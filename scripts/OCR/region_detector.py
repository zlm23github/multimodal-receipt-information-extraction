#!/usr/bin/env python3
"""
Receipt Region Detection
Detect different regions in receipt images (header, items, total, etc.)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import easyocr


class ReceiptRegionDetector:
    def __init__(self):
        self.region_types = {
            'header': 'Store name and basic info',
            'items': 'Product list and prices', 
            'total': 'Total amount and summary',
            'footer': 'Date, time, and other info'
        }
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'])
    
    def detect_regions(self, image_path: str) -> Dict[str, np.ndarray]:
        """
        Detect different regions in a receipt image
        """
        print(f"Detecting regions in: {Path(image_path).name}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return {}
        
        height, width = image.shape[:2]
        print(f"  Image size: {width}x{height}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        processed = self.preprocess_image(gray)
        
        text_regions = self.detect_text_regions_with_ocr(processed)
        
        classified_regions = self.classify_regions(image, text_regions)
        
        print(f"  Detected {len(classified_regions)} regions")
        for region_type, region_img in classified_regions.items():
            print(f"    - {region_type}: {region_img.shape}")
        
        return classified_regions
    
    def preprocess_image(self, gray_image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def detect_text_regions(self, processed_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        min_area = 100  # Minimum area for text regions
        max_area = 50000  # Maximum area for text regions
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if min_area < area < max_area:
                aspect_ratio = w / h
                if 0.1 < aspect_ratio < 10:  # Reasonable aspect ratio for text
                    text_regions.append((x, y, w, h))
        
        text_regions.sort(key=lambda x: x[1])
        
        return text_regions
    
    def detect_text_regions_with_ocr(self, processed_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        results = self.reader.readtext(processed_image)
        
        text_regions = []
        for (bbox, text, confidence) in results:
            x_coords = [coord[0] for coord in bbox]
            y_coords = [coord[1] for coord in bbox]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            w = int(x_max - x_min)
            h = int(y_max - y_min)
            x = int(x_min)
            y = int(y_min)
            
            text_regions.append((x, y, w, h))
        
        text_regions.sort(key=lambda x: x[1])
        
        return text_regions
    
    def classify_regions(self, original_image: np.ndarray, text_regions: List[Tuple[int, int, int, int]]) -> Dict[str, np.ndarray]:
        height, width = original_image.shape[:2]
        regions = {}
        
        if not text_regions:
            return regions
        
        y_coords = [y for x, y, w, h in text_regions]
        y_min, y_max = min(y_coords), max(y_coords)
        
        region_boundaries = self._detect_adaptive_boundaries(original_image, text_regions, y_min, y_max)
        
        print(f"  Adaptive region detection:")
        print(f"     Header: 0 - {region_boundaries['header_end']} ({region_boundaries['header_end']/height*100:.1f}%)")
        print(f"     Items: {region_boundaries['header_end']} - {region_boundaries['items_end']} ({(region_boundaries['items_end']-region_boundaries['header_end'])/height*100:.1f}%)")
        print(f"     Total: {region_boundaries['items_end']} - {region_boundaries['total_end']} ({(region_boundaries['total_end']-region_boundaries['items_end'])/height*100:.1f}%)")
        print(f"     Footer: {region_boundaries['total_end']} - {height} ({(height-region_boundaries['total_end'])/height*100:.1f}%)")
        
        regions['header'] = original_image[0:region_boundaries['header_end'], :]
        regions['items'] = original_image[region_boundaries['header_end']:region_boundaries['items_end'], :]
        regions['total'] = original_image[region_boundaries['items_end']:region_boundaries['total_end'], :]
        regions['footer'] = original_image[region_boundaries['total_end']:, :]
        
        regions = {k: v for k, v in regions.items() if v.size > 0}
        
        return regions
    
    def _detect_adaptive_boundaries(self, image: np.ndarray, text_regions: List[Tuple[int, int, int, int]], y_min: int, y_max: int) -> Dict[str, int]:
        height, width = image.shape[:2]
        
        strip_height = 15
        strips = []
        
        for y in range(y_min, y_max, strip_height):
            strip_text_count = 0
            strip_has_numbers = False
            strip_has_large_text = False
            
            for x, region_y, w, h in text_regions:
                if region_y <= y < region_y + h:
                    strip_text_count += 1
                    if w < 80 and h < 25:  # Small regions often contain numbers
                        strip_has_numbers = True
                    if w > 150 and h > 20:  # Large regions often contain headers
                        strip_has_large_text = True
            
            strips.append({
                'y': y,
                'text_count': strip_text_count,
                'has_numbers': strip_has_numbers,
                'has_large_text': strip_has_large_text
            })
        
        header_end = y_min
        
        member_found = False
        print(f"  Checking {len(text_regions)} text regions for 'Member'...")
        
        for i, (x, region_y, w, h) in enumerate(text_regions):
            if region_y < y_min + int((y_max - y_min) * 0.4):  # First 40% of image
                try:
                    region_img = image[region_y:region_y+h, x:x+w]
                    results = self.reader.readtext(region_img)
                    for (bbox, text, confidence) in results:
                        print(f"    Region {i}: '{text}' (confidence: {confidence:.2f})")
                        if ('member' in text.lower() or 
                            '2x' in text.lower()):
                            header_end = region_y + h
                            member_found = True
                            print(f"  Found 'Member' at y={region_y + h}, text='{text}'")
                            break
                    if member_found:
                        break
                except Exception as e:
                    print(f"  OCR error for region {x},{region_y},{w},{h}: {e}")
                    pass
        
        if not member_found:
            print("  'Member' not found, using fallback logic")
            large_text_strips = [i for i, strip in enumerate(strips) if strip['has_large_text']]
            
            if large_text_strips:
                last_large_text = large_text_strips[-1]
                header_end = strips[last_large_text]['y'] + strip_height
            else:
                max_density = max(strip['text_count'] for strip in strips)
                for i, strip in enumerate(strips):
                    if strip['text_count'] < max_density * 0.4:  # 40% of max density
                        header_end = strip['y']
                        break
        
        total_start = y_max
        total_end = y_max
        
        subtotal_found = False
        print(f"  Looking for 'SUBTOTAL' keyword...")
        
        for i, (x, region_y, w, h) in enumerate(text_regions):
            if region_y > y_min + int((y_max - y_min) * 0.4):
                try:
                    region_img = image[region_y:region_y+h, x:x+w]
                    results = self.reader.readtext(region_img)
                    for (bbox, text, confidence) in results:
                        print(f"    Region {i}: '{text}' (confidence: {confidence:.2f})")
                        if 'subtotal' in text.lower():
                            total_start = region_y  # SUBTOTAL这一行的开始
                            subtotal_found = True
                            print(f"  Found 'SUBTOTAL' at y={region_y}, text='{text}'")
                            break
                    if subtotal_found:
                        break
                except Exception as e:
                    print(f"  OCR error for region {x},{region_y},{w},{h}: {e}")
                    pass
        
        if not subtotal_found:
            print("  'SUBTOTAL' not found, using number-based detection")
            number_strips = [i for i, strip in enumerate(strips) if strip['has_numbers']]
            
            if number_strips:
                clusters = []
                current_cluster = [number_strips[0]]
                
                for i in range(1, len(number_strips)):
                    if number_strips[i] - number_strips[i-1] <= 2:  # Within 2 strips (30px)
                        current_cluster.append(number_strips[i])
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [number_strips[i]]
                clusters.append(current_cluster)
                
                if clusters:
                    largest_cluster = max(clusters, key=len)
                    total_start = strips[largest_cluster[0]]['y']
                    total_end = strips[largest_cluster[-1]]['y'] + strip_height
        
        items_end = total_start
        
        header_end = max(header_end, y_min + int((y_max - y_min) * 0.15))
        header_end = min(header_end, y_min + int((y_max - y_min) * 0.35))
        
        if subtotal_found:
            pass
        else:
            items_end = max(items_end, header_end + int((y_max - y_min) * 0.4))
            items_end = min(items_end, y_max - int((y_max - y_min) * 0.2))
        
        total_end = max(total_end, items_end + int((y_max - y_min) * 0.15))
        total_end = min(total_end, y_max - int((y_max - y_min) * 0.1))
        
        return {
            'header_end': header_end,
            'items_end': items_end,
            'total_end': total_end
        }
    
    def visualize_regions(self, image_path: str, output_dir: str = "data/results/visualization/ocr_results"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        regions = self.detect_regions(image_path)
        
        if not regions:
            print("No regions detected")
            return
        
        image_name = Path(image_path).stem
        for region_type, region_img in regions.items():
            output_file = output_path / f"{image_name}_{region_type}.jpg"
            cv2.imwrite(str(output_file), region_img)
            print(f"  Saved {region_type} region: {output_file}")
        
        original = cv2.imread(image_path)
        height, width = original.shape[:2]
        
        regions = self.detect_regions(image_path)
        if not regions:
            print("No regions detected for visualization")
            return
        
        header_end = regions['header'].shape[0]
        items_end = header_end + regions['items'].shape[0]
        total_end = items_end + regions['total'].shape[0]
        
        cv2.line(original, (0, header_end), (width, header_end), (0, 255, 0), 2)
        cv2.line(original, (0, items_end), (width, items_end), (0, 255, 0), 2)
        cv2.line(original, (0, total_end), (width, total_end), (0, 255, 0), 2)
        
        cv2.putText(original, "HEADER", (10, header_end - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(original, "ITEMS", (10, items_end - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(original, "TOTAL", (10, total_end - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(original, "FOOTER", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        viz_file = output_path / f"{image_name}_regions_visualization.jpg"
        cv2.imwrite(str(viz_file), original)
        print(f"  Saved visualization: {viz_file}")


def main():
    detector = ReceiptRegionDetector()
    
    image_path = "data/corrected_images/8_corrected.jpg"
    if Path(image_path).exists():
        detector.visualize_regions(image_path)
    else:
        print(f"Image not found: {image_path}")


if __name__ == "__main__":
    main()
