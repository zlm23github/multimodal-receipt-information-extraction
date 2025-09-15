#!/usr/bin/env python3
"""
Receipt Text Extraction
Extract text from different receipt regions using OCR
"""

import cv2
import easyocr
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re


class ReceiptTextExtractor:
    def __init__(self):
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'])
        print("EasyOCR initialized")
        
        self.extraction_strategies = {
            'header': self.extract_header_text,
            'items': self.extract_items_text,
            'total': self.extract_total_text,
            'footer': self.extract_footer_text
        }
    
    def extract_text_from_regions(self, regions: Dict[str, np.ndarray], image_name: str) -> Dict[str, Dict]:
        """
        Extract text from different receipt regions
        """
        print(f"Extracting text from {len(regions)} regions")
        
        results = {}
        raw_results = {}
        
        for region_type, region_img in regions.items():
            print(f"  Processing {region_type} region...")
            
            if region_type in self.extraction_strategies:
                region_text = self.extraction_strategies[region_type](region_img)
            else:
                region_text = self.extract_generic_text(region_img)
            
            raw_results[region_type] = {
                'text_elements': region_text,
                'count': len(region_text)
            }
            
            results[region_type] = {
                'raw_text': region_text,
                'processed_text': self.process_region_text(region_type, region_text),
                'confidence': self.calculate_region_confidence(region_text)
            }
            
            print(f"    Extracted {len(region_text)} text elements")
        
        self.save_raw_text_results(raw_results, image_name)
        
        return results
    
    def save_raw_text_results(self, raw_results: Dict, image_name: str):
        """
        Save raw unprocessed text results to JSON for debugging
        """
        import json
        from pathlib import Path
        
        output_dir = Path("data/results/visualization/ocr_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        serializable_results = {}
        for region_type, data in raw_results.items():
            serializable_results[region_type] = {
                'count': data['count'],
                'text_elements': []
            }
            
            for element in data['text_elements']:
                serializable_element = {
                    'text': element['text'],
                    'confidence': float(element['confidence']),
                    'x_min': float(element['x_min']),
                    'x_max': float(element['x_max']),
                    'y_min': float(element['y_min']),
                    'y_max': float(element['y_max']),
                    'rel_x': float(element['rel_x']),
                    'rel_y': float(element['rel_y']),
                    'rel_width': float(element['rel_width']),
                    'rel_height': float(element['rel_height']),
                    'area': float(element['area'])
                }
                serializable_results[region_type]['text_elements'].append(serializable_element)
        
        output_file = output_dir / f"{image_name}_raw_text.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"  Raw text saved: {output_file}")
    
    def extract_header_text(self, header_img: np.ndarray) -> List[Dict]:
        """
        Extract text from header region (store name, etc.)
        """
        results = self.reader.readtext(
            header_img,
            width_ths=0.7,
            height_ths=0.7,
            text_threshold=0.4,
            low_text=0.4,
            link_threshold=0.4
        )
        
        return self.process_ocr_results(results, header_img.shape)


    def extract_items_text(self, items_img: np.ndarray) -> List[Dict]:
        """
        Extract text from items region (product names and prices)
        """
        results = self.reader.readtext(
            items_img,
            width_ths=0.5,
            height_ths=0.5,
            text_threshold=0.3,
            low_text=0.3,
            link_threshold=0.6
        )
        
        return self.process_ocr_results(results, items_img.shape)
    
    def extract_total_text(self, total_img: np.ndarray) -> List[Dict]:
        """
        Extract text from total region (amounts, totals)
        """
        results = self.reader.readtext(
            total_img,
            width_ths=0.6,
            height_ths=0.6,
            text_threshold=0.4,
            low_text=0.4,
            link_threshold=0.5
        )
        
        return self.process_ocr_results(results, total_img.shape)
    
    def extract_footer_text(self, footer_img: np.ndarray) -> List[Dict]:
        """
        Extract text from footer region (date, time, etc.)
        """
        results = self.reader.readtext(
            footer_img,
            width_ths=0.8,
            height_ths=0.8,
            text_threshold=0.5,
            low_text=0.5,
            link_threshold=0.4
        )
        
        return self.process_ocr_results(results, footer_img.shape)
    
    def extract_generic_text(self, img: np.ndarray) -> List[Dict]:
        """
        Generic text extraction for unknown regions
        """
        results = self.reader.readtext(img)
        return self.process_ocr_results(results, img.shape)
    
    def process_ocr_results(self, results: List, img_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        Process OCR results into structured format
        """
        processed_results = []
        height, width = img_shape[:2]
        
        for result in results:
            if len(result) == 3:
                bbox, text, confidence = result
            elif len(result) == 2:
                bbox, text = result
                confidence = 0.5
            else:
                continue
            
            x_coords = [coord[0] for coord in bbox]
            y_coords = [coord[1] for coord in bbox]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            rel_x = (x_min + x_max) / 2 / width
            rel_y = (y_min + y_max) / 2 / height
            rel_width = (x_max - x_min) / width
            rel_height = (y_max - y_min) / height
            
            processed_results.append({
                'text': text.strip(),
                'confidence': confidence,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'rel_x': rel_x,
                'rel_y': rel_y,
                'rel_width': rel_width,
                'rel_height': rel_height,
                'area': (x_max - x_min) * (y_max - y_min)
            })
        
        return processed_results
    
    def process_region_text(self, region_type: str, text_data: List[Dict]) -> Dict:
        """
        Process extracted text based on region type
        """
        if not text_data:
            return {}
        
        if region_type == 'header':
            return self.process_header_text(text_data)
        elif region_type == 'items':
            return self.process_items_text(text_data)
        elif region_type == 'total':
            return self.process_total_text(text_data)
        elif region_type == 'footer':
            return self.process_footer_text(text_data)
        else:
            return self.process_generic_text(text_data)
    
    def process_header_text(self, text_data: List[Dict]) -> Dict:
        """
        Process header text to extract store name
        """
        high_conf = [t for t in text_data if t['confidence'] > 0.6]
        
        if not high_conf:
            high_conf = sorted(text_data, key=lambda x: x['confidence'], reverse=True)[:3]
        
        top_text = [t for t in high_conf if t['rel_y'] < 0.5]
        
        store_parts = []
        for text_item in top_text:
            text = text_item['text']
            if len(text) > 1 and any(c.isalpha() for c in text):
                store_parts.append(text)
        
        store_name = " ".join(store_parts[:3])
        
        return {
            'store_name': store_name if store_name else "Unknown Store",
            'raw_text': [t['text'] for t in text_data]
        }
    
    def process_items_text(self, text_data: List[Dict]) -> Dict:
        """
        Process items text to extract product names and prices
        """
        items = []
        
        sorted_text = sorted(text_data, key=lambda x: x['rel_y'])
        
        for element in sorted_text:
            text = element['text'].strip()
            
            
            price_match = re.match(r'^(\d+\.\d{2})\s*([EA])?$', text)
            if price_match:
                if items and items[-1].get('price') is None:
                    price = float(price_match.group(1))
                    tax_indicator = price_match.group(2) if price_match.group(2) else None
                    items[-1]['price'] = price
                    items[-1]['tax_indicator'] = tax_indicator
                continue
            
            item_match = re.match(r'^(\d{6})\s+(.+)$', text)
            if item_match:
                item_number = item_match.group(1)
                item_name = item_match.group(2)
                
                items.append({
                    'item_number': item_number,
                    'name': f"{item_number} {item_name}",
                    'price': None,
                    'tax_indicator': None,
                    'confidence': element['confidence']
                })
                continue
            
            if len(text) > 3 and not re.match(r'^\d+\.\d{2}', text) and not text in ['E', 'A']:
                if items and items[-1].get('price') is None:
                    items[-1]['name'] += ' ' + text
                else:
                    items.append({
                        'item_number': None,
                        'name': text,
                        'price': None,
                        'tax_indicator': None,
                        'confidence': element['confidence']
                    })
        
        return {
            'items': items,
            'item_count': len(items)
        }
    
    def is_valid_item(self, name: str, price: float) -> bool:
        """
        Check if an item looks valid
        """
        name = name.strip()
        
        if not name:
            return False
        
        if len(name) > 100:
            return False
        
        non_items = ['SUBTOTAL', 'TAX', 'TOTAL', 'Check', 'Member', 'Prntd', 'CHANGE', 'Thank', 'You', 'Please', 'Come', 'Again', 'E', 'A']
        if name.upper() in non_items:
            return False
        
        alpha_count = sum(1 for c in name if c.isalpha())
        if len(name) > 0 and alpha_count / len(name) < 0.3:
            return False
        
        if price > 0 and (price < 0.01 or price > 1000):
            return False
        
        return True
    
    def process_total_text(self, text_data: List[Dict]) -> Dict:
        """
        Process total text to extract amounts
        """
        amounts = []
        
        for text_item in text_data:
            text = text_item['text']
            numbers = re.findall(r'[\$]?\d+\.?\d*', text)
            for num_str in numbers:
                try:
                    amount = float(num_str.replace('$', ''))
                    if 0.01 <= amount <= 9999.99:
                        amounts.append({
                            'value': amount,
                            'confidence': text_item['confidence'],
                            'text': text
                        })
                except ValueError:
                    continue
        
        amounts.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'total_amount': amounts[0]['value'] if amounts else 0.0,
            'all_amounts': amounts
        }
    
    def process_footer_text(self, text_data: List[Dict]) -> Dict:
        """
        Process footer text to extract date and other info
        """
        dates = []
        
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{1,2}-\d{1,2}-\d{2,4}'
        ]
        
        for text_item in text_data:
            text = text_item['text']
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    dates.append({
                        'date': match.group(),
                        'confidence': text_item['confidence']
                    })
        
        dates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'date': dates[0]['date'] if dates else "Unknown Date",
            'all_dates': dates
        }
    
    def process_generic_text(self, text_data: List[Dict]) -> Dict:
        """
        Generic text processing
        """
        return {
            'text_elements': [t['text'] for t in text_data],
            'count': len(text_data)
        }
    
    def extract_price(self, text: str) -> float:
        """
        Extract price from text
        """
        numbers = re.findall(r'[\$]?\d+\.?\d*', text)
        for num_str in numbers:
            try:
                return float(num_str.replace('$', ''))
            except ValueError:
                continue
        return 0.0
    
    def calculate_region_confidence(self, text_data: List[Dict]) -> float:
        """
        Calculate average confidence for a region
        """
        if not text_data:
            return 0.0
        return np.mean([t['confidence'] for t in text_data])
    
    def save_results(self, results: Dict, image_name: str, output_dir: str = "data/results/visualization/ocr_results"):
        """
        Save extraction results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import json
        json_file = output_path / f"{image_name}_text_extraction.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Text extraction results saved: {json_file}")
        
        # Save summary as CSV
        summary_data = []
        for region_type, region_data in results.items():
            summary_data.append({
                'region': region_type,
                'text_count': len(region_data.get('raw_text', [])),
                'confidence': region_data.get('confidence', 0.0),
                'processed_info': str(region_data.get('processed_text', {}))
            })
        
        df = pd.DataFrame(summary_data)
        csv_file = output_path / f"{image_name}_text_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"Text summary saved: {csv_file}")


def main():
    from region_detector import ReceiptRegionDetector
    
    # Detect regions first
    detector = ReceiptRegionDetector()
    image_path = "data/corrected_images/8_corrected.jpg"
    
    if Path(image_path).exists():
        regions = detector.detect_regions(image_path)
        
        # Extract text from regions
        extractor = ReceiptTextExtractor()
        results = extractor.extract_text_from_regions(regions, Path(image_path).stem)
        
        extractor.save_results(results, Path(image_path).stem)
    else:
        print(f"Image not found: {image_path}")


if __name__ == "__main__":
    main()
