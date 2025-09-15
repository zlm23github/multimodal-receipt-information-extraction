#!/usr/bin/env python3
"""
Image Orientation Corrector
Correct the orientation of receipt images to make them straight
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class ImageOrientationCorrector:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def detect_skew_angle(self, image: np.ndarray) -> float:
        """
        Detect the skew angle using projection profile method
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        best_angle = 0.0
        best_score = 0.0
        
        for angle in np.arange(-30, 31, 0.5):
            h, w = binary.shape
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(binary, rotation_matrix, (w, h), 
                                   flags=cv2.INTER_LANCZOS4, 
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
            
            horizontal_projection = np.sum(rotated, axis=1)
            
            projection_variance = np.var(horizontal_projection)
            
            threshold = np.max(horizontal_projection) * 0.1
            peaks = []
            for i in range(1, len(horizontal_projection) - 1):
                if (horizontal_projection[i] > threshold and 
                    horizontal_projection[i] > horizontal_projection[i-1] and 
                    horizontal_projection[i] > horizontal_projection[i+1]):
                    peaks.append(i)
            peak_count = len(peaks)
            
            score = projection_variance + peak_count * 100
            
            if score > best_score:
                best_score = score
                best_angle = angle
        
        return best_angle
    
    def correct_orientation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Correct the orientation of the image by rotating it
        """
        if abs(angle) < 0.5:  # No correction needed
            return image
        
        # Get image dimensions
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix (rotate by +angle to correct the skew)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions to avoid cropping
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        corrected = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                 flags=cv2.INTER_LANCZOS4, 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))  # White background
        
        return corrected
    
    def process_single_image(self, input_path: str, output_path: str) -> bool:
        """
        Process a single image for orientation correction
        """
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image: {input_path}")
                return False
            
            print(f"📷 Processing: {Path(input_path).name}")
            print(f"   Original size: {image.shape[1]}x{image.shape[0]}")
            
            angle = self.detect_skew_angle(image)
            print(f"   Detected skew: {angle:.2f} degrees")
            
            # Correct orientation
            if abs(angle) > 0.5:
                corrected = self.correct_orientation(image, angle)
                print(f"   Corrected size: {corrected.shape[1]}x{corrected.shape[0]}")
                print(f"   Applied correction: {angle:.2f} degrees")
            else:
                corrected = image
                print(f"   No correction needed")
            
            success = cv2.imwrite(output_path, corrected)
            if success:
                print(f"   Saved to: {output_path}")
                return True
            else:
                print(f"   Failed to save: {output_path}")
                return False
                
        except Exception as e:
            print(f"   Error processing {input_path}: {e}")
            return False
    
    def process_directory(self, input_dir: str, output_dir: str) -> dict:
        """
        Process all images in a directory
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            print(f"Input directory does not exist: {input_dir}")
            return {"success": 0, "failed": 0, "total": 0}
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in: {input_dir}")
            return {"success": 0, "failed": 0, "total": 0}
        
        print(f"🔍 Found {len(image_files)} image files")
        print(f"📁 Input directory: {input_dir}")
        print(f"📁 Output directory: {output_dir}")
        print("-" * 50)
        
        success_count = 0
        failed_count = 0
        
        for image_file in sorted(image_files):
            # Generate output filename
            output_filename = f"{image_file.stem}_corrected.jpg"
            output_file_path = output_path / output_filename
            
            # Process image
            if self.process_single_image(str(image_file), str(output_file_path)):
                success_count += 1
            else:
                failed_count += 1
            print()  # Empty line for readability
        
        # Summary
        total = success_count + failed_count
        print("=" * 50)
        print(f"Processing Summary:")
        print(f"   Total images: {total}")
        print(f"   Successfully processed: {success_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Success rate: {success_count/total*100:.1f}%")
        
        return {
            "success": success_count,
            "failed": failed_count,
            "total": total
        }


def main():
    """
    Correct orientation of all images in data/raw_images directory
    """
    corrector = ImageOrientationCorrector()
    
    input_dir = "data/raw_images"
    output_dir = "data/corrected_images"
    
    print(f"Starting orientation correction...")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    
    result = corrector.process_directory(input_dir, output_dir)
    
    if result["total"] > 0:
        print(f"Processing completed! {result['success']}/{result['total']} images processed successfully.")
    else:
        print(f"No images were processed.")


if __name__ == "__main__":
    main()
