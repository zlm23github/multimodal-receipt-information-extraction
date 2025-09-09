import os
import easyocr
import cv2
import pandas as pd
import numpy as np
import re
from tqdm import tqdm


class OCRProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.raw_image_dir = os.path.join(self.data_dir, "raw_images")
        self.ocr_results_dir = os.path.join(self.data_dir, "ocr_results")

        # initialize the EasyOCR with optimized parameters
        print(f"Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'], 
                                   gpu=False,  # Use CPU for stability
                                   model_storage_directory=None,
                                   user_network_directory=None,
                                   download_enabled=True)
        print(f"EasyOCR initialized successfully")

        os.makedirs(self.ocr_results_dir, exist_ok=True)

    def detect_and_correct_skew(self, img):
        """
        Detect and correct image skew for receipt images
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None or len(lines) == 0:
                print(f"  No lines detected, skipping skew correction")
                return img, 0
            
            # Calculate angles of detected lines
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                # Normalize angle to [-45, 45] range
                if angle > 45:
                    angle = angle - 90
                elif angle < -45:
                    angle = angle + 90
                
                # Only consider horizontal lines (close to 0 degrees)
                if abs(angle) < 30:
                    angles.append(angle)
            
            # Calculate median angle (more robust than mean)
            median_angle = np.median(angles)
            
            # Only correct if angle is significant (> 1 degree)
            if abs(median_angle) > 1:
                print(f"  Detected skew: {median_angle:.2f} degrees, correcting...")
                
                # Get image dimensions
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                
                # Create rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                
                # Calculate new dimensions to avoid cropping
                cos_angle = abs(rotation_matrix[0, 0])
                sin_angle = abs(rotation_matrix[0, 1])
                new_w = int((h * sin_angle) + (w * cos_angle))
                new_h = int((h * cos_angle) + (w * sin_angle))
                
                # Adjust rotation matrix for new dimensions
                rotation_matrix[0, 2] += (new_w / 2) - center[0]
                rotation_matrix[1, 2] += (new_h / 2) - center[1]
                
                # Apply rotation
                corrected_img = cv2.warpAffine(img, rotation_matrix, (new_w, new_h), 
                                             flags=cv2.INTER_CUBIC, 
                                             borderMode=cv2.BORDER_REPLICATE)
                
                print(f"  Skew correction applied: {median_angle:.2f} degrees")
                return corrected_img, median_angle
            else:
                print(f"  No significant skew detected: {median_angle:.2f} degrees")
                return img, 0
                
        except Exception as e:
            print(f"  Error in skew correction: {e}")
            return img, 0

    def preprocess_image(self, image_path):
        """
        Preprocess the image with skew correction and scaling to improve OCR accuracy
        """
        try:
            # read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"  Error: Could not read image {image_path}")
                return None

            print(f"  Processing: {os.path.basename(image_path)}")
            
            # Step 1: Detect and correct skew
            corrected_img, skew_angle = self.detect_and_correct_skew(img)
            
            # Step 2: Scale up the image for better OCR accuracy
            scale_factor = 2.0
            height, width = corrected_img.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            scaled_img = cv2.resize(corrected_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Calculate actual scale factor from original to final processed image
            orig_height, orig_width = img.shape[:2]
            final_height, final_width = scaled_img.shape[:2]
            actual_scale_x = orig_width / final_width
            actual_scale_y = orig_height / final_height
            
            # Step 3: Convert to grayscale
            gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)

            # Step 4: Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Step 5: Binary thresholding
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            print(f"  Skew correction: {skew_angle:.2f} degrees, Scale factor: {scale_factor}")
            print(f"  Actual scale factors: x={actual_scale_x:.3f}, y={actual_scale_y:.3f}")
            return binary, (actual_scale_x, actual_scale_y)

        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None, 1.0

        

    def extract_text_easyocr(self, image_path):
        """
        Extract text from the image using EasyOCR
        """
        try:
            # Load original image to get dimensions
            original_img = cv2.imread(image_path)
            if original_img is None:
                print(f"Error: Could not read image {image_path}")
                return [], False
            
            orig_h, orig_w = original_img.shape[:2]
            
            # preprocess the image (returns processed image and scale factors)
            result = self.preprocess_image(image_path)
            if result is None:
                return [], False
            
            processed_image, scale_factors = result
            if processed_image is None:
                return [], False
            
            # Get processed image dimensions
            proc_h, proc_w = processed_image.shape[:2]
            
            # Use the actual scale factors from preprocessing
            scale_x, scale_y = scale_factors
            
            print(f"  Original size: {orig_w}x{orig_h}, Processed size: {proc_w}x{proc_h}")
            print(f"  Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

            # OCR with parameters to get larger bounding boxes
            results = self.reader.readtext(processed_image, 
                                         width_ths=0.5,      # Lower threshold to merge nearby text
                                         height_ths=0.5,     # Lower threshold to merge nearby text
                                         text_threshold=0.3, # Lower threshold for text detection
                                         low_text=0.3,       # Lower threshold for low text
                                         link_threshold=0.6, # Higher threshold to link more text
                                         canvas_size=2560,   # Larger canvas size
                                         mag_ratio=1.0)      # No additional magnification
            
            ocr_data = []
            for result in results:
                if len(result) == 3:
                    bbox, text, confidence = result
                elif len(result) == 2:
                    bbox, text = result
                    confidence = 0.5  # Default confidence
                else:
                    continue
                
                # Get coordinates from processed image
                x_coords = [coord[0] for coord in bbox]
                y_coords = [coord[1] for coord in bbox]
                proc_x_min, proc_x_max = min(x_coords), max(x_coords)
                proc_y_min, proc_y_max = min(y_coords), max(y_coords)
                
                # Convert coordinates back to original image scale
                orig_x_min = proc_x_min * scale_x
                orig_x_max = proc_x_max * scale_x
                orig_y_min = proc_y_min * scale_y
                orig_y_max = proc_y_max * scale_y
                
                ocr_data.append({
                    "text": text,
                    "confidence": confidence,
                    "x_min": orig_x_min,
                    "x_max": orig_x_max,
                    "y_min": orig_y_min,
                    "y_max": orig_y_max
                })

            return ocr_data, True

        except Exception as e:
            print(f"Error extracting text from image {image_path}: {e}")
            return [], False
        

    def process_all_images(self):
        """
        Process all images in the raw image directory
        """
        try:
            metadata_path = os.path.join(self.data_dir, 'metadata', 'image_metadata.csv')

            df_metadata = pd.read_csv(metadata_path)
            all_results = []

            print(f"Starting to process {len(df_metadata)} images")

            for _, row in tqdm(df_metadata.iterrows(), total=len(df_metadata)):
                image_path = row['file_path']
                image_id = row['image_id']

                ocr_data, success = self.extract_text_easyocr(image_path)

                if success:
                    for item in ocr_data:
                        item['image_id'] = image_id
                        all_results.append(item)
                    print(f"Processed {row['file_name']} successfully")
                else:
                    print(f"Failed to process {row['file_name']}")

            # save the results
            df_results = pd.DataFrame(all_results)
            output_path = os.path.join(self.ocr_results_dir, 'ocr_results.csv')
            df_results.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"\n=== OCR Processing Summary ===")
            print(f"Total images processed: {len(df_metadata)}")
            print(f"Total OCR results collected: {len(all_results)}")
            print(f"Saved OCR results to: {output_path}")

            return df_results

        except Exception as e:
            print(f"Error processing all images: {e}")
            return None
            


def main():
    """
    Main function
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    print(f"Processing data from: {data_dir}")

    ocr_processor = OCRProcessor(data_dir)
    df_results = ocr_processor.process_all_images()
    print(f"OCR processing completed")
    print(df_results.head())


if __name__ == "__main__":
    main()