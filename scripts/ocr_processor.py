import os
import easyocr
import cv2
import pandas as pd
from tqdm import tqdm


class OCRProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.raw_image_dir = os.path.join(self.data_dir, "raw_images")
        self.ocr_results_dir = os.path.join(self.data_dir, "ocr_results")

        # initialize the EasyOCR
        print(f"Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'])
        print(f"EasyOCR initialized successfully")

        os.makedirs(self.ocr_results_dir, exist_ok=True)

    def preprocess_image(self, image_path):
        """
        Preprocess the image to improve the OCR accuracy
        """
        try:
            # read the image
            img = cv2.imread(image_path)

            # convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # binary thresholding
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            return binary

        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None

        

    def extract_text_easyocr(self, image_path):
        """
        Extract text from the image using EasyOCR
        """
        try:
            # preprocess the image
            processed_image = self.preprocess_image(image_path)

            # OCR
            results = self.reader.readtext(processed_image)
            
            ocr_data = []
            for(bbox, text, confidence) in results:
                x_coords = [coord[0] for coord in bbox]
                y_coords = [coord[1] for coord in bbox]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                ocr_data.append({
                    "text": text,
                    "confidence": confidence,
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max
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