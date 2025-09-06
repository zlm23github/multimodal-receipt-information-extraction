import os
import hashlib
import pandas as pd
from PIL import Image

def collect_image_metadata(data_dir):
    """
    Collect metadata from the image
    """
    metadata = []
    raw_image_dir = os.path.join(data_dir, "raw_images")

    if not os.path.exists(raw_image_dir):
        raise FileNotFoundError(f"Raw image directory {raw_image_dir} not found")

    # support all image extensions
    image_extensions = (".jpg", ".jpeg", ".png", ".webp")

    print(f"Starting to scan raw image directory: {raw_image_dir}")

    for file_name in os.listdir(raw_image_dir):
        if file_name.endswith(image_extensions):
            file_path = os.path.join(raw_image_dir, file_name)

            try:
                # open the image and get the width, height, and mode
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode

                # get the file size
                file_size = os.path.getsize(file_path)
                
                # generate a unique id for the image
                image_id = hashlib.md5(file_path.encode()).hexdigest()[:8]

                metadata.append({
                    "image_id": image_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "file_size": file_size
                })

                print(f"Collected metadata for image: {file_name}")
            except Exception as e:
                print(f"Error collecting metadata for image: {file_name}")
                print(e)
    
    if not metadata:
        print("No metadata collected")
        return
    
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(data_dir, 'metadata', 'image_metadata.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print("\n=== Metadata Collection Summary ===")
    print(f"Collected {len(metadata)} images")
    print(f"Saved metadata to: {csv_path}")

    return df

def main():
    """
    Main function
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = collect_image_metadata(data_dir)


if __name__ == "__main__":
    main()