#!/usr/bin/env python3
"""
OCR Results Visualizer
Visualize OCR results by overlaying text and bounding boxes on original images
"""

import pandas as pd
import cv2
import numpy as np
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class OCRVisualizer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_images_dir = os.path.join(data_dir, "raw_images")
        self.cleaned_data_dir = os.path.join(data_dir, "cleaned_data")
        self.visualization_dir = os.path.join(data_dir, "visualization")
        os.makedirs(self.visualization_dir, exist_ok=True)
    
    def load_cleaned_data(self) -> pd.DataFrame:
        """Load cleaned OCR data"""
        file_path = os.path.join(self.cleaned_data_dir, "cleaned_ocr_results.csv")
        df = pd.read_csv(file_path)
        print(f"Loaded cleaned OCR data: {len(df)} records")
        return df
    
    def get_image_files(self) -> List[str]:
        """Get list of image files"""
        image_files = []
        for file in os.listdir(self.raw_images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(file)
        return sorted(image_files)
    
    def get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence level"""
        if confidence >= 0.8:
            return 'green'
        elif confidence >= 0.6:
            return 'blue'
        elif confidence >= 0.4:
            return 'orange'
        else:
            return 'red'
    
    def visualize_image_ocr(self, image_file: str, df: pd.DataFrame, save_path: str = None):
        """Visualize OCR results for a single image"""
        # Load image
        image_path = os.path.join(self.raw_images_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        # Get original image dimensions
        orig_h, orig_w = image.shape[:2]
        print(f"Original image size: {orig_w} x {orig_h}")
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get OCR data for this image
        image_ocr_data = df  # Use all data for now since we only have one image
        
        if len(image_ocr_data) == 0:
            print(f"No OCR data found for image: {image_file}")
            return None
        
        # Debug: Show coordinate ranges
        print(f"OCR coordinate ranges:")
        print(f"  X: {df['x_min'].min():.1f} - {df['x_max'].max():.1f}")
        print(f"  Y: {df['y_min'].min():.1f} - {df['y_max'].max():.1f}")
        print(f"  Image bounds: X: 0-{orig_w}, Y: 0-{orig_h}")
        
        # Create figure with larger size
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        ax.imshow(image_rgb)
        ax.set_title(f"OCR Results: {image_file}", fontsize=18, fontweight='bold')
        
        # Set axis limits to match image dimensions
        ax.set_xlim(0, orig_w)
        ax.set_ylim(orig_h, 0)  # Invert Y axis for image coordinates
        
        # Filter for meaningful results (lower confidence but larger boxes)
        meaningful_data = image_ocr_data[image_ocr_data['confidence'] >= 0.6].copy()
        print(f"  Showing {len(meaningful_data)} meaningful results (confidence >= 0.6)")
        
        # Draw bounding boxes and text
        text_positions = {}  # Track text positions to avoid overlap
        
        for i, (_, row) in enumerate(meaningful_data.iterrows()):
            # Use coordinates with padding to make boxes larger
            padding = 5  # Add 5 pixels padding around each box
            x_min = max(0, int(row['x_min']) - padding)
            y_min = max(0, int(row['y_min']) - padding)
            x_max = min(orig_w, int(row['x_max']) + padding)
            y_max = min(orig_h, int(row['y_max']) + padding)
            
            # Ensure coordinates are within image bounds
            x_min = max(0, min(x_min, orig_w))
            y_min = max(0, min(y_min, orig_h))
            x_max = max(0, min(x_max, orig_w))
            y_max = max(0, min(y_max, orig_h))
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Skip very small boxes (only show meaningful text regions)
            if width < 20 or height < 15:
                continue
            
            # Get color based on confidence
            color = self.get_confidence_color(row['confidence'])
            
            # Draw bounding box with thicker lines
            rect = Rectangle((x_min, y_min), width, height, 
                           linewidth=3, edgecolor=color, facecolor='none', alpha=0.9)
            ax.add_patch(rect)
            
            # Add text label with content and confidence
            text = f"{row['text']} ({row['confidence']:.2f})"
            
            # Position text above the box
            text_x = x_min
            text_y = y_min - 15
            
            # Check for overlap and adjust position
            while (text_x, text_y) in text_positions:
                text_y -= 25
                if text_y < 40:  # If too high, move to right
                    text_x += 80
                    text_y = y_min - 15
            
            text_positions[(text_x, text_y)] = True
            
            # Add text with better background
            ax.text(text_x, text_y, text, fontsize=9, color=color, 
                   fontweight='bold', verticalalignment='bottom',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.95, edgecolor=color))
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='green', label='High Confidence (≥0.8)'),
            mpatches.Patch(color='blue', label='Good Confidence (0.6-0.8)'),
            mpatches.Patch(color='orange', label='Fair Confidence (0.4-0.6)'),
            mpatches.Patch(color='red', label='Low Confidence (<0.4)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_all_images(self):
        """Visualize OCR results for all images"""
        print("Starting OCR visualization...")
        
        # Load data
        df = self.load_cleaned_data()
        
        # Get image files
        image_files = self.get_image_files()
        print(f"Found {len(image_files)} images to visualize")
        
        # Visualize each image
        for image_file in image_files:
            print(f"Processing: {image_file}")
            save_path = os.path.join(self.visualization_dir, f"{image_file.split('.')[0]}_ocr_result.jpg")
            self.visualize_image_ocr(image_file, df, save_path)
        
        print(f"\nVisualization completed!")
        print(f"Results saved in: {self.visualization_dir}")
        print(f"Generated {len(image_files)} individual visualizations")

def main():
    """Main function"""
    visualizer = OCRVisualizer()
    visualizer.visualize_all_images()

if __name__ == "__main__":
    main()