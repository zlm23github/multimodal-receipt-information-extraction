#!/usr/bin/env python3
"""
OCR-only Baseline for Receipt Text Extraction
Complete OCR pipeline with region detection and text extraction
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import json

from region_detector import ReceiptRegionDetector
from text_extractor import ReceiptTextExtractor


class OCRBaseline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_images_dir = self.data_dir / "corrected_images"
        self.results_dir = self.data_dir / "results" / "ocr_baseline"
        self.visualization_dir = self.data_dir / "results" / "visualization" / "ocr_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        self.region_detector = ReceiptRegionDetector()
        self.text_extractor = ReceiptTextExtractor()
        
        print("OCR Baseline initialized")
    
    def process_single_image(self, image_path: str) -> Dict:
        """
        Process a single image through the complete OCR pipeline
        """
        print(f"\n{'='*60}")
        print(f"Processing: {Path(image_path).name}")
        print(f"{'='*60}")
        
        print("Step 1: Detecting regions...")
        regions = self.region_detector.detect_regions(image_path)
        
        if not regions:
            print("No regions detected")
            return {
                'store_name': "No Regions",
                'total_amount': 0.0,
                'date': "No Date",
                'items': [],
                'confidence': 0.0
            }
        
        print("Step 2: Extracting text from regions...")
        text_results = self.text_extractor.extract_text_from_regions(regions, Path(image_path).stem)
        
        print("Step 3: Combining results...")
        combined_info = self.combine_region_results(text_results)
        
        overall_confidence = self.calculate_overall_confidence(text_results)
        combined_info['confidence'] = overall_confidence
        
        print(f"Processing completed (confidence: {overall_confidence:.3f})")
        
        return combined_info
    
    def combine_region_results(self, text_results: Dict[str, Dict]) -> Dict:
        """
        Combine results from different regions
        """
        combined = {
            'store_name': "Unknown Store",
            'total_amount': 0.0,
            'date': "Unknown Date",
            'items': []
        }
        
        if 'header' in text_results:
            header_info = text_results['header'].get('processed_text', {})
            combined['store_name'] = header_info.get('store_name', "Unknown Store")
        
        if 'items' in text_results:
            items_info = text_results['items'].get('processed_text', {})
            combined['items'] = items_info.get('items', [])
        
        if 'total' in text_results:
            total_info = text_results['total'].get('processed_text', {})
            combined['total_amount'] = total_info.get('total_amount', 0.0)
        
        if 'footer' in text_results:
            footer_info = text_results['footer'].get('processed_text', {})
            combined['date'] = footer_info.get('date', "Unknown Date")
        
        return combined
    
    def calculate_overall_confidence(self, text_results: Dict[str, Dict]) -> float:
        """
        Calculate overall confidence from all regions
        """
        confidences = []
        for region_data in text_results.values():
            if 'confidence' in region_data:
                confidences.append(region_data['confidence'])
        
        return np.mean(confidences) if confidences else 0.0
    
    def process_all_images(self) -> Dict[str, Dict]:
        """
        Process all images in the corrected_images directory
        """
        print("Starting OCR-only baseline processing...")
        
        image_files = list(self.raw_images_dir.glob("*.jpg")) + list(self.raw_images_dir.glob("*.png"))
        
        if not image_files:
            print("No images found in corrected_images directory")
            return {}
        
        print(f"Found {len(image_files)} images to process")
        
        results = {}
        
        for img_file in image_files:
            try:
                result = self.process_single_image(str(img_file))
                results[img_file.stem] = result
                
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                results[img_file.stem] = {
                    'store_name': "Error",
                    'total_amount': 0.0,
                    'date': "Error",
                    'items': [],
                    'confidence': 0.0
                }
        
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, Dict]):
        """
        Save OCR baseline results
        """
        json_file = self.results_dir / "ocr_baseline_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved: {json_file}")
        
        summary_data = []
        for image_id, info in results.items():
            summary_data.append({
                'image_id': image_id,
                'store_name': info['store_name'],
                'total_amount': info['total_amount'],
                'date': info['date'],
                'items_count': len(info['items']),
                'items': '; '.join([f"{item['name']}:${item['price']:.2f}" if item['price'] is not None else f"{item['name']}:N/A" for item in info['items']]),
                'confidence': info['confidence']
            })
        
        df = pd.DataFrame(summary_data)
        csv_file = self.results_dir / "ocr_baseline_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"Summary saved: {csv_file}")
    
    def calculate_accuracy(self, extracted_info: Dict, ground_truth: Dict) -> Dict:
        """
        Calculate accuracy against ground truth
        """
        if not ground_truth:
            return {
                'store_name': 0.0,
                'total_amount': 0.0,
                'date': 0.0,
                'overall': 0.0
            }
        
        accuracies = {}
        
        store_gt = ground_truth.get('store_name', '').lower()
        store_pred = extracted_info.get('store_name', '').lower()
        accuracies['store_name'] = 1.0 if store_gt == store_pred else 0.0
        
        amount_gt = ground_truth.get('total_amount', 0.0)
        amount_pred = extracted_info.get('total_amount', 0.0)
        if amount_gt > 0:
            tolerance = abs(amount_gt - amount_pred) / amount_gt
            accuracies['total_amount'] = 1.0 if tolerance <= 0.15 else 0.0
        else:
            accuracies['total_amount'] = 0.0
        
        date_gt = ground_truth.get('date', '')
        date_pred = extracted_info.get('date', '')
        accuracies['date'] = 1.0 if date_gt == date_pred else 0.0
        
        accuracies['overall'] = np.mean(list(accuracies.values()))
        
        return accuracies
    
    def run_baseline_evaluation(self, ground_truth: Dict[str, Dict] = None) -> Dict:
        """
        Run OCR baseline with evaluation
        """
        print("Running OCR baseline evaluation...")
        
        results = self.process_all_images()
        
        if not results:
            print("No results to evaluate")
            return {}
        
        if ground_truth:
            print("\nCalculating accuracies...")
            accuracies = {}
            for image_id, result in results.items():
                if image_id in ground_truth:
                    accuracies[image_id] = self.calculate_accuracy(result, ground_truth[image_id])
            
            if accuracies:
                avg_accuracy = {
                    'store_name': np.mean([acc['store_name'] for acc in accuracies.values()]),
                    'total_amount': np.mean([acc['total_amount'] for acc in accuracies.values()]),
                    'date': np.mean([acc['date'] for acc in accuracies.values()]),
                    'overall': np.mean([acc['overall'] for acc in accuracies.values()])
                }
                
                print(f"\nAverage Accuracy:")
                print(f"   Store Name: {avg_accuracy['store_name']:.3f}")
                print(f"   Total Amount: {avg_accuracy['total_amount']:.3f}")
                print(f"   Date: {avg_accuracy['date']:.3f}")
                print(f"   Overall: {avg_accuracy['overall']:.3f}")
                
                accuracy_file = self.results_dir / "ocr_accuracy_results.json"
                with open(accuracy_file, 'w') as f:
                    json.dump({
                        'individual_accuracies': accuracies,
                        'average_accuracy': avg_accuracy
                    }, f, indent=2, default=str)
                print(f"Accuracy results saved: {accuracy_file}")
        
        return results


def main():
    """Test OCR baseline"""
    ocr_baseline = OCRBaseline()
    
    results = ocr_baseline.process_all_images()
    
    if results:
        print(f"\n🎉 OCR baseline processing completed!")
        print(f"Processed {len(results)} images")
        
        for image_id, info in results.items():
            print(f"\n{image_id}:")
            print(f"   Store: {info['store_name']}")
            print(f"   Amount: ${info['total_amount']:.2f}")
            print(f"   Date: {info['date']}")
            print(f"   Items: {len(info['items'])}")
            print(f"   Confidence: {info['confidence']:.3f}")
    else:
        print("No results generated")


if __name__ == "__main__":
    main()
