#!/usr/bin/env python3
"""
OCR Accuracy Calculator
Calculate accuracy of OCR results against ground truth annotations
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re


class OCRAccuracyCalculator:
    def __init__(self):
        self.ground_truth_dir = Path("data/ground_truth")
        self.ocr_results_dir = Path("data/results/ocr_baseline")
        self.output_dir = Path("data/results/ocr_baseline")
        
    def calculate_accuracy(self) -> Dict[str, Any]:
        """
        Calculate accuracy for all OCR results against ground truth
        """
        print("Calculating OCR accuracy...")
        
        ground_truth = self.load_ground_truth()
        if not ground_truth:
            print("No ground truth found")
            return {}
        
        ocr_results = self.load_ocr_results()
        if not ocr_results:
            print("No OCR results found")
            return {}
        all_metrics = {}
        
        for image_name, gt_data in ground_truth.items():
            print(f"\nProcessing: {image_name}")
            
            ocr_key = None
            for key in ocr_results.keys():
                if image_name.replace('.jpg', '') in key or key.replace('_corrected', '') in image_name:
                    ocr_key = key
                    break
            
            if not ocr_key:
                print(f" No OCR result found for {image_name}")
                continue
                
            ocr_data = ocr_results[ocr_key]
            
            metrics = self.calculate_image_metrics(gt_data, ocr_data, image_name)
            all_metrics[image_name] = metrics
            
            self.print_image_summary(image_name, metrics)
        
        overall_metrics = self.calculate_overall_metrics(all_metrics)
        
        self.save_accuracy_results(all_metrics, overall_metrics)
        
        return overall_metrics
    
    def load_ground_truth(self) -> Dict[str, Any]:
        gt_file = self.ground_truth_dir / "annotations.json"
        if not gt_file.exists():
            print(f"Ground truth file not found: {gt_file}")
            return {}
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_ocr_results(self) -> Dict[str, Any]:
        ocr_file = self.ocr_results_dir / "ocr_baseline_results.json"
        if not ocr_file.exists():
            print(f"OCR results file not found: {ocr_file}")
            return {}
        
        with open(ocr_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_image_metrics(self, gt_data: Dict, ocr_data: Dict, image_name: str) -> Dict[str, Any]:
        metrics = {
            'image_name': image_name,
            'store_name_accuracy': 0.0,
            'date_accuracy': 0.0,
            'total_accuracy': 0.0,
            'tax_accuracy': 0.0,
            'items_accuracy': {'overall_accuracy': 0.0},
            'price_accuracy': 0.0,
            'overall_accuracy': 0.0
        }
        
        weights = {
            'store_name': 0.1,
            'date': 0.04,
            'total': 0.05,
            'tax': 0.01,
            'items': 0.4,
            'price': 0.4
        }
        
        if gt_data.get('store_name', '').lower() not in ['unknown', 'unknown store']:
            metrics['store_name_accuracy'] = self.calculate_text_accuracy(gt_data.get('store_name', ''), ocr_data.get('store_name', ''))
        
        if gt_data.get('date', '').lower() not in ['unknown', 'unknown date']:
            metrics['date_accuracy'] = self.calculate_text_accuracy(gt_data.get('date', ''), ocr_data.get('date', ''))
        
        if gt_data.get('total', 0) != 0:
            metrics['total_accuracy'] = self.calculate_numeric_accuracy(gt_data.get('total', 0), ocr_data.get('total_amount', 0))
        
        if gt_data.get('tax', 0) != 0:
            metrics['tax_accuracy'] = self.calculate_numeric_accuracy(gt_data.get('tax', 0), ocr_data.get('tax', 0))
        
        metrics['items_accuracy'] = self.calculate_items_accuracy(gt_data.get('items', []), ocr_data.get('items', []))
        metrics['price_accuracy'] = self.calculate_price_accuracy(gt_data.get('items', []), ocr_data.get('items', []))
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        if gt_data.get('store_name', '').lower() not in ['unknown', 'unknown store']:
            weighted_sum += metrics['store_name_accuracy'] * weights['store_name']
            total_weight += weights['store_name']
        
        if gt_data.get('date', '').lower() not in ['unknown', 'unknown date']:
            weighted_sum += metrics['date_accuracy'] * weights['date']
            total_weight += weights['date']
        
        if gt_data.get('total', 0) != 0:
            weighted_sum += metrics['total_accuracy'] * weights['total']
            total_weight += weights['total']
        
        if gt_data.get('tax', 0) != 0:
            weighted_sum += metrics['tax_accuracy'] * weights['tax']
            total_weight += weights['tax']
        
        weighted_sum += metrics['items_accuracy']['overall_accuracy'] * weights['items']
        weighted_sum += metrics['price_accuracy'] * weights['price']
        total_weight += weights['items'] + weights['price']
        
        if total_weight > 0:
            metrics['overall_accuracy'] = weighted_sum / total_weight
        else:
            metrics['overall_accuracy'] = 0.0
        
        return metrics
    
    def calculate_text_accuracy(self, gt_text: str, ocr_text: str) -> float:
        if not gt_text or not ocr_text:
            return 0.0
        
        fallback_values = ['unknown store', 'unknown date', 'unknown', 'n/a', 'not found']
        ocr_lower = ocr_text.lower().strip()
        if ocr_lower in fallback_values:
            return 0.0
        
        gt_norm = self.normalize_text(gt_text)
        ocr_norm = self.normalize_text(ocr_text)
        
        if gt_norm == ocr_norm:
            return 1.0
        
        common_chars = sum(1 for c in gt_norm if c in ocr_norm)
        total_chars = max(len(gt_norm), len(ocr_norm))
        
        return common_chars / total_chars if total_chars > 0 else 0.0
    
    def calculate_numeric_accuracy(self, gt_value: float, ocr_value: float) -> float:
        if gt_value == 0 and ocr_value == 0:
            return 1.0
        
        if gt_value == 0 or ocr_value == 0:
            return 0.0
        
        tolerance = 0.01
        if abs(gt_value - ocr_value) <= tolerance:
            return 1.0
        
        error = abs(gt_value - ocr_value) / max(gt_value, ocr_value)
        return max(0.0, 1.0 - error)
    
    def calculate_price_accuracy(self, gt_items: List[Dict], ocr_items: List[Dict]) -> float:
        if not gt_items and not ocr_items:
            return 1.0
        
        if not gt_items or not ocr_items:
            return 0.0
        
        gt_prices = [item.get('price', 0) for item in gt_items if item.get('price') is not None and item.get('price', 0) > 0]
        ocr_prices = [item.get('price', 0) for item in ocr_items if item.get('price') is not None and item.get('price', 0) > 0]
        
        if not gt_prices and not ocr_prices:
            return 1.0
        
        if not gt_prices or not ocr_prices:
            return 0.0
        
        price_accuracies = []
        for gt_item in gt_items:
            gt_price = gt_item.get('price', 0)
            if gt_price is None or gt_price <= 0:
                continue
                
            best_price_accuracy = 0.0
            for ocr_item in ocr_items:
                ocr_price = ocr_item.get('price', 0)
                if ocr_price is None or ocr_price <= 0:
                    continue
                
                gt_name = self.normalize_text(gt_item.get('name', ''))
                ocr_name = self.normalize_text(ocr_item.get('name', ''))
                name_similarity = self.calculate_text_accuracy(gt_name, ocr_name)
                
                if name_similarity > 0.3:
                    price_accuracy = self.calculate_numeric_accuracy(gt_price, ocr_price)
                    best_price_accuracy = max(best_price_accuracy, price_accuracy)
            
            price_accuracies.append(best_price_accuracy)
        
        return sum(price_accuracies) / len(price_accuracies) if price_accuracies else 0.0
    
    def calculate_items_accuracy(self, gt_items: List[Dict], ocr_items: List[Dict]) -> Dict[str, Any]:
        if not gt_items and not ocr_items:
            return {'overall_accuracy': 1.0, 'item_count_accuracy': 1.0, 'item_matching': []}
        
        if not gt_items or not ocr_items:
            return {'overall_accuracy': 0.0, 'item_count_accuracy': 0.0, 'item_matching': []}
        
        count_accuracy = min(len(ocr_items), len(gt_items)) / max(len(ocr_items), len(gt_items))
        
        matched_items = []
        unmatched_gt = gt_items.copy()
        unmatched_ocr = ocr_items.copy()
        
        for gt_item in gt_items:
            best_match = None
            best_score = 0.0
            
            for i, ocr_item in enumerate(ocr_items):
                if ocr_item in unmatched_ocr:
                    score = self.calculate_item_similarity(gt_item, ocr_item)
                    if score > best_score:
                        best_score = score
                        best_match = (i, ocr_item)
            
            if best_match and best_score > 0.5:  # Threshold for matching
                matched_items.append({
                    'gt_item': gt_item,
                    'ocr_item': best_match[1],
                    'similarity': best_score
                })
                unmatched_ocr.remove(best_match[1])
                unmatched_gt.remove(gt_item)
        
        if matched_items:
            avg_similarity = sum(item['similarity'] for item in matched_items) / len(matched_items)
        else:
            avg_similarity = 0.0
        
        overall_accuracy = (count_accuracy * 0.3 + avg_similarity * 0.7)
        
        return {
            'overall_accuracy': overall_accuracy,
            'item_count_accuracy': count_accuracy,
            'avg_similarity': avg_similarity,
            'matched_count': len(matched_items),
            'total_gt_items': len(gt_items),
            'total_ocr_items': len(ocr_items),
            'item_matching': matched_items
        }
    
    def calculate_item_similarity(self, gt_item: Dict, ocr_item: Dict) -> float:
        scores = []
        
        gt_number = str(gt_item.get('item_number', ''))
        ocr_number = str(ocr_item.get('item_number', ''))
        if gt_number and ocr_number:
            number_score = 1.0 if gt_number == ocr_number else 0.0
        else:
            number_score = 0.5  # Partial credit if one is missing
        scores.append(number_score)
        
        gt_name = self.normalize_text(gt_item.get('name', ''))
        ocr_name = self.normalize_text(ocr_item.get('name', ''))
        name_score = self.calculate_text_accuracy(gt_name, ocr_name)
        scores.append(name_score)
        
        gt_price = gt_item.get('price', 0)
        ocr_price = ocr_item.get('price', 0)
        price_score = self.calculate_numeric_accuracy(gt_price, ocr_price) if ocr_price else 0.0
        scores.append(price_score)
        
        weights = [0.2, 0.5, 0.3]  # number, name, price
        return sum(s * w for s, w in zip(scores, weights))
    
    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
    
    def calculate_overall_metrics(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        if not all_metrics:
            return {}
        
        store_name_accuracies = [m['store_name_accuracy'] for m in all_metrics.values() if m['store_name_accuracy'] > 0]
        date_accuracies = [m['date_accuracy'] for m in all_metrics.values() if m['date_accuracy'] > 0]
        total_accuracies = [m['total_accuracy'] for m in all_metrics.values() if m['total_accuracy'] > 0]
        tax_accuracies = [m['tax_accuracy'] for m in all_metrics.values() if m['tax_accuracy'] > 0]
        
        overall = {
            'total_images': len(all_metrics),
            'avg_store_name_accuracy': sum(store_name_accuracies) / len(store_name_accuracies) if store_name_accuracies else 0.0,
            'avg_date_accuracy': sum(date_accuracies) / len(date_accuracies) if date_accuracies else 0.0,
            'avg_total_accuracy': sum(total_accuracies) / len(total_accuracies) if total_accuracies else 0.0,
            'avg_tax_accuracy': sum(tax_accuracies) / len(tax_accuracies) if tax_accuracies else 0.0,
            'avg_items_accuracy': sum(m['items_accuracy']['overall_accuracy'] for m in all_metrics.values()) / len(all_metrics),
            'avg_price_accuracy': sum(m['price_accuracy'] for m in all_metrics.values()) / len(all_metrics),
            'avg_overall_accuracy': sum(m['overall_accuracy'] for m in all_metrics.values()) / len(all_metrics)
        }
        
        return overall
    
    def print_image_summary(self, image_name: str, metrics: Dict[str, Any]):
        print(f"  Store Name: {metrics['store_name_accuracy']:.3f}")
        print(f"  Date: {metrics['date_accuracy']:.3f}")
        print(f"  Total: {metrics['total_accuracy']:.3f}")
        print(f"  Tax: {metrics['tax_accuracy']:.3f}")
        print(f"  Items: {metrics['items_accuracy']['overall_accuracy']:.3f}")
        print(f"  Price: {metrics['price_accuracy']:.3f}")
        print(f"  Overall: {metrics['overall_accuracy']:.3f}")
    
    def save_accuracy_results(self, all_metrics: Dict[str, Any], overall_metrics: Dict[str, Any]):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        detailed_file = self.output_dir / "accuracy_detailed.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        
        summary_file = self.output_dir / "accuracy_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_metrics, f, indent=2, ensure_ascii=False)
        
        csv_data = []
        for image_name, metrics in all_metrics.items():
            row = {
                'image_name': image_name,
                'store_name_accuracy': metrics['store_name_accuracy'],
                'date_accuracy': metrics['date_accuracy'],
                'total_accuracy': metrics['total_accuracy'],
                'tax_accuracy': metrics['tax_accuracy'],
                'items_accuracy': metrics['items_accuracy']['overall_accuracy'],
                'price_accuracy': metrics['price_accuracy'],
                'overall_accuracy': metrics['overall_accuracy']
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / "accuracy_summary.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved:")
        print(f"  Detailed: {detailed_file}")
        print(f"  Summary: {summary_file}")
        print(f"  CSV: {csv_file}")


def main():
    calculator = OCRAccuracyCalculator()
    results = calculator.calculate_accuracy()
    
    if results:
        print(f"\nOverall OCR Accuracy Results:")
        print(f"  Total Images: {results['total_images']}")
        print(f"  Store Name: {results['avg_store_name_accuracy']:.3f}")
        print(f"  Date: {results['avg_date_accuracy']:.3f}")
        print(f"  Total Amount: {results['avg_total_accuracy']:.3f}")
        print(f"  Tax: {results['avg_tax_accuracy']:.3f}")
        print(f"  Items: {results['avg_items_accuracy']:.3f}")
        print(f"  Price: {results['avg_price_accuracy']:.3f}")
        print(f"  Overall: {results['avg_overall_accuracy']:.3f}")


if __name__ == "__main__":
    main()
