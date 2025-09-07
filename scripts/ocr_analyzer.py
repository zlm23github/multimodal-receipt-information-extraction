#!/usr/bin/env python3
"""
Simple OCR Results Analyzer
Two main functions: record results and analyze performance
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def record_ocr_result(ocr_file_path, experiment_name=None):
    """
    Record OCR results with basic information
    
    Args:
        ocr_file_path: Path to OCR results CSV file
        experiment_name: Name for this experiment (optional)
    
    Returns:
        dict: Recorded experiment data
    """
    # Read OCR results
    df = pd.read_csv(ocr_file_path)
    
    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Calculate performance metrics
    total_texts = len(df)
    avg_confidence = df['confidence'].mean()
    max_confidence = df['confidence'].max()
    min_confidence = df['confidence'].min()
    
    # Confidence distribution
    excellent = len(df[df['confidence'] >= 0.9])
    good = len(df[(df['confidence'] >= 0.7) & (df['confidence'] < 0.9)])
    fair = len(df[(df['confidence'] >= 0.5) & (df['confidence'] < 0.7)])
    poor = len(df[df['confidence'] < 0.5])
    
    # Count unique images
    unique_images = df['image_id'].nunique() if 'image_id' in df.columns else 0
    
    # Identify issues
    issues = []
    if avg_confidence < 0.5:
        issues.append("Very low average confidence")
    if poor > total_texts * 0.3:
        issues.append("High percentage of poor recognition")
    if max_confidence - min_confidence > 0.8:
        issues.append("High variance in recognition quality")
    
    # Create experiment record
    experiment_record = {
        'experiment_info': {
            'name': experiment_name,
            'recorded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ocr_file': ocr_file_path,
            'total_images': unique_images,
            'total_texts': total_texts
        },
        'performance_metrics': {
            'avg_confidence': round(avg_confidence, 3),
            'max_confidence': round(max_confidence, 3),
            'min_confidence': round(min_confidence, 3),
            'excellent_count': excellent,
            'excellent_rate': round(excellent / total_texts, 3),
            'good_count': good,
            'good_rate': round(good / total_texts, 3),
            'fair_count': fair,
            'fair_rate': round(fair / total_texts, 3),
            'poor_count': poor,
            'poor_rate': round(poor / total_texts, 3)
        },
        'quality_assessment': {
            'overall_quality': 'Excellent' if avg_confidence >= 0.8 else 
                             'Good' if avg_confidence >= 0.6 else
                             'Fair' if avg_confidence >= 0.4 else 'Poor',
            'issues_found': issues,
            'needs_improvement': avg_confidence < 0.6
        }
    }
    
    
    # Save experiment record
    record_file = f"data/analysis/{experiment_name}_record.json"
    os.makedirs("data/analysis", exist_ok=True)
    
    import json
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert the experiment record
    experiment_record = convert_numpy_types(experiment_record)
    
    with open(record_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_record, f, ensure_ascii=False, indent=2)
    
    # Also save detailed results with experiment info
    df['experiment_name'] = experiment_name
    df['recorded_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    detailed_file = f"data/analysis/{experiment_name}_detailed.csv"
    df.to_csv(detailed_file, index=False, encoding='utf-8')
    
    print(f"✅ OCR experiment recorded:")
    print(f"  Record file: {record_file}")
    print(f"  Detailed results: {detailed_file}")
    print(f"  Overall quality: {experiment_record['quality_assessment']['overall_quality']}")
    print(f"  Average confidence: {avg_confidence:.3f}")
    
    return experiment_record

def analyze_performance(ocr_file_path):
    """
    Analyze OCR performance metrics
    
    Args:
        ocr_file_path: Path to OCR results CSV file
    
    Returns:
        dict: Performance metrics
    """
    # Read OCR results
    df = pd.read_csv(ocr_file_path)
    
    # Basic metrics
    total_texts = len(df)
    avg_confidence = df['confidence'].mean()
    max_confidence = df['confidence'].max()
    min_confidence = df['confidence'].min()
    
    # Confidence distribution
    excellent = len(df[df['confidence'] >= 0.9])
    good = len(df[(df['confidence'] >= 0.7) & (df['confidence'] < 0.9)])
    fair = len(df[(df['confidence'] >= 0.5) & (df['confidence'] < 0.7)])
    poor = len(df[df['confidence'] < 0.5])
    
    # Print results
    print(f"\n=== OCR Performance Analysis ===")
    print(f"Total texts: {total_texts}")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Highest confidence: {max_confidence:.3f}")
    print(f"Lowest confidence: {min_confidence:.3f}")
    
    print(f"\nConfidence Distribution:")
    print(f"  Excellent (0.9-1.0): {excellent} ({excellent/total_texts*100:.1f}%)")
    print(f"  Good (0.7-0.9): {good} ({good/total_texts*100:.1f}%)")
    print(f"  Fair (0.5-0.7): {fair} ({fair/total_texts*100:.1f}%)")
    print(f"  Poor (0.0-0.5): {poor} ({poor/total_texts*100:.1f}%)")
    
    # Quality assessment
    if avg_confidence >= 0.8:
        quality = "Excellent"
    elif avg_confidence >= 0.6:
        quality = "Good"
    elif avg_confidence >= 0.4:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"\nOverall Quality: {quality}")
    
    # Return metrics
    return {
        'total_texts': total_texts,
        'avg_confidence': avg_confidence,
        'max_confidence': max_confidence,
        'min_confidence': min_confidence,
        'excellent_count': excellent,
        'good_count': good,
        'fair_count': fair,
        'poor_count': poor,
        'quality': quality
    }

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple OCR Results Analyzer')
    parser.add_argument('--file', type=str, default='data/ocr_results/ocr_results.csv', help='OCR results file path')
    parser.add_argument('--record', action='store_true', help='Record the results')
    parser.add_argument('--analyze', action='store_true', help='Analyze performance')
    parser.add_argument('--name', type=str, help='Experiment name for recording')
    
    args = parser.parse_args()
    
    if args.record:
        record_ocr_result(args.file, args.name)
    
    if args.analyze:
        analyze_performance(args.file)
    
    if not args.record and not args.analyze:
        # Default: analyze current results
        analyze_performance(args.file)

if __name__ == "__main__":
    main()
