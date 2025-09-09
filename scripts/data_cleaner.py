#!/usr/bin/env python3
"""
OCR Data Cleaning Tool
Clean OCR text, handle special characters, remove noise and duplicates
"""

import pandas as pd
import numpy as np
import re
import os
from typing import Dict

class OCRDataCleaner:
    def __init__(self):
        self.data_dir = "data"
        self.ocr_results_dir = os.path.join(self.data_dir, "ocr_results")
        self.cleaned_data_dir = os.path.join(self.data_dir, "cleaned_data")
        os.makedirs(self.cleaned_data_dir, exist_ok=True)
    
    def load_ocr_data(self) -> pd.DataFrame:
        """Load OCR data"""
        file_path = os.path.join(self.ocr_results_dir, "ocr_results.csv")
        df = pd.read_csv(file_path)
        print(f"Loaded OCR data: {len(df)} records")
        return df
    
    def clean_text(self, text: str) -> Dict:
        """Clean individual text"""
        if pd.isna(text) or text == '':
            return {
                'original_text': text,
                'cleaned_text': '',
                'is_valid': False,
                'cleaning_actions': ['empty_text'],
                'confidence_score': 0
            }
        
        original_text = text
        cleaning_actions = []
        confidence_score = 1.0
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        if cleaned != text:
            cleaning_actions.append('remove_extra_spaces')
        
        # Remove special characters (keep basic punctuation)
        cleaned = re.sub(r'[^\w\s.,!?$%()-]', '', cleaned)
        if cleaned != text:
            cleaning_actions.append('remove_special_chars')
        
        # Remove too short meaningless text
        if len(cleaned) < 2:
            cleaning_actions.append('remove_too_short')
            confidence_score = 0.1
        elif len(cleaned) > 100:
            cleaning_actions.append('truncate_too_long')
            cleaned = cleaned[:100]
            confidence_score = 0.8
        
        # Detect and remove gibberish
        if re.search(r'[^a-zA-Z0-9\s.,!?$%()-]{3,}', cleaned):
            cleaning_actions.append('remove_gibberish')
            confidence_score = 0.2
        
        # Check digit ratio (might be gibberish)
        digit_ratio = len(re.findall(r'\d', cleaned)) / len(cleaned) if len(cleaned) > 0 else 0
        if digit_ratio > 0.8:
            cleaning_actions.append('high_digit_ratio')
            confidence_score *= 0.5
        
        # Check letter ratio
        letter_ratio = len(re.findall(r'[a-zA-Z]', cleaned)) / len(cleaned) if len(cleaned) > 0 else 0
        if letter_ratio < 0.1 and len(cleaned) > 5:
            cleaning_actions.append('low_letter_ratio')
            confidence_score *= 0.3
        
        # Final validation
        is_valid = (
            len(cleaned) >= 2 and 
            confidence_score > 0.3 and 
            len(cleaning_actions) <= 3
        )
        
        return {
            'original_text': original_text,
            'cleaned_text': cleaned,
            'is_valid': is_valid,
            'cleaning_actions': cleaning_actions,
            'confidence_score': confidence_score,
            'text_length': len(cleaned),
            'digit_ratio': digit_ratio,
            'letter_ratio': letter_ratio
        }
    
    def filter_by_confidence(self, df: pd.DataFrame, min_confidence: float = 0.1) -> pd.DataFrame:
        """Filter by confidence threshold"""
        print(f"Filtering by confidence (>= {min_confidence})...")
        
        before_count = len(df)
        df_filtered = df[df['confidence'] >= min_confidence].copy()
        after_count = len(df_filtered)
        
        print(f"After filtering: {after_count} records (removed {before_count - after_count})")
        
        return df_filtered
    
    def clean_dataset(self) -> pd.DataFrame:
        """Clean entire dataset"""
        print("Starting data cleaning...")
        
        # Load data
        df = self.load_ocr_data()
        
        # Clean text
        print("Cleaning text...")
        cleaning_results = df['text'].apply(self.clean_text)
        cleaning_df = pd.DataFrame(cleaning_results.tolist())
        
        # Merge cleaning results
        df_cleaned = df.copy()
        for col in cleaning_df.columns:
            df_cleaned[col] = cleaning_df[col]
        
        # Filter valid text (more lenient)
        print("Filtering valid text...")
        before_count = len(df_cleaned)
        # Keep records with confidence >= 0.3 or text length >= 2
        df_cleaned = df_cleaned[
            (df_cleaned['is_valid'] == True) | 
            (df_cleaned['confidence'] >= 0.3) |
            (df_cleaned['text'].str.len() >= 2)
        ].copy()
        after_count = len(df_cleaned)
        print(f"Valid text: {after_count} records (removed {before_count - after_count})")
        
        # Filter by confidence
        df_cleaned = self.filter_by_confidence(df_cleaned, min_confidence=0.1)
        
        # Save cleaned data
        output_file = os.path.join(self.cleaned_data_dir, "cleaned_ocr_results.csv")
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
    

def main():
    """Main function"""
    # Create cleaner
    cleaner = OCRDataCleaner()
    
    # Execute cleaning
    cleaned_df = cleaner.clean_dataset()
    
    print(f"\nData cleaning completed!")
    print(f"Cleaned data shape: {cleaned_df.shape}")

if __name__ == "__main__":
    main()
