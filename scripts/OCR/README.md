# OCR Module

This module handles receipt text extraction using a region-based approach.

## Architecture

### 1. Region Detection (`region_detector.py`)
- **Purpose**: Detect different regions in receipt images
- **Regions**: Header, Items, Total, Footer
- **Method**: Image preprocessing + contour detection + position-based classification

### 2. Text Extraction (`text_extractor.py`)
- **Purpose**: Extract text from each region using OCR
- **Method**: Region-specific OCR parameters + intelligent text processing
- **Features**: Text merging, confidence scoring, pattern matching

### 3. OCR Baseline (`ocr_baseline.py`)
- **Purpose**: Complete OCR pipeline with evaluation
- **Features**: Region detection → Text extraction → Result combination → Accuracy calculation

## Key Features

### Region-Based Processing
- **Header**: Store name and basic info (higher OCR thresholds)
- **Items**: Product names and prices (lower thresholds for more text)
- **Total**: Amounts and totals (optimized for numbers)
- **Footer**: Date and other info (optimized for small text)

### Text Processing
- **Fragmentation Handling**: Merge split words (e.g., "tom" + "ato" → "tomato")
- **Pattern Matching**: Extract dates, prices, store names
- **Confidence Scoring**: Weight results by OCR confidence
- **Error Correction**: Common OCR error fixes

### Evaluation
- **Accuracy Metrics**: Store name, amount, date, overall
- **Confidence Analysis**: Per-region and overall confidence
- **Result Visualization**: Region boundaries and text overlays

## Usage

### Basic Usage
```python
from scripts.ocr.ocr_baseline import OCRBaseline

# Initialize
ocr = OCRBaseline()

# Process single image
result = ocr.process_single_image("data/corrected_images/8_corrected.jpg")

# Process all images
results = ocr.process_all_images()
```

### Region Detection Only
```python
from scripts.ocr.region_detector import ReceiptRegionDetector

detector = ReceiptRegionDetector()
regions = detector.detect_regions("data/corrected_images/8_corrected.jpg")
detector.visualize_regions("data/corrected_images/8_corrected.jpg")
```

### Text Extraction Only
```python
from scripts.ocr.text_extractor import ReceiptTextExtractor

extractor = ReceiptTextExtractor()
text_results = extractor.extract_text_from_regions(regions, "image_name")
```

## Output

### Files Generated
- `results/ocr/ocr_baseline_results.json` - Detailed results
- `results/ocr/ocr_baseline_summary.csv` - Summary table
- `results/ocr/{image_name}_regions_visualization.jpg` - Region visualization
- `results/ocr/{image_name}_{region}.jpg` - Individual region images

### Result Format
```json
{
  "image_id": {
    "store_name": "Store 5",
    "total_amount": 12.34,
    "date": "2024-01-15",
    "items": [
      {"name": "tomato", "price": 2.50, "confidence": 0.85},
      {"name": "bread", "price": 3.99, "confidence": 0.92}
    ],
    "confidence": 0.88
  }
}
```

## Advantages

1. **Higher Accuracy**: Region-specific processing improves OCR quality
2. **Better Text Merging**: Handles fragmented text more effectively
3. **Structured Output**: Organized by receipt sections
4. **Confidence Scoring**: Quantifies extraction reliability
5. **Visualization**: Easy to debug and validate results

## Dependencies

- `opencv-python` - Image processing
- `easyocr` - OCR engine
- `pandas` - Data handling
- `numpy` - Numerical operations
