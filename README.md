# Multimodal Receipt Information Extraction

A comprehensive system for extracting and analyzing information from receipt images using multimodal approaches (OCR + Image features). This project demonstrates how multimodal fusion can enhance information extraction accuracy, especially when single-modality OCR performance is low and fragmented.

## 🎯 Project Overview

This project implements a complete pipeline for receipt information extraction and analysis, including:

- **OCR Baseline**: Region-based information extraction using EasyOCR
- **Image Feature Processing**: Deep learning-based feature extraction for visual analysis
- **Multimodal Fusion**: Combining text and image features for improved accuracy
- **Performance Evaluation**: Comprehensive accuracy metrics and comparison

## 🚀 Quick Start

### 1. Environment Setup

First, activate the virtual environment:
```bash
conda activate snapledger
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Complete Workflow

**Step 1: Image Preprocessing**
```bash
python scripts/image_orientation_corrector.py
```
This processes all images in `data/raw_images/` and saves corrected versions to `data/corrected_images/`.

**Step 2: OCR Processing**
```bash
python scripts/OCR/ocr_baseline.py
```
This runs the complete OCR pipeline on all corrected images.

**Step 3: Accuracy Evaluation**
```bash
python scripts/OCR/accuracy_calculator.py
```
This compares OCR results with ground truth annotations.

## 🔧 Individual Component Usage

You can also run individual components separately for testing or debugging:

### Region Detection Only

Detect and visualize receipt regions:
```bash
python scripts/OCR/region_detector.py
```
This will process `data/corrected_images/8_corrected.jpg` and generate region visualizations.

### Text Extraction Only

Extract text from pre-detected regions:
```bash
python scripts/OCR/text_extractor.py
```
This requires regions to be detected first using the region detector.

### Image Orientation Correction Only

Correct image orientation without running OCR:
```bash
python scripts/image_orientation_corrector.py
```
This processes all images in `data/raw_images/` and saves corrected versions.

## 📋 Complete Workflow

1. **Place images** in `data/raw_images/` directory
2. **Run orientation correction**: `python scripts/image_orientation_corrector.py`
3. **Run OCR pipeline**: `python scripts/OCR/ocr_baseline.py`
4. **Evaluate accuracy**: `python scripts/OCR/accuracy_calculator.py`
5. **Check results** in `data/results/` directory

## 📁 Project Structure

```
receipt-multimodal-classification/
├── data/
│   ├── raw_images/              # Original receipt images
│   ├── corrected_images/        # Orientation-corrected images
│   ├── ground_truth/           # Manual annotations
│   ├── image_features/         # Extracted image feature vectors
│   └── results/
│       ├── ocr_baseline/       # OCR results and accuracy
│       ├── image_baseline/     # Image-only results
│       ├── multimodal/         # Multimodal fusion results
│       └── visualization/      # Visual outputs
├── scripts/
│   ├── OCR/                    # OCR processing modules
│   │   ├── region_detector.py  # Receipt region detection
│   │   ├── text_extractor.py   # Text extraction from regions
│   │   ├── ocr_baseline.py     # Complete OCR pipeline
│   │   └── accuracy_calculator.py # Accuracy evaluation
│   ├── image_orientation_corrector.py # Image preprocessing
│   └── IFP/                    # Image Feature Processing
│       ├── feature_extractor.py # Deep learning feature extraction
│       ├── image_baseline.py   # Image-only baseline
│       └── multimodal_fusion.py # Text + Image fusion
├── notebooks/                  # Jupyter notebooks for analysis
└── requirements.txt           # Python dependencies
```

## 🔧 Core Components

### OCR Module (`scripts/OCR/`)

**Region Detection** (`region_detector.py`)
- Detects Header, Items, Total, and Footer regions
- Uses adaptive boundary detection based on text density
- Keyword-based region identification ("Member", "SUBTOTAL")

**Text Extraction** (`text_extractor.py`)
- Region-specific OCR parameters for optimal results
- Intelligent text merging for fragmented words
- Pattern matching for structured data extraction

**OCR Baseline** (`ocr_baseline.py`)
- Complete end-to-end OCR pipeline
- Combines region detection and text extraction
- Generates structured output with confidence scores

**Accuracy Calculator** (`accuracy_calculator.py`)
- Compares OCR results with ground truth annotations
- Weighted accuracy metrics across different fields
- Detailed performance analysis and reporting


### Image Feature Processing (`scripts/IFP/`)

**Feature Extractor** (`feature_extractor.py`)
- Deep learning-based image feature extraction using pre-trained models
- ResNet50/ResNet101 backbone for visual feature representation
- Region-specific feature extraction for different receipt sections

**Image Baseline** (`image_baseline.py`)
- Image-only baseline for information extraction
- Visual pattern recognition for store names, amounts, and items
- Comparison baseline against OCR performance

**Multimodal Fusion** (`multimodal_fusion.py`)
- Combines text features from OCR with image features
- Attention-based fusion mechanisms
- Enhanced accuracy through complementary modalities

## 📊 Usage Examples

### Complete Pipeline

```python
# Step 1: Correct image orientation
from scripts.image_orientation_corrector import ImageOrientationCorrector
corrector = ImageOrientationCorrector()
corrector.process_directory("data/raw_images", "data/corrected_images")

# Step 2: Run OCR pipeline
from scripts.OCR.ocr_baseline import OCRBaseline
ocr = OCRBaseline()
result = ocr.process_single_image("data/corrected_images/8_corrected.jpg")

print(f"Store: {result['store_name']}")
print(f"Total: ${result['total_amount']}")
print(f"Items: {len(result['items'])}")
```

### Individual Components

**Region Detection Only:**
```python
from scripts.OCR.region_detector import ReceiptRegionDetector

detector = ReceiptRegionDetector()
regions = detector.detect_regions("data/corrected_images/8_corrected.jpg")

# Visualize detected regions
detector.visualize_regions("data/corrected_images/8_corrected.jpg")
```

**Text Extraction Only:**
```python
from scripts.OCR.text_extractor import ReceiptTextExtractor

extractor = ReceiptTextExtractor()
text_results = extractor.extract_text_from_regions(regions, "image_name")
```

**Calculate Accuracy:**
```python
from scripts.OCR.accuracy_calculator import OCRAccuracyCalculator

calculator = OCRAccuracyCalculator()
results = calculator.calculate_accuracy()

print(f"Overall Accuracy: {results['avg_overall_accuracy']:.3f}")
print(f"Store Name: {results['avg_store_name_accuracy']:.3f}")
print(f"Items: {results['avg_items_accuracy']:.3f}")
```

## 📈 Performance Metrics

The system uses weighted accuracy evaluation:

- **Store Name**: 10% weight
- **Date**: 4% weight  
- **Total Amount**: 5% weight
- **Tax**: 1% weight
- **Items**: 40% weight
- **Price Accuracy**: 40% weight

### Sample Results

```
Overall OCR Accuracy Results:
  Total Images: 1
  Store Name: 0.000
  Date: 0.000
  Total Amount: 1.000
  Tax: 0.000
  Items: 0.790
  Price: 0.778
  Overall: 0.677
```

## 🛠️ Dependencies

- `opencv-python>=4.6.0` - Image processing
- `easyocr>=1.6.0` - OCR engine
- `pandas>=1.5.0` - Data handling
- `numpy<2.0` - Numerical operations
- `torch>=1.12.0` - Deep learning framework
- `matplotlib>=3.5.0` - Visualization

## 📝 Output Files

### OCR Results
- `ocr_baseline_results.json` - Detailed extraction results
- `ocr_baseline_summary.csv` - Summary table
- `accuracy_detailed.json` - Per-image accuracy metrics
- `accuracy_summary.json` - Overall accuracy statistics

### Image Features
- `image_features.csv` - Extracted feature vectors
- `image_baseline_results.json` - Image-only extraction results
- `multimodal_features.csv` - Combined text and image features

### Visualizations
- `{image_id}_regions_visualization.jpg` - Region boundaries
- `{image_id}_{region}.jpg` - Individual region images
- `{image_id}_raw_text.json` - Raw OCR text elements

## 🔬 Research Context

This project demonstrates the effectiveness of multimodal fusion for receipt text extraction, particularly addressing challenges such as:

- **Poor OCR Quality**: Fragmented text like "tom, ato" instead of "tomato"
- **Garbled Characters**: Chinese characters appearing as "乱码"
- **Low Confidence**: OCR results with insufficient reliability

The multimodal approach combines:
1. **Text Features**: Extracted from OCR results
2. **Image Features**: Visual patterns and layout information
3. **Fusion Strategy**: Intelligent combination for improved accuracy

## 🚧 Development Status

### ✅ Completed
- OCR baseline with region-based processing
- Image orientation correction
- Accuracy evaluation system
- Text extraction and processing

### 🚧 In Progress
- Image feature processing (IFP module)
- Image-only baseline implementation
- Multimodal fusion module

### 📋 Planned
- Advanced text repair algorithms
- Real-time processing capabilities
- Mobile app integration
- Performance optimization

## 📄 License

This project is part of the SnapLedger multimodal AI research initiative.

## 🤝 Contributing

For questions or contributions, please refer to the project documentation in the `scripts/OCR/README.md` file.