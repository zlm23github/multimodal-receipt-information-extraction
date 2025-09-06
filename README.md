# SnapLedger - Receipt Multimodal Classification Project

## Project Overview

SnapLedger is a multimodal AI-based receipt classification system that can automatically identify and categorize various types of receipts, invoices, and documents.

## Project Structure

```
receipt-multimodal-classification/
├── data/                          # Data directory
│   ├── raw_images/               # Original images
│   ├── processed_images/         # Processed images
│   ├── ocr_results/              # OCR recognition results
│   └── metadata/                 # Metadata files
├── scripts/                      # Python scripts
├── notebooks/                    # Jupyter Notebooks
├── experiments/                  # Experiment records
├── reports/                      # Reports and results
└── requirements.txt              # Dependencies list
```

## Environment Setup

### 1. Create Conda Environment

```bash
# Create project-specific environment
conda create -n snapledger python=3.10 -y

# Activate environment
conda activate snapledger

```

### 2. Install Dependencies

#### Day 1: Data Collection Phase
```bash
# Activate environment
conda activate snapledger

# Install basic dependencies
pip install -r requirements.txt

```
