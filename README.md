# Transformer Scaling Laws on SVG Data

This repository contains a complete, end-to-end machine learning pipeline for training decoder-only Transformer language models on Scalable Vector Graphics (SVG) data. The project investigates whether the scaling laws and capabilities observed in natural language generation translate to the structured, geometric, and deterministic XML modality of SVGs.

For a detailed analysis of the results, architectures, and findings, please read the [**Final Report (REPORT_Final.md)**](REPORT_Final.md).

---

## Project Overview

The pipeline is designed to be fully reproducible and runs through four distinct stages:

1. **Data Collection & Preparation (`01` - `05`)**
   - Downloads over 430,000 raw SVG files from HuggingFace (icons, fonts, emoji).
   - Cleans and normalizes the XML, filters outliers, and removes structural bloat.
   - Trains a custom Byte-Level BPE tokenizer with a constrained 2,048 vocabulary.
   - Generates a ~110.5M token dataset split into Train/Val/Test.

2. **Standard Parameterization Scaling Study (`06` - `08`)**
   - Performs a learning rate sweep on a Tiny model (1.2M params).
   - Trains 5 model sizes (Tiny, Small, Medium, Large, XL up to 87M params) using Standard Parameterization (SP).
   - Evaluates whether fixed hyperparameters can successfully transfer across widths (spoiler: they diverge).

3. **Maximal Update Parameterization (μP) Scaling Study (`09` - `11`)**
   - Implements the `mup` package to decouple learning rate from model width.
   - Repeats the LR sweep and 5-model scaling study.
   - Successfully demonstrates reliable hyperparameter transfer and derives an empirical scaling law for SVGs.

4. **Extended Training & Generation Evaluation (`12` - `14`)**
   - Takes the best-performing model (μP XL) and trains it for 15 epochs.
   - Generates unconditional, prefix-conditioned, and temperature-varied SVG samples.
   - Evaluates samples for XML validity, structural correctness, and renderability (achieving 93-100% validity).

---

## Pipeline Scripts

The project is structured into 14 sequential, minimalist Python scripts that strictly adhere to a clear, explicit "computer science student" aesthetic.

| Script | Description |
|--------|-------------|
| `01_download_data.py` | Downloads SVG datasets from HuggingFace. |
| `02_clean_normalize.py` | Parses, normalizes coordinates, and filters extreme SVGs. |
| `03_train_tokenizer.py` | Trains the 2,048-vocab Byte-Level BPE tokenizer. |
| `04_tokenize_split.py` | Tokenizes the corpus and splits into Train/Val/Test numpy arrays. |
| `05_statistics.py` | Generates dataset statistics, histograms, and complexity percentiles. |
| `06_lr_sweep.py` | Performs an LR sweep for the SP Tiny model. |
| `07_train_all_models.py`| Trains all 5 SP models for 1 epoch using the best LR. |
| `08_scaling_analysis.py`| Analyzes SP scaling (and divergence). |
| `09_mup_lr_sweep.py` | Performs an LR sweep for the μP Tiny model. |
| `10_mup_train_all.py` | Trains all 5 μP models for 1 epoch using the best LR. |
| `11_scaling_analysis.py`| Derives scaling laws comparing SP vs μP. |
| `12_train_best.py` | Trains the μP XL model for 15 epochs. |
| `13_generate_samples.py`| Uses the trained XL model to generate novel SVGs with varying settings. |
| `14_evaluate_report.py` | Renders the generated SVGs, computes perplexity, and writes the final evaluation. |

There are also helper architectures:
- `model.py` and `train.py`: Standard Parameterization implementation.
- `model_mup.py` and `train_mup.py`: Maximal Update Parameterization implementation.

---

## Environment Setup

1. **Python version**: Python 3.10+ recommended.
2. **Dependencies**: 
   - `torch`, `numpy`, `matplotlib`, `sentencepiece`, `datasets`
   - `lxml` (for robust XML parsing)
   - `cairosvg` (for rendering SVGs to PNG)
   - `mup` (for Maximal Update Parameterization)

*(Note: Windows users may require Cairo/GTK+ libraries installed system-wide for `cairosvg` to function properly).*

## How to Run

The scripts are designed to be run sequentially from `01` to `14`. 
The pipeline expects to be run in an environment with a capable GPU (NVIDIA A100 or equivalent) due to the heavy computational requirements of the XL model training.

```bash
python 01_download_data.py
python 02_clean_normalize.py
# ... and so on
```

Output checkpoints, plots, rendered samples, and Markdown reports will be generated in the `checkpoints/`, `data/`, and `generated/` directories.

---

## Author
*Ananya Singh*
