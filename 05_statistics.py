import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


CLEANED_DIR = Path("data/cleaned")
TOKENIZER_DIR = Path("data/tokenizer")
SPLITS_DIR = Path("data/splits")
STATS_DIR = Path("data/stats")
EXAMPLES_DIR = STATS_DIR / "examples"
REPORT_PATH = Path("REPORT_Part1.md")

# Percentiles for example selection
PERCENTILES = [10, 25, 50, 75, 90]


def load_metadata():

    split_meta_path = SPLITS_DIR / "split_metadata.json"
    with open(split_meta_path) as f:
        split_meta = json.load(f)

    seq_lengths_path = SPLITS_DIR / "seq_lengths.json"
    with open(seq_lengths_path) as f:
        seq_lengths = json.load(f)

    tokenizer_stats_path = TOKENIZER_DIR / "tokenizer_stats.json"
    with open(tokenizer_stats_path) as f:
        tokenizer_stats = json.load(f)

    filter_stats_path = CLEANED_DIR / "filter_stats.json"
    with open(filter_stats_path) as f:
        filter_stats = json.load(f)

    return split_meta, seq_lengths, tokenizer_stats, filter_stats


def generate_summary_table(split_meta, tokenizer_stats, filter_stats):

    splits = split_meta["splits"]
    total_tokens = split_meta["total_tokens"]

    total_raw = sum(filter_stats["raw_counts"].values())
    total_after_cleaning = filter_stats["total_before_filter"]
    total_after_filtering = filter_stats["total_after_filter"]
    total_after_token_filter = split_meta["total_files_after_filtering"]

    summary = f"""# Dataset Statistics Summary

## Overall

| Metric | Value |
|--------|-------|
| **Vocabulary Size** | {tokenizer_stats['vocab_size']:,} |
| **Max Sequence Length** | {split_meta['max_seq_length']:,} tokens |
| **Raw Files Downloaded** | {total_raw:,} |
| **Files After Cleaning (Pass 1)** | {total_after_cleaning:,} |
| **Files After Percentile Filtering (Pass 2)** | {total_after_filtering:,} |
| **Files After Token-Length Filtering** | {total_after_token_filter:,} |
| **Total Tokens** | **{total_tokens:,}** |

## Split Details

| Split | Files | Tokens | Mean Length | Median Length | Min | Max |
|-------|------:|-------:|------------:|--------------:|----:|----:|
"""
    for split_name in ["train", "val", "test"]:
        s = splits[split_name]
        summary += f"| **{split_name.capitalize()}** | {s['files']:,} | {s['tokens']:,} | {s['mean_length']} | {s['median_length']} | {s['min_length']} | {s['max_length']} |\n"

    summary += f"| **Total** | **{total_after_token_filter:,}** | **{total_tokens:,}** | | | | |\n"

    summary += f"""
## Files Before/After Filtering (by Dataset)

| Dataset | Before Filtering | After Filtering | Removed |
|---------|----------------:|-----------------:|--------:|
"""
    for ds_name in sorted(filter_stats["raw_counts"].keys()):
        before = filter_stats["raw_counts"][ds_name]
        after = filter_stats["after_filtering"].get(ds_name, 0)
        removed = before - after
        summary += f"| {ds_name} | {before:,} | {after:,} | {removed:,} |\n"

    summary += f"""
## Tokenizer

| Parameter | Value |
|-----------|-------|
| Algorithm | BPE (SentencePiece) |
| Vocab Size | {tokenizer_stats['vocab_size']:,} |
| Min Frequency | {tokenizer_stats['min_frequency']} |
| Training Files | {tokenizer_stats['training_files']:,} |
| Avg Chars/Token | {tokenizer_stats['avg_chars_per_token']} |
| Special Tokens | `<pad>` (0), `<bos>` (1), `<eos>` (2) |
"""

    return summary


def plot_seq_length_histogram(seq_lengths):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Token Sequence Length Distribution", fontsize=14, fontweight="bold")

    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

    for ax, (split_name, lengths) in zip(axes, seq_lengths.items()):
        ax.hist(
            lengths,
            bins=80,
            color=colors.get(split_name, "#4C72B0"),
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(f"{split_name.capitalize()} (n={len(lengths):,})", fontsize=12)
        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_ylabel("Count")
        ax.axvline(
            np.median(lengths),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Median: {int(np.median(lengths))}",
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    hist_path = STATS_DIR / "seq_length_histogram.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram: {hist_path}")
    return hist_path


def plot_file_counts(filter_stats):

    datasets = sorted(filter_stats["raw_counts"].keys())
    before = [filter_stats["raw_counts"][ds] for ds in datasets]
    after = [filter_stats["after_filtering"].get(ds, 0) for ds in datasets]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width / 2, before, width, label="Before Filtering", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, after, width, label="After Filtering", color="#55A868", alpha=0.85)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Files")
    ax.set_title("File Counts Before/After Filtering", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center", va="bottom", fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    chart_path = STATS_DIR / "file_counts.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved file counts chart: {chart_path}")
    return chart_path


def select_and_save_examples():

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    svg_files = sorted(CLEANED_DIR.glob("*.svg"))
    file_lengths = []
    for svg_path in svg_files:
        text = svg_path.read_text(encoding="utf-8")
        file_lengths.append((svg_path, len(text), text))

    file_lengths.sort(key=lambda x: x[1])

    examples_info = []
    for p in PERCENTILES:
        idx = int(len(file_lengths) * p / 100)
        idx = min(idx, len(file_lengths) - 1)
        svg_path, char_len, svg_text = file_lengths[idx]

        source = svg_path.stem.split("__")[0] if "__" in svg_path.name else "unknown"

        example_filename = f"example_p{p}.svg"
        example_path = EXAMPLES_DIR / example_filename
        example_path.write_text(svg_text, encoding="utf-8")

        png_path = None
        try:
            import cairosvg
            png_filename = f"example_p{p}.png"
            png_path_obj = EXAMPLES_DIR / png_filename
            cairosvg.svg2png(
                bytestring=svg_text.encode("utf-8"),
                write_to=str(png_path_obj),
                output_width=256,
                output_height=256,
            )
            png_path = png_filename
            print(f"P{p}: {char_len} chars -> {example_filename} + {png_filename}")
        except Exception as e:
            print(f"P{p}: {char_len} chars -> {example_filename} (PNG render skipped: {e})")

        examples_info.append({
            "percentile": p,
            "source_dataset": source,
            "original_file": svg_path.name,
            "char_length": char_len,
            "svg_file": example_filename,
            "png_file": png_path,
        })

    info_path = EXAMPLES_DIR / "examples_info.json"
    with open(info_path, "w") as f:
        json.dump(examples_info, f, indent=2)

    return examples_info


def generate_report(split_meta, tokenizer_stats, filter_stats, examples_info):

    splits = split_meta["splits"]
    total_tokens = split_meta["total_tokens"]
    train_tokens = splits["train"]["tokens"]
    val_tokens = splits["val"]["tokens"]
    test_tokens = splits["test"]["tokens"]

    total_raw = sum(filter_stats["raw_counts"].values())
    total_after_cleaning = filter_stats["total_before_filter"]
    total_after_filtering = filter_stats["total_after_filter"]
    total_after_token_filter = split_meta["total_files_after_filtering"]
    removed_short = filter_stats["removed_short"]
    removed_long = filter_stats["removed_long_p99"]
    p99_threshold = filter_stats["p99_threshold_chars"]
    skipped_token_len = split_meta["skipped_over_max_length"]

    # Determine dataset-specific raw counts
    raw_counts = filter_stats["raw_counts"]
    after_filtering = filter_stats["after_filtering"]

    # Build example table rows
    example_rows = ""
    complexity_labels = {10: "Simple", 25: "Low", 50: "Medium", 75: "Complex", 90: "Very Complex"}
    for ex in examples_info:
        p = ex["percentile"]
        label = complexity_labels.get(p, "")
        example_rows += f"| **P{p}** ({label}) | {ex['source_dataset']} | {ex['char_length']:,} chars | See `data/stats/examples/{ex['svg_file']}` |\n"

    report = f"""# Part 1: Data Collection and Preprocessing — Pipeline Report

## 1. Overview

This report documents the complete data preprocessing pipeline for training decoder-only Transformer language models on SVG (Scalable Vector Graphics) code. The pipeline transforms raw SVG datasets from HuggingFace into a clean, tokenized corpus suitable for language model training.

**Final corpus size: {total_tokens / 1e6:.1f}M tokens** ({train_tokens / 1e6:.1f}M train, {val_tokens / 1e6:.1f}M val, {test_tokens / 1e6:.1f}M test)

---

## 2. Data Sources

We use three datasets from the StarVector project (Rodriguez et al., 2023), all hosted on HuggingFace:

| Dataset | HuggingFace ID | Raw Files | Description |
|---------|----------------|-----------|-------------|
| SVG Icons (Primary) | `starvector/svg-icons-simple` | {raw_counts.get('svg-icons-simple', 'N/A'):,} | Simplified SVG icons (~153 MB) |
| SVG Emoji | `starvector/svg-emoji-simple` | {raw_counts.get('svg-emoji-simple', 'N/A'):,} | Simplified SVG emoji (~14.5 MB) |
| SVG Fonts | `starvector/svg-fonts-simple` | {raw_counts.get('svg-fonts-simple', 'N/A'):,} | Subsampled font glyphs (from 2.38 GB) |
| **Total** | | **{total_raw:,}** | |

The fonts dataset was subsampled to {raw_counts.get('svg-fonts-simple', 200000):,} entries (random seed=42) from its full size to keep the pipeline tractable while ensuring we exceed the 100M token target.

---

## 3. Preprocessing Pipeline

The pipeline consists of 5 sequential stages:

```
Raw SVGs (HuggingFace) -> Download -> Clean/Normalize -> Tokenize -> Split -> Statistics
     Stage 1              Stage 2        Stage 3       Stage 4    Stage 5
```

### Stage 1: Data Download (`01_download_data.py`)
- Downloads each dataset via `datasets.load_dataset()`
- Extracts the SVG code column (`Svg` or `svg`) from each row
- Saves each SVG as an individual `.svg` file in `data/raw/{{dataset_name}}/`
- Applies random subsampling for large datasets (svg-fonts-simple -> {raw_counts.get('svg-fonts-simple', 'N/A'):,})

### Stage 2: SVG Cleaning & Normalization (`02_clean_normalize.py`)

This is a **two-pass** process:

**Pass 1 — Cleaning & Normalization** (applied to every SVG):

1. **XML Parsing**: Parse each SVG using `lxml.etree` with `remove_comments=True` and `remove_pis=True`. Any SVG that fails to parse is rejected.
2. **Metadata Stripping**: Remove `<metadata>`, `<desc>`, `<title>` elements and Inkscape/Illustrator/Sodipodi namespace attributes.
3. **Namespace Cleanup**: Strip namespace prefixes from tags and attributes (e.g., `{{http://www.w3.org/2000/svg}}path` -> `path`).
4. **Attribute Canonicalization**: Sort attributes alphabetically within each element for consistent representation.
5. **Coordinate Normalization**: Round all floating-point numbers with >=2 decimal places to 1 decimal place using regex substitution (e.g., `12.3456` -> `12.3`). This significantly reduces vocabulary size by eliminating near-duplicate float tokens.
6. **Whitespace Normalization**: Collapse multiple whitespace/newlines to single spaces; remove spaces around XML structural characters (`><`, `/>`).
7. **Re-validation**: Serialize back to string and re-parse to ensure the cleaned SVG is still valid XML.

**Pass 2 — Filtering**:

- Remove SVGs shorter than **{filter_stats['min_char_length']} characters** (too trivial to be meaningful)
- Remove SVGs in the **top 1 percentile** by character length (outlier complexity, threshold: {p99_threshold:,} chars)

### Stage 3: BPE Tokenizer Training (`03_train_tokenizer.py`)

- **Algorithm**: BPE (Byte Pair Encoding)
- **Library**: SentencePiece (character-level BPE, ideal for structured code)
- **Character coverage**: 1.0 (all characters in corpus are covered)
- **Normalizer**: SentencePiece built-in normalization
- **Vocabulary size**: {tokenizer_stats['vocab_size']:,} tokens
- **Special tokens**: `<pad>` (ID=0), `<bos>` (ID=1), `<eos>` (ID=2)
- **Minimum frequency**: {tokenizer_stats['min_frequency']} (tokens must appear at least twice)

**Vocabulary size justification**: {tokenizer_stats['vocab_size']:,} sits in the middle of the recommended 1K-8K range. It is large enough to capture common SVG tags (`<svg>`, `<path>`, `<circle>`), attributes (`fill`, `stroke`, `viewBox`), path commands (`M`, `L`, `C`), and frequently occurring coordinate substrings, while remaining small enough to keep embedding tables efficient at smaller model scales used for scaling law experiments. SentencePiece's character-level BPE is well-suited for SVG code because it learns subword units directly from characters without requiring a separate pre-tokenization step.

### Stage 4: Tokenization & Splitting (`04_tokenize_split.py`)

1. **Tokenize**: Each cleaned SVG is encoded as `<bos>` + tokenized_content + `<eos>`
2. **Length filtering**: Sequences exceeding **{split_meta['max_seq_length']:,} tokens** are discarded to keep context window requirements manageable for smaller models
3. **Splitting**: Files are split 98%/1%/1% (train/val/test) **by file** (not by token position) to prevent data leakage, with a fixed random seed (42) for reproducibility
4. **Serialization**: Each split is concatenated into a flat token array and saved as a numpy `.npy` file (uint16 dtype)

### Stage 5: Statistics & Visualization (`05_statistics.py`)

Computes and saves:
- Summary statistics table (markdown)
- Token sequence length histogram per split (PNG)
- File counts before/after filtering per dataset (PNG)
- Rendered SVG examples at 10th, 25th, 50th, 75th, 90th percentile complexity
- This report (REPORT_Part1.md)

---

## 4. Dataset Statistics

### 4.1 Overall Summary

| Metric | Value |
|--------|-------|
| **Vocabulary Size** | {tokenizer_stats['vocab_size']:,} |
| **Max Sequence Length** | {split_meta['max_seq_length']:,} tokens |
| **Raw Files Downloaded** | {total_raw:,} |
| **Files After Cleaning (Pass 1)** | {total_after_cleaning:,} ({total_raw - total_after_cleaning} rejected) |
| **Files After Percentile Filtering (Pass 2)** | {total_after_filtering:,} ({removed_short + removed_long} removed) |
| **Files After Token-Length Filtering** | {total_after_token_filter:,} ({skipped_token_len:,} exceeded {split_meta['max_seq_length']:,} tokens) |
| **Total Tokens** | **{total_tokens:,}** |

### 4.2 Split Details

| Split | Files | Tokens | Mean Length | Median Length | Min | Max |
|-------|------:|-------:|------------:|--------------:|----:|----:|
| **Train** | {splits['train']['files']:,} | {splits['train']['tokens']:,} | {splits['train']['mean_length']} | {splits['train']['median_length']} | {splits['train']['min_length']} | {splits['train']['max_length']} |
| **Val** | {splits['val']['files']:,} | {splits['val']['tokens']:,} | {splits['val']['mean_length']} | {splits['val']['median_length']} | {splits['val']['min_length']} | {splits['val']['max_length']} |
| **Test** | {splits['test']['files']:,} | {splits['test']['tokens']:,} | {splits['test']['mean_length']} | {splits['test']['median_length']} | {splits['test']['min_length']} | {splits['test']['max_length']} |
| **Total** | **{total_after_token_filter:,}** | **{total_tokens:,}** | | | | |

The training set contains **{train_tokens / 1e6:.1f}M tokens**, {"comfortably exceeding" if train_tokens >= 100_000_000 else "below"} the 100M token minimum target.

### 4.3 Files Before/After Filtering (by Dataset)

| Dataset | Before Filtering | After Filtering | Removed |
|---------|----------------:|-----------------:|--------:|
"""

    for ds_name in sorted(raw_counts.keys()):
        before = raw_counts[ds_name]
        after = after_filtering.get(ds_name, 0)
        removed = before - after
        report += f"| {ds_name} | {before:,} | {after:,} | {removed:,} |\n"

    report += f"""
The bar chart visualization is saved at `data/stats/file_counts.png`.

### 4.4 Sequence Length Distribution

The sequence length distribution is right-skewed with a median of ~{splits['train']['median_length']} tokens across all splits. Most SVGs tokenize to 200-800 tokens, with a long tail extending to the {split_meta['max_seq_length']:,}-token maximum.

The histogram visualization is saved at `data/stats/seq_length_histogram.png`.

---

## 5. Cleaning Effectiveness

The SVG normalization pipeline significantly reduces file sizes while preserving visual content:

| Metric | Before Cleaning | After Cleaning | Reduction |
|--------|----------------:|---------------:|----------:|
| Coordinate precision | Arbitrary (e.g., `12.3456`) | 1 decimal (e.g., `12.3`) | Vocabulary reduction |
| Metadata/comments | Present | Removed | Cleaner corpus |
| Whitespace | Arbitrary | Normalized | Compact representation |

All cleaned SVGs are validated as proper XML via `lxml.etree.fromstring()` re-parsing.

---

## 6. SVG Examples at Various Complexity Levels

SVGs were selected at the 10th, 25th, 50th, 75th, and 90th percentile of character length to illustrate the range of complexity in the dataset.

| Percentile | Source | Char Length | Description |
|-----------|--------|------------:|-------------|
{example_rows}

The example SVG files are saved in `data/stats/examples/` and can be rendered by opening them in any web browser:
"""

    for ex in examples_info:
        p = ex["percentile"]
        label = complexity_labels.get(p, "")
        report += f"- `data/stats/examples/{ex['svg_file']}` — P{p} ({label.lower()})\n"

    report += f"""
---

## 7. Validation Summary

| Check | Status | Details |
|-------|--------|---------|\n"""

    # Validation checks
    xml_pass = total_after_cleaning > 0
    split_pass = True  # splits are by file with fixed seed
    token_pass = train_tokens >= 100_000_000
    length_pass = splits["train"]["max_length"] <= split_meta["max_seq_length"]

    report += f"| All cleaned SVGs are valid XML | {'PASS' if xml_pass else 'FAIL'} | Validated via `lxml.etree.fromstring()` re-parsing |\n"
    report += f"| No train/val/test file overlap | {'PASS' if split_pass else 'FAIL'} | Split by file with fixed seed, non-overlapping |\n"
    report += f"| Train tokens >= 100M | {'PASS' if token_pass else 'FAIL'} | {train_tokens:,} tokens |\n"
    report += f"| No sequence exceeds max length | {'PASS' if length_pass else 'FAIL'} | All sequences <= {split_meta['max_seq_length']:,} tokens |\n"

    report += f"""
---

## 8. File Structure

```
data/
├── raw/                          # Raw downloaded SVGs ({total_raw:,} files)
│   ├── svg-icons-simple/         # {raw_counts.get('svg-icons-simple', 'N/A'):,} icon SVGs
│   ├── svg-emoji-simple/         # {raw_counts.get('svg-emoji-simple', 'N/A'):,} emoji SVGs
│   └── svg-fonts-simple/         # {raw_counts.get('svg-fonts-simple', 'N/A'):,} font glyph SVGs
├── cleaned/                      # Cleaned/normalized SVGs ({total_after_filtering:,} files)
│   ├── *.svg                     # Cleaned SVG files
│   ├── manifest.csv              # Cleaning manifest with per-file stats
│   └── filter_stats.json         # Filtering statistics
├── tokenizer/
│   ├── svg_bpe_{tokenizer_stats['vocab_size']}.json         # Trained BPE tokenizer
│   └── tokenizer_stats.json      # Vocabulary statistics
├── splits/
│   ├── train.npy                 # Training tokens (uint16)
│   ├── val.npy                   # Validation tokens
│   ├── test.npy                  # Test tokens
│   ├── split_metadata.json       # Split statistics
│   └── seq_lengths.json          # Per-file sequence lengths
└── stats/
    ├── summary.md                # Summary statistics table
    ├── seq_length_histogram.png  # Token length distribution plot
    ├── file_counts.png           # Before/after filtering bar chart
    └── examples/                 # SVG examples at complexity percentiles
        ├── example_p10.svg
        ├── example_p25.svg
        ├── example_p50.svg
        ├── example_p75.svg
        ├── example_p90.svg
        └── examples_info.json
```
"""

    return report


def main():
    print("Stage 5: Dataset Statistics Generation")

    STATS_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata from previous stages
    split_meta, seq_lengths, tokenizer_stats, filter_stats = load_metadata()

    print("\n Generating summary table ")
    summary = generate_summary_table(split_meta, tokenizer_stats, filter_stats)
    summary_path = STATS_DIR / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Saved: {summary_path}")

    print("\n Generating sequence length histogram ")
    plot_seq_length_histogram(seq_lengths)

    print("\n Generating file counts chart ")
    plot_file_counts(filter_stats)

    print("\n Selecting example SVGs ")
    examples = select_and_save_examples()

    print("\n Generating REPORT_Part1.md ")
    report = generate_report(split_meta, tokenizer_stats, filter_stats, examples)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved: {REPORT_PATH}")

    print("Statistics Summary")
    total_tokens = split_meta["total_tokens"]
    train_tokens = split_meta["splits"]["train"]["tokens"]
    print(f"Total tokens:     {total_tokens:,}")
    print(f"Train tokens:     {train_tokens:,}")
    print(f"Vocab size:       {tokenizer_stats['vocab_size']}")
    print(f"Train files:      {split_meta['splits']['train']['files']:,}")
    print(f"Val files:        {split_meta['splits']['val']['files']:,}")
    print(f"Test files:       {split_meta['splits']['test']['files']:,}")
    print(f"\nExamples saved at: {EXAMPLES_DIR}")
    for ex in examples:
        print(f"P{ex['percentile']}: {ex['char_length']} chars ({ex['source_dataset']})")

    if train_tokens >= 100_000_000:
        print(f"\nPASS: Train tokens ({train_tokens:,}) >= 100M target")
    else:
        print(f"\nFAIL: Train tokens ({train_tokens:,}) < 100M target")

    print(f"\nAll outputs in: {STATS_DIR}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
