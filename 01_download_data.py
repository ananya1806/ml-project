import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path("data/raw")
FONTS_SUBSAMPLE = 350_000
RANDOM_SEED = 42

DATASETS = [
    {
        "name": "svg-icons-simple",
        "hf_id": "starvector/svg-icons-simple",
        "svg_column": None,  # auto-detect
        "subsample": None,
    },
    {
        "name": "svg-emoji-simple",
        "hf_id": "starvector/svg-emoji-simple",
        "svg_column": None,
        "subsample": None,
    },
    {
        "name": "svg-fonts-simple",
        "hf_id": "starvector/svg-fonts-simple",
        "svg_column": None,
        "subsample": FONTS_SUBSAMPLE,
    },
]


def detect_svg_column(dataset):
    columns = dataset.column_names
    for candidate in ["svg", "Svg", "SVG", "text", "content", "code"]:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not detect SVG column. Available columns: {columns}")


def download_dataset(config):
    name = config["name"]
    hf_id = config["hf_id"]
    subsample = config["subsample"]
    
    out_dir = RAW_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading: {hf_id}")
    
    # Load dataset
    ds = load_dataset(hf_id, split="train")
    print(f"Loaded {len(ds)} rows")
    
    # Detect SVG column
    svg_col = detect_svg_column(ds)
    print(f"SVG column: '{svg_col}'")
    
    # Subsample if needed
    if subsample and len(ds) > subsample:
        print(f"Subsampling {len(ds)} → {subsample} (seed={RANDOM_SEED})")
        random.seed(RANDOM_SEED)
        indices = sorted(random.sample(range(len(ds)), subsample))
        ds = ds.select(indices)
    
    # Save individual SVG files
    saved = 0
    for idx, row in enumerate(tqdm(ds, desc=f"  Saving {name}")):
        svg_text = row[svg_col]
        if svg_text and isinstance(svg_text, str) and len(svg_text.strip()) > 0:
            filepath = out_dir / f"{idx:06d}.svg"
            filepath.write_text(svg_text, encoding="utf-8")
            saved += 1
    
    print(f"Saved {saved} SVG files to {out_dir}")
    return {"name": name, "total_rows": len(ds), "saved": saved}


def main():
    print("Stage 1: Data Download")
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    for config in DATASETS:
        result = download_dataset(config)
        results.append(result)
    
    # Save download manifest
    manifest = {
        "datasets": results,
        "total_files": sum(r["saved"] for r in results),
        "random_seed": RANDOM_SEED,
        "fonts_subsample": FONTS_SUBSAMPLE,
    }
    manifest_path = RAW_DIR / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("Download Summary")
    for r in results:
        print(f"{r['name']}: {r['saved']} files")
    print(f"TOTAL: {manifest['total_files']} files")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
