import re
import csv
import json
from pathlib import Path
from lxml import etree
from tqdm import tqdm
from collections import defaultdict

RAW_DIR = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")
MIN_CHAR_LENGTH = 50
PERCENTILE_CUTOFF = 99  # Remove top 1% by character length

# Namespaces to strip (editor metadata)
STRIP_NAMESPACES = {
    "http://www.inkscape.org/namespaces/inkscape",
    "http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd",
    "http://ns.adobe.com/AdobeIllustrator/10.0/",
    "http://ns.adobe.com/AdobeSVGViewerExtensions/3.0/",
    "http://www.bohemiancoding.com/sketch/ns",
    "http://creativecommons.org/ns#",
    "http://purl.org/dc/elements/1.1/",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
}

# Elements to remove entirely
STRIP_ELEMENTS = {"metadata", "desc", "title", "defs"}


def strip_namespace(tag):
    if tag and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def clean_svg(svg_text):
    try:
        # Parse with comment/PI removal
        parser = etree.XMLParser(
            remove_comments=True,
            remove_pis=True,
            remove_blank_text=True,
        )
        root = etree.fromstring(svg_text.encode("utf-8"), parser)
    except etree.XMLSyntaxError as e:
        return None, False, f"XML parse error: {e}"

    # Recursive cleaning
    _clean_element(root)

    # Serialize back to string
    cleaned = etree.tostring(root, encoding="unicode")

    # Coordinate normalization: round floats with >=2 decimal places to 1 decimal
    cleaned = re.sub(
        r"(\d+\.\d{2,})",
        lambda m: f"{float(m.group(1)):.1f}",
        cleaned,
    )

    # Whitespace normalization
    cleaned = re.sub(r"\s+", " ", cleaned)  # collapse whitespace
    cleaned = cleaned.replace("> <", "><")   # remove space between tags
    cleaned = cleaned.strip()

    # Re-validate
    try:
        etree.fromstring(cleaned.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        return None, False, f"Re-validation failed: {e}"

    return cleaned, True, None


def _clean_element(element):
    # Strip namespace from tag
    element.tag = strip_namespace(element.tag)

    # Remove elements we don't want (metadata, desc, title)
    children_to_remove = []
    for child in element:
        child_tag = strip_namespace(child.tag)
        if child_tag in STRIP_ELEMENTS:
            children_to_remove.append(child)
        else:
            _clean_element(child)

    for child in children_to_remove:
        element.remove(child)

    # Clean attributes
    new_attribs = {}
    for attr_name, attr_value in element.attrib.items():
        # Strip namespace from attribute name
        clean_name = strip_namespace(attr_name)

        # Skip editor-specific namespace attributes
        if attr_name.startswith("{"):
            ns = attr_name.split("}")[0][1:]
            if ns in STRIP_NAMESPACES:
                continue

        new_attribs[clean_name] = attr_value

    # Replace attributes with cleaned + sorted version
    element.attrib.clear()
    for key in sorted(new_attribs.keys()):
        element.set(key, new_attribs[key])


def main():
    print("Stage 2: SVG Cleaning & Normalization")

    # Clear old cleaned directory to prevent orphaned files from previous runs
    if CLEANED_DIR.exists():
        shutil.rmtree(CLEANED_DIR)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all raw SVG files
    raw_files = []
    for dataset_dir in sorted(RAW_DIR.iterdir()):
        if dataset_dir.is_dir():
            svgs = sorted(dataset_dir.glob("*.svg"))
            for svg_path in svgs:
                raw_files.append((dataset_dir.name, svg_path))

    print(f"Found {len(raw_files)} raw SVG files")

    print("\n Pass 1: Cleaning & Normalization ")
    cleaned_data = []  # (dataset_name, filename, cleaned_text, original_len, cleaned_len)
    stats = defaultdict(lambda: {"total": 0, "cleaned": 0, "failed": 0})

    for dataset_name, svg_path in tqdm(raw_files, desc="  Cleaning"):
        stats[dataset_name]["total"] += 1
        svg_text = svg_path.read_text(encoding="utf-8", errors="replace")
        original_len = len(svg_text)

        cleaned_text, success, error = clean_svg(svg_text)

        if success:
            stats[dataset_name]["cleaned"] += 1
            cleaned_data.append((
                dataset_name,
                svg_path.name,
                cleaned_text,
                original_len,
                len(cleaned_text),
            ))
        else:
            stats[dataset_name]["failed"] += 1

    print(f"\n  Pass 1 results:")
    for ds_name, s in stats.items():
        print(f"{ds_name}: {s['cleaned']}/{s['total']} cleaned, {s['failed']} failed")

    print("\n Pass 2: Filtering ")

    # Filter by minimum length
    before_filter = len(cleaned_data)
    cleaned_data = [d for d in cleaned_data if len(d[2]) >= MIN_CHAR_LENGTH]
    removed_short = before_filter - len(cleaned_data)
    print(f"Removed {removed_short} SVGs shorter than {MIN_CHAR_LENGTH} chars")

    # Compute percentile threshold
    char_lengths = [len(d[2]) for d in cleaned_data]
    char_lengths_sorted = sorted(char_lengths)
    p99_idx = int(len(char_lengths_sorted) * PERCENTILE_CUTOFF / 100)
    p99_threshold = char_lengths_sorted[min(p99_idx, len(char_lengths_sorted) - 1)]
    print(f"99th percentile threshold: {p99_threshold} chars")

    before_p99 = len(cleaned_data)
    cleaned_data = [d for d in cleaned_data if len(d[2]) <= p99_threshold]
    removed_long = before_p99 - len(cleaned_data)
    print(f"Removed {removed_long} SVGs above 99th percentile")

    print(f"\n Saving {len(cleaned_data)} cleaned SVGs ")

    # Save all cleaned SVGs to a flat directory with dataset prefix
    manifest_rows = []
    dataset_counts = defaultdict(int)

    for dataset_name, filename, cleaned_text, orig_len, clean_len in tqdm(
        cleaned_data, desc="  Saving"
    ):
        out_filename = f"{dataset_name}__{filename}"
        out_path = CLEANED_DIR / out_filename
        out_path.write_text(cleaned_text, encoding="utf-8")
        dataset_counts[dataset_name] += 1

        manifest_rows.append({
            "dataset": dataset_name,
            "original_file": filename,
            "cleaned_file": out_filename,
            "original_chars": orig_len,
            "cleaned_chars": clean_len,
            "reduction_pct": round((1 - clean_len / orig_len) * 100, 1) if orig_len > 0 else 0,
        })

    # Save manifest CSV
    manifest_path = CLEANED_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Save filtering stats JSON (used by later stages)
    filter_stats = {
        "raw_counts": {ds: s["total"] for ds, s in stats.items()},
        "after_cleaning": {ds: s["cleaned"] for ds, s in stats.items()},
        "cleaning_failures": {ds: s["failed"] for ds, s in stats.items()},
        "after_filtering": dict(dataset_counts),
        "removed_short": removed_short,
        "removed_long_p99": removed_long,
        "p99_threshold_chars": p99_threshold,
        "min_char_length": MIN_CHAR_LENGTH,
        "total_before_filter": before_filter,
        "total_after_filter": len(cleaned_data),
    }
    filter_stats_path = CLEANED_DIR / "filter_stats.json"
    with open(filter_stats_path, "w") as f:
        json.dump(filter_stats, f, indent=2)

    print("Cleaning Summary")
    print(f"  Raw files:           {sum(s['total'] for s in stats.values())}")
    print(f"  After cleaning:      {before_filter}")
    print(f"  Removed (too short): {removed_short}")
    print(f"  Removed (top 1%):    {removed_long}")
    print(f"  Final cleaned set:   {len(cleaned_data)}")
    print(f"  Manifest:            {manifest_path}")
    print(f"  Filter stats:        {filter_stats_path}")
    for ds, count in sorted(dataset_counts.items()):
        print(f"    {ds}: {count}")


if __name__ == "__main__":
    main()
