import json
import random
import numpy as np
from pathlib import Path
import sentencepiece as spm
from tqdm import tqdm

CLEANED_DIR = Path("data/cleaned")
TOKENIZER_DIR = Path("data/tokenizer")
SPLITS_DIR = Path("data/splits")
MAX_SEQ_LENGTH = 2048
TRAIN_RATIO = 0.98
VAL_RATIO = 0.01
TEST_RATIO = 0.01
RANDOM_SEED = 42
MODEL_PREFIX = "svg_bpe_2048"


def main():
    print("Stage 4: Tokenization & Splitting")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Load SentencePiece tokenizer
    model_path = TOKENIZER_DIR / (MODEL_PREFIX + ".model")
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    vocab_size = sp.get_piece_size()
    print(f"Loaded tokenizer: vocab={vocab_size}, <bos>={bos_id}, <eos>={eos_id}")

    # Collect all cleaned SVG files
    svg_files = sorted(CLEANED_DIR.glob("*.svg"))
    print(f"Found {len(svg_files)} cleaned SVG files")

    print("\n Tokenizing ")
    all_sequences = []  # List of (filename, token_ids_list)
    skipped_length = 0

    for svg_path in tqdm(svg_files, desc="  Tokenizing"):
        text = svg_path.read_text(encoding="utf-8").strip()
        encoded = sp.encode(text, out_type=int)
        token_ids = [bos_id] + encoded + [eos_id]

        if len(token_ids) > MAX_SEQ_LENGTH:
            skipped_length += 1
            continue

        all_sequences.append((svg_path.name, token_ids))

    print(f"Tokenized: {len(all_sequences)} sequences")
    print(f"Skipped (>{MAX_SEQ_LENGTH} tokens): {skipped_length}")

    print("\n Splitting (98/1/1 by file) ")
    random.seed(RANDOM_SEED)
    random.shuffle(all_sequences)

    n = len(all_sequences)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_seqs = all_sequences[:n_train]
    val_seqs = all_sequences[n_train:n_train + n_val]
    test_seqs = all_sequences[n_train + n_val:]

    print(f"Train: {len(train_seqs)} files")
    print(f"Val:   {len(val_seqs)} files")
    print(f"Test:  {len(test_seqs)} files")

    def compute_length_stats(seqs, name):
        lengths = [len(s[1]) for s in seqs]
        tokens = sum(lengths)
        stats = {
            "name": name,
            "files": len(seqs),
            "tokens": tokens,
            "mean_length": round(np.mean(lengths), 1) if lengths else 0,
            "median_length": int(np.median(lengths)) if lengths else 0,
            "min_length": int(np.min(lengths)) if lengths else 0,
            "max_length": int(np.max(lengths)) if lengths else 0,
            "std_length": round(np.std(lengths), 1) if lengths else 0,
        }
        return stats, lengths

    train_stats, train_lengths = compute_length_stats(train_seqs, "train")
    val_stats, val_lengths = compute_length_stats(val_seqs, "val")
    test_stats, test_lengths = compute_length_stats(test_seqs, "test")

    total_tokens = train_stats["tokens"] + val_stats["tokens"] + test_stats["tokens"]

    print("\n Serializing to .npy ")

    def save_split(seqs, name):
        all_tokens = []
        for _, token_ids in seqs:
            all_tokens.extend(token_ids)
        arr = np.array(all_tokens, dtype=np.uint16)
        path = SPLITS_DIR / f"{name}.npy"
        np.save(path, arr)
        size_mb = arr.nbytes / (1024 * 1024)
        print(f"{name}.npy: {len(arr):,} tokens ({size_mb:.1f} MB)")
        return len(arr)

    save_split(train_seqs, "train")
    save_split(val_seqs, "val")
    save_split(test_seqs, "test")

    metadata = {
        "max_seq_length": MAX_SEQ_LENGTH,
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "random_seed": RANDOM_SEED,
        "skipped_over_max_length": skipped_length,
        "total_files_after_filtering": len(all_sequences),
        "total_tokens": total_tokens,
        "splits": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
    }
    metadata_path = SPLITS_DIR / "split_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save sequence lengths for histogram generation
    seq_lengths = {
        "train": train_lengths,
        "val": val_lengths,
        "test": test_lengths,
    }
    lengths_path = SPLITS_DIR / "seq_lengths.json"
    with open(lengths_path, "w") as f:
        json.dump(seq_lengths, f)

    print("Tokenization & Splitting Summary")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Train: {train_stats['tokens']:,} tokens ({train_stats['files']} files)")
    print(f"Val:   {val_stats['tokens']:,} tokens ({val_stats['files']} files)")
    print(f"Test:  {test_stats['tokens']:,} tokens ({test_stats['files']} files)")
    print(f"Mean seq length: {train_stats['mean_length']}")
    print(f"Median seq length: {train_stats['median_length']}")
    print(f"Metadata: {metadata_path}")
    print(f"Seq lengths: {lengths_path}")

    if train_stats["tokens"] >= 100_000_000:
        print(f"\n  Training set exceeds 100M tokens target!")
    else:
        print(f"\n  WARNING: Training set ({train_stats['tokens']:,}) is below 100M tokens target!")
        print(f"Consider adding more data or adjusting filters.")


if __name__ == "__main__":
    main()
