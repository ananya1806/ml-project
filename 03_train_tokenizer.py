import os
import json
from pathlib import Path
import sentencepiece as spm
from tqdm import tqdm

CLEANED_DIR = Path("data/cleaned")
TOKENIZER_DIR = Path("data/tokenizer")
VOCAB_SIZE = 2048
MODEL_PREFIX = "svg_bpe_2048"


def main():
    print("Stage 3: BPE Tokenizer Training (SentencePiece)")

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # Collect cleaned SVG files
    svg_files = sorted(CLEANED_DIR.glob("*.svg"))
    num_files = len(svg_files)
    print(f"Training corpus: {num_files} cleaned SVG files")

    # Write training corpus (one SVG per line)
    corpus_path = TOKENIZER_DIR / "train_corpus.txt"
    print(f"Writing training corpus to {corpus_path}...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for svg_path in tqdm(svg_files, desc="  Preparing corpus"):
            text = svg_path.read_text(encoding="utf-8").strip()
            text = text.replace("\n", " ").replace("\r", "")
            f.write(text + "\n")

    corpus_size_mb = corpus_path.stat().st_size / (1024 * 1024)
    print(f"Corpus size: {corpus_size_mb:.1f} MB")

    # Train SentencePiece BPE model
    model_prefix = str(TOKENIZER_DIR / MODEL_PREFIX)
    print(f"\n  Training SentencePiece BPE (vocab_size={VOCAB_SIZE})...")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=False,
        max_sentence_length=16384,
        input_sentence_size=0,  # use all sentences
    )

    print(f"Training complete!")

    # Load and verify
    sp = spm.SentencePieceProcessor()
    sp.load(model_prefix + ".model")

    actual_vocab_size = sp.get_piece_size()
    print(f"Vocabulary size: {actual_vocab_size}")
    print(f"<pad>={sp.pad_id()}, <bos>={sp.bos_id()}, <eos>={sp.eos_id()}, <unk>={sp.unk_id()}")

    # Test encode/decode on samples
    print(f"\n  Sample encodings:")
    sample_files = svg_files[:5]
    sample_stats = []
    for sf in sample_files:
        text = sf.read_text(encoding="utf-8").strip()
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        chars_per_token = round(len(text) / len(ids), 2) if ids else 0
        roundtrip = text == decoded
        sample_stats.append({
            "file": sf.name,
            "chars": len(text),
            "tokens": len(ids),
            "chars_per_token": chars_per_token,
            "roundtrip_match": roundtrip,
        })
        print(f"{sf.name}: {len(text)} chars -> {len(ids)} tokens ({chars_per_token} chars/tok)")

    # Save statistics
    stats = {
        "vocab_size": actual_vocab_size,
        "special_tokens": {
            "<pad>": sp.pad_id(),
            "<bos>": sp.bos_id(),
            "<eos>": sp.eos_id(),
            "<unk>": sp.unk_id(),
        },
        "min_frequency": "auto (SentencePiece)",
        "training_files": num_files,
        "sample_encoding_stats": sample_stats,
        "avg_chars_per_token": round(
            sum(s["chars_per_token"] for s in sample_stats) / len(sample_stats), 2
        ) if sample_stats else 0,
    }
    stats_path = TOKENIZER_DIR / "tokenizer_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Cleanup corpus file (large)
    corpus_path.unlink()
    print(f"\n  Cleaned up training corpus file")

    print("Tokenizer Summary")
    print(f"Vocabulary size:    {actual_vocab_size}")
    print(f"Model:              {model_prefix}.model")
    print(f"Training files:     {num_files}")
    print(f"Avg chars/token:    {stats['avg_chars_per_token']}")
    print(f"Stats saved to:     {stats_path}")

    if actual_vocab_size >= VOCAB_SIZE:
        print(f"\n  Vocab size target reached: {actual_vocab_size}")
    else:
        print(f"\n  WARNING: Actual vocab ({actual_vocab_size}) < target ({VOCAB_SIZE})")


if __name__ == "__main__":
    main()
