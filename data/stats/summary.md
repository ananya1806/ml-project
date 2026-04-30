# Dataset Statistics Summary

## Overall

| Metric | Value |
|--------|-------|
| **Vocabulary Size** | 2,048 |
| **Max Sequence Length** | 2,048 tokens |
| **Raw Files Downloaded** | 434,548 |
| **Files After Cleaning (Pass 1)** | 434,548 |
| **Files After Percentile Filtering (Pass 2)** | 430,204 |
| **Files After Token-Length Filtering** | 430,204 |
| **Total Tokens** | **110,475,857** |

## Split Details

| Split | Files | Tokens | Mean Length | Median Length | Min | Max |
|-------|------:|-------:|------------:|--------------:|----:|----:|
| **Train** | 421,599 | 108,279,355 | 256.8 | 198 | 61 | 1370 |
| **Val** | 4,302 | 1,109,964 | 258.0 | 198 | 70 | 1292 |
| **Test** | 4,303 | 1,086,538 | 252.5 | 196 | 67 | 1293 |
| **Total** | **430,204** | **110,475,857** | | | | |

## Files Before/After Filtering (by Dataset)

| Dataset | Before Filtering | After Filtering | Removed |
|---------|----------------:|-----------------:|--------:|
| svg-emoji-simple | 4,114 | 3,630 | 484 |
| svg-fonts-simple | 350,000 | 349,831 | 169 |
| svg-icons-simple | 80,434 | 76,743 | 3,691 |

## Tokenizer

| Parameter | Value |
|-----------|-------|
| Algorithm | BPE (SentencePiece) |
| Vocab Size | 2,048 |
| Min Frequency | auto (SentencePiece) |
| Training Files | 430,204 |
| Avg Chars/Token | 3.95 |
| Special Tokens | `<pad>` (0), `<bos>` (1), `<eos>` (2) |
