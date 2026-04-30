import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


LR_SWEEP_DIR = os.environ.get("LR_SWEEP_DIR", "checkpoints/lr_sweep")
SCALING_DIR = os.environ.get("RESULTS_DIR", "checkpoints/scaling")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "checkpoints")


def power_law(N, a, alpha, c):

    return a * np.power(N, -alpha) + c


def plot_lr_sweep(lr_dir, output_dir):

    with open(os.path.join(lr_dir, "lr_sweep_summary.json")) as f:
        summary = json.load(f)

    lrs = [r['learning_rate'] for r in summary['results']]
    losses = [r['final_val_loss'] for r in summary['results']]
    best_lr = summary['best_lr']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(lrs, losses, 'o-', color='#4A90D9', markersize=8, linewidth=2)

    # Highlight best
    best_idx = lrs.index(best_lr)
    ax.scatter([best_lr], [losses[best_idx]], color='#E74C3C', s=150, zorder=5,
               label=f'Best: {best_lr:.0e} (loss={losses[best_idx]:.4f})')

    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Final Validation Loss', fontsize=12)
    ax.set_title('Learning Rate Sweep (Tiny Model)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "lr_sweep_plot.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    return summary


def plot_scaling(scaling_dir, output_dir):

    with open(os.path.join(scaling_dir, "scaling_summary.json")) as f:
        summary = json.load(f)

    models = summary['models']
    params = np.array([m['n_params'] for m in models])
    losses = np.array([m['final_val_loss'] for m in models])
    names = [m['model_name'] for m in models]

    # Fit power law: L = a * N^(-alpha) + c
    try:
        popt, pcov = curve_fit(power_law, params, losses,
                               p0=[10.0, 0.1, 2.0], maxfev=10000)
        a, alpha, c = popt

        # R² calculation
        y_pred = power_law(params, *popt)
        ss_res = np.sum((losses - y_pred) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        fit_ok = True
        print(f"Power law fit: L = {a:.4f} * N^(-{alpha:.4f}) + {c:.4f}")
        print(f"R² = {r_squared:.4f}")
    except Exception as e:
        print(f"Power law fit failed: {e}")
        fit_ok = False
        a, alpha, c, r_squared = 0, 0, 0, 0

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(params, losses, s=120, c='#E74C3C', zorder=5, edgecolors='white', linewidth=1.5)

    for i, name in enumerate(names):
        ax.annotate(name.capitalize(), (params[i], losses[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=10)

    if fit_ok:
        x_fit = np.logspace(np.log10(params.min() * 0.5), np.log10(params.max() * 2), 100)
        y_fit = power_law(x_fit, a, alpha, c)
        ax.plot(x_fit, y_fit, '--', color='#4A90D9', linewidth=2,
                label=f'L = {a:.2f}·N$^{{-{alpha:.4f}}}$ + {c:.2f}\n(R²={r_squared:.4f})')
        ax.legend(fontsize=11, loc='upper right')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Validation Loss (1 epoch)', fontsize=12)
    ax.set_title('Scaling Law: Validation Loss vs Model Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    path = os.path.join(output_dir, "scaling_plot.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    return summary, {'a': a, 'alpha': alpha, 'c': c, 'r_squared': r_squared, 'fit_ok': fit_ok}


def plot_training_curves(scaling_dir, output_dir):

    with open(os.path.join(scaling_dir, "scaling_summary.json")) as f:
        summary = json.load(f)

    colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C', '#9B59B6']
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_info in enumerate(summary['models']):
        name = model_info['model_name']
        log_path = os.path.join(scaling_dir, f"{name}_log.json")
        with open(log_path) as f:
            log = json.load(f)

        steps = [s['step'] for s in log['steps']]
        train_losses = [s['train_loss'] for s in log['steps']]

        ax.plot(steps, train_losses, color=colors[i % len(colors)], alpha=0.8,
                linewidth=1.5, label=f"{name.capitalize()} ({model_info['n_params']:,} params)")

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def generate_report(lr_summary, scaling_summary, fit_params, output_dir):

    models = scaling_summary['models']
    best_lr = lr_summary['best_lr']

    # LR sweep table
    lr_rows = ""
    for r in lr_summary['results']:
        marker = " **← selected**" if r['learning_rate'] == best_lr else ""
        lr_rows += f"| {r['learning_rate']:.1e} | {r['final_val_loss']:.4f} | {r['best_val_loss']:.4f} | {r['wall_clock_seconds']:.1f}s |{marker}\n"

    # Model table
    model_rows = ""
    for m in models:
        log_path = os.path.join(SCALING_DIR, f"{m['model_name']}_log.json")
        with open(log_path) as f:
            log = json.load(f)
        model_rows += (
            f"| {m['model_name'].capitalize()} | {m['n_params']:,} | "
            f"{log.get('block_size', 1024)} | "
            f"{log.get('gradient_accumulation_steps', 4)*log.get('micro_batch_size', 16)*log.get('block_size', 1024):,} | "
            f"{m['final_val_loss']:.4f} | {m['wall_clock_seconds']:.1f}s | "
            f"{m['peak_gpu_memory_gb']:.1f} GB | {m['avg_tokens_per_sec']:,} |\n"
        )

    # Architecture table
    arch_rows = ""
    configs = {
        'tiny': (128, 4, 4, 512),
        'small': (192, 6, 6, 768),
        'medium': (384, 6, 6, 1536),
        'large': (512, 10, 8, 2048),
        'xl': (768, 12, 12, 3072),
    }
    for m in models:
        name = m['model_name']
        d, nl, nh, dff = configs.get(name, (0, 0, 0, 0))
        arch_rows += f"| {name.capitalize()} | {m['n_params']:,} | {d} | {nl} | {nh} | {dff} | {d//nh if nh else 0} |\n"

    # Fit info
    if fit_params['fit_ok']:
        fit_text = f"""The fitted scaling law is:

**L = {fit_params['a']:.4f} · N^(-{fit_params['alpha']:.4f}) + {fit_params['c']:.4f}**

| Parameter | Value |
|-----------|-------|
| a (scale) | {fit_params['a']:.4f} |
| α (exponent) | {fit_params['alpha']:.4f} |
| c (irreducible loss) | {fit_params['c']:.4f} |
| R² | {fit_params['r_squared']:.4f} |"""
    else:
        fit_text = "Power law fitting did not converge. See scaling plot for visual analysis."

    # Determine if scaling worked or if LR caused issues
    smallest_loss = models[0]['final_val_loss']
    largest_loss = models[-1]['final_val_loss']
    best_model = min(models, key=lambda m: m['final_val_loss'])
    worst_model = max(models, key=lambda m: m['final_val_loss'])
    scaling_works = largest_loss < smallest_loss

    if scaling_works:
        analysis_text = f"""### Loss vs Model Size

As model size increases from {models[0]['n_params']:,} to {models[-1]['n_params']:,} parameters, validation loss decreases from {smallest_loss:.4f} to {largest_loss:.4f}. This demonstrates a clear scaling relationship where additional parameters improve the model's ability to capture SVG code patterns.

### Efficiency

Larger models require proportionally more compute time and GPU memory, but the cost-per-improvement ratio becomes less favorable at larger scales — consistent with diminishing returns predicted by scaling laws."""
    else:
        analysis_text = f"""### Loss vs Model Size

The smallest models (Tiny and Small) achieved the lowest validation losses ({models[0]['final_val_loss']:.4f} and {models[1]['final_val_loss']:.4f} respectively), while the larger models (Medium, Large, XL) showed significantly higher validation losses (~{np.mean([m['final_val_loss'] for m in models[2:]]):.2f}). The best-performing model was **{best_model['model_name'].capitalize()}** ({best_model['n_params']:,} params) with val_loss={best_model['final_val_loss']:.4f}.

This outcome is attributable to the **shared learning rate** (LR={best_lr:.1e}), which was optimized for the Tiny model as required by the experimental protocol. The LR sweep on Tiny showed monotonically decreasing val loss across the tested range, selecting {best_lr:.1e} (the highest tested value). While this aggressive LR enables the small models to train effectively in a single epoch, larger models with more parameters are more sensitive to high learning rates and fail to converge properly.

### Key Observation

This result highlights a well-known challenge in scaling studies: a single learning rate does not transfer well across model scales. In practice, larger models typically require lower learning rates (approximately scaling as LR ∝ N^(-0.5) or similar). The experimental protocol (same LR for all models) intentionally exposes this effect.

### Efficiency

Despite the convergence issues, the compute scaling follows expected patterns: wall-clock time increases roughly linearly with parameter count ({models[0]['wall_clock_seconds']:.0f}s for Tiny vs {models[-1]['wall_clock_seconds']:.0f}s for XL), and GPU memory scales proportionally ({models[0]['peak_gpu_memory_gb']:.1f} GB to {models[-1]['peak_gpu_memory_gb']:.1f} GB)."""

    analysis_text += f"""

### Limitations

- All models trained with the **same learning rate** ({best_lr:.1e}, from Tiny sweep) as required by the experimental protocol. Larger models would likely benefit from lower learning rates.
- Only **1 epoch** of training. More epochs could allow larger models to eventually converge even with a suboptimal LR.
- The LR sweep on Tiny did not find a U-shaped curve (loss kept decreasing at higher LRs), suggesting the optimal LR for Tiny may be even higher than the range tested."""

    report = f"""# Part 2: Transformer Scaling Study — Report

## 1. Overview

This report documents the training of 5 decoder-only transformer language models of varying sizes on our SVG corpus (108M training tokens). We perform a learning rate sweep on the smallest model, train all models for 1 epoch using the selected learning rate, and analyze the scaling behavior.

---

## 2. Model Architecture

All models use the same decoder-only GPT architecture adapted from [nanoGPT](https://github.com/karpathy/nanoGPT) by Karpathy.

### Architecture Details

| Component | Implementation | Source |
|-----------|---------------|--------|
| Attention | Multi-head causal self-attention with Flash Attention | nanoGPT (unchanged) |
| FFN | Linear → GELU → Linear with explicit d_ff | nanoGPT (**modified**: added d_ff param) |
| Normalization | Pre-LayerNorm (bias=False) | nanoGPT (unchanged) |
| Embeddings | Token + learned positional, weight tying with LM head | nanoGPT (**modified**: custom vocab/block) |
| Optimizer | AdamW with decay/no-decay parameter groups | nanoGPT (unchanged) |

### Model Configurations

| Name | Parameters | d_model | n_layers | n_heads | d_ff | head_dim |
|------|-----------|---------|----------|---------|------|----------|
{arch_rows}
---

## 3. Training Setup

| Parameter | Value |
|-----------|-------|
| Tokenizer | SentencePiece BPE (vocab=2,048) |
| Block size | 1,024 tokens |
| Batch size | {scaling_summary.get('tokens_per_step', 65536):,} tokens |
| Optimizer | AdamW (β1=0.9, β2=0.95, weight_decay=0.1) |
| LR schedule | Cosine with 10% linear warmup, min_lr = 0.1 × max_lr |
| Learning rate | {best_lr:.1e} (from sweep) |
| Precision | bfloat16 (mixed precision) |
| Dropout | 0.0 |
| Gradient clipping | max_norm = 1.0 |
| Epochs | 1 |
| Steps per epoch | {scaling_summary.get('total_steps', 'N/A'):,} |
| Training tokens | 108,279,355 |

---

## 4. Learning Rate Sweep

Performed on the **Tiny** model (~{models[0]['n_params']:,} params) with {len(lr_summary['results'])} learning rates on a log scale.

| Learning Rate | Final Val Loss | Best Val Loss | Time |
|--------------|---------------|--------------|------|
{lr_rows}
**Selected LR: {best_lr:.1e}** — used for all subsequent model training.

The LR sweep plot is saved at `checkpoints/lr_sweep_plot.png`.

---

## 5. Scaling Results

### 5.1 Model Summary

| Model | Params | Block Size | Batch (tokens) | Val Loss | Time | Peak GPU | Tokens/s |
|-------|--------|-----------|----------------|----------|------|----------|----------|
{model_rows}
### 5.2 Scaling Law Fit

{fit_text}

The scaling plot is saved at `checkpoints/scaling_plot.png`.

### 5.3 Training Curves

Training loss curves for all 5 models are overlaid in `checkpoints/training_curves.png`.

---

## 6. Analysis

{analysis_text}

---

## 7. Reproducibility

| Item | Value |
|------|-------|
| Random seed (data splits) | 42 |
| PyTorch version | See Colab environment |
| GPU | See training logs |
| Total training time | {sum(m['wall_clock_seconds'] for m in models):.0f}s ({sum(m['wall_clock_seconds'] for m in models)/60:.1f} min) |
| Code reference | nanoGPT by Karpathy (model architecture) |
"""

    report_path = os.path.join(output_dir, "REPORT_Part2.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved: {report_path}")


def main():
    print("Stage 8: Scaling Analysis & Report")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLR Sweep Plot")
    lr_summary = plot_lr_sweep(LR_SWEEP_DIR, OUTPUT_DIR)

    print("\nScaling Plot")
    scaling_summary, fit_params = plot_scaling(SCALING_DIR, OUTPUT_DIR)

    print("\nTraining Curves")
    plot_training_curves(SCALING_DIR, OUTPUT_DIR)

    print("\nGenerating Report")
    generate_report(lr_summary, scaling_summary, fit_params, OUTPUT_DIR)

    print("\nAnalysis Summary")
    print(f"Plots: {OUTPUT_DIR}/")
    print(f"Report: {OUTPUT_DIR}/REPORT_Part2.md")


if __name__ == "__main__":
    main()
