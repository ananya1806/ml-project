import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


SP_LR_DIR = os.environ.get("SP_LR_DIR", "checkpoints/lr_sweep")
SP_SCALING_DIR = os.environ.get("SP_SCALING_DIR", "checkpoints/scaling")
MUP_LR_DIR = os.environ.get("MUP_LR_DIR", "checkpoints/mup_lr_sweep")
MUP_SCALING_DIR = os.environ.get("MUP_SCALING_DIR", "checkpoints/mup_scaling")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "checkpoints")


def power_law(N, a, alpha, c):

    return a * np.power(N, -alpha) + c


def fit_power_law(params, losses, label=""):

    try:
        popt, pcov = curve_fit(power_law, params, losses,
                               p0=[10.0, 0.1, 2.0], maxfev=10000)
        a, alpha, c = popt
        y_pred = power_law(params, *popt)
        ss_res = np.sum((losses - y_pred) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        perr = np.sqrt(np.diag(pcov))
        print(f"{label} fit: L = {a:.4f} * N^(-{alpha:.4f}) + {c:.4f}, R²={r_squared:.4f}")
        return {
            'fit_ok': True, 'a': a, 'alpha': alpha, 'c': c,
            'r_squared': r_squared, 'popt': popt, 'pcov': pcov,
            'a_err': perr[0], 'alpha_err': perr[1], 'c_err': perr[2],
        }
    except Exception as e:
        print(f"{label} fit failed: {e}")
        return {'fit_ok': False, 'a': 0, 'alpha': 0, 'c': 0, 'r_squared': 0}


def extrapolate(fit_result, largest_params, factor=10, n_samples=10000):

    N_target = largest_params * factor

    if not fit_result['fit_ok']:
        return {'N_target': N_target, 'predicted_loss': None, 'ci_low': None, 'ci_high': None}

    # Point prediction
    predicted = power_law(N_target, *fit_result['popt'])

    # Monte Carlo from parameter covariance
    try:
        samples = np.random.multivariate_normal(
            fit_result['popt'], fit_result['pcov'], size=n_samples
        )
        # Filter valid samples (a > 0, alpha > 0, c > 0)
        valid = samples[(samples[:, 0] > 0) & (samples[:, 1] > 0) & (samples[:, 2] > 0)]
        if len(valid) > 100:
            predictions = np.array([power_law(N_target, *s) for s in valid])
            ci_low, ci_high = np.percentile(predictions, [2.5, 97.5])
        else:
            ci_low, ci_high = None, None
    except Exception:
        ci_low, ci_high = None, None

    return {
        'N_target': int(N_target),
        'predicted_loss': round(predicted, 4),
        'ci_low': round(ci_low, 4) if ci_low is not None else None,
        'ci_high': round(ci_high, 4) if ci_high is not None else None,
    }


def plot_lr_comparison(sp_lr_dir, mup_lr_dir, output_dir):

    with open(os.path.join(sp_lr_dir, "lr_sweep_summary.json")) as f:
        sp_data = json.load(f)
    with open(os.path.join(mup_lr_dir, "mup_lr_sweep_summary.json")) as f:
        mup_data = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 6))

    sp_lrs = [r['learning_rate'] for r in sp_data['results']]
    sp_losses = [r['final_val_loss'] for r in sp_data['results']]
    mup_lrs = [r['learning_rate'] for r in mup_data['results']]
    mup_losses = [r['final_val_loss'] for r in mup_data['results']]

    ax.semilogx(sp_lrs, sp_losses, 'o-', color='#E74C3C', markersize=8, linewidth=2,
                label=f"SP (best={sp_data['best_lr']:.0e})")
    ax.semilogx(mup_lrs, mup_losses, 's-', color='#3498DB', markersize=8, linewidth=2,
                label=f"μP (best={mup_data['best_lr']:.0e})")

    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Final Validation Loss', fontsize=12)
    ax.set_title('LR Sweep: Standard Param. vs μP (Tiny Model)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "lr_sweep_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    return sp_data, mup_data


def plot_scaling_comparison(sp_dir, mup_dir, output_dir):

    with open(os.path.join(sp_dir, "scaling_summary.json")) as f:
        sp_data = json.load(f)
    with open(os.path.join(mup_dir, "mup_scaling_summary.json")) as f:
        mup_data = json.load(f)

    sp_params = np.array([m['n_params'] for m in sp_data['models']])
    sp_losses = np.array([m['final_val_loss'] for m in sp_data['models']])
    mup_params = np.array([m['n_params'] for m in mup_data['models']])
    mup_losses = np.array([m['final_val_loss'] for m in mup_data['models']])

    # Fit power laws
    print("\nPower Law Fits")
    sp_fit = fit_power_law(sp_params, sp_losses, "SP")
    mup_fit = fit_power_law(mup_params, mup_losses, "μP")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # SP points and fit
    ax.scatter(sp_params, sp_losses, s=120, c='#E74C3C', zorder=5,
               edgecolors='white', linewidth=1.5, label='SP (Standard Param.)')
    if sp_fit['fit_ok']:
        x_fit = np.logspace(np.log10(sp_params.min() * 0.5), np.log10(sp_params.max() * 2), 100)
        y_fit = power_law(x_fit, *sp_fit['popt'])
        ax.plot(x_fit, y_fit, '--', color='#E74C3C', linewidth=1.5, alpha=0.7,
                label=f"SP fit: α={sp_fit['alpha']:.3f}, R²={sp_fit['r_squared']:.3f}")

    # μP points and fit
    ax.scatter(mup_params, mup_losses, s=120, c='#3498DB', zorder=5,
               marker='s', edgecolors='white', linewidth=1.5, label='μP (Maximal Update)')
    if mup_fit['fit_ok']:
        x_fit = np.logspace(np.log10(mup_params.min() * 0.5), np.log10(mup_params.max() * 2), 100)
        y_fit = power_law(x_fit, *mup_fit['popt'])
        ax.plot(x_fit, y_fit, '--', color='#3498DB', linewidth=1.5, alpha=0.7,
                label=f"μP fit: α={mup_fit['alpha']:.3f}, R²={mup_fit['r_squared']:.3f}")

    # Annotate model names
    for i, m in enumerate(sp_data['models']):
        ax.annotate(m['model_name'].capitalize(), (sp_params[i], sp_losses[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=9, color='#E74C3C')
    for i, m in enumerate(mup_data['models']):
        ax.annotate(m['model_name'].capitalize(), (mup_params[i], mup_losses[i]),
                    textcoords="offset points", xytext=(10, -12), fontsize=9, color='#3498DB')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Validation Loss (1 epoch)', fontsize=12)
    ax.set_title('Scaling Law: SP vs μP', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    path = os.path.join(output_dir, "scaling_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    return sp_data, mup_data, sp_fit, mup_fit


def plot_mup_training_curves(mup_dir, output_dir):

    with open(os.path.join(mup_dir, "mup_scaling_summary.json")) as f:
        summary = json.load(f)

    colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C', '#9B59B6']
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_info in enumerate(summary['models']):
        name = model_info['model_name']
        log_path = os.path.join(mup_dir, f"{name}_log.json")
        with open(log_path) as f:
            log = json.load(f)

        steps = [s['step'] for s in log['steps']]
        train_losses = [s['train_loss'] for s in log['steps']]
        ax.plot(steps, train_losses, color=colors[i % len(colors)], alpha=0.8,
                linewidth=1.5, label=f"{name.capitalize()} ({model_info['n_params']:,} params)")

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('μP Training Loss Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "mup_training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def generate_report(sp_lr, mup_lr, sp_data, mup_data, sp_fit, mup_fit,
                    extrap, output_dir):

    sp_best_lr = sp_lr['best_lr']
    mup_best_lr = mup_lr['best_lr']
    sp_models = sp_data['models']
    mup_models = mup_data['models']

    # LR sweep tables
    sp_lr_rows = ""
    for r in sp_lr['results']:
        marker = " **← selected**" if r['learning_rate'] == sp_best_lr else ""
        sp_lr_rows += f"| {r['learning_rate']:.1e} | {r['final_val_loss']:.4f} |{marker}\n"

    mup_lr_rows = ""
    for r in mup_lr['results']:
        marker = " **← selected**" if r['learning_rate'] == mup_best_lr else ""
        mup_lr_rows += f"| {r['learning_rate']:.1e} | {r['final_val_loss']:.4f} |{marker}\n"

    # Model comparison table
    comp_rows = ""
    for sp_m, mup_m in zip(sp_models, mup_models):
        name = sp_m['model_name'].capitalize()
        delta = mup_m['final_val_loss'] - sp_m['final_val_loss']
        comp_rows += (f"| {name} | {sp_m['n_params']:,} | "
                      f"{sp_m['final_val_loss']:.4f} | {mup_m['final_val_loss']:.4f} | "
                      f"{delta:+.4f} |\n")

    # Fit comparison
    sp_fit_text = (f"L = {sp_fit['a']:.4f} · N^(-{sp_fit['alpha']:.4f}) + {sp_fit['c']:.4f} "
                   f"(R²={sp_fit['r_squared']:.4f})") if sp_fit['fit_ok'] else "Did not converge"
    mup_fit_text = (f"L = {mup_fit['a']:.4f} · N^(-{mup_fit['alpha']:.4f}) + {mup_fit['c']:.4f} "
                    f"(R²={mup_fit['r_squared']:.4f})") if mup_fit['fit_ok'] else "Did not converge"

    # Determine best fit for extrapolation
    best_label = "μP" if (mup_fit.get('r_squared', 0) > sp_fit.get('r_squared', 0)) else "SP"

    # Extrapolation text
    if extrap['predicted_loss'] is not None:
        if extrap['ci_low'] is not None:
            extrap_text = (f"Using the **{best_label}** scaling law (higher R²), we predict:\n\n"
                          f"**Predicted loss at {extrap['N_target']:,} params: "
                          f"{extrap['predicted_loss']:.4f}**\n\n"
                          f"95% confidence interval: [{extrap['ci_low']:.4f}, {extrap['ci_high']:.4f}]\n\n"
                          f"(Computed via Monte Carlo sampling from the parameter covariance matrix, "
                          f"10,000 samples)")
        else:
            extrap_text = (f"Using the **{best_label}** scaling law, we predict:\n\n"
                          f"**Predicted loss at {extrap['N_target']:,} params: "
                          f"{extrap['predicted_loss']:.4f}**\n\n"
                          f"Confidence interval could not be computed (insufficient valid parameter samples).")
    else:
        extrap_text = "Extrapolation not possible — no valid power law fit."

    report = f"""# Part 3: μP Scaling and Extrapolation — Report

## 1. Overview

This report compares Standard Parameterization (SP, Part 2) with Maximal Update Parameterization (μP) for scaling decoder-only transformers on SVG code. We investigate whether μP enables principled learning rate transfer across model widths and use the scaling laws to extrapolate beyond our trained model sizes.

**Reference**: Yang et al. (2022), "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"

---

## 2. μP Implementation

Three changes were made to the SP model (model.py → model_mup.py):

| Change | SP (Part 2) | μP (Part 3) | Why |
|--------|-------------|-------------|-----|
| Output head | `nn.Linear` (weight-tied) | `MuSharedReadout` (weight-tied, μP-scaled) | μP scales output logits by 1/width_mult |
| Attention scaling | 1/√d_head | 1/d_head | Prevents attention logits from growing with width |
| Optimizer | `AdamW` | `MuAdamW` | Scales LR by 1/width_mult for hidden weights |
| LR schedule | Absolute LR setting | Relative multiplier | Preserves MuAdamW's per-group LR ratios |

All other hyperparameters (batch size, weight decay, warmup, gradient clipping, etc.) are identical to Part 2.

---

## 3. Learning Rate Sweeps

### SP Sweep (Part 2)

| Learning Rate | Val Loss |
|--------------|----------|
{sp_lr_rows}
**SP selected: {sp_best_lr:.1e}**

### μP Sweep (Part 3)

| Learning Rate | Val Loss |
|--------------|----------|
{mup_lr_rows}
**μP selected: {mup_best_lr:.1e}**

The LR sweep comparison plot is saved at `checkpoints/lr_sweep_comparison.png`.

---

## 4. Scaling Results Comparison

### Model-by-Model

| Model | Params | SP Val Loss | μP Val Loss | Δ (μP − SP) |
|-------|--------|-------------|-------------|-------------|
{comp_rows}
### Power Law Fits

| Parameterization | Scaling Law | R² |
|-----------------|-------------|-----|
| SP | {sp_fit_text} |
| μP | {mup_fit_text} |

The comparison scaling plot is saved at `checkpoints/scaling_comparison.png`.

---

## 5. Scaling Law Extrapolation

Our largest model (XL) has {sp_models[-1]['n_params']:,} parameters. We extrapolate to **10× this size** ({extrap['N_target']:,} parameters).

{extrap_text}

### Confidence Discussion

The reliability of this extrapolation depends on several factors:

1. **Within-range reliability**: The power law fits our 5 data points from {sp_models[0]['n_params']:,} to {sp_models[-1]['n_params']:,} parameters. Extrapolating 10× beyond this range increases uncertainty substantially.

2. **Factors that could cause deviation**:
   - **Compute-optimal training**: With only 1 epoch on 108M tokens, larger models are increasingly undertrained. A 870M-param model would need far more data to reach its potential (Chinchilla scaling).
   - **Architecture effects**: Our models vary in both width and depth. The power law assumes a smooth relationship, but depth introduces discrete jumps in capacity.
   - **Tokenizer capacity**: With vocab=2,048, very large models may hit the information bottleneck of the tokenizer before the scaling law predicts.

3. **How far is reliable?**: Power laws are empirically reliable for 2-5× extrapolation in well-controlled settings. A 10× extrapolation should be treated as a rough estimate, not a precise prediction.

---

## 6. Analysis: Why μP Helps

### Why does a fixed LR degrade for larger models?

Under Standard Parameterization, when model width (d_model) increases:
- Weight matrices grow as d_model × d_model
- The gradient ∇L scales as O(1/√d_model) per entry, but there are O(d_model²) entries
- The total parameter update ∝ lr × ∇L changes scale with width
- Result: an LR optimal for d_model=128 causes excessively large updates at d_model=768

This is exactly what we observed in Part 2: LR=3e-2 worked for Tiny (d_model=128) but caused Medium/Large/XL to diverge.

### How does μP address this?

μP rescales three components to keep updates O(1) regardless of width:

1. **Output logits**: Scaled by 1/width_mult via MuReadout, preventing output explosion
2. **Attention scores**: Scaled by 1/d instead of 1/√d, keeping attention distributions stable
3. **Learning rate**: MuAdamW divides LR by width_mult for matrix-like weights, ensuring weight updates remain proportional regardless of width

The key insight: under μP, the optimal LR found at any width transfers to all other widths because the parameterization normalizes the update scale.

---

## 7. μP Training Curves

Training curves for all μP models are saved at `checkpoints/mup_training_curves.png`.

---

## 8. Reproducibility

| Item | Value |
|------|-------|
| mup package version | pip install mup |
| Base shapes | Tiny (d=128) → Delta (d=256) |
| GPU | NVIDIA A100 (Google Colab Pro) |
| SP training time | {sum(m['wall_clock_seconds'] for m in sp_models):.0f}s |
| μP training time | {sum(m['wall_clock_seconds'] for m in mup_models):.0f}s |
| Total Part 3 time | ~{sum(m['wall_clock_seconds'] for m in mup_models)/60:.0f} min (μP models only) |
"""

    report_path = os.path.join(output_dir, "REPORT_Part3.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved: {report_path}")


def main():
    print("Stage 11: Part 3 Scaling Analysis")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLR Sweep Comparison")
    sp_lr, mup_lr = plot_lr_comparison(SP_LR_DIR, MUP_LR_DIR, OUTPUT_DIR)

    print("\nScaling Comparison")
    sp_data, mup_data, sp_fit, mup_fit = plot_scaling_comparison(
        SP_SCALING_DIR, MUP_SCALING_DIR, OUTPUT_DIR)

    print("\nμP Training Curves")
    plot_mup_training_curves(MUP_SCALING_DIR, OUTPUT_DIR)

    print("\nExtrapolation")
    best_fit = mup_fit if mup_fit.get('r_squared', 0) > sp_fit.get('r_squared', 0) else sp_fit
    largest_params = max(m['n_params'] for m in sp_data['models'])
    extrap = extrapolate(best_fit, largest_params, factor=10)
    if extrap['predicted_loss'] is not None:
        print(f"  Predicted loss at {extrap['N_target']:,} params: {extrap['predicted_loss']:.4f}")
        if extrap['ci_low'] is not None:
            print(f"  95% CI: [{extrap['ci_low']:.4f}, {extrap['ci_high']:.4f}]")

    # Save extrapolation
    extrap_path = os.path.join(OUTPUT_DIR, "extrapolation.json")
    with open(extrap_path, 'w') as f:
        json.dump(extrap, f, indent=2)

    print("\nGenerating Report")
    generate_report(sp_lr, mup_lr, sp_data, mup_data, sp_fit, mup_fit, extrap, OUTPUT_DIR)

    print("\nStage 11 Complete")
    print(f"Plots: {OUTPUT_DIR}/")
    print(f"Report: {OUTPUT_DIR}/REPORT_Part3.md")


if __name__ == "__main__":
    main()
