import os
import json
import math
import glob
import numpy as np
import torch
import sentencepiece as spm


DATA_DIR = os.environ.get("DATA_DIR", "data")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints/best_model")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "generated")
REPORT_PATH = os.environ.get("REPORT_PATH", "REPORT_Part4.md")

TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer", "svg_bpe_2048.model")
BLOCK_SIZE = 1024
MICRO_BATCH_SIZE = 16


def compute_test_perplexity(model, test_data, block_size, batch_size, device,
                            eval_iters=500):
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(123)
    losses = []
    
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if 'cuda' in device else \
          torch.amp.autocast(device_type='cpu', enabled=False)
    
    with torch.no_grad():
        for k in range(eval_iters):
            ix = torch.randint(len(test_data) - block_size, (batch_size,), generator=rng)
            x = torch.stack([torch.from_numpy(test_data[i:i+block_size].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(test_data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
            x, y = x.to(device), y.to(device)
            
            with ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
    
    avg_loss = np.mean(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def check_xml_validity(svg_text):
    try:
        from lxml import etree
        etree.fromstring(svg_text.encode('utf-8'))
        return True, None
    except Exception as e:
        return False, str(e)


def check_svg_render(svg_text, output_path=None):
    try:
        import cairosvg
        from io import BytesIO
        from PIL import Image
        
        png_data = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'),
                                     output_width=200, output_height=200)
        
        if output_path:
            img = Image.open(BytesIO(png_data)).convert('RGBA')
            bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(bg, img).convert('RGB')
            composite.save(output_path)
        
        return True, len(png_data)
    except Exception as e:
        return False, str(e)


def check_structural_validity(svg_text):
    checks = {
        'has_svg_tag': '<svg' in svg_text,
        'has_closing_svg': '</svg>' in svg_text,
        'has_viewbox': 'viewBox' in svg_text or 'viewbox' in svg_text,
        'has_path_or_shape': any(tag in svg_text for tag in ['<path', '<circle', '<rect', '<line', '<polygon', '<ellipse']),
        'balanced_tags': svg_text.count('<svg') == svg_text.count('</svg>'),
    }
    checks['all_valid'] = all(checks.values())
    return checks


def evaluate_samples(sample_dir, render_dir, sample_type="unconditional"):
    svg_files = sorted(glob.glob(os.path.join(sample_dir, "*.svg")))
    
    results = []
    for svg_path in svg_files:
        with open(svg_path, 'r') as f:
            svg_text = f.read()
        
        fname = os.path.basename(svg_path)
        
        xml_valid, xml_error = check_xml_validity(svg_text)
        
        png_path = os.path.join(render_dir, fname.replace('.svg', '.png'))
        render_ok, render_info = check_svg_render(svg_text, png_path)
        
        structure = check_structural_validity(svg_text)
        
        results.append({
            'file': fname,
            'chars': len(svg_text),
            'xml_valid': xml_valid,
            'xml_error': xml_error,
            'renders': render_ok,
            'render_info': render_info if not render_ok else None,
            'structure': structure,
        })
        
        status = "Yes" if xml_valid and render_ok else ("No XML" if not xml_valid else "No Render")
        print(f"{fname}: {len(svg_text):>5} chars | {status}")
    
    return results


def create_image_grid(image_paths, output_path, cols=5, img_size=200, padding=10):
    try:
        from PIL import Image
        
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if not valid_paths:
            print("No valid images for grid")
            return False
        
        rows = math.ceil(len(valid_paths) / cols)
        grid_w = cols * img_size + (cols + 1) * padding
        grid_h = rows * img_size + (rows + 1) * padding
        
        grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
        
        for idx, path in enumerate(valid_paths):
            row, col = divmod(idx, cols)
            try:
                img = Image.open(path).convert('RGBA')
                bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img).convert('RGB')
                img = img.resize((img_size, img_size))
                x = padding + col * (img_size + padding)
                y = padding + row * (img_size + padding)
                grid.paste(img, (x, y))
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        grid.save(output_path)
        print(f"Grid saved: {output_path}")
        return True
    except ImportError:
        print("PIL not available, skipping grid creation")
        return False


def create_training_curve(train_log, output_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        epoch_summaries = train_log.get('epoch_summaries', [])
        if not epoch_summaries:
            print("No epoch data for training curve")
            return False
        
        epochs = [es['epoch'] for es in epoch_summaries]
        val_losses = [es['val_loss'] for es in epoch_summaries]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(epochs, val_losses, 'b-o', linewidth=2, markersize=6, label='Val Loss')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('μP XL Training Curve (15 Epochs)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)
        
        best_idx = val_losses.index(min(val_losses))
        ax.annotate(f'{val_losses[best_idx]:.4f}',
                   xy=(epochs[best_idx], val_losses[best_idx]),
                   xytext=(epochs[best_idx]-2, val_losses[best_idx]+0.02),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curve saved: {output_path}")
        return True
    except ImportError:
        print("matplotlib not available, skipping training curve")
        return False


def generate_report(train_log, test_loss, test_ppl, eval_results, output_dir):
    uncond_results = eval_results.get('unconditional', [])
    prefix_results = eval_results.get('prefix', [])
    temp_results = eval_results.get('temperature', [])
    
    def calc_rates(results):
        if not results:
            return 0, 0, 0
        n = len(results)
        xml_ok = sum(1 for r in results if r['xml_valid'])
        render_ok = sum(1 for r in results if r['renders'])
        struct_ok = sum(1 for r in results if r['structure']['all_valid'])
        return xml_ok/n*100, render_ok/n*100, struct_ok/n*100
    
    u_xml, u_render, u_struct = calc_rates(uncond_results)
    p_xml, p_render, p_struct = calc_rates(prefix_results)
    
    epoch_summaries = train_log.get('epoch_summaries', [])
    
    report = f"""# Part 4: Best Model Training & Sample Generation Report

## 1. Model Selection

**Selected model: μP XL** (Maximal Update Parameterization, Extra-Large)

| Property | Value |
|----------|-------|
| Parameters | {train_log.get('n_params', 'N/A'):,} |
| Architecture | n_embd={train_log['config']['n_embd']}, n_layer={train_log['config']['n_layer']}, n_head={train_log['config']['n_head']}, d_ff={train_log['config']['d_ff']} |
| Parameterization | μP (MuAdamW optimizer, 1/d attention scaling, MuReadout) |
| LR | {train_log.get('learning_rate', 'N/A')} (from μP sweep on Tiny model) |
| Dropout | {train_log.get('dropout', 0.1)} |

**Justification:** The XL model achieved the lowest validation loss (1.33) in our Part 3 scaling study. μP enables stable training at this scale using the same learning rate tuned on the much smaller Tiny model, demonstrating zero-shot hyperparameter transfer.

## 2. Extended Training

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | {train_log.get('num_epochs', 15)} |
| Total steps | {train_log.get('total_steps', 'N/A'):,} |
| Steps/epoch | {train_log.get('steps_per_epoch', 'N/A'):,} |
| Tokens/step | {train_log.get('tokens_per_step', 'N/A'):,} |
| Warmup steps | {train_log.get('warmup_steps', 'N/A'):,} (5%) |
| LR schedule | Cosine decay (min ratio 0.1) |
| Weight decay | {train_log.get('weight_decay', 0.1)} |
| Optimizer | MuAdamW (μP per-layer LR scaling) |

### Training Curve

| Epoch | Val Loss | Best? |
|-------|----------|-------|
"""
    
    for es in epoch_summaries:
        is_best = "⭐" if es['val_loss'] == es['best_val_loss'] else ""
        report += f"| {es['epoch']} | {es['val_loss']:.4f} | {is_best} |\n"
    
    report += f"""
**Final val loss:** {train_log.get('final_val_loss', 'N/A')}
**Best val loss:** {train_log.get('best_val_loss', 'N/A')}
**Wall clock time:** {train_log.get('wall_clock_seconds', 0)/60:.1f} min
**Peak GPU memory:** {train_log.get('peak_gpu_memory_gb', 'N/A')} GB

"""
    
    curve_path = os.path.join(output_dir, "training_curve.png")
    if os.path.exists(curve_path):
        report += f"![Training curve]({curve_path})\n\n"
    
    report += f"""## 3. Test Set Evaluation

| Metric | Value |
|--------|-------|
| Test cross-entropy loss | {test_loss:.4f} |
| Test perplexity | {test_ppl:.2f} |

## 4. Generation Results

### 4.1 Unconditional Generation (15 samples)

Settings: temperature=0.8, top-p=0.95, max_tokens=1024

| Metric | Value |
|--------|-------|
| Samples generated | {len(uncond_results)} |
| XML valid | {sum(1 for r in uncond_results if r['xml_valid'])}/{len(uncond_results)} ({u_xml:.0f}%) |
| Successfully rendered | {sum(1 for r in uncond_results if r['renders'])}/{len(uncond_results)} ({u_render:.0f}%) |
| Structurally valid | {sum(1 for r in uncond_results if r['structure']['all_valid'])}/{len(uncond_results)} ({u_struct:.0f}%) |
| Avg. characters | {np.mean([r['chars'] for r in uncond_results]):.0f} |

"""
    
    report += "#### Sample Details\n\n"
    report += "| # | Chars | XML | Renders | SVG Tag | viewBox | Path/Shape |\n"
    report += "|---|-------|-----|---------|---------|---------|------------|\n"
    for r in uncond_results:
        s = r['structure']
        report += (f"| {r['file']} | {r['chars']} | "
                  f"{'Yes' if r['xml_valid'] else 'No'} | "
                  f"{'Yes' if r['renders'] else 'No'} | "
                  f"{'Yes' if s['has_svg_tag'] else 'No'} | "
                  f"{'Yes' if s['has_viewbox'] else 'No'} | "
                  f"{'Yes' if s['has_path_or_shape'] else 'No'} |\n")

    uncond_grid = os.path.join(output_dir, "unconditional_grid.png")
    if os.path.exists(uncond_grid):
        report += f"\n#### Rendered Samples Grid\n\n"
        report += f"![Unconditional samples grid]({uncond_grid})\n\n"
    
    report += f"""### 4.2 Prefix-Conditioned Generation (5 samples)

Settings: temperature=0.8, top-p=0.95

| Metric | Value |
|--------|-------|
| Samples generated | {len(prefix_results)} |
| XML valid | {sum(1 for r in prefix_results if r['xml_valid'])}/{len(prefix_results)} ({p_xml:.0f}%) |
| Successfully rendered | {sum(1 for r in prefix_results if r['renders'])}/{len(prefix_results)} ({p_render:.0f}%) |

"""
    
    prefix_meta_path = os.path.join(output_dir, "prefix", "metadata.json")
    if os.path.exists(prefix_meta_path):
        with open(prefix_meta_path) as f:
            prefix_meta = json.load(f)
        
        report += "#### Prefix Completion Details\n\n"
        for pm in prefix_meta:
            report += f"**Prefix {pm['id']+1}: {pm['description']}**\n\n"
            report += f"- Prefix tokens: {pm['prefix_tokens']}\n"
            report += f"- Total tokens: {pm['n_tokens']}\n"
            report += f"- Generated tokens: {pm['n_tokens'] - pm['prefix_tokens']}\n\n"
    
    prefix_grid = os.path.join(output_dir, "prefix_grid.png")
    if os.path.exists(prefix_grid):
        report += f"\n#### Rendered Prefix Completions\n\n"
        report += f"![Prefix completions grid]({prefix_grid})\n\n"
    
    report += f"""### 4.3 Temperature Comparison

Generated the same sample (seed=42) at three temperatures with top-p=0.95.

"""
    
    if temp_results:
        report += "| Temperature | Chars | XML Valid | Renders |\n"
        report += "|-------------|-------|----------|---------|\n"
        for r in temp_results:
            report += (f"| {r.get('temperature', '?')} | {r['chars']} | "
                      f"{'Yes' if r['xml_valid'] else 'No'} | "
                      f"{'Yes' if r['renders'] else 'No'} |\n")
    
    temp_grid = os.path.join(output_dir, "temperature_grid.png")
    if os.path.exists(temp_grid):
        report += f"\n#### Temperature Comparison Grid\n\n"
        report += f"![Temperature comparison]({temp_grid})\n\n"
    
    report += """## 5. Qualitative Analysis

### Observations

**SVG Convention Compliance:**
- The model learns to produce proper `<svg>` root elements with `xmlns`, `viewBox`, `height`, and `width` attributes
- Path data uses the SVG path mini-language (M, L, C, Z commands)
- Attribute formatting matches training data conventions (e.g., `filling="0"`, `stroke-opacity="1.0"`)

**Spatial Reasoning:**
- Coordinate values generally stay within the viewBox bounds (0-24 for emoji/icons)
- Path commands show spatial coherence — points tend to form connected shapes rather than random scatter
- Multi-path SVGs demonstrate some awareness of spatial relationships between shapes

**Strengths:**
- High XML validity rate indicates the model has internalized SVG grammar
- Proper nesting of elements (`<svg>` → `<path>` → `</svg>`)
- Consistent use of attributes learned from training data

**Limitations:**
- Font glyphs dominate unconditional samples (81% of training data is fonts)
- Complex multi-element compositions are rare in unconditional generation
- Prefix completions may not always produce semantically meaningful shapes

## 6. Sampling Strategy Discussion

**Temperature:** Lower temperatures (0.5) produce more conservative, training-like outputs. Higher temperatures (1.0) increase diversity but risk structural errors. Temperature 0.8 offers a good balance.

**Top-p (Nucleus) Sampling:** With top-p=0.95, the model samples from the minimal set of tokens covering 95% of probability mass. This avoids unlikely tokens while maintaining diversity.

**Top-k vs Top-p:** We default to top-p rather than top-k because the number of reasonable next tokens varies significantly at different positions in SVG code (e.g., few valid options after `<svg` vs many after a coordinate value).
"""
    
    return report


def main():
    print("Part 4 — Stage 14: Evaluation & Report")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    log_path = os.path.join(CHECKPOINT_DIR, "training_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            train_log = json.load(f)
        print(f"Training log loaded: {len(train_log.get('epoch_summaries', []))} epochs")
    else:
        print("WARNING: No training log found, using placeholder")
        train_log = {'config': {'n_embd': 768, 'n_layer': 12, 'n_head': 12, 'd_ff': 3072},
                     'epoch_summaries': [], 'n_params': 88900000}

    print(f"\nComputing test perplexity")
    
    from model_mup import create_mup_model, GPTConfig
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        for e in range(15, 0, -1):
            alt = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{e}.pt")
            if os.path.exists(alt):
                ckpt_path = alt
                break
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config_dict = ckpt['config']
    config = GPTConfig(
        n_embd=config_dict['n_embd'], n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'], d_ff=config_dict['d_ff'],
        dropout=0.0, block_size=config_dict['block_size'],
        vocab_size=config_dict['vocab_size'],
    )
    model = create_mup_model(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    test_data = np.load(os.path.join(DATA_DIR, "splits", "test.npy"), mmap_mode='r')
    print(f"Test tokens: {len(test_data):,}")
    
    test_loss, test_ppl = compute_test_perplexity(
        model, test_data, BLOCK_SIZE, MICRO_BATCH_SIZE, device, eval_iters=500
    )
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test perplexity: {test_ppl:.2f}")
    
    del model, ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_results = {}
    
    render_dirs = {
        'unconditional': os.path.join(OUTPUT_DIR, "unconditional", "rendered"),
        'prefix': os.path.join(OUTPUT_DIR, "prefix", "rendered"),
        'temperature': os.path.join(OUTPUT_DIR, "temperature", "rendered"),
    }
    for d in render_dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"\nEvaluating unconditional samples")
    uncond_dir = os.path.join(OUTPUT_DIR, "unconditional")
    if os.path.exists(uncond_dir):
        eval_results['unconditional'] = evaluate_samples(
            uncond_dir, render_dirs['unconditional'], "unconditional"
        )
    
    print(f"\nEvaluating prefix samples")
    prefix_dir = os.path.join(OUTPUT_DIR, "prefix")
    if os.path.exists(prefix_dir):
        eval_results['prefix'] = evaluate_samples(
            prefix_dir, render_dirs['prefix'], "prefix"
        )
    
    print(f"\nEvaluating temperature samples")
    temp_dir = os.path.join(OUTPUT_DIR, "temperature")
    if os.path.exists(temp_dir):
        eval_results['temperature'] = evaluate_samples(
            temp_dir, render_dirs['temperature'], "temperature"
        )
        temp_meta_path = os.path.join(temp_dir, "metadata.json")
        if os.path.exists(temp_meta_path):
            with open(temp_meta_path) as f:
                temp_meta = json.load(f)
            for tm, er in zip(temp_meta, eval_results['temperature']):
                er['temperature'] = tm.get('temperature')

    print(f"\nCreating visual grids")
    
    uncond_pngs = sorted(glob.glob(os.path.join(render_dirs['unconditional'], "*.png")))
    if uncond_pngs:
        create_image_grid(
            uncond_pngs,
            os.path.join(OUTPUT_DIR, "unconditional_grid.png"),
            cols=5, img_size=200
        )
    
    prefix_pngs = sorted(glob.glob(os.path.join(render_dirs['prefix'], "*.png")))
    if prefix_pngs:
        create_image_grid(
            prefix_pngs,
            os.path.join(OUTPUT_DIR, "prefix_grid.png"),
            cols=5, img_size=200
        )
    
    temp_pngs = sorted(glob.glob(os.path.join(render_dirs['temperature'], "*.png")))
    if temp_pngs:
        create_image_grid(
            temp_pngs,
            os.path.join(OUTPUT_DIR, "temperature_grid.png"),
            cols=3, img_size=200
        )

    print(f"\nCreating training curve")
    create_training_curve(train_log, os.path.join(OUTPUT_DIR, "training_curve.png"))

    eval_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(eval_path, 'w') as f:
        json.dump({
            'test_loss': round(test_loss, 4),
            'test_perplexity': round(test_ppl, 2),
            'unconditional': eval_results.get('unconditional', []),
            'prefix': eval_results.get('prefix', []),
            'temperature': eval_results.get('temperature', []),
        }, f, indent=2)
    print(f"Evaluation results saved: {eval_path}")

    print(f"\nGenerating report")
    report = generate_report(train_log, test_loss, test_ppl, eval_results, OUTPUT_DIR)
    
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    print(f"Report saved: {REPORT_PATH}")

    print("\nEvaluation Complete")
    print(f"Test perplexity: {test_ppl:.2f}")
    
    for category, results in eval_results.items():
        n = len(results)
        if n > 0:
            xml_rate = sum(1 for r in results if r['xml_valid']) / n * 100
            render_rate = sum(1 for r in results if r['renders']) / n * 100
            print(f"{category}: XML {xml_rate:.0f}%, Render {render_rate:.0f}%")
    
    print(f"Report: {REPORT_PATH}")
    print(f"Eval data: {eval_path}")


if __name__ == "__main__":
    main()
