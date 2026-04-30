import os
import json
import time
import torch
import numpy as np
import sentencepiece as spm

from model_mup import GPTConfig, create_mup_model


DATA_DIR = os.environ.get("DATA_DIR", "data")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints/best_model")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "generated")

TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer", "svg_bpe_2048.model")

BOS_ID = 1
EOS_ID = 2
PAD_ID = 0

MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95


def repair_svg(text):
    original = text
    text = text.strip()
    
    if not text:
        return text
    
    if text.endswith('</svg>'):
        return text
    
    import re
    
    quote_count = text.count('"')
    if quote_count % 2 == 1:
        text += '"'
    
    last_lt = text.rfind('<')
    last_gt = text.rfind('>')
    if last_lt > last_gt:
        tag_content = text[last_lt:]
        if tag_content.startswith('</'):  
            text += '>'
        elif '/' in tag_content:
            text += '>'
        else:
            text += '/>'
    
    if '</svg>' not in text:
        text += '</svg>'
    
    return text




@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=0.0,
             eos_id=None, device='cuda'):
    model.eval()
    block_size = model.config.block_size
    
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature  
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('inf')
        
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  
        idx = torch.cat((idx, idx_next), dim=1)
        
        if eos_id is not None and (idx_next == eos_id).all():
            break
    
    return idx


def load_model(checkpoint_path, device='cuda'):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config_dict = ckpt['config']
    config = GPTConfig(
        n_embd=config_dict['n_embd'],
        n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'],
        d_ff=config_dict['d_ff'],
        dropout=0.0,  
        block_size=config_dict['block_size'],
        vocab_size=config_dict['vocab_size'],
    )
    
    model = create_mup_model(config)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model.count_parameters():,} params")
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}, val_loss: {ckpt.get('val_loss', '?')}")
    
    return model


def generate_unconditional(model, sp, device, n_samples=15, temperature=0.8,
                           top_p=0.95, top_k=0, seed=None):
    samples = []
    
    for i in range(n_samples):
        if seed is not None:
            torch.manual_seed(seed + i)
        
        start = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
        
        tokens = generate(
            model, start, MAX_NEW_TOKENS,
            temperature=temperature, top_k=top_k, top_p=top_p,
            eos_id=EOS_ID, device=device
        )
        
        token_ids = tokens[0].tolist()
        if token_ids[0] == BOS_ID:
            token_ids = token_ids[1:]
        if EOS_ID in token_ids:
            token_ids = token_ids[:token_ids.index(EOS_ID)]
        
        text = sp.decode(token_ids)
        hit_eos = len(token_ids) < MAX_NEW_TOKENS
        
        if not hit_eos:
            text = repair_svg(text)
        
        samples.append({
            'id': i,
            'text': text,
            'n_tokens': len(token_ids),
            'hit_eos': hit_eos,
            'repaired': not hit_eos,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
        })
        
        print(f"    Sample {i+1}/{n_samples}: {len(token_ids)} tokens, "
              f"{len(text)} chars{' (repaired)' if not hit_eos else ''}")
    
    return samples


def generate_prefix_conditioned(model, sp, device, prefixes, temperature=0.8,
                                top_p=0.95, top_k=0, seed=42):
    samples = []
    
    for i, (prefix_text, description) in enumerate(prefixes):
        torch.manual_seed(seed + i)
        
        prefix_ids = sp.encode(prefix_text)
        start = torch.tensor([[BOS_ID] + prefix_ids], dtype=torch.long, device=device)
        
        tokens = generate(
            model, start, MAX_NEW_TOKENS,
            temperature=temperature, top_k=top_k, top_p=top_p,
            eos_id=EOS_ID, device=device
        )
        
        token_ids = tokens[0].tolist()
        if token_ids[0] == BOS_ID:
            token_ids = token_ids[1:]
        if EOS_ID in token_ids:
            token_ids = token_ids[:token_ids.index(EOS_ID)]
        
        full_text = sp.decode(token_ids)
        hit_eos = (len(token_ids) - len(prefix_ids)) < MAX_NEW_TOKENS
        
        if not hit_eos:
            full_text = repair_svg(full_text)
        
        completion_text = full_text[len(prefix_text):]  
        
        samples.append({
            'id': i,
            'prefix': prefix_text,
            'description': description,
            'full_text': full_text,
            'completion': completion_text,
            'n_tokens': len(token_ids),
            'prefix_tokens': len(prefix_ids),
            'hit_eos': hit_eos,
            'repaired': not hit_eos,
        })
        
        print(f"    Prefix {i+1}: '{description}' → "
              f"{len(token_ids) - len(prefix_ids)} new tokens"
              f"{' (repaired)' if not hit_eos else ''}")
    
    return samples


def generate_temperature_comparison(model, sp, device, temperatures=[0.5, 0.8, 1.0],
                                     seed=42, top_p=0.95):
    samples = []
    
    for temp in temperatures:
        torch.manual_seed(seed)  
        
        start = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
        tokens = generate(
            model, start, MAX_NEW_TOKENS,
            temperature=temp, top_k=0, top_p=top_p,
            eos_id=EOS_ID, device=device
        )
        
        token_ids = tokens[0].tolist()
        if token_ids[0] == BOS_ID:
            token_ids = token_ids[1:]
        if EOS_ID in token_ids:
            token_ids = token_ids[:token_ids.index(EOS_ID)]
        
        text = sp.decode(token_ids)
        hit_eos = len(token_ids) < MAX_NEW_TOKENS
        if not hit_eos:
            text = repair_svg(text)
        
        samples.append({
            'temperature': temp,
            'text': text,
            'n_tokens': len(token_ids),
            'hit_eos': hit_eos,
            'repaired': not hit_eos,
        })
        
        print(f"Temp={temp}: {len(token_ids)} tokens, {len(text)} chars"
              f"{' (repaired)' if not hit_eos else ''}")
    
    return samples


def main():
    print("Part 4 — Stage 13: Generate SVG Samples")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print(f"\nLoading tokenizer")
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    print(f"Vocab size: {sp.get_piece_size()}")

    print(f"\nLoading model")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        for e in range(15, 0, -1):
            alt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{e}.pt")
            if os.path.exists(alt_path):
                ckpt_path = alt_path
                break
    
    model = load_model(ckpt_path, device)

    os.makedirs(os.path.join(OUTPUT_DIR, "unconditional"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "prefix"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "temperature"), exist_ok=True)


    print("\n1. Unconditional Generation (15 samples)")
    
    t0 = time.time()
    unconditional = generate_unconditional(
        model, sp, device, n_samples=15,
        temperature=0.8, top_p=0.95, seed=42
    )
    print(f"Time: {time.time()-t0:.1f}s")
    
    for s in unconditional:
        path = os.path.join(OUTPUT_DIR, "unconditional", f"sample_{s['id']:02d}.svg")
        with open(path, 'w') as f:
            f.write(s['text'])
    
    with open(os.path.join(OUTPUT_DIR, "unconditional", "metadata.json"), 'w') as f:
        json.dump(unconditional, f, indent=2)

    print("\n2. Prefix-Conditioned Generation (5 samples)")

    prefixes = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" height="200px" viewBox="0.0 0.0 24.0 24.0" width="200px"><path d="M12.0 2.0 L12.0 2.0 C14.0 2.0',
            "Open path (emoji-style) → complete the icon"
        ),
        (
            '<svg xmlns="http://www.w3.org/2000/svg" height="200px" viewBox="0.0 0.0 24.0 24.0" width="200px"><path d="M3.6 1.2 L9.2 1.2 L14.8 1.2 L20.4 1.2',
            "Partial rectangle → complete the shape"
        ),
        (
            '<svg xmlns="http://www.w3.org/2000/svg" height="200px" viewBox="0.0 0.0 24.0 24.0" width="200px"><path d="M12.0 12.0',
            "Centered point → what shape emerges?"
        ),
        (
            '<svg xmlns="http://www.w3.org/2000/svg" height="200px" viewBox="0.0 0.0 24.0 24.0" width="200px"><path d="M3.0 12.0 L21.0 12.0',
            "Horizontal line → add vertical strokes?"
        ),
        (
            '<svg xmlns="http://www.w3.org/2000/svg" height="200px" viewBox="0.0 0.0 24.0 24.0" width="200px"><path d="M12.0 2.0 L12.0 2.0 C14.0 3.0 16.0 5.0 16.0 8.0" fill="none" filling="0" stroke="black" stroke-opacity="1.0" stroke-width=".3"/><path d="M12.0 12.0',
            "One completed path + start of second → related shapes?"
        ),
    ]

    t0 = time.time()
    prefix_samples = generate_prefix_conditioned(
        model, sp, device, prefixes,
        temperature=0.8, top_p=0.95, seed=42
    )
    print(f"Time: {time.time()-t0:.1f}s")
    
    for s in prefix_samples:
        path = os.path.join(OUTPUT_DIR, "prefix", f"prefix_{s['id']:02d}.svg")
        with open(path, 'w') as f:
            f.write(s['full_text'])
    
    with open(os.path.join(OUTPUT_DIR, "prefix", "metadata.json"), 'w') as f:
        json.dump(prefix_samples, f, indent=2)

    print("\n3. Temperature Comparison (0.5, 0.8, 1.0)")
    
    t0 = time.time()
    temp_samples = generate_temperature_comparison(
        model, sp, device,
        temperatures=[0.5, 0.8, 1.0],
        seed=42, top_p=0.95
    )
    print(f"Time: {time.time()-t0:.1f}s")
    
    # Save each SVG
    for s in temp_samples:
        temp_str = str(s['temperature']).replace('.', '_')
        path = os.path.join(OUTPUT_DIR, "temperature", f"temp_{temp_str}.svg")
        with open(path, 'w') as f:
            f.write(s['text'])
    
    with open(os.path.join(OUTPUT_DIR, "temperature", "metadata.json"), 'w') as f:
        json.dump(temp_samples, f, indent=2)

    # Summary
    print("\nGeneration Complete")
    print(f"Unconditional: {len(unconditional)} samples → {OUTPUT_DIR}/unconditional/")
    print(f"Prefix: {len(prefix_samples)} samples → {OUTPUT_DIR}/prefix/")
    print(f"Temperature: {len(temp_samples)} samples → {OUTPUT_DIR}/temperature/")
    
    # Quick stats
    avg_tokens = np.mean([s['n_tokens'] for s in unconditional])
    avg_chars = np.mean([len(s['text']) for s in unconditional])
    print(f"\n  Unconditional avg: {avg_tokens:.0f} tokens, {avg_chars:.0f} chars")


if __name__ == "__main__":
    main()
