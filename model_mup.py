import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from mup import MuReadout, set_base_shapes
from mup.optim import MuAdamW


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 2048
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    d_ff: int = 512
    dropout: float = 0.0
    bias: bool = False




MODEL_CONFIGS = {
    "tiny": GPTConfig(n_embd=128, n_layer=4, n_head=4, d_ff=512),
    "small": GPTConfig(n_embd=192, n_layer=6, n_head=6, d_ff=768),
    "medium": GPTConfig(n_embd=384, n_layer=6, n_head=6, d_ff=1536),
    "large": GPTConfig(n_embd=512, n_layer=10, n_head=8, d_ff=2048),
    "xl": GPTConfig(n_embd=768, n_layer=12, n_head=12, d_ff=3072),
}




class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention_MuP(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # μP: use 1/d scaling instead of 1/√d
        mup_scale = 1.0 / self.head_dim

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
                scale=mup_scale,  # μP change
            )
        else:
            att = (q @ k.transpose(-2, -1)) * mup_scale
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention_MuP(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




class GPT_MuP(nn.Module):


    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # μP: MuReadout for output head with μP output scaling
        # No weight tying — MuReadout needs independent weight for proper μP scaling
        self.lm_head = MuReadout(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights (set_base_shapes will rescale for μP)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def count_parameters(self):

        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Decay params: {num_decay:,}, No-decay params: {num_nodecay:,}")

        # μP: MuAdamW scales LR per layer based on width_mult
        optimizer = MuAdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer




def create_mup_model(config):

    # Base and delta must match target depth
    base_config = GPTConfig(
        n_embd=128, n_layer=config.n_layer, n_head=4, d_ff=512,
        block_size=config.block_size, vocab_size=config.vocab_size,
    )
    delta_config = GPTConfig(
        n_embd=256, n_layer=config.n_layer, n_head=4, d_ff=1024,
        block_size=config.block_size, vocab_size=config.vocab_size,
    )

    base_model = GPT_MuP(base_config)
    delta_model = GPT_MuP(delta_config)
    model = GPT_MuP(config)

    # set_base_shapes rescales model parameters for μP
    set_base_shapes(model, base_model, delta=delta_model)

    del base_model, delta_model
    return model
