"""
NextLat from Scratch: Next-Latent Prediction in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - grid_step
def grid_step(pos: tuple, action: int, G: int) -> tuple:
    """
    Apply one action to a (row, col) position on a G x G grid.

    Args:
        pos: (row, col) current position
        action: 0=up, 1=down, 2=left, 3=right
        G: grid size (grid is G x G, valid coords are 0..G-1)

    Returns:
        (new_pos, legal) where new_pos is a tuple of plain ints,
        and legal is False (with new_pos == pos) if the move
        would leave the grid.
    """
    row, col = pos

    # Map action id to a (d_row, d_col) delta
    deltas = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }
    d_row, d_col = deltas[action]

    new_row = row + d_row
    new_col = col + d_col

    # Check bounds
    if 0 <= new_row < G and 0 <= new_col < G:
        return (int(new_row), int(new_col)), True
    else:
        return (int(row), int(col)), False

# Step 2 - legal_actions
def legal_actions(pos: tuple, G: int) -> list:
    """
    Return the sorted list of legal action ids from a position on a G x G grid.

    Args:
        pos: (row, col) current position
        G: grid size

    Returns:
        Sorted list of action ids (subset of [0, 1, 2, 3]) that are legal from pos.
    """
    legal = []
    for action in range(4):
        _, is_legal = grid_step(pos, action, G)
        if is_legal:
            legal.append(action)

    return sorted(legal)

# Step 3 - random_walk_to_goal
def random_walk_to_goal(start: tuple, goal: tuple, G: int, max_len: int, rng) -> list:
    """
    Generate a random sequence of legal moves from start until goal is reached
    or max_len moves have been taken.

    Args:
        start: (row, col) starting position
        goal: (row, col) goal position
        G: grid size
        max_len: maximum number of moves to take
        rng: np.random.Generator instance

    Returns:
        List of action ids taken (possibly empty if start == goal).
    """
    if start == goal:
        return []

    actions_taken = []
    pos = start

    for _ in range(max_len):
        actions = legal_actions(pos, G)
        action = int(rng.choice(actions))
        pos, _ = grid_step(pos, action, G)
        actions_taken.append(action)

        if pos == goal:
            break

    return actions_taken

# Step 4 - encode_sequence
import torch

def encode_sequence(start: tuple, goal: tuple, moves: list, G: int, T: int) -> tuple:
    """
    Build [start_cell, goal_cell, moves..., EOS, pad...] as a (T,) long tensor plus a bool mask.

    Vocabulary:
        actions: 0..3
        cell tokens: 4 + row*G + col
        EOS (also used as padding): 4 + G*G

    Args:
        start: (row, col) start position
        goal: (row, col) goal position
        moves: list of action ids taken
        G: grid size
        T: fixed output sequence length

    Returns:
        (tokens, mask): tokens is torch.long shape (T,), mask is torch.bool shape (T,)
    """
    EOS = 4 + G * G

    def cell(pos):
        row, col = pos
        return 4 + row * G + col

    base = [cell(start), cell(goal)] + list(moves)

    # Truncate so that one EOS still fits within length T
    base = base[:T - 1]

    real_tokens = base + [EOS]
    num_real = len(real_tokens)

    # Pad with EOS up to length T
    pad_len = T - num_real
    tokens_list = real_tokens + [EOS] * pad_len

    tokens = torch.tensor(tokens_list, dtype=torch.long)
    mask = torch.tensor([True] * num_real + [False] * pad_len, dtype=torch.bool)

    return tokens, mask

# Step 5 - make_dataset
import numpy as np
import torch

def make_dataset(n: int, G: int, T: int, seed: int = 0) -> dict:
    """
    Generate n encoded goal-directed walks on a G x G grid, plus the true
    cell index the walker occupies after consuming every token.

    Args:
        n: number of walks (dataset size)
        G: grid size
        T: fixed encoded sequence length
        seed: RNG seed

    Returns:
        dict with:
            'tokens': (n, T) long
            'mask':   (n, T) bool
            'states': (n, T) long  -- cell index row*G+col after each token
            'G': int
    """
    rng = np.random.default_rng(seed)
    max_len = T - 3

    all_tokens = []
    all_masks = []
    all_states = []

    for _ in range(n):
        start = tuple(int(x) for x in rng.integers(0, G, size=2))
        goal = tuple(int(x) for x in rng.integers(0, G, size=2))

        moves = random_walk_to_goal(start, goal, G, max_len, rng)
        tokens, mask = encode_sequence(start, goal, moves, G, T)

        states_row = torch.zeros(T, dtype=torch.long)

        pos = start
        cell_idx = pos[0] * G + pos[1]
        states_row[0] = cell_idx  # start token: walker hasn't moved
        states_row[1] = cell_idx  # goal token: walker still hasn't moved

        for j, action in enumerate(moves):
            pos, _ = grid_step(pos, action, G)
            states_row[2 + j] = pos[0] * G + pos[1]

        # EOS and any padding: position stays fixed at wherever the walk ended
        last_filled = 2 + len(moves)
        final_cell_idx = pos[0] * G + pos[1]
        for k in range(last_filled, T):
            states_row[k] = final_cell_idx

        all_tokens.append(tokens)
        all_masks.append(mask)
        all_states.append(states_row)

    return {
        'tokens': torch.stack(all_tokens),
        'mask': torch.stack(all_masks),
        'states': torch.stack(all_states),
        'G': G
    }

# Step 6 - get_batch
def get_batch(dataset: dict, batch_size: int, step: int) -> dict:
    """
    Slice a deterministic training batch out of the dataset, cyclically by row,
    and return the next-token-prediction shifted views.

    Args:
        dataset: dict with 'tokens' (n, T), 'mask' (n, T), 'states' (n, T), 'G'
        batch_size: number of rows in the batch
        step: which batch step (determines which rows are selected, cyclically)

    Returns:
        dict with:
            'x': tokens[:, :-1] for the selected rows      -> (batch_size, T-1)
            'y': tokens[:, 1:]  for the selected rows      -> (batch_size, T-1)
            'mask': mask[:, 1:] for the selected rows      -> (batch_size, T-1)
            'states': states[:, :-1] for the selected rows -> (batch_size, T-1)
    """
    n = dataset['tokens'].shape[0]

    indices = [(step * batch_size + i) % n for i in range(batch_size)]

    tokens = dataset['tokens'][indices]
    mask = dataset['mask'][indices]
    states = dataset['states'][indices]

    x = tokens[:, :-1]
    y = tokens[:, 1:]
    y_mask = mask[:, 1:]
    x_states = states[:, :-1]

    return {
        'x': x,
        'y': y,
        'mask': y_mask,
        'states': x_states
    }

# Step 7 - causal_mask
import torch

def causal_mask(T: int):
    """
    Return a (T, T) bool tensor, True where key index <= query index.

    Row i = query position i, column j = key position j.
    Entry (i, j) is True exactly when j <= i (lower-triangular, including diagonal).

    Args:
        T: sequence length

    Returns:
        (T, T) torch.bool tensor
    """
    return torch.tril(torch.ones(T, T, dtype=torch.bool))

# Step 8 - init_gpt_params
import torch

def init_gpt_params(vocab_size: int, d_model: int, n_layers: int, max_len: int, seed: int = 0) -> dict:
    """
    Create the parameter dictionary for a small pre-LN GPT.

    Args:
        vocab_size: size of the token vocabulary
        d_model: model (embedding) dimension
        n_layers: number of transformer blocks
        max_len: maximum sequence length (for position embeddings)
        seed: RNG seed for reproducibility

    Returns:
        dict of float32, requires_grad=True tensors, keyed as described.
    """
    torch.manual_seed(seed)

    params = {}

    params['wte'] = (torch.randn(vocab_size, d_model) * 0.02).float().requires_grad_(True)
    params['wpe'] = (torch.randn(max_len, d_model) * 0.02).float().requires_grad_(True)

    for l in range(n_layers):
        params[f'ln1_w{l}'] = torch.ones(d_model, dtype=torch.float32).requires_grad_(True)
        params[f'ln1_b{l}'] = torch.zeros(d_model, dtype=torch.float32).requires_grad_(True)

        params[f'qkv_w{l}'] = (torch.randn(d_model, 3 * d_model) * 0.02).float().requires_grad_(True)
        params[f'qkv_b{l}'] = torch.zeros(3 * d_model, dtype=torch.float32).requires_grad_(True)

        params[f'proj_w{l}'] = (torch.randn(d_model, d_model) * 0.02).float().requires_grad_(True)
        params[f'proj_b{l}'] = torch.zeros(d_model, dtype=torch.float32).requires_grad_(True)

        params[f'ln2_w{l}'] = torch.ones(d_model, dtype=torch.float32).requires_grad_(True)
        params[f'ln2_b{l}'] = torch.zeros(d_model, dtype=torch.float32).requires_grad_(True)

        params[f'fc_w{l}'] = (torch.randn(d_model, 4 * d_model) * 0.02).float().requires_grad_(True)
        params[f'fc_b{l}'] = torch.zeros(4 * d_model, dtype=torch.float32).requires_grad_(True)

        params[f'fc2_w{l}'] = (torch.randn(4 * d_model, d_model) * 0.02).float().requires_grad_(True)
        params[f'fc2_b{l}'] = torch.zeros(d_model, dtype=torch.float32).requires_grad_(True)

    params['lnf_w'] = torch.ones(d_model, dtype=torch.float32).requires_grad_(True)
    params['lnf_b'] = torch.zeros(d_model, dtype=torch.float32).requires_grad_(True)

    params['head_w'] = (torch.randn(d_model, vocab_size) * 0.02).float().requires_grad_(True)
    params['head_b'] = torch.zeros(vocab_size, dtype=torch.float32).requires_grad_(True)

    return params

# Step 9 - attention_block
import torch
import torch.nn.functional as F
import math

def attention_block(x, params: dict, layer: int, n_heads: int):
    """
    One pre-LayerNorm causal multi-head self-attention block with a residual
    connection: x + Proj(CausalMultiHeadAttn(LayerNorm(x))).

    Args:
        x: (B, T, d) input
        params: parameter dict from init_gpt_params
        layer: which layer's parameters to use
        n_heads: number of attention heads (d must be divisible by n_heads)

    Returns:
        (B, T, d) output
    """
    B, T, d = x.shape
    head_dim = d // n_heads

    ln1_w = params[f'ln1_w{layer}']
    ln1_b = params[f'ln1_b{layer}']
    qkv_w = params[f'qkv_w{layer}']
    qkv_b = params[f'qkv_b{layer}']
    proj_w = params[f'proj_w{layer}']
    proj_b = params[f'proj_b{layer}']

    # Pre-LayerNorm
    z = F.layer_norm(x, (d,), ln1_w, ln1_b, eps=1e-5)

    # Fused QKV projection, then split
    qkv = z @ qkv_w + qkv_b            # (B, T, 3*d)
    q, k, v = qkv.split(d, dim=-1)     # each (B, T, d)

    # Reshape into heads: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
    q = q.reshape(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, T, n_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention scores
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)  # (B, n_heads, T, T)

    # Causal mask: block attending to future positions
    mask = causal_mask(T).to(x.device)          # (T, T), True = allowed
    scores = scores.masked_fill(~mask, float('-inf'))

    attn = F.softmax(scores, dim=-1)
    out = attn @ v                                # (B, n_heads, T, head_dim)

    # Merge heads back
    out = out.transpose(1, 2).reshape(B, T, d)    # (B, T, d)

    # Output projection
    out = out @ proj_w + proj_b

    # Residual connection
    return x + out

# Step 10 - mlp_block
import torch
import torch.nn.functional as F

def mlp_block(x, params: dict, layer: int):
    """
    One pre-LayerNorm position-wise feed-forward block with a residual
    connection: x + FC2(gelu_tanh(FC1(LayerNorm(x)))).

    Args:
        x: (B, T, d) input
        params: parameter dict from init_gpt_params
        layer: which layer's parameters to use

    Returns:
        (B, T, d) output
    """
    B, T, d = x.shape

    ln2_w = params[f'ln2_w{layer}']
    ln2_b = params[f'ln2_b{layer}']
    fc_w = params[f'fc_w{layer}']
    fc_b = params[f'fc_b{layer}']
    fc2_w = params[f'fc2_w{layer}']
    fc2_b = params[f'fc2_b{layer}']

    # Pre-LayerNorm
    z = F.layer_norm(x, (d,), ln2_w, ln2_b, eps=1e-5)

    # Expand to 4*d, apply GELU (tanh approximation)
    h = F.gelu(z @ fc_w + fc_b, approximate='tanh')

    # Project back down to d
    out = h @ fc2_w + fc2_b

    # Residual connection
    return x + out

# Step 11 - gpt_hidden_states
import torch
import torch.nn.functional as F

def gpt_hidden_states(tokens, params: dict, n_heads: int):
    """
    Run the full transformer and return the final hidden states (B, T, d).

    Args:
        tokens: (B, T) long tensor of token ids
        params: parameter dict from init_gpt_params
        n_heads: number of attention heads

    Returns:
        (B, T, d) hidden states h_t -- the pre-logit activations that
        parameterize the next-token distribution at each position.
    """
    B, T = tokens.shape

    n_layers = sum(1 for k in params if k.startswith('ln1_w'))

    # Token + position embeddings
    x = params['wte'][tokens] + params['wpe'][:T]

    # Stack of pre-LN transformer blocks
    for layer in range(n_layers):
        x = attention_block(x, params, layer, n_heads)
        x = mlp_block(x, params, layer)

    # Final LayerNorm
    d = x.shape[-1]
    h = F.layer_norm(x, (d,), params['lnf_w'], params['lnf_b'], eps=1e-5)

    return h

# Step 12 - output_head
def output_head(h, params: dict):
    """
    Map hidden states to next-token logits: h @ head_w + head_b.

    Args:
        h: (..., d) hidden states, any number of leading dimensions
        params: parameter dict containing 'head_w' (d, vocab_size) and 'head_b' (vocab_size,)

    Returns:
        (..., vocab_size) logits
    """
    return h @ params['head_w'] + params['head_b']

# Step 13 - next_token_loss
import torch
import torch.nn.functional as F

def next_token_loss(logits, targets, mask):
    """
    Masked mean cross-entropy between logits and targets.

    Args:
        logits: (B, T, V) raw (unnormalized) logits
        targets: (B, T) long, target token ids
        mask: (B, T) bool, True where the position should be scored

    Returns:
        0-dim tensor: mean cross-entropy over masked-in positions,
        or torch.tensor(0.0) if no position is masked in.
    """
    B, T, V = logits.shape

    per_position_loss = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        reduction='none'
    )  # (B*T,)

    mask_flat = mask.reshape(-1).float()

    total_mask = mask_flat.sum()
    if total_mask.item() == 0:
        return torch.tensor(0.0)

    masked_loss = (per_position_loss * mask_flat).sum() / total_mask

    return masked_loss

# Step 14 - init_dynamics_params
import torch

def init_dynamics_params(d_model: int, hidden: int, seed: int = 0) -> dict:
    """
    Create the parameter dictionary for the latent transition MLP (p_psi).

    Args:
        d_model: dimension of the GPT hidden states
        hidden: hidden layer width of the dynamics MLP
        seed: RNG seed

    Returns:
        dict with keys 'W1','b1','W2','b2','W3','b3', all float32, requires_grad=True.
    """
    torch.manual_seed(seed)

    dyn = {}

    dyn['W1'] = (torch.randn(2 * d_model, hidden) * 0.02).float().requires_grad_(True)
    dyn['b1'] = torch.zeros(hidden, dtype=torch.float32).requires_grad_(True)

    dyn['W2'] = (torch.randn(hidden, hidden) * 0.02).float().requires_grad_(True)
    dyn['b2'] = torch.zeros(hidden, dtype=torch.float32).requires_grad_(True)

    dyn['W3'] = (torch.randn(hidden, d_model) * 0.02).float().requires_grad_(True)
    dyn['b3'] = torch.zeros(d_model, dtype=torch.float32).requires_grad_(True)

    return dyn

# Step 15 - latent_transition
import torch
import torch.nn.functional as F

def latent_transition(h, x_emb, dyn: dict):
    """
    Predict the next hidden state as a residual delta:
    h_hat_{t+1} = f_psi(h_t, x_{t+1}) + h_t

    Args:
        h: (..., d_model) current hidden state
        x_emb: (..., d_model) embedding of the next token
        dyn: parameter dict from init_dynamics_params

    Returns:
        (..., d_model) predicted next hidden state
    """
    z = torch.cat([h, x_emb], dim=-1)                       # (..., 2*d_model)
    z = F.layer_norm(z, (z.shape[-1],), eps=1e-5)            # no learned scale/shift

    a1 = F.gelu(z @ dyn['W1'] + dyn['b1'], approximate='tanh')
    a2 = F.gelu(a1 @ dyn['W2'] + dyn['b2'], approximate='tanh')
    delta = a2 @ dyn['W3'] + dyn['b3']

    return delta + h

# Step 16 - rollout_latents
def rollout_latents(h, x, params: dict, dyn: dict, d_steps: int) -> list:
    """
    Recursively roll the latent transition forward d_steps steps from every
    start position, feeding predictions back in (not the true hidden states).

    Args:
        h: (B, T, d) hidden states aligned with input tokens x
        x: (B, T) long, input token ids
        params: GPT parameter dict (used here just for 'wte')
        dyn: dynamics parameter dict from init_dynamics_params
        d_steps: rollout horizon

    Returns:
        List of d_steps tensors, each (B, T-d_steps, d). The i-th entry
        (1-indexed) predicts h[:, i:T-d_steps+i].
    """
    B, T, d = h.shape

    h_hat = h[:, :T - d_steps]
    outs = []

    for i in range(1, d_steps + 1):
        emb = params['wte'][x[:, i:T - d_steps + i]]
        h_hat = latent_transition(h_hat, emb, dyn)
        outs.append(h_hat)

    return outs

# Step 17 - next_hidden_loss
import torch
import torch.nn.functional as F

def next_hidden_loss(h, h_hats: list, mask, beta: float = 1.0):
    """
    Stop-gradient Smooth L1 loss between rolled-out latents and the real
    hidden states, averaged over masked positions and rollout steps.

    Args:
        h: (B, T, d) real hidden states
        h_hats: list of d_steps predicted tensors from rollout_latents,
                 each (B, T-d_steps, d)
        mask: (B, T) bool, True at real (non-padding) input positions
        beta: Smooth L1 transition point

    Returns:
        0-dim tensor loss; torch.tensor(0.0) if h_hats is empty.
    """
    d_steps = len(h_hats)

    if d_steps == 0:
        return torch.tensor(0.0)

    B, T, d = h.shape

    step_losses = []

    for idx in range(d_steps):
        i = idx + 1  # 1-based step index

        target = h[:, i:T - d_steps + i].detach()
        m = mask[:, i:T - d_steps + i].float()

        pred = h_hats[idx]

        per_elem = F.smooth_l1_loss(pred, target, reduction='none', beta=beta)  # (B, T-d_steps, d)
        per_position = per_elem.mean(dim=-1)  # (B, T-d_steps)

        masked_mean = (per_position * m).sum() / m.sum()
        step_losses.append(masked_mean)

    total = torch.stack(step_losses).mean()

    return total

# Step 18 - kl_alignment_loss
import torch
import torch.nn.functional as F

def kl_alignment_loss(h, h_hats: list, mask, params: dict):
    """
    Forward KL divergence, in token space, between the distributions induced
    by the real and predicted latents, using a frozen (detached) output head.

    Args:
        h: (B, T, d) real hidden states
        h_hats: list of d_steps predicted tensors from rollout_latents,
                 each (B, T-d_steps, d)
        mask: (B, T) bool, True at real (non-padding) input positions
        params: GPT parameter dict (contains 'head_w', 'head_b')

    Returns:
        0-dim tensor loss; torch.tensor(0.0) if h_hats is empty.
    """
    d_steps = len(h_hats)

    if d_steps == 0:
        return torch.tensor(0.0)

    B, T, d = h.shape

    frozen = {
        'head_w': params['head_w'].detach(),
        'head_b': params['head_b'].detach()
    }

    step_losses = []

    for idx in range(d_steps):
        i = idx + 1  # 1-based step index

        h_true_slice = h[:, i:T - d_steps + i].detach()
        h_pred_slice = h_hats[idx]
        m = mask[:, i:T - d_steps + i].float()

        logits_true = output_head(h_true_slice, frozen)
        logits_pred = output_head(h_pred_slice, frozen)

        lp = F.log_softmax(logits_true, dim=-1)
        lq = F.log_softmax(logits_pred, dim=-1)

        kl = (lp.exp() * (lp - lq)).sum(dim=-1)  # (B, T-d_steps)

        masked_mean = (kl * m).sum() / m.sum()
        step_losses.append(masked_mean)

    total = torch.stack(step_losses).mean()

    return total

# Step 19 - nextlat_loss
import torch

def nextlat_loss(batch: dict, params: dict, dyn: dict, n_heads: int, d_steps: int,
                 lam_h: float, lam_kl: float, beta: float = 1.0) -> dict:
    """
    Compute the full NextLat training objective on one batch:
    total = next_token + lam_h * next_h + lam_kl * kl

    Args:
        batch: dict with 'x' (B, T), 'y' (B, T), 'mask' (B, T) from get_batch
        params: GPT parameter dict
        dyn: dynamics parameter dict
        n_heads: number of attention heads
        d_steps: rollout horizon (0 disables the auxiliary terms entirely)
        lam_h: weight on the next-hidden Smooth L1 loss
        lam_kl: weight on the KL alignment loss
        beta: Smooth L1 beta parameter

    Returns:
        dict of 0-dim tensors: 'total', 'next_token', 'next_h', 'kl'
    """
    x = batch['x']
    y = batch['y']
    mask = batch['mask']

    h = gpt_hidden_states(x, params, n_heads)
    logits = output_head(h, params)
    next_token = next_token_loss(logits, y, mask)

    if d_steps > 0:
        eos = params['head_b'].shape[0] - 1
        mask_x = x != eos

        hats = rollout_latents(h, x, params=params, dyn=dyn, d_steps=d_steps)

        next_h = next_hidden_loss(h, hats, mask_x, beta=beta)
        kl = kl_alignment_loss(h, hats, mask_x, params)
    else:
        next_h = torch.tensor(0.0)
        kl = torch.tensor(0.0)

    total = next_token + lam_h * next_h + lam_kl * kl

    return {
        'total': total,
        'next_token': next_token,
        'next_h': next_h,
        'kl': kl
    }

# Step 20 - train_step
def train_step(batch: dict, params: dict, dyn: dict, opt, n_heads: int, d_steps: int,
               lam_h: float, lam_kl: float, beta: float = 1.0) -> dict:
    """
    Perform one optimizer update on the NextLat objective.

    Args:
        batch: dict with 'x', 'y', 'mask' from get_batch
        params: GPT parameter dict
        dyn: dynamics parameter dict
        opt: torch.optim optimizer holding both params' and dyn's tensors
        n_heads: number of attention heads
        d_steps: rollout horizon
        lam_h: weight on next-hidden loss
        lam_kl: weight on KL alignment loss
        beta: Smooth L1 beta parameter

    Returns:
        dict with keys 'total', 'next_token', 'next_h', 'kl', each a Python float.
    """
    opt.zero_grad()

    out = nextlat_loss(batch, params, dyn, n_heads, d_steps, lam_h, lam_kl, beta)

    out['total'].backward()
    opt.step()

    return {k: v.item() for k, v in out.items()}

# Step 21 - train_model
import torch

def train_model(dataset: dict, cfg: dict, seed: int = 0) -> tuple:
    """
    Train a GPT or a NextLat model on the dataset.

    Args:
        dataset: dict with 'tokens' (n, T), 'mask' (n, T), 'states' (n, T), 'G'
        cfg: dict with keys d_model, n_layers, n_heads, hidden, steps,
             batch_size, lr, d_steps, lam_h, lam_kl, beta
        seed: RNG seed for parameter initialization

    Returns:
        (params, dyn, history) where history is a list of per-step loss dicts.
    """
    G = dataset['G']
    T = dataset['tokens'].shape[1]

    vocab_size = 4 + G * G + 1
    max_len = T

    params = init_gpt_params(vocab_size, cfg['d_model'], cfg['n_layers'], max_len, seed=seed)
    dyn = init_dynamics_params(cfg['d_model'], cfg['hidden'], seed=seed)

    opt = torch.optim.Adam(list(params.values()) + list(dyn.values()), lr=cfg['lr'])

    history = []

    for step in range(cfg['steps']):
        batch = get_batch(dataset, cfg['batch_size'], step)
        out = train_step(
            batch, params, dyn, opt,
            n_heads=cfg['n_heads'],
            d_steps=cfg['d_steps'],
            lam_h=cfg['lam_h'],
            lam_kl=cfg['lam_kl'],
            beta=cfg['beta']
        )
        history.append(out)

    return params, dyn, history

# Step 22 - greedy_decode
import torch

def greedy_decode(params: dict, n_heads: int, prefix: list, n_tokens: int) -> list:
    """
    Generate n_tokens tokens autoregressively using the transformer alone,
    via greedy (argmax) decoding.

    Args:
        params: GPT parameter dict
        n_heads: number of attention heads
        prefix: list of int token ids to start from
        n_tokens: number of new tokens to generate

    Returns:
        List of the n_tokens generated ints (not including the prefix).
    """
    seq = list(prefix)
    generated = []

    with torch.no_grad():
        for _ in range(n_tokens):
            x = torch.tensor([seq], dtype=torch.long)
            h = gpt_hidden_states(x, params, n_heads)
            logits = output_head(h[0, -1], params)
            tok = int(torch.argmax(logits))

            seq.append(tok)
            generated.append(tok)

    return generated

# Step 23 - effective_rank
import torch

def effective_rank(H, tol: float = 1e-12) -> float:
    """
    Roy-Vetterli effective rank: exp of the Shannon entropy (natural log)
    of the normalized singular values above tol.

    Args:
        H: (N, D) matrix (e.g. stacked hidden states)
        tol: singular values <= tol are discarded

    Returns:
        Python float; 0.0 if no singular value survives.
    """
    s = torch.linalg.svdvals(H)
    s = s[s > tol]

    if s.numel() == 0:
        return 0.0

    p = s / s.sum()
    entropy = -(p * p.log()).sum()

    return float(torch.exp(entropy))

# Step 24 - eval_hidden_states
import torch

def eval_hidden_states(dataset: dict, params: dict, n_heads: int, n_rows: int):
    """
    Collect the transformer's hidden states at every real input position
    of the first n_rows sequences.

    Args:
        dataset: dict with 'tokens' (n, T), 'mask' (n, T), 'states' (n, T), 'G'
        params: GPT parameter dict
        n_heads: number of attention heads
        n_rows: how many sequences to use (from the start of the dataset)

    Returns:
        Detached (N, d) matrix, where N is the number of real input positions.
    """
    x = dataset['tokens'][:n_rows, :-1]
    mask_x = dataset['mask'][:n_rows, :-1]

    with torch.no_grad():
        h = gpt_hidden_states(x, params, n_heads)

    return h[mask_x]

# Step 25 - valid_move_rate
import torch

def valid_move_rate(dataset: dict, params: dict, n_heads: int, n_rows: int) -> float:
    """
    Fraction of top-1 next-token predictions that are legal under the true
    world model, scored at positions t >= 1 with real targets.

    Args:
        dataset: dict with 'tokens', 'mask', 'states', 'G'
        params: GPT parameter dict
        n_heads: number of attention heads
        n_rows: number of sequences to evaluate (from the start)

    Returns:
        Float in [0, 1]; 0.0 if nothing is scored.
    """
    G = dataset['G']
    EOS = 4 + G * G

    tokens = dataset['tokens'][:n_rows]
    x = tokens[:, :-1]
    y_mask = dataset['mask'][:n_rows, 1:]
    pos = dataset['states'][:n_rows, :-1]
    goal = tokens[:, 1] - 4  # goal cell index per row

    with torch.no_grad():
        h = gpt_hidden_states(x, params, n_heads)
        logits = output_head(h, params)
        pred = logits.argmax(dim=-1)  # (n_rows, T-1)

    n_rows_actual, L = pred.shape
    legal_count = 0
    total = 0

    for i in range(n_rows_actual):
        g = int(goal[i])
        for t in range(1, L):
            if not bool(y_mask[i, t]):
                continue

            p_idx = int(pos[i, t])
            p = int(pred[i, t])
            total += 1

            if p_idx == g:
                # Standing on the goal: only EOS is legal
                if p == EOS:
                    legal_count += 1
            else:
                # Elsewhere: must be an action that stays on the grid
                cell = divmod(p_idx, G)
                if p in legal_actions(cell, G):
                    legal_count += 1

    if total == 0:
        return 0.0

    return legal_count / total

# Step 26 - sequence_compression
def sequence_compression(dataset: dict, params: dict, n_heads: int, n_tokens: int, max_pairs: int) -> float:
    """
    Fraction of prefix pairs that reach the same true state with the same goal
    for which greedy decoding produces identical continuations.

    Args:
        dataset: dict with 'tokens', 'mask', 'states', 'G'
        params: GPT parameter dict
        n_heads: number of attention heads
        n_tokens: number of tokens to decode from each prefix
        max_pairs: maximum number of pairs to evaluate

    Returns:
        Float in [0, 1]; 0.0 if there are no pairs.
    """
    tokens = dataset['tokens']
    mask = dataset['mask']
    states = dataset['states']
    n, T = tokens.shape

    groups = {}

    for i in range(n):
        for t in range(2, T):
            if not bool(mask[i, t]):
                continue
            if int(tokens[i, t]) >= 4:
                continue
            if t + 1 + n_tokens > T:
                continue

            prefix = tokens[i, :t + 1].tolist()
            key = (int(states[i, t]), int(tokens[i, 1]))

            if key not in groups:
                groups[key] = [prefix]
            elif len(groups[key]) < 2 and prefix != groups[key][0]:
                groups[key].append(prefix)

    pairs = [v for v in groups.values() if len(v) == 2][:max_pairs]

    if len(pairs) == 0:
        return 0.0

    matches = 0
    for p1, p2 in pairs:
        c1 = greedy_decode(params, n_heads, p1, n_tokens)
        c2 = greedy_decode(params, n_heads, p2, n_tokens)
        if c1 == c2:
            matches += 1

    return matches / len(pairs)

# Step 27 - detour_robustness
import numpy as np

def detour_robustness(params: dict, n_heads: int, G: int, max_steps: int, n_trials: int,
                      detour_prob: float = 0.75, seed: int = 0) -> float:
    """
    Fraction of episodes in which the model still reaches the goal by legal
    moves when its choices are replaced by random legal moves with
    probability detour_prob.

    Args:
        params: GPT parameter dict
        n_heads: number of attention heads
        G: grid size
        max_steps: step budget per episode
        n_trials: number of episodes
        detour_prob: probability of replacing the model's move with a random legal move
        seed: RNG seed

    Returns:
        Success fraction as a float.
    """
    rng = np.random.default_rng(seed)

    def cell(pos):
        return 4 + pos[0] * G + pos[1]

    successes = 0

    for _ in range(n_trials):
        start = tuple(int(v) for v in rng.integers(0, G, size=2))
        goal = tuple(int(v) for v in rng.integers(0, G, size=2))

        seq = [cell(start), cell(goal)]
        pos = start
        failed = False

        for _ in range(max_steps):
            if pos == goal:
                break

            if rng.random() < detour_prob:
                tok = int(rng.choice(legal_actions(pos, G)))
            else:
                tok = greedy_decode(params, n_heads, seq, 1)[0]

            if tok not in legal_actions(pos, G):
                failed = True
                break

            pos, _ = grid_step(pos, tok, G)
            seq.append(tok)

        if not failed and pos == goal:
            successes += 1

    return successes / n_trials

# Step 28 - world_model_report
def world_model_report(dataset: dict, params: dict, n_heads: int, n_rows: int, n_tokens: int,
                       max_pairs: int, n_trials: int, seed: int = 0) -> dict:
    """
    Bundle the four world-model metrics into one dict, each rounded to 4 decimals.

    Args:
        dataset: dict with 'tokens', 'mask', 'states', 'G'
        params: GPT parameter dict
        n_heads: number of attention heads
        n_rows: number of sequences for the legality and effective-rank metrics
        n_tokens: continuation length for the compression metric
        max_pairs: maximum prefix pairs for the compression metric
        n_trials: number of episodes for the detour metric
        seed: RNG seed for the detour metric

    Returns:
        dict with keys 'valid_move_rate', 'effective_rank',
        'sequence_compression', 'detour_robustness'.
    """
    G = dataset['G']
    T = dataset['tokens'].shape[1]

    vmr = valid_move_rate(dataset, params, n_heads, n_rows)

    H = eval_hidden_states(dataset, params, n_heads, n_rows)
    erank = effective_rank(H)

    seq_comp = sequence_compression(dataset, params, n_heads, n_tokens, max_pairs)

    detour = detour_robustness(params, n_heads, G, max_steps=T - 3,
                               n_trials=n_trials, detour_prob=0.75, seed=seed)

    return {
        'valid_move_rate': round(vmr, 4),
        'effective_rank': round(erank, 4),
        'sequence_compression': round(seq_comp, 4),
        'detour_robustness': round(detour, 4),
    }

# Step 29 - draft_from_latent
import torch

def draft_from_latent(h_last, dyn: dict, params: dict, max_draft: int) -> tuple:
    """
    Draft tokens by rolling the latent dynamics forward from the last hidden state.

    Args:
        h_last: (d,) final hidden state at the last position of the verified sequence
        dyn: dynamics parameter dict
        params: GPT parameter dict (used for 'wte' and the output head)
        max_draft: number of extra draft tokens to propose

    Returns:
        (next_token, drafts): next_token is the transformer's own prediction (int),
        drafts is a list of max_draft ints proposed by the dynamics model.
    """
    drafts = []

    with torch.no_grad():
        # Free token: the transformer's own prediction from h_last
        next_token = int(torch.argmax(output_head(h_last, params)))

        # Step the latent forward using the token we just chose
        h = latent_transition(h_last, params['wte'][next_token], dyn)

        for _ in range(max_draft):
            tok = int(torch.argmax(output_head(h, params)))
            drafts.append(tok)
            h = latent_transition(h, params['wte'][tok], dyn)

    return next_token, drafts

# Step 30 - verify_draft
import torch

def verify_draft(params: dict, n_heads: int, prefix: list, next_token: int, drafts: list) -> tuple:
    """
    Check a draft with a single transformer pass.

    Args:
        params: GPT parameter dict
        n_heads: number of attention heads
        prefix: verified token ids so far
        next_token: the transformer's own (free) next token
        drafts: tokens proposed by the latent dynamics model

    Returns:
        (n_accepted, correction): number of drafts accepted (int), and the
        transformer's own token right after the last accepted one (int).
    """
    P = len(prefix)
    seq = list(prefix) + [next_token] + list(drafts)

    with torch.no_grad():
        x = torch.tensor([seq], dtype=torch.long)
        h = gpt_hidden_states(x, params, n_heads)
        preds = torch.argmax(output_head(h[0], params), dim=-1)  # (len(seq),)

    n_accepted = 0
    for j, d in enumerate(drafts):
        if int(preds[P + j]) == d:
            n_accepted += 1
        else:
            break

    correction = int(preds[P + n_accepted])

    return n_accepted, correction

# Step 31 - self_speculative_generate
import torch

def self_speculative_generate(params: dict, dyn: dict, n_heads: int, prefix: list,
                              n_tokens: int, max_draft: int) -> dict:
    """
    Generate n_tokens tokens with variable-length self-speculative decoding.

    Args:
        params: GPT parameter dict
        dyn: dynamics parameter dict
        n_heads: number of attention heads
        prefix: starting token ids
        n_tokens: number of tokens to generate
        max_draft: maximum number of drafted tokens per cycle

    Returns:
        dict with 'tokens' (generated tokens, length n_tokens),
        'cycles' (number of loop iterations), and
        'accepted' (n_accepted for each cycle).
    """
    seq = list(prefix)
    P0 = len(prefix)
    max_len = params['wpe'].shape[0]

    cycles = 0
    accepted = []

    while len(seq) - P0 < n_tokens:
        with torch.no_grad():
            x = torch.tensor([seq], dtype=torch.long)
            h_last = gpt_hidden_states(x, params, n_heads)[0, -1]

        k = max(0, min(max_draft, max_len - len(seq) - 2))

        next_token, drafts = draft_from_latent(h_last, dyn, params, k)
        n_accepted, correction = verify_draft(params, n_heads, seq, next_token, drafts)

        seq.extend([next_token] + drafts[:n_accepted] + [correction])

        accepted.append(n_accepted)
        cycles += 1

    return {
        'tokens': seq[P0:][:n_tokens],
        'cycles': cycles,
        'accepted': accepted,
    }

# Step 32 - speculative_stats
def speculative_stats(result: dict, n_tokens: int) -> dict:
    """
    Summarize a self-speculative run.

    Args:
        result: dict from self_speculative_generate with 'tokens', 'cycles', 'accepted'
        n_tokens: number of tokens that were requested

    Returns:
        dict with 'cycles' (int), 'mean_accepted' (float, 4 decimals),
        'speedup' (float, 4 decimals).
    """
    cycles = result['cycles']
    accepted = result['accepted']

    mean_accepted = sum(accepted) / len(accepted) if len(accepted) > 0 else 0.0
    speedup = n_tokens / cycles if cycles > 0 else 0.0

    return {
        'cycles': cycles,
        'mean_accepted': round(mean_accepted, 4),
        'speedup': round(speedup, 4),
    }

