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

# Step 8 - init_gpt_params (not yet solved)
# TODO: implement

# Step 9 - attention_block (not yet solved)
# TODO: implement

# Step 10 - mlp_block (not yet solved)
# TODO: implement

# Step 11 - gpt_hidden_states (not yet solved)
# TODO: implement

# Step 12 - output_head (not yet solved)
# TODO: implement

# Step 13 - next_token_loss (not yet solved)
# TODO: implement

# Step 14 - init_dynamics_params (not yet solved)
# TODO: implement

# Step 15 - latent_transition (not yet solved)
# TODO: implement

# Step 16 - rollout_latents (not yet solved)
# TODO: implement

# Step 17 - next_hidden_loss (not yet solved)
# TODO: implement

# Step 18 - kl_alignment_loss (not yet solved)
# TODO: implement

# Step 19 - nextlat_loss (not yet solved)
# TODO: implement

# Step 20 - train_step (not yet solved)
# TODO: implement

# Step 21 - train_model (not yet solved)
# TODO: implement

# Step 22 - greedy_decode (not yet solved)
# TODO: implement

# Step 23 - effective_rank (not yet solved)
# TODO: implement

# Step 24 - eval_hidden_states (not yet solved)
# TODO: implement

# Step 25 - valid_move_rate (not yet solved)
# TODO: implement

# Step 26 - sequence_compression (not yet solved)
# TODO: implement

# Step 27 - detour_robustness (not yet solved)
# TODO: implement

# Step 28 - world_model_report (not yet solved)
# TODO: implement

# Step 29 - draft_from_latent (not yet solved)
# TODO: implement

# Step 30 - verify_draft (not yet solved)
# TODO: implement

# Step 31 - self_speculative_generate (not yet solved)
# TODO: implement

# Step 32 - speculative_stats (not yet solved)
# TODO: implement

