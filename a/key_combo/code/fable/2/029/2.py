import math

import numpy as np


RNG = np.random.default_rng()
BACKGROUND = 0
FOREGROUND_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
N_CHOICES = (3, 4, 5)
GENERATION_ATTEMPTS = 500


def _split_blocks(binary: np.ndarray, n: int) -> np.ndarray:
    """Return blocks as (block_r, block_c, inner_r, inner_c)."""
    return binary.reshape(n, n, n, n).transpose(0, 2, 1, 3)


def _majority_seed(blocks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    stack = np.array([blocks[r, c] for r, c in coords], dtype=np.uint8)
    votes = stack.sum(axis=0)
    return (votes * 2 > len(stack)).astype(np.uint8)


def _objective_vec(seed_vec: np.ndarray, pair_matrix: np.ndarray, total_ones: int) -> int:
    k = int(seed_vec.sum())
    quadratic = int(seed_vec @ pair_matrix @ seed_vec)
    return int(total_ones + k * k - 2 * quadratic)


def _hill_climb(seed: np.ndarray, pair_matrix: np.ndarray, total_ones: int) -> tuple[np.ndarray, int]:
    """Greedy single-bit descent for Hamming distance to the clean self-expansion."""
    seed_vec = seed.astype(np.int32).ravel().copy()
    k = int(seed_vec.sum())
    row_sums = pair_matrix @ seed_vec
    col_sums = pair_matrix.T @ seed_vec
    quadratic = int(seed_vec @ row_sums)
    current = int(total_ones + k * k - 2 * quadratic)
    size = len(seed_vec)

    while True:
        best_delta = 0
        best_index = -1
        for idx in range(size):
            delta = 1 - 2 * int(seed_vec[idx])
            score_delta = 2 * delta * (k - int(row_sums[idx]) - int(col_sums[idx])) + 1 - 2 * int(pair_matrix[idx, idx])
            if score_delta < best_delta:
                best_delta = int(score_delta)
                best_index = idx

        if best_index < 0:
            return seed_vec.reshape(seed.shape).astype(np.uint8), current

        delta = 1 - 2 * int(seed_vec[best_index])
        seed_vec[best_index] += delta
        k += delta
        quadratic += delta * (int(row_sums[best_index]) + int(col_sums[best_index])) + int(pair_matrix[best_index, best_index])
        row_sums = row_sums + delta * pair_matrix[:, best_index]
        col_sums = col_sums + delta * pair_matrix[best_index, :]
        current += best_delta


def _candidate_masks(counts: np.ndarray) -> list[np.ndarray]:
    n = counts.shape[0]
    total = n * n
    order = np.argsort(-counts.ravel(), kind="stable")
    seen: set[tuple[int, ...]] = set()
    masks: list[np.ndarray] = []

    def add(mask: np.ndarray) -> None:
        flat = tuple(int(x) for x in mask.ravel())
        if 0 < sum(flat) < total and flat not in seen:
            seen.add(flat)
            masks.append(mask.astype(np.uint8, copy=True))

    for threshold in sorted(set(map(int, counts.ravel())), reverse=True):
        add(counts >= threshold)
    add(counts > 0)

    tail_size = 2
    head_size = 2
    for k in range(1, total):
        base = np.zeros(total, dtype=np.uint8)
        base[order[:k]] = 1
        add(base.reshape(n, n))

        tail = order[max(0, k - tail_size) : k]
        head = order[k : min(total, k + head_size)]
        for src in tail:
            for dst in head:
                swapped = base.copy()
                swapped[src] = 0
                swapped[dst] = 1
                add(swapped.reshape(n, n))

    return masks


def _recover_seed_binary(binary: np.ndarray) -> np.ndarray:
    side = int(binary.shape[0])
    n = int(math.isqrt(side))
    if binary.ndim != 2 or binary.shape[0] != binary.shape[1] or n * n != side:
        raise ValueError("expected a square n^2 x n^2 grid")

    blocks = _split_blocks(binary.astype(np.uint8), n)
    counts = blocks.sum(axis=(2, 3)).astype(int)
    pair_matrix = blocks.reshape(n * n, n * n).astype(np.int32)
    total_ones = int(binary.sum())

    seed_candidates: list[np.ndarray] = []
    seen_seeds: set[tuple[int, ...]] = set()
    for mask in _candidate_masks(counts):
        selected = mask.copy()
        for _ in range(4):
            if not selected.any():
                break
            voted = _majority_seed(blocks, selected.astype(bool))
            if np.array_equal(voted, selected):
                selected = voted
                break
            selected = voted
        if not selected.any():
            continue
        seed = _majority_seed(blocks, selected.astype(bool))
        key = tuple(int(x) for x in seed.ravel())
        if 0 < sum(key) < n * n and key not in seen_seeds:
            seen_seeds.add(key)
            seed_candidates.append(seed)

    if not seed_candidates:
        fallback = (counts > 0).astype(np.uint8)
        seed_candidates.append(fallback)

    best_seed = None
    best_error = None
    best_key = None
    for candidate in seed_candidates:
        refined, error = _hill_climb(candidate, pair_matrix, total_ones)
        key = tuple(int(x) for x in refined.ravel())
        if best_error is None or error < best_error or (error == best_error and key < best_key):
            best_seed = refined
            best_error = error
            best_key = key

    assert best_seed is not None
    return best_seed.astype(np.uint8)


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the unique n x n seed sprite from a noisy self-similar n^2 x n^2 expansion.

    Input structure:
    - The grid is square with side n^2, black background, and exactly one foreground color.
    - Splitting the grid into n x n blocks yields n^2 subgrids, each of shape n x n.
    - Before sparse deletions and additions, every occupied block was a full copy of the same seed, while every empty block was all background.

    Rule:
    - Infer n from the grid side and use block foreground counts to propose which blocks carry seed copies.
    - For each proposal, majority-vote the corresponding block pixels to recover a seed candidate, then enforce the self-similarity constraint that the block occupancy pattern must be that same seed.
    - Choose the self-consistent seed whose clean self-expansion best matches the whole noisy input, and output that n x n seed in the original foreground color.

    Key insight:
    - The occupancy map of the block array and the contents of every occupied block are the same picture, so the true seed is the unique pattern that simultaneously explains both signals while ignoring sparse noise.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("expected a square 2D grid")

    colors = [int(c) for c in np.unique(g) if int(c) != BACKGROUND]
    if len(colors) != 1:
        raise ValueError("expected exactly one foreground color")

    fg = colors[0]
    seed = _recover_seed_binary((g == fg).astype(np.uint8))
    return (seed * fg).astype(int)


def _seed_area_bounds(n: int) -> tuple[int, int]:
    total = n * n
    lo = max(4, math.ceil(total * 0.35))
    hi = min(total - 4, math.floor(total * 0.65))
    return lo, hi


def _random_seed(n: int) -> np.ndarray:
    lo, hi = _seed_area_bounds(n)
    total = n * n
    for _ in range(400):
        area = int(RNG.integers(lo, hi + 1))
        seed = np.zeros((n, n), dtype=np.uint8)
        chosen = RNG.choice(total, size=area, replace=False)
        seed.ravel()[chosen] = 1

        rows = seed.sum(axis=1)
        cols = seed.sum(axis=0)
        empty_rows = n - rows
        empty_cols = n - cols
        if np.count_nonzero(rows) < 2 or np.count_nonzero(cols) < 2:
            continue
        if np.count_nonzero(empty_rows) < 2 or np.count_nonzero(empty_cols) < 2:
            continue
        if len({tuple(row) for row in seed}) < 2:
            continue
        if len({tuple(col) for col in seed.T}) < 2:
            continue
        if np.array_equal(seed, np.rot90(seed)):
            continue
        return seed
    raise RuntimeError("failed to generate a diverse seed")


def _expand_seed(seed: np.ndarray) -> np.ndarray:
    return np.kron(seed, seed).astype(np.uint8)


def _block_to_global(block_rc: tuple[int, int], cell_rc: tuple[int, int], n: int) -> tuple[int, int]:
    br, bc = block_rc
    r, c = cell_rc
    return br * n + r, bc * n + c


def _round_robin_cells(cells: list[tuple[int, int]], count: int, cap: int) -> list[tuple[int, int]]:
    if not cells or count <= 0 or cap <= 0:
        return []
    shuffled = list(cells)
    RNG.shuffle(shuffled)
    used = {cell: 0 for cell in shuffled}
    picked: list[tuple[int, int]] = []
    cursor = 0
    while len(picked) < count:
        cell = shuffled[cursor % len(shuffled)]
        cursor += 1
        if used[cell] >= cap:
            if all(used[c] >= cap for c in shuffled):
                break
            continue
        used[cell] += 1
        picked.append(cell)
    return picked


def _apply_noise(seed: np.ndarray, mode: str) -> np.ndarray:
    n = seed.shape[0]
    area = int(seed.sum())
    noisy = _expand_seed(seed)
    occupied = [tuple(map(int, rc)) for rc in np.argwhere(seed == 1)]
    empty = [tuple(map(int, rc)) for rc in np.argwhere(seed == 0)]
    true_cells = occupied.copy()
    false_cells = empty.copy()

    remove_cap = max(1, (area - 1) // 2)
    add_cap = max(1, (area - 1) // 2)
    remove_counts = {cell: 0 for cell in true_cells}
    add_counts = {cell: 0 for cell in false_cells}

    removal_plan = true_cells.copy()
    RNG.shuffle(removal_plan)
    for block_rc, removed_cell in zip(occupied, removal_plan):
        gr, gc = _block_to_global(block_rc, removed_cell, n)
        noisy[gr, gc] = 0
        remove_counts[removed_cell] += 1

    if mode == "ambiguous" and false_cells:
        add_blocks = occupied.copy()
        RNG.shuffle(add_blocks)
        add_total = int(RNG.integers(max(1, area // 3), max(2, area // 2) + 1))
        add_total = min(add_total, len(add_blocks), len(false_cells) * add_cap)
        add_plan = _round_robin_cells(false_cells, add_total, add_cap)
        for block_rc, added_cell in zip(add_blocks[: len(add_plan)], add_plan):
            gr, gc = _block_to_global(block_rc, added_cell, n)
            noisy[gr, gc] = 1
            add_counts[added_cell] += 1

    if mode == "ambiguous":
        special_blocks = occupied.copy()
        RNG.shuffle(special_blocks)
        extra_done = False
        for block_rc in special_blocks:
            candidates = [cell for cell in true_cells if remove_counts[cell] < remove_cap]
            if not candidates:
                break
            extra_cell = candidates[int(RNG.integers(len(candidates)))]
            gr, gc = _block_to_global(block_rc, extra_cell, n)
            if noisy[gr, gc] == 1:
                noisy[gr, gc] = 0
                remove_counts[extra_cell] += 1
                extra_done = True
                break
        if not extra_done and occupied:
            block_rc = occupied[0]
            removed_cell = removal_plan[0]
            gr, gc = _block_to_global(block_rc, removed_cell, n)
            noisy[gr, gc] = 0

    if empty:
        noisy_empty = empty.copy()
        RNG.shuffle(noisy_empty)
        chosen_blocks = noisy_empty[: int(RNG.integers(1, min(len(noisy_empty), max(2, n - 1)) + 1))]
        if mode == "ambiguous" and noisy_empty:
            chosen_blocks = [noisy_empty[0], *chosen_blocks]

        for block_index, block_rc in enumerate(chosen_blocks):
            if mode == "ambiguous" and block_index == 0:
                count = min(len(false_cells), max(2, area - 1))
            else:
                count = 1 if mode == "clear" else int(RNG.integers(1, min(len(false_cells), max(2, n // 2 + 1)) + 1))
            chosen_cells = RNG.choice(len(false_cells), size=count, replace=False)
            for idx in np.atleast_1d(chosen_cells):
                cell_rc = false_cells[int(idx)]
                gr, gc = _block_to_global(block_rc, cell_rc, n)
                noisy[gr, gc] = 1

    return noisy.astype(np.uint8)


def _blocks_from_binary(binary: np.ndarray) -> np.ndarray:
    n = int(math.isqrt(binary.shape[0]))
    return _split_blocks(binary, n)


def _topk_mask(counts: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(counts.size, dtype=np.uint8)
    flat_counts = counts.astype(int, copy=False).ravel()
    order = np.argsort(-flat_counts, kind="stable")
    mask[order[:k]] = 1
    return mask.reshape(counts.shape)


def _validate_case(seed: np.ndarray, noisy_binary: np.ndarray, mode: str) -> bool:
    n = seed.shape[0]
    area = int(seed.sum())
    if area == 0 or area == n * n:
        return False

    blocks = _blocks_from_binary(noisy_binary)
    counts = blocks.sum(axis=(2, 3)).astype(int)
    occupied = np.argwhere(seed == 1)
    empty = np.argwhere(seed == 0)

    if len(empty) == 0:
        return False
    if not np.any(counts[seed == 0] > 0):
        return False

    for br, bc in occupied:
        if np.array_equal(blocks[br, bc], seed):
            return False

    occupied_stack = np.array([blocks[br, bc] for br, bc in occupied], dtype=np.uint8)
    if not np.array_equal((occupied_stack.sum(axis=0) * 2 > area).astype(np.uint8), seed):
        return False

    densest_br, densest_bc = np.unravel_index(int(np.argmax(counts)), counts.shape)
    if np.array_equal(blocks[densest_br, densest_bc], seed):
        return False

    topk = _topk_mask(counts, area)
    if mode == "ambiguous" and np.array_equal(topk, seed):
        return False
    if mode == "clear" and not np.array_equal(topk, seed):
        return False

    recovered = _recover_seed_binary(noisy_binary)
    if not np.array_equal(recovered, seed):
        return False

    pair_matrix = blocks.reshape(n * n, n * n).astype(np.int32)
    truth = _objective_vec(seed.ravel().astype(np.int32), pair_matrix, int(noisy_binary.sum()))
    shortcut = _objective_vec(topk.ravel().astype(np.int32), pair_matrix, int(noisy_binary.sum()))
    if mode == "ambiguous" and shortcut <= truth:
        return False

    return not np.array_equal(noisy_binary, _expand_seed(seed))


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    n = int(RNG.choice(N_CHOICES))
    seed = _random_seed(n)
    mode = "ambiguous" if RNG.random() < 0.55 else "clear"
    noisy_binary = _apply_noise(seed, mode)
    if not _validate_case(seed, noisy_binary, mode):
        raise RuntimeError("generated case failed semantic validation")
    color = int(RNG.choice(FOREGROUND_COLORS))
    return noisy_binary.astype(int) * color, seed.astype(int) * color


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            grid, _seed = _generate_case()
        except RuntimeError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid self-similar noisy expansion")
