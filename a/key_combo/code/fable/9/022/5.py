from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _apply_d4(arr: np.ndarray, op: int) -> np.ndarray:
    op = int(op)
    if op == 0:
        return arr.copy()
    if op == 1:
        return np.rot90(arr, 1)
    if op == 2:
        return np.rot90(arr, 2)
    if op == 3:
        return np.rot90(arr, 3)
    if op == 4:
        return np.fliplr(arr)
    if op == 5:
        return np.flipud(arr)
    if op == 6:
        return arr.T.copy()
    if op == 7:
        return np.fliplr(np.rot90(arr, 1))
    raise ValueError(f"unknown D4 op: {op}")


def _build_inverse_ops() -> list[int]:
    probe = np.arange(9, dtype=int).reshape(3, 3)
    inv = [-1] * 8
    for op in range(8):
        for cand in range(8):
            if np.array_equal(_apply_d4(_apply_d4(probe, op), cand), probe):
                inv[op] = cand
                break
        assert inv[op] != -1
    return inv


INV_OPS = _build_inverse_ops()


def _mode_color(values: np.ndarray) -> int:
    counts = np.bincount(values.astype(int), minlength=10)
    best = int(counts.max())
    winners = np.flatnonzero(counts == best)
    assert len(winners) == 1, "expected a unique majority color"
    return int(winners[0])


def _majority_grid(blocks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(blocks, axis=0)
    h, w = stack.shape[1:]
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = _mode_color(stack[:, r, c])
    return out


def _regions_between(size: int, lines: list[int]) -> list[tuple[int, int]]:
    assert lines
    regions: list[tuple[int, int]] = []
    start = 0
    for line in sorted(lines):
        assert 0 < line < size - 1
        regions.append((start, line))
        start = line + 1
    regions.append((start, size))
    return regions


def _parse_slots(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[np.ndarray]]:
    h, w = grid.shape
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))

    assert len(candidates) == 1, "expected a unique separator color"
    separator, row_lines, col_lines = candidates[0]
    row_regions = _regions_between(h, row_lines)
    col_regions = _regions_between(w, col_lines)

    cell_heights = {r1 - r0 for r0, r1 in row_regions}
    cell_widths = {c1 - c0 for c0, c1 in col_regions}
    assert len(cell_heights) == 1 and len(cell_widths) == 1
    cell_size = next(iter(cell_heights))
    assert cell_size == next(iter(cell_widths)), "expected square cells"

    cells = [grid[r0:r1, c0:c1].copy() for r0, r1 in row_regions for c0, c1 in col_regions]
    return int(separator), row_regions, col_regions, cells


def _best_match(proto: np.ndarray, variants: list[np.ndarray]) -> tuple[int, bool, int]:
    scores = [int(np.count_nonzero(v == proto)) for v in variants]
    best = max(scores)
    winners = [idx for idx, score in enumerate(scores) if score == best]
    return winners[0], len(winners) == 1, best


def _infer_orientations(cells: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    aligned_variants = [[_apply_d4(cell, INV_OPS[op]) for op in range(8)] for cell in cells]

    best_key = None
    best_proto = None
    best_ops = None
    for anchor_idx in range(len(cells)):
        for anchor_op in range(8):
            proto = aligned_variants[anchor_idx][anchor_op]
            for _ in range(3):
                ops = []
                for variants in aligned_variants:
                    op, _unique, _score = _best_match(proto, variants)
                    ops.append(op)
                aligned = [aligned_variants[i][ops[i]] for i in range(len(cells))]
                proto2 = _majority_grid(aligned)
                if np.array_equal(proto2, proto):
                    break
                proto = proto2

            ops = []
            unique_hits = 0
            total = 0
            for variants in aligned_variants:
                op, unique, score = _best_match(proto, variants)
                ops.append(op)
                unique_hits += int(unique)
                total += score
            aligned = [aligned_variants[i][ops[i]] for i in range(len(cells))]
            proto = _majority_grid(aligned)
            total = sum(int(np.count_nonzero(block == proto)) for block in aligned)
            key = (total, unique_hits)
            if best_key is None or key > best_key:
                best_key = key
                best_proto = proto.copy()
                best_ops = ops[:]

    assert best_proto is not None and best_ops is not None
    return best_proto, best_ops


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a separator-grid of noisy rotated or mirrored sprite copies by aligning them, voting in a common pose, then restoring each slot's own pose.

    Input structure:
    - One separator color forms full rows and full columns that partition the canvas into equal square cells.
    - Each cell contains the same multi-color non-symmetric sprite under one of the eight D4 orientations.
    - Sparse noise changes a few cell pixels to other colors or to background, but the intended orientation of each cell remains uniquely recoverable.

    Rule:
    - Detect the separator grid and extract every square cell.
    - For each cell, choose the orientation explanation that makes it agree best with the shared sprite after alignment to a common pose.
    - In that common pose, take a per-pixel majority vote across cells to recover the clean sprite.
    - Reapply each cell's inferred orientation to the recovered sprite, write it back into the corresponding slot, and keep the separator lines unchanged.

    Key insight:
    - Raw same-position voting fails because the slots are not in a common pose; the only reliable path is to undo each rotation or reflection first, vote in aligned coordinates, then repaint each slot in its original orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    separator, row_regions, col_regions, cells = _parse_slots(g)
    clean_sprite, ops = _infer_orientations(cells)

    out = g.copy()
    idx = 0
    for r0, r1 in row_regions:
        for c0, c1 in col_regions:
            out[r0:r1, c0:c1] = _apply_d4(clean_sprite, ops[idx])
            idx += 1

    for r in range(g.shape[0]):
        if np.all(g[r, :] == separator):
            out[r, :] = separator
    for c in range(g.shape[1]):
        if np.all(g[:, c] == separator):
            out[:, c] = separator
    return out


GEN_LAYOUTS = [(2, 2), (2, 3), (3, 2), (3, 3)]
GEN_SIZE_RANGE = (5, 7)
GEN_MAX_ATTEMPTS = 400


def _grow_color_regions(size: int, colors: list[int], rng: random.Random) -> np.ndarray:
    k = len(colors)
    owner = -np.ones((size, size), dtype=int)
    all_cells = [(r, c) for r in range(size) for c in range(size)]
    seeds = rng.sample(all_cells, k)
    frontiers = [set() for _ in range(k)]
    counts = [0] * k

    for idx, (r, c) in enumerate(seeds):
        owner[r, c] = idx
        counts[idx] = 1

    def refresh(idx: int) -> None:
        cells = np.argwhere(owner == idx)
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < size and 0 <= cc < size and owner[rr, cc] == -1:
                    frontier.add((rr, cc))
        frontiers[idx] = frontier

    for idx in range(k):
        refresh(idx)

    while np.any(owner == -1):
        choices = [idx for idx in range(k) if frontiers[idx]]
        weights = [1.0 / counts[idx] for idx in choices]
        total = sum(weights)
        pick = rng.random() * total
        acc = 0.0
        chosen = choices[0]
        for idx, weight in zip(choices, weights):
            acc += weight
            if pick <= acc:
                chosen = idx
                break
        cell = rng.choice(tuple(frontiers[chosen]))
        owner[cell] = chosen
        counts[chosen] += 1
        refresh(chosen)
        for idx in range(k):
            if cell in frontiers[idx]:
                refresh(idx)

    sprite = np.zeros((size, size), dtype=int)
    for idx, color in enumerate(colors):
        sprite[owner == idx] = int(color)
    return sprite


def _gen_sprite(size: int, palette: list[int], rng: random.Random) -> np.ndarray | None:
    fg_count = rng.randint(3, 4)
    fg_colors = rng.sample(palette, fg_count)
    sprite = _grow_color_regions(size, fg_colors, rng)

    color_counts = Counter(map(int, sprite.ravel()))
    target_bg = rng.randint(max(2, size), max(size + 1, (size * size) // 3))
    cells = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(cells)
    bg_added = 0
    for r, c in cells:
        color = int(sprite[r, c])
        if color_counts[color] <= 2:
            continue
        sprite[r, c] = 0
        color_counts[color] -= 1
        bg_added += 1
        if bg_added >= target_bg:
            break

    if bg_added < 2:
        return None
    if np.count_nonzero(sprite) < (size * size) // 2:
        return None
    if len({int(v) for v in sprite.ravel()} - {0}) < 3:
        return None

    variants = [_apply_d4(sprite, op) for op in range(8)]
    if len({v.tobytes() for v in variants}) != 8:
        return None

    min_dist = size + 2
    for i in range(8):
        for j in range(i + 1, 8):
            dist = int(np.count_nonzero(variants[i] != variants[j]))
            if dist < min_dist:
                return None
    return sprite


def _choose_orientations(n_slots: int, rng: random.Random) -> list[int]:
    require_reflection = rng.random() < 0.65
    require_quarter_turn = rng.random() < 0.65
    while True:
        distinct = rng.randint(2, min(6, n_slots, 8))
        chosen = rng.sample(range(8), distinct)
        ops = [rng.choice(chosen) for _ in range(n_slots)]
        if len(set(ops)) < 2:
            continue
        if require_reflection and not any(op >= 4 for op in ops):
            continue
        if require_quarter_turn and not any(op in {1, 3, 6, 7} for op in ops):
            continue
        return ops


def _corrupt_canonical(
    sprite: np.ndarray,
    n_slots: int,
    rng: random.Random,
) -> list[np.ndarray]:
    size = sprite.shape[0]
    fg_colors = sorted({int(v) for v in sprite.ravel()} - {0})
    cap = max(1, (n_slots - 1) // 2)
    used_per_pos = np.zeros((size, size), dtype=int)
    max_noise = 2 if n_slots <= 4 else 3

    noisy = []
    for _ in range(n_slots):
        block = sprite.copy()
        candidates = [(r, c) for r in range(size) for c in range(size) if used_per_pos[r, c] < cap]
        rng.shuffle(candidates)
        n_noise = rng.randint(1, min(max_noise, len(candidates)))
        for r, c in candidates[:n_noise]:
            original = int(block[r, c])
            options = [0, *fg_colors]
            options = [color for color in options if color != original]
            block[r, c] = rng.choice(options)
            used_per_pos[r, c] += 1
        noisy.append(block)
    return noisy


def _render_grid(
    canonical_noisy: list[np.ndarray],
    sprite: np.ndarray,
    ops: list[int],
    layout: tuple[int, int],
    separator: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = layout
    size = sprite.shape[0]
    h = rows * size + (rows - 1)
    w = cols * size + (cols - 1)
    inp = np.full((h, w), separator, dtype=int)
    out = np.full((h, w), separator, dtype=int)

    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * (size + 1)
            c0 = cc * (size + 1)
            inp[r0 : r0 + size, c0 : c0 + size] = _apply_d4(canonical_noisy[idx], ops[idx])
            out[r0 : r0 + size, c0 : c0 + size] = _apply_d4(sprite, ops[idx])
            idx += 1
    return inp, out


def _semantic_ok(inp: np.ndarray, out: np.ndarray, sprite: np.ndarray, ops: list[int]) -> bool:
    try:
        separator, row_regions, col_regions, cells = _parse_slots(inp)
    except Exception:
        return False

    if not np.array_equal(main(inp), out):
        return False

    aligned = []
    for cell, true_op in zip(cells, ops):
        scores = [int(np.count_nonzero(_apply_d4(cell, INV_OPS[op]) == sprite)) for op in range(8)]
        if scores[true_op] != max(scores):
            return False
        if scores.count(scores[true_op]) != 1:
            return False
        aligned.append(_apply_d4(cell, INV_OPS[true_op]))

    try:
        if not np.array_equal(_majority_grid(aligned), sprite):
            return False
    except AssertionError:
        return False

    try:
        raw_vote = _majority_grid(cells)
    except AssertionError:
        raw_vote = None
    if raw_vote is not None:
        if any(np.array_equal(raw_vote, _apply_d4(sprite, op)) for op in range(8)):
            return False

        tiled_raw = inp.copy()
        for r0, r1 in row_regions:
            for c0, c1 in col_regions:
                tiled_raw[r0:r1, c0:c1] = raw_vote
        if np.array_equal(tiled_raw, out):
            return False

    if len(set(ops)) < 2:
        return False
    if not any(op >= 4 for op in ops):
        return False
    return int(separator) not in {int(v) for v in sprite.ravel()}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout = rng.choice(GEN_LAYOUTS)
    size = rng.randint(*GEN_SIZE_RANGE)
    separator = rng.choice([c for c in range(1, 10)])
    palette = [c for c in range(1, 10) if c != separator]

    sprite = _gen_sprite(size, palette, rng)
    if sprite is None:
        return None

    n_slots = layout[0] * layout[1]
    ops = _choose_orientations(n_slots, rng)
    canonical_noisy = _corrupt_canonical(sprite, n_slots, rng)
    inp, out = _render_grid(canonical_noisy, sprite, ops, layout, separator)

    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if not _semantic_ok(inp, out, sprite, ops):
        return None
    return inp


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid alignment-vote ARC input")
