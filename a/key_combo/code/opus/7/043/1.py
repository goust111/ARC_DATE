from __future__ import annotations

import random

import numpy as np


BG = 0
RING_GRAY = 5
RED_SEED = 2
GREEN_SEED = 3
PALETTE_POOL = (1, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400


def _extract_palette(grid: np.ndarray) -> np.ndarray:
    row = np.asarray(grid[0], dtype=int)
    end = 0
    while end < row.size and int(row[end]) != BG:
        end += 1
    assert end >= 1, "palette row must start with a non-background run"
    assert np.all(row[end:] == BG), "palette row must be left-aligned"
    return row[:end].astype(int)


def _seed_positions(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    rows, cols = np.where(grid[2:] == int(color))
    return [(int(r + 2), int(c)) for r, c in zip(rows, cols)]


def _paint_rays(grid: np.ndarray, red_cols: list[int], green_rows: list[int], seed_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for c in red_cols:
        out[1:, int(c)] = RING_GRAY
    for r in green_rows:
        out[int(r), :] = RING_GRAY
    out[0] = grid[0]
    out[seed_mask] = grid[seed_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand emergent red-green ray crossings into palette-colored concentric square rings.

    Input structure:
    - The background is black, with a left-aligned palette run on row 0 and empty cells after it.
    - From row 2 downward, isolated red seed pixels and green seed pixels are scattered.
    - Red seeds have unique columns, green seeds have unique rows, and their row/column cartesian product defines implicit crossings.

    Rule:
    - Each red seed emits a full vertical gray ray from row 1 to the bottom; each green seed emits a full horizontal gray ray across its row.
    - Seed pixels keep their original colors instead of turning gray.
    - Every crossing between a green-seed row and a red-seed column becomes the center of concentric Chebyshev rings, where distance d uses palette[d].
    - Ring pixels overwrite gray rays but never overwrite seed pixels; when multiple centers compete, prefer the one with smaller Euclidean distance, then smaller center row, then smaller center column.

    Key insight:
    - The important anchors are not marked directly in the input: they emerge from the cartesian product of red columns and green rows, and those hidden crossings drive the ring expansion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    palette = _extract_palette(g)
    radius = int(len(palette) - 1)

    red_seeds = _seed_positions(g, RED_SEED)
    green_seeds = _seed_positions(g, GREEN_SEED)
    red_cols = sorted({c for _r, c in red_seeds})
    green_rows = sorted({r for r, _c in green_seeds})
    assert len(red_cols) == len(red_seeds) and len(green_rows) == len(green_seeds)

    seed_mask = np.zeros_like(g, dtype=bool)
    seed_mask[2:] = (g[2:] == RED_SEED) | (g[2:] == GREEN_SEED)
    out = _paint_rays(g, red_cols, green_rows, seed_mask)

    centers = [(int(r), int(c)) for r in green_rows for c in red_cols]
    h, w = map(int, g.shape)
    for r in range(h):
        for c in range(w):
            if seed_mask[r, c]:
                continue
            best_key: tuple[int, int, int] | None = None
            best_color: int | None = None
            for cr, cc in centers:
                cheb = max(abs(r - cr), abs(c - cc))
                if cheb > radius:
                    continue
                key = ((r - cr) * (r - cr) + (c - cc) * (c - cc), int(cr), int(cc))
                if best_key is None or key < best_key:
                    best_key = key
                    best_color = int(palette[cheb])
            if best_color is not None:
                out[r, c] = best_color
    return out


def _sample_positions(
    lo: int,
    hi: int,
    count: int,
    min_gap: int,
    *,
    fixed: tuple[int, ...] = (),
    require_even_gap: bool = False,
) -> list[int] | None:
    values = list(range(int(lo), int(hi) + 1))
    fixed_vals = sorted(set(int(v) for v in fixed))
    if count < len(fixed_vals):
        return None
    if not values:
        return None

    for _ in range(500):
        remaining = [v for v in values if v not in fixed_vals]
        if len(remaining) < count - len(fixed_vals):
            return None
        chosen = sorted(fixed_vals + random.sample(remaining, count - len(fixed_vals)))
        if any(chosen[i + 1] - chosen[i] < int(min_gap) for i in range(len(chosen) - 1)):
            continue
        if require_even_gap and count >= 2:
            if not any((chosen[i + 1] - chosen[i]) % 2 == 0 for i in range(len(chosen) - 1)):
                continue
        return chosen
    return None


def _far_enough(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in placed)


def _place_seeds(
    red_cols: list[int],
    green_rows: list[int],
    h: int,
    w: int,
    *,
    red_seed_on_cross: bool,
    green_seed_on_cross: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    placed: list[tuple[int, int]] = []
    red_seeds: list[tuple[int, int]] = []
    green_seeds: list[tuple[int, int]] = []

    forced_red_index = random.randrange(len(red_cols)) if red_seed_on_cross else None
    forced_green_index = random.randrange(len(green_rows)) if green_seed_on_cross else None

    for idx, c in enumerate(red_cols):
        candidates = list(range(2, h))
        random.shuffle(candidates)
        if forced_red_index == idx:
            preferred = green_rows[:]
            random.shuffle(preferred)
            candidates = preferred + [r for r in candidates if r not in green_rows]
        placed_here = False
        for r in candidates:
            pos = (int(r), int(c))
            if pos in placed or not _far_enough(pos, placed):
                continue
            red_seeds.append(pos)
            placed.append(pos)
            placed_here = True
            break
        if not placed_here:
            return None

    for idx, r in enumerate(green_rows):
        candidates = list(range(w))
        random.shuffle(candidates)
        if forced_green_index == idx:
            preferred = red_cols[:]
            random.shuffle(preferred)
            candidates = preferred + [c for c in candidates if c not in red_cols]
        placed_here = False
        for c in candidates:
            pos = (int(r), int(c))
            if pos in placed or not _far_enough(pos, placed):
                continue
            green_seeds.append(pos)
            placed.append(pos)
            placed_here = True
            break
        if not placed_here:
            return None

    if red_seed_on_cross and not any(r in set(green_rows) for r, _c in red_seeds):
        return None
    if green_seed_on_cross and not any(c in set(red_cols) for _r, c in green_seeds):
        return None
    return red_seeds, green_seeds


def _coverage_stats(shape: tuple[int, int], centers: list[tuple[int, int]], radius: int) -> tuple[int, int]:
    h, w = map(int, shape)
    overlap = 0
    tie = 0
    for r in range(h):
        for c in range(w):
            d2s: list[int] = []
            for cr, cc in centers:
                if max(abs(r - cr), abs(c - cc)) <= int(radius):
                    d2s.append((r - cr) * (r - cr) + (c - cc) * (c - cc))
            if len(d2s) >= 2:
                overlap += 1
                d2s.sort()
                if d2s[0] == d2s[1]:
                    tie += 1
    return overlap, tie


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(13, 19)
    w = random.randint(13, 19)
    palette_len = random.choice((3, 4))
    radius = palette_len - 1
    palette = random.sample(PALETTE_POOL, palette_len)

    n_red = 2 if w < 16 else random.choice((2, 3))
    n_green = 2 if h < 16 else random.choice((2, 3))

    top_overlap = random.random() < 0.4
    side_clip = random.random() < 0.5
    red_seed_on_cross = random.random() < 0.6
    green_seed_on_cross = random.random() < 0.6

    forced_green_rows: tuple[int, ...] = ()
    if top_overlap:
        forced_green_rows = (random.randint(2, radius),)
        row_lo = 2
    else:
        row_lo = max(2, radius + 1)

    if row_lo > h - 1:
        return None

    forced_red_cols: tuple[int, ...] = ()
    if side_clip:
        edge_choices = list(range(0, radius + 1)) + list(range(w - radius - 1, w))
        forced_red_cols = (random.choice(edge_choices),)

    red_cols = _sample_positions(0, w - 1, n_red, 2, fixed=forced_red_cols, require_even_gap=True)
    green_rows = _sample_positions(row_lo, h - 1, n_green, 2, fixed=forced_green_rows, require_even_gap=True)
    if red_cols is None or green_rows is None:
        return None

    seed_pairs = _place_seeds(
        red_cols,
        green_rows,
        h,
        w,
        red_seed_on_cross=red_seed_on_cross,
        green_seed_on_cross=green_seed_on_cross,
    )
    if seed_pairs is None:
        return None
    red_seeds, green_seeds = seed_pairs

    inp = np.zeros((h, w), dtype=int)
    inp[0, :palette_len] = np.array(palette, dtype=int)
    for r, c in red_seeds:
        inp[r, c] = RED_SEED
    for r, c in green_seeds:
        inp[r, c] = GREEN_SEED

    centers = [(int(r), int(c)) for r in green_rows for c in red_cols]
    overlap_cells, tie_cells = _coverage_stats(inp.shape, centers, radius)
    if overlap_cells < 10 or tie_cells < 2:
        return None

    out = main(inp)
    gray_visible = int(np.count_nonzero(out == RING_GRAY))
    non_bg_ratio = float(np.count_nonzero(out != BG)) / float(out.size)
    if gray_visible < max(6, n_red + n_green):
        return None
    if not (0.18 <= non_bg_ratio <= 0.78):
        return None
    if not all(np.any(out == color) for color in palette):
        return None
    if np.array_equal(inp, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
