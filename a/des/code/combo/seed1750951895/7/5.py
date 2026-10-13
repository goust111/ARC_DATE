from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
SEP = 5
PALETTE = (2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    pts = [-1] + sorted(separators) + [length]
    spans = []
    for a, b in zip(pts, pts[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _separator_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not rows or not cols:
            continue
        row_spans = _spans(h, rows)
        col_spans = _spans(w, cols)
        if len(row_spans) < 2 or len(col_spans) < 2:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) == 1 and len(col_sizes) == 1:
            candidates.append((color, row_spans, col_spans))
    assert len(candidates) == 1, "expected one separator color"
    return candidates[0]


def _unique_mode(values: np.ndarray) -> int:
    ranked = Counter(map(int, values.reshape(-1))).most_common()
    assert ranked
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the consensus sprite from repeated grid cells and mark the damaged pixels in blue.
    """
    g = np.asarray(grid, dtype=int)
    sep, row_spans, col_spans = _separator_layout(g)
    _ = sep

    tiles = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tiles.append(g[r0 : r1 + 1, c0 : c1 + 1].copy())

    assert len(tiles) >= 3, "need enough copies for majority voting"
    tile_shape = tiles[0].shape
    assert all(tile.shape == tile_shape for tile in tiles)

    stack = np.stack(tiles, axis=0)
    th, tw = tile_shape
    consensus = np.zeros((th, tw), dtype=int)
    for r in range(th):
        for c in range(tw):
            consensus[r, c] = _unique_mode(stack[:, r, c])

    mismatch_counts = [int(np.sum(tile != consensus)) for tile in tiles]
    damaged = [i for i, count in enumerate(mismatch_counts) if count > 0]
    assert len(damaged) == 1, "expected exactly one damaged copy"

    out = consensus.copy()
    out[tiles[damaged[0]] != consensus] = BLUE
    return out


def _make_mask(height: int, width: int) -> np.ndarray | None:
    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        prev_l = prev_r = None
        ok = True
        for r in range(height):
            seg_w = random.randint(max(2, width // 3), width - 1)
            if r == 0:
                left = random.randint(0, width - seg_w)
            else:
                lo = max(0, prev_l - seg_w + 1)
                hi = min(width - seg_w, prev_r)
                if lo > hi:
                    ok = False
                    break
                left = random.randint(lo, hi)
            right = left + seg_w - 1
            mask[r, left : right + 1] = True
            prev_l, prev_r = left, right
        if not ok:
            continue
        if not np.all(mask.any(axis=0)):
            continue
        if np.any(mask.all(axis=1)) or np.any(mask.all(axis=0)):
            continue
        return mask
    return None


def _colorize(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    h, w = map(int, mask.shape)
    tile = np.full((h, w), BG, dtype=int)
    mode = random.choice(("solid", "stripes", "checker", "bands"))
    axis = random.choice((0, 1))
    for r, c in np.argwhere(mask):
        if mode == "solid":
            value = colors[0]
        elif mode == "stripes":
            idx = (r if axis == 0 else c) % len(colors)
            value = colors[idx]
        elif mode == "checker":
            value = colors[(r + c) % len(colors)]
        else:
            pivot = (h - 1) * c - (w - 1) * r
            idx = 0 if pivot >= 0 else min(1, len(colors) - 1)
            value = colors[idx]
        tile[r, c] = value
    used = set(map(int, tile[mask]))
    if set(colors[: len(used)]) != used and len(colors) > 1 and len(used) < len(colors):
        return _colorize(mask, colors)
    return tile


def _template() -> np.ndarray | None:
    height = random.randint(5, 8)
    width = random.randint(5, 8)
    mask = _make_mask(height, width)
    if mask is None:
        return None
    color_count = random.randint(1, 3)
    colors = random.sample(PALETTE, color_count)
    tile = _colorize(mask, colors)
    if np.any(tile == BLUE) or np.any(tile == SEP):
        return None
    if not np.all((tile != BG).any(axis=0)) or not np.all((tile != BG).any(axis=1)):
        return None
    if np.any((tile != BG).all(axis=0)) or np.any((tile != BG).all(axis=1)):
        return None
    return tile


def _damage_positions(template: np.ndarray) -> tuple[list[tuple[int, int]], list[int]] | None:
    sprite_cells = [tuple(map(int, rc)) for rc in np.argwhere(template != BG)]
    if len(sprite_cells) < 3:
        return None
    count = random.randint(1, max(1, min(5, len(sprite_cells) // 3)))
    picked = random.sample(sprite_cells, count)
    mode = random.choice(("bg", "alt", "mixed"))
    alt_choices = [c for c in PALETTE if c not in set(map(int, np.unique(template)))]
    if mode != "bg" and not alt_choices:
        return None
    repl = []
    for _ in picked:
        if mode == "bg":
            repl.append(BG)
        elif mode == "alt":
            repl.append(random.choice(alt_choices))
        else:
            repl.append(random.choice([BG, random.choice(alt_choices)]))
    return picked, repl


def _build_expected(template: np.ndarray, damage_pts: list[tuple[int, int]]) -> np.ndarray:
    out = template.copy()
    for r, c in damage_pts:
        out[r, c] = BLUE
    return out


def _gen_attempt() -> np.ndarray | None:
    cell = _template()
    if cell is None:
        return None

    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    if n_rows * n_cols < 3:
        return None

    damage = _damage_positions(cell)
    if damage is None:
        return None
    damage_pts, replacements = damage

    h, w = map(int, cell.shape)
    grid_h = n_rows * h + (n_rows - 1)
    grid_w = n_cols * w + (n_cols - 1)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[h:: h + 1, :] = SEP
    grid[:, w:: w + 1] = SEP

    damaged_index = random.randrange(n_rows * n_cols)
    idx = 0
    for rr in range(n_rows):
        for cc in range(n_cols):
            r0 = rr * (h + 1)
            c0 = cc * (w + 1)
            patch = cell.copy()
            if idx == damaged_index:
                for (r, c), value in zip(damage_pts, replacements):
                    patch[r, c] = value
            grid[r0 : r0 + h, c0 : c0 + w] = patch
            idx += 1

    expected = _build_expected(cell, damage_pts)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
