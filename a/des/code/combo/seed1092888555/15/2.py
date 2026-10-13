from __future__ import annotations

from collections import Counter
import random

import numpy as np


SEP = 9
GEN_BG = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 300


def _split_segments(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + separators + [length]
    segments: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        start = a + 1
        end = b
        if start >= end:
            raise AssertionError("invalid separator layout")
        segments.append((start, end))
    return segments


def _find_panel_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    row_colors = [int(grid[r, 0]) for r in range(h) if np.all(grid[r] == grid[r, 0])]
    col_colors = [int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])]
    common = set(row_colors) & set(col_colors)
    if not common:
        raise AssertionError("no separator color found")

    sep = None
    best_score = -1
    for color in common:
        rows = [r for r in range(h) if np.all(grid[r] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        score = len(rows) + len(cols)
        if score > best_score:
            sep = color
            best_score = score
    assert sep is not None

    sep_rows = [r for r in range(h) if np.all(grid[r] == sep)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == sep)]
    row_segments = _split_segments(h, sep_rows)
    col_segments = _split_segments(w, sep_cols)
    panel_hs = {r1 - r0 for r0, r1 in row_segments}
    panel_ws = {c1 - c0 for c0, c1 in col_segments}
    if len(panel_hs) != 1 or len(panel_ws) != 1:
        raise AssertionError("panels must be equal-sized")
    return sep, row_segments, col_segments


def _panel_bars(panel: np.ndarray, bg: int) -> np.ndarray:
    h, w = panel.shape
    counts = Counter(map(int, panel[panel != bg].ravel()))
    if not counts:
        raise AssertionError("each panel must contain at least one foreground pixel")

    first_seen: dict[int, tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if color != bg and color not in first_seen:
                first_seen[color] = (r, c)

    order = sorted(counts, key=lambda color: (counts[color], first_seen[color]))
    out = np.full((h, len(order)), bg, dtype=int)
    for c, color in enumerate(order):
        height = counts[color]
        if height > h:
            raise AssertionError("bar height exceeds panel height")
        out[h - height :, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each separator-bounded panel into a count-sorted bar chart.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    sep, row_segments, col_segments = _find_panel_layout(g)
    non_sep = g[g != sep]
    if non_sep.size == 0:
        raise AssertionError("missing panel contents")
    bg = Counter(map(int, non_sep.ravel())).most_common(1)[0][0]
    panel_outputs: list[np.ndarray] = []

    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            panel = g[r0:r1, c0:c1]
            if np.any(panel == sep):
                raise AssertionError("separator color cannot appear inside panels")
            panel_outputs.append(_panel_bars(panel, bg))

    panel_h = row_segments[0][1] - row_segments[0][0]
    out_w = sum(p.shape[1] for p in panel_outputs) + len(panel_outputs) - 1
    out = np.full((panel_h, out_w), sep, dtype=int)

    cursor = 0
    for i, panel_out in enumerate(panel_outputs):
        width = panel_out.shape[1]
        out[:, cursor : cursor + width] = panel_out
        cursor += width
        if i != len(panel_outputs) - 1:
            out[:, cursor] = sep
            cursor += 1

    return out


def _sample_counts(rng: random.Random, panel_h: int, panel_w: int, n_colors: int) -> list[int] | None:
    candidates = list(range(1, panel_h + 1))
    rng.shuffle(candidates)
    counts = sorted(candidates[:n_colors])
    if sum(counts) > panel_h * panel_w:
        return None
    return counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_rows = rng.randint(1, 3)
    panel_cols = rng.randint(2, 4)
    panel_h = rng.randint(4, 7)
    panel_w = rng.randint(4, 8)

    h = panel_rows * panel_h + (panel_rows - 1)
    w = panel_cols * panel_w + (panel_cols - 1)
    grid = np.full((h, w), GEN_BG, dtype=int)

    for r in range(1, panel_rows):
        grid[r * panel_h + (r - 1), :] = SEP
    for c in range(1, panel_cols):
        grid[:, c * panel_w + (c - 1)] = SEP

    for pr in range(panel_rows):
        for pc in range(panel_cols):
            r0 = pr * (panel_h + 1)
            c0 = pc * (panel_w + 1)
            n_colors = rng.randint(1, min(4, panel_h, panel_w))
            counts = _sample_counts(rng, panel_h, panel_w, n_colors)
            if counts is None:
                return None
            colors = rng.sample(PALETTE, n_colors)
            cells = [(r, c) for r in range(panel_h) for c in range(panel_w)]
            rng.shuffle(cells)
            needed = sum(counts)
            chosen = cells[:needed]
            idx = 0
            for color, count in zip(colors, counts):
                for _ in range(count):
                    rr, cc = chosen[idx]
                    idx += 1
                    grid[r0 + rr, c0 + cc] = color

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape[0] != panel_h:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
