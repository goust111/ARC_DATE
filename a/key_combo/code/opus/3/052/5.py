from __future__ import annotations

import random

import numpy as np


def _panel_ranges(grid: np.ndarray) -> list[tuple[int, int]]:
    sep = [r for r in range(grid.shape[0]) if np.all(grid[r] == 5)]
    cuts = [-1] + sep + [grid.shape[0]]
    return [(cuts[i] + 1, cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i] + 1 < cuts[i + 1]]


def _slide_panel(panel: np.ndarray) -> np.ndarray:
    fg = [(int(r), int(c)) for r, c in np.argwhere(panel != 0)]
    fg_set: set[tuple[int, int]] = set()
    for r, c in fg:
        has_southeast_support = any(rr > r and cc > c for rr, cc in fg)
        nc = min(c + 1, panel.shape[1] - 1) if has_southeast_support else c
        fg_set.add((r, nc))
    out = np.zeros_like(panel)
    color = int(panel[fg[0]]) if fg else 0
    for r, c in fg_set:
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each stacked panel's foreground pixels east if southeast-supported, then overlay middle, top, bottom."""
    ranges = _panel_ranges(grid)
    panels = [_slide_panel(grid[r0:r1]) for r0, r1 in ranges]
    if not panels:
        return np.zeros((0, 0), dtype=grid.dtype)
    height = max(p.shape[0] for p in panels)
    width = max(p.shape[1] for p in panels)
    out = np.zeros((height, width), dtype=grid.dtype)
    order = [1, 0, 2]
    for idx in order:
        if idx >= len(panels):
            continue
        panel = panels[idx]
        mask = panel != 0
        out[: panel.shape[0], : panel.shape[1]][mask] = panel[mask]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((21, 7), dtype=int)
    grid[7, :] = 5
    grid[14, :] = 5
    shapes = [
        [(1, 1), (1, 2), (2, 1), (3, 2), (4, 2)],
        [(1, 1), (2, 1), (2, 2), (3, 3), (4, 3)],
        [(1, 2), (2, 2), (3, 1), (3, 2), (4, 4)],
    ]
    random.shuffle(shapes)
    colors = random.sample([1, 2, 3, 4, 6], 3)
    offsets = [0, 8, 15]
    for shape, color, off in zip(shapes, colors, offsets):
        for r, c in shape:
            if off + r < grid.shape[0]:
                grid[off + r, c] = color
    return grid
