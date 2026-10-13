from __future__ import annotations

import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat the bottom foreground motif upward until it fills the whole grid.

    Input structure
    - The background is the dominant color.
    - The only foreground pixels occupy a contiguous block of rows at the bottom.
    - That bottom block is the complete vertical motif, including its background cells.

    Rule
    - Copy the bottom motif rows repeatedly upward across the full grid.
    - Preserve horizontal alignment and vertical phase: the original first motif row remains at its input row.
    - The output shape is unchanged.

    Key insight
    The blank rows above the bottom foreground block are missing repetitions of the same row motif, not empty space.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    row_has_fg = np.any(g != bg, axis=1)
    assert np.any(row_has_fg), "expected a bottom foreground motif"
    top = int(np.flatnonzero(row_has_fg)[0])
    assert top > 0, "expected background rows above the motif"
    assert np.all(row_has_fg[top:]), "motif rows must be contiguous through the bottom"

    motif = g[top:].copy()
    motif_h = int(motif.shape[0])

    out = np.empty_like(g)
    for r in range(g.shape[0]):
        out[r] = motif[(int(r) - top) % motif_h]
    return out

# --------------- generator (att2) ---------------

# Canvas and motif size bounds.
HEIGHT_RANGE = (8, 14)
WIDTH_RANGE = (5, 9)
MOTIF_H_RANGE = (3, 6)
# Foreground support and density limits.
BBOX_W_RANGE = (3, 5)
FG_DENSITY_RANGE = (0.30, 0.50)
# Generation retry limit.
MAX_ATTEMPTS = 300

def _sample_dims(rng) -> tuple[int, int, int, int] | None:
    """Return height, width, blank rows, and motif height."""
    motif_h = int(rng.integers(MOTIF_H_RANGE[0], MOTIF_H_RANGE[1] + 1))
    min_h = max(HEIGHT_RANGE[0], motif_h * 2)
    if min_h > HEIGHT_RANGE[1]:
        return None
    height = int(rng.integers(min_h, HEIGHT_RANGE[1] + 1))
    width = int(rng.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    return height, width, height - motif_h, motif_h

def _sample_fg_cells(rng, motif_h: int, width: int, bbox_w: int) -> set[tuple[int, int]] | None:
    """Choose bbox-local cells while forcing every row and both bbox edges."""
    min_fg = max(motif_h, int(np.ceil(FG_DENSITY_RANGE[0] * motif_h * width)))
    max_fg = min(motif_h * bbox_w, int(np.floor(FG_DENSITY_RANGE[1] * motif_h * width)))
    if max_fg < min_fg:
        return None
    target_fg = int(rng.integers(min_fg, max_fg + 1))
    selected = {(int(rng.integers(0, motif_h)), 0), (int(rng.integers(0, motif_h)), bbox_w - 1)}
    for r in range(motif_h):
        if not any(rr == r for rr, _ in selected):
            selected.add((r, int(rng.integers(0, bbox_w))))
    if len(selected) > target_fg:
        return None
    remaining = [
        (r, c)
        for r in range(motif_h)
        for c in range(bbox_w)
        if (r, c) not in selected
    ]
    need = target_fg - len(selected)
    if need:
        for i in rng.choice(len(remaining), size=need, replace=False):
            selected.add(remaining[int(i)])
    return selected

def _paint_motif(rng, motif_h: int, width: int, c0: int, selected: set[tuple[int, int]], bg: int, fg_colors: list[int]) -> np.ndarray:
    """Render selected bbox-local cells into a full-width motif."""
    motif = np.full((motif_h, width), bg, dtype=int)
    coords = list(selected)
    coords = [coords[int(i)] for i in rng.permutation(len(coords))]
    for i, (r, c) in enumerate(coords):
        if i == 0:
            color = fg_colors[0]
        elif i == 1:
            color = fg_colors[1]
        else:
            color = int(rng.choice(fg_colors))
        motif[r, c0 + c] = color
    return motif

def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    dims = _sample_dims(rng)
    if dims is None:
        return None
    height, width, blank_rows, motif_h = dims
    colors = rng.choice(np.arange(10), size=3, replace=False)
    bg = int(colors[0])
    fg_colors = [int(colors[1]), int(colors[2])]
    bbox_w = int(rng.integers(BBOX_W_RANGE[0], min(BBOX_W_RANGE[1], width) + 1))
    c0 = int(rng.integers(0, width - bbox_w + 1))
    selected = _sample_fg_cells(rng, motif_h, width, bbox_w)
    if selected is None:
        return None
    motif = _paint_motif(rng, motif_h, width, c0, selected, bg, fg_colors)
    expected = motif[(np.arange(height) - blank_rows) % motif_h]
    inp = expected.copy()
    inp[:blank_rows] = bg
    if np.array_equal(main(inp), expected):
        return inp
    return None

def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 9b30e358 input")
