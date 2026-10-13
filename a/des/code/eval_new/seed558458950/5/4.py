"""Reconstruct overlapping axis-aligned rectangles from visible pixels and redraw them by the layer order stored in the top-left legend column.

Input structure:
- Background color is the most frequent value (bg, usually 0).
- A vertical legend is placed in column 0 at the top: one pixel per layer color, top-to-bottom from bottom layer to top layer.
- The rest of the grid contains full rectangles of those colors in some hidden true stacking order; overlap and occlusion can hide parts of lower rectangles.
- Only the legend column is guaranteed to be out-of-band metadata; all other non-bg pixels belong to rectangle layers.

Rule:
- Detect bg as the most frequent color, then read the legend colors from top to bottom.
- For each legend color, infer its full rectangle by taking all visible pixels of that color (excluding the legend column), forming an initial bounding box, and expanding that box outward while every added outer border row/column stays non-bg.
- Start from an empty bg canvas, then draw each inferred rectangle in the order from legend-bottom to legend-top, so later draws overwrite earlier ones in overlaps.
- Clear the legend column in the output to bg.

Key insight:
- The legend is not a color key but a reordering instruction; once rectangle extents are reconstructed, deterministic re-rendering in that prescribed order resolves every hidden overlap.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


Rect = Tuple[int, int, int, int]


def _most_frequent_color(grid: np.ndarray) -> int:
    """Return the background color (most common value)."""
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _read_legend_colors(grid: np.ndarray, bg: int) -> List[int]:
    """Read the ordered legend from the top of column 0."""
    colors: List[int] = []
    for value in grid[:, 0]:
        color = int(value)
        if color == bg:
            if colors:
                break
            continue
        if color in colors:
            continue
        colors.append(color)
    return colors


def _expand_box(grid: np.ndarray, bg: int, box: Rect) -> Rect:
    """Expand a color box outward while every added border line is non-bg."""
    r0, r1, c0, c1 = box
    h, w = grid.shape

    changed = True
    while changed:
        changed = False
        # Expand up
        if r0 > 0 and np.all(grid[r0 - 1, c0 : c1 + 1] != bg):
            r0 -= 1
            changed = True
        # Expand down
        if r1 + 1 < h and np.all(grid[r1 + 1, c0 : c1 + 1] != bg):
            r1 += 1
            changed = True
        # Expand left
        if c0 > 1 and np.all(grid[r0 : r1 + 1, c0 - 1] != bg):
            c0 -= 1
            changed = True
        # Expand right
        if c1 + 1 < w and np.all(grid[r0 : r1 + 1, c1 + 1] != bg):
            c1 += 1
            changed = True
    return (r0, r1, c0, c1)


def _infer_rectangle(grid: np.ndarray, bg: int, color: int) -> Optional[Rect]:
    """Infer the full axis-aligned rectangle for a layer color."""
    # Ignore legend cells in column 0 when inferring geometry.
    mask = np.zeros_like(grid, dtype=bool)
    mask[:, 1:] = grid[:, 1:] == color

    positions = np.argwhere(mask)
    if positions.size == 0:
        return None

    r0, c0 = positions.min(axis=0)
    r1, c1 = positions.max(axis=0)
    return _expand_box(grid, bg, (int(r0), int(r1), int(c0), int(c1)))


def main(grid: np.ndarray) -> np.ndarray:
    """Recompose overlapping colored rectangles by inferring hidden full extents and repainting in legend order.

    Input structure:
    - A background color is dominant (bg, usually 0).
    - The top-left vertical legend in column 0 defines the target stack: one cell per layer color, listed from bottom layer to top layer.
    - The rest of the grid contains axis-aligned rectangles in those colors, already rendered by a different hidden stacking order that may cause occlusion.

    Rule:
    - Read bg as the most frequent color and parse the legend color sequence from top to bottom.
    - For each legend color, gather visible pixels (excluding the legend column), infer the full rectangle by expanding the visible bounding box while every added border row/column remains non-bg.
    - Draw rectangles on an empty bg grid from legend-bottom to legend-top (last drawn = top layer), overwriting overlaps deterministically.
    - Remove the legend column in the output.

    Key insight:
    - The legend gives a reordering of layer draws, not a fixed mapping; once each rectangle extent is reconstructed, replaying rectangles in the indicated order resolves the obscured overlaps.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("grid must be 2D")

    bg = _most_frequent_color(grid)
    order_colors = _read_legend_colors(grid, bg)

    out = np.full_like(grid, bg)
    if not order_colors:
        return out

    inferred = {}
    for color in order_colors:
        rect = _infer_rectangle(grid, bg, color)
        if rect is not None:
            inferred[color] = rect

    for color in order_colors:
        rect = inferred.get(color)
        if rect is None:
            continue
        r0, r1, c0, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    # Remove the legend marker column.
    out[:, 0] = bg
    return out


def _bbox_intersects(a: Rect, b: Rect) -> bool:
    a0, a1, b0, b1 = a
    c0, c1, d0, d1 = b
    return not (a1 < c0 or c1 < a0 or b1 < d0 or d1 < b0)


def _rect_inside(parent: Rect, child: Rect) -> bool:
    a0, a1, b0, b1 = parent
    c0, c1, d0, d1 = child
    return c0 >= a0 and c1 <= a1 and d0 >= b0 and d1 <= b1


def _generate_nested_chain(rng: np.random.Generator, h: int, w: int, count: int) -> Optional[List[Rect]]:
    """Generate a strictly shrinking contained chain, largest to smallest."""
    max_trials = 80
    for _ in range(max_trials):
        min_side = max(5, count + 3)
        if h - 1 <= min_side or w - 1 <= min_side:
            return None

        outer_h = int(rng.integers(min_side, min(h - 1, min_side + 12) + 1))
        outer_w = int(rng.integers(min_side, min(w - 1, min_side + 12) + 1))
        if outer_h <= 2 or outer_w <= 2:
            continue

        max_r0 = h - outer_h
        max_c0 = w - outer_w
        if max_r0 < 1 or max_c0 < 2:
            continue

        r0 = int(rng.integers(1, max_r0 + 1))
        c0 = int(rng.integers(1, max_c0 + 1))
        chain: List[Rect] = [(r0, r0 + outer_h - 1, c0, c0 + outer_w - 1)]

        ok = True
        for _ in range(1, count):
            pr0, pr1, pc0, pc1 = chain[-1]
            ph = pr1 - pr0 + 1
            pw = pc1 - pc0 + 1
            if ph < 4 or pw < 4:
                ok = False
                break

            nh = int(rng.integers(2, ph))
            nw = int(rng.integers(2, pw))
            nr0 = int(rng.integers(pr0, pr1 - nh + 2))
            nc0 = int(rng.integers(pc0, pc1 - nw + 2))
            candidate = (nr0, nr0 + nh - 1, nc0, nc0 + nw - 1)

            if _rect_inside((pr0, pr1, pc0, pc1), candidate):
                chain.append(candidate)
            else:
                ok = False
                break

        if ok and len(chain) == count:
            return chain

    return None


def _rect_size_rect(rng: np.random.Generator, h: int, w: int) -> Optional[Rect]:
    """Sample a random background-free rectangle for a side object."""
    min_size = 3
    max_size_h = max(min_size + 1, h - 2)
    max_size_w = max(min_size + 1, w - 2)

    if max_size_h <= min_size or max_size_w <= min_size:
        return None

    rect_h = int(rng.integers(min_size, min(max_size_h, min_size + 8) + 1))
    rect_w = int(rng.integers(min_size, min(max_size_w, min_size + 8) + 1))
    max_r0 = h - rect_h
    max_c0 = w - rect_w
    if max_r0 < 1 or max_c0 < 2:
        return None

    r0 = int(rng.integers(1, max_r0 + 1))
    c0 = int(rng.integers(1, max_c0 + 1))
    return (r0, r0 + rect_h - 1, c0, c0 + rect_w - 1)


def _find_non_overlapping_rect(rng: np.random.Generator, h: int, w: int, blocked: List[Rect]) -> Optional[Rect]:
    """Find a random rectangle that does not intersect any blocked rectangle."""
    for _ in range(120):
        rect = _rect_size_rect(rng, h, w)
        if rect is None:
            return None
        if any(_bbox_intersects(rect, b) for b in blocked):
            continue
        return rect
    return None


def generate_input() -> np.ndarray:
    """Generate random valid inputs for this task."""
    rng = np.random.default_rng()

    h = int(rng.integers(16, 30))
    w = int(rng.integers(16, 30))

    bg = 0
    max_layers = min(5, max(2, min((h - 3) // 3, (w - 3) // 3)))
    n_layers = int(rng.integers(2, max_layers + 1))

    palette = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
    colors = rng.choice(palette, size=n_layers, replace=False).tolist()

    # Branch 1: pure nested overlap chain.
    # Branch 2: chain + one disjoint side rectangle (extra geometric variety).
    add_side = n_layers >= 3 and rng.random() < 0.5
    chain_count = n_layers - 1 if add_side else n_layers

    chain = _generate_nested_chain(rng, h, w, chain_count)
    while chain is None:
        # Retry with a slightly new grid size if the requested chain is impossible.
        h = int(rng.integers(16, 30))
        w = int(rng.integers(16, 30))
        max_layers = min(5, max(2, min((h - 3) // 3, (w - 3) // 3)))
        n_layers = min(n_layers, max_layers)
        colors = rng.choice(palette, size=n_layers, replace=False).tolist()
        chain_count = n_layers - 1 if add_side and n_layers >= 3 else n_layers
        chain = _generate_nested_chain(rng, h, w, chain_count)

    rects = chain.copy()
    if add_side and chain_count < n_layers:
        side = _find_non_overlapping_rect(rng, h, w, chain)
        if side is not None:
            rects.append(side)
        else:
            colors = colors[:chain_count]
            n_layers = chain_count
            add_side = False

    # True hidden stacking: chain depth order (outer -> inner), side inserted at random position.
    # This guarantees all rectangles remain partially visible.
    core_order = list(range(len(rects)))
    core_order = core_order[:chain_count]
    if len(rects) > chain_count:
        side_idx = len(rects) - 1
        insert_at = int(rng.integers(0, len(rects)))
        core_order = core_order[:]
        core_order.insert(insert_at, side_idx)

    true_order = core_order

    grid = np.full((h, w), bg, dtype=int)
    for idx in reversed(true_order):
        r0, r1, c0, c1 = rects[idx]
        grid[r0 : r1 + 1, c0 : c1 + 1] = colors[idx]

    true_order_colors = [colors[idx] for idx in true_order]
    # Branch for whether legend order matches true order.
    if rng.random() < 0.5:
        legend_colors = true_order_colors
    else:
        legend_colors = true_order_colors.copy()
        while True:
            rng.shuffle(legend_colors)
            if legend_colors != true_order_colors:
                break

    for row, color in enumerate(legend_colors):
        grid[row, 0] = color

    return grid
