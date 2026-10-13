from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
POSITIVE_MASK_COLOR = 1
NEGATIVE_MASK_COLOR = 2
DIVIDER_COLOR = 5
SOURCE_PALETTE = (3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

PANEL_HEIGHT_RANGE = (3, 5)
PANEL_WIDTH_RANGE = (3, 5)
GEN_ATTEMPTS = 400


def _foreground_count(panel: np.ndarray) -> int:
    return int(np.count_nonzero(panel != BG))


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    divider_rows = np.flatnonzero(np.all(grid == DIVIDER_COLOR, axis=1))
    assert divider_rows.size == 1, "expected exactly one full-width grey divider row"
    divider_r = int(divider_rows[0])
    assert divider_r * 2 + 1 == grid.shape[0], "divider must split the grid into equal top/bottom panels"
    top = grid[:divider_r, :]
    bottom = grid[divider_r + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0
    return top, bottom


def _render_output(mask_panel: np.ndarray, source_panel: np.ndarray) -> np.ndarray:
    panel_h, panel_w = map(int, mask_panel.shape)
    source_mask = source_panel != BG
    assert source_mask.any() and (~source_mask).any(), "source template must contain both foreground and background"

    out = np.full((panel_h * panel_h, panel_w * panel_w), BG, dtype=int)
    for r in range(panel_h):
        for c in range(panel_w):
            color = int(mask_panel[r, c])
            if color == BG:
                continue
            block = out[r * panel_h : (r + 1) * panel_h, c * panel_w : (c + 1) * panel_w]
            if color == NEGATIVE_MASK_COLOR:
                block[~source_mask] = color
            else:
                block[source_mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the sparser panel as a blue/red mask that stamps the denser panel's binary template into output blocks.

    Input structure:
    - A full-width grey divider row splits the grid into equal top and bottom panels on a black background.
    - One panel has strictly fewer non-black cells and acts as the mask; its foreground cells are blue and/or red.
    - The other panel is a denser source template whose own foreground colors may vary but whose binary silhouette is what matters.

    Rule:
    - Compare the two panels by non-black cell count; the sparser one is the mask and the denser one is the source.
    - Produce an output of shape `(panel_h * panel_h, panel_w * panel_w)`, interpreted as a `panel_h x panel_w` array of `panel_h x panel_w` blocks.
    - Each blue mask cell stamps the source foreground as a positive stencil in its corresponding block, recolored blue.
    - Each red mask cell stamps the complement of the source silhouette as a negative stencil in its corresponding block, recolored red; mask-background positions leave their blocks black.

    Key insight:
    - Density decides which panel gives per-cell stamping instructions, and the mask color controls positive versus negative stamping while the source panel's original colors are discarded.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top, bottom = _split_panels(g)
    top_count = _foreground_count(top)
    bottom_count = _foreground_count(bottom)
    assert top_count != bottom_count, "exactly one panel must be sparser"

    if top_count < bottom_count:
        mask_panel, source_panel = top, bottom
    else:
        mask_panel, source_panel = bottom, top

    mask_colors = set(map(int, np.unique(mask_panel))) - {BG}
    assert mask_colors and mask_colors <= {POSITIVE_MASK_COLOR, NEGATIVE_MASK_COLOR}
    assert _foreground_count(source_panel) > _foreground_count(mask_panel)
    return _render_output(mask_panel, source_panel)


def _random_cells(h: int, w: int, count: int) -> list[tuple[int, int]]:
    return random.sample([(r, c) for r in range(h) for c in range(w)], count)


def _good_spread(cells: Iterable[tuple[int, int]]) -> bool:
    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    return len(rows) > 1 and len(cols) > 1


def _sample_mask_cells(h: int, w: int, count: int) -> list[tuple[int, int]]:
    for _ in range(80):
        cells = _random_cells(h, w, count)
        if _good_spread(cells):
            return cells
    return _random_cells(h, w, count)


def _sample_connected_shape(h: int, w: int, count: int) -> list[tuple[int, int]]:
    all_cells = {(r, c) for r in range(h) for c in range(w)}
    for _ in range(120):
        chosen = {random.choice(tuple(all_cells))}
        while len(chosen) < count:
            frontier = set()
            for r, c in chosen:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in all_cells and (nr, nc) not in chosen:
                        frontier.add((nr, nc))
            chosen.add(random.choice(tuple(frontier)))
        if _good_spread(chosen):
            return sorted(chosen)
    return sorted(chosen)


def _paint_mask_panel(h: int, w: int, cells: list[tuple[int, int]]) -> np.ndarray:
    panel = np.full((h, w), BG, dtype=int)
    mode = random.choices(("positive", "negative", "mixed"), weights=(3, 2, 5), k=1)[0]
    if mode == "positive":
        for r, c in cells:
            panel[r, c] = POSITIVE_MASK_COLOR
    elif mode == "negative":
        for r, c in cells:
            panel[r, c] = NEGATIVE_MASK_COLOR
    else:
        neg_count = random.randint(1, len(cells) - 1)
        neg_cells = set(random.sample(cells, neg_count))
        for r, c in cells:
            panel[r, c] = NEGATIVE_MASK_COLOR if (r, c) in neg_cells else POSITIVE_MASK_COLOR
    return panel


def _paint_source_panel(h: int, w: int, cells: list[tuple[int, int]]) -> np.ndarray:
    panel = np.full((h, w), BG, dtype=int)
    palette = random.sample(SOURCE_PALETTE, random.choice((1, 2)))
    if len(palette) == 1:
        for r, c in cells:
            panel[r, c] = palette[0]
        return panel

    split = random.randint(1, len(cells) - 1)
    alt_cells = set(random.sample(cells, split))
    for r, c in cells:
        panel[r, c] = palette[1] if (r, c) in alt_cells else palette[0]
    return panel


def _make_example() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    area = panel_h * panel_w

    mask_count_max = max(2, area // 3)
    mask_count = random.randint(2, mask_count_max)

    source_min = max(mask_count + 1, (area * 7 + 19) // 20)
    source_max = min(area - 2, (area * 13) // 20)
    if source_min > source_max:
        return None
    source_count = random.randint(source_min, source_max)

    mask_cells = _sample_mask_cells(panel_h, panel_w, mask_count)
    source_cells = _sample_connected_shape(panel_h, panel_w, source_count)

    mask_panel = _paint_mask_panel(panel_h, panel_w, mask_cells)
    source_panel = _paint_source_panel(panel_h, panel_w, source_cells)

    if _foreground_count(mask_panel) >= _foreground_count(source_panel):
        return None

    divider = np.full((1, panel_w), DIVIDER_COLOR, dtype=int)
    if random.choice((False, True)):
        grid = np.vstack([mask_panel, divider, source_panel])
    else:
        grid = np.vstack([source_panel, divider, mask_panel])

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
