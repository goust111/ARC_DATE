from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


BG = Color.BLACK
SEP = Color.GREY
PANEL_COUNT = 3
GEN_RETRIES = 400
PANEL_H_RANGE = (7, 9)
PANEL_W_RANGE = (9, 14)
COLOR_POOL = [c for c in Color.NOT_BLACK if c != SEP]

# Canonical 6x7 overlap scaffold. Prefixes of length 4..12 are connected.
_A_ORDER = [
    (1, 2), (2, 2), (3, 2), (3, 1),
    (4, 1), (4, 2), (5, 2), (5, 1),
    (4, 0), (5, 0), (2, 1), (1, 1),
]
_B_ORDER = [
    (1, 2), (1, 3), (1, 4), (2, 4),
    (0, 4), (0, 5), (1, 5), (2, 5),
    (0, 3), (0, 6), (1, 6), (2, 6),
]
_C_ORDER = [
    (2, 2), (1, 3), (2, 3), (3, 3),
    (3, 4), (4, 4), (4, 3), (5, 3),
    (5, 4), (2, 4), (3, 5), (4, 5),
]


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEP)]
    assert len(sep_rows) == 2, "expected exactly two full-width grey separator rows"
    top_h = sep_rows[0]
    mid_h = sep_rows[1] - sep_rows[0] - 1
    bot_h = g.shape[0] - sep_rows[1] - 1
    assert top_h == mid_h == bot_h and top_h > 0, "panels must have equal positive height"

    return [
        g[: sep_rows[0], :].copy(),
        g[sep_rows[0] + 1 : sep_rows[1], :].copy(),
        g[sep_rows[1] + 1 :, :].copy(),
    ]


def _panel_info(panel: np.ndarray) -> dict[str, object]:
    assert not np.any(panel == SEP), "separator color must not appear inside a panel"
    colors = sorted(set(map(int, np.unique(panel))) - {BG})
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    color = int(colors[0])

    comps = find_components(panel, bg=BG, connectivity=4, monochromatic=True, colors=[color])
    assert len(comps) == 1, "each panel must contain exactly one 4-connected sprite"
    comp = comps[0]
    assert comp["area"] == int(np.sum(panel == color)), "all foreground cells must belong to the sprite"

    return {
        "color": color,
        "area": int(comp["area"]),
        "mask": panel == color,
    }


def _compose(panels: list[np.ndarray], draw_order: tuple[int, int, int] | list[int]) -> np.ndarray:
    out = np.zeros_like(panels[0])
    for idx in draw_order:
        mask = panels[int(idx)] != BG
        out[mask] = panels[int(idx)][mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite three stacked panels by using sprite area as the layer priority.

    Input structure:
    - The grid has a black background and two full-width grey separator rows.
    - Those separators split the grid into three equal-height panels stacked top to bottom.
    - Each panel contains exactly one single-color 4-connected sprite, and the three sprite colors are distinct.

    Rule:
    - Measure the area of the unique sprite in each panel.
    - Create a black output canvas with the same shape as one panel.
    - Draw the three panels onto that canvas at matching coordinates: largest-area sprite first, then the middle area, then the smallest-area sprite last so it has the highest priority wherever shapes overlap.

    Key insight:
    - Layer priority is not encoded by panel position; it is encoded only by comparing the three sprite areas across panels.
    """
    panels = _split_panels(np.asarray(grid, dtype=int))
    infos = [_panel_info(panel) for panel in panels]
    areas = [int(info["area"]) for info in infos]
    assert len(set(areas)) == PANEL_COUNT, "sprite areas must be pairwise distinct"

    draw_order = tuple(sorted(range(PANEL_COUNT), key=lambda idx: areas[idx], reverse=True))
    return _compose(panels, draw_order)


def _prefix_mask(order: list[tuple[int, int]], area: int) -> np.ndarray:
    mask = np.zeros((6, 7), dtype=bool)
    for r, c in order[:area]:
        mask[r, c] = True
    return mask


def _transform_mask(mask: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(mask, k=int(rot_k))
    if flip_lr:
        out = np.fliplr(out)
    return out


def _sample_areas() -> list[int]:
    while True:
        areas = random.sample(range(4, 13), PANEL_COUNT)
        draw_order = tuple(sorted(range(PANEL_COUNT), key=lambda idx: areas[idx], reverse=True))
        if draw_order not in ((0, 1, 2), (2, 1, 0)):
            return areas


def _build_panels() -> list[np.ndarray]:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    areas = _sample_areas()
    colors = random.sample(COLOR_POOL, PANEL_COUNT)
    rot_k = random.randrange(4)
    flip_lr = bool(random.getrandbits(1))

    masks = [
        _transform_mask(_prefix_mask(order, area), rot_k, flip_lr)
        for order, area in zip((_A_ORDER, _B_ORDER, _C_ORDER), areas)
    ]
    sprite_h, sprite_w = masks[0].shape
    assert all(mask.shape == (sprite_h, sprite_w) for mask in masks)
    assert sprite_h <= panel_h and sprite_w <= panel_w

    r0 = random.randint(0, panel_h - sprite_h)
    c0 = random.randint(0, panel_w - sprite_w)
    panels = [np.zeros((panel_h, panel_w), dtype=int) for _ in range(PANEL_COUNT)]
    for panel, mask, color in zip(panels, masks, colors):
        view = panel[r0 : r0 + sprite_h, c0 : c0 + sprite_w]
        view[mask] = int(color)

    return panels


def _build_grid(panels: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    grid = np.zeros((panel_h * PANEL_COUNT + 2, panel_w), dtype=int)
    grid[:panel_h, :] = panels[0]
    grid[panel_h, :] = SEP
    grid[panel_h + 1 : 2 * panel_h + 1, :] = panels[1]
    grid[2 * panel_h + 1, :] = SEP
    grid[2 * panel_h + 2 :, :] = panels[2]
    return grid


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        panels = _split_panels(grid)
        infos = [_panel_info(panel) for panel in panels]
    except AssertionError:
        return False

    areas = [int(info["area"]) for info in infos]
    colors = [int(info["color"]) for info in infos]
    masks = [np.asarray(info["mask"], dtype=bool) for info in infos]

    if len(set(areas)) != PANEL_COUNT or len(set(colors)) != PANEL_COUNT:
        return False

    draw_order = tuple(sorted(range(PANEL_COUNT), key=lambda idx: areas[idx], reverse=True))
    if draw_order in ((0, 1, 2), (2, 1, 0)):
        return False

    correct = _compose(panels, draw_order)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, correct):
        return False

    for idx in range(PANEL_COUNT):
        others = [j for j in range(PANEL_COUNT) if j != idx]
        exclusive = masks[idx] & ~masks[others[0]] & ~masks[others[1]]
        if not exclusive.any():
            return False
        if int(colors[idx]) not in set(map(int, np.unique(correct))):
            return False

    for i, j in ((0, 1), (0, 2), (1, 2)):
        k = ({0, 1, 2} - {i, j}).pop()
        pair_only = masks[i] & masks[j] & ~masks[k]
        if not pair_only.any():
            return False

    for perm in itertools.permutations(range(PANEL_COUNT)):
        comp = _compose(panels, perm)
        if perm == draw_order:
            if not np.array_equal(comp, correct):
                return False
        else:
            if np.array_equal(comp, correct):
                return False

    if any(np.array_equal(correct, panel) for panel in panels):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        panels = _build_panels()
        grid = _build_grid(panels)
        if _semantic_check(grid):
            return grid
    raise RuntimeError("failed to generate a valid area-priority panel composite puzzle")
