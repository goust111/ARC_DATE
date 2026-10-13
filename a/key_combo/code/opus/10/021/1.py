from __future__ import annotations

from collections import deque
from itertools import permutations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

PANEL_COLORS = (BLUE, RED, GREEN)
MAX_GENERATION_ATTEMPTS = 400
PANEL_HEIGHT_RANGE = (13, 15)
PANEL_WIDTH = 9
OBJECT_HEIGHT_RANGE = (8, 9)
OBJECT_WIDTH_RANGE = (6, 7)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[tuple[list[tuple[int, int]], bool]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append((cells, touches_border))
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a foreground object"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_holes(panel: np.ndarray) -> int:
    fg = panel != BLACK
    r0, c0, r1, c1 = _bbox(fg)
    crop = fg[r0 : r1 + 1, c0 : c1 + 1]
    zero_mask = ~crop
    return sum(1 for _cells, touches_border in _components(zero_mask) if not touches_border)


def _separator_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    full_sep_cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    assert len(full_sep_cols) >= 2, "expected two grey separator lines"
    runs: list[tuple[int, int]] = []
    start = int(full_sep_cols[0])
    prev = start
    for col in map(int, full_sep_cols[1:]):
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev))
        start = prev = col
    runs.append((start, prev))
    assert len(runs) == 2, "expected exactly two separator runs"
    return runs


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    assert h > 0 and w > 0, "expected a non-empty grid"
    runs = _separator_runs(grid)
    left_end = runs[0][0]
    middle_start = runs[0][1] + 1
    middle_end = runs[1][0]
    right_start = runs[1][1] + 1
    panels = [
        grid[:, :left_end],
        grid[:, middle_start:middle_end],
        grid[:, right_start:],
    ]
    panel_widths = {panel.shape[1] for panel in panels}
    assert len(panel_widths) == 1 and next(iter(panel_widths)) > 0, "panels must have equal positive width"
    return panels


def _compose(panels: list[np.ndarray], order: tuple[int, int, int] | list[int]) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    out = np.full((panel_h, panel_w), BLACK, dtype=int)
    for idx in order:
        mask = panels[idx] != BLACK
        out[mask] = panels[idx][mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite three separated panels by hole-count-defined transparent layer order.

    Input structure:
    - A black background grid contains three equal-width panels separated by full-height grey vertical lines.
    - Each panel contains exactly one 4-connected single-color foreground object in blue, red, or green.
    - The three objects have the same bounding-box size, slightly shifted positions, and 1, 2, and 3 enclosed black holes respectively.

    Rule:
    - Detect the grey separator columns, remove them, and split the grid into three equal panels.
    - In each panel, count the number of enclosed 4-connected black regions inside the foreground object's bounding box.
    - Create a black output canvas the size of one panel and overlay the panels from fewest holes to most holes, painting only non-black pixels so later layers overwrite earlier ones.

    Key insight:
    - Panel depth is determined only by each object's enclosed-hole count; position and color do not define the stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panels = _split_panels(g)
    hole_counts = [_count_holes(panel) for panel in panels]
    assert len(set(hole_counts)) == 3, "hole counts must be distinct"
    order = sorted(range(3), key=lambda idx: hole_counts[idx])
    return _compose(panels, order)


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, hh, ww = rect
    return {(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)}


def _slot_rects(obj_h: int, obj_w: int) -> dict[str, tuple[int, int, int, int]]:
    center_col = max(1, min(obj_w - 3, obj_w // 2 - 1))
    return {
        "shared": (1, 1, 2, 1),
        "top_only": (1, obj_w - 3, 1, 2),
        "bottom_only": (obj_h // 2, 1, 1, 2),
        "middle_extra": (obj_h // 2 - 1, obj_w - 2, 2, 1),
        "top_extra": (obj_h - 2, center_col, 1, 2),
    }


def _transform_rect(rect: tuple[int, int, int, int], obj_h: int, obj_w: int, flip_v: bool, flip_h: bool) -> tuple[int, int, int, int]:
    r0, c0, hh, ww = rect
    if flip_v:
        r0 = obj_h - r0 - hh
    if flip_h:
        c0 = obj_w - c0 - ww
    return r0, c0, hh, ww


def _build_object(
    obj_h: int,
    obj_w: int,
    color: int,
    slot_names: tuple[str, ...],
    flip_v: bool,
    flip_h: bool,
) -> np.ndarray:
    obj = np.full((obj_h, obj_w), color, dtype=int)
    slots = _slot_rects(obj_h, obj_w)
    used_cells: set[tuple[int, int]] = set()
    for name in slot_names:
        rect = _transform_rect(slots[name], obj_h, obj_w, flip_v, flip_h)
        cells = _rect_cells(rect)
        assert used_cells.isdisjoint(cells), "holes must be disjoint"
        used_cells |= cells
        for r, c in cells:
            obj[r, c] = BLACK
    assert _count_holes(obj) == len(slot_names), "object must realize the requested hole count"
    return obj


def _place_object(panel_h: int, panel_w: int, patch: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    panel = np.full((panel_h, panel_w), BLACK, dtype=int)
    r0, c0 = offset
    h, w = patch.shape
    panel[r0 : r0 + h, c0 : c0 + w] = patch
    return panel


def _build_input(panels: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    grid = np.full((panel_h, panel_w * 3 + 2), BLACK, dtype=int)
    grid[:, panel_w] = GREY
    grid[:, panel_w * 2 + 1] = GREY
    grid[:, :panel_w] = panels[0]
    grid[:, panel_w + 1 : panel_w * 2 + 1] = panels[1]
    grid[:, panel_w * 2 + 2 :] = panels[2]
    return grid


def _visible_color_counts(out: np.ndarray) -> dict[int, int]:
    colors, counts = np.unique(out[out != BLACK], return_counts=True)
    return {int(color): int(count) for color, count in zip(colors, counts)}


def _actual_order_from_counts(counts: list[int]) -> tuple[int, int, int]:
    return tuple(sorted(range(3), key=lambda idx: counts[idx]))


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = PANEL_WIDTH
    obj_h = random.randint(*OBJECT_HEIGHT_RANGE)
    obj_w = random.randint(*OBJECT_WIDTH_RANGE)

    possible_offsets = [
        (r, c)
        for r in range(panel_h - obj_h + 1)
        for c in range(panel_w - obj_w + 1)
    ]
    shared_offset = random.choice(possible_offsets)
    shifted_offsets = [
        offset
        for offset in possible_offsets
        if offset != shared_offset and abs(offset[0] - shared_offset[0]) <= 3 and abs(offset[1] - shared_offset[1]) <= 2
    ]
    if not shifted_offsets:
        return None
    bottom_offset = random.choice(shifted_offsets)

    flip_v = bool(random.getrandbits(1))
    flip_h = bool(random.getrandbits(1))

    layer_colors = list(PANEL_COLORS)
    random.shuffle(layer_colors)
    layers = [
        _build_object(obj_h, obj_w, layer_colors[0], ("bottom_only",), flip_v, flip_h),
        _build_object(obj_h, obj_w, layer_colors[1], ("shared", "middle_extra"), flip_v, flip_h),
        _build_object(obj_h, obj_w, layer_colors[2], ("shared", "top_only", "top_extra"), flip_v, flip_h),
    ]
    layer_offsets = [bottom_offset, shared_offset, shared_offset]

    layer_to_position = list(range(3))
    random.shuffle(layer_to_position)
    panels = [None, None, None]
    hole_counts = [0, 0, 0]
    panel_colors = [0, 0, 0]
    for layer_idx, pos_idx in enumerate(layer_to_position):
        panel = _place_object(panel_h, panel_w, layers[layer_idx], layer_offsets[layer_idx])
        panels[pos_idx] = panel
        hole_counts[pos_idx] = layer_idx + 1
        panel_colors[pos_idx] = int(layer_colors[layer_idx])
    assert all(panel is not None for panel in panels)

    actual_order = _actual_order_from_counts(hole_counts)
    if actual_order in ((0, 1, 2), (2, 1, 0)):
        return None

    color_order = tuple(idx for idx, _color in sorted(enumerate(panel_colors), key=lambda item: item[1]))
    if actual_order == color_order or actual_order == color_order[::-1]:
        return None

    expected = _compose(panels, actual_order)
    visible = _visible_color_counts(expected)
    if len(visible) != 3:
        return None

    for order in permutations(range(3)):
        if order == actual_order:
            continue
        if np.array_equal(expected, _compose(panels, order)):
            return None

    grid = _build_input(panels)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _sample_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
