from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, components_mask


FRAME = 5
BG = 0
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 400
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _find_structure(g: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    candidates: list[tuple[tuple[int, int, int, int], int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        b = bbox(g == color)
        if b is None:
            continue
        r0, c0, r1, c1 = b
        if r1 - r0 < 2 or c1 - c0 < 5:
            continue
        if not np.all(g[r0, c0 : c1 + 1] == color):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == color):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == color):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == color):
            continue
        dividers = [c for c in range(c0 + 1, c1) if np.all(g[r0 : r1 + 1, c] == color)]
        if len(dividers) != 1:
            continue
        candidates.append((b, int(dividers[0])))
    assert len(candidates) == 1, "expected exactly one framed panel with one divider"
    return candidates[0]


def _left_bar_rows(g: np.ndarray, frame_bbox: tuple[int, int, int, int], divider: int) -> dict[int, int]:
    r0, c0, r1, _c1 = frame_bbox
    left = g[r0 + 1 : r1, c0 + 1 : divider]
    color_to_row: dict[int, int] = {}
    for rr in range(left.shape[0]):
        cols = np.flatnonzero(left[rr] != BG)
        if cols.size == 0:
            continue
        assert cols.size == 2 and int(cols[1]) == int(cols[0]) + 1, "each source strip must be 2 cells wide"
        color = int(left[rr, cols[0]])
        assert int(left[rr, cols[1]]) == color and color != FRAME
        abs_row = r0 + 1 + rr
        assert color not in color_to_row, "bar colors must be unique"
        color_to_row[color] = abs_row
    assert 3 <= len(color_to_row) <= 4, "expected 3-4 source strips"
    return color_to_row


def _outside_components(g: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> dict[int, int]:
    r0, c0, r1, c1 = frame_bbox
    outside = np.ones(g.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = components_mask(outside & (g != BG), connectivity=4)
    color_to_area: dict[int, int] = {}
    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "outside sprite must be monochrome"
        color = int(next(iter(colors)))
        assert color != FRAME and color not in color_to_area, "outside sprite colors must be unique"
        color_to_area[color] = int(comp.shape[0])
    assert 3 <= len(color_to_area) <= 4, "expected 3-4 outside sprites"
    return color_to_area


def _transform(g: np.ndarray) -> np.ndarray:
    frame_bbox, divider = _find_structure(g)
    color_to_row = _left_bar_rows(g, frame_bbox, divider)
    color_to_area = _outside_components(g, frame_bbox)
    assert set(color_to_row) == set(color_to_area), "bar colors and sprite colors must match exactly"

    r0, c0, r1, c1 = frame_bbox
    right_start = divider + 1
    right_width = c1 - divider - 1
    assert right_width >= 1

    out = np.array(g, copy=True)
    outside = np.ones(g.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    out[outside] = BG

    for color, row in color_to_row.items():
        width = min(color_to_area[color], right_width)
        out[row, right_start : right_start + width] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match outside monochrome sprites to same-color source bars and stretch right-panel bars by sprite area.

    Input structure:
    - A black background contains one grey rectangular frame with a full-height grey divider, splitting a narrow left panel from a wider right panel.
    - The left panel contains 3 or 4 one-row horizontal color bars, each exactly 2 cells wide and each with a distinct color.
    - The right panel is blank, while the area outside the frame contains 3 or 4 separate monochrome 4-connected sprites whose colors match the left-panel bars one-to-one.

    Rule:
    - Find every outside sprite, measure its 4-connected area, and use its color to choose the matching source bar in the left panel.
    - On the matching row inside the right panel, draw a same-color horizontal bar starting immediately to the right of the divider.
    - The new bar's width equals the matched sprite area, clipped by the right-panel width; clear all outside sprites to black and keep the frame, divider, and left panel unchanged.

    Key insight:
    - Each outside sprite carries two independent pieces of information at once: its color says which row it controls, and its area says how far that row must extend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _transform(g)


def _grow_shape(area: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))
    rs = [r for r, _c in cells]
    cs = [c for _r, c in cells]
    min_r, min_c = min(rs), min(cs)
    normalized = {(r - min_r, c - min_c) for r, c in cells}
    return normalized


def _shape_array(area: int) -> np.ndarray:
    for _ in range(80):
        cells = _grow_shape(area)
        h = max(r for r, _c in cells) + 1
        w = max(c for _r, c in cells) + 1
        arr = np.zeros((h, w), dtype=int)
        for r, c in cells:
            arr[r, c] = 1
        if area >= 4 and h == 1:
            continue
        if area >= 4 and w == 1:
            continue
        if area >= 5 and h * w == area and random.random() < 0.85:
            continue
        return arr
    raise RuntimeError("failed to build a varied 4-connected shape")


def _rows_for_bars(n_bars: int) -> tuple[int, list[int]]:
    for _ in range(100):
        top_gap = random.randint(0, 1)
        bottom_gap = random.randint(0, 1)
        middle_gaps = [random.randint(1, 2) for _ in range(n_bars - 1)]
        inner_h = top_gap + n_bars + sum(middle_gaps) + bottom_gap
        if 7 <= inner_h <= 12:
            rows: list[int] = []
            cur = top_gap
            for idx in range(n_bars):
                rows.append(cur)
                cur += 1
                if idx < n_bars - 1:
                    cur += middle_gaps[idx]
            return inner_h, rows
    raise RuntimeError("failed to sample bar rows")


def _choose_areas(right_width: int, n_bars: int) -> list[int]:
    for _ in range(120):
        force_truncation = random.random() < 0.4
        small_pool = list(range(2, 8))
        random.shuffle(small_pool)
        areas = small_pool[:n_bars]
        if force_truncation:
            big = random.randint(right_width + 1, min(12, right_width + 3))
            replace_idx = random.randrange(n_bars)
            areas[replace_idx] = big
        if len(set(areas)) != n_bars:
            continue
        if force_truncation and max(areas) <= right_width:
            continue
        return areas
    raise RuntimeError("failed to choose distinct sprite areas")


def _can_place(mask: np.ndarray, occupied: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        if occupied[rr, cc]:
            return False
    return True


def _place_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    H, W = grid.shape
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        grid[rr, cc] = color
        for dr, dc in DIRS8:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < H and 0 <= nc < W:
                occupied[nr, nc] = True


def _place_sprites(
    grid: np.ndarray,
    occupied: np.ndarray,
    placements: list[tuple[int, int, np.ndarray, int]],
) -> bool:
    H, W = grid.shape
    order = list(range(len(placements)))
    random.shuffle(order)
    for idx in order:
        _area, _color, mask, _row = placements[idx]
        h, w = mask.shape
        options = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(options)
        chosen: tuple[int, int] | None = None
        for top, left in options:
            if not _can_place(mask, occupied, top, left):
                continue
            chosen = (top, left)
            break
        if chosen is None:
            return False
        top, left = chosen
        _place_mask(grid, occupied, mask, _color, top, left)
    return True


def _centroid_order(g: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = frame_bbox
    outside = np.ones(g.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = components_mask(outside & (g != BG), connectivity=4)
    keyed = []
    for comp in comps:
        color = int(g[tuple(comp[0])])
        centroid_r = float(np.mean(comp[:, 0]))
        centroid_c = float(np.mean(comp[:, 1]))
        keyed.append((centroid_r, centroid_c, color))
    keyed.sort()
    return [color for _r, _c, color in keyed]


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    colors_by_row: list[int],
    areas_by_row: list[int],
    frame_bbox: tuple[int, int, int, int],
    divider: int,
) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    r0, c0, r1, c1 = frame_bbox
    right = grid[r0 + 1 : r1, divider + 1 : c1]
    if np.any(right != BG):
        return False

    if areas_by_row == sorted(areas_by_row) or areas_by_row == sorted(areas_by_row, reverse=True):
        return False

    color_order_by_sprite = _centroid_order(grid, frame_bbox)
    if color_order_by_sprite == colors_by_row:
        return False

    out_right = expected[r0 + 1 : r1, divider + 1 : c1]
    if len({tuple(row.tolist()) for row in out_right if np.any(row != BG)}) != len(colors_by_row):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(3, 4)
    left_w = random.randint(3, 4)
    right_w = random.randint(8, 10)
    inner_h, row_offsets = _rows_for_bars(n_bars)
    outer_h = inner_h + 2
    outer_w = left_w + right_w + 3

    H_lo = max(15, outer_h + 4)
    H_hi = min(28, outer_h + 10)
    W_lo = max(20, outer_w + 6)
    W_hi = min(30, outer_w + 12)
    if H_lo > H_hi or W_lo > W_hi:
        return None

    H = random.randint(H_lo, H_hi)
    W = random.randint(W_lo, W_hi)
    frame_top = random.randint(2, H - outer_h - 2)
    frame_left = random.randint(2, W - outer_w - 2)
    divider = frame_left + 1 + left_w
    frame_bbox = (frame_top, frame_left, frame_top + outer_h - 1, frame_left + outer_w - 1)

    colors = random.sample(BAR_COLORS, n_bars)
    areas = _choose_areas(right_w, n_bars)
    paired = list(zip(row_offsets, colors, areas))
    random.shuffle(paired)
    row_offsets = [row for row, _color, _area in paired]
    colors = [color for _row, color, _area in paired]
    areas = [area for _row, _color, area in paired]

    grid = np.zeros((H, W), dtype=int)
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0 : r1 + 1, divider] = FRAME

    bar_starts: list[int] = []
    colors_by_row = [color for _, color, _ in sorted(zip(row_offsets, colors, areas))]
    areas_by_row = [area for _, _color, area in sorted(zip(row_offsets, colors, areas))]
    for row_offset, color, _area in zip(row_offsets, colors, areas):
        row = r0 + 1 + row_offset
        start = random.randint(0, left_w - 2)
        bar_starts.append(start)
        abs_start = c0 + 1 + start
        grid[row, abs_start : abs_start + 2] = color

    occupied = np.zeros_like(grid, dtype=bool)
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r1 + 2)
    cc1 = min(W, c1 + 2)
    occupied[rr0:rr1, cc0:cc1] = True

    sprite_specs: list[tuple[int, int, np.ndarray, int]] = []
    for row_offset, color, area in zip(row_offsets, colors, areas):
        sprite_specs.append((area, color, _shape_array(area).astype(bool), r0 + 1 + row_offset))
    sprite_specs.sort(reverse=True)
    if not _place_sprites(grid, occupied, sprite_specs):
        return None

    expected = _transform(grid)
    if not _semantic_checks(grid, expected, colors_by_row, areas_by_row, frame_bbox, divider):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
