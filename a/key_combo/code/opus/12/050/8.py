from __future__ import annotations

import random
from collections import Counter

import numpy as np


BBox = tuple[int, int, int, int]

_BG = 0
_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_RECT_COUNT_CHOICES = [3, 4, 5]
_GRID_SIDE_RANGE = (13, 18)
_SHORT_SIDE_RANGE = (3, 5)
_LONG_SIDE_RANGE = (4, 7)
_MARGIN = 1
_GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _non_bg_colors(grid: np.ndarray, bg: int) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != int(bg)]


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_center(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _bbox_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _paint_solid(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _paint_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _frame_cells(box: BBox) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _intersection_area(a: BBox, b: BBox) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _contains(a: BBox, b: BBox) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _recover_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    bg = _get_bg(grid)
    rects: list[dict[str, object]] = []
    for color in _non_bg_colors(grid, bg):
        coords = np.argwhere(grid == int(color))
        assert coords.size > 0, "each foreground color must remain visible"
        box = _bbox_from_coords(coords)
        h = int(box[2] - box[0] + 1)
        w = int(box[3] - box[1] + 1)
        assert h >= 3 and w >= 3, "rectangles must have a non-empty interior"
        rects.append(
            {
                "color": int(color),
                "bbox": box,
                "center": _bbox_center(box),
                "area": _bbox_area(box),
            }
        )
    rects.sort(key=lambda rect: rect["color"])
    return rects


def _render(shape: tuple[int, int], rects: list[dict[str, object]], *, hollow: bool, low_to_high: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    for rect in low_to_high:
        if hollow:
            _paint_frame(grid, rect["bbox"], int(rect["color"]))
        else:
            _paint_solid(grid, rect["bbox"], int(rect["color"]))
    return grid


def _center_priority_low_to_high(rects: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rects,
        key=lambda rect: (
            int(rect["center"][0]),
            int(rect["center"][1]),
            int(rect["color"]),
        ),
        reverse=True,
    )


def _order_by_color_low_to_high(rects: list[dict[str, object]], *, smaller_color_on_top: bool) -> list[dict[str, object]]:
    return sorted(
        rects,
        key=lambda rect: (int(rect["color"]), int(rect["center"][0]), int(rect["center"][1])),
        reverse=smaller_color_on_top,
    )


def _order_by_area_low_to_high(rects: list[dict[str, object]], *, smaller_area_on_top: bool) -> list[dict[str, object]]:
    return sorted(
        rects,
        key=lambda rect: (int(rect["area"]), int(rect["center"][0]), int(rect["center"][1]), int(rect["color"])),
        reverse=smaller_area_on_top,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn overlapped solid rectangles into hollow frames whose visible depth is determined by center position.

    Input structure:
    - The background is black.
    - Each non-black color belongs to one solid rectangle, and several such rectangles overlap on the same canvas.
    - Every rectangle has a distinct center row and a distinct center column when computed from its bounding box midpoint with floor rounding.

    Rule:
    - For each foreground color, recover its rectangle from the bounding box of all visible pixels of that color.
    - Erase the rectangle interior, leaving only the one-pixel border frame.
    - Compute each frame center `(cr, cc)` from its bbox and sort centers lexicographically by row then column; earlier centers are higher priority.
    - Draw frames from lowest priority to highest priority so later, higher-priority frames overwrite earlier ones at shared border pixels.

    Key insight:
    - The hidden ordering signal is the frame center's position relative to the upper-left corner, not the color id or the rectangle size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg == _BG, "expected black background"
    rects = _recover_rectangles(g)
    assert 3 <= len(rects) <= 5, "expected 3 to 5 rectangles"

    centers = [tuple(map(int, rect["center"])) for rect in rects]
    center_rows = [cr for cr, _cc in centers]
    center_cols = [cc for _cr, cc in centers]
    assert len(set(centers)) == len(centers), "rectangle centers must be unique"
    assert len(set(center_rows)) == len(rects), "center rows must be distinct"
    assert len(set(center_cols)) == len(rects), "center cols must be distinct"

    return _render(g.shape, rects, hollow=True, low_to_high=_center_priority_low_to_high(rects))


def _sample_size() -> tuple[int, int]:
    short = random.randint(*_SHORT_SIDE_RANGE)
    long = random.randint(*_LONG_SIDE_RANGE)
    if random.random() < 0.5:
        return short, long
    return long, short


def _sample_rectangles(shape: tuple[int, int], n_rects: int) -> list[dict[str, object]]:
    H, W = shape
    colors = random.sample(_COLOR_POOL, n_rects)
    rects: list[dict[str, object]] = []
    for color in colors:
        h, w = _sample_size()
        max_r0 = H - h - _MARGIN
        max_c0 = W - w - _MARGIN
        if max_r0 < _MARGIN or max_c0 < _MARGIN:
            return []
        r0 = random.randint(_MARGIN, max_r0)
        c0 = random.randint(_MARGIN, max_c0)
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        rects.append(
            {
                "color": int(color),
                "bbox": box,
                "center": _bbox_center(box),
                "area": _bbox_area(box),
            }
        )
    return rects


def _visible_bbox_matches_original(inp: np.ndarray, rects: list[dict[str, object]]) -> bool:
    try:
        recovered_rects = _recover_rectangles(inp)
    except AssertionError:
        return False
    recovered = {int(rect["color"]): rect["bbox"] for rect in recovered_rects}
    if set(recovered) != {int(rect["color"]) for rect in rects}:
        return False
    for rect in rects:
        if recovered[int(rect["color"])] != rect["bbox"]:
            return False
    return True


def _visible_side_witnesses(inp: np.ndarray, rect: dict[str, object]) -> bool:
    color = int(rect["color"])
    r0, c0, r1, c1 = rect["bbox"]
    return (
        bool(np.any(inp[r0, c0 : c1 + 1] == color))
        and bool(np.any(inp[r1, c0 : c1 + 1] == color))
        and bool(np.any(inp[r0 : r1 + 1, c0] == color))
        and bool(np.any(inp[r0 : r1 + 1, c1] == color))
    )


def _overlap_pairs(rects: list[dict[str, object]]) -> int:
    pairs = 0
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _intersection_area(rects[i]["bbox"], rects[j]["bbox"]) > 0:
                pairs += 1
    return pairs


def _total_overlap_area(rects: list[dict[str, object]]) -> int:
    total = 0
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            total += _intersection_area(rects[i]["bbox"], rects[j]["bbox"])
    return total


def _frame_conflict_pixels(rects: list[dict[str, object]]) -> int:
    counts: Counter[tuple[int, int]] = Counter()
    for rect in rects:
        for cell in _frame_cells(rect["bbox"]):
            counts[cell] += 1
    return sum(1 for count in counts.values() if count >= 2)


def _no_containment(rects: list[dict[str, object]]) -> bool:
    for i in range(len(rects)):
        for j in range(len(rects)):
            if i == j:
                continue
            if _contains(rects[i]["bbox"], rects[j]["bbox"]):
                return False
    return True


def _centers_unambiguous(rects: list[dict[str, object]]) -> bool:
    centers = [tuple(map(int, rect["center"])) for rect in rects]
    if len(set(centers)) != len(centers):
        return False
    rows = [cr for cr, _ in centers]
    cols = [cc for _, cc in centers]
    return len(set(rows)) == len(rows) and len(set(cols)) == len(cols)


def _visual_clarity_ok(shape: tuple[int, int], rects: list[dict[str, object]], inp: np.ndarray) -> bool:
    H, W = shape
    occupied = int(np.count_nonzero(inp))
    if occupied > int(0.62 * H * W):
        return False
    if _total_overlap_area(rects) > int(0.22 * H * W):
        return False
    if _frame_conflict_pixels(rects) > 10:
        return False
    return True


def _semantic_order_is_unique(shape: tuple[int, int], rects: list[dict[str, object]], expected: np.ndarray) -> bool:
    alternatives = [
        _render(shape, rects, hollow=True, low_to_high=_order_by_color_low_to_high(rects, smaller_color_on_top=True)),
        _render(shape, rects, hollow=True, low_to_high=_order_by_color_low_to_high(rects, smaller_color_on_top=False)),
        _render(shape, rects, hollow=True, low_to_high=_order_by_area_low_to_high(rects, smaller_area_on_top=True)),
        _render(shape, rects, hollow=True, low_to_high=_order_by_area_low_to_high(rects, smaller_area_on_top=False)),
    ]
    return all(not np.array_equal(expected, alt) for alt in alternatives)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_GRID_SIDE_RANGE)
    W = random.randint(*_GRID_SIDE_RANGE)
    n_rects = random.choice(_RECT_COUNT_CHOICES)
    rects = _sample_rectangles((H, W), n_rects)
    if len(rects) != n_rects:
        return None
    if not _centers_unambiguous(rects):
        return None
    if not _no_containment(rects):
        return None
    if _overlap_pairs(rects) < 2:
        return None
    if _frame_conflict_pixels(rects) == 0:
        return None

    input_order = _center_priority_low_to_high(rects)
    inp = _render((H, W), rects, hollow=False, low_to_high=input_order)
    if _get_bg(inp) != _BG:
        return None

    expected = _render((H, W), rects, hollow=True, low_to_high=input_order)
    if not _visible_bbox_matches_original(inp, rects):
        return None
    if any(not _visible_side_witnesses(inp, rect) for rect in rects):
        return None
    if any(int(np.count_nonzero(inp == int(rect["color"]))) < max(6, int(rect["area"]) // 4) for rect in rects):
        return None
    if not _visual_clarity_ok((H, W), rects, inp):
        return None
    if not _semantic_order_is_unique((H, W), rects, expected):
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
