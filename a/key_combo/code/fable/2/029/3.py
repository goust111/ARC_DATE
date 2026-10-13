from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, get_bg


Rect = tuple[int, int, int, int]


def _paint_rect(grid: np.ndarray, box: Rect, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _intersection_area(a: Rect, b: Rect) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _recover_rectangles(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = get_bg(grid)
    items: list[dict] = []
    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == int(color))
        assert coords.size > 0, "every foreground color must remain visible"
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box

        # The task guarantee says each true edge is still visible somewhere.
        assert np.any(coords[:, 0] == r0), "top edge must remain visible"
        assert np.any(coords[:, 0] == r1), "bottom edge must remain visible"
        assert np.any(coords[:, 1] == c0), "left edge must remain visible"
        assert np.any(coords[:, 1] == c1), "right edge must remain visible"

        items.append(
            {
                "color": int(color),
                "bbox": box,
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "visible_area": int(coords.shape[0]),
            }
        )

    assert items, "expected at least one rectangle"
    areas = [item["area"] for item in items]
    assert len(areas) == len(set(areas)), "true rectangle areas must be unique"
    return int(bg), items


def _render_by_order(shape: tuple[int, int], bg: int, items: list[dict], order: list[int]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for idx in order:
        item = items[int(idx)]
        _paint_rect(out, item["bbox"], item["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the hidden full rectangles and redraw them in descending true-area layer order.

    Input structure:
    - The background is the most common color.
    - Each remaining color belongs to one solid axis-aligned rectangle, but rectangles overlap and upper layers hide lower ones.
    - Every rectangle still exposes at least one visible pixel on each of its four true edges, so the visible color extrema recover its full bbox.

    Rule:
    - For each foreground color, take the bbox of all currently visible pixels of that color as the rectangle's true bbox.
    - Compute each rectangle's true area from that recovered bbox.
    - Keep all rectangle positions fixed, then redraw them from largest true area first to smallest true area last.

    Key insight:
    - Visible pixel count is a decoy: occlusion changes what remains exposed, but the visible extrema still reveal each rectangle's full size, and that recovered area is the real layering key.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, items = _recover_rectangles(g)
    order = sorted(range(len(items)), key=lambda idx: items[idx]["area"], reverse=True)
    return _render_by_order(g.shape, bg, items, order)


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
RECT_COUNT_RANGE = (3, 5)
CORE_HEIGHT_RANGE = (3, 5)
CORE_WIDTH_RANGE = (3, 5)
RECT_HEIGHT_RANGE = (5, 13)
RECT_WIDTH_RANGE = (5, 13)
MAX_SCENE_ATTEMPTS = 1200
MAX_RECT_ATTEMPTS = 80
MAX_INPUT_ATTEMPTS = 300
MIN_BACKGROUND_RATIO = 0.38
ANCHOR_LEVELS = (0.0, 0.5, 1.0)
PALETTE = tuple(range(1, 10))


def _choose_anchor_value(lo: int, hi: int, level: float) -> int:
    if lo == hi:
        return int(lo)
    pos = lo + int(round((hi - lo) * float(level)))
    return int(min(max(pos, lo), hi))


def _sample_rectangles(
    height: int,
    width: int,
    count: int,
) -> list[dict] | None:
    core_h = random.randint(*CORE_HEIGHT_RANGE)
    core_w = random.randint(*CORE_WIDTH_RANGE)
    core_r0 = random.randint(2, height - core_h - 3)
    core_c0 = random.randint(2, width - core_w - 3)
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    colors = random.sample(PALETTE, count + 1)
    bg = colors[0]
    rect_colors = colors[1:]

    anchor_pairs = [(vr, vc) for vr in ANCHOR_LEVELS for vc in ANCHOR_LEVELS]
    random.shuffle(anchor_pairs)

    items: list[dict] = []
    used_areas: set[int] = set()
    used_boxes: set[Rect] = set()

    for idx, color in enumerate(rect_colors):
        anchor_r, anchor_c = anchor_pairs[idx]
        made = False
        for _ in range(MAX_RECT_ATTEMPTS):
            rect_h = random.randint(*RECT_HEIGHT_RANGE)
            rect_w = random.randint(*RECT_WIDTH_RANGE)
            if rect_h > height - 2 or rect_w > width - 2:
                continue

            r_lo = max(1, core_r1 - rect_h + 1)
            r_hi = min(core_r0, height - rect_h - 1)
            c_lo = max(1, core_c1 - rect_w + 1)
            c_hi = min(core_c0, width - rect_w - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue

            r0 = _choose_anchor_value(r_lo, r_hi, anchor_r)
            c0 = _choose_anchor_value(c_lo, c_hi, anchor_c)
            if r_hi > r_lo:
                r0 = int(min(max(r0 + random.randint(-1, 1), r_lo), r_hi))
            if c_hi > c_lo:
                c0 = int(min(max(c0 + random.randint(-1, 1), c_lo), c_hi))

            box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            area = rect_h * rect_w
            if area in used_areas or box in used_boxes:
                continue

            items.append({"color": int(color), "bbox": box, "area": int(area)})
            used_areas.add(int(area))
            used_boxes.add(box)
            made = True
            break

        if not made:
            return None

    return [{"color": int(bg), "bbox": (-1, -1, -1, -1), "area": 0}] + items


def _render_scene(
    shape: tuple[int, int],
    bg: int,
    items: list[dict],
    order: list[int],
) -> np.ndarray:
    canvas = np.full(shape, int(bg), dtype=int)
    for idx in order:
        item = items[int(idx)]
        _paint_rect(canvas, item["bbox"], item["color"])
    return canvas


def _sort_by_visible_area(grid: np.ndarray) -> np.ndarray:
    bg, items = _recover_rectangles(grid)
    visible_areas = [item["visible_area"] for item in items]
    if len(set(visible_areas)) != len(visible_areas):
        raise AssertionError("visible counts must be unique in generated scenes")
    order = sorted(range(len(items)), key=lambda idx: items[idx]["visible_area"], reverse=True)
    return _render_by_order(grid.shape, bg, items, order)


def _visible_edge_ok(grid: np.ndarray, color: int, box: Rect) -> bool:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0 or tuple(map(int, bbox_coords(coords))) != tuple(map(int, box)):
        return False
    r0, c0, r1, c1 = box
    return bool(
        np.any(coords[:, 0] == r0)
        and np.any(coords[:, 0] == r1)
        and np.any(coords[:, 1] == c0)
        and np.any(coords[:, 1] == c1)
    )


def _scene_is_valid(shape: tuple[int, int], bg: int, rects: list[dict], order: list[int]) -> bool:
    grid = _render_scene(shape, bg, rects, order)
    area_desc = sorted(range(len(rects)), key=lambda idx: rects[idx]["area"], reverse=True)
    expected = _render_scene(shape, bg, rects, area_desc)

    if np.array_equal(grid, expected):
        return False
    if get_bg(grid) != bg:
        return False
    if np.count_nonzero(grid == bg) <= int(grid.size * MIN_BACKGROUND_RATIO):
        return False

    overlaps = 0
    hidden = 0
    visible_areas = []
    for item in rects:
        box = item["bbox"]
        color = item["color"]
        if not _visible_edge_ok(grid, color, box):
            return False
        visible_area = int(np.sum(grid == color))
        visible_areas.append(visible_area)
        if visible_area < item["area"]:
            hidden += 1

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _intersection_area(rects[i]["bbox"], rects[j]["bbox"]) > 0:
                overlaps += 1
    if overlaps == 0 or hidden == 0:
        return False

    if len(set(visible_areas)) != len(visible_areas):
        return False
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False
    try:
        wrong = _sort_by_visible_area(grid)
    except AssertionError:
        return False
    if np.array_equal(wrong, expected):
        return False

    # At least one true-area inversion relative to the observed input layer stack.
    if list(order) == area_desc:
        return False
    return True


def _build_scene() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    count = random.randint(*RECT_COUNT_RANGE)

    packed = _sample_rectangles(height, width, count)
    if packed is None:
        return None

    bg = packed[0]["color"]
    rects = packed[1:]
    order = list(range(len(rects)))
    random.shuffle(order)
    if not _scene_is_valid((height, width), bg, rects, order):
        return None
    return _render_scene((height, width), bg, rects, order)


def generate_input() -> np.ndarray:
    for _ in range(MAX_INPUT_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occluded-rectangle ordering task")
