from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
RECT_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE)
MAX_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_dividers(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    divs = []
    for c in range(1, w):
        if np.all(grid[1:, c] == GREY) and int(grid[0, c]) in (BLUE, GREY):
            divs.append(int(c))
    assert len(divs) == 2
    return divs


def _panel_ranges(grid: np.ndarray) -> list[tuple[int, int]]:
    d1, d2 = _find_dividers(grid)
    panels = [(1, d1 - 1), (d1 + 1, d2 - 1), (d2 + 1, grid.shape[1] - 1)]
    assert all(c0 <= c1 for c0, c1 in panels)
    widths = [c1 - c0 + 1 for c0, c1 in panels]
    assert len(set(widths)) == 1
    return panels


def _compress_panel(panel: np.ndarray) -> tuple[np.ndarray, list[int], dict[int, int]]:
    kept_rows = [int(r) for r in range(panel.shape[0]) if np.any(panel[r] != BG)]
    compressed = np.full_like(panel, BG)
    row_map: dict[int, int] = {}
    start = panel.shape[0] - len(kept_rows)
    for idx, src_r in enumerate(kept_rows):
        dst_r = int(start + idx)
        compressed[dst_r] = panel[src_r]
        row_map[int(src_r)] = dst_r
    return compressed, kept_rows, row_map


def _extract_rectangles(panel: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for coords in _components_mask(panel != BG):
        bbox = _bbox_coords(coords)
        arr = _crop_bbox(panel, bbox)
        color = int(arr[0, 0])
        assert color != BG
        assert np.all(arr == color)
        rects.append(
            {
                "bbox": bbox,
                "arr": arr,
                "color": color,
                "h": int(arr.shape[0]),
                "w": int(arr.shape[1]),
            }
        )
    rects.sort(key=lambda rect: (rect["bbox"][0], rect["bbox"][1]))
    return rects


def _build_output(rects: list[dict], rotate_key: str, order_key: str) -> np.ndarray:
    ordered = sorted(
        rects,
        key=lambda rect: (
            rect[order_key],
            rect["panel_index"],
            rect["bbox_global"][0],
            rect["bbox_global"][1],
            rect["color"],
        ),
    )
    patches: list[np.ndarray] = []
    for rect in ordered:
        arr = rect["arr"]
        if rect[rotate_key]:
            arr = np.rot90(arr, 1)
        patches.append(arr)
    out_h = int(sum(arr.shape[0] for arr in patches))
    out_w = int(max(arr.shape[1] for arr in patches))
    out = np.full((out_h, out_w), BG, dtype=int)
    cursor = 0
    for arr in patches:
        h, w = arr.shape
        left = (out_w - w) // 2
        out[cursor : cursor + h, left : left + w] = arr
        cursor += h
    return out


def _analyze(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 6 and w >= 10
    assert int(g[0, 0]) == MAGENTA
    assert np.all(g[1:, 0] == RED)

    rects: list[dict] = []
    for panel_index, (c0, c1) in enumerate(_panel_ranges(g)):
        panel = g[1:, c0 : c1 + 1]
        compressed, kept_rows, row_map = _compress_panel(panel)
        _ = compressed
        comp_rects = _extract_rectangles(panel)
        panel_center = (c0 + c1) / 2.0
        panel_rotate = bool(panel_center > (w / 2.0))
        for rect in comp_rects:
            r0, lc0, r1, lc1 = rect["bbox"]
            bbox_global = (r0 + 1, lc0 + c0, r1 + 1, lc1 + c0)
            rect_center = (bbox_global[1] + bbox_global[3]) / 2.0
            rect.update(
                {
                    "panel_index": int(panel_index),
                    "panel_cols": (int(c0), int(c1)),
                    "panel_center": float(panel_center),
                    "panel_rotate": panel_rotate,
                    "bbox_global": tuple(int(v) for v in bbox_global),
                    "blue_dist": int(r0 + 1),
                    "compressed_top": int(row_map[r0] + 1),
                    "rect_center_rotate": bool(rect_center > (w / 2.0)),
                    "kept_rows": tuple(kept_rows),
                }
            )
            rects.append(rect)
    assert rects
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each panel vertically, then sort and stack the rectangles using the original axis distances.

    Input structure:
    - The background is black.
    - A blue top edge line and a red left edge line meet at a magenta top-left corner.
    - Two full-height grey divider columns split the interior into three equal-width panels.
    - Each panel contains one or two solid monochrome rectangles separated by background-only rows.

    Rule:
    - Treat the top blue line and left red line as coordinate axes, and record every rectangle's original distance to the blue axis before any compression.
    - Compress each panel independently by discarding all-empty interior rows and packing the remaining rows to the panel bottom.
    - Sort all rectangles by their original distance to the blue axis.
    - Rotate every rectangle from a panel whose center column lies farther than half the grid width from the red axis by 90 degrees counterclockwise.
    - Stack the sorted rectangles top-to-bottom in a new black grid, centered horizontally.

    Key insight:
    - The visible panel compression is a decoy for ordering: it changes where rectangles appear afterward, but the stack order still comes from the pre-compression distances, while rotation is decided once per panel from the panel's axis position.
    """
    rects = _analyze(np.asarray(grid, dtype=int))
    return _build_output(rects, rotate_key="panel_rotate", order_key="blue_dist")


def _sample_counts() -> list[int]:
    return random.choice(([2, 2, 1], [1, 2, 2], [2, 2, 2]))


def _random_partition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    prev = -1
    out = []
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _place_vertical(interior_h: int, heights: list[int]) -> list[tuple[int, int]]:
    count = len(heights)
    min_needed = sum(heights) + max(0, count - 1)
    if min_needed > interior_h:
        raise ValueError("panel too short")
    extra = interior_h - min_needed
    pads = _random_partition(extra, count + 1)
    top = pads[0]
    boxes: list[tuple[int, int]] = []
    for idx, height in enumerate(heights):
        r0 = int(top)
        r1 = int(r0 + height - 1)
        boxes.append((r0, r1))
        top = r1 + 1
        if idx < count - 1:
            top += 1 + pads[idx + 1]
    return boxes


def _choose_panel_widths(panel_w: int, count: int, panel_index: int) -> list[int]:
    hi = min(6, panel_w)
    if panel_index == 1 and count == 2:
        return [random.randint(3, min(4, hi)), random.randint(3, min(4, hi))]
    return [random.randint(3, hi) for _ in range(count)]


def _choose_lefts(panel_w: int, widths: list[int], panel_index: int) -> list[int]:
    if panel_index == 1 and len(widths) == 2:
        left_width = widths[0]
        right_width = widths[1]
        left_pos = 0
        right_pos = panel_w - right_width
        if left_pos + (left_width - 1) / 2.0 >= (panel_w - 1) / 2.0:
            left_pos = max(0, left_pos - 1)
        if right_pos + (right_width - 1) / 2.0 <= (panel_w - 1) / 2.0:
            right_pos = min(panel_w - right_width, right_pos + 1)
        return [left_pos, right_pos]
    return [random.randint(0, panel_w - width) for width in widths]


def _render_instance(
    h: int,
    panel_w: int,
    panel_specs: list[list[dict]],
) -> np.ndarray:
    w = 1 + 3 * panel_w + 2
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MAGENTA
    divs = [1 + panel_w, 1 + 2 * panel_w + 1]
    for dc in divs:
        grid[:, dc] = GREY
    panel_ranges = [(1, panel_w), (panel_w + 2, 2 * panel_w + 1), (2 * panel_w + 3, 3 * panel_w + 2)]
    for panel_index, rects in enumerate(panel_specs):
        c0, _c1 = panel_ranges[panel_index]
        for rect in rects:
            rr0 = rect["top"]
            rr1 = rr0 + rect["h"]
            cc0 = c0 + rect["left"]
            cc1 = cc0 + rect["w"]
            grid[rr0:rr1, cc0:cc1] = rect["color"]
    return grid


def _make_attempt() -> np.ndarray | None:
    h = random.randint(12, 18)
    panel_w = random.randint(6, 8)
    interior_h = h - 1
    counts = _sample_counts()
    total_rects = sum(counts)
    colors = random.sample(list(RECT_COLORS), total_rects)
    color_iter = iter(colors)

    panel_specs: list[list[dict]] = []
    for panel_index, count in enumerate(counts):
        widths = _choose_panel_widths(panel_w, count, panel_index)
        heights = [random.randint(2, 4) for _ in range(count)]
        try:
            row_boxes = _place_vertical(interior_h, heights)
        except ValueError:
            return None
        lefts = _choose_lefts(panel_w, widths, panel_index)
        rects = []
        for idx in range(count):
            top_local, _bottom_local = row_boxes[idx]
            rects.append(
                {
                    "top": int(top_local + 1),
                    "left": int(lefts[idx]),
                    "h": int(heights[idx]),
                    "w": int(widths[idx]),
                    "color": int(next(color_iter)),
                }
            )
        panel_specs.append(rects)

    grid = _render_instance(h, panel_w, panel_specs)
    rects = _analyze(grid)
    if len(rects) != total_rects:
        return None

    blue_dists = [rect["blue_dist"] for rect in rects]
    if len(set(blue_dists)) != len(blue_dists):
        return None

    counts_by_panel = [sum(rect["panel_index"] == idx for rect in rects) for idx in range(3)]
    if sum(count == 2 for count in counts_by_panel) < 2:
        return None

    correct = _build_output(rects, rotate_key="panel_rotate", order_key="blue_dist")
    if max(correct.shape) > 30:
        return None

    wrong_compressed = _build_output(rects, rotate_key="panel_rotate", order_key="compressed_top")
    if np.array_equal(correct, wrong_compressed):
        return None

    wrong_object_rotate = _build_output(rects, rotate_key="rect_center_rotate", order_key="blue_dist")
    if np.array_equal(correct, wrong_object_rotate):
        return None

    panel2 = [rect for rect in rects if rect["panel_index"] == 1]
    if len(panel2) != 2:
        return None
    if any(rect["panel_rotate"] for rect in panel2):
        return None
    if len({rect["rect_center_rotate"] for rect in panel2}) != 2:
        return None

    if not any(rect["panel_rotate"] for rect in rects):
        return None
    if not any(not rect["panel_rotate"] for rect in rects):
        return None

    if not np.array_equal(main(grid), correct):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for panel-compress stack puzzle")
