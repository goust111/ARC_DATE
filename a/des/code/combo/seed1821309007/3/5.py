import random
from collections import deque

import numpy as np


BG = 0
RED = 2
DIR_LABELS = ("left", "right", "up", "down")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, RED))
    for color in colors:
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if len(comp) != h * w:
                continue
            rects.append(
                {
                    "color": color,
                    "cells": comp,
                    "bbox": (r0, c0, r1, c1),
                    "height": h,
                    "width": w,
                }
            )
    return rects


def _detect_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for comp in _components(grid == RED):
        if len(comp) != 3:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        present = set(comp)
        all_cells = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        missing = list(all_cells - present)
        if len(missing) != 1:
            continue
        mr, mc = missing[0]
        arrows.append(
            {
                "cells": comp,
                "bbox": (r0, c0, r1, c1),
                "missing": (mr - r0, mc - c0),
            }
        )
    return arrows


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _classify_arrow(arrow: dict, rects: list[dict]) -> tuple[str, str, list[dict]]:
    r0, c0, r1, c1 = arrow["bbox"]
    row_overlap_rects = [
        rect
        for rect in rects
        if _overlap_1d(rect["bbox"][0], rect["bbox"][2], r0, r1)
    ]
    col_overlap_rects = [
        rect
        for rect in rects
        if _overlap_1d(rect["bbox"][1], rect["bbox"][3], c0, c1)
    ]

    left_count = sum(rect["bbox"][3] < c0 for rect in row_overlap_rects)
    right_count = sum(rect["bbox"][1] > c1 for rect in row_overlap_rects)
    up_count = sum(rect["bbox"][2] < r0 for rect in col_overlap_rects)
    down_count = sum(rect["bbox"][0] > r1 for rect in col_overlap_rects)

    horiz = left_count > 0 and right_count > 0
    vert = up_count > 0 and down_count > 0
    assert horiz != vert, "arrow axis must be unambiguous"

    mr, mc = arrow["missing"]
    if horiz:
        direction = "left" if mc == 0 else "right"
        return "horizontal", direction, row_overlap_rects
    direction = "up" if mr == 0 else "down"
    return "vertical", direction, col_overlap_rects


def _paint_horizontal(out: np.ndarray, rects: list[dict], direction: str) -> None:
    ordered = sorted(rects, key=lambda rect: rect["bbox"][1])
    total = sum(rect["width"] for rect in ordered) + len(ordered) - 1
    start = 0 if direction == "left" else out.shape[1] - total
    span_r0 = min(rect["bbox"][0] for rect in ordered)
    span_r1 = max(rect["bbox"][2] for rect in ordered)

    cursor = start
    for i, rect in enumerate(ordered):
        r0, _c0, r1, _c1 = rect["bbox"]
        out[r0 : r1 + 1, cursor : cursor + rect["width"]] = rect["color"]
        cursor += rect["width"]
        if i + 1 < len(ordered):
            out[span_r0 : span_r1 + 1, cursor] = RED
            cursor += 1


def _paint_vertical(out: np.ndarray, rects: list[dict], direction: str) -> None:
    ordered = sorted(rects, key=lambda rect: rect["bbox"][0])
    total = sum(rect["height"] for rect in ordered) + len(ordered) - 1
    start = 0 if direction == "up" else out.shape[0] - total
    span_c0 = min(rect["bbox"][1] for rect in ordered)
    span_c1 = max(rect["bbox"][3] for rect in ordered)

    cursor = start
    for i, rect in enumerate(ordered):
        r0, c0, _r1, c1 = rect["bbox"]
        height = rect["height"]
        width = rect["width"]
        out[cursor : cursor + height, c0 : c1 + 1] = rect["color"]
        cursor += height
        if i + 1 < len(ordered):
            out[cursor, span_c0 : span_c1 + 1] = RED
            cursor += 1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    rects = _detect_rectangles(g)
    arrows = _detect_arrows(g)
    assert rects and arrows

    out = g.copy()
    moved_ids: set[int] = set()
    arrow_cells: set[tuple[int, int]] = set()

    grouped: list[tuple[str, str, list[dict]]] = []
    for arrow in arrows:
        axis, direction, members = _classify_arrow(arrow, rects)
        assert len(members) >= 2
        grouped.append((axis, direction, members))
        arrow_cells.update(arrow["cells"])
        for rect in members:
            moved_ids.add(id(rect))

    for rect in rects:
        if id(rect) in moved_ids:
            for r, c in rect["cells"]:
                out[r, c] = BG
    for r, c in arrow_cells:
        out[r, c] = BG

    for axis, direction, members in grouped:
        if axis == "horizontal":
            _paint_horizontal(out, members, direction)
        else:
            _paint_vertical(out, members, direction)

    return out


def _arrow_pattern(direction: str, variant: int) -> np.ndarray:
    arr = np.full((2, 2), BG, dtype=int)
    if direction == "left":
        missing = (variant, 0)
    elif direction == "right":
        missing = (variant, 1)
    elif direction == "up":
        missing = (0, variant)
    else:
        missing = (1, variant)
    for r in range(2):
        for c in range(2):
            if (r, c) != missing:
                arr[r, c] = RED
    return arr


def _choose_palette() -> list[int]:
    colors = [1, 3, 4, 5, 6, 7, 8, 9]
    random.shuffle(colors)
    return colors


def _render_horizontal_group(
    inp: np.ndarray,
    out: np.ndarray,
    top: int,
    left: int,
    widths: list[int],
    colors: list[int],
    direction: str,
    variant: int,
) -> None:
    gap_choices = [random.randint(1, 3) for _ in range(len(widths) - 1)]
    arrow_gap = random.randrange(len(widths) - 1)
    gap_choices[arrow_gap] = 2
    x = left
    rect_boxes: list[tuple[int, int, int]] = []
    for i, width in enumerate(widths):
        inp[top : top + 2, x : x + width] = colors[i]
        rect_boxes.append((x, x + width - 1, colors[i]))
        x += width
        if i + 1 < len(widths):
            if i == arrow_gap:
                inp[top : top + 2, x : x + 2] = _arrow_pattern(direction, variant)
            x += gap_choices[i]

    total = sum(widths) + len(widths) - 1
    start = 0 if direction == "left" else inp.shape[1] - total
    cursor = start
    for i, width in enumerate(widths):
        out[top : top + 2, cursor : cursor + width] = colors[i]
        cursor += width
        if i + 1 < len(widths):
            out[top : top + 2, cursor] = RED
            cursor += 1


def _render_vertical_group(
    inp: np.ndarray,
    out: np.ndarray,
    top: int,
    left: int,
    heights: list[int],
    colors: list[int],
    direction: str,
    variant: int,
) -> None:
    gap_choices = [random.randint(1, 3) for _ in range(len(heights) - 1)]
    arrow_gap = random.randrange(len(heights) - 1)
    gap_choices[arrow_gap] = 2
    y = top
    for i, height in enumerate(heights):
        inp[y : y + height, left : left + 2] = colors[i]
        y += height
        if i + 1 < len(heights):
            if i == arrow_gap:
                inp[y : y + 2, left : left + 2] = _arrow_pattern(direction, variant)
            y += gap_choices[i]

    total = sum(heights) + len(heights) - 1
    start = 0 if direction == "up" else inp.shape[0] - total
    cursor = start
    for i, height in enumerate(heights):
        out[cursor : cursor + height, left : left + 2] = colors[i]
        cursor += height
        if i + 1 < len(heights):
            out[cursor, left : left + 2] = RED
            cursor += 1


def _can_place_horizontal(
    occ: np.ndarray, h: int, w: int, top: int, left: int, widths: list[int]
) -> bool:
    if not (0 <= top and top + 2 <= h and 0 <= left and left + sum(widths) + len(widths) + 4 <= w):
        return False
    right = w - 1
    total = sum(widths) + len(widths) - 1
    source_w = sum(widths) + (len(widths) - 2) * 3 + 2
    src_left = left
    src_right = src_left + source_w - 1
    dst_left = 0
    dst_right = total - 1
    if occ[top : top + 2, src_left : src_right + 1].any():
        return False
    if occ[top : top + 2, dst_left : dst_right + 1].any():
        return False
    if occ[top : top + 2, right - total + 1 : right + 1].any():
        return False
    return True


def _can_place_vertical(
    occ: np.ndarray, h: int, w: int, top: int, left: int, heights: list[int]
) -> bool:
    if not (0 <= left and left + 2 <= w and 0 <= top and top + sum(heights) + len(heights) + 4 <= h):
        return False
    bottom = h - 1
    total = sum(heights) + len(heights) - 1
    source_h = sum(heights) + (len(heights) - 2) * 3 + 2
    src_top = top
    src_bottom = src_top + source_h - 1
    dst_top = 0
    dst_bottom = total - 1
    if occ[src_top : src_bottom + 1, left : left + 2].any():
        return False
    if occ[dst_top : dst_bottom + 1, left : left + 2].any():
        return False
    if occ[bottom - total + 1 : bottom + 1, left : left + 2].any():
        return False
    return True


def _mark_horizontal(occ: np.ndarray, top: int, left: int, widths: list[int]) -> None:
    total = sum(widths) + len(widths) - 1
    source_w = sum(widths) + (len(widths) - 2) * 3 + 2
    occ[top : top + 2, left : left + source_w] = True
    occ[top : top + 2, 0:total] = True
    occ[top : top + 2, occ.shape[1] - total : occ.shape[1]] = True


def _mark_vertical(occ: np.ndarray, top: int, left: int, heights: list[int]) -> None:
    total = sum(heights) + len(heights) - 1
    source_h = sum(heights) + (len(heights) - 2) * 3 + 2
    occ[top : top + source_h, left : left + 2] = True
    occ[0:total, left : left + 2] = True
    occ[occ.shape[0] - total : occ.shape[0], left : left + 2] = True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    direction_cycle = list(DIR_LABELS)
    random.shuffle(direction_cycle)
    n_groups = random.randint(1, 3)
    palette = _choose_palette()
    color_idx = 0

    for group_idx in range(n_groups):
        direction = direction_cycle[group_idx % len(direction_cycle)]
        if direction in ("left", "right"):
            n_rects = random.randint(2, 4)
            widths = [random.randint(2, 4) for _ in range(n_rects)]
            needed = sum(widths) + 2 + (n_rects - 2) * 3
            if needed > w - 2:
                return None
            top = random.randint(0, h - 2)
            left = random.randint(1, max(1, w - needed - 1))
            if not _can_place_horizontal(occ, h, w, top, left, widths):
                return None
            colors = palette[color_idx : color_idx + n_rects]
            if len(colors) < n_rects:
                return None
            color_idx += n_rects
            variant = random.randint(0, 1)
            _render_horizontal_group(inp, out, top, left, widths, colors, direction, variant)
            _mark_horizontal(occ, top, left, widths)
        else:
            n_rects = random.randint(2, 4)
            heights = [random.randint(2, 4) for _ in range(n_rects)]
            needed = sum(heights) + 2 + (n_rects - 2) * 3
            if needed > h - 2:
                return None
            top = random.randint(1, max(1, h - needed - 1))
            left = random.randint(0, w - 2)
            if not _can_place_vertical(occ, h, w, top, left, heights):
                return None
            colors = palette[color_idx : color_idx + n_rects]
            if len(colors) < n_rects:
                return None
            color_idx += n_rects
            variant = random.randint(0, 1)
            _render_vertical_group(inp, out, top, left, heights, colors, direction, variant)
            _mark_vertical(occ, top, left, heights)

    if np.count_nonzero(inp == RED) < 3:
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
