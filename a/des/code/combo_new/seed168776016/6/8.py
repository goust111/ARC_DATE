from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _box_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _intersection_box(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    total = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            total += 1
            stack = [(int(r), int(c))]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
    return int(total)


def _occupied_interval(line: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(line)
    assert cols.size > 0
    start = int(cols[0])
    end = int(cols[-1])
    assert end - start + 1 == int(cols.size), "rows and columns must stay contiguous"
    return start, end


def _compress_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int, tuple[int, int]]]:
    assert intervals
    runs: list[tuple[int, int, tuple[int, int]]] = []
    start = 0
    current = intervals[0]
    for idx in range(1, len(intervals) + 1):
        if idx == len(intervals) or intervals[idx] != current:
            runs.append((int(start), int(idx - 1), current))
            if idx < len(intervals):
                start = idx
                current = intervals[idx]
    return runs


def _center_col(box: tuple[int, int, int, int]) -> int:
    return int((int(box[1]) + int(box[3])) // 2)


def _center_row(box: tuple[int, int, int, int]) -> int:
    return int((int(box[0]) + int(box[2])) // 2)


def _decompose_staggered_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, sub.shape)
    assert h >= 3 and w >= 3
    assert np.all(sub.any(axis=1)) and np.all(sub.any(axis=0))

    row_intervals = [_occupied_interval(sub[rr, :]) for rr in range(h)]
    col_intervals = [_occupied_interval(sub[:, cc]) for cc in range(w)]
    row_runs = _compress_intervals(row_intervals)
    col_runs = _compress_intervals(col_intervals)
    assert len(row_runs) == 3 and len(col_runs) == 3

    top_start, top_end, top_int = row_runs[0]
    mid_start, mid_end, mid_int = row_runs[1]
    bot_start, bot_end, bot_int = row_runs[2]
    left_start, left_end, left_int = col_runs[0]
    midc_start, midc_end, midc_int = col_runs[1]
    right_start, right_end, right_int = col_runs[2]

    assert top_start == 0 and top_end < mid_start <= mid_end < bot_start and bot_end == h - 1
    assert left_start == 0 and left_end < midc_start <= midc_end < right_start and right_end == w - 1
    assert mid_int == (0, w - 1)
    assert midc_int == (0, h - 1)
    assert top_int != bot_int

    candidates: list[list[tuple[int, int, int, int]]] = []

    if top_int[0] == 0 and bot_int[1] == w - 1:
        rect_top = (int(left_int[0]), int(top_int[0]), int(left_int[1]), int(top_int[1]))
        rect_bottom = (int(right_int[0]), int(bot_int[0]), int(right_int[1]), int(bot_int[1]))
        candidates.append([rect_top, rect_bottom])

    if top_int[1] == w - 1 and bot_int[0] == 0:
        rect_top = (int(right_int[0]), int(top_int[0]), int(right_int[1]), int(top_int[1]))
        rect_bottom = (int(left_int[0]), int(bot_int[0]), int(left_int[1]), int(bot_int[1]))
        candidates.append([rect_top, rect_bottom])

    valid: list[list[tuple[int, int, int, int]]] = []
    for rects in candidates:
        local_a, local_b = rects
        inter = _intersection_box(local_a, local_b)
        if inter is None:
            continue
        if _box_area(inter) == _box_area(local_a) or _box_area(inter) == _box_area(local_b):
            continue
        rebuilt = _box_mask((h, w), local_a) | _box_mask((h, w), local_b)
        if np.array_equal(rebuilt, sub):
            valid.append(rects)

    assert len(valid) == 1
    chosen = valid[0]
    return [
        (int(r0 + rr0), int(c0 + cc0), int(r0 + rr1), int(c0 + cc1))
        for rr0, cc0, rr1, cc1 in chosen
    ]


def _legend_columns(grid: np.ndarray) -> dict[int, list[int]]:
    legend: dict[int, list[int]] = {}
    for col, value in enumerate(map(int, grid[-1, :])):
        if value != BG:
            legend.setdefault(int(value), []).append(int(col))
    assert legend
    for cols in legend.values():
        assert len(cols) == 2
    return legend


def _marker_blocks(
    grid: np.ndarray,
    shape_colors: set[int],
) -> list[tuple[int, tuple[int, int, int, int]]]:
    markers: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in sorted(int(v) for v in np.unique(grid[:-1, :]) if int(v) not in shape_colors and int(v) != BG):
        mask = grid[:-1, :] == int(color)
        assert int(mask.sum()) == 4
        assert _component_count(mask) == 1
        box = _bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        assert r1 - r0 == 1 and c1 - c0 == 1
        assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1])
        markers.append((int(color), box))
    return markers


def _min_distance(mask_a: np.ndarray, box_b: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box_b
    block_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    coords = np.argwhere(mask_a)
    best = None
    for rr, cc in coords:
        for br, bc in block_cells:
            dist = abs(int(rr) - int(br)) + abs(int(cc) - int(bc))
            if best is None or dist < best:
                best = dist
    assert best is not None
    return int(best)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hidden rectangle intersection with its nearby marker color and erase the smaller rectangle's exclusive pixels.

    Input structure:
    - The background is black.
    - Each active color above the last row forms one 4-connected union made from two same-colored, partially overlapping solid rectangles whose seam is hidden by the shared color.
    - A separate 2x2 block of another color sits nearest to that union and acts as the group's intersection color.
    - The bottom row contains exactly two legend pixels of each shape color; they identify which colors are shape unions and are cleared in the output.

    Rule:
    - Read the bottom-row legend to identify the shape colors, then ignore that legend row while analyzing the objects.
    - Pair every shape union with the nearest 2x2 marker block of a different color.
    - Reconstruct the two hidden rectangles from the union's three-phase row and column interval profile, compute their overlap, recolor that overlap with the paired marker color, and compare the two rectangles' non-overlap areas.
    - Erase the smaller rectangle's exclusive pixels to black, keep the larger rectangle's exclusive pixels in the original shape color, and clear the entire bottom legend row to black.

    Key insight:
    - Although the overlap is invisible, the union of two diagonally staggered rectangles has a distinctive top/full/bottom and left/full/right interval structure, which lets the solver recover both rectangles before doing the area-based keep/erase decision.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    legend = _legend_columns(g)
    shape_colors = set(legend)
    markers = _marker_blocks(g, shape_colors)

    group_masks: dict[int, np.ndarray] = {}
    rects_by_color: dict[int, list[tuple[int, int, int, int]]] = {}
    for color in shape_colors:
        mask = g[:-1, :] == int(color)
        assert mask.any()
        assert _component_count(mask) == 1
        rects = _decompose_staggered_rectangles(mask)
        rects_by_color[int(color)] = rects
        full_mask = np.zeros_like(g, dtype=bool)
        full_mask[:-1, :] = mask
        group_masks[int(color)] = full_mask

    marker_for_color: dict[int, int] = {}
    for color, mask in group_masks.items():
        distances = [(_min_distance(mask, box), marker_color) for marker_color, box in markers]
        distances.sort()
        assert len(distances) >= 1
        if len(distances) > 1:
            assert distances[0][0] < distances[1][0]
        marker_for_color[int(color)] = int(distances[0][1])

    out[-1, :] = BG
    for color in sorted(shape_colors):
        rects = rects_by_color[int(color)]
        rects = sorted(rects, key=_center_col)
        left_rect, right_rect = rects
        legend_cols = sorted(legend[int(color)])
        assert _center_col(left_rect) <= _center_col(right_rect)
        assert legend_cols[0] <= legend_cols[1]

        inter = _intersection_box(left_rect, right_rect)
        assert inter is not None
        inter_mask = _box_mask(g.shape, inter)
        left_mask = _box_mask(g.shape, left_rect)
        right_mask = _box_mask(g.shape, right_rect)
        left_only = left_mask & ~inter_mask
        right_only = right_mask & ~inter_mask
        left_area = int(left_only.sum())
        right_area = int(right_only.sum())

        out[inter_mask] = marker_for_color[int(color)]
        if left_area <= right_area:
            out[left_only] = BG
        if right_area < left_area:
            out[right_only] = BG

    return out


def _build_group(diag: str, smaller_side: str) -> dict[str, object] | None:
    for _ in range(200):
        top_h = random.randint(3, 5)
        bottom_h = random.randint(3, 5)
        top_w = random.randint(3, 6)
        bottom_w = random.randint(3, 6)
        overlap_h = random.randint(1, min(top_h, bottom_h) - 1)
        overlap_w = random.randint(1, min(top_w, bottom_w) - 1)

        union_h = top_h + bottom_h - overlap_h
        union_w = top_w + bottom_w - overlap_w
        if union_h > 8 or union_w > 10:
            continue

        top_area = top_h * top_w - overlap_h * overlap_w
        bottom_area = bottom_h * bottom_w - overlap_h * overlap_w
        if top_area == bottom_area:
            continue
        if smaller_side == "left" and not (
            (diag == "lr" and top_area < bottom_area) or (diag == "rl" and bottom_area < top_area)
        ):
            continue
        if smaller_side == "right" and not (
            (diag == "lr" and bottom_area < top_area) or (diag == "rl" and top_area < bottom_area)
        ):
            continue

        if diag == "lr":
            top_box = (0, 0, top_h - 1, top_w - 1)
            bottom_box = (top_h - overlap_h, top_w - overlap_w, union_h - 1, union_w - 1)
        else:
            top_box = (0, bottom_w - overlap_w, top_h - 1, union_w - 1)
            bottom_box = (top_h - overlap_h, 0, union_h - 1, bottom_w - 1)

        union = _box_mask((union_h, union_w), top_box) | _box_mask((union_h, union_w), bottom_box)
        if _component_count(union) != 1:
            continue
        if _decompose_staggered_rectangles(union) is None:
            continue

        rects = sorted([top_box, bottom_box], key=_center_col)
        if _center_col(rects[0]) == _center_col(rects[1]):
            continue
        return {
            "union_h": int(union_h),
            "union_w": int(union_w),
            "rects_local": [(int(a), int(b), int(c), int(d)) for a, b, c, d in (top_box, bottom_box)],
        }
    return None


def _place_groups(groups: list[dict[str, object]]) -> tuple[int, int, list[dict[str, object]]] | None:
    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 2)
    gaps = [random.randint(2, 4) for _ in range(max(0, len(groups) - 1))]
    total_h = top_pad + bottom_pad + 1 + sum(int(g["union_h"]) for g in groups) + sum(gaps)
    if total_h > 30:
        return None

    placed: list[dict[str, object]] = []
    for group in groups:
        placed.append(dict(group))

    current_r = top_pad
    min_width = 0
    for idx, group in enumerate(placed):
        group["marker_side"] = random.choice(["left", "right"])
        group["marker_gap"] = random.randint(1, 3)
        required = int(group["union_w"]) + int(group["marker_gap"]) + 2
        min_width = max(min_width, required + 2)
        group["union_r0"] = int(current_r)
        current_r += int(group["union_h"])
        if idx < len(gaps):
            current_r += int(gaps[idx])

    legend_need = 2 * len(groups) + max(0, len(groups) - 1)
    min_width = max(min_width, legend_need + 2)
    if min_width > 30:
        return None

    width = random.randint(min_width, min(30, min_width + 6))
    left_pad = random.randint(1, 2)
    right_pad = random.randint(1, 2)
    if left_pad + right_pad + 3 > width:
        return None

    for group in placed:
        union_w = int(group["union_w"])
        marker_gap = int(group["marker_gap"])
        if group["marker_side"] == "left":
            low = left_pad + marker_gap + 2
            high = width - right_pad - union_w
        else:
            low = left_pad
            high = width - right_pad - union_w - marker_gap - 2
        if low > high:
            return None
        union_c0 = random.randint(low, high)
        group["union_c0"] = int(union_c0)
        union_r0 = int(group["union_r0"])
        marker_r0 = union_r0 + random.randint(0, int(group["union_h"]) - 2)
        if group["marker_side"] == "left":
            marker_c0 = union_c0 - marker_gap - 2
        else:
            marker_c0 = union_c0 + union_w + marker_gap
        group["marker_box"] = (int(marker_r0), int(marker_c0), int(marker_r0 + 1), int(marker_c0 + 1))

        rects_global = []
        for r0, c0, r1, c1 in group["rects_local"]:
            rects_global.append(
                (
                    int(union_r0 + r0),
                    int(union_c0 + c0),
                    int(union_r0 + r1),
                    int(union_c0 + c1),
                )
            )
        group["rects_global"] = rects_global

    legend_cols: list[int] = []
    for group in placed:
        rects_sorted = sorted(group["rects_global"], key=_center_col)
        cols = [_center_col(rects_sorted[0]), _center_col(rects_sorted[1])]
        if cols[0] == cols[1]:
            return None
        group["legend_cols"] = cols
        legend_cols.extend(cols)
    if len(set(legend_cols)) != len(legend_cols):
        return None

    occupied = np.zeros((total_h, width), dtype=bool)
    for group in placed:
        union_mask = np.zeros_like(occupied)
        for box in group["rects_global"]:
            union_mask |= _box_mask((total_h, width), box)
        marker_mask = _box_mask((total_h, width), group["marker_box"])
        if np.any(occupied & union_mask) or np.any(occupied & marker_mask):
            return None
        occupied |= union_mask
        occupied |= marker_mask

    for idx, group in enumerate(placed):
        union_mask = np.zeros((total_h, width), dtype=bool)
        for box in group["rects_global"]:
            union_mask |= _box_mask((total_h, width), box)
        own_dist = _min_distance(union_mask, group["marker_box"])
        other_dists = []
        for jdx, other in enumerate(placed):
            if jdx == idx:
                continue
            other_mask = np.zeros((total_h, width), dtype=bool)
            for box in other["rects_global"]:
                other_mask |= _box_mask((total_h, width), box)
            other_dists.append(_min_distance(other_mask, group["marker_box"]))
        if other_dists and own_dist >= min(other_dists):
            return None

    return int(total_h), int(width), placed


def _render_case(groups: list[dict[str, object]], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)

    for group in groups:
        shape_color = int(group["shape_color"])
        marker_color = int(group["marker_color"])
        rects = [tuple(map(int, box)) for box in group["rects_global"]]

        union_mask = np.zeros_like(grid, dtype=bool)
        for box in rects:
            union_mask |= _box_mask(grid.shape, box)
        marker_mask = _box_mask(grid.shape, group["marker_box"])
        grid[union_mask] = shape_color
        grid[marker_mask] = marker_color

        expected[union_mask] = shape_color
        expected[marker_mask] = marker_color

        left_rect, right_rect = sorted(rects, key=_center_col)
        inter = _intersection_box(left_rect, right_rect)
        assert inter is not None
        inter_mask = _box_mask(grid.shape, inter)
        left_only = _box_mask(grid.shape, left_rect) & ~inter_mask
        right_only = _box_mask(grid.shape, right_rect) & ~inter_mask
        left_area = int(left_only.sum())
        right_area = int(right_only.sum())
        expected[inter_mask] = marker_color
        if left_area <= right_area:
            expected[left_only] = BG
        if right_area < left_area:
            expected[right_only] = BG

        for col in group["legend_cols"]:
            grid[-1, int(col)] = shape_color

    expected[-1, :] = BG
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    n_groups = random.randint(2, 3)
    if 2 * n_groups > len(PALETTE):
        return None

    chosen_colors = random.sample(list(PALETTE), 2 * n_groups)
    shape_colors = chosen_colors[:n_groups]
    marker_colors = chosen_colors[n_groups:]

    groups: list[dict[str, object]] = []
    for idx in range(n_groups):
        diag = random.choice(["lr", "rl"])
        smaller_side = random.choice(["left", "right"])
        group = _build_group(diag, smaller_side)
        if group is None:
            return None
        group["shape_color"] = int(shape_colors[idx])
        group["marker_color"] = int(marker_colors[idx])
        groups.append(group)

    placed = _place_groups(groups)
    if placed is None:
        return None
    height, width, ready_groups = placed
    grid, expected = _render_case(ready_groups, height, width)

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
