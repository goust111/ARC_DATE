from __future__ import annotations

import random
from collections import Counter

import numpy as np


FRAME_COLOR = 5
MARKER_COLOR = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _find_target_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == FRAME_COLOR):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if border != set(comp):
            continue
        inside = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inside.size == 0 or not bool(np.all(inside == int(bg))):
            continue
        candidates.append((int(r0), int(c0), int(r1), int(c1)))
    assert candidates, "expected a gray target frame"
    candidates.sort(key=lambda box: (box[1], box[0], -(box[2] - box[0]) * (box[3] - box[1])))
    return candidates[0]


def _is_vertical_run(coords: list[tuple[int, int]]) -> bool:
    cols = {c for _r, c in coords}
    if len(cols) != 1:
        return False
    rows = sorted(r for r, _c in coords)
    return rows == list(range(rows[0], rows[-1] + 1))


def _extract_fragments(grid: np.ndarray, bg: int) -> list[dict]:
    mask = (grid != int(bg)) & (grid != FRAME_COLOR)
    fragments: list[dict] = []
    for comp in _components(mask):
        orange = [(r, c) for r, c in comp if int(grid[r, c]) == MARKER_COLOR]
        body = [(r, c) for r, c in comp if int(grid[r, c]) != MARKER_COLOR]
        if not orange or not body:
            continue
        body_colors = {int(grid[r, c]) for r, c in body}
        if len(body_colors) != 1 or not _is_vertical_run(orange):
            continue
        br0, bc0, _br1, _bc1 = _bbox(body)
        rel = sorted((int(r - br0), int(c - bc0)) for r, c in body)
        fragments.append(
            {
                "order": int(len(orange)),
                "color": int(next(iter(body_colors))),
                "cells": rel,
            }
        )
    fragments.sort(key=lambda item: int(item["order"]))
    assert fragments, "expected at least one fragment"
    assert len({int(item["order"]) for item in fragments}) == len(fragments), "orders must be unique"
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack colored fragments into the gray target frame using the orange marker count as layer order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    r0, c0, r1, c1 = _find_target_frame(g, bg)
    ir0, ic0, ir1, ic1 = int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)

    out = np.full_like(g, int(bg))
    _draw_frame(out, (r0, c0, r1, c1), FRAME_COLOR)

    for frag in _extract_fragments(g, bg):
        color = int(frag["color"])
        for rr, cc in frag["cells"]:
            tr, tc = int(ir0 + rr), int(ic0 + cc)
            if ir0 <= tr <= ir1 and ic0 <= tc <= ic1:
                out[tr, tc] = int(color)
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _shape_bbox(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _shape_right_boundary_rows(cells: set[tuple[int, int]]) -> list[int]:
    max_c = max(c for _r, c in cells)
    return [int(r) for r, c in cells if int(c) == int(max_c)]


def _shape_bottom_boundary_cols(cells: set[tuple[int, int]]) -> list[int]:
    max_r = max(r for r, _c in cells)
    return [int(c) for r, c in cells if int(r) == int(max_r)]


def _random_shape(max_h: int, max_w: int) -> set[tuple[int, int]] | None:
    for _ in range(80):
        h = random.randint(2, int(max_h))
        w = random.randint(2, int(max_w))
        max_area = min(h * w - 1, max(4, (h * w * 3) // 4))
        if max_area < 3:
            continue
        area = random.randint(3, int(max_area))
        cells: set[tuple[int, int]] = {(0, 0)}
        frontier = [(0, 1), (1, 0)]
        frontier = [p for p in frontier if p[0] < h and p[1] < w]
        while len(cells) < area and frontier:
            idx = random.randrange(len(frontier))
            r, c = frontier.pop(idx)
            if (r, c) in cells:
                continue
            cells.add((int(r), int(c)))
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells = _normalize(cells)
        sh, sw = _shape_bbox(cells)
        if sh < 2 or sw < 2:
            continue
        if len(cells) == sh * sw:
            continue
        if (0, 0) not in cells:
            continue
        return cells
    return None


def _make_fragment_spec(
    color: int,
    order: int,
    body_cells: set[tuple[int, int]],
    orientation: str,
) -> dict:
    body_h, body_w = _shape_bbox(body_cells)
    if orientation == "right":
        attach_row = random.choice(_shape_right_boundary_rows(body_cells))
        marker_cells = {(int(attach_row + k), int(body_w)) for k in range(int(order))}
    else:
        attach_col = random.choice(_shape_bottom_boundary_cols(body_cells))
        marker_cells = {(int(body_h + k), int(attach_col)) for k in range(int(order))}
    patch_h = max(max(r for r, _c in body_cells), max(r for r, _c in marker_cells)) + 1
    patch_w = max(max(c for _r, c in body_cells), max(c for _r, c in marker_cells)) + 1
    return {
        "color": int(color),
        "order": int(order),
        "body_cells": set(body_cells),
        "marker_cells": set(marker_cells),
        "patch_h": int(patch_h),
        "patch_w": int(patch_w),
        "orientation": orientation,
    }


def _render_expected(
    shape_specs: list[dict],
    grid_shape: tuple[int, int],
    bg: int,
    frame_box: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.full(grid_shape, int(bg), dtype=int)
    _draw_frame(out, frame_box, FRAME_COLOR)
    ir0, ic0 = int(frame_box[0] + 1), int(frame_box[1] + 1)
    for spec in sorted(shape_specs, key=lambda item: int(item["order"])):
        for rr, cc in spec["body_cells"]:
            out[int(ir0 + rr), int(ic0 + cc)] = int(spec["color"])
    return out


def _visible_color_count(shape_specs: list[dict]) -> int:
    top_by_cell: dict[tuple[int, int], int] = {}
    for spec in sorted(shape_specs, key=lambda item: int(item["order"])):
        for cell in spec["body_cells"]:
            top_by_cell[(int(cell[0]), int(cell[1]))] = int(spec["color"])
    return len(set(top_by_cell.values()))


def _has_occlusion(shape_specs: list[dict]) -> bool:
    seen: set[tuple[int, int]] = set()
    for spec in sorted(shape_specs, key=lambda item: int(item["order"])):
        cells = set(spec["body_cells"])
        if seen & cells:
            return True
        seen |= cells
    return False


def _place_objects(frame_w: int, objects: list[dict]) -> tuple[int, int, tuple[int, int, int, int], dict[int, tuple[int, int]]]:
    frame_h = int(objects[0]["frame_h"])
    frame_col = random.randint(1, 2)
    gap_to_objects = random.randint(2, 4)

    cols = 1 if len(objects) == 2 else random.choice([1, 2])
    shuffled = list(range(len(objects)))
    random.shuffle(shuffled)
    groups: list[list[int]] = [[] for _ in range(int(cols))]
    for idx, obj_idx in enumerate(shuffled):
        groups[idx % cols].append(int(obj_idx))

    col_widths = [max(objects[i]["patch_w"] for i in group) if group else 0 for group in groups]
    inter_col_gap = random.randint(2, 4)
    right_width = sum(col_widths) + inter_col_gap * max(0, cols - 1)

    group_gaps: list[list[int]] = []
    group_heights: list[int] = []
    for group in groups:
        gaps = [random.randint(1, 3) for _ in range(max(0, len(group) - 1))]
        total = sum(objects[i]["patch_h"] for i in group) + sum(gaps)
        group_gaps.append(gaps)
        group_heights.append(int(total))

    min_h = max(frame_h + 2, max(group_heights) + 2)
    grid_h = random.randint(int(min_h), int(min_h + 6))
    frame_row = random.randint(1, int(grid_h - frame_h - 1))
    frame_box = (int(frame_row), int(frame_col), int(frame_row + frame_h - 1), int(frame_col + frame_w - 1))

    right_start = int(frame_col + frame_w + gap_to_objects)
    grid_w = int(right_start + right_width + random.randint(2, 4))
    placements: dict[int, tuple[int, int]] = {}
    x = int(right_start)
    for col_idx, group in enumerate(groups):
        col_w = int(col_widths[col_idx])
        total_h = int(group_heights[col_idx])
        y = random.randint(1, int(grid_h - total_h - 1))
        for j, obj_idx in enumerate(group):
            jitter = random.randint(0, int(col_w - objects[obj_idx]["patch_w"]))
            placements[int(obj_idx)] = (int(y), int(x + jitter))
            y += int(objects[obj_idx]["patch_h"])
            if j < len(group_gaps[col_idx]):
                y += int(group_gaps[col_idx][j])
        x += col_w + inter_col_gap

    return int(grid_h), int(grid_w), frame_box, placements


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([0, 1, 2, 3, 4, 6, 8, 9])
    n_fragments = random.randint(2, 5)
    palette = [c for c in range(10) if c not in {bg, FRAME_COLOR, MARKER_COLOR}]
    colors = random.sample(palette, int(n_fragments))

    interior_h = random.randint(6, 10)
    interior_w = random.randint(6, 10)
    frame_h = int(interior_h + 2)
    frame_w = int(interior_w + 2)

    shape_specs: list[dict] = []
    for order, color in enumerate(colors, start=1):
        body = _random_shape(interior_h, interior_w)
        if body is None:
            return None
        orientation = random.choice(["right", "below"])
        spec = _make_fragment_spec(int(color), int(order), body, orientation)
        spec["frame_h"] = int(frame_h)
        shape_specs.append(spec)

    if not _has_occlusion(shape_specs):
        return None
    if _visible_color_count(shape_specs) < 2:
        return None
    union_cells = set().union(*(spec["body_cells"] for spec in shape_specs))
    if len(union_cells) >= interior_h * interior_w:
        return None

    grid_h, grid_w, frame_box, placements = _place_objects(frame_w, shape_specs)
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    _draw_frame(grid, frame_box, FRAME_COLOR)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    fr0, fc0, fr1, fc1 = frame_box
    occupied[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    right_count = 0
    below_count = 0
    for idx, spec in enumerate(shape_specs):
        top, left = placements[idx]
        coords = [
            (int(top + r), int(left + c), int(spec["color"]))
            for r, c in spec["body_cells"]
        ] + [
            (int(top + r), int(left + c), MARKER_COLOR)
            for r, c in spec["marker_cells"]
        ]
        rows = [r for r, _c, _v in coords]
        cols = [c for _r, c, _v in coords]
        if min(rows) < 0 or max(rows) >= grid_h or min(cols) < 0 or max(cols) >= grid_w:
            return None
        for r, c, _v in coords:
            rr0, rr1 = max(0, r - 1), min(grid_h - 1, r + 1)
            cc0, cc1 = max(0, c - 1), min(grid_w - 1, c + 1)
            if bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                return None
        for r, c, v in coords:
            grid[r, c] = int(v)
            occupied[r, c] = True
        if spec["orientation"] == "right":
            right_count += 1
        else:
            below_count += 1

    if right_count == 0 or below_count == 0:
        return None

    expected = _render_expected(shape_specs, grid.shape, bg, frame_box)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
