from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


SHAPE_LIBRARY: dict[int, list[tuple[tuple[int, int], ...]]] = {
    2: [((0, 0), (0, 1))],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
    ],
    5: [
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ],
    6: [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
    ],
    7: [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)),
    ],
    8: [
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
    ],
}


GEN_ATTEMPTS = 400
GROUP_PLACE_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _rotate90(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize([(c, -r) for r, c in cells])


def _flip_horizontal(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize([(r, -c) for r, c in cells])


def _shape_info(cells: tuple[tuple[int, int], ...], area: int | None = None) -> dict[str, object]:
    norm = _normalize(cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return {
        "cells": norm,
        "area": int(area if area is not None else len(norm)),
        "h": int(h),
        "w": int(w),
    }


def _component_info(cells: list[tuple[int, int]]) -> dict[str, object]:
    coords = np.array(sorted(cells), dtype=int)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return {
        "coords": coords,
        "area": int(len(cells)),
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "norm": norm,
    }


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
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(cells)
    return comps


def _components_of_color(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    return [_component_info(comp) for comp in _components(grid == int(color))]


def _is_straight_segment(comp: dict[str, object]) -> bool:
    h = int(comp["h"])
    w = int(comp["w"])
    return (h == 1 or w == 1) and int(comp["area"]) == max(h, w)


def _detect_slots(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    h, _w = grid.shape
    candidates: list[tuple[int, list[int]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components_of_color(grid, color)
        if not comps:
            continue
        cols: list[int] = []
        ok = True
        for comp in comps:
            if int(comp["area"]) != 2 or int(comp["h"]) != 2 or int(comp["w"]) != 1:
                ok = False
                break
            if int(comp["r0"]) != h - 2 or int(comp["r1"]) != h - 1:
                ok = False
                break
            cols.append(int(comp["c0"]))
        if not ok:
            continue
        cols.sort()
        if len(cols) >= 2:
            gaps = [int(cols[i + 1] - cols[i]) for i in range(len(cols) - 1)]
            if len(set(gaps)) != 1:
                continue
        if np.count_nonzero(grid == int(color)) != 2 * len(cols):
            continue
        candidates.append((int(color), cols))
    assert candidates, "slot markers not found"
    color, cols = max(candidates, key=lambda item: (len(item[1]), -item[0]))
    return int(color), [int(c) for c in cols]


def _score_shape_line_assignment(grid: np.ndarray, shape_color: int, line_color: int | None) -> tuple[int, ...]:
    shape_comps = _components_of_color(grid, shape_color)
    h, w = grid.shape
    shape_map = -np.ones((h, w), dtype=int)
    for idx, comp in enumerate(shape_comps):
        for r, c in comp["coords"]:
            shape_map[int(r), int(c)] = int(idx)

    line_comps = _components_of_color(grid, line_color) if line_color is not None else []
    shape_area_ok = sum(1 for comp in shape_comps if 2 <= int(comp["area"]) <= 8)
    shape_non_1d = sum(1 for comp in shape_comps if int(comp["h"]) > 1 and int(comp["w"]) > 1)
    shape_bad = sum(1 for comp in shape_comps if not (2 <= int(comp["area"]) <= 8))
    touch_good = 0
    touch_any = 0
    line_good = 0
    line_bad = 0
    for comp in line_comps:
        if _is_straight_segment(comp):
            line_good += 1
        else:
            line_bad += 1
        adj: set[int] = set()
        for r, c in comp["coords"]:
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and int(shape_map[nr, nc]) >= 0:
                    adj.add(int(shape_map[nr, nc]))
        if len(adj) >= 2:
            touch_good += 1
        elif len(adj) == 1:
            touch_any += 1
    return (
        int(shape_non_1d),
        int(shape_area_ok),
        int(touch_good),
        int(line_good),
        int(touch_any),
        int(len(shape_comps)),
        -int(shape_bad),
        -int(line_bad),
        -int(len(line_comps)),
    )


def _detect_shape_and_line_colors(grid: np.ndarray, bg: int, slot_color: int) -> tuple[int, int | None]:
    colors = [int(v) for v in np.unique(grid) if int(v) not in (bg, slot_color)]
    assert colors, "shape color not found"
    if len(colors) == 1:
        return int(colors[0]), None
    best: tuple[tuple[int, ...], int, int | None] | None = None
    for shape_color in colors:
        remaining = [color for color in colors if color != shape_color]
        line_color = int(remaining[0]) if remaining else None
        score = _score_shape_line_assignment(grid, int(shape_color), line_color)
        candidate = (score, int(shape_color), line_color)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return int(best[1]), best[2]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    slot_color, slot_cols = _detect_slots(g, bg)
    shape_color, line_color = _detect_shape_and_line_colors(g, bg, slot_color)

    shape_comps = _components_of_color(g, shape_color)
    shape_map = -np.ones((h, w), dtype=int)
    for idx, comp in enumerate(shape_comps):
        for r, c in comp["coords"]:
            shape_map[int(r), int(c)] = int(idx)

    parent = list(range(len(shape_comps)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    if line_color is not None:
        for comp in _components_of_color(g, line_color):
            adj: set[int] = set()
            for r, c in comp["coords"]:
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and int(shape_map[nr, nc]) >= 0:
                        adj.add(int(shape_map[nr, nc]))
            adj_list = sorted(adj)
            if len(adj_list) >= 2:
                base = int(adj_list[0])
                for other in adj_list[1:]:
                    union(base, int(other))

    groups_by_root: dict[int, list[dict[str, object]]] = defaultdict(list)
    for idx, comp in enumerate(shape_comps):
        groups_by_root[find(int(idx))].append(comp)
    groups = list(groups_by_root.values())
    assert len(groups) == len(slot_cols), "group count must match slot count"

    def shape_key(comp: dict[str, object]) -> tuple[object, ...]:
        return (int(comp["area"]), int(comp["h"]), int(comp["w"]), comp["norm"])

    def group_key(group: list[dict[str, object]]) -> tuple[object, ...]:
        max_area = max(int(comp["area"]) for comp in group)
        anchor = min((int(comp["r0"]), int(comp["c0"])) for comp in group)
        return (-int(max_area), int(anchor[0]), int(anchor[1]))

    ordered_groups = sorted(groups, key=group_key)
    out = np.full_like(g, bg)
    out[g == int(slot_color)] = int(slot_color)

    for slot_col, group in zip(slot_cols, ordered_groups):
        shapes = sorted(group, key=shape_key)
        stack_height = sum(int(shape["h"]) for shape in shapes) + (len(shapes) - 1)
        top = int(h - 2 - stack_height)
        cursor = int(top)
        for shape in shapes:
            left = int(slot_col - int(shape["w"]) // 2)
            for dr, dc in shape["norm"]:
                rr = int(cursor + dr)
                cc = int(left + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(shape_color)
            cursor += int(shape["h"]) + 1

    return out


def _random_shape(area: int) -> dict[str, object]:
    cells = tuple(random.choice(SHAPE_LIBRARY[int(area)]))
    if random.random() < 0.5:
        cells = _flip_horizontal(cells)
    for _ in range(random.randint(0, 3)):
        cells = _rotate90(cells)
    return _shape_info(cells, area)


def _edge_rows(shape: dict[str, object], side: str) -> list[int]:
    cells = shape["cells"]
    if side == "left":
        target = 0
        return [int(r) for r, c in cells if int(c) == target]
    target = int(shape["w"]) - 1
    return [int(r) for r, c in cells if int(c) == target]


def _edge_cols(shape: dict[str, object], side: str) -> list[int]:
    cells = shape["cells"]
    if side == "top":
        target = 0
        return [int(c) for r, c in cells if int(r) == target]
    target = int(shape["h"]) - 1
    return [int(c) for r, c in cells if int(r) == target]


def _build_group_pattern(shapes: list[dict[str, object]], orientation: str | None) -> dict[str, object]:
    placements: list[tuple[dict[str, object], int, int]] = [(shapes[0], 0, 0)]
    line_cells: list[tuple[int, int]] = []
    prev_shape = shapes[0]
    prev_top = 0
    prev_left = 0

    for shape in shapes[1:]:
        gap = random.randint(2, 4)
        if orientation == "h":
            prev_row = random.choice(_edge_rows(prev_shape, "right"))
            next_row = random.choice(_edge_rows(shape, "left"))
            top = int(prev_top + prev_row - next_row)
            left = int(prev_left + int(prev_shape["w"]) - 1 + gap)
            line_row = int(prev_top + prev_row)
            for col in range(int(prev_left + int(prev_shape["w"])), int(left)):
                line_cells.append((int(line_row), int(col)))
        else:
            prev_col = random.choice(_edge_cols(prev_shape, "bottom"))
            next_col = random.choice(_edge_cols(shape, "top"))
            top = int(prev_top + int(prev_shape["h"]) - 1 + gap)
            left = int(prev_left + prev_col - next_col)
            line_col = int(prev_left + prev_col)
            for row in range(int(prev_top + int(prev_shape["h"])), int(top)):
                line_cells.append((int(row), int(line_col)))
        placements.append((shape, int(top), int(left)))
        prev_shape = shape
        prev_top = int(top)
        prev_left = int(left)

    all_cells: list[tuple[int, int]] = list(line_cells)
    for shape, top, left in placements:
        all_cells.extend((int(top + r), int(left + c)) for r, c in shape["cells"])

    r0 = min(r for r, _ in all_cells)
    c0 = min(c for _, c in all_cells)
    placed_shapes: list[dict[str, object]] = []
    union_cells: set[tuple[int, int]] = set()
    for shape, top, left in placements:
        abs_cells = tuple(sorted((int(top - r0 + r), int(left - c0 + c)) for r, c in shape["cells"]))
        placed_shapes.append({"shape": shape, "cells": abs_cells})
        union_cells.update(abs_cells)
    norm_lines = tuple(sorted((int(r - r0), int(c - c0)) for r, c in line_cells))
    union_cells.update(norm_lines)

    height = max(r for r, _ in union_cells) + 1
    width = max(c for _, c in union_cells) + 1
    return {
        "shapes": placed_shapes,
        "line_cells": norm_lines,
        "cells": tuple(sorted(union_cells)),
        "h": int(height),
        "w": int(width),
    }


def _sample_group_sizes(group_count: int, total_shapes: int) -> list[int]:
    sizes = [1] * int(group_count)
    for _ in range(int(total_shapes - group_count)):
        choices = [idx for idx, size in enumerate(sizes) if size < 3]
        sizes[random.choice(choices)] += 1
    random.shuffle(sizes)
    return [int(size) for size in sizes]


def _sample_groups() -> list[dict[str, object]] | None:
    group_count = random.randint(2, 4)
    total_shapes = random.randint(group_count + 1, min(7, group_count + 3))
    areas = random.sample(list(range(2, 9)), total_shapes)
    random.shuffle(areas)
    shapes = [_random_shape(int(area)) for area in areas]
    if not any(int(shape["h"]) > 1 and int(shape["w"]) > 1 for shape in shapes):
        return None

    groups: list[dict[str, object]] = []
    offset = 0
    for size in _sample_group_sizes(group_count, total_shapes):
        group_shapes = [shapes[offset + i] for i in range(size)]
        offset += size
        input_order = list(group_shapes)
        random.shuffle(input_order)
        orientation = random.choice(["h", "v"]) if size > 1 else None
        pattern = _build_group_pattern(input_order, orientation)
        stack_height = sum(int(shape["h"]) for shape in group_shapes) + (len(group_shapes) - 1)
        max_width = max(int(shape["w"]) for shape in group_shapes)
        groups.append(
            {
                "shapes": group_shapes,
                "pattern": pattern,
                "max_area": max(int(shape["area"]) for shape in group_shapes),
                "stack_height": int(stack_height),
                "max_width": int(max_width),
            }
        )
    return groups


def _sample_palette() -> tuple[int, int, int, int]:
    colors = random.sample(range(10), 4)
    return int(colors[0]), int(colors[1]), int(colors[2]), int(colors[3])


def _dilate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR8 + ((0, 0),):
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out.add((int(nr), int(nc)))
    return out


def _place_groups(
    grid: np.ndarray,
    bg: int,
    shape_color: int,
    line_color: int,
    groups: list[dict[str, object]],
) -> bool:
    h, w = grid.shape
    marker_cells = set(map(tuple, np.argwhere(grid != int(bg))))
    forbidden = _dilate(marker_cells, h, w)
    order = sorted(
        groups,
        key=lambda group: (int(group["pattern"]["h"]) * int(group["pattern"]["w"]), int(group["max_area"])),
        reverse=True,
    )

    for group in order:
        pattern = group["pattern"]
        gh = int(pattern["h"])
        gw = int(pattern["w"])
        row_min = 1 if h - 3 - gh >= 1 else 0
        row_max = h - 3 - gh
        col_min = 1 if w - 1 - gw >= 1 else 0
        col_max = w - 1 - gw
        if row_max < row_min or col_max < col_min:
            return False

        placed = False
        for _ in range(GROUP_PLACE_ATTEMPTS):
            top = random.randint(row_min, row_max)
            left = random.randint(col_min, col_max)
            abs_cells = {(int(top + r), int(left + c)) for r, c in pattern["cells"]}
            if any(cell in forbidden for cell in abs_cells):
                continue
            for item in pattern["shapes"]:
                for r, c in item["cells"]:
                    grid[int(top + r), int(left + c)] = int(shape_color)
            for r, c in pattern["line_cells"]:
                grid[int(top + r), int(left + c)] = int(line_color)
            forbidden |= _dilate(abs_cells, h, w)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_canvas(groups: list[dict[str, object]]) -> tuple[np.ndarray, int, int, int, int]:
    bg, shape_color, line_color, slot_color = _sample_palette()
    ordered = sorted(groups, key=lambda group: int(group["max_area"]), reverse=True)
    max_width = max(int(group["max_width"]) for group in ordered)
    left_reach = max(int(group["max_width"]) // 2 for group in ordered)
    right_reach = max((int(group["max_width"]) - 1) - int(group["max_width"]) // 2 for group in ordered)
    spacing = int(max_width + random.randint(3, 6))
    left_pad = int(left_reach + random.randint(1, 3))
    right_pad = int(right_reach + random.randint(1, 3))
    width = int(left_pad + spacing * (len(ordered) - 1) + right_pad + 1)

    output_need = max(int(group["stack_height"]) for group in ordered) + 2
    input_need = max(int(group["pattern"]["h"]) for group in ordered) + 3
    height = int(max(output_need + random.randint(4, 8), input_need + random.randint(2, 5)))

    grid = np.full((height, width), bg, dtype=int)
    slot_cols = [int(left_pad + spacing * idx) for idx in range(len(ordered))]
    for col in slot_cols:
        grid[height - 2 : height, int(col)] = int(slot_color)
    return grid, bg, shape_color, line_color, slot_color


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        groups = _sample_groups()
        if groups is None:
            continue
        grid, bg, shape_color, line_color, _slot_color = _build_canvas(groups)
        if not _place_groups(grid, bg, shape_color, line_color, groups):
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
