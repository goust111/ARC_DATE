from __future__ import annotations

import random
from itertools import combinations

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_EDGE = {
    (-1, 0): "top",
    (1, 0): "bottom",
    (0, -1): "right",
    (0, 1): "left",
}
RECT_SHAPES = [(1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1), (2, 2)]
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _monochrome_components(
    grid: np.ndarray, valid_mask: np.ndarray, connectivity: int = 4
) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if not valid_mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and valid_mask[nr, nc]
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append({"color": color, "cells": sorted(cells)})
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return (r1 - r0 + 1) * (c1 - c0 + 1) == len(cells)


def _legend_info(grid: np.ndarray, bg: int) -> tuple[np.ndarray, dict[frozenset[int], int]]:
    h, w = grid.shape
    non_bg = grid != bg
    best: tuple[int, tuple[int, int, int, int]] | None = None
    for cells in _components(non_bg, connectivity=8):
        r0, c0, r1, c1 = _bbox(cells)
        if c1 - c0 + 1 != 2:
            continue
        if len(cells) != (r1 - r0 + 1) * 2:
            continue
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            continue
        if not any(r in (0, h - 1) and c in (0, w - 1) for r, c in cells):
            continue
        area = len(cells)
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        raise AssertionError("expected a corner 2-column legend")

    r0, c0, r1, c1 = best[1]
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    mix_map: dict[frozenset[int], int] = {}
    for r in range(r0, r1 + 1):
        left = int(grid[r, c0])
        right = int(grid[r, c0 + 1])
        if left == bg or right == bg or left == right:
            continue
        key = frozenset((left, right))
        if key not in mix_map:
            mix_map[key] = left
    return legend_mask, mix_map


def _arrow_direction(cells: list[tuple[int, int]], h: int, w: int) -> tuple[int, int] | None:
    if len(cells) != 3:
        return None
    cell_set = set(cells)
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})

    if len(rows) == 2 and len(cols) == 3:
        top_row, bottom_row = rows
        row_counts = {r: sum(1 for rr, _ in cells if rr == r) for r in rows}
        if top_row == 0 and row_counts[top_row] == 2 and row_counts[bottom_row] == 1:
            tip_col = next(c for r, c in cells if r == bottom_row)
            expected = {(0, tip_col - 1), (0, tip_col + 1), (1, tip_col)}
            if cell_set == expected:
                return (1, 0)
        if bottom_row == h - 1 and row_counts[bottom_row] == 2 and row_counts[top_row] == 1:
            tip_col = next(c for r, c in cells if r == top_row)
            expected = {(h - 1, tip_col - 1), (h - 1, tip_col + 1), (h - 2, tip_col)}
            if cell_set == expected:
                return (-1, 0)

    if len(cols) == 2 and len(rows) == 3:
        left_col, right_col = cols
        col_counts = {c: sum(1 for _, cc in cells if cc == c) for c in cols}
        if left_col == 0 and col_counts[left_col] == 2 and col_counts[right_col] == 1:
            tip_row = next(r for r, c in cells if c == right_col)
            expected = {(tip_row - 1, 0), (tip_row + 1, 0), (tip_row, 1)}
            if cell_set == expected:
                return (0, 1)
        if right_col == w - 1 and col_counts[right_col] == 2 and col_counts[left_col] == 1:
            tip_row = next(r for r, c in cells if c == left_col)
            expected = {(tip_row - 1, w - 1), (tip_row + 1, w - 1), (tip_row, w - 2)}
            if cell_set == expected:
                return (0, -1)

    return None


def _parse_scene(
    grid: np.ndarray,
) -> tuple[int, np.ndarray, dict[frozenset[int], int], list[dict[str, object]], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    legend_mask, mix_map = _legend_info(g, bg)

    comp_mask = (g != bg) & ~legend_mask
    mono = _monochrome_components(g, comp_mask, connectivity=8)

    directions: dict[int, tuple[int, int]] = {}
    arrow_cells: set[tuple[int, int]] = set()
    for comp in mono:
        cells = comp["cells"]
        if not isinstance(cells, list):
            continue
        direction = _arrow_direction(cells, h, w)
        if direction is None:
            continue
        color = int(comp["color"])
        directions[color] = direction
        arrow_cells.update(cells)

    object_mask = (g != bg) & ~legend_mask
    for r, c in arrow_cells:
        object_mask[r, c] = False

    objects: list[dict[str, object]] = []
    for comp in _monochrome_components(g, object_mask, connectivity=8):
        color = int(comp["color"])
        cells = comp["cells"]
        if not isinstance(cells, list):
            continue
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        if len(cells) < 2 or len(cells) > 4 or not _is_rectangular(cells):
            continue
        objects.append({"color": color, "cells": cells})

    return bg, legend_mask, mix_map, objects, directions


def _distance_to_target(
    cells: list[tuple[int, int]], direction: tuple[int, int], h: int, w: int
) -> int:
    dr, dc = direction
    if dr == -1:
        return min(r for r, _ in cells)
    if dr == 1:
        return (h - 1) - max(r for r, _ in cells)
    if dc == -1:
        return min(c for _, c in cells)
    return (w - 1) - max(c for _, c in cells)


def _slide_objects(
    objects: list[dict[str, object]], directions: dict[int, tuple[int, int]], h: int, w: int
) -> list[dict[str, object]]:
    stationary = [obj for obj in objects if int(obj["color"]) not in directions]
    movable = [obj for obj in objects if int(obj["color"]) in directions]
    occupied = {cell for obj in stationary for cell in obj["cells"]}  # type: ignore[arg-type]
    final_cells: dict[int, list[tuple[int, int]]] = {
        int(obj["color"]): list(obj["cells"]) for obj in stationary  # type: ignore[arg-type]
    }

    movable_sorted = sorted(
        movable,
        key=lambda obj: (
            _distance_to_target(obj["cells"], directions[int(obj["color"])], h, w),  # type: ignore[arg-type]
            _bbox(obj["cells"])[0],  # type: ignore[arg-type]
            _bbox(obj["cells"])[1],  # type: ignore[arg-type]
        ),
    )

    for obj in movable_sorted:
        color = int(obj["color"])
        dr, dc = directions[color]
        current = list(obj["cells"])  # type: ignore[arg-type]
        while True:
            shifted = [(r + dr, c + dc) for r, c in current]
            if any(not (0 <= r < h and 0 <= c < w) for r, c in shifted):
                break
            if any((r, c) in occupied for r, c in shifted):
                break
            current = shifted
        final_cells[color] = current
        occupied.update(current)

    return [{"color": int(obj["color"]), "cells": final_cells[int(obj["color"])]} for obj in objects]


def _adjacent(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    bset = set(b)
    for r, c in a:
        if (r - 1, c) in bset or (r + 1, c) in bset or (r, c - 1) in bset or (r, c + 1) in bset:
            return True
    return False


def _apply_mixing(
    objects: list[dict[str, object]], mix_map: dict[frozenset[int], int]
) -> list[int]:
    targets = [set() for _ in objects]
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            color_i = int(objects[i]["color"])
            color_j = int(objects[j]["color"])
            key = frozenset((color_i, color_j))
            if key not in mix_map:
                continue
            if _adjacent(objects[i]["cells"], objects[j]["cells"]):  # type: ignore[arg-type]
                target = mix_map[key]
                targets[i].add(target)
                targets[j].add(target)

    final_colors: list[int] = []
    for obj, options in zip(objects, targets):
        if len(options) == 1:
            final_colors.append(next(iter(options)))
        else:
            final_colors.append(int(obj["color"]))
    return final_colors


def _render(h: int, w: int, bg: int, objects: list[dict[str, object]], colors: list[int]) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    for obj, color in zip(objects, colors):
        for r, c in obj["cells"]:  # type: ignore[arg-type]
            out[r, c] = color
    return out


def _solve_from_parts(
    h: int,
    w: int,
    bg: int,
    objects: list[dict[str, object]],
    directions: dict[int, tuple[int, int]],
    mix_map: dict[frozenset[int], int],
) -> np.ndarray:
    final_objects = _slide_objects(objects, directions, h, w)
    final_colors = _apply_mixing(final_objects, mix_map)
    return _render(h, w, bg, final_objects, final_colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use edge arrows to slide same-colored interior rectangles, then recolor adjacent pairs by a corner legend.

    - Each border arrow is a 3-pixel V whose color matches exactly one interior monochrome rectangle.
    - The arrow's inward direction gives that object's slide direction.
    - Objects slide in increasing order of distance to their target edge, stopping at the border or at an already
      stopped object.
    - After sliding, any 4-neighbor-adjacent object pair whose unordered color pair appears in the 2-column corner
      legend is recolored so both objects become the legend row's left color.
    - Arrows and the legend are erased in the output.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, _legend_mask, mix_map, objects, directions = _parse_scene(g)
    out = _solve_from_parts(h, w, bg, objects, directions, mix_map)
    return out


def _legend_rows_for_mode(
    final_objects: list[dict[str, object]], mode: str
) -> list[tuple[int, int]] | None:
    n = len(final_objects)
    colors = [int(obj["color"]) for obj in final_objects]
    adjacent_pairs: list[tuple[int, int]] = []
    for i, j in combinations(range(n), 2):
        if _adjacent(final_objects[i]["cells"], final_objects[j]["cells"]):  # type: ignore[arg-type]
            adjacent_pairs.append((i, j))

    all_pairs = list(combinations(range(n), 2))
    adjacent_set = {frozenset(pair) for pair in adjacent_pairs}
    non_adjacent_pairs = [pair for pair in all_pairs if frozenset(pair) not in adjacent_set]

    disjoint_adj2: list[list[tuple[int, int]]] = []
    for a, b in combinations(adjacent_pairs, 2):
        if len({a[0], a[1], b[0], b[1]}) == 4:
            disjoint_adj2.append([a, b])

    if mode == "two":
        if not disjoint_adj2:
            return None
        chosen = random.choice(disjoint_adj2)
        rows = [(colors[i], colors[j]) for i, j in chosen]
    elif mode == "one":
        if not adjacent_pairs:
            return None
        i, j = random.choice(adjacent_pairs)
        rows = [(colors[i], colors[j])]
        used = {i, j}
        extras = [pair for pair in non_adjacent_pairs if pair[0] not in used and pair[1] not in used]
        if extras and random.random() < 0.5:
            ei, ej = random.choice(extras)
            rows.append((colors[ei], colors[ej]))
    else:
        if not non_adjacent_pairs:
            return None
        rows = []
        pair_count = 2 if len(non_adjacent_pairs) >= 2 and random.random() < 0.4 else 1
        used: set[int] = set()
        candidates = non_adjacent_pairs[:]
        random.shuffle(candidates)
        for i, j in candidates:
            if i in used or j in used:
                continue
            rows.append((colors[i], colors[j]))
            used.update((i, j))
            if len(rows) == pair_count:
                break
        if not rows:
            return None

    randomized_rows = []
    for a, b in rows:
        randomized_rows.append((a, b) if random.random() < 0.5 else (b, a))
    random.shuffle(randomized_rows)
    return randomized_rows


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                out[nr, nc] = True
    return out


def _place_objects(
    h: int, w: int, colors: list[int], directions: list[tuple[int, int]]
) -> list[dict[str, object]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []
    interior_r0, interior_r1 = 3, h - 4
    interior_c0, interior_c1 = 3, w - 4
    if interior_r0 > interior_r1 or interior_c0 > interior_c1:
        return None

    for color, (obj_h, obj_w) in zip(colors, random.choices(RECT_SHAPES, k=len(colors))):
        candidates: list[tuple[int, int]] = []
        for r0 in range(interior_r0, interior_r1 - obj_h + 2):
            for c0 in range(interior_c0, interior_c1 - obj_w + 2):
                r1 = r0 + obj_h - 1
                c1 = c0 + obj_w - 1
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h - 1, r1 + 1)
                cc1 = min(w - 1, c1 + 1)
                if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        cells = [(r0 + dr, c0 + dc) for dr in range(obj_h) for dc in range(obj_w)]
        for r, c in cells:
            occupied[r, c] = True
        objects.append({"color": color, "cells": cells})

    distances = [
        _distance_to_target(obj["cells"], direction, h, w)  # type: ignore[arg-type]
        for obj, direction in zip(objects, directions)
    ]
    if len(set(distances)) != len(distances):
        return None
    return objects


def _draw_legend(
    grid: np.ndarray, corner: str, rows: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    n = len(rows)
    mask = np.zeros((h, w), dtype=bool)
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - 2
    elif corner == "bl":
        r0, c0 = h - n, 0
    else:
        r0, c0 = h - n, w - 2

    for i, (left, right) in enumerate(rows):
        grid[r0 + i, c0] = left
        grid[r0 + i, c0 + 1] = right
        mask[r0 + i, c0] = True
        mask[r0 + i, c0 + 1] = True
    return grid, mask


def _arrow_cells_for(direction: tuple[int, int], center: int, h: int, w: int) -> list[tuple[int, int]]:
    if direction == (1, 0):
        return [(0, center - 1), (0, center + 1), (1, center)]
    if direction == (-1, 0):
        return [(h - 1, center - 1), (h - 1, center + 1), (h - 2, center)]
    if direction == (0, 1):
        return [(center - 1, 0), (center + 1, 0), (center, 1)]
    return [(center - 1, w - 1), (center + 1, w - 1), (center, w - 2)]


def _place_arrows(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    colors: list[int],
    directions: list[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    reserved = _dilate_mask(legend_mask)
    edge_to_colors: dict[str, list[tuple[int, tuple[int, int]]]] = {"top": [], "bottom": [], "left": [], "right": []}
    for color, direction in zip(colors, directions):
        edge_to_colors[DIR_TO_EDGE[direction]].append((color, direction))

    for edge, items in edge_to_colors.items():
        candidates = list(range(1, w - 1)) if edge in ("top", "bottom") else list(range(1, h - 1))
        random.shuffle(candidates)
        for color, direction in items:
            placed = False
            for center in candidates:
                cells = _arrow_cells_for(direction, center, h, w)
                if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                    continue
                if any(reserved[r, c] for r, c in cells):
                    continue
                local_mask = np.zeros((h, w), dtype=bool)
                for r, c in cells:
                    local_mask[r, c] = True
                if (_dilate_mask(local_mask) & reserved).any():
                    continue
                for r, c in cells:
                    grid[r, c] = color
                reserved |= _dilate_mask(local_mask)
                placed = True
                break
            if not placed:
                return False
    return True


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(13, 19)
    w = random.randint(13, 19)
    bg = random.randint(0, 9)
    palette = [color for color in range(10) if color != bg]

    n_objects = random.randint(3, 5)
    colors = random.sample(palette, n_objects)

    top_capacity = max(1, (w - 5) // 3)
    side_capacity = max(1, (h - 5) // 3)
    directions: list[tuple[int, int]] = []
    counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    limits = {"top": top_capacity, "bottom": top_capacity, "left": side_capacity, "right": side_capacity}
    for _ in range(n_objects):
        options = [direction for direction in DIRS4 if counts[DIR_TO_EDGE[direction]] < limits[DIR_TO_EDGE[direction]]]
        if not options:
            return None
        direction = random.choice(options)
        directions.append(direction)
        counts[DIR_TO_EDGE[direction]] += 1
    if len(set(directions)) == 1 and random.random() < 0.7:
        return None

    objects = _place_objects(h, w, colors, directions)
    if objects is None:
        return None

    final_objects = _slide_objects(objects, {color: direction for color, direction in zip(colors, directions)}, h, w)
    has_adjacency = any(
        _adjacent(final_objects[i]["cells"], final_objects[j]["cells"])  # type: ignore[arg-type]
        for i, j in combinations(range(len(final_objects)), 2)
    )
    if not has_adjacency and random.random() < 0.65:
        return None

    mode_choices = ["none"]
    if has_adjacency:
        mode_choices.extend(["one", "one"])
    rows_two = _legend_rows_for_mode(final_objects, "two")
    if rows_two is not None:
        mode_choices.append("two")
    mode = random.choice(mode_choices)
    rows = rows_two if mode == "two" else _legend_rows_for_mode(final_objects, mode)
    if rows is None:
        return None

    corner = random.choice(["tl", "tr", "bl", "br"])
    grid = np.full((h, w), bg, dtype=int)
    grid, legend_mask = _draw_legend(grid, corner, rows)

    for obj in objects:
        for r, c in obj["cells"]:  # type: ignore[arg-type]
            grid[r, c] = int(obj["color"])

    if not _place_arrows(grid, legend_mask, colors, directions):
        return None

    mix_map = {frozenset(pair): pair[0] for pair in rows}
    expected = _solve_from_parts(h, w, bg, objects, {color: direction for color, direction in zip(colors, directions)}, mix_map)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _sample_scene()
        if sample is None:
            continue
        grid, expected = sample
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
