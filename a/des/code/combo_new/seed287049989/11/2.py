from __future__ import annotations

import random
from itertools import combinations

import numpy as np


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cell = stack.pop()
                comp.append(cell)
                for rr, cc in _neighbors4(cell):
                    if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not bool(seen[rr, cc]):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(comp)

    return comps


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _parse_mix_table(grid: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    assert grid.shape[0] >= 4 and grid.shape[1] >= 4, "expected room for a 3x3 legend and separator"
    table = np.asarray(grid[:3, :3], dtype=int)
    assert bool(np.all(table != int(bg))), "legend must be a full 3x3 color block"
    assert bool(np.all(grid[3, :] == int(bg)) or np.all(grid[:, 3] == int(bg))), (
        "legend must be separated from the rest by a background row or column"
    )

    mix: dict[tuple[int, int], int] = {}
    for row in table:
        a, b, m = map(int, row)
        assert a != b, "legend input colors must differ"
        mix[(a, b)] = m
        mix[(b, a)] = m
    return mix


def _order_path(component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(component) == 1:
        return list(component)

    comp_set = set(component)
    degree = {cell: sum((nbr in comp_set) for nbr in _neighbors4(cell)) for cell in comp_set}
    assert all(d in (1, 2) for d in degree.values()), "path cells must form a single thin chain"
    terminals = [cell for cell, d in degree.items() if d == 1]
    assert len(terminals) == 2, "path must have exactly two ends"

    ordered = [terminals[0]]
    prev: tuple[int, int] | None = None
    cur = terminals[0]
    while len(ordered) < len(component):
        next_cells = [nbr for nbr in _neighbors4(cur) if nbr in comp_set and nbr != prev]
        assert len(next_cells) == 1, "path chain must be unambiguous"
        prev, cur = cur, next_cells[0]
        ordered.append(cur)
    return ordered


def _path_candidate_data(
    grid: np.ndarray,
    color: int,
    bg: int,
    active_mask: np.ndarray,
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]] | None:
    path_mask = (grid == int(color)) & active_mask
    components = _components4(path_mask)
    if not components:
        return None

    out: list[tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]] = []
    h, w = grid.shape

    for component in components:
        comp_set = set(component)

        if len(component) == 1:
            cell = component[0]
            endpoint_cells = [
                (rr, cc)
                for rr, cc in _neighbors4(cell)
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in comp_set and int(grid[rr, cc]) != int(bg)
            ]
            if len(endpoint_cells) != 2:
                return None
            endpoint_colors = [int(grid[cell]) for cell in endpoint_cells]
            if len(set(endpoint_colors)) != 2 or int(color) in endpoint_colors:
                return None
            out.append((component, [cell], (endpoint_colors[0], endpoint_colors[1])))
            continue

        degree = {cell: sum((nbr in comp_set) for nbr in _neighbors4(cell)) for cell in comp_set}
        if any(d not in (1, 2) for d in degree.values()):
            return None
        terminals = [cell for cell, d in degree.items() if d == 1]
        if len(terminals) != 2:
            return None

        terminal_set = set(terminals)
        endpoint_colors: list[int] = []
        for cell in component:
            external = [
                (rr, cc)
                for rr, cc in _neighbors4(cell)
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in comp_set and int(grid[rr, cc]) != int(bg)
            ]
            if cell in terminal_set:
                if len(external) != 1:
                    return None
                endpoint_colors.append(int(grid[external[0]]))
            elif external:
                return None

        if len(endpoint_colors) != 2 or len(set(endpoint_colors)) != 2 or int(color) in endpoint_colors:
            return None

        ordered = _order_path(component)
        start = ordered[0]
        end = ordered[-1]
        start_endpoint = [
            nbr
            for nbr in _neighbors4(start)
            if 0 <= nbr[0] < h and 0 <= nbr[1] < w and nbr not in comp_set and int(grid[nbr]) != int(bg)
        ]
        end_endpoint = [
            nbr
            for nbr in _neighbors4(end)
            if 0 <= nbr[0] < h and 0 <= nbr[1] < w and nbr not in comp_set and int(grid[nbr]) != int(bg)
        ]
        out.append((component, ordered, (int(grid[start_endpoint[0]]), int(grid[end_endpoint[0]]))))

    return out


def _find_path_color(
    grid: np.ndarray,
    bg: int,
    active_mask: np.ndarray,
) -> tuple[int, list[tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]]]:
    candidates: list[tuple[int, list[tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]]]] = []
    colors = sorted(set(int(v) for v in grid[active_mask]) - {int(bg)})
    for color in colors:
        data = _path_candidate_data(grid, color, bg, active_mask)
        if data is not None:
            candidates.append((int(color), data))

    assert len(candidates) == 1, "expected exactly one color to behave like the path carrier"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate endpoint colors inward along each thin path and use the corner legend to color odd midpoints.

    Input structure:
    - The background is the dominant color.
    - A 3x3 legend sits in the upper-left corner; each row shows two endpoint colors
      followed by their mixed result color, and the legend is separated from the rest
      of the grid by a background row or column.
    - Elsewhere, one non-background color forms several one-cell-wide orthogonal paths.
    - Each path has exactly two colored endpoint pixels attached to its ends, and those
      endpoint colors differ from the background and from the shared path color.

    Rule:
    - Read the 3x3 legend as an unordered mapping from endpoint-color pairs to a mix color.
    - For every path, identify its ordered chain of path-color cells and its two endpoint colors.
    - Recolor the first half of the path cells from one endpoint side with that endpoint's color,
      and recolor the second half from the opposite side with the other endpoint's color.
    - If the number of path cells is odd, recolor the unique middle path cell to the legend's
      mix color for that unordered endpoint pair. Endpoints and the legend remain unchanged.

    Key insight:
    - The thin carrier color marks a chain that must be split symmetrically from its two ends,
      but the center of an odd-length chain is not owned by either endpoint: it is decoded from
      the compact 3x3 legend in the corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    mix_table = _parse_mix_table(g, bg)

    active = np.ones_like(g, dtype=bool)
    active[:3, :3] = False
    path_color, path_data = _find_path_color(g, bg, active)

    out = g.copy()
    for _component, ordered, (color_a, color_b) in path_data:
        length = len(ordered)
        half = length // 2
        for idx in range(half):
            out[ordered[idx]] = int(color_a)
            out[ordered[-1 - idx]] = int(color_b)
        if length % 2 == 1:
            out[ordered[half]] = int(mix_table[(int(color_a), int(color_b))])

    return out


# --------------- generator ---------------

GRID_SIDE_RANGE = (18, 30)
PATH_COUNT_RANGE = (2, 4)
PATH_LENGTH_BASE_RANGE = (4, 11)
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 500

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PERPENDICULARS = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}


def _step(cell: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    return cell[0] + direction[0], cell[1] + direction[1]


def _shape_line(length: int) -> list[tuple[int, int]]:
    direction = random.choice(DIRS)
    pts = [(0, 0)]
    cur = (0, 0)
    for _ in range(length - 1):
        cur = _step(cur, direction)
        pts.append(cur)
    return pts


def _shape_l(length: int) -> list[tuple[int, int]]:
    splits = [(a, length - a + 1) for a in range(2, length) if length - a + 1 >= 2]
    seg_a, seg_b = random.choice(splits)
    dir_a = random.choice(DIRS)
    dir_b = random.choice(PERPENDICULARS[dir_a])

    pts = [(0, 0)]
    cur = (0, 0)
    for direction, steps in ((dir_a, seg_a - 1), (dir_b, seg_b - 1)):
        for _ in range(steps):
            cur = _step(cur, direction)
            pts.append(cur)
    return pts


def _shape_z(length: int) -> list[tuple[int, int]]:
    splits = [
        (a, b, length - a - b + 2)
        for a in range(2, length)
        for b in range(2, length)
        if length - a - b + 2 >= 2
    ]
    seg_a, seg_b, seg_c = random.choice(splits)
    dir_a = random.choice(DIRS)
    dir_b = random.choice(PERPENDICULARS[dir_a])
    dir_c = dir_a

    pts = [(0, 0)]
    cur = (0, 0)
    for direction, steps in ((dir_a, seg_a - 1), (dir_b, seg_b - 1), (dir_c, seg_c - 1)):
        for _ in range(steps):
            cur = _step(cur, direction)
            pts.append(cur)
    return pts


def _build_path_shape(
    length: int,
    shape_kind: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], int, int] | None:
    for _ in range(100):
        if shape_kind == 0:
            ordered = _shape_line(length)
        elif shape_kind == 1:
            ordered = _shape_l(length)
        else:
            ordered = _shape_z(length)

        if len(set(ordered)) != len(ordered):
            continue

        start_dir = (ordered[1][0] - ordered[0][0], ordered[1][1] - ordered[0][1])
        end_dir = (ordered[-1][0] - ordered[-2][0], ordered[-1][1] - ordered[-2][1])
        endpoint_a = (ordered[0][0] - start_dir[0], ordered[0][1] - start_dir[1])
        endpoint_b = (ordered[-1][0] + end_dir[0], ordered[-1][1] + end_dir[1])
        if endpoint_a in ordered or endpoint_b in ordered or endpoint_a == endpoint_b:
            continue

        path_set = set(ordered)
        if sum((nbr in path_set) for nbr in _neighbors4(endpoint_a)) != 1:
            continue
        if sum((nbr in path_set) for nbr in _neighbors4(endpoint_b)) != 1:
            continue

        footprint = ordered + [endpoint_a, endpoint_b]
        min_r = min(r for r, _c in footprint)
        max_r = max(r for r, _c in footprint)
        min_c = min(c for _r, c in footprint)
        max_c = max(c for _r, c in footprint)

        ordered = [(r - min_r, c - min_c) for r, c in ordered]
        endpoint_a = (endpoint_a[0] - min_r, endpoint_a[1] - min_c)
        endpoint_b = (endpoint_b[0] - min_r, endpoint_b[1] - min_c)
        return ordered, endpoint_a, endpoint_b, max_r - min_r + 1, max_c - min_c + 1

    return None


def _buffer_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    buffered = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    buffered.add((rr, cc))
    return buffered


def _choose_pairs(endpoint_colors: list[int]) -> list[tuple[int, int]]:
    all_pairs = list(combinations(endpoint_colors, 2))
    valid = [triplet for triplet in combinations(all_pairs, 3) if set(sum(triplet, ())) == set(endpoint_colors)]
    return list(random.choice(valid))


def _parities_for_path_count(path_count: int) -> list[int]:
    if path_count == 2:
        parities = [1, 0]
    elif path_count == 3:
        parities = [1, 0, random.randint(0, 1)]
    else:
        parities = [1, 1, 0, 0]
    random.shuffle(parities)
    return parities


def _sample_length_with_parity(parity: int) -> int:
    base = random.randint(*PATH_LENGTH_BASE_RANGE)
    return int(base + ((int(parity) - base) % 2))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    bg, path_color = random.sample(list(range(10)), 2)

    available = [color for color in range(10) if color not in (bg, path_color)]
    endpoint_colors = random.sample(available, 4)
    remaining = [color for color in available if color not in endpoint_colors]
    mix_colors = random.sample(remaining, 3)
    pairs = _choose_pairs(endpoint_colors)

    grid = np.full((h, w), int(bg), dtype=int)
    rows = list(zip(pairs, mix_colors))
    random.shuffle(rows)
    for r, ((color_a, color_b), mix_color) in enumerate(rows):
        grid[r, 0] = int(color_a)
        grid[r, 1] = int(color_b)
        grid[r, 2] = int(mix_color)

    path_count = random.randint(*PATH_COUNT_RANGE)
    parities = _parities_for_path_count(path_count)

    blocked: set[tuple[int, int]] = {(r, c) for r in range(5) for c in range(5)}
    for path_idx in range(path_count):
        pair = random.choice(pairs)
        if random.random() < 0.5:
            endpoint_a, endpoint_b = pair
        else:
            endpoint_b, endpoint_a = pair

        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            length = _sample_length_with_parity(parities[path_idx])
            shape_kind = random.choices([0, 1, 2], weights=[2, 5, 3], k=1)[0]
            shape = _build_path_shape(length, shape_kind)
            if shape is None:
                continue

            ordered, local_a, local_b, shape_h, shape_w = shape
            if h - shape_h < 4 or w - shape_w < 4:
                continue

            top = random.randint(4, h - shape_h)
            left = random.randint(4, w - shape_w)
            path_cells = [(top + r, left + c) for r, c in ordered]
            endpoint_cell_a = (top + local_a[0], left + local_a[1])
            endpoint_cell_b = (top + local_b[0], left + local_b[1])
            footprint = set(path_cells + [endpoint_cell_a, endpoint_cell_b])
            if footprint & blocked:
                continue

            buffered = _buffer_cells(footprint, h, w)
            if buffered & blocked:
                continue

            for r, c in path_cells:
                grid[r, c] = int(path_color)
            grid[endpoint_cell_a] = int(endpoint_a)
            grid[endpoint_cell_b] = int(endpoint_b)
            blocked |= buffered
            placed = True
            break

        if not placed:
            return None

    try:
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-mixing ARC input")
