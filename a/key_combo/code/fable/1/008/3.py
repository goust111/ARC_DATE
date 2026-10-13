from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]

BG = 0
GREY = 5
FOUR_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_DIRS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _mask_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    dirs = FOUR_DIRS if connectivity == 4 else EIGHT_DIRS
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        q = deque([(rr, cc)])
        coords: list[Coord] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc + dc
                if not _in_bounds(mask.shape, nr, nc) or seen[nr, nc] or not mask[nr, nc]:
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _color_components(grid: np.ndarray, color: int, connectivity: int = 4) -> list[np.ndarray]:
    return _mask_components(grid == int(color), connectivity)


def _touches_component_8(cell: Coord, component_set: set[Coord]) -> bool:
    r, c = cell
    return any((r + dr, c + dc) in component_set for dr, dc in EIGHT_DIRS)


def _components_touch_8(comp_a: set[Coord], comp_b: set[Coord]) -> bool:
    return any(_touches_component_8(cell, comp_b) for cell in comp_a)


def _line_colors(grid: np.ndarray, coords: np.ndarray) -> list[int]:
    rows = sorted({int(r) for r in coords[:, 0]})
    cols = sorted({int(c) for c in coords[:, 1]})
    assert len(rows) == 1 or len(cols) == 1, "legend must be a straight line"
    assert not (len(rows) == 1 and len(cols) == 1), "legend needs multiple pixels"

    if len(rows) == 1:
        row = rows[0]
        ordered = sorted(int(c) for c in coords[:, 1])
        assert ordered == list(range(ordered[0], ordered[-1] + 1)), "legend pixels must be consecutive"
        return [int(grid[row, c]) for c in ordered]

    col = cols[0]
    ordered = sorted(int(r) for r in coords[:, 0])
    assert ordered == list(range(ordered[0], ordered[-1] + 1)), "legend pixels must be consecutive"
    return [int(grid[r, col]) for r in ordered]


def _marker_and_legend(grid: np.ndarray, bg: int) -> tuple[list[int], Coord, int]:
    color_mask = (grid != bg) & (grid != GREY)
    comps = _mask_components(color_mask, connectivity=4)
    legend_parts = [comp for comp in comps if len(comp) > 1]
    marker_parts = [comp for comp in comps if len(comp) == 1]
    assert len(legend_parts) == 1, "expected exactly one legend"
    assert len(marker_parts) == 1, "expected exactly one marker"

    legend = legend_parts[0]
    marker = marker_parts[0]
    legend_colors = _line_colors(grid, legend)
    assert len(legend_colors) == len(set(legend_colors)), "legend colors must be distinct"
    marker_coord = tuple(map(int, marker[0]))
    marker_color = int(grid[marker_coord])
    assert marker_color in legend_colors, "marker color must appear in legend"
    return legend_colors, marker_coord, marker_color


def _gray_graph(grid: np.ndarray, marker_coord: Coord) -> tuple[list[np.ndarray], list[list[int]], int]:
    comps = _color_components(grid, GREY, connectivity=4)
    assert len(comps) >= 2, "expected a chain of gray blocks"
    comp_sets = [{tuple(map(int, cell)) for cell in comp} for comp in comps]

    adjacency = [[] for _ in comps]
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if _components_touch_8(comp_sets[i], comp_sets[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)

    degrees = [len(neigh) for neigh in adjacency]
    assert degrees.count(1) == 2, "gray-block graph must have two endpoints"
    assert all(deg in (1, 2) for deg in degrees), "gray-block graph must be an unbranched chain"

    start_candidates = [
        i
        for i, comp_set in enumerate(comp_sets)
        if _touches_component_8(marker_coord, comp_set)
    ]
    assert len(start_candidates) == 1, "marker must touch exactly one endpoint block"
    start = start_candidates[0]
    assert len(adjacency[start]) == 1, "marker must indicate a chain endpoint"

    visited = set()
    order: list[int] = []
    prev = -1
    cur = start
    while True:
        visited.add(cur)
        order.append(cur)
        next_nodes = [nxt for nxt in adjacency[cur] if nxt != prev]
        if not next_nodes:
            break
        assert len(next_nodes) == 1, "path walk must be unique"
        prev, cur = cur, next_nodes[0]

    assert len(visited) == len(comps), "gray chain must be connected"
    return comps, adjacency, start


def _recolor_chain(
    grid: np.ndarray,
    gray_components: list[np.ndarray],
    adjacency: list[list[int]],
    start: int,
    legend_colors: list[int],
    marker_coord: Coord,
    marker_color: int,
) -> np.ndarray:
    order: list[int] = []
    prev = -1
    cur = start
    while True:
        order.append(cur)
        next_nodes = [nxt for nxt in adjacency[cur] if nxt != prev]
        if not next_nodes:
            break
        prev, cur = cur, next_nodes[0]

    start_idx = legend_colors.index(marker_color)
    out = np.array(grid, dtype=int, copy=True)
    out[marker_coord] = BG
    for step, comp_idx in enumerate(order):
        color = legend_colors[(start_idx + step) % len(legend_colors)]
        coords = gray_components[comp_idx]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the gray-block chain by relaying the marker color through the legend's cyclic successor order.

    Input structure:
    - The background is black.
    - One isolated straight legend bar contains several distinct colors in order.
    - Several gray 4-connected blocks form a non-branching chain through block-to-block adjacency, and one colored marker pixel touches an endpoint block.

    Rule:
    - Keep the legend unchanged.
    - Recolor the endpoint block touching the marker with the marker color, then erase the marker to black.
    - Walk from that endpoint block along the gray-block chain; each next block takes the next legend color after the current one, wrapping from the legend end back to the legend start.

    Key insight:
    - The legend is a cyclic successor table, so the marker fixes the entry point in that cycle and the chain order comes from block adjacency rather than from global row or column sorting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _get_bg(g)
    assert bg == BG, "background must be black"

    legend_colors, marker_coord, marker_color = _marker_and_legend(g, bg)
    gray_components, adjacency, start = _gray_graph(g, marker_coord)
    return _recolor_chain(g, gray_components, adjacency, start, legend_colors, marker_coord, marker_color)


def _count_turns(path: list[Coord]) -> int:
    turns = 0
    for i in range(2, len(path)):
        d1 = (path[i - 1][0] - path[i - 2][0], path[i - 1][1] - path[i - 2][1])
        d2 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        if d1 != d2:
            turns += 1
    return turns


def _generate_diagonal_path(length: int, side: int) -> list[Coord] | None:
    cells = [(r, c) for r in range(side) for c in range(side)]
    random.shuffle(cells)

    def can_add(nxt: Coord, used: set[Coord], prev: Coord) -> bool:
        r, c = nxt
        if not (0 <= r < side and 0 <= c < side) or nxt in used:
            return False
        for other in used:
            if other == prev:
                continue
            if abs(other[0] - r) == 1 and abs(other[1] - c) == 1:
                return False
        return True

    def search(path: list[Coord], used: set[Coord], turns: int) -> list[Coord] | None:
        if len(path) == length:
            return list(path) if turns >= 1 else None

        dirs = list(DIAG_DIRS)
        random.shuffle(dirs)
        for dr, dc in dirs:
            nxt = (path[-1][0] + dr, path[-1][1] + dc)
            if not can_add(nxt, used, path[-1]):
                continue
            new_turns = turns
            if len(path) >= 2:
                prev_dir = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
                if prev_dir != (dr, dc):
                    new_turns += 1
            path.append(nxt)
            used.add(nxt)
            found = search(path, used, new_turns)
            if found is not None:
                return found
            used.remove(nxt)
            path.pop()
        return None

    for start in cells:
        found = search([start], {start}, 0)
        if found is not None and _count_turns(found) >= 1:
            return found
    return None


def _path_to_components(path: list[Coord], block_size: int) -> list[list[Coord]]:
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    r0 = min(rows)
    c0 = min(cols)
    components: list[list[Coord]] = []
    for pr, pc in path:
        top = (pr - r0) * block_size
        left = (pc - c0) * block_size
        coords = [
            (top + dr, left + dc)
            for dr in range(block_size)
            for dc in range(block_size)
        ]
        components.append(coords)
    return components


def _marker_candidates(components: list[list[Coord]], start_index: int) -> list[Coord]:
    comp_sets = [set(comp) for comp in components]
    gray_cells = set().union(*comp_sets)
    candidates: set[Coord] = set()
    for r, c in components[start_index]:
        for dr, dc in FOUR_DIRS:
            nr = r + dr
            nc = c + dc
            if (nr, nc) in gray_cells:
                continue
            touched = [
                idx
                for idx, comp_set in enumerate(comp_sets)
                if _touches_component_8((nr, nc), comp_set)
            ]
            if touched == [start_index]:
                candidates.add((nr, nc))
    return sorted(candidates)


def _tight_grid(cells_by_color: list[tuple[int, list[Coord]]]) -> tuple[np.ndarray, dict[int, list[Coord]]]:
    all_cells = [cell for _color, coords in cells_by_color for cell in coords]
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_r = max(r for r, _ in all_cells)
    max_c = max(c for _, c in all_cells)
    grid = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    shifted: dict[int, list[Coord]] = {}
    for color, coords in cells_by_color:
        moved = [(r - min_r, c - min_c) for r, c in coords]
        shifted[color] = moved
        for rr, cc in moved:
            grid[rr, cc] = color
    return grid, shifted


def _sprite_cells(sprite: np.ndarray) -> list[tuple[Coord, int]]:
    cells: list[tuple[Coord, int]] = []
    for r, c in np.argwhere(sprite != BG):
        rr = int(r)
        cc = int(c)
        cells.append(((rr, cc), int(sprite[rr, cc])))
    return cells


def _can_place(
    base_cells: set[Coord],
    sprite_cells: list[tuple[Coord, int]],
    pos: Coord,
    shape: tuple[int, int],
    canvas: tuple[int, int],
) -> bool:
    r0, c0 = pos
    sh, sw = shape
    h, w = canvas
    if r0 < 0 or c0 < 0 or r0 + sh > h or c0 + sw > w:
        return False
    for (dr, dc), _color in sprite_cells:
        rr = r0 + dr
        cc = c0 + dc
        if (rr, cc) in base_cells:
            return False
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if (nr, nc) in base_cells:
                    return False
    return True


def _paste_sprite(dst: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    nonzero = sprite != BG
    view = dst[top : top + sprite.shape[0], left : left + sprite.shape[1]]
    view[nonzero] = sprite[nonzero]


def _layout_case(chain_sprite: np.ndarray, legend_sprite: np.ndarray) -> np.ndarray | None:
    chain_cells = _sprite_cells(chain_sprite)
    legend_cells = _sprite_cells(legend_sprite)
    chain_shape = chain_sprite.shape
    legend_shape = legend_sprite.shape
    gap = 2
    padding = 1

    stacked = random.choice([False, True])
    if stacked:
        min_h = chain_shape[0] + legend_shape[0] + gap + 2 * padding
        min_w = max(chain_shape[1], legend_shape[1]) + 2 * padding
    else:
        min_h = max(chain_shape[0], legend_shape[0]) + 2 * padding
        min_w = chain_shape[1] + legend_shape[1] + gap + 2 * padding

    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, min(30, min_h + 6))
    w = random.randint(min_w, min(30, min_w + 6))
    canvas = (h, w)

    chain_positions = [
        (r, c)
        for r in range(h - chain_shape[0] + 1)
        for c in range(w - chain_shape[1] + 1)
    ]
    random.shuffle(chain_positions)

    for chain_pos in chain_positions[:80]:
        placed_chain = {(chain_pos[0] + dr, chain_pos[1] + dc) for (dr, dc), _ in chain_cells}
        legend_positions = [
            (r, c)
            for r in range(h - legend_shape[0] + 1)
            for c in range(w - legend_shape[1] + 1)
        ]
        random.shuffle(legend_positions)
        for legend_pos in legend_positions:
            if not _can_place(placed_chain, legend_cells, legend_pos, legend_shape, canvas):
                continue
            grid = np.full(canvas, BG, dtype=int)
            _paste_sprite(grid, chain_sprite, chain_pos[0], chain_pos[1])
            _paste_sprite(grid, legend_sprite, legend_pos[0], legend_pos[1])
            return grid
    return None


def _component_centroid(coords: list[Coord]) -> tuple[float, float]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (sum(rs) / len(rs), sum(cs) / len(cs))


def _render_from_order(
    grid: np.ndarray,
    components: list[list[Coord]],
    order: list[int],
    legend_colors: list[int],
    start_idx: int,
    marker: Coord,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[marker] = BG
    for step, comp_idx in enumerate(order):
        color = legend_colors[(start_idx + step) % len(legend_colors)]
        for r, c in components[comp_idx]:
            out[r, c] = color
    return out


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    components: list[list[Coord]],
    path_order: list[int],
    legend_colors: list[int],
    marker_index: int,
    marker: Coord,
) -> bool:
    if marker_index == 0:
        return False
    if len(path_order) <= len(legend_colors):
        return False
    if _count_turns([_component_centroid(components[i]) for i in path_order]) < 1:
        return False

    if not np.array_equal(main(grid), expected):
        return False

    centroids = [_component_centroid(comp) for comp in components]
    candidate_orders = [
        sorted(range(len(components)), key=lambda i: (centroids[i][0], centroids[i][1])),
        sorted(range(len(components)), key=lambda i: (centroids[i][1], centroids[i][0])),
    ]
    candidate_orders += [list(reversed(order)) for order in candidate_orders]
    for order in candidate_orders:
        if np.array_equal(
            _render_from_order(grid, components, order, legend_colors, marker_index, marker),
            expected,
        ):
            return False

    if np.array_equal(
        _render_from_order(grid, components, path_order, legend_colors, 0, marker),
        expected,
    ):
        return False

    return True


def _generate_case() -> np.ndarray | None:
    legend_len = random.randint(3, 5)
    block_size = random.choice([1, 2])
    chain_len = random.randint(legend_len + 1, min(8, legend_len + 3))
    path = _generate_diagonal_path(chain_len, side=max(6, chain_len + 1))
    if path is None:
        return None
    if random.choice([False, True]):
        path = list(reversed(path))

    components = _path_to_components(path, block_size)
    marker_candidates = _marker_candidates(components, 0)
    if not marker_candidates:
        return None
    marker = random.choice(marker_candidates)

    legend_colors = random.sample(LEGEND_COLORS, legend_len)
    marker_index = random.randint(1, legend_len - 1)
    marker_color = legend_colors[marker_index]

    chain_sprite, _shifted = _tight_grid(
        [(GREY, [cell for comp in components for cell in comp]), (marker_color, [marker])]
    )

    legend_sprite = (
        np.array(legend_colors, dtype=int)[:, None]
        if random.choice([False, True])
        else np.array(legend_colors, dtype=int)[None, :]
    )
    grid = _layout_case(chain_sprite, legend_sprite)
    if grid is None:
        return None

    gray_components = [comp for comp in _color_components(grid, GREY, connectivity=4)]
    if len(gray_components) != chain_len:
        return None
    legend_colors_found, marker_found, marker_color_found = _marker_and_legend(grid, BG)
    if legend_colors_found != legend_colors or marker_color_found != marker_color:
        return None

    parsed_gray = _color_components(grid, GREY, connectivity=4)
    gray_sets = [set(map(tuple, comp.tolist())) for comp in parsed_gray]
    start_idx = None
    for idx, comp_set in enumerate(gray_sets):
        if _touches_component_8(marker_found, comp_set):
            start_idx = idx
            break
    if start_idx is None:
        return None

    prev = -1
    cur = start_idx
    path_order: list[int] = []
    adjacency = [[] for _ in gray_sets]
    for i in range(len(gray_sets)):
        for j in range(i + 1, len(gray_sets)):
            if _components_touch_8(gray_sets[i], gray_sets[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)
    while True:
        path_order.append(cur)
        nxt = [node for node in adjacency[cur] if node != prev]
        if not nxt:
            break
        if len(nxt) != 1:
            return None
        prev, cur = cur, nxt[0]

    parsed_components = [sorted(map(tuple, comp.tolist())) for comp in parsed_gray]
    component_sets = [set(comp) for comp in parsed_components]
    if len(component_sets) != chain_len:
        return None
    parsed_lists = [sorted(comp) for comp in component_sets]
    centroids = [_component_centroid(comp) for comp in parsed_lists]
    mapped = sorted(range(chain_len), key=lambda i: (centroids[i][0], centroids[i][1]))
    if len(mapped) != chain_len:
        return None

    expected = _render_from_order(
        grid,
        parsed_lists,
        path_order,
        legend_colors,
        marker_index,
        marker_found,
    )
    if not _validate_case(
        grid,
        expected,
        parsed_lists,
        path_order,
        legend_colors,
        marker_index,
        marker_found,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-relay ARC input")
