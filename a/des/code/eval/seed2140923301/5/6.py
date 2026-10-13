from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
Quadrant = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_dividers(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        r for r in range(h) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    for r in row_candidates:
        color = int(grid[r, 0])
        for c in col_candidates:
            if int(grid[0, c]) == color:
                return int(r), int(c), color
    raise AssertionError("expected one full-row and one full-column divider of the same color")


def _parse_problem(
    grid: np.ndarray,
) -> tuple[int, int, int, list[tuple[int, int]], dict[int, Coord]]:
    bg = _get_bg(grid)
    divider_row, divider_col, grey = _find_dividers(grid, bg)
    comps = _components_mask((grid != bg) & (grid != grey))

    pairs_with_keys: list[tuple[tuple[int, int], int, int]] = []
    endpoints: dict[int, Coord] = {}

    for comp in comps:
        coords = [(int(r), int(c)) for r, c in comp.tolist()]
        colors = [int(grid[r, c]) for r, c in coords]
        if len(coords) == 1:
            color = colors[0]
            assert color not in endpoints, "endpoint colors must be unique"
            endpoints[color] = coords[0]
            continue

        if len(coords) != 2:
            raise AssertionError("non-divider foreground must be singleton endpoints or 2-cell legend pairs")

        (r0, c0), (r1, c1) = coords
        assert abs(r0 - r1) + abs(c0 - c1) == 1, "legend cells must be orthogonally adjacent"
        assert colors[0] != colors[1], "legend pair colors must differ"

        if r0 == r1:
            if c0 < c1:
                src_color, dst_color = colors[0], colors[1]
            else:
                src_color, dst_color = colors[1], colors[0]
        else:
            if r0 < r1:
                src_color, dst_color = colors[0], colors[1]
            else:
                src_color, dst_color = colors[1], colors[0]

        pairs_with_keys.append(((min(r0, r1), min(c0, c1)), src_color, dst_color))

    pairs_with_keys.sort(key=lambda item: item[0])
    pairs = [(src_color, dst_color) for _, src_color, dst_color in pairs_with_keys]
    pair_colors = {color for pair in pairs for color in pair}
    assert pair_colors == set(endpoints), "legend colors and endpoint colors must match exactly"
    return divider_row, divider_col, grey, pairs, endpoints


def _quadrant(coord: Coord, divider_row: int, divider_col: int) -> Quadrant:
    r, c = coord
    assert r != divider_row and c != divider_col, "endpoints cannot lie on dividers"
    return (0 if r < divider_row else 1, 0 if c < divider_col else 1)


def _segment_cells(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segments must be axis-aligned")


def _route_cells(kind: str, src: Coord, dst: Coord, divider_col: int) -> list[Coord]:
    if kind == "adj":
        bend = (src[0], dst[1])
        return _segment_cells(src, bend) + _segment_cells(bend, dst)[1:]
    if kind == "diag":
        bend_a = (src[0], divider_col)
        bend_b = (dst[0], divider_col)
        return (
            _segment_cells(src, bend_a)
            + _segment_cells(bend_a, bend_b)[1:]
            + _segment_cells(bend_b, dst)[1:]
        )
    raise AssertionError("unknown route kind")


def _render_solution(
    grid: np.ndarray,
    divider_row: int,
    divider_col: int,
    pairs: list[tuple[int, int]],
    endpoints: dict[int, Coord],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for src_color, dst_color in pairs:
        assert src_color in endpoints and dst_color in endpoints
        src = endpoints[src_color]
        dst = endpoints[dst_color]
        q_src = _quadrant(src, divider_row, divider_col)
        q_dst = _quadrant(dst, divider_row, divider_col)
        quad_dist = abs(q_src[0] - q_dst[0]) + abs(q_src[1] - q_dst[1])
        if quad_dist == 1:
            cells = _route_cells("adj", src, dst, divider_col)
        elif quad_dist == 2:
            cells = _route_cells("diag", src, dst, divider_col)
        else:
            raise AssertionError("paired endpoints must be in adjacent or diagonal quadrants")

        for r, c in cells:
            out[r, c] = src_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect color-paired singleton endpoints across quadrant dividers.

    The input contains:
    - a dominant background,
    - one full-row and one full-column divider of the same grey color,
    - singleton colored endpoints scattered in the quadrants,
    - a corner legend made of 2-cell color pairs that specifies which endpoint
      colors should be connected.

    Adjacent-quadrant pairs use an L path that goes horizontal first, then
    vertical. Diagonal-quadrant pairs use a Z path that goes from the source
    horizontally to the vertical divider, then vertically on that divider, then
    horizontally to the target. Paths use the legend pair's first color.
    """
    g = np.asarray(grid, dtype=int)
    divider_row, divider_col, _grey, pairs, endpoints = _parse_problem(g)
    return _render_solution(g, divider_row, divider_col, pairs, endpoints)


GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
PAIR_COUNT_CHOICES = (2, 3)
GEN_ATTEMPTS = 400
PAIR_PLACE_ATTEMPTS = 300
ALL_COLORS = tuple(range(10))
ADJACENT_OPTIONS: tuple[tuple[Quadrant, Quadrant], ...] = (
    ((0, 0), (0, 1)),
    ((0, 1), (0, 0)),
    ((1, 0), (1, 1)),
    ((1, 1), (1, 0)),
    ((0, 0), (1, 0)),
    ((1, 0), (0, 0)),
    ((0, 1), (1, 1)),
    ((1, 1), (0, 1)),
)
DIAGONAL_OPTIONS: tuple[tuple[Quadrant, Quadrant], ...] = (
    ((0, 0), (1, 1)),
    ((1, 1), (0, 0)),
    ((0, 1), (1, 0)),
    ((1, 0), (0, 1)),
)
LEGEND_CORNERS = ("tl", "tr", "bl", "br")


def _expand4(cells: set[Coord], h: int, w: int) -> set[Coord]:
    expanded: set[Coord] = set()
    for r, c in cells:
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                expanded.add((nr, nc))
    return expanded


def _corner_halves(corner: str) -> tuple[bool, bool]:
    return corner in {"tl", "tr"}, corner in {"tl", "bl"}


def _candidate_cells(q: Quadrant, h: int, w: int, divider_row: int, divider_col: int) -> list[Coord]:
    if q == (0, 0):
        rows = range(1, divider_row - 1)
        cols = range(1, divider_col - 1)
    elif q == (0, 1):
        rows = range(1, divider_row - 1)
        cols = range(divider_col + 2, w - 1)
    elif q == (1, 0):
        rows = range(divider_row + 2, h - 1)
        cols = range(1, divider_col - 1)
    else:
        rows = range(divider_row + 2, h - 1)
        cols = range(divider_col + 2, w - 1)
    return [(r, c) for r in rows for c in cols]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    legend_corner = rng.choice(LEGEND_CORNERS)
    legend_height = 2 * pair_count - 1
    legend_on_top, legend_on_left = _corner_halves(legend_corner)

    top_need = legend_height + 5 if legend_on_top else 6
    bottom_need = legend_height + 5 if not legend_on_top else 6
    left_need = 7 if legend_on_left else 6
    right_need = 7 if not legend_on_left else 6

    min_h = max(GRID_H_RANGE[0], top_need + bottom_need + 1)
    min_w = max(GRID_W_RANGE[0], left_need + right_need + 1)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])
    divider_row = rng.randint(top_need, h - bottom_need - 1)
    divider_col = rng.randint(left_need, w - right_need - 1)

    bg, grey = rng.sample(ALL_COLORS, 2)
    endpoint_palette = [color for color in ALL_COLORS if color not in {bg, grey}]
    colors = rng.sample(endpoint_palette, 2 * pair_count)

    diag_pick = rng.choice(DIAGONAL_OPTIONS)
    adj_picks = rng.sample(list(ADJACENT_OPTIONS), pair_count - 1)
    relation_specs = [("diag", diag_pick)] + [("adj", pick) for pick in adj_picks]
    rng.shuffle(relation_specs)

    specs: list[dict[str, object]] = []
    for i, (kind, (src_q, dst_q)) in enumerate(relation_specs):
        specs.append(
            {
                "kind": kind,
                "src_q": src_q,
                "dst_q": dst_q,
                "src_color": colors[2 * i],
                "dst_color": colors[2 * i + 1],
            }
        )

    grid = np.full((h, w), bg, dtype=int)
    grid[divider_row, :] = grey
    grid[:, divider_col] = grey

    legend_r0 = 1 if legend_on_top else h - legend_height - 1
    legend_c0 = 1 if legend_on_left else w - 3
    legend_cells: set[Coord] = set()
    for i, spec in enumerate(specs):
        rr = legend_r0 + 2 * i
        left = legend_c0
        right = legend_c0 + 1
        src_color = int(spec["src_color"])
        dst_color = int(spec["dst_color"])
        grid[rr, left] = src_color
        grid[rr, right] = dst_color
        legend_cells.add((rr, left))
        legend_cells.add((rr, right))

    quadrant_cells = {
        q: _candidate_cells(q, h, w, divider_row, divider_col)
        for q in ((0, 0), (0, 1), (1, 0), (1, 1))
    }
    if any(not cells for cells in quadrant_cells.values()):
        return None

    endpoint_cells: set[Coord] = set()
    endpoint_buffer = set()
    legend_buffer = _expand4(legend_cells, h, w)
    used_path_cells: set[Coord] = set()
    used_path_buffer: set[Coord] = set()
    endpoints: dict[int, Coord] = {}

    place_order = sorted(range(pair_count), key=lambda i: 0 if specs[i]["kind"] == "diag" else 1)
    for idx in place_order:
        spec = specs[idx]
        kind = str(spec["kind"])
        src_q = spec["src_q"]
        dst_q = spec["dst_q"]
        src_options = quadrant_cells[src_q]  # type: ignore[index]
        dst_options = quadrant_cells[dst_q]  # type: ignore[index]

        placed = False
        for _ in range(PAIR_PLACE_ATTEMPTS):
            src = rng.choice(src_options)
            dst = rng.choice(dst_options)
            if src == dst:
                continue
            if src in legend_buffer or dst in legend_buffer:
                continue
            if src in endpoint_buffer or dst in endpoint_buffer:
                continue
            if src in used_path_buffer or dst in used_path_buffer:
                continue
            if kind == "adj" and (src[0] == dst[0] or src[1] == dst[1]):
                continue

            cells = _route_cells(kind, src, dst, divider_col)
            cell_set = set(cells)
            interior = cell_set - {src, dst}

            if cell_set & used_path_buffer:
                continue
            if interior & legend_buffer:
                continue
            if interior & endpoint_buffer:
                continue

            grid[src] = int(spec["src_color"])
            grid[dst] = int(spec["dst_color"])
            endpoints[int(spec["src_color"])] = src
            endpoints[int(spec["dst_color"])] = dst
            endpoint_cells.update((src, dst))
            endpoint_buffer = _expand4(endpoint_cells, h, w)
            used_path_cells |= cell_set
            used_path_buffer = _expand4(used_path_cells, h, w)
            placed = True
            break

        if not placed:
            return None

    pair_list = [(int(spec["src_color"]), int(spec["dst_color"])) for spec in specs]
    expected = _render_solution(grid, divider_row, divider_col, pair_list, endpoints)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
