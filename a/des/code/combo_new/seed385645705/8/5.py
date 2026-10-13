from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]

BG_COLOR = 0
PATH_COLOR = 3
PORTAL_COLOR_POOL = [1, 2, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 500
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _touches_edge(cell: Coord, shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _coords_for_color(grid: np.ndarray, color: int) -> list[Coord]:
    return [tuple(map(int, coord)) for coord in np.argwhere(grid == int(color))]


def _is_consecutive(values: list[int]) -> bool:
    if not values:
        return False
    start = values[0]
    return values == list(range(start, start + len(values)))


def _is_path_candidate(coords: list[Coord], shape: tuple[int, int]) -> bool:
    if not coords:
        return False
    if len(coords) == 1:
        return _touches_edge(coords[0], shape)

    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    if len(rows) == 1:
        ordered_cols = sorted(c for _r, c in coords)
        return _is_consecutive(ordered_cols) and (
            _touches_edge((rows[0], ordered_cols[0]), shape)
            or _touches_edge((rows[0], ordered_cols[-1]), shape)
        )
    if len(cols) == 1:
        ordered_rows = sorted(r for r, _c in coords)
        return _is_consecutive(ordered_rows) and (
            _touches_edge((ordered_rows[0], cols[0]), shape)
            or _touches_edge((ordered_rows[-1], cols[0]), shape)
        )
    return False


def _find_path_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        coords = _coords_for_color(grid, color)
        if _is_path_candidate(coords, grid.shape):
            candidates.append(int(color))
    assert len(candidates) == 1, "expected exactly one edge-anchored straight path color"
    return candidates[0]


def _portal_map(grid: np.ndarray, bg: int, path_color: int) -> tuple[dict[Coord, Coord], set[Coord]]:
    mapping: dict[Coord, Coord] = {}
    portal_cells: set[Coord] = set()
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (int(bg), int(path_color))):
        coords = _coords_for_color(grid, color)
        assert len(coords) == 2, "each portal color must appear exactly twice"
        a, b = coords
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1, "portal pair pixels must be separate"
        mapping[a] = b
        mapping[b] = a
        portal_cells.add(a)
        portal_cells.add(b)
    return mapping, portal_cells


def _line_orientation(path_coords: list[Coord]) -> str:
    rows = {r for r, _c in path_coords}
    cols = {c for _r, c in path_coords}
    if len(path_coords) == 1:
        return "point"
    if len(rows) == 1:
        return "horizontal"
    if len(cols) == 1:
        return "vertical"
    raise AssertionError("path must be horizontal or vertical")


def _endpoint_and_direction(path_coords: list[Coord], portal_cells: set[Coord]) -> tuple[Coord, Coord]:
    if len(path_coords) == 1:
        cell = path_coords[0]
        candidates = []
        for dr, dc in DIR4:
            nbr = (cell[0] + dr, cell[1] + dc)
            if nbr in portal_cells:
                candidates.append((cell, (dr, dc)))
        assert len(candidates) == 1, "single-cell path must face exactly one portal"
        return candidates[0]

    orientation = _line_orientation(path_coords)
    candidates: list[tuple[Coord, Coord]] = []
    if orientation == "horizontal":
        row = path_coords[0][0]
        cols = sorted(c for _r, c in path_coords)
        left = (row, cols[0])
        right = (row, cols[-1])
        if (row, cols[0] - 1) in portal_cells:
            candidates.append((left, (0, -1)))
        if (row, cols[-1] + 1) in portal_cells:
            candidates.append((right, (0, 1)))
    else:
        col = path_coords[0][1]
        rows = sorted(r for r, _c in path_coords)
        top = (rows[0], col)
        bottom = (rows[-1], col)
        if (rows[0] - 1, col) in portal_cells:
            candidates.append((top, (-1, 0)))
        if (rows[-1] + 1, col) in portal_cells:
            candidates.append((bottom, (1, 0)))

    assert len(candidates) == 1, "path must have one portal directly ahead of an endpoint"
    return candidates[0]


def _trace_extension(shape: tuple[int, int], start: Coord, direction: Coord, portal_by_cell: dict[Coord, Coord]) -> list[Coord]:
    dr, dc = direction
    current = start
    drawn: list[Coord] = []
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (current[0], current[1], dr, dc)
        assert state not in seen_states, "portal loop detected"
        seen_states.add(state)

        ahead = (current[0] + dr, current[1] + dc)
        if not _in_bounds(shape, ahead[0], ahead[1]):
            break

        if ahead in portal_by_cell:
            pair = portal_by_cell[ahead]
            current = (pair[0] + dr, pair[1] + dc)
            if not _in_bounds(shape, current[0], current[1]):
                break
            drawn.append(current)
            continue

        current = ahead
        drawn.append(current)

    return drawn


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the straight green route by teleporting through same-colored portal pairs.

    Input structure:
    - The background is a dominant solid color.
    - One non-background color forms a single straight path segment that touches
      a grid edge and ends immediately before a portal.
    - Every other non-background color appears exactly twice as two isolated
      single-pixel portals, with each pair sharing a color.

    Rule:
    - Keep the existing path and all portal pixels.
    - Follow the path's travel direction from its exposed endpoint.
    - When the next cell in that direction is a portal pixel, jump to the
      same-colored partner and continue from the cell on the far side while
      preserving direction.
    - Otherwise extend the path one cell at a time until the next teleport or
      until the path reaches the grid boundary.

    Key insight:
    - The colored singleton pairs are not obstacles; they define a portal graph
      for one straight-moving path whose direction never changes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    path_color = _find_path_color(g, bg)
    portal_by_cell, portal_cells = _portal_map(g, bg, path_color)
    path_coords = _coords_for_color(g, path_color)
    path_end, direction = _endpoint_and_direction(path_coords, portal_cells)

    out = g.copy()
    for r, c in _trace_extension(g.shape, path_end, direction, portal_by_cell):
        assert (r, c) not in portal_cells, "path may not overwrite a portal"
        assert int(out[r, c]) in (int(bg), int(path_color)), "path may only extend through empty cells"
        out[r, c] = int(path_color)
    return out


def _sample_spaced_values(count: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    span = hi - lo + 1
    compressed = span - (count - 1) * (min_gap - 1)
    if compressed < count:
        return None
    picks = sorted(random.sample(range(compressed), count))
    return [lo + picks[i] + i * (min_gap - 1) for i in range(count)]


def _transform_grid(grid: np.ndarray, rotations: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, rotations)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _segment_starts_from_exits(exit_cols: list[int]) -> list[int]:
    return [0] + [int(c) + 1 for c in exit_cols]


def _render_expected(
    shape: tuple[int, int],
    rows: list[int],
    entry_cols: list[int],
    exit_cols: list[int],
    pair_positions: list[tuple[Coord, Coord, int]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    inp = np.full((h, w), BG_COLOR, dtype=int)
    out = np.full((h, w), BG_COLOR, dtype=int)

    starts = _segment_starts_from_exits(exit_cols)
    ends = [entry - 1 for entry in entry_cols] + [w - 1]

    for i, row in enumerate(rows):
        start = starts[i]
        end = ends[i]
        assert start <= end, "each path segment must contain at least one cell"
        out[row, start : end + 1] = PATH_COLOR

    inp[rows[0], 0 : entry_cols[0]] = PATH_COLOR

    for a, b, color in pair_positions:
        inp[a] = int(color)
        inp[b] = int(color)
        out[a] = int(color)
        out[b] = int(color)

    return inp, out


def _portal_cell_is_clear(grid: np.ndarray, cell: Coord) -> bool:
    r, c = cell
    if int(grid[r, c]) != BG_COLOR:
        return False
    h, w = grid.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if (rr, cc) == cell:
                continue
            if int(grid[rr, cc]) != BG_COLOR:
                return False
    return True


def _place_extra_pairs(inp: np.ndarray, out: np.ndarray, colors: list[int], blocked_rows: set[int]) -> bool:
    h, w = inp.shape
    allowed_cells = [(r, c) for r in range(h) for c in range(w) if r not in blocked_rows]
    for color in colors:
        placed = False
        for _ in range(200):
            first = random.choice(allowed_cells)
            second = random.choice(allowed_cells)
            if first == second:
                continue
            if abs(first[0] - second[0]) <= 1 and abs(first[1] - second[1]) <= 1:
                continue
            if not _portal_cell_is_clear(inp, first):
                continue
            inp[first] = int(color)
            out[first] = int(color)
            if not _portal_cell_is_clear(inp, second):
                inp[first] = BG_COLOR
                out[first] = BG_COLOR
                continue
            inp[second] = int(color)
            out[second] = int(color)
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_input_structure(inp: np.ndarray) -> bool:
    try:
        bg = _most_common_color(inp)
        path_color = _find_path_color(inp, bg)
        portal_by_cell, portal_cells = _portal_map(inp, bg, path_color)
        path_coords = _coords_for_color(inp, path_color)
        path_end, direction = _endpoint_and_direction(path_coords, portal_cells)
    except AssertionError:
        return False

    if int(np.count_nonzero(inp == bg)) <= int(np.count_nonzero(inp != bg)):
        return False

    allowed_portal = (path_end[0] + direction[0], path_end[1] + direction[1])
    for portal in portal_cells:
        neighbors = []
        for dr, dc in DIR4:
            nbr = (portal[0] + dr, portal[1] + dc)
            if _in_bounds(inp.shape, nbr[0], nbr[1]) and int(inp[nbr]) != bg:
                neighbors.append(nbr)
        if not neighbors:
            continue
        if portal != allowed_portal or neighbors != [path_end]:
            return False
        if int(inp[path_end]) != path_color:
            return False

    # A valid teleport chain should not loop.
    try:
        _trace_extension(inp.shape, path_end, direction, portal_by_cell)
    except AssertionError:
        return False
    return True


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(8, 22)
    w = random.randint(8, 22)
    max_chain = min(4, (h + 1) // 2 - 1)
    if max_chain < 1 or w < 7:
        return None

    chain_len = random.randint(1, max_chain)
    rows = _sample_spaced_values(chain_len + 1, 0, h - 1, 2)
    if rows is None:
        return None
    rows = list(rows)
    random.shuffle(rows)

    extra_max = min(2, len(PORTAL_COLOR_POOL) - chain_len)
    extra_pairs = random.randint(0, extra_max)
    colors = random.sample(PORTAL_COLOR_POOL, chain_len + extra_pairs)
    used_colors = colors[:chain_len]
    extra_colors = colors[chain_len:]

    entry_cols: list[int] = []
    exit_cols: list[int] = []
    segment_start = 0

    for i in range(chain_len):
        entry_hi = w - 2
        if segment_start + 1 > entry_hi:
            return None
        entry = random.randint(segment_start + 1, entry_hi)
        entry_cols.append(entry)

        exit_hi = w - 4 if i < chain_len - 1 else w - 2
        if exit_hi < 0:
            return None
        exit_col = random.randint(0, exit_hi)
        exit_cols.append(exit_col)
        segment_start = exit_col + 1

    pair_positions = [
        ((rows[i], entry_cols[i]), (rows[i + 1], exit_cols[i]), used_colors[i])
        for i in range(chain_len)
    ]

    inp, out = _render_expected((h, w), rows, entry_cols, exit_cols, pair_positions)
    if extra_colors and not _place_extra_pairs(inp, out, extra_colors, blocked_rows=set(rows)):
        return None
    if not _validate_input_structure(inp):
        return None

    rotations = random.randrange(4)
    flip_lr = random.random() < 0.5
    inp = _transform_grid(inp, rotations, flip_lr)
    out = _transform_grid(out, rotations, flip_lr)
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair = _build_attempt()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid teleport-path input")
