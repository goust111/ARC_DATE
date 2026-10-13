from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FG_COLORS = tuple(range(1, 10))

GRID_MIN = 14
GRID_MAX = 24
LINE_COUNT_CHOICES = (2, 3)

SEGMENT_COUNT_CHOICES = (1, 2, 3)
SEGMENT_COUNT_WEIGHTS = (2, 3, 2)
SEGMENT_LEN_MIN = 2
SEGMENT_LEN_MAX = 6

LOCAL_PATH_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 120
SAMPLE_ATTEMPTS = 400

Coord = tuple[int, int]


@dataclass(frozen=True)
class PlacedLine:
    path: tuple[Coord, ...]
    markers: tuple[Coord, Coord]
    body_color: int
    marker_colors: tuple[int, int]


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    out: list[Coord] = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _neighbors4_unbounded(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _components(grid: np.ndarray) -> list[tuple[int, list[Coord]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[Coord]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, grid.shape):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            comps.append((color, coords))

    return comps


def _line_endpoints(
    grid: np.ndarray,
    coords: list[Coord],
    markers: dict[Coord, int],
) -> list[tuple[Coord, Coord, Coord]]:
    color = int(grid[coords[0]])
    body = set(coords)
    endpoints: list[tuple[Coord, Coord, Coord]] = []

    for cell in coords:
        r, c = cell
        same = [nbr for nbr in _neighbors4(r, c, grid.shape) if nbr in body]
        foreign = [
            nbr
            for nbr in _neighbors4(r, c, grid.shape)
            if int(grid[nbr]) != BG and nbr not in body
        ]

        assert len(same) in (1, 2), "polyline body must be an open non-branching path"
        if len(same) == 1:
            assert len(foreign) == 1, "each endpoint must have exactly one marker"
            marker = foreign[0]
            assert marker in markers, "endpoint marker must be a singleton component"
            assert markers[marker] != color, "marker color must differ from the body color"
            endpoints.append((cell, same[0], marker))
        else:
            assert not foreign, "only endpoints may touch non-body foreground pixels"

    assert len(endpoints) == 2, "each polyline must have exactly two endpoints"
    return endpoints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each open rectilinear polyline beyond both endpoints using the colors of its endpoint markers.

    Input structure:
    - The background is black.
    - Each polyline body is a one-cell-thick monochrome 4-connected open path made of horizontal and vertical segments.
    - Every body has exactly two endpoints, and each endpoint touches one adjacent single-pixel marker of a different color.
    - Marker pixels are not part of the polyline bodies and the two markers of one body have different colors.

    Rule:
    - Split foreground objects into multi-pixel polyline bodies and singleton endpoint markers.
    - For each endpoint, use the final body step direction from the penultimate body pixel into the endpoint as the outward extension direction.
    - Starting just beyond the endpoint, draw marker-colored pixels along that direction until the next step would leave the grid or hit any non-black input pixel.
    - After all extensions are drawn, erase every endpoint marker back to black and keep the original bodies unchanged.

    Key insight:
    - The marker decides only the extension color, while the polyline's own last segment decides only the extension direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g)
    bodies: list[list[Coord]] = []
    markers: dict[Coord, int] = {}
    for color, coords in comps:
        if len(coords) == 1:
            markers[coords[0]] = color
        else:
            bodies.append(coords)

    assert bodies, "expected at least one polyline body"
    assert markers, "expected endpoint markers"

    out = g.copy()
    used_markers: set[Coord] = set()

    for coords in bodies:
        for endpoint, prev, marker in _line_endpoints(g, coords, markers):
            assert marker not in used_markers, "markers may belong to only one endpoint"
            used_markers.add(marker)

            er, ec = endpoint
            pr, pc = prev
            dr, dc = er - pr, ec - pc
            paint = int(markers[marker])

            r, c = er + dr, ec + dc
            while 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and int(g[r, c]) == BG:
                assert int(out[r, c]) == BG, "extension paths must not overlap"
                out[r, c] = paint
                r += dr
                c += dc

    assert used_markers == set(markers), "every marker must be attached to an endpoint"
    for marker in markers:
        out[marker] = BG
    return out


def _perpendicular_dirs(direction: Coord) -> list[Coord]:
    dr, dc = direction
    if dr != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _generate_local_path() -> list[Coord] | None:
    for _ in range(LOCAL_PATH_ATTEMPTS):
        segment_count = random.choices(SEGMENT_COUNT_CHOICES, weights=SEGMENT_COUNT_WEIGHTS, k=1)[0]
        path = [(0, 0)]
        used = {(0, 0)}
        prev_dir: Coord | None = None
        ok = True

        for _segment in range(segment_count):
            directions = list(DIR4 if prev_dir is None else _perpendicular_dirs(prev_dir))
            random.shuffle(directions)
            candidates: list[tuple[Coord, list[Coord]]] = []

            for direction in directions:
                lengths = list(range(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX + 1))
                random.shuffle(lengths)
                for length in lengths:
                    trial = list(path)
                    trial_used = set(used)
                    prev = trial[-1]
                    new_cells: list[Coord] = []
                    valid = True

                    for step in range(1, length + 1):
                        cell = (path[-1][0] + direction[0] * step, path[-1][1] + direction[1] * step)
                        if cell in trial_used:
                            valid = False
                            break
                        touched = {nbr for nbr in _neighbors4_unbounded(cell) if nbr in trial_used}
                        if touched != {prev}:
                            valid = False
                            break
                        new_cells.append(cell)
                        trial_used.add(cell)
                        prev = cell

                    if valid:
                        candidates.append((direction, new_cells))

            if not candidates:
                ok = False
                break

            prev_dir, new_cells = random.choice(candidates)
            path.extend(new_cells)
            used.update(new_cells)

        if not ok:
            continue
        if random.random() < 0.5:
            path.reverse()
        if _marker_pairs_local(path):
            return path
    return None


def _marker_options_local(path: list[Coord], endpoint_index: int) -> list[Coord]:
    body = set(path)
    endpoint = path[0] if endpoint_index == 0 else path[-1]
    prev = path[1] if endpoint_index == 0 else path[-2]
    direction = (endpoint[0] - prev[0], endpoint[1] - prev[1])

    options: list[Coord] = []
    for offset in _perpendicular_dirs(direction):
        marker = (endpoint[0] + offset[0], endpoint[1] + offset[1])
        if marker in body:
            continue
        touched = {nbr for nbr in _neighbors4_unbounded(marker) if nbr in body}
        if touched == {endpoint}:
            options.append(marker)
    return options


def _marker_pairs_local(path: list[Coord]) -> list[tuple[Coord, Coord]]:
    body = set(path)
    start = path[0]
    end = path[-1]
    pairs: list[tuple[Coord, Coord]] = []

    for m0 in _marker_options_local(path, 0):
        for m1 in _marker_options_local(path, 1):
            if m0 == m1:
                continue
            occupied = body | {m0, m1}
            if {nbr for nbr in _neighbors4_unbounded(m0) if nbr in occupied} != {start}:
                continue
            if {nbr for nbr in _neighbors4_unbounded(m1) if nbr in occupied} != {end}:
                continue
            pairs.append((m0, m1))

    return pairs


def _bbox(coords: list[Coord] | tuple[Coord, ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _translate(coords: list[Coord] | tuple[Coord, ...], dr: int, dc: int) -> tuple[Coord, ...]:
    return tuple((r + dr, c + dc) for r, c in coords)


def _expanded_cells(cells: tuple[Coord, ...], shape: tuple[int, int]) -> set[Coord]:
    h, w = shape
    out: set[Coord] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _candidate_clear(
    body: tuple[Coord, ...],
    markers: tuple[Coord, Coord],
    shape: tuple[int, int],
    occupied: set[Coord],
    forbidden: set[Coord],
) -> bool:
    h, w = shape
    cells = body + markers

    if len(set(cells)) != len(cells):
        return False
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    if any(cell in forbidden or cell in occupied for cell in cells):
        return False

    body_set = set(body)
    occupied_set = body_set | set(markers)
    if {nbr for nbr in _neighbors4_unbounded(markers[0]) if nbr in occupied_set} != {body[0]}:
        return False
    if {nbr for nbr in _neighbors4_unbounded(markers[1]) if nbr in occupied_set} != {body[-1]}:
        return False
    return True


def _translation_choices(
    local_body: list[Coord],
    local_markers: tuple[Coord, Coord],
    shape: tuple[int, int],
) -> list[Coord]:
    all_cells = list(local_body) + [local_markers[0], local_markers[1]]
    min_r, min_c, max_r, max_c = _bbox(all_cells)
    max_dr = shape[0] - 1 - max_r
    max_dc = shape[1] - 1 - max_c
    min_dr = -min_r
    min_dc = -min_c
    if min_dr > max_dr or min_dc > max_dc:
        return []

    shifts = [(dr, dc) for dr in range(min_dr, max_dr + 1) for dc in range(min_dc, max_dc + 1)]
    random.shuffle(shifts)
    return shifts


def _make_line(
    path: tuple[Coord, ...],
    markers: tuple[Coord, Coord],
    body_color: int,
    marker_colors: tuple[int, int],
) -> PlacedLine:
    return PlacedLine(path=path, markers=markers, body_color=body_color, marker_colors=marker_colors)


def _place_random_line(
    shape: tuple[int, int],
    occupied: set[Coord],
    forbidden: set[Coord],
    body_color: int,
    marker_colors: tuple[int, int],
) -> PlacedLine | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        local_path = _generate_local_path()
        if local_path is None:
            continue

        marker_pairs = _marker_pairs_local(local_path)
        random.shuffle(marker_pairs)
        for local_markers in marker_pairs:
            for shift_r, shift_c in _translation_choices(local_path, local_markers, shape):
                body = _translate(local_path, shift_r, shift_c)
                markers = _translate(local_markers, shift_r, shift_c)
                if not _candidate_clear(body, markers, shape, occupied, forbidden):
                    continue
                return _make_line(body, markers, body_color, marker_colors)
    return None


def _place_blocked_line(
    shape: tuple[int, int],
    occupied: set[Coord],
    forbidden: set[Coord],
    existing_lines: list[PlacedLine],
    body_color: int,
    marker_colors: tuple[int, int],
) -> PlacedLine | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS * 2):
        blocker_line = random.choice(existing_lines)
        blocker = random.choice(list(blocker_line.path))
        local_path = _generate_local_path()
        if local_path is None:
            continue

        marker_pairs = _marker_pairs_local(local_path)
        if not marker_pairs:
            continue

        endpoint_index = random.choice((0, 1))
        endpoint_local = local_path[0] if endpoint_index == 0 else local_path[-1]
        prev_local = local_path[1] if endpoint_index == 0 else local_path[-2]
        direction = (endpoint_local[0] - prev_local[0], endpoint_local[1] - prev_local[1])
        distance = random.randint(2, 6)
        target_endpoint = (blocker[0] - direction[0] * distance, blocker[1] - direction[1] * distance)
        if not (0 <= target_endpoint[0] < h and 0 <= target_endpoint[1] < w):
            continue

        shift_r = target_endpoint[0] - endpoint_local[0]
        shift_c = target_endpoint[1] - endpoint_local[1]
        body = _translate(local_path, shift_r, shift_c)

        for local_markers in random.sample(marker_pairs, k=len(marker_pairs)):
            markers = _translate(local_markers, shift_r, shift_c)
            if not _candidate_clear(body, markers, shape, occupied, forbidden):
                continue

            clear = True
            for step in range(1, distance):
                cell = (target_endpoint[0] + direction[0] * step, target_endpoint[1] + direction[1] * step)
                if not (0 <= cell[0] < h and 0 <= cell[1] < w) or cell in occupied:
                    clear = False
                    break
            if not clear:
                continue
            if blocker not in occupied:
                continue
            return _make_line(body, markers, body_color, marker_colors)
    return None


def _commit_line(line: PlacedLine, shape: tuple[int, int], occupied: set[Coord], forbidden: set[Coord]) -> None:
    cells = line.path + line.markers
    occupied.update(cells)
    forbidden.update(_expanded_cells(cells, shape))


def _render_input(shape: tuple[int, int], lines: list[PlacedLine]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for line in lines:
        for cell in line.path:
            grid[cell] = line.body_color
        grid[line.markers[0]] = line.marker_colors[0]
        grid[line.markers[1]] = line.marker_colors[1]
    return grid


def _trace_endpoint(grid: np.ndarray, endpoint: Coord, prev: Coord) -> tuple[str, int]:
    dr = endpoint[0] - prev[0]
    dc = endpoint[1] - prev[1]
    r = endpoint[0] + dr
    c = endpoint[1] + dc
    length = 0

    while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and int(grid[r, c]) == BG:
        length += 1
        r += dr
        c += dc

    branch = "boundary" if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) else "blocked"
    return branch, length


def _branch_counts(grid: np.ndarray, lines: list[PlacedLine]) -> tuple[int, int] | None:
    boundary = 0
    blocked = 0
    for line in lines:
        for endpoint, prev in ((line.path[0], line.path[1]), (line.path[-1], line.path[-2])):
            branch, length = _trace_endpoint(grid, endpoint, prev)
            if length < 1:
                return None
            if branch == "boundary":
                boundary += 1
            else:
                blocked += 1
    return boundary, blocked


def _render_expected(grid: np.ndarray, lines: list[PlacedLine]) -> np.ndarray:
    out = grid.copy()
    for line in lines:
        endpoint_data = (
            (line.path[0], line.path[1], line.marker_colors[0]),
            (line.path[-1], line.path[-2], line.marker_colors[1]),
        )
        for endpoint, prev, color in endpoint_data:
            dr = endpoint[0] - prev[0]
            dc = endpoint[1] - prev[1]
            r = endpoint[0] + dr
            c = endpoint[1] + dc
            while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and int(grid[r, c]) == BG:
                out[r, c] = color
                r += dr
                c += dc

    for line in lines:
        out[line.markers[0]] = BG
        out[line.markers[1]] = BG
    return out


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    line_count = random.choice(LINE_COUNT_CHOICES)
    force_blocked = random.random() < 0.5

    body_colors = random.sample(FG_COLORS, line_count)
    marker_palette = [color for color in FG_COLORS if color not in body_colors]
    marker_color_pairs = [tuple(random.sample(marker_palette, 2)) for _ in range(line_count)]

    occupied: set[Coord] = set()
    forbidden: set[Coord] = set()
    lines: list[PlacedLine] = []

    first = _place_random_line(shape, occupied, forbidden, body_colors[0], marker_color_pairs[0])
    if first is None:
        return None
    lines.append(first)
    _commit_line(first, shape, occupied, forbidden)

    next_index = 1
    if force_blocked:
        blocked_line = _place_blocked_line(
            shape,
            occupied,
            forbidden,
            lines,
            body_colors[next_index],
            marker_color_pairs[next_index],
        )
        if blocked_line is None:
            return None
        lines.append(blocked_line)
        _commit_line(blocked_line, shape, occupied, forbidden)
        next_index += 1

    while next_index < line_count:
        line = _place_random_line(shape, occupied, forbidden, body_colors[next_index], marker_color_pairs[next_index])
        if line is None:
            return None
        lines.append(line)
        _commit_line(line, shape, occupied, forbidden)
        next_index += 1

    grid = _render_input(shape, lines)
    counts = _branch_counts(grid, lines)
    if counts is None:
        return None
    boundary, blocked = counts
    if boundary == 0:
        return None
    if force_blocked and blocked == 0:
        return None

    expected = _render_expected(grid, lines)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(SAMPLE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
