from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]
DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8: tuple[Coord, ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

BG = 0
SEPARATOR_COLOR = 6
BODY_TOP = 2
USABLE_COLORS = (1, 2, 3, 4, 5, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def neighbors8(r: int, c: int) -> list[Coord]:
    return [(r + dr, c + dc) for dr, dc in DIRS8]


def _monochromatic_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or seen[r, c]:
                continue

            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if not in_bounds((h, w), nr, nc) or seen[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _cells(component: dict) -> set[Coord]:
    return set(component["coords"])


def _neighbors_in(cell: Coord, cells: set[Coord]) -> list[Coord]:
    return [nbr for nbr in neighbors4(*cell) if nbr in cells]


def _endpoints(cells: set[Coord]) -> tuple[Coord, Coord]:
    degrees = {cell: len(_neighbors_in(cell, cells)) for cell in cells}
    assert all(degree in (1, 2) for degree in degrees.values()), "path must be a simple orthogonal chain"
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    return endpoints[0], endpoints[1]


def _ordered_path(cells: set[Coord], start: Coord) -> list[Coord]:
    assert start in cells
    path = [start]
    prev: Coord | None = None
    cur = start
    while True:
        nxt = [nbr for nbr in _neighbors_in(cur, cells) if nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path must not branch"
        prev, cur = cur, nxt[0]
        path.append(cur)
    assert len(path) == len(cells), "path traversal must cover the full chain"
    return path


def _extract_palette_and_separator(grid: np.ndarray, bg: int) -> tuple[list[int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h >= 6 and w >= 6

    separator_rows = [
        r for r in range(h) if int(g[r, 0]) != bg and np.all(g[r] == g[r, 0])
    ]
    assert separator_rows, "expected a full-width separator row"
    separator_r = int(separator_rows[0])
    assert separator_r == 1, "separator must sit directly below the palette row"

    palette_end = 0
    while palette_end < w and int(g[0, palette_end]) != bg:
        palette_end += 1
    assert 4 <= palette_end <= 6, "palette row must contain 4-6 left-aligned colors"
    assert np.all(g[0, palette_end:] == bg), "palette row must end after the contiguous prefix"

    palette = [int(v) for v in g[0, :palette_end]]
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert all(color != bg and color != int(g[separator_r, 0]) for color in palette)
    return palette, separator_r


def _body_components(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    _palette, separator_r = _extract_palette_and_separator(g, bg)
    body = g[separator_r + 1 :, :]
    comps = _monochromatic_components(body, bg)
    lifted: list[dict] = []
    for comp in comps:
        coords = tuple((r + separator_r + 1, c) for r, c in comp["coords"])
        r0, c0, r1, c1 = comp["bbox"]
        lifted.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "area": int(comp["area"]),
                "bbox": (r0 + separator_r + 1, c0, r1 + separator_r + 1, c1),
            }
        )
    paths = [comp for comp in lifted if int(comp["area"]) > 1]
    markers = [comp for comp in lifted if int(comp["area"]) == 1]
    return bg, paths, markers


def _visible_endpoint_candidates(
    grid: np.ndarray,
    bg: int,
    marker: Coord,
    endpoint_to_object: dict[Coord, int],
) -> dict[int, tuple[int, Coord, tuple[Coord, ...]]]:
    h, w = map(int, grid.shape)
    mr, mc = marker
    candidates: dict[int, tuple[int, Coord, tuple[Coord, ...]]] = {}

    for dr, dc in DIRS4:
        gap: list[Coord] = []
        r = mr + dr
        c = mc + dc
        while 0 <= r < h and 0 <= c < w:
            if int(grid[r, c]) != bg:
                hit = (r, c)
                if hit in endpoint_to_object:
                    object_index = endpoint_to_object[hit]
                    assert object_index not in candidates, "a marker must not see both endpoints of the same path"
                    candidates[object_index] = (object_index, hit, tuple(gap))
                break
            gap.append((r, c))
            r += dr
            c += dc

    assert candidates, "each marker must see at least one path endpoint"
    return candidates


def _unique_matching(
    candidate_maps: list[dict[int, tuple[int, Coord, tuple[Coord, ...]]]],
    object_count: int,
) -> list[tuple[int, Coord, tuple[Coord, ...]]]:
    solutions: list[tuple[tuple[int, Coord, tuple[Coord, ...]], ...]] = []

    def backtrack(i: int, used: set[int], current: list[tuple[int, Coord, tuple[Coord, ...]]]) -> None:
        if i == len(candidate_maps):
            if len(used) == object_count:
                solutions.append(tuple(current))
            return
        for object_index, candidate in candidate_maps[i].items():
            if object_index in used:
                continue
            used.add(object_index)
            current.append(candidate)
            backtrack(i + 1, used, current)
            current.pop()
            used.remove(object_index)

    backtrack(0, set(), [])
    assert len(solutions) == 1, "marker-to-path assignment must be unique"
    return list(solutions[0])


def _apply_rule(
    grid: np.ndarray,
    palette: list[int],
    paths: list[dict],
    markers: list[dict],
    assignment: list[tuple[int, Coord, tuple[Coord, ...]]],
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    path_cells = [_cells(path) for path in paths]
    object_endpoints = [_endpoints(cells) for cells in path_cells]

    gap_plans: dict[Coord, int] = {}
    ring_plans: dict[Coord, int] = {}
    tail_plans: dict[Coord, int] = {}

    blocked_for_rings: set[Coord] = set()
    for path in paths:
        blocked_for_rings.update(path["coords"])
    for marker in markers:
        blocked_for_rings.update(marker["coords"])

    for path_index, _hit, gap in assignment:
        blocked_for_rings.update(gap)

    for marker_component, (path_index, hit_endpoint, gap) in zip(markers, assignment):
        marker = marker_component["coords"][0]
        mr, mc = marker
        d = len(gap)
        assert 2 <= d <= 4, "marker gap must encode a distance in [2, 4]"
        path_color = int(paths[path_index]["color"])
        outer_color = int(palette[(d - 1) % len(palette)])

        for cell in gap:
            assert int(g[cell]) == bg
            gap_plans[cell] = path_color

        for ring_d in range(1, d + 1):
            ring_color = int(palette[(ring_d - 1) % len(palette)])
            for rr in range(mr - ring_d, mr + ring_d + 1):
                for cc in range(mc - ring_d, mc + ring_d + 1):
                    if not in_bounds(g.shape, rr, cc):
                        continue
                    if max(abs(rr - mr), abs(cc - mc)) != ring_d:
                        continue
                    cell = (rr, cc)
                    if cell in blocked_for_rings or int(g[cell]) != bg:
                        continue
                    assert cell not in ring_plans, "ring layers from different markers must stay disjoint"
                    ring_plans[cell] = ring_color

        ordered_from_visible = _ordered_path(path_cells[path_index], hit_endpoint)
        tail = ordered_from_visible[::-1][:d]
        for cell in tail:
            tail_plans[cell] = outer_color

    for (r, c), color in gap_plans.items():
        out[r, c] = int(color)
    for (r, c), color in ring_plans.items():
        out[r, c] = int(color)
    for (r, c), color in tail_plans.items():
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each marker pixel to a path endpoint by clear sightline, then use that gap length to grow rings and recolor the far tail.

    Input structure
    - The top row is a left-aligned palette of 4-6 distinct non-background colors, and row 1 is a full-width separator.
    - Below the separator are 2-3 monochrome one-pixel-wide orthogonal path objects and the same number of isolated marker pixels.
    - Each marker shares exactly one clear row or column sightline with a path endpoint, with a background gap of length 2-4.

    Rule
    - Uniquely match every marker to a path endpoint by orthogonal sightline through background only.
    - Let D be the number of background cells between the marker and that endpoint; fill those D gap cells with the path color.
    - Around the marker, paint the Chebyshev-distance rings 1..D using palette[0], palette[1], ..., cycling if needed, but only onto background cells.
    - Take the opposite endpoint of the matched path and recolor its first D path pixels with the outer-ring color palette[(D-1) mod L].

    Key insight
    - A single visible distance D simultaneously determines connector length, the number and colors of concentric rings, and how many cells of the far path tail are recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, paths, markers = _body_components(g)
    palette, _separator_r = _extract_palette_and_separator(g, bg)
    assert 2 <= len(paths) <= 3
    assert len(paths) == len(markers)

    path_cells = [_cells(path) for path in paths]
    object_endpoints = [_endpoints(cells) for cells in path_cells]
    endpoint_to_object = {
        endpoint: object_index
        for object_index, endpoints in enumerate(object_endpoints)
        for endpoint in endpoints
    }
    marker_coords = [marker["coords"][0] for marker in markers]
    candidate_maps = [
        _visible_endpoint_candidates(g, bg, marker, endpoint_to_object) for marker in marker_coords
    ]
    assignment = _unique_matching(candidate_maps, len(paths))
    return _apply_rule(g, palette, paths, markers, assignment)


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _square_cells(center: Coord, radius: int, shape: tuple[int, int]) -> set[Coord]:
    cr, cc = center
    h, w = map(int, shape)
    cells = set()
    for r in range(cr - radius, cr + radius + 1):
        for c in range(cc - radius, cc + radius + 1):
            if 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
    return cells


def _allocate_colors(pair_count: int) -> tuple[list[int], list[int], list[int]]:
    max_palette_len = 5 if pair_count == 3 else 6
    palette_len = random.randint(4, max_palette_len)
    palette = random.sample(list(USABLE_COLORS), palette_len)
    remaining = [c for c in USABLE_COLORS if c not in palette]
    path_colors = random.sample(remaining, pair_count)
    marker_pool = [c for c in USABLE_COLORS if c not in path_colors]
    marker_colors = random.sample(marker_pool, pair_count)
    return palette, path_colors, marker_colors


def _sample_distances(pair_count: int) -> list[int]:
    return random.sample([2, 3, 4], pair_count)


def _sample_directions(pair_count: int) -> list[Coord]:
    while True:
        dirs = random.sample(list(DIRS4), pair_count)
        axes = {0 if dr else 1 for dr, _dc in dirs}
        if pair_count == 1 or len(axes) >= 2:
            return dirs


def _buffer_cells(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    h, w = map(int, shape)
    out = set(cells)
    for r, c in list(cells):
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _segment_lengths(total_steps: int, segment_count: int) -> list[int]:
    if segment_count == 1:
        return [total_steps]
    cuts = sorted(random.sample(range(1, total_steps), segment_count - 1))
    points = [0, *cuts, total_steps]
    return [points[i + 1] - points[i] for i in range(segment_count)]


def _path_turn_count(path: list[Coord]) -> int:
    turns = 0
    prev_dir: Coord | None = None
    for a, b in zip(path, path[1:]):
        cur_dir = (b[0] - a[0], b[1] - a[1])
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return turns


def _extend_segment(
    cur_path: list[Coord],
    direction: Coord,
    steps: int,
    marker: Coord,
    d: int,
    shape: tuple[int, int],
    forbidden: set[Coord],
) -> list[Coord] | None:
    path = list(cur_path)
    used = set(cur_path)
    h, w = map(int, shape)
    r, c = path[-1]

    for _ in range(steps):
        r += direction[0]
        c += direction[1]
        cell = (r, c)
        if not (BODY_TOP <= r < h and 0 <= c < w):
            return None
        if cell in forbidden or cell in used:
            return None
        if _chebyshev(cell, marker) <= d:
            return None
        for nbr in neighbors4(r, c):
            if nbr in used and nbr != path[-1]:
                return None
        path.append(cell)
        used.add(cell)
    return path


def _build_path(
    start: Coord,
    marker: Coord,
    d: int,
    total_len: int,
    first_dir: Coord,
    shape: tuple[int, int],
    forbidden: set[Coord],
) -> list[Coord] | None:
    total_steps = total_len - 1
    max_segments = min(4, total_steps)
    segment_count = random.randint(2, max_segments)

    for _ in range(24):
        lengths = _segment_lengths(total_steps, segment_count)

        def dfs(seg_index: int, prev_dir: Coord, path: list[Coord]) -> list[Coord] | None:
            if seg_index == segment_count:
                return path if len(path) == total_len else None

            dirs: list[Coord]
            if seg_index == 0:
                dirs = [first_dir]
            else:
                dirs = [
                    cand
                    for cand in DIRS4
                    if cand != (-prev_dir[0], -prev_dir[1]) and cand != prev_dir
                ]
                random.shuffle(dirs)

            for cand in dirs:
                extended = _extend_segment(path, cand, lengths[seg_index], marker, d, shape, forbidden)
                if extended is None:
                    continue
                result = dfs(seg_index + 1, cand, extended)
                if result is not None:
                    return result
            return None

        candidate = dfs(0, first_dir, [start])
        if candidate is not None and 1 <= _path_turn_count(candidate) <= 3:
            return candidate
    return None


def _place_pair(
    shape: tuple[int, int],
    d: int,
    direction: Coord,
    path_color: int,
    marker_color: int,
    occupied: set[Coord],
    reserved_rings: set[Coord],
    occupied_buffer: set[Coord],
) -> dict | None:
    h, w = map(int, shape)
    toward_marker = (-direction[0], -direction[1])

    for _ in range(120):
        mr = random.randint(BODY_TOP + d, h - 1 - d)
        mc = random.randint(d, w - 1 - d)
        marker = (mr, mc)
        if marker in occupied_buffer or marker in reserved_rings:
            continue

        gap = [(mr + direction[0] * step, mc + direction[1] * step) for step in range(1, d + 1)]
        visible = (mr + direction[0] * (d + 1), mc + direction[1] * (d + 1))
        all_corridor = set(gap) | {visible}
        if not all(in_bounds(shape, r, c) and r >= BODY_TOP for r, c in all_corridor):
            continue
        if any(cell in occupied_buffer or cell in reserved_rings for cell in all_corridor):
            continue

        ring_square = _square_cells(marker, d, shape)
        if ring_square & occupied:
            continue
        if ring_square & reserved_rings:
            continue

        first_dir_options = [
            cand for cand in DIRS4 if cand != toward_marker and cand != (-toward_marker[0], -toward_marker[1])
        ]
        if not first_dir_options:
            continue
        random.shuffle(first_dir_options)

        path_len = random.randint(5, 10)
        local_forbidden = set(occupied_buffer) | set(reserved_rings) | set(gap) | {marker}
        for first_dir in first_dir_options:
            path = _build_path(visible, marker, d, path_len, first_dir, shape, local_forbidden)
            if path is None:
                continue

            path_cells = set(path)
            if any(cell in ring_square for cell in path_cells if cell != visible):
                continue
            if any(_manhattan(marker, cell) <= 1 for cell in path_cells):
                continue
            return {
                "marker": marker,
                "marker_color": int(marker_color),
                "direction": direction,
                "d": int(d),
                "gap": tuple(gap),
                "visible": visible,
                "path": tuple(path),
                "path_color": int(path_color),
                "ring_square": tuple(sorted(ring_square)),
            }
    return None


def _render_input(shape: tuple[int, int], palette: list[int], pairs: list[dict]) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), BG, dtype=int)
    grid[0, : len(palette)] = np.array(palette, dtype=int)
    grid[1, :] = SEPARATOR_COLOR
    for pair in pairs:
        for r, c in pair["path"]:
            grid[r, c] = int(pair["path_color"])
        mr, mc = pair["marker"]
        grid[mr, mc] = int(pair["marker_color"])
    return grid


def _expected_output(shape: tuple[int, int], palette: list[int], pairs: list[dict]) -> np.ndarray:
    inp = _render_input(shape, palette, pairs)
    out = inp.copy()

    ring_used: set[Coord] = set()
    for pair in pairs:
        for r, c in pair["gap"]:
            out[r, c] = int(pair["path_color"])

    blocked = {
        cell
        for pair in pairs
        for cell in (set(pair["path"]) | {pair["marker"]} | set(pair["gap"]))
    }
    for pair in pairs:
        mr, mc = pair["marker"]
        d = int(pair["d"])
        for ring_d in range(1, d + 1):
            color = int(palette[(ring_d - 1) % len(palette)])
            for rr in range(mr - ring_d, mr + ring_d + 1):
                for cc in range(mc - ring_d, mc + ring_d + 1):
                    cell = (rr, cc)
                    if not in_bounds(shape, rr, cc):
                        continue
                    if max(abs(rr - mr), abs(cc - mc)) != ring_d:
                        continue
                    if cell in blocked or cell in ring_used:
                        continue
                    out[rr, cc] = color
                    ring_used.add(cell)

    for pair in pairs:
        d = int(pair["d"])
        outer = int(palette[(d - 1) % len(palette)])
        tail = list(reversed(pair["path"]))[:d]
        for r, c in tail:
            out[r, c] = outer
    return out


def _validate_semantics(grid: np.ndarray, palette: list[int], pairs: list[dict]) -> None:
    bg, paths, markers = _body_components(grid)
    assert bg == BG
    assert len(paths) == len(pairs)
    assert len(markers) == len(pairs)

    d_values = {int(pair["d"]) for pair in pairs}
    assert len(d_values) >= min(2, len(pairs)), "need multiple gap lengths to disallow distance shortcuts"
    axes = {0 if int(pair["direction"][0]) else 1 for pair in pairs}
    assert len(axes) >= min(2, len(pairs)), "need both row and column sightlines across the sample"

    path_cells = [_cells(path) for path in paths]
    object_endpoints = [_endpoints(cells) for cells in path_cells]
    endpoint_to_object = {
        endpoint: object_index
        for object_index, endpoints in enumerate(object_endpoints)
        for endpoint in endpoints
    }
    marker_coords = [marker["coords"][0] for marker in markers]
    candidate_maps = [
        _visible_endpoint_candidates(grid, BG, marker, endpoint_to_object) for marker in marker_coords
    ]
    assignment = _unique_matching(candidate_maps, len(paths))

    path_by_color = {int(path["color"]): path_index for path_index, path in enumerate(paths)}
    pair_by_path = {path_by_color[int(pair["path_color"])]: pair for pair in pairs}
    for path_index, hit_endpoint, gap in assignment:
        pair = pair_by_path[path_index]
        assert hit_endpoint == pair["visible"]
        assert tuple(gap) == tuple(pair["gap"])

    expected = _expected_output(grid.shape, palette, pairs)
    solved = main(grid)
    assert np.array_equal(solved, expected), "main() must match the construction semantics"


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.choice((2, 3))
    palette, path_colors, marker_colors = _allocate_colors(pair_count)
    d_values = _sample_distances(pair_count)
    directions = _sample_directions(pair_count)

    if pair_count == 2:
        h = random.randint(18, 22)
        w = random.randint(18, 22)
    else:
        h = random.randint(20, 24)
        w = random.randint(20, 24)
    shape = (h, w)

    occupied: set[Coord] = set()
    reserved_rings: set[Coord] = set()
    occupied_buffer: set[Coord] = set()
    pairs: list[dict] = []

    order = list(range(pair_count))
    random.shuffle(order)
    for idx in order:
        pair = _place_pair(
            shape=shape,
            d=d_values[idx],
            direction=directions[idx],
            path_color=path_colors[idx],
            marker_color=marker_colors[idx],
            occupied=occupied,
            reserved_rings=reserved_rings,
            occupied_buffer=occupied_buffer,
        )
        if pair is None:
            return None
        pairs.append(pair)

        actual_cells = set(pair["path"]) | {pair["marker"]}
        occupied |= actual_cells
        reserved_rings |= set(pair["ring_square"]) | set(pair["gap"])
        occupied_buffer |= _buffer_cells(actual_cells | set(pair["gap"]), shape)

    if len({int(pair["path_color"]) for pair in pairs}) != pair_count:
        return None
    if len({int(pair["marker_color"]) for pair in pairs}) != pair_count:
        return None

    grid = _render_input(shape, palette, pairs)
    _validate_semantics(grid, palette, pairs)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _generate_attempt()
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
