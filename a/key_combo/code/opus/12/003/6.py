import random
from collections import Counter
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
DIRECTIONS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG = 0
PATH_COLOR = 5


def neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_of_mask(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def ordered_path(cells: set[Coord], start: Coord) -> list[Coord]:
    assert start in cells
    path = [start]
    prev = None
    cur = start
    while True:
        nxt = [nbr for nbr in neighbors4(*cur) if nbr in cells and nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path must be an unbranched chain"
        prev, cur = cur, nxt[0]
        path.append(cur)
    assert len(path) == len(cells), "path walk must cover the entire component"
    return path


def path_endpoints(cells: set[Coord]) -> list[Coord]:
    endpoints = [cell for cell in cells if sum(nbr in cells for nbr in neighbors4(*cell)) == 1]
    degrees = [sum(nbr in cells for nbr in neighbors4(*cell)) for cell in cells]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degrees), "path must be one-pixel wide and unbranched"
    return endpoints


def parse_palette(grid: np.ndarray) -> tuple[list[int], list[Coord]]:
    row = [int(x) for x in grid[0]]
    runs: list[list[int]] = []
    coords_runs: list[list[Coord]] = []
    current: list[int] = []
    current_coords: list[Coord] = []
    for c, value in enumerate(row):
        if value not in (BG, PATH_COLOR):
            current.append(value)
            current_coords.append((0, c))
        elif current:
            runs.append(current)
            coords_runs.append(current_coords)
            current = []
            current_coords = []
    if current:
        runs.append(current)
        coords_runs.append(current_coords)
    assert len(runs) == 1, "expected exactly one palette run in the top row"
    palette = runs[0]
    assert 2 <= len(palette) <= 3, "palette length must be 2 or 3"
    assert len(set(palette)) == len(palette), "palette colors must be unique"
    return palette, coords_runs[0]


def extract_paths(grid: np.ndarray) -> list[dict]:
    comps = components_of_mask(grid == PATH_COLOR)
    paths = []
    for comp in comps:
        cells = set(comp)
        endpoints = path_endpoints(cells)
        paths.append({"cells": cells, "endpoints": endpoints})
    assert 2 <= len(paths) <= 3, "expected 2 or 3 paths"
    return paths


def extract_markers(grid: np.ndarray, palette_coords: set[Coord]) -> list[Coord]:
    h, w = grid.shape
    markers: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in palette_coords:
                continue
            value = int(grid[r, c])
            if value not in (BG, PATH_COLOR):
                markers.append((r, c))
    assert 2 <= len(markers) <= 3, "expected 2 or 3 markers"
    return markers


def visible_candidates(
    grid: np.ndarray,
    marker: Coord,
    endpoint_to_path: dict[Coord, int],
) -> list[tuple[int, Coord, tuple[Coord, ...]]]:
    h, w = grid.shape
    mr, mc = marker
    candidates = []
    seen_paths = set()
    for dr, dc in DIRECTIONS:
        gap: list[Coord] = []
        r, c = mr + dr, mc + dc
        while 0 <= r < h and 0 <= c < w:
            value = int(grid[r, c])
            if value != BG:
                hit = (r, c)
                if value == PATH_COLOR and hit in endpoint_to_path and gap:
                    path_index = endpoint_to_path[hit]
                    assert path_index not in seen_paths, "marker sees the same path from multiple directions"
                    candidates.append((path_index, hit, tuple(gap)))
                    seen_paths.add(path_index)
                break
            gap.append((r, c))
            r += dr
            c += dc
    assert candidates, "each marker must see at least one endpoint"
    return candidates


def unique_assignment(
    candidate_lists: list[list[tuple[int, Coord, tuple[Coord, ...]]]],
    path_count: int,
) -> list[tuple[int, Coord, tuple[Coord, ...]]]:
    order = sorted(range(len(candidate_lists)), key=lambda i: len(candidate_lists[i]))
    chosen = [None] * len(candidate_lists)
    solutions: list[tuple[tuple[int, Coord, tuple[Coord, ...]], ...]] = []

    def backtrack(i: int, used: set[int]) -> None:
        if i == len(order):
            if len(used) == path_count:
                solutions.append(tuple(chosen))  # type: ignore[arg-type]
            return
        marker_index = order[i]
        for candidate in candidate_lists[marker_index]:
            path_index = candidate[0]
            if path_index in used:
                continue
            chosen[marker_index] = candidate
            used.add(path_index)
            backtrack(i + 1, used)
            used.remove(path_index)
            chosen[marker_index] = None
            if len(solutions) > 1:
                return

    backtrack(0, set())
    assert len(solutions) == 1, "marker-path matching must be unique"
    return list(solutions[0])


def solve_structure(grid: np.ndarray) -> tuple[list[int], list[Coord], list[dict], list[tuple[int, Coord, tuple[Coord, ...]]]]:
    palette, palette_coords_list = parse_palette(grid)
    palette_coords = set(palette_coords_list)
    paths = extract_paths(grid)
    markers = extract_markers(grid, palette_coords)

    endpoint_to_path: dict[Coord, int] = {}
    for i, path in enumerate(paths):
        for endpoint in path["endpoints"]:
            assert endpoint not in endpoint_to_path
            endpoint_to_path[endpoint] = i

    candidate_lists = [visible_candidates(grid, marker, endpoint_to_path) for marker in markers]
    assignment = unique_assignment(candidate_lists, len(paths))
    return palette, markers, paths, assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each marker to its uniquely visible path endpoint, then cyclically recolor the whole extended path by walking outward from the marker.

    Input structure:
    - Black (0) is the background.
    - The top row contains one contiguous palette run of 2 or 3 colored pixels; left-to-right order defines the cyclic palette.
    - Elsewhere there are 2 or 3 separate grey (5) one-pixel-wide orthogonal path objects, each an unbranched 4-connected chain with exactly two endpoints, plus the same number of isolated colored marker pixels.

    Rule:
    - Match each marker to the unique path endpoint it can see along a straight horizontal or vertical background gap, with the full marker-to-path assignment required to be one-to-one.
    - Fill the gap between the marker and its matched endpoint with grey, without changing the marker pixel itself.
    - Starting from the first path cell next to the marker and walking through the filled gap and then along the original path to the far endpoint, recolor step-distance d with palette[(d-1) mod L].
    - Keep the marker pixels, the palette row, and all remaining background unchanged.

    Key insight:
    - The marker is both the sightline anchor and the distance origin: the gap length shifts the coloring phase for every later cell on that path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    palette, markers, paths, assignment = solve_structure(g)

    out = g.copy()
    for marker, (path_index, near_endpoint, gap) in zip(markers, assignment):
        cells = paths[path_index]["cells"]
        endpoint_a, endpoint_b = paths[path_index]["endpoints"]
        far_endpoint = endpoint_b if near_endpoint == endpoint_a else endpoint_a
        path_order = ordered_path(cells, near_endpoint)
        assert path_order[-1] == far_endpoint
        extended = list(gap) + path_order
        for i, cell in enumerate(extended):
            out[cell] = palette[i % len(palette)]
        out[marker] = g[marker]
    return out


GRID_H_RANGE = (13, 19)
GRID_W_RANGE = (13, 19)
PATH_COUNT_CHOICES = (2, 3)
PALETTE_LEN_CHOICES = (2, 3)
PATH_LENGTH_RANGE = (5, 12)
TURN_CHOICES = (0, 1, 2, 3)
TURN_WEIGHTS = (1, 3, 3, 2)
GAP_LENGTHS = (1, 2, 3, 4, 5)
MIN_PATH_ROW = 2
EDGE_MARGIN = 1
MIN_OBJECT_DISTANCE = 3
MAX_GENERATION_ATTEMPTS = 500
PAIR_PLACEMENT_ATTEMPTS = 80
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def segment_lengths(total_steps: int, segment_count: int) -> list[int]:
    if segment_count == 1:
        return [total_steps]
    cuts = sorted(random.sample(range(1, total_steps), segment_count - 1))
    points = [0, *cuts, total_steps]
    return [points[i + 1] - points[i] for i in range(segment_count)]


def relative_path(length: int, turns: int) -> list[Coord]:
    pieces = segment_lengths(length - 1, turns + 1)
    start_axis = random.randrange(2)
    axis_signs = [random.choice((-1, 1)), random.choice((-1, 1))]
    reverse_candidates = []
    if turns >= 2 and pieces[1] > 1:
        reverse_candidates.append(2)
    if turns >= 3 and pieces[2] > 1:
        reverse_candidates.append(3)
    reverse_segment = random.choice([None, *reverse_candidates])
    coords = [(0, 0)]
    r = c = 0
    for i, steps in enumerate(pieces):
        axis = (start_axis + i) % 2
        sign = axis_signs[axis]
        if i == reverse_segment:
            sign *= -1
        dr, dc = (sign, 0) if axis == 0 else (0, sign)
        for _ in range(steps):
            r += dr
            c += dc
            coords.append((r, c))
    return coords


def is_chain(coords: list[Coord]) -> bool:
    cells = set(coords)
    if len(cells) != len(coords):
        return False
    degrees = [sum(nbr in cells for nbr in neighbors4(*cell)) for cell in cells]
    return degrees.count(1) == 2 and all(deg in (1, 2) for deg in degrees)


def clear_gap(path_cells: set[Coord], endpoint: Coord, direction: Coord, gap_len: int) -> tuple[list[Coord], Coord] | None:
    dr, dc = direction
    forward = []
    for step in range(1, gap_len + 1):
        cell = (endpoint[0] + dr * step, endpoint[1] + dc * step)
        if cell in path_cells:
            return None
        forward.append(cell)
    marker = (endpoint[0] + dr * (gap_len + 1), endpoint[1] + dc * (gap_len + 1))
    if marker in path_cells:
        return None
    return list(reversed(forward)), marker


def gap_options(path: list[Coord], visible: Coord, inward: Coord, gap_len: int) -> list[tuple[list[Coord], Coord, Coord]]:
    cells = set(path)
    options = []
    for direction in DIRECTIONS:
        if direction == inward:
            continue
        found = clear_gap(cells, visible, direction, gap_len)
        if found is not None:
            gap, marker = found
            options.append((gap, marker, direction))
    return options


def translation_bounds(h: int, w: int, footprint: list[Coord]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in footprint]
    cols = [c for _, c in footprint]
    min_tr = max(-min(rows), MIN_PATH_ROW - min(rows))
    max_tr = min(h - 1 - max(rows), h - 1 - EDGE_MARGIN - max(rows))
    min_tc = max(-min(cols), EDGE_MARGIN - min(cols))
    max_tc = min(w - 1 - max(cols), w - 1 - EDGE_MARGIN - max(cols))
    if min_tr > max_tr or min_tc > max_tc:
        return None
    return min_tr, max_tr, min_tc, max_tc


def min_distance(a: set[Coord], b: set[Coord]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def path_has_turn(path: list[Coord]) -> bool:
    if len(path) < 3:
        return False
    for i in range(1, len(path) - 1):
        a = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        b = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if a != b:
            return True
    return False


def try_place_pair(h: int, w: int, gap_len: int, existing_objects: list[set[Coord]]):
    length = random.randint(*PATH_LENGTH_RANGE)
    turns = random.choices(TURN_CHOICES, weights=TURN_WEIGHTS, k=1)[0]
    if turns > length - 2:
        turns = max(0, length - 2)
    rel_path = relative_path(length, turns)
    if not is_chain(rel_path):
        return None

    visible_at_start = random.choice((True, False))
    visible = rel_path[0] if visible_at_start else rel_path[-1]
    inward_nbr = rel_path[1] if visible_at_start else rel_path[-2]
    inward = (inward_nbr[0] - visible[0], inward_nbr[1] - visible[1])
    options = gap_options(rel_path, visible, inward, gap_len)
    if not options:
        return None
    gap, marker, direction = random.choice(options)

    footprint = [*rel_path, *gap, marker]
    bounds = translation_bounds(h, w, footprint)
    if bounds is None:
        return None
    min_tr, max_tr, min_tc, max_tc = bounds

    for _ in range(20):
        tr = random.randint(min_tr, max_tr)
        tc = random.randint(min_tc, max_tc)
        path = [(r + tr, c + tc) for r, c in rel_path]
        gap_abs = [(r + tr, c + tc) for r, c in gap]
        marker_abs = (marker[0] + tr, marker[1] + tc)
        path_set = set(path)
        marker_set = {marker_abs}
        if any(min_distance(path_set, obj) < MIN_OBJECT_DISTANCE for obj in existing_objects):
            continue
        if any(min_distance(marker_set, obj) < MIN_OBJECT_DISTANCE for obj in existing_objects):
            continue
        if any(cell in obj for cell in gap_abs for obj in existing_objects):
            continue
        return {
            "path": path,
            "gap": gap_abs,
            "marker": marker_abs,
            "direction": direction,
            "visible_at_start": visible_at_start,
            "gap_len": gap_len,
            "has_turn": path_has_turn(path),
        }
    return None


def explicit_output(inp: np.ndarray, records: list[dict], palette: list[int]) -> np.ndarray:
    out = inp.copy()
    for rec in records:
        path = rec["path"] if rec["visible_at_start"] else list(reversed(rec["path"]))
        extended = list(rec["gap"]) + path
        for i, cell in enumerate(extended):
            out[cell] = palette[i % len(palette)]
    return out


def semantic_check(inp: np.ndarray, expected: np.ndarray, intended_gap_lengths: list[int]) -> bool:
    try:
        palette, markers, paths, assignment = solve_structure(inp)
    except Exception:
        return False
    if len(markers) != len(paths):
        return False
    gaps = sorted(len(gap) for _path_index, _endpoint, gap in assignment)
    if gaps != sorted(intended_gap_lengths):
        return False
    axes = {0 if marker[0] != endpoint[0] else 1 for marker, (_pi, endpoint, _gap) in zip(markers, assignment)}
    if len(markers) >= 2 and axes != {0, 1}:
        return False
    if not any(path_has_turn(ordered_path(path["cells"], path_endpoints(path["cells"])[0])) for path in paths):
        return False
    try:
        out = main(inp)
    except Exception:
        return False
    return np.array_equal(out, expected)


def gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    pair_count = random.choice(PATH_COUNT_CHOICES)
    palette_len = random.choice(PALETTE_LEN_CHOICES)
    palette = random.sample(PALETTE_COLORS, palette_len)
    marker_colors = random.sample([c for c in PALETTE_COLORS if c not in palette], pair_count)
    gap_lengths = random.sample(GAP_LENGTHS, pair_count)

    if w < palette_len + 4:
        return None
    palette_start = random.randint(1, w - palette_len - 1)
    grid = np.zeros((h, w), dtype=int)
    grid[0, palette_start : palette_start + palette_len] = palette

    records: list[dict] = []
    existing_objects: list[set[Coord]] = []
    for gap_len, marker_color in zip(gap_lengths, marker_colors):
        placed = None
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            placed = try_place_pair(h, w, gap_len, existing_objects)
            if placed is not None:
                break
        if placed is None:
            return None
        records.append(placed)
        existing_objects.append(set(placed["path"]))
        existing_objects.append({placed["marker"]})
        for r, c in placed["path"]:
            grid[r, c] = PATH_COLOR
        mr, mc = placed["marker"]
        grid[mr, mc] = marker_color

    axes = {0 if rec["direction"][0] != 0 else 1 for rec in records}
    if len(records) >= 2 and axes != {0, 1}:
        return None
    if not any(rec["has_turn"] for rec in records):
        return None

    expected = explicit_output(grid, records, palette)
    if not semantic_check(grid, expected, gap_lengths):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
