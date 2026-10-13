import random
from collections import Counter

import numpy as np


BG_COLOR = 0
MARKER_COLOR = 9
PATH_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_step(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_simple_path(order: list[tuple[int, int]]) -> bool:
    if len(order) < 2 or len(set(order)) != len(order):
        return False
    if any(not _is_step(a, b) for a, b in zip(order, order[1:])):
        return False

    coord_set = set(order)
    degree_counts = Counter(
        sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c))
        for r, c in coord_set
    )
    return degree_counts.get(1, 0) == 2 and degree_counts.get(0, 0) == 0 and all(
        degree <= 2 for degree in degree_counts
    )


def _ordered_visible_path(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    coords = [tuple(map(int, pos)) for pos in np.argwhere(grid == color)]
    assert coords, "each path color must appear at least once"
    coord_set = set(coords)

    degrees = {
        coord: sum((nr, nc) in coord_set for nr, nc in _neighbors4(*coord))
        for coord in coord_set
    }
    assert all(degree <= 2 for degree in degrees.values()), "path must be one-cell-thick and non-branching"
    endpoints = [coord for coord, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "each colored path must have exactly two endpoints"

    marked_endpoints = [
        coord
        for coord in endpoints
        if sum(
            0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == MARKER_COLOR
            for nr, nc in _neighbors4(*coord)
        )
        == 1
    ]
    assert len(marked_endpoints) == 1, "exactly one endpoint must touch the maroon marker"
    start = marked_endpoints[0]
    finish = endpoints[0] if endpoints[1] == start else endpoints[1]

    order = [start]
    prev = None
    cur = start
    while cur != finish:
        next_cells = [nbr for nbr in _neighbors4(*cur) if nbr in coord_set and nbr != prev]
        assert len(next_cells) == 1, "path traversal must be unambiguous"
        prev, cur = cur, next_cells[0]
        order.append(cur)

    assert len(order) == len(coords), "each colored path must be a single connected component"
    return order


def _odd_full_from_prefix(prefix: list[tuple[int, int]]) -> list[tuple[int, int]]:
    center_r, center_c = prefix[-1]
    mirrored = [(2 * center_r - r, 2 * center_c - c) for r, c in prefix[-2::-1]]
    return list(prefix) + mirrored


def _even_full_from_prefix(prefix: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if len(prefix) < 2:
        return None
    prev_r, prev_c = prefix[-2]
    left_r, left_c = prefix[-1]
    sum_r = 3 * left_r - prev_r
    sum_c = 3 * left_c - prev_c
    mirrored = [(sum_r - r, sum_c - c) for r, c in prefix[::-1]]
    return list(prefix) + mirrored


def _candidate_full_orders(order: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    n = len(order)
    found: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for prefix_len in range(2, n):
        prefix = order[:prefix_len]
        odd_full = _odd_full_from_prefix(prefix)
        if len(odd_full) > n and odd_full[:n] == order and _is_simple_path(odd_full):
            found.setdefault(tuple(odd_full), odd_full)

        even_full = _even_full_from_prefix(prefix)
        if even_full is not None and len(even_full) > n and even_full[:n] == order and _is_simple_path(even_full):
            found.setdefault(tuple(even_full), even_full)
    return list(found.values())


def _infer_completed_order(order: list[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates = _candidate_full_orders(order)
    assert len(candidates) == 1, "path center must be uniquely inferable from the visible turn pattern"
    return candidates[0]


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    assert bg == BG_COLOR
    assert MARKER_COLOR in set(map(int, np.unique(g)))

    out = g.copy()
    path_colors = [int(color) for color in np.unique(g) if int(color) not in {bg, MARKER_COLOR}]
    assert path_colors, "grid must contain at least one colored path"

    for color in path_colors:
        visible_order = _ordered_visible_path(g, color)
        full_order = _infer_completed_order(visible_order)
        for r, c in full_order:
            if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
                continue
            assert out[r, c] in {bg, color}, "completed path must not overwrite other paths or markers"
            out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marked colored path by drawing its missing 180-degree-symmetric tail.

    Input structure:
    - The background is black.
    - Several colors each form one one-pixel-thick 4-connected path made of horizontal and vertical segments.
    - Exactly one endpoint of each path touches a maroon marker pixel; that marked side is the complete reference side.
    - The opposite endpoint is truncated, so only an initial prefix of the symmetric far half is still visible.

    Rule:
    - Keep every existing path pixel and every maroon marker.
    - For each path, order its pixels from the maroon-adjacent endpoint to the unmarked endpoint.
    - Infer the unique midpoint geometry for which the visible path is a prefix of a 180-degree-rotationally symmetric full path; when the true midpoint lies between two cells, use the truncated-side middle cell as the representative center while continuing the same straight run through that gap.
    - Paint the missing suffix of that inferred full path in the same color, clipping any cells that would fall outside the grid.

    Key insight:
    - The maroon marker identifies which end is the untouched reference end, while the sequence of turns reveals the hidden midpoint and therefore the correct 180-degree continuation.
    """
    return _solve_grid(grid)


# --------------- generator (o5x) ---------------

GRID_HEIGHT_RANGE = (16, 30)
GRID_WIDTH_RANGE = (16, 30)
PATH_COUNT_RANGE = (1, 4)
SEGMENT_COUNT_RANGE = (2, 5)
SEGMENT_LENGTH_RANGE = (1, 4)
LOCAL_COORD_LIMIT = 10
LOCAL_SHAPE_ATTEMPTS = 250
PLACEMENT_ATTEMPTS = 250
GEN_ATTEMPTS = 400


def _random_half_path() -> list[tuple[int, int]] | None:
    target_segments = random.randint(*SEGMENT_COUNT_RANGE)
    current = (0, 0)
    order = [current]
    used = {current}

    axis = random.choice((0, 1))
    direction = random.choice(((0, 1), (0, -1))) if axis == 0 else random.choice(((1, 0), (-1, 0)))

    for seg_idx in range(target_segments):
        seg_len = random.randint(*SEGMENT_LENGTH_RANGE)
        for _ in range(seg_len):
            nxt = (current[0] + direction[0], current[1] + direction[1])
            if nxt in used:
                return None
            if max(abs(nxt[0]), abs(nxt[1])) > LOCAL_COORD_LIMIT:
                return None
            used.add(nxt)
            order.append(nxt)
            current = nxt

        if seg_idx == target_segments - 1:
            break

        axis = 1 - axis
        choices = [(0, 1), (0, -1)] if axis == 0 else [(1, 0), (-1, 0)]
        random.shuffle(choices)
        for choice in choices:
            nxt = (current[0] + choice[0], current[1] + choice[1])
            if nxt not in used and max(abs(nxt[0]), abs(nxt[1])) <= LOCAL_COORD_LIMIT:
                direction = choice
                break
        else:
            return None

    return order


def _turn_count(order: list[tuple[int, int]]) -> int:
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(order, order[1:])]
    return sum(d1 != d2 for d1, d2 in zip(dirs, dirs[1:]))


def _choose_marker_cell(full_order: list[tuple[int, int]]) -> tuple[int, int] | None:
    complete_end = full_order[0]
    full_set = set(full_order)
    candidates = []
    for marker in _neighbors4(*complete_end):
        if marker in full_set:
            continue
        if sum(nbr in full_set for nbr in _neighbors4(*marker)) != 1:
            continue
        candidates.append(marker)
    if not candidates:
        return None
    return random.choice(candidates)


def _build_local_spec() -> dict | None:
    for _ in range(LOCAL_SHAPE_ATTEMPTS):
        parity = random.choice(("odd", "even"))
        half = _random_half_path()
        if half is None:
            continue

        if parity == "odd":
            if len(half) < 4:
                continue
            prefix = half[::-1]
            full_order = _odd_full_from_prefix(prefix)
        else:
            if len(half) < 3:
                continue
            prefix = half[::-1]
            full_order = _even_full_from_prefix(prefix)
            assert full_order is not None

        if not _is_simple_path(full_order):
            continue
        if _turn_count(full_order) < 2:
            continue

        reserved_probe = set(full_order)
        r0, c0, r1, c1 = _bbox(reserved_probe)
        if max(r1 - r0 + 1, c1 - c0 + 1) > 13:
            continue

        valid_inputs = []
        min_visible = len(prefix) + 1
        for cut_len in range(min_visible, len(full_order)):
            visible_order = full_order[:cut_len]
            if _turn_count(visible_order) < 1:
                continue
            candidates = _candidate_full_orders(visible_order)
            if len(candidates) == 1 and candidates[0] == full_order:
                valid_inputs.append(visible_order)
        if not valid_inputs:
            continue

        input_order = random.choice(valid_inputs)
        marker = _choose_marker_cell(full_order)
        if marker is None:
            continue

        reserved = set(full_order)
        reserved.add(marker)
        return {
            "input_order": input_order,
            "full_order": full_order,
            "marker": marker,
            "reserved": reserved,
            "parity": parity,
        }
    return None


def _translate_coords(coords: list[tuple[int, int]] | set[tuple[int, int]], dr: int, dc: int):
    if isinstance(coords, set):
        return {(r + dr, c + dc) for r, c in coords}
    return [(r + dr, c + dc) for r, c in coords]


def _buffer_cells(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _place_specs(specs: list[dict], height: int, width: int) -> list[dict] | None:
    blocked: set[tuple[int, int]] = set()
    placed = []
    indexed_specs = list(enumerate(specs))
    indexed_specs.sort(key=lambda item: len(item[1]["reserved"]), reverse=True)

    for original_idx, spec in indexed_specs:
        r0, c0, r1, c1 = _bbox(spec["reserved"])
        min_dr = -r0
        max_dr = height - 1 - r1
        min_dc = -c0
        max_dc = width - 1 - c1
        if min_dr > max_dr or min_dc > max_dc:
            return None

        success = None
        for _ in range(PLACEMENT_ATTEMPTS):
            dr = random.randint(min_dr, max_dr)
            dc = random.randint(min_dc, max_dc)
            translated_reserved = _translate_coords(spec["reserved"], dr, dc)
            if translated_reserved & blocked:
                continue
            success = {
                "idx": original_idx,
                "input_order": _translate_coords(spec["input_order"], dr, dc),
                "full_order": _translate_coords(spec["full_order"], dr, dc),
                "marker": (spec["marker"][0] + dr, spec["marker"][1] + dc),
                "reserved": translated_reserved,
            }
            blocked |= _buffer_cells(translated_reserved, height, width)
            break
        if success is None:
            return None
        placed.append(success)

    placed.sort(key=lambda item: item["idx"])
    return placed


def _render_grid(height: int, width: int, placed_specs: list[dict], use_full: bool) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for spec in placed_specs:
        color = spec["color"]
        order = spec["full_order"] if use_full else spec["input_order"]
        for r, c in order:
            assert 0 <= r < height and 0 <= c < width
            assert grid[r, c] == BG_COLOR
            grid[r, c] = color
        mr, mc = spec["marker"]
        assert grid[mr, mc] == BG_COLOR
        grid[mr, mc] = MARKER_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    path_count = random.randint(*PATH_COUNT_RANGE)

    specs = []
    for _ in range(path_count):
        spec = _build_local_spec()
        if spec is None:
            return None
        specs.append(spec)

    placed_specs = _place_specs(specs, height, width)
    if placed_specs is None:
        return None
    colors = random.sample(PATH_COLORS, len(placed_specs))
    for color, spec in zip(colors, placed_specs):
        spec["color"] = color

    inp = _render_grid(height, width, placed_specs, use_full=False)
    expected = _render_grid(height, width, placed_specs, use_full=True)
    got = _solve_grid(inp)
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the rotational path-completion puzzle")
