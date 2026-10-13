import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

HEIGHT_MIN = 10
HEIGHT_MAX = 24
WIDTH_MIN = 10
WIDTH_MAX = 24
PAIR_COUNT_CHOICES = (2, 3)
SEED_COUNT_CHOICES = (1, 2)
PATH_MARGIN = 1
MAX_DISTANCE_EXTRA = 8
PLACEMENT_TRIES = 250
GEN_RETRIES = 500


def _foreground_order_and_counts(grid: np.ndarray) -> tuple[list[int], dict[int, int]]:
    order: list[int] = []
    counts: dict[int, int] = {}
    for raw in np.asarray(grid, dtype=int).ravel():
        color = int(raw)
        if color == BG:
            continue
        if color not in counts:
            counts[color] = 0
            order.append(color)
        counts[color] += 1
    return order, counts


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        row = int(a[0])
        c0, c1 = sorted((int(a[1]), int(b[1])))
        return [(row, c) for c in range(c0, c1 + 1)]
    if a[1] == b[1]:
        col = int(a[1])
        r0, r1 = sorted((int(a[0]), int(b[0])))
        return [(r, col) for r in range(r0, r1 + 1)]
    raise AssertionError("endpoint pair must be axis-aligned")


def _extract_paths(grid: np.ndarray, strict: bool = True) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    order, counts = _foreground_order_and_counts(g)
    if strict:
        assert all(count in (1, 2) for count in counts.values()), "foreground colors must appear once or twice"

    endpoint_colors = [color for color in order if counts[color] == 2]
    assert endpoint_colors, "expected at least one endpoint pair"

    paths: list[dict[str, object]] = []
    covered_cells: set[tuple[int, int]] = set()
    used_seed_coords: set[tuple[int, int]] = set()

    for color in endpoint_colors:
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        assert len(coords) == 2, "endpoint color must appear exactly twice"
        cells = _segment_cells(coords[0], coords[1])
        markers: list[tuple[int, int]] = [(0, int(color))]

        for idx in range(1, len(cells) - 1):
            r, c = cells[idx]
            value = int(g[r, c])
            if value == BG:
                continue
            if strict:
                assert counts.get(value, 0) == 1, "path markers between endpoints must be singleton seeds"
            markers.append((idx, value))
            used_seed_coords.add((r, c))

        markers.append((len(cells) - 1, int(color)))

        if strict:
            overlap = covered_cells & set(cells)
            assert not overlap, "endpoint paths must not overlap"
            covered_cells.update(cells)

        paths.append(
            {
                "endpoint_color": int(color),
                "cells": cells,
                "markers": markers,
            }
        )

    if strict:
        singleton_coords = {
            tuple(map(int, coord))
            for color in order
            if counts[color] == 1
            for coord in np.argwhere(g == color)
        }
        assert singleton_coords == used_seed_coords, "singleton colors must lie on endpoint paths"

    return paths


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for path in _extract_paths(g, strict=True):
        cells = path["cells"]
        markers = path["markers"]
        assert isinstance(cells, list)
        assert isinstance(markers, list)

        for (start_idx, fill_color), (stop_idx, _next_color) in zip(markers, markers[1:]):
            for idx in range(int(start_idx) + 1, int(stop_idx)):
                r, c = cells[idx]
                assert int(g[r, c]) == BG, "only black cells may be filled between markers"
                out[r, c] = int(fill_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each straight same-color endpoint pair, using singleton seed pixels on the path as ordered color handoff markers.

    Input structure:
    - The background is black.
    - Several foreground colors appear exactly twice as singleton endpoints.
    - Each endpoint pair lies on one row or one column, and some of those straight corridors also contain singleton seed pixels of other colors.

    Rule:
    - Keep every endpoint pixel and every seed pixel unchanged.
    - For each same-color endpoint pair, inspect the straight segment between the earlier endpoint and the later one along that row or column.
    - If no seed lies on the segment, fill all black cells between the endpoints with the endpoint color.
    - If one or more seeds lie on the segment, sort them by position; fill each black gap after a marker with that marker's color, starting with the endpoint color before the first seed.

    Key insight:
    - The off-color singleton pixels on a connector are not noise or extra endpoints; they are divider markers that transfer the paint color from one segment to the next.
    """
    return _apply_rule(np.asarray(grid, dtype=int))


def _choose_seed_positions(distance: int, seed_count: int) -> list[int]:
    if seed_count == 0:
        return []

    black_total = distance - 1 - seed_count
    assert black_total >= seed_count + 1
    groups = [1] * (seed_count + 1)
    extra = black_total - (seed_count + 1)
    for _ in range(extra):
        groups[random.randrange(seed_count + 1)] += 1

    positions: list[int] = []
    pos = groups[0] + 1
    positions.append(pos)
    for idx in range(1, seed_count):
        pos += groups[idx] + 1
        positions.append(pos)
    return positions


def _reserve_cells(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _min_distance(seed_count: int) -> int:
    if seed_count == 0:
        return 2
    return 2 * seed_count + 2


def _build_specs() -> list[dict[str, object]] | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    seeded_flags = [0, 1]
    if pair_count == 3:
        seeded_flags.append(random.randint(0, 1))
    random.shuffle(seeded_flags)

    seed_counts = [
        random.choice(SEED_COUNT_CHOICES) if has_seed else 0
        for has_seed in seeded_flags
    ]
    total_colors = pair_count + sum(seed_counts)
    if total_colors > len(PALETTE):
        return None

    chosen = random.sample(PALETTE, total_colors)
    endpoint_colors = chosen[:pair_count]
    seed_palette = chosen[pair_count:]

    if pair_count > 1 and random.random() < 0.5:
        orientations = ["h", "v"] + [random.choice(("h", "v")) for _ in range(pair_count - 2)]
        random.shuffle(orientations)
    else:
        orientations = [random.choice(("h", "v")) for _ in range(pair_count)]

    specs: list[dict[str, object]] = []
    seed_idx = 0
    for endpoint_color, orientation, seed_count in zip(endpoint_colors, orientations, seed_counts):
        seed_colors = seed_palette[seed_idx : seed_idx + seed_count]
        seed_idx += seed_count
        specs.append(
            {
                "endpoint_color": int(endpoint_color),
                "orientation": orientation,
                "seed_colors": [int(color) for color in seed_colors],
            }
        )

    specs.sort(key=lambda spec: (_min_distance(len(spec["seed_colors"])), random.random()), reverse=True)
    return specs


def _path_cells(
    orientation: str,
    fixed: int,
    start: int,
    distance: int,
) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(fixed, start + offset) for offset in range(distance + 1)]
    return [(start + offset, fixed) for offset in range(distance + 1)]


def _place_spec(
    grid: np.ndarray,
    blocked: np.ndarray,
    spec: dict[str, object],
) -> dict[str, object] | None:
    h, w = grid.shape
    orientation = str(spec["orientation"])
    seed_colors = list(spec["seed_colors"])
    endpoint_color = int(spec["endpoint_color"])

    axial_limit = w if orientation == "h" else h
    cross_limit = h if orientation == "h" else w
    max_distance = axial_limit - 1 - 2 * PATH_MARGIN
    min_distance = _min_distance(len(seed_colors))

    if cross_limit <= 2 * PATH_MARGIN or max_distance < min_distance:
        return None

    max_extra = min(MAX_DISTANCE_EXTRA, max_distance - min_distance)

    for _ in range(PLACEMENT_TRIES):
        distance = min_distance + random.randint(0, max_extra)
        fixed = random.randint(PATH_MARGIN, cross_limit - 1 - PATH_MARGIN)
        start = random.randint(PATH_MARGIN, axial_limit - 1 - PATH_MARGIN - distance)
        cells = _path_cells(orientation, fixed, start, distance)

        if any(blocked[r, c] for r, c in cells):
            continue

        seed_positions = _choose_seed_positions(distance, len(seed_colors))
        grid[cells[0]] = endpoint_color
        grid[cells[-1]] = endpoint_color
        for pos, color in zip(seed_positions, seed_colors):
            r, c = cells[pos]
            grid[r, c] = int(color)

        _reserve_cells(blocked, cells)
        return {
            "endpoint_color": endpoint_color,
            "orientation": orientation,
            "cells": cells,
            "seed_positions": seed_positions,
            "seed_colors": seed_colors,
        }

    return None


def _render_expected(
    grid: np.ndarray,
    placed_specs: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for spec in placed_specs:
        endpoint_color = int(spec["endpoint_color"])
        cells = list(spec["cells"])
        seed_positions = list(spec["seed_positions"])
        seed_colors = list(spec["seed_colors"])

        markers = [(0, endpoint_color)]
        markers.extend((int(pos), int(color)) for pos, color in zip(seed_positions, seed_colors))
        markers.append((len(cells) - 1, endpoint_color))

        for (start_idx, fill_color), (stop_idx, _next_color) in zip(markers, markers[1:]):
            for idx in range(int(start_idx) + 1, int(stop_idx)):
                r, c = cells[idx]
                out[r, c] = int(fill_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    specs = _build_specs()
    if specs is None:
        return None

    h = random.randint(HEIGHT_MIN, HEIGHT_MAX)
    w = random.randint(WIDTH_MIN, WIDTH_MAX)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    placed_specs: list[dict[str, object]] = []
    for spec in specs:
        placed = _place_spec(grid, blocked, spec)
        if placed is None:
            return None
        placed_specs.append(placed)

    expected = _render_expected(grid, placed_specs)
    if np.array_equal(expected, grid):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
