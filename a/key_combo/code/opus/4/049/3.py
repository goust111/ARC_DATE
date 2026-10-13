from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
LEGEND_WIDTH = 2
SIZE_CHOICES = (13, 15, 17, 19)
ENTRY_COUNT_CHOICES = (3, 4)
COLOR_POOL = tuple(range(1, 10))
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
SLOPE_POSITIVE = {(-1, -1), (1, 1)}
SLOPE_NEGATIVE = {(-1, 1), (1, -1)}
MAX_GENERATION_ATTEMPTS = 300

# Two incomparable offsets in the same diagonal quadrant uniquely identify the source.
SATELLITE_PAIRS = (
    ((1, 2), (2, 1)),
    ((1, 3), (3, 1)),
    ((2, 3), (3, 2)),
    ((1, 3), (2, 1)),
    ((1, 2), (3, 1)),
    ((1, 3), (2, 2)),
    ((2, 2), (3, 1)),
)


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _ray_cells(source: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r, c = map(int, source)
    dr, dc = map(int, direction)
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _legend_entries(grid: np.ndarray) -> list[tuple[int, int, int]]:
    rows = [
        (int(r), int(grid[r, 0]), int(grid[r, 1]))
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != BG and int(grid[r, 1]) != BG
    ]
    assert 3 <= len(rows) <= 4, "expected 3-4 legend rows in the left two columns"
    markers = [marker for _, marker, _ in rows]
    assert len(set(markers)) == len(markers), "legend marker colors must be unique"
    return rows


def _infer_source_and_direction(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = [tuple(map(int, pt)) for pt in coords]
    assert len(pts) >= 3, "each source color needs one source and at least two satellites"
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for sr, sc in pts:
        signs = {
            (_sign(r - sr), _sign(c - sc))
            for r, c in pts
            if (r, c) != (sr, sc)
        }
        if len(signs) != 1:
            continue
        direction = next(iter(signs))
        if 0 in direction:
            continue
        candidates.append(((sr, sc), direction))
    assert len(candidates) == 1, "same-color cluster must imply a unique source pixel"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the source clusters and redraw color-mapped diagonal rays using legend order as crossing priority.

    Input structure:
    - The grid is an odd square on a black background.
    - The left two columns contain 3-4 spaced legend rows; each row is a marker-color pixel beside its ray-color pixel.
    - In the main area (from column 2 onward), each legend marker color appears as one source pixel plus same-color satellite pixels in a single diagonal quadrant.

    Rule:
    - Read the legend rows to build marker-color -> ray-color pairs and keep their row positions.
    - For each marker color in the main area, identify the unique pixel from which all other same-color pixels lie in one diagonal quadrant; that quadrant gives the ray direction.
    - Clear the entire main area to background while leaving the legend untouched.
    - Draw one diagonal ray per source, starting at the source and continuing to the grid edge, using the mapped ray color.
    - Process legend rows from bottom to top; later rays overwrite earlier ones, so intersections keep the color of the higher legend row.

    Key insight:
    - The satellites encode direction, the legend encodes the visible ray color, and the legend's vertical order silently encodes overwrite priority at crossings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 1, "expected an odd square grid"
    assert w >= LEGEND_WIDTH + 3, "grid must have a legend zone and a main area"

    entries = _legend_entries(g)
    main_mask = np.ones_like(g, dtype=bool)
    main_mask[:, :LEGEND_WIDTH] = False

    source_info: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for _row, marker_color, _ray_color in entries:
        coords = np.argwhere((g == marker_color) & main_mask)
        assert coords.shape[0] >= 3, "each marker color must have a source with satellites"
        source_info[marker_color] = _infer_source_and_direction(map(tuple, coords.tolist()))

    out = g.copy()
    out[:, LEGEND_WIDTH:] = BG
    for row, marker_color, ray_color in sorted(entries, key=lambda item: item[0], reverse=True):
        del row
        source, direction = source_info[marker_color]
        for r, c in _ray_cells(source, direction, g.shape):
            out[r, c] = int(ray_color)
    return out


def _cluster_cells(source: tuple[int, int], direction: tuple[int, int], pair: tuple[tuple[int, int], tuple[int, int]]) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    sr, sc = map(int, source)
    cells = [source]
    for ar, ac in pair:
        cells.append((sr + dr * ar, sc + dc * ac))
    return cells


def _cluster_clear(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    for r, c in cells:
        if not _in_bounds(shape, r, c) or c < LEGEND_WIDTH:
            return False
        for rr, cc in occupied:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return False
    return len(set(cells)) == len(cells)


def _all_ray_cells_main_area(source: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> bool:
    return all(c >= LEGEND_WIDTH for _, c in _ray_cells(source, direction, shape))


def _place_satellites(
    rng: random.Random,
    source: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    pairs = list(SATELLITE_PAIRS)
    rng.shuffle(pairs)
    for pair in pairs:
        cells = _cluster_cells(source, direction, pair)
        if _cluster_clear(cells, occupied, shape):
            return cells[1:]
    return None


def _choose_legend_rows(rng: random.Random, side: int, count: int) -> list[int]:
    candidates = list(range(side))
    for _ in range(200):
        rows = sorted(rng.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    raise RuntimeError("failed to place spaced legend rows")


def _crossing_pair(
    rng: random.Random,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[dict] | None:
    h, w = map(int, shape)
    for _ in range(300):
        intersection = (
            rng.randint(2, h - 3),
            rng.randint(LEGEND_WIDTH + 2, w - 3),
        )
        d1 = rng.choice(DIRECTIONS)
        d2_pool = tuple(SLOPE_NEGATIVE if d1 in SLOPE_POSITIVE else SLOPE_POSITIVE)
        d2 = rng.choice(d2_pool)

        choices1 = []
        choices2 = []
        for step_back in range(2, 7):
            s1 = (intersection[0] - step_back * d1[0], intersection[1] - step_back * d1[1])
            if _in_bounds(shape, *s1) and _all_ray_cells_main_area(s1, d1, shape):
                choices1.append(s1)
            s2 = (intersection[0] - step_back * d2[0], intersection[1] - step_back * d2[1])
            if _in_bounds(shape, *s2) and _all_ray_cells_main_area(s2, d2, shape):
                choices2.append(s2)
        rng.shuffle(choices1)
        rng.shuffle(choices2)

        for s1 in choices1:
            sats1 = _place_satellites(rng, s1, d1, occupied, shape)
            if sats1 is None:
                continue
            occ1 = occupied | {s1, *sats1}
            for s2 in choices2:
                if s2 == s1:
                    continue
                sats2 = _place_satellites(rng, s2, d2, occ1, shape)
                if sats2 is None:
                    continue
                if set(_ray_cells(s1, d1, shape)) & set(_ray_cells(s2, d2, shape)) != {intersection}:
                    continue
                return [
                    {"source": s1, "direction": d1, "satellites": sats1},
                    {"source": s2, "direction": d2, "satellites": sats2},
                ]
    return None


def _random_ray(
    rng: random.Random,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    specs: list[dict],
) -> dict | None:
    h, w = map(int, shape)
    for _ in range(400):
        direction = rng.choice(DIRECTIONS)
        source = (
            rng.randint(1, h - 2),
            rng.randint(LEGEND_WIDTH + 1, w - 2),
        )
        if not _all_ray_cells_main_area(source, direction, shape):
            continue
        satellites = _place_satellites(rng, source, direction, occupied, shape)
        if satellites is None:
            continue
        new_path = set(_ray_cells(source, direction, shape))
        bad = False
        for spec in specs:
            old_path = set(_ray_cells(spec["source"], spec["direction"], shape))
            overlap = new_path & old_path
            if len(overlap) > 1:
                bad = True
                break
        if not bad:
            return {"source": source, "direction": direction, "satellites": satellites}
    return None


def _render_input(side: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((side, side), dtype=int)
    for spec in specs:
        row = int(spec["legend_row"])
        marker = int(spec["marker"])
        ray = int(spec["ray"])
        grid[row, 0] = marker
        grid[row, 1] = ray
        for r, c in [spec["source"], *spec["satellites"]]:
            grid[r, c] = marker
    return grid


def _render_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    out[:, LEGEND_WIDTH:] = BG
    for spec in sorted(specs, key=lambda item: item["legend_row"], reverse=True):
        for r, c in _ray_cells(spec["source"], spec["direction"], inp.shape):
            out[r, c] = int(spec["ray"])
    return out


def _render_with_order(inp: np.ndarray, specs: list[dict], reverse: bool) -> np.ndarray:
    out = inp.copy()
    out[:, LEGEND_WIDTH:] = BG
    for spec in sorted(specs, key=lambda item: item["legend_row"], reverse=reverse):
        for r, c in _ray_cells(spec["source"], spec["direction"], inp.shape):
            out[r, c] = int(spec["ray"])
    return out


def _intersection_map(specs: list[dict], shape: tuple[int, int]) -> dict[tuple[int, int], list[dict]]:
    paths = [set(_ray_cells(spec["source"], spec["direction"], shape)) for spec in specs]
    out: dict[tuple[int, int], list[dict]] = {}
    for i, spec_i in enumerate(specs):
        for j in range(i + 1, len(specs)):
            cells = paths[i] & paths[j]
            if cells:
                for cell in cells:
                    out.setdefault(cell, [])
                    if spec_i not in out[cell]:
                        out[cell].append(spec_i)
                    if specs[j] not in out[cell]:
                        out[cell].append(specs[j])
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    intersections = _intersection_map(specs, inp.shape)
    if not intersections:
        return False

    changed_by_priority = _render_with_order(inp, specs, reverse=False)
    if np.array_equal(changed_by_priority, expected):
        return False

    for spec in specs:
        cluster = [spec["source"], *spec["satellites"]]
        try:
            inferred_source, inferred_dir = _infer_source_and_direction(cluster)
        except AssertionError:
            return False
        if inferred_source != spec["source"] or inferred_dir != spec["direction"]:
            return False
        if spec["marker"] == spec["ray"]:
            return False

    for cell, owners in intersections.items():
        if len(owners) < 2:
            continue
        winner = min(owners, key=lambda item: item["legend_row"])
        if int(expected[cell]) != int(winner["ray"]):
            return False

    return True


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(SIZE_CHOICES)
    count = rng.choice(ENTRY_COUNT_CHOICES)
    colors = list(rng.sample(COLOR_POOL, 2 * count))
    marker_colors = colors[:count]
    ray_colors = colors[count:]
    rng.shuffle(ray_colors)

    specs = _crossing_pair(rng, (side, side), set())
    if specs is None:
        return None

    occupied = set()
    for spec in specs:
        occupied.update([spec["source"], *spec["satellites"]])

    while len(specs) < count:
        spec = _random_ray(rng, (side, side), occupied, specs)
        if spec is None:
            return None
        occupied.update([spec["source"], *spec["satellites"]])
        specs.append(spec)

    rows = _choose_legend_rows(rng, side, count)
    rng.shuffle(specs)
    for spec, row, marker, ray in zip(specs, rows, marker_colors, ray_colors):
        spec["legend_row"] = row
        spec["marker"] = marker
        spec["ray"] = ray

    inp = _render_input(side, specs)
    expected = _render_expected(inp, specs)
    if not _semantic_checks(inp, expected, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-ray ARC input")
