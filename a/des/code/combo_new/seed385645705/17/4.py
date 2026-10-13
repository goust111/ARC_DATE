from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
PairSpec = dict[str, object]


GRID_MIN = 12
GRID_MAX = 18
MAX_PAIRS = 5
MAX_ATTEMPTS = 400
PALETTE = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _pair_key(points: tuple[Coord, Coord]) -> Coord:
    return min(points)


def _pair_cells(a: Coord, b: Coord) -> list[Coord]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]

    start, end = sorted((a, b))
    sr, sc = start
    er, ec = end
    h_step = 1 if ec >= sc else -1
    cells = [(sr, c) for c in range(sc, ec + h_step, h_step)]
    cells.extend((r, ec) for r in range(sr + 1, er + 1))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored marker pair with a straight or horizontal-then-vertical L path, letting later pairs overwrite earlier crossings.

    Input structure:
    - The background is the dominant color.
    - Every non-background color appears exactly twice, as two singleton marker pixels.
    - Each same-color pair is either row-aligned, column-aligned, or forms an L whose upper marker is connected first horizontally and then vertically.

    Rule:
    - Sort the color pairs by the lexicographically earlier marker: smaller row first, then smaller column.
    - For a same-row pair, fill every cell between the two markers on that row; for a same-column pair, fill every cell between them on that column.
    - For any remaining pair, start from the upper marker, draw horizontally to the other marker's column, then draw vertically to the lower marker; later paths overwrite earlier ones at intersections.

    Key insight:
    - The task has two layers: infer which geometric connector each pair implies, then resolve crossings using the pair order induced by marker positions rather than by the color ids themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one non-background pair"

    pairs: list[tuple[Coord, int, tuple[Coord, Coord]]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each non-background color must appear exactly twice"
        points = tuple((int(r), int(c)) for r, c in coords)
        pairs.append((_pair_key(points), color, points))

    out = np.full_like(g, bg)
    for _key, color, points in sorted(pairs, key=lambda item: item[0]):
        for r, c in _pair_cells(points[0], points[1]):
            out[r, c] = color
    return out


def _well_spaced(points: tuple[Coord, Coord], used_markers: set[Coord]) -> bool:
    for r, c in points:
        for rr, cc in used_markers:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return False
    return True


def _add_pair(kind: str, endpoints: tuple[Coord, Coord]) -> PairSpec:
    cells = tuple(_pair_cells(endpoints[0], endpoints[1]))
    return {"kind": kind, "endpoints": endpoints, "cells": cells, "key": _pair_key(endpoints)}


def _can_place_disjoint(spec: PairSpec, used_markers: set[Coord], used_paths: set[Coord]) -> bool:
    endpoints = tuple(spec["endpoints"])  # type: ignore[arg-type]
    cells = set(spec["cells"])  # type: ignore[arg-type]
    if not _well_spaced(endpoints, used_markers):
        return False
    if any(point in used_markers for point in endpoints):
        return False
    if any(cell in used_markers for cell in cells):
        return False
    if cells & used_paths:
        return False
    return True


def _sample_disjoint_pair(
    kind: str,
    height: int,
    width: int,
    used_markers: set[Coord],
    used_paths: set[Coord],
) -> PairSpec | None:
    for _ in range(200):
        if kind == "row":
            r = random.randint(1, height - 2)
            c0 = random.randint(1, width - 4)
            c1 = random.randint(c0 + 2, width - 2)
            endpoints = ((r, c0), (r, c1))
        elif kind == "col":
            c = random.randint(1, width - 2)
            r0 = random.randint(1, height - 4)
            r1 = random.randint(r0 + 2, height - 2)
            endpoints = ((r0, c), (r1, c))
        else:
            r0 = random.randint(1, height - 4)
            r1 = random.randint(r0 + 2, height - 2)
            c0 = random.randint(1, width - 2)
            options = [c for c in range(1, width - 1) if abs(c - c0) >= 2]
            if not options:
                continue
            c1 = random.choice(options)
            endpoints = ((r0, c0), (r1, c1))

        spec = _add_pair(kind, endpoints)
        if _can_place_disjoint(spec, used_markers, used_paths):
            return spec
    return None


def _sample_hv_cross(height: int, width: int) -> list[PairSpec] | None:
    for _ in range(100):
        cross_r = random.randint(4, height - 5)
        cross_c = random.randint(4, width - 5)

        c0 = random.randint(1, cross_c - 2)
        c1 = random.randint(cross_c + 2, width - 2)
        r0 = random.randint(1, cross_r - 2)
        r1 = random.randint(cross_r + 2, height - 2)

        row_spec = _add_pair("row", ((cross_r, c0), (cross_r, c1)))
        col_spec = _add_pair("col", ((r0, cross_c), (r1, cross_c)))
        if set(row_spec["cells"]) & set(col_spec["cells"]) == {(cross_r, cross_c)}:
            return [row_spec, col_spec]
    return None


def _sample_ll_cross(height: int, width: int) -> list[PairSpec] | None:
    for _ in range(100):
        early_row = random.randint(1, height - 7)
        late_row = random.randint(early_row + 2, height - 4)
        early_end_row = random.randint(late_row + 2, height - 2)
        late_end_row = random.randint(late_row + 2, height - 2)

        cross_col = random.randint(3, width - 4)
        if random.random() < 0.5:
            early_start_col = random.randint(1, cross_col - 2)
        else:
            early_start_col = random.randint(cross_col + 2, width - 2)

        if random.random() < 0.5:
            late_start_col = random.randint(1, cross_col - 1)
            late_end_col = random.randint(cross_col + 1, width - 2)
        else:
            late_start_col = random.randint(cross_col + 1, width - 2)
            late_end_col = random.randint(1, cross_col - 1)

        early_spec = _add_pair("l", ((early_row, early_start_col), (early_end_row, cross_col)))
        late_spec = _add_pair("l", ((late_row, late_start_col), (late_end_row, late_end_col)))
        if set(early_spec["cells"]) & set(late_spec["cells"]) == {(late_row, cross_col)}:
            return [early_spec, late_spec]
    return None


def _register(spec: PairSpec, pair_specs: list[PairSpec], used_markers: set[Coord], used_paths: set[Coord]) -> None:
    pair_specs.append(spec)
    used_markers.update(spec["endpoints"])  # type: ignore[arg-type]
    used_paths.update(spec["cells"])  # type: ignore[arg-type]


def _crossing_count(pair_specs: list[PairSpec]) -> int:
    total = 0
    for i, spec_a in enumerate(pair_specs):
        cells_a = set(spec_a["cells"])  # type: ignore[arg-type]
        for spec_b in pair_specs[i + 1 :]:
            cells_b = set(spec_b["cells"])  # type: ignore[arg-type]
            total += len(cells_a & cells_b)
    return total


def _oracle_ok(grid: np.ndarray, pair_specs: list[PairSpec], bg: int) -> bool:
    non_bg = [int(color) for color in np.unique(grid) if int(color) != bg]
    if len(non_bg) != len(pair_specs):
        return False
    if any(int(np.sum(grid == color)) != 2 for color in non_bg):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if _crossing_count(pair_specs) < 1:
        return False
    return int(np.sum(out != bg)) > int(np.sum(grid != bg))


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        height = random.randint(GRID_MIN, GRID_MAX)
        width = random.randint(GRID_MIN, GRID_MAX)
        motif = random.choices(("hv", "ll"), weights=(2, 1))[0]
        min_pairs = 4 if motif == "ll" else 4
        pair_count = random.randint(min_pairs, MAX_PAIRS)

        bg = random.choice(PALETTE)
        colors = random.sample([color for color in PALETTE if color != bg], pair_count)
        grid = np.full((height, width), bg, dtype=int)

        pair_specs: list[PairSpec] = []
        used_markers: set[Coord] = set()
        used_paths: set[Coord] = set()

        motif_specs = _sample_hv_cross(height, width) if motif == "hv" else _sample_ll_cross(height, width)
        if motif_specs is None:
            continue
        for spec in motif_specs:
            _register(spec, pair_specs, used_markers, used_paths)

        required = ["l"] if motif == "hv" else ["row", "col"]
        failed = False
        for kind in required:
            spec = _sample_disjoint_pair(kind, height, width, used_markers, used_paths)
            if spec is None:
                failed = True
                break
            _register(spec, pair_specs, used_markers, used_paths)
        if failed:
            continue

        while len(pair_specs) < pair_count:
            extra_kinds = ("row", "col", "l") if motif == "hv" else ("row", "col")
            kinds = list(extra_kinds)
            random.shuffle(kinds)
            added = False
            for kind in kinds:
                spec = _sample_disjoint_pair(kind, height, width, used_markers, used_paths)
                if spec is None:
                    continue
                _register(spec, pair_specs, used_markers, used_paths)
                added = True
                break
            if not added:
                failed = True
                break
        if failed:
            continue

        if len(pair_specs) != pair_count:
            continue

        random.shuffle(colors)
        for color, spec in zip(colors, pair_specs):
            for r, c in spec["endpoints"]:  # type: ignore[assignment]
                grid[r, c] = color

        if _oracle_ok(grid, pair_specs, bg):
            return grid

    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
