from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 500
LAYOUT_RETRIES = 600
GRID_HEIGHT_RANGE = (10, 22)
GRID_WIDTH_RANGE = (10, 22)
LINE_COUNT_RANGE = (2, 4)
MIN_LINE_LENGTH = 5
MIN_CROSSINGS = 2
MAX_CROSSINGS = 6


def _is_border_cell(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _is_corner(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}


def _diag_cells(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert abs(dr) == abs(dc) and dr != 0
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    return tuple((r0 + k * step_r, c0 + k * step_c) for k in range(abs(dr) + 1))


def _unique_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for point in points:
        if point not in out:
            out.append(point)
    return out


def _line_endpoints_main(shape: tuple[int, int], key: int) -> list[tuple[int, int]]:
    h, w = shape
    points: list[tuple[int, int]] = []
    if 0 <= -key < w:
        points.append((0, -key))
    if 0 <= h - 1 - key < w:
        points.append((h - 1, h - 1 - key))
    if 0 <= key < h:
        points.append((key, 0))
    if 0 <= key + w - 1 < h:
        points.append((key + w - 1, w - 1))
    return _unique_points(points)


def _line_endpoints_anti(shape: tuple[int, int], key: int) -> list[tuple[int, int]]:
    h, w = shape
    points: list[tuple[int, int]] = []
    if 0 <= key < w:
        points.append((0, key))
    if 0 <= key - (h - 1) < w:
        points.append((h - 1, key - (h - 1)))
    if 0 <= key < h:
        points.append((key, 0))
    if 0 <= key - (w - 1) < h:
        points.append((key - (w - 1), w - 1))
    return _unique_points(points)


def _candidate_line(shape: tuple[int, int], orientation: str, key: int) -> dict[str, object] | None:
    if orientation == "main":
        endpoints = _line_endpoints_main(shape, key)
    else:
        endpoints = _line_endpoints_anti(shape, key)
    if len(endpoints) != 2:
        return None
    if any(_is_corner(shape, r, c) for r, c in endpoints):
        return None
    a, b = sorted(endpoints)
    cells = _diag_cells(a, b)
    if len(cells) < MIN_LINE_LENGTH:
        return None
    return {
        "orientation": orientation,
        "key": int(key),
        "endpoints": (a, b),
        "cells": cells,
        "cell_set": frozenset(cells),
    }


def _enumerate_lines(shape: tuple[int, int], orientation: str) -> list[dict[str, object]]:
    h, w = shape
    lines: list[dict[str, object]] = []
    if orientation == "main":
        keys = range(-(w - 1), h)
    else:
        keys = range(h + w - 1)
    for key in keys:
        line = _candidate_line(shape, orientation, key)
        if line is not None:
            lines.append(line)
    return lines


def _recover_lines(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    lines: list[dict[str, object]] = []
    seen_endpoints: set[tuple[int, int]] = set()
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each color must appear exactly twice"
        points = [tuple(map(int, coord)) for coord in coords.tolist()]
        for r, c in points:
            assert _is_border_cell((h, w), r, c), "markers must lie on the outer border"
            assert (r, c) not in seen_endpoints, "border markers must occupy distinct cells"
            seen_endpoints.add((r, c))
        a, b = sorted(points)
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        assert abs(dr) == abs(dc) and dr != 0, "same-color markers must define a 45-degree segment"
        orientation = "main" if dr * dc > 0 else "anti"
        cells = _diag_cells(a, b)
        line = {
            "color": color,
            "orientation": orientation,
            "key": int(a[0] - a[1] if orientation == "main" else a[0] + a[1]),
            "endpoints": (a, b),
            "cells": cells,
            "cell_set": frozenset(cells),
        }
        lines.append(line)

    main_cells: set[tuple[int, int]] = set()
    anti_cells: set[tuple[int, int]] = set()
    main_keys: set[int] = set()
    anti_keys: set[int] = set()
    for line in lines:
        if line["orientation"] == "main":
            assert int(line["key"]) not in main_keys
            main_keys.add(int(line["key"]))
            assert main_cells.isdisjoint(line["cell_set"])
            main_cells.update(line["cell_set"])
        else:
            assert int(line["key"]) not in anti_keys
            anti_keys.add(int(line["key"]))
            assert anti_cells.isdisjoint(line["cell_set"])
            anti_cells.update(line["cell_set"])
    return lines


def _crossings_from_lines(
    shape: tuple[int, int], lines: list[dict[str, object]]
) -> tuple[np.ndarray, list[int], list[int], list[tuple[int, int]]]:
    h, w = shape
    mains = [line for line in lines if line["orientation"] == "main"]
    antis = [line for line in lines if line["orientation"] == "anti"]
    out = np.zeros((h, w), dtype=int)
    main_degrees = [0] * len(mains)
    anti_degrees = [0] * len(antis)
    cross_cells: list[tuple[int, int]] = []
    for i, main_line in enumerate(mains):
        for j, anti_line in enumerate(antis):
            shared = main_line["cell_set"] & anti_line["cell_set"]
            assert len(shared) <= 1, "opposite diagonal families may share at most one cell"
            if not shared:
                continue
            cell = next(iter(shared))
            out[cell] = int(main_line.get("color", 1))
            main_degrees[i] += 1
            anti_degrees[j] += 1
            cross_cells.append(cell)
    return out, main_degrees, anti_degrees, cross_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each invisible border-to-border diagonal and keep only the cells where a main diagonal crosses an anti-diagonal.

    Input structure:
    - The background is black.
    - Every non-black color appears exactly twice, and both same-color cells lie on the outer border.
    - Each same-color border pair can be connected by one 45-degree diagonal segment; such segments belong either to the main-diagonal family or the anti-diagonal family.

    Rule:
    - For every same-color pair, reconstruct the diagonal segment between the two border markers, but do not draw that segment itself.
    - Whenever one main-diagonal segment and one anti-diagonal segment share a grid cell, color that shared cell with the main-diagonal segment's color.
    - Turn every other cell black, including all original border markers.

    Key insight:
    - The visible border markers are only endpoints of hidden diagonals; the output is determined entirely by where those unseen diagonals intersect and by the fact that main-diagonal color overrides anti-diagonal color at each crossing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    lines = _recover_lines(g)
    out, _main_degrees, _anti_degrees, _cross_cells = _crossings_from_lines(g.shape, lines)
    return out


def _endpoints_disjoint(mains: list[dict[str, object]], antis: list[dict[str, object]]) -> bool:
    occupied: set[tuple[int, int]] = set()
    for line in mains + antis:
        for point in line["endpoints"]:
            if point in occupied:
                return False
            occupied.add(point)
    return True


def _points_are_isolated(points: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _build_input(shape: tuple[int, int], lines: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for line in lines:
        color = int(line["color"])
        for r, c in line["endpoints"]:
            grid[r, c] = color
    return grid


def _analyze_generated(shape: tuple[int, int], lines: list[dict[str, object]]) -> dict[str, object]:
    output, main_degrees, anti_degrees, cross_cells = _crossings_from_lines(shape, lines)
    return {
        "output": output,
        "main_degrees": main_degrees,
        "anti_degrees": anti_degrees,
        "cross_cells": cross_cells,
        "cross_total": len(cross_cells),
        "crossed_main_count": sum(deg > 0 for deg in main_degrees),
    }


def _layout_ok(shape: tuple[int, int], mains: list[dict[str, object]], antis: list[dict[str, object]]) -> bool:
    if not _endpoints_disjoint(mains, antis):
        return False
    if len(mains) < 2 or len(antis) < 2:
        return False
    analysis = _analyze_generated(shape, mains + antis)
    main_degrees = analysis["main_degrees"]
    anti_degrees = analysis["anti_degrees"]
    cross_cells = analysis["cross_cells"]
    cross_total = int(analysis["cross_total"])
    crossed_main_count = int(analysis["crossed_main_count"])
    if not (MIN_CROSSINGS <= cross_total <= MAX_CROSSINGS):
        return False
    if cross_total >= len(mains) * len(antis):
        return False
    if max(main_degrees, default=0) < 2 or max(anti_degrees, default=0) < 2:
        return False
    if not any(deg == 0 for deg in main_degrees + anti_degrees):
        return False
    if crossed_main_count < 2:
        return False
    if not _points_are_isolated(cross_cells):
        return False
    return True


def _sample_layout(shape: tuple[int, int]) -> tuple[list[dict[str, object]], list[dict[str, object]]] | None:
    main_candidates = _enumerate_lines(shape, "main")
    anti_candidates = _enumerate_lines(shape, "anti")
    if len(main_candidates) < 4 or len(anti_candidates) < 4:
        return None

    for _ in range(LAYOUT_RETRIES):
        main_count = random.randint(*LINE_COUNT_RANGE)
        anti_count = random.randint(*LINE_COUNT_RANGE)
        if main_count + anti_count < 5:
            continue
        mains = random.sample(main_candidates, main_count)
        antis = random.sample(anti_candidates, anti_count)
        if _layout_ok(shape, mains, antis):
            return mains, antis
    return None


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        recovered = _recover_lines(inp)
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    analysis = _analyze_generated(inp.shape, recovered)
    if not np.array_equal(analysis["output"], expected):
        return False
    if int(np.count_nonzero(inp)) != 2 * len(recovered):
        return False
    if int(np.count_nonzero(expected)) != int(analysis["cross_total"]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_HEIGHT_RANGE),
        random.randint(*GRID_WIDTH_RANGE),
    )
    layout = _sample_layout(shape)
    if layout is None:
        return None
    mains, antis = layout

    colors = random.sample(COLOR_POOL, len(mains) + len(antis))
    colored_lines: list[dict[str, object]] = []
    for line, color in zip(mains + antis, colors):
        inst = dict(line)
        inst["color"] = int(color)
        colored_lines.append(inst)

    analysis = _analyze_generated(shape, colored_lines)
    expected = analysis["output"]
    if int(analysis["cross_total"]) < MIN_CROSSINGS:
        return None

    inp = _build_input(shape, colored_lines)
    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-diagonal puzzle input")
