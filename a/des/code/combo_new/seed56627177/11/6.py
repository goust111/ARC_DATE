from __future__ import annotations

import random

import numpy as np


BG = 0
CROSS_COLOR = 4
PAIR_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]

Coord = tuple[int, int]
PathRecord = dict[str, object]


def _ordered_endpoints(a: Coord, b: Coord) -> tuple[Coord, Coord]:
    if a[1] < b[1]:
        return a, b
    if a[1] > b[1]:
        return b, a
    return (a, b) if a[0] <= b[0] else (b, a)


def _inclusive_range(start: int, end: int) -> range:
    step = 1 if end >= start else -1
    return range(start, end + step, step)


def _make_path(color: int, a: Coord, b: Coord) -> PathRecord:
    left, right = _ordered_endpoints(a, b)
    r0, c0 = left
    r1, c1 = right

    full: set[Coord] = set()
    for c in _inclusive_range(c0, c1):
        full.add((r0, c))
    for r in _inclusive_range(r0, r1):
        full.add((r, c1))

    if r0 == r1:
        kind = "h"
    elif c0 == c1:
        kind = "v"
    else:
        kind = "l"

    endpoints = {left, right}
    inner = full - endpoints
    return {
        "color": int(color),
        "left": left,
        "right": right,
        "endpoints": endpoints,
        "full": full,
        "inner": inner,
        "kind": kind,
    }


def _extract_paths(grid: np.ndarray) -> list[PathRecord]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    paths: list[PathRecord] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 2, "each non-black color must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        paths.append(_make_path(color, a, b))

    assert paths, "expected at least one color pair"
    return paths


def _render_paths(grid: np.ndarray, paths: list[PathRecord]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    endpoint_cells = {
        cell
        for path in paths
        for cell in path["endpoints"]  # type: ignore[index]
    }

    coverage: dict[Coord, set[int]] = {}
    for path in paths:
        color = int(path["color"])
        for cell in path["inner"]:  # type: ignore[index]
            coverage.setdefault(cell, set()).add(color)

    for cell, colors in coverage.items():
        if cell in endpoint_cells:
            continue
        assert int(out[cell]) == BG, "non-endpoint path cells must start black"
        out[cell] = CROSS_COLOR if len(colors) >= 2 else next(iter(colors))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair with a left-to-right L path and mark path crossings in yellow.

    Input structure:
    - Black background with several non-black colors, each appearing in exactly two single-pixel endpoints.
    - Each color pair defines one hidden orthogonal connection.
    - Different color pairs are distinct, and only the endpoints are visible in the input.

    Rule:
    - For each color pair, choose the left endpoint as the one with the smaller column; if both endpoints share a column, use the upper one first.
    - Draw that pair's path by moving horizontally from the left endpoint to the right endpoint's column, then vertically along that column to the right endpoint.
    - Fill every black non-endpoint cell on exactly one path with that path's color; if a non-endpoint cell belongs to multiple paths, color it yellow.
    - Keep all original endpoint pixels unchanged.

    Key insight:
    - The hidden connection is not a symmetric shortest path: every pair uses the same ordered "horizontal first, then vertical" construction, and only shared path cells become the uniform yellow crossing marker.
    """
    g = np.asarray(grid, dtype=int)
    paths = _extract_paths(g)
    return _render_paths(g, paths)


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 22
PAIR_COUNT_BY_MODE = {
    "cross": (2, 4),
    "horizontal": (2, 4),
    "vertical": (2, 4),
    "mixed": (3, 4),
}
GEN_RETRIES = 400
PATH_RETRIES = 120
CROSS_RETRIES = 200
MODE_NAMES = ["cross", "horizontal", "vertical", "mixed"]
_MODE_BAG: list[str] = []


def _next_mode() -> str:
    global _MODE_BAG
    if not _MODE_BAG:
        _MODE_BAG = random.sample(MODE_NAMES, k=len(MODE_NAMES))
    return _MODE_BAG.pop()


def _render_input(shape: tuple[int, int], paths: list[PathRecord]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for path in paths:
        color = int(path["color"])
        for cell in path["endpoints"]:  # type: ignore[index]
            assert grid[cell] == BG, "endpoints must be distinct"
            grid[cell] = color
    return grid


def _cross_cells(paths: list[PathRecord]) -> set[Coord]:
    coverage: dict[Coord, set[int]] = {}
    for path in paths:
        color = int(path["color"])
        for cell in path["inner"]:  # type: ignore[index]
            coverage.setdefault(cell, set()).add(color)
    return {cell for cell, colors in coverage.items() if len(colors) >= 2}


def _path_fits(existing: list[PathRecord], cand: PathRecord) -> bool:
    cand_end = cand["endpoints"]  # type: ignore[assignment]
    cand_full = cand["full"]  # type: ignore[assignment]
    cand_inner = cand["inner"]  # type: ignore[assignment]

    for path in existing:
        path_end = path["endpoints"]  # type: ignore[assignment]
        path_full = path["full"]  # type: ignore[assignment]
        path_inner = path["inner"]  # type: ignore[assignment]
        if cand_end & path_full:
            return False
        if path_end & cand_full:
            return False
        if cand_inner & path_inner:
            return False
    return True


def _sample_random_path(shape: tuple[int, int], color: int, kind: str) -> PathRecord | None:
    h, w = shape
    if h < 6 or w < 6:
        return None

    for _ in range(PATH_RETRIES):
        if kind == "h":
            if h < 3 or w < 7:
                return None
            row = random.randint(1, h - 2)
            c0 = random.randint(1, w - 5)
            c1 = random.randint(c0 + 3, w - 2)
            return _make_path(color, (row, c0), (row, c1))

        if kind == "v":
            if h < 7 or w < 3:
                return None
            col = random.randint(1, w - 2)
            rows = [r for r in range(1, h - 1)]
            r0, r1 = random.sample(rows, 2)
            if abs(r0 - r1) < 3:
                continue
            return _make_path(color, (r0, col), (r1, col))

        if kind == "l":
            if h < 6 or w < 7:
                return None
            c0 = random.randint(1, w - 5)
            c1 = random.randint(c0 + 2, w - 2)
            r0 = random.randint(1, h - 2)
            candidates = [r for r in range(1, h - 1) if abs(r - r0) >= 2]
            if not candidates:
                continue
            r1 = random.choice(candidates)
            return _make_path(color, (r0, c0), (r1, c1))

        raise ValueError(f"unknown path kind: {kind}")

    return None


def _build_crossing_pair(shape: tuple[int, int], color_a: int, color_b: int) -> list[PathRecord] | None:
    h, w = shape
    if h < 8 or w < 8:
        return None

    for _ in range(CROSS_RETRIES):
        c2a = random.randint(3, w - 4)
        c1a = random.randint(1, c2a - 2)
        c1b = random.randint(1, c2a - 1)
        c2b = random.randint(c2a + 2, w - 2)

        ra0 = random.randint(1, h - 2)
        ra1_choices = [r for r in range(1, h - 1) if abs(r - ra0) >= 3]
        if not ra1_choices:
            continue
        ra1 = random.choice(ra1_choices)

        lo, hi = sorted((ra0, ra1))
        rb0_choices = [r for r in range(lo + 1, hi)]
        if not rb0_choices:
            continue
        rb0 = random.choice(rb0_choices)

        rb1_choices = [r for r in range(1, h - 1) if abs(r - rb0) >= 2]
        if not rb1_choices:
            continue
        rb1 = random.choice(rb1_choices)

        first = _make_path(color_a, (ra0, c1a), (ra1, c2a))
        second = _make_path(color_b, (rb0, c1b), (rb1, c2b))
        overlap = first["inner"] & second["inner"]  # type: ignore[operator]
        if len(overlap) != 1:
            continue
        if first["endpoints"] & second["full"]:  # type: ignore[operator]
            continue
        if second["endpoints"] & first["full"]:  # type: ignore[operator]
            continue
        if first["kind"] != "l" or second["kind"] != "l":
            continue
        return [first, second]

    return None


def _choose_shape(pair_count: int) -> tuple[int, int]:
    low = max(GRID_MIN, 8 + pair_count * 2)
    high = max(low, GRID_MAX)
    return random.randint(low, high), random.randint(low, high)


def _build_paths_for_mode(mode: str) -> tuple[tuple[int, int], list[PathRecord]] | None:
    pair_lo, pair_hi = PAIR_COUNT_BY_MODE[mode]
    pair_count = random.randint(pair_lo, pair_hi)
    shape = _choose_shape(pair_count)
    colors = random.sample(PAIR_COLORS, pair_count)
    paths: list[PathRecord] = []

    if mode == "cross":
        crossing = _build_crossing_pair(shape, colors[0], colors[1])
        if crossing is None:
            return None
        paths.extend(crossing)
        extra_colors = colors[2:]
        extra_kinds = [random.choice(["l", "h", "v"]) for _ in extra_colors]
    elif mode == "horizontal":
        extra_colors = colors
        extra_kinds = ["h", "l"] + [random.choice(["l", "v", "h"]) for _ in range(pair_count - 2)]
    elif mode == "vertical":
        extra_colors = colors
        extra_kinds = ["v", "l"] + [random.choice(["l", "h", "v"]) for _ in range(pair_count - 2)]
    elif mode == "mixed":
        extra_colors = colors
        extra_kinds = ["h", "v", "l"] + [random.choice(["l", "h", "v"]) for _ in range(pair_count - 3)]
    else:
        raise ValueError(f"unknown mode: {mode}")

    for color, kind in zip(extra_colors, extra_kinds, strict=False):
        cand = None
        for _ in range(PATH_RETRIES):
            cand = _sample_random_path(shape, color, kind)
            if cand is not None and _path_fits(paths, cand):
                break
            cand = None
        if cand is None:
            return None
        paths.append(cand)

    return shape, paths


def _validate_generated(inp: np.ndarray, paths: list[PathRecord], mode: str) -> bool:
    expected = _render_paths(inp, paths)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    kinds = {str(path["kind"]) for path in paths}
    crossing = _cross_cells(paths)
    if mode == "cross" and not ("l" in kinds and len(crossing) == 1):
        return False
    if mode == "horizontal" and not ("h" in kinds and "l" in kinds and not crossing):
        return False
    if mode == "vertical" and not ("v" in kinds and "l" in kinds and not crossing):
        return False
    if mode == "mixed" and not ({"h", "v", "l"} <= kinds and not crossing):
        return False

    if np.count_nonzero(actual) <= np.count_nonzero(inp):
        return False
    if np.count_nonzero(actual) > (actual.size * 2) // 3:
        return False

    return True


def generate_input() -> np.ndarray:
    mode = _next_mode()
    for _ in range(GEN_RETRIES):
        built = _build_paths_for_mode(mode)
        if built is None:
            continue
        shape, paths = built
        inp = _render_input(shape, paths)
        if _validate_generated(inp, paths, mode):
            return inp
    else:
        raise RuntimeError(f"failed to generate a valid {mode} instance")
