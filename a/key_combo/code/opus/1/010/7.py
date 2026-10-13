from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from typing import NamedTuple
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
MAGENTA = 6
TEAL = 8

# --------------- solver ---------------


def _cell_ranges(length: int, wall_indices: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for wall in sorted(wall_indices):
        if start <= wall - 1:
            ranges.append((start, wall - 1))
        start = wall + 1
    if start <= length - 1:
        ranges.append((start, length - 1))
    return ranges


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int:
    matches = [i for i, (lo, hi) in enumerate(bands) if lo <= pos <= hi]
    assert len(matches) == 1, "marker must belong to exactly one lattice cell band"
    return matches[0]


def _unique_dominant_color(colors: list[int], counts: Counter[int]) -> int:
    max_count = max(counts[color] for color in colors)
    candidates = [color for color in colors if counts[color] == max_count]
    assert len(candidates) == 1, "expected a unique dominant non-lattice color"
    return candidates[0]


def _fill_cells_from_markers(
    marker_cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]
) -> frozenset[tuple[int, int]]:
    markers = tuple((int(r), int(c)) for r, c in marker_cells)
    assert len(markers) == len(set(markers)), "marker cells must be unique"

    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for row_idx, col_idx in markers:
        by_row[row_idx].append(col_idx)
        by_col[col_idx].append(row_idx)

    assert by_row and by_col, "each color must contain at least one marker"
    assert all(len(cols) == 2 for cols in by_row.values()), "each occupied lattice row must contain exactly two markers of a color"
    assert all(len(rows) == 2 for rows in by_col.values()), "each occupied lattice column must contain exactly two markers of a color"

    filled: set[tuple[int, int]] = set()
    for row_idx, cols in by_row.items():
        c0, c1 = sorted(cols)
        for col_idx in range(c0, c1 + 1):
            filled.add((row_idx, col_idx))
    for col_idx, rows in by_col.items():
        r0, r1 = sorted(rows)
        for row_idx in range(r0, r1 + 1):
            filled.add((row_idx, col_idx))
    return frozenset(filled)


def _summary_grid(red_count: int, green_count: int, magenta_count: int) -> np.ndarray:
    width = max(red_count, green_count, magenta_count)
    assert width > 0, "summary must have at least one column"
    out = np.zeros((3, width), dtype=int)
    out[0, :red_count] = RED
    out[1, :green_count] = GREEN
    out[2, :magenta_count] = MAGENTA
    return out


def _extract_markers_by_color(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[int, tuple[tuple[int, int], ...]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = g.shape

    colors = [int(color) for color in np.unique(g)]
    lattice_candidates = []
    for color in colors:
        wall_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        wall_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if wall_rows and wall_cols:
            lattice_candidates.append((color, wall_rows, wall_cols))
    assert len(lattice_candidates) == 1, "expected one lattice color spanning full rows and columns"

    lattice_color, wall_rows, wall_cols = lattice_candidates[0]
    assert lattice_color == TEAL, "expected the lattice to be teal"

    counts = Counter(map(int, g.ravel()))
    empty_color = _unique_dominant_color([color for color in colors if color != lattice_color], counts)
    assert empty_color == BLACK, "expected empty lattice cells to be black"

    marker_colors = {color for color in colors if color not in {lattice_color, empty_color}}
    assert marker_colors == {RED, GREEN}, "expected red and green marker colors"

    row_bands = _cell_ranges(height, wall_rows)
    col_bands = _cell_ranges(width, wall_cols)
    assert row_bands and col_bands, "lattice must contain at least one cell row and one cell column"

    markers_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r_raw, c_raw in np.argwhere((g != lattice_color) & (g != empty_color)):
        r = int(r_raw)
        c = int(c_raw)
        color = int(g[r, c])
        markers_by_color[color].append((_band_index(r, row_bands), _band_index(c, col_bands)))

    assert tuple(sorted(markers_by_color)) == (RED, GREEN), "both marker colors must appear"
    return row_bands, col_bands, {
        color: tuple(sorted(cells))
        for color, cells in markers_by_color.items()
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace red and green lattice-cell runs, then summarize the counts of pure red, pure green, and magenta overlap cells.

    Input structure:
    - A teal lattice of full-width rows and full-height columns partitions the canvas into regular rectangular cells.
    - Black cells are empty, and red (2) plus green (3) singleton marker pixels sit inside cell interiors.
    - For each color separately, every occupied lattice row and every occupied lattice column contains exactly two markers.

    Rule:
    - Convert each marker pixel to its lattice-cell coordinate.
    - For each color, fill every inclusive cell interval between the two same-color markers in any occupied lattice row, and also every inclusive cell interval in any occupied lattice column.
    - Any cell reached only by red counts as pure red, any cell reached only by green counts as pure green, and any cell reached by both becomes magenta.
    - Remove the original lattice scene and output a 3-row summary bar: row 0 has the red count in red pixels, row 1 has the green count in green pixels, and row 2 has the magenta count in magenta pixels, all left-aligned on black.

    Key insight:
    - The counted magenta category does not exist in the input; it emerges only after tracing the two colors on the coarser lattice-cell coordinate system and measuring the overlap there.
    """
    _row_bands, _col_bands, markers_by_color = _extract_markers_by_color(grid)

    red_fill = _fill_cells_from_markers(markers_by_color[RED])
    green_fill = _fill_cells_from_markers(markers_by_color[GREEN])
    overlap = red_fill & green_fill

    red_only = len(red_fill - overlap)
    green_only = len(green_fill - overlap)
    magenta = len(overlap)
    return _summary_grid(red_only, green_only, magenta)


# --------------- generator ---------------

MIN_FILL_CELLS = 6
MAX_FILL_CELLS = 16
MIN_PURE_CELLS = 2
MAX_INTERSECTION_CELLS = 6
GRID_CHOICES = ((4, 4), (4, 5), (5, 4), (5, 5))
CELL_HEIGHT_CHOICES = (2, 3, 4)
CELL_WIDTH_CHOICES = (2, 3, 4)
PROFILE_COUNT = 6
RANDOM_PAIR_ATTEMPTS = 600
MAX_GENERATION_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


class Candidate(NamedTuple):
    shape: str
    complexity: int
    markers: tuple[tuple[int, int], ...]
    marker_set: frozenset[tuple[int, int]]
    fill: frozenset[tuple[int, int]]


def _center_offset(size: int, rng: random.Random) -> int:
    if size % 2 == 1:
        return size // 2
    return rng.choice((size // 2 - 1, size // 2))


def _make_candidate(shape: str, complexity: int, markers: tuple[tuple[int, int], ...]) -> Candidate | None:
    marker_tuple = tuple(sorted((int(r), int(c)) for r, c in markers))
    if len(marker_tuple) != len(set(marker_tuple)):
        return None

    try:
        fill = _fill_cells_from_markers(marker_tuple)
    except AssertionError:
        return None

    if len(fill) < MIN_FILL_CELLS or len(fill) > MAX_FILL_CELLS:
        return None
    if len(fill) <= len(marker_tuple):
        return None

    return Candidate(shape, complexity, marker_tuple, frozenset(marker_tuple), fill)


@lru_cache(maxsize=None)
def _enumerate_candidates(n_rows: int, n_cols: int) -> tuple[Candidate, ...]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    candidates: list[Candidate] = []

    def add(shape: str, complexity: int, markers: tuple[tuple[int, int], ...]) -> None:
        key = tuple(sorted(markers))
        if key in seen:
            return
        candidate = _make_candidate(shape, complexity, markers)
        if candidate is None:
            return
        seen.add(key)
        candidates.append(candidate)

    for r0, r1 in combinations(range(n_rows), 2):
        for c0, c1 in combinations(range(n_cols), 2):
            if (r1 - r0) + (c1 - c0) < 3:
                continue
            add("rectangle", 0, ((r0, c0), (r0, c1), (r1, c0), (r1, c1)))

    for r0, r1, r2 in combinations(range(n_rows), 3):
        for c0, c1, c2 in combinations(range(n_cols), 3):
            add("hex_a", 1, ((r0, c0), (r0, c1), (r1, c1), (r1, c2), (r2, c2), (r2, c0)))
            add("hex_b", 1, ((r0, c1), (r0, c2), (r1, c0), (r1, c1), (r2, c0), (r2, c2)))
            add("hex_c", 1, ((r0, c0), (r0, c2), (r1, c0), (r1, c1), (r2, c1), (r2, c2)))

    if n_rows >= 4 and n_cols >= 4:
        for r0, r1, r2, r3 in combinations(range(n_rows), 4):
            for c0, c1, c2, c3 in combinations(range(n_cols), 4):
                add(
                    "oct_a",
                    2,
                    (
                        (r0, c0),
                        (r0, c1),
                        (r1, c1),
                        (r1, c2),
                        (r2, c2),
                        (r2, c3),
                        (r3, c3),
                        (r3, c0),
                    ),
                )
                add(
                    "oct_b",
                    2,
                    (
                        (r0, c2),
                        (r0, c3),
                        (r1, c1),
                        (r1, c2),
                        (r2, c0),
                        (r2, c1),
                        (r3, c0),
                        (r3, c3),
                    ),
                )

    assert candidates, "expected at least one candidate shape"
    candidates.sort(key=lambda cand: (cand.complexity, cand.shape, cand.markers))
    return tuple(candidates)


def _pair_stats(red: Candidate, green: Candidate) -> tuple[int, int, int, bool, int]:
    overlap = red.fill & green.fill
    magenta = len(overlap)
    red_only = len(red.fill) - magenta
    green_only = len(green.fill) - magenta
    marker_overlap = bool(overlap & (red.marker_set | green.marker_set))
    covered = len(red.fill | green.fill)
    return red_only, green_only, magenta, marker_overlap, covered


def _profile_constraints(profile: int) -> dict[str, object]:
    return (
        {"marker_overlap": True, "p_bucket": "small", "complexity": "at_least_one_complex"},
        {"marker_overlap": False, "p_bucket": "medium", "complexity": "at_least_one_complex"},
        {"marker_overlap": None, "p_bucket": "small", "complexity": "both_complex"},
        {"marker_overlap": True, "p_bucket": "any", "complexity": "exactly_one_rectangle"},
        {"marker_overlap": False, "p_bucket": "any", "complexity": "any"},
        {"marker_overlap": None, "p_bucket": "medium", "complexity": "any"},
    )[profile % PROFILE_COUNT]


def _complexity_ok(red: Candidate, green: Candidate, mode: str) -> bool:
    rectangle_flags = (red.shape == "rectangle", green.shape == "rectangle")
    if mode == "any":
        return True
    if mode == "at_least_one_complex":
        return red.complexity > 0 or green.complexity > 0
    if mode == "both_complex":
        return red.complexity > 0 and green.complexity > 0
    if mode == "exactly_one_rectangle":
        return rectangle_flags.count(True) == 1
    raise AssertionError(f"unknown complexity mode: {mode}")


def _bucket_ok(magenta: int, bucket: str) -> bool:
    if bucket == "any":
        return 1 <= magenta <= MAX_INTERSECTION_CELLS
    if bucket == "small":
        return 1 <= magenta <= 2
    if bucket == "medium":
        return 3 <= magenta <= MAX_INTERSECTION_CELLS
    raise AssertionError(f"unknown bucket: {bucket}")


def _pair_ok(
    red: Candidate,
    green: Candidate,
    n_rows: int,
    n_cols: int,
    profile: int,
) -> bool:
    if red.marker_set & green.marker_set:
        return False

    red_only, green_only, magenta, marker_overlap, covered = _pair_stats(red, green)
    if red_only < MIN_PURE_CELLS or green_only < MIN_PURE_CELLS:
        return False
    if not _bucket_ok(magenta, str(_profile_constraints(profile)["p_bucket"])):
        return False
    target_marker_overlap = _profile_constraints(profile)["marker_overlap"]
    if target_marker_overlap is not None and marker_overlap != target_marker_overlap:
        return False
    if not _complexity_ok(red, green, str(_profile_constraints(profile)["complexity"])):
        return False

    total_cells = n_rows * n_cols
    if covered > total_cells - 3:
        return False
    return True


def _pick_grid_shape(profile: int, rng: random.Random) -> tuple[int, int]:
    if profile % PROFILE_COUNT in {1, 2, 5}:
        return rng.choice(((4, 5), (5, 4), (5, 5)))
    return rng.choice(GRID_CHOICES)


def _find_candidate_pair(
    n_rows: int,
    n_cols: int,
    profile: int,
    rng: random.Random,
) -> tuple[Candidate, Candidate] | None:
    candidates = _enumerate_candidates(n_rows, n_cols)
    if len(candidates) < 2:
        return None

    for _ in range(RANDOM_PAIR_ATTEMPTS):
        red, green = rng.sample(candidates, 2)
        if _pair_ok(red, green, n_rows, n_cols, profile):
            return red, green

    valid_pairs: list[tuple[int, int]] = []
    for i in range(len(candidates)):
        red = candidates[i]
        for j in range(i + 1, len(candidates)):
            green = candidates[j]
            if _pair_ok(red, green, n_rows, n_cols, profile):
                valid_pairs.append((i, j))

    if not valid_pairs:
        return None

    i, j = rng.choice(valid_pairs)
    red = candidates[i]
    green = candidates[j]
    if rng.random() < 0.5:
        red, green = green, red
    return red, green


def _build_lattice_grid(
    n_rows: int,
    n_cols: int,
    cell_height: int,
    cell_width: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = n_rows * cell_height + (n_rows - 1)
    width = n_cols * cell_width + (n_cols - 1)
    assert height <= 30 and width <= 30, "input grid must stay within ARC bounds"

    grid = np.zeros((height, width), dtype=int)
    wall_rows = [cell_height + i * (cell_height + 1) for i in range(n_rows - 1)]
    wall_cols = [cell_width + i * (cell_width + 1) for i in range(n_cols - 1)]
    grid[wall_rows, :] = TEAL
    grid[:, wall_cols] = TEAL
    row_bands = _cell_ranges(height, wall_rows)
    col_bands = _cell_ranges(width, wall_cols)
    assert len(row_bands) == n_rows and len(col_bands) == n_cols
    return grid, row_bands, col_bands


def _place_markers(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    markers: tuple[tuple[int, int], ...],
    color: int,
    row_offset: int,
    col_offset: int,
) -> None:
    for row_idx, col_idx in markers:
        r0, _r1 = row_bands[row_idx]
        c0, _c1 = col_bands[col_idx]
        grid[r0 + row_offset, c0 + col_offset] = color


def _expected_summary_from_candidates(red: Candidate, green: Candidate) -> np.ndarray:
    overlap = red.fill & green.fill
    return _summary_grid(
        len(red.fill - overlap),
        len(green.fill - overlap),
        len(overlap),
    )


def _validate_instance(
    grid: np.ndarray,
    red: Candidate,
    green: Candidate,
    expected: np.ndarray,
) -> bool:
    overlap = red.fill & green.fill
    if not overlap:
        return False
    if len(red.fill - overlap) < MIN_PURE_CELLS or len(green.fill - overlap) < MIN_PURE_CELLS:
        return False

    try:
        _row_bands, _col_bands, markers_by_color = _extract_markers_by_color(grid)
    except AssertionError:
        return False

    if frozenset(markers_by_color[RED]) != red.marker_set:
        return False
    if frozenset(markers_by_color[GREEN]) != green.marker_set:
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    return True


def _gen_attempt(rng: random.Random, profile: int) -> np.ndarray | None:
    n_rows, n_cols = _pick_grid_shape(profile, rng)
    red, green = _find_candidate_pair(n_rows, n_cols, profile, rng) or (None, None)
    if red is None or green is None:
        return None

    cell_height = rng.choice(CELL_HEIGHT_CHOICES)
    cell_width = rng.choice(CELL_WIDTH_CHOICES)
    grid, row_bands, col_bands = _build_lattice_grid(n_rows, n_cols, cell_height, cell_width)

    row_offset = _center_offset(cell_height, rng)
    col_offset = _center_offset(cell_width, rng)
    _place_markers(grid, row_bands, col_bands, red.markers, RED, row_offset, col_offset)
    _place_markers(grid, row_bands, col_bands, green.markers, GREEN, row_offset, col_offset)

    expected = _expected_summary_from_candidates(red, green)
    if expected.shape[1] > 30:
        return None
    if not _validate_instance(grid, red, green, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    rng = random.Random()
    profile = _GEN_CALL_COUNT % PROFILE_COUNT
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng, profile)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
