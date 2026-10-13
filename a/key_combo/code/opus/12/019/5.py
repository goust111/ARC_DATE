from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
VOTE_COLOR = 7
MARKER_COLOR_POOL = (1, 2, 3, 4, 5, 6, 8, 9)

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (12, 20)
MARKER_COUNT_RANGE = (3, 5)
VOTE_TOTAL_RANGE = (8, 15)
MIN_MARKER_GAP = 3
MIN_BAND_HEIGHT = 3
MAX_BAND_HEIGHT = 10
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 200


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _marker_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != BACKGROUND) & (grid != VOTE_COLOR))
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    assert markers, "expected at least one non-vote marker"
    rows = [r for r, _c, _color in markers]
    assert len(rows) == len(set(rows)), "marker rows must be unique"
    markers.sort(key=lambda item: item[0])
    return markers


def _band_ranges(marker_rows: list[int], height: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for i, row in enumerate(marker_rows):
        end = height - 1 if i == len(marker_rows) - 1 else (row + marker_rows[i + 1]) // 2
        assert start <= row <= end, "marker must lie inside its band"
        bands.append((start, end))
        start = end + 1
    assert start == height
    return bands


def _vote_counts_by_band(grid: np.ndarray, bands: list[tuple[int, int]]) -> list[int]:
    return [int(np.count_nonzero(grid[start : end + 1] == VOTE_COLOR)) for start, end in bands]


def _winner_index(vote_counts: list[int]) -> int:
    best = max(vote_counts)
    assert vote_counts.count(best) == 1, "expected a unique winning band"
    return vote_counts.index(best)


def _render_output(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    bands: list[tuple[int, int]],
    winner: int,
) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), BACKGROUND, dtype=int)

    for i, (row, _col, color) in enumerate(markers):
        if i != winner:
            out[row, :] = color

    win_row, _win_col, win_color = markers[winner]
    win_start, win_end = bands[winner]
    out[win_row, :] = win_color
    out[win_start, :] = win_color
    out[win_end, :] = win_color
    out[win_start : win_end + 1, 0] = win_color
    out[win_start : win_end + 1, width - 1] = win_color
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    markers = _marker_cells(grid)
    bands = _band_ranges([row for row, _c, _color in markers], int(grid.shape[0]))
    vote_counts = _vote_counts_by_band(grid, bands)
    winner = _winner_index(vote_counts)
    return _render_output(tuple(map(int, grid.shape)), markers, bands, winner)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use orange vote counts inside marker-defined horizontal bands to decide which band gets upgraded to a full frame.

    Input structure:
    - A black background contains 3-5 isolated non-orange marker pixels, each on a distinct row.
    - Orange vote pixels are scattered as single cells across the grid and disappear in the output.
    - The marker rows divide the grid into horizontal bands by midpoint boundaries between consecutive markers.

    Rule:
    - Sort the non-background, non-orange markers by row and split the rows at floor midpoints between consecutive marker rows.
    - Count the orange vote pixels inside each band; the unique highest-count band is the winner.
    - Draw a full-width horizontal line on every marker row, but only the winning band also gets its top row, bottom row, and both side columns painted in its marker color.

    Key insight:
    - The orange pixels are invisible votes: they are erased from the output, but their band-wise counts decide which marker's simple row line is promoted into a complete band frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid exceeds ARC size limits"

    markers = _marker_cells(g)
    bands = _band_ranges([row for row, _c, _color in markers], int(g.shape[0]))
    vote_counts = _vote_counts_by_band(g, bands)
    winner = _winner_index(vote_counts)
    return _render_output(tuple(map(int, g.shape)), markers, bands, winner)


def _distinct_positive_sums(
    n_parts: int,
    total: int,
    start: int = 1,
) -> list[tuple[int, ...]]:
    if n_parts == 0:
        return [()] if total == 0 else []

    results: list[tuple[int, ...]] = []
    min_rest = sum(range(start + 1, start + n_parts))
    for value in range(start, total + 1):
        remaining = total - value
        if remaining < min_rest:
            break
        for tail in _distinct_positive_sums(n_parts - 1, remaining, value + 1):
            results.append((value,) + tail)
    return results


_COUNT_PATTERNS = {
    n_markers: [
        counts
        for total in range(VOTE_TOTAL_RANGE[0], VOTE_TOTAL_RANGE[1] + 1)
        for counts in _distinct_positive_sums(n_markers, total)
    ]
    for n_markers in range(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1] + 1)
}


def _sample_vote_counts(n_markers: int) -> list[int]:
    counts = list(random.choice(_COUNT_PATTERNS[n_markers]))
    random.shuffle(counts)
    return counts


def _sample_marker_rows(height: int, n_markers: int) -> list[int] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        rows = sorted(random.sample(range(1, height - 1), n_markers))
        if any(rows[i + 1] - rows[i] < MIN_MARKER_GAP for i in range(n_markers - 1)):
            continue
        bands = _band_ranges(rows, height)
        band_heights = [end - start + 1 for start, end in bands]
        if all(MIN_BAND_HEIGHT <= band_h <= MAX_BAND_HEIGHT for band_h in band_heights):
            return rows
    return None


def _valid_vote_cell(
    cell: tuple[int, int],
    markers: Iterable[tuple[int, int]],
    votes: Iterable[tuple[int, int]],
) -> bool:
    return all(_chebyshev(cell, marker) > 1 for marker in markers) and all(
        _chebyshev(cell, vote) > 1 for vote in votes
    )


def _pick_band_votes(
    band: tuple[int, int],
    width: int,
    count: int,
    marker_points: list[tuple[int, int]],
    taken_votes: list[tuple[int, int]],
    preferred_row: int | None,
) -> list[tuple[int, int]] | None:
    start, end = band
    band_cells = [(r, c) for r in range(start, end + 1) for c in range(width)]
    band_cells = [cell for cell in band_cells if _valid_vote_cell(cell, marker_points, ())]
    if len(band_cells) < count:
        return None

    preferred_cells = [cell for cell in band_cells if preferred_row is not None and cell[0] == preferred_row]
    for _ in range(PLACEMENT_ATTEMPTS):
        chosen: list[tuple[int, int]] = []
        occupied = list(taken_votes)

        if preferred_row is not None:
            random.shuffle(preferred_cells)
            forced = next((cell for cell in preferred_cells if _valid_vote_cell(cell, marker_points, occupied)), None)
            if forced is None:
                continue
            chosen.append(forced)
            occupied.append(forced)

        candidates = band_cells[:]
        random.shuffle(candidates)
        for cell in candidates:
            if cell in chosen:
                continue
            if _valid_vote_cell(cell, marker_points, occupied):
                chosen.append(cell)
                occupied.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _place_votes(
    shape: tuple[int, int],
    bands: list[tuple[int, int]],
    vote_counts: list[int],
    marker_points: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    height, width = map(int, shape)
    assert height > 0 and width > 0

    preferred_rows: dict[int, int] = {}
    boundary_mode = random.choice(("none", "midpoint", "edge"))
    if boundary_mode == "midpoint":
        band_index = random.randrange(len(bands) - 1)
        preferred_rows[band_index] = bands[band_index][1]
    elif boundary_mode == "edge":
        if random.choice([False, True]):
            preferred_rows[0] = 0
        else:
            preferred_rows[len(bands) - 1] = height - 1

    order = sorted(range(len(bands)), key=lambda idx: (vote_counts[idx], idx in preferred_rows), reverse=True)
    placed: list[tuple[int, int]] = []
    for idx in order:
        cells = _pick_band_votes(
            bands[idx],
            width,
            vote_counts[idx],
            marker_points,
            placed,
            preferred_rows.get(idx),
        )
        if cells is None:
            return None
        placed.extend(cells)
    return placed


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    markers = _marker_cells(grid)
    if not (MARKER_COUNT_RANGE[0] <= len(markers) <= MARKER_COUNT_RANGE[1]):
        return False
    if len({color for _r, _c, color in markers}) != len(markers):
        return False

    marker_points = [(r, c) for r, c, _color in markers]
    if any(_chebyshev(a, b) <= 1 for i, a in enumerate(marker_points) for b in marker_points[i + 1 :]):
        return False

    vote_points = [tuple(map(int, rc)) for rc in np.argwhere(grid == VOTE_COLOR)]
    if not (VOTE_TOTAL_RANGE[0] <= len(vote_points) <= VOTE_TOTAL_RANGE[1]):
        return False
    if any(_chebyshev(vote, marker) <= 1 for vote in vote_points for marker in marker_points):
        return False
    if any(_chebyshev(a, b) <= 1 for i, a in enumerate(vote_points) for b in vote_points[i + 1 :]):
        return False

    bands = _band_ranges([row for row, _c, _color in markers], int(grid.shape[0]))
    band_heights = [end - start + 1 for start, end in bands]
    if not all(MIN_BAND_HEIGHT <= band_h <= MAX_BAND_HEIGHT for band_h in band_heights):
        return False

    vote_counts = _vote_counts_by_band(grid, bands)
    if any(count <= 0 for count in vote_counts):
        return False
    if len(set(vote_counts)) != len(vote_counts):
        return False

    winner = _winner_index(vote_counts)
    expected = _render_output(tuple(map(int, grid.shape)), markers, bands, winner)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    n_markers = random.randint(*MARKER_COUNT_RANGE)

    rows = _sample_marker_rows(height, n_markers)
    if rows is None:
        return None

    cols = random.sample(range(1, width - 1), n_markers)
    colors = random.sample(MARKER_COLOR_POOL, n_markers)
    markers = [(row, col, color) for row, col, color in zip(rows, cols, colors)]
    marker_points = [(row, col) for row, col, _color in markers]
    bands = _band_ranges(rows, height)

    vote_counts = _sample_vote_counts(n_markers)
    votes = _place_votes((height, width), bands, vote_counts, marker_points)
    if votes is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for row, col, color in markers:
        grid[row, col] = color
    for row, col in votes:
        grid[row, col] = VOTE_COLOR

    return grid if _validate_generated_grid(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
