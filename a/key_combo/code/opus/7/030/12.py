from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
MAROON = 9


def _terrain_tops(grid: np.ndarray) -> np.ndarray:
    """Return the row index of the top terrain pixel in each column."""
    h, w = grid.shape
    tops = np.empty(w, dtype=int)
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] == MAROON)
        assert rows.size > 0, "each column must contain terrain"
        top = int(rows[0])
        assert np.array_equal(rows, np.arange(top, h)), "terrain must be a contiguous suffix in every column"
        tops[c] = top
    return tops


def _fill_counts(tops: np.ndarray, rlo: int, rhi: int, clo: int, chi: int) -> tuple[int, int]:
    fill = 0
    total = (rhi - rlo - 1) * (chi - clo - 1)
    if total <= 0:
        return 0, 0
    for c in range(clo + 1, chi):
        fill += max(0, min(rhi, int(tops[c])) - (rlo + 1))
    return fill, total - fill


def _apply_rule(grid: np.ndarray, tops: np.ndarray, rows: tuple[int, int], cols: tuple[int, int]) -> np.ndarray:
    rlo, rhi = rows
    clo, chi = cols

    out = grid.copy()
    for c in range(clo + 1, chi):
        fill_stop = min(rhi, int(tops[c]))
        if fill_stop > rlo + 1:
            out[rlo + 1 : fill_stop, c] = BLUE

    out[[rlo, rhi], :] = RED
    out[:, [clo, chi]] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw divider lines from two diagonal markers and fill only the part of the marked rectangle that lies above the terrain contour.

    Input structure:
    - The grid has a black background with a maroon terrain suffix in every column.
    - Exactly two red single-pixel markers lie in empty air, on different rows and different columns.
    - The two markers are opposite corners of a rectangle, while blue is absent from the input.

    Rule:
    - Use the marker rows as full-width red divider lines and the marker columns as full-height red divider lines.
    - Consider the strict interior of the rectangle between those two rows and columns.
    - Fill an interior cell blue iff it is above the top maroon terrain cell in its column; keep terrain cells and non-filled air unchanged, then let the red divider lines override everything.

    Key insight:
    - The red markers define a rectangle, but the blue region is that rectangle clipped by a per-column terrain skyline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == BLUE), "blue must be absent from the input"
    assert set(map(int, np.unique(g))).issubset({BLACK, RED, MAROON}), "unexpected input colors"

    tops = _terrain_tops(g)

    pts = np.argwhere(g == RED)
    assert pts.shape == (2, 2), "expected exactly two red marker pixels"

    (r0, c0), (r1, c1) = pts
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert r0 != r1 and c0 != c1, "markers must be diagonal corners"
    assert r0 < int(tops[c0]) and r1 < int(tops[c1]), "markers must lie above the terrain"

    rlo, rhi = sorted((r0, r1))
    clo, chi = sorted((c0, c1))
    assert rhi - rlo >= 2 and chi - clo >= 2, "rectangle must have a strict interior"

    return _apply_rule(g, tops, (rlo, rhi), (clo, chi))


# --------------- generator (gpt-5.4 high) ---------------

HEIGHT_RANGE = (14, 22)
WIDTH_RANGE = (14, 22)
MIN_TERRAIN_HEIGHT = 2
MAX_TERRAIN_HEIGHT = 10
MIN_AIR_MARGIN = 5
MIN_PROFILE_SPAN = 3
MAX_PROFILE_STEP = 2
MIN_ROW_GAP = 4
MIN_COL_GAP = 5
MIN_VISIBLE_FILL = 4
MIN_VISIBLE_BLOCKED = 4
GENERATION_RETRIES = 300


def _gen_profile(height: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    max_height = min(MAX_TERRAIN_HEIGHT, height - MIN_AIR_MARGIN)
    if max_height < MIN_TERRAIN_HEIGHT + MIN_PROFILE_SPAN:
        return None

    for _ in range(80):
        cur = int(rng.integers(MIN_TERRAIN_HEIGHT, max_height + 1))
        heights = [cur]
        for _ in range(1, width):
            step = int(rng.choice([-2, -1, 0, 1, 2], p=[0.10, 0.22, 0.36, 0.22, 0.10]))
            cur = int(np.clip(cur + step, MIN_TERRAIN_HEIGHT, max_height))
            heights.append(cur)

        profile = np.array(heights, dtype=int)
        if int(profile.max() - profile.min()) < MIN_PROFILE_SPAN:
            continue
        if len(set(map(int, profile))) < 4:
            continue
        return profile
    return None


def _candidate_rectangles(tops: np.ndarray, height: int, width: int) -> list[tuple[int, int, int, int, int]]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for clo in range(1, width - MIN_COL_GAP):
        for chi in range(clo + MIN_COL_GAP, width - 1):
            interior_tops = tops[clo + 1 : chi]
            if interior_tops.size == 0 or len(set(map(int, interior_tops))) < 2:
                continue

            for rlo in range(1, height - MIN_ROW_GAP):
                for rhi in range(rlo + MIN_ROW_GAP, height - 1):
                    crossing = (interior_tops > rlo + 1) & (interior_tops < rhi)
                    if not np.any(crossing):
                        continue

                    fill, blocked = _fill_counts(tops, rlo, rhi, clo, chi)
                    if fill < MIN_VISIBLE_FILL or blocked < MIN_VISIBLE_BLOCKED:
                        continue

                    if int(tops[clo]) > rlo and int(tops[chi]) > rhi:
                        candidates.append((rlo, rhi, clo, chi, 0))
                    if int(tops[clo]) > rhi and int(tops[chi]) > rlo:
                        candidates.append((rlo, rhi, clo, chi, 1))
    return candidates


def _build_input(
    height: int,
    width: int,
    terrain_heights: np.ndarray,
    rect: tuple[int, int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((height, width), dtype=int)
    for c, terrain_height in enumerate(terrain_heights):
        grid[height - int(terrain_height) :, c] = MAROON

    rlo, rhi, clo, chi, orientation = rect
    if orientation == 0:
        markers = ((rlo, clo), (rhi, chi))
    else:
        markers = ((rlo, chi), (rhi, clo))
    for r, c in markers:
        grid[r, c] = RED

    tops = height - terrain_heights
    expected = _apply_rule(grid, tops, (rlo, rhi), (clo, chi))
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(*HEIGHT_RANGE))
    width = int(rng.integers(*WIDTH_RANGE))

    terrain_heights = _gen_profile(height, width, rng)
    if terrain_heights is None:
        return None
    tops = height - terrain_heights

    candidates = _candidate_rectangles(tops, height, width)
    if not candidates:
        return None

    weights = np.array(
        [min(_fill_counts(tops, rlo, rhi, clo, chi)) for rlo, rhi, clo, chi, _ in candidates],
        dtype=float,
    )
    weights /= weights.sum()
    rect = candidates[int(rng.choice(len(candidates), p=weights))]

    grid, expected = _build_input(height, width, terrain_heights, rect)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 12")
