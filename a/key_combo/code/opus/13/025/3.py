from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

GEN_HEIGHT_RANGE = (10, 13)
GEN_BAND_COUNT_CHOICES = (4, 5)
GEN_BAND_WIDTH_CHOICES = (3, 4)
GEN_ATTEMPTS = 400
GEN_BAND_ATTEMPTS = 250


def _separator_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _band_slices(grid: np.ndarray) -> list[tuple[int, int]]:
    sep_cols = _separator_columns(grid)
    assert sep_cols, "expected at least one full-height separator column"
    assert all(b - a > 1 for a, b in zip(sep_cols, sep_cols[1:])), "separators must be isolated"

    starts = [0] + [c + 1 for c in sep_cols]
    ends = sep_cols + [grid.shape[1]]
    bands = [(int(c0), int(c1)) for c0, c1 in zip(starts, ends)]
    assert all(c0 < c1 for c0, c1 in bands), "each band must have positive width"

    widths = {c1 - c0 for c0, c1 in bands}
    assert len(widths) == 1, "all bands must have equal width"
    return bands


def _fill_band(band: np.ndarray) -> np.ndarray:
    out = band.copy()
    for r in range(band.shape[0]):
        cols = np.flatnonzero(band[r] == BLUE)
        for left, right in zip(cols, cols[1:]):
            if right - left > 1:
                span = out[r, left + 1 : right]
                assert np.all(span == BLACK), "fillable spans must be background"
                out[r, left + 1 : right] = RED
    return out


def _filled_band_area(band: np.ndarray) -> int:
    return int(np.count_nonzero(_fill_band(band) == RED))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each separated outline band, then reorder the bands by filled area.

    Input structure:
    - The grid has a black background and several equal-width vertical bands.
    - Full-height grey columns separate the bands.
    - Inside each band, blue diagonal outline pixels form a closed or near-closed shape.

    Rule:
    - In each band independently, scan every row from left to right.
    - Between each adjacent pair of blue outline pixels on that row, fill the black cells with red.
    - Count the total number of red filled cells per band.
    - Reorder the entire filled bands from left to right by ascending red count, while keeping the grey separators fixed in place.

    Key insight:
    - The sort key is not visible in the raw outline alone; it appears only after scanline filling reveals each band's enclosed area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bands = _band_slices(g)
    filled_bands = []
    areas = []
    for c0, c1 in bands:
        filled = _fill_band(g[:, c0:c1])
        filled_bands.append(filled)
        areas.append(int(np.count_nonzero(filled == RED)))

    assert len(set(areas)) == len(areas), "band areas must be unique for an unambiguous ordering"

    order = np.argsort(areas, kind="stable")
    out = g.copy()
    for dst_idx, src_idx in enumerate(order):
        c0, c1 = bands[dst_idx]
        out[:, c0:c1] = filled_bands[int(src_idx)]
    return out


@lru_cache(maxsize=None)
def _states(width: int) -> tuple[tuple[int, int], ...]:
    return tuple((left, right) for left in range(width) for right in range(left, width))


@lru_cache(maxsize=None)
def _transitions(width: int) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    trans: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for left, right in _states(width):
        nxt = []
        for nleft in range(max(0, left - 1), min(width - 1, left + 1) + 1):
            lo = max(nleft, right - 1)
            hi = min(width - 1, right + 1)
            for nright in range(lo, hi + 1):
                nxt.append((nleft, nright))
        trans[(left, right)] = tuple(nxt)
    return trans


def _path_between(
    start: tuple[int, int], end: tuple[int, int], steps: int, width: int
) -> list[tuple[int, int]] | None:
    cur = start
    path = [cur]
    for remaining in range(steps, 0, -1):
        candidates = [
            state
            for state in _transitions(width)[cur]
            if abs(end[0] - state[0]) <= remaining - 1 and abs(end[1] - state[1]) <= remaining - 1
        ]
        if not candidates:
            return None

        current_dist = abs(end[0] - cur[0]) + abs(end[1] - cur[1])
        weights = []
        for state in candidates:
            next_dist = abs(end[0] - state[0]) + abs(end[1] - state[1])
            progress = current_dist - next_dist
            weight = 1
            if progress > 0:
                weight += 3
            if state == cur:
                weight += 1
            if state[1] - state[0] >= 2:
                weight += 1
            weights.append(weight)
        cur = random.choices(candidates, weights=weights, k=1)[0]
        path.append(cur)
    return path if cur == end else None


def _positive_partition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    points = [0] + cuts + [total]
    return [points[i + 1] - points[i] for i in range(parts)]


def _build_profile(height: int, width: int) -> tuple[list[tuple[int, int]], str] | None:
    states = _states(width)
    narrow = [state for state in states if state[1] - state[0] <= 1]
    open_states = [state for state in states if state[1] - state[0] >= 2]
    full_states = [state for state in open_states if state[1] - state[0] == width - 1]
    if not open_states:
        return None

    family = random.choice(("single_peak", "sweep", "pinch"))
    steps = height - 1
    if steps < 3:
        return None
    seg_a, seg_b, seg_c = _positive_partition(steps, 3)

    if family == "single_peak":
        state0 = random.choice(narrow)
        state1 = random.choice(open_states)
        open_neighbors = [state for state in _transitions(width)[state1] if state[1] - state[0] >= 2]
        state2 = random.choice(open_neighbors)
        state3 = random.choice(narrow)
    elif family == "sweep":
        state0 = random.choice(narrow)
        state1 = random.choice(full_states or open_states)
        shifted = [
            state for state in _transitions(width)[state1] if state[1] - state[0] >= 2 and state != state1
        ]
        state2 = random.choice(shifted or open_states)
        state3 = random.choice(narrow)
    else:
        state0 = random.choice(open_states)
        state1 = random.choice(narrow)
        state2 = random.choice(open_states)
        state3 = random.choice(narrow)

    path1 = _path_between(state0, state1, seg_a, width)
    path2 = _path_between(state1, state2, seg_b, width)
    path3 = _path_between(state2, state3, seg_c, width)
    if path1 is None or path2 is None or path3 is None:
        return None

    profile = path1[:-1] + path2[:-1] + path3
    if len(profile) != height:
        return None

    fill_rows = sum(1 for left, right in profile if right - left >= 2)
    fill_area = sum(max(0, right - left - 1) for left, right in profile)
    min_area = 2 if width == 3 else 3
    if fill_rows < 2 or fill_area < min_area:
        return None
    return profile, family


def _render_band(profile: list[tuple[int, int]], width: int) -> np.ndarray:
    band = np.zeros((len(profile), width), dtype=int)
    for r, (left, right) in enumerate(profile):
        band[r, left] = BLUE
        band[r, right] = BLUE
    return band


def _generate_band(height: int, width: int) -> tuple[np.ndarray, int, str] | None:
    for _ in range(GEN_BAND_ATTEMPTS):
        built = _build_profile(height, width)
        if built is None:
            continue
        profile, family = built
        band = _render_band(profile, width)
        area = _filled_band_area(band)
        if area <= 0:
            continue
        return band, area, family
    return None


def _shuffle_non_sorted(items: list[tuple[np.ndarray, int, str]]) -> list[tuple[np.ndarray, int, str]]:
    shuffled = items[:]
    random.shuffle(shuffled)
    if [area for _, area, _ in shuffled] == sorted(area for _, area, _ in shuffled):
        i = random.randrange(len(shuffled) - 1)
        shuffled[i], shuffled[i + 1] = shuffled[i + 1], shuffled[i]
    return shuffled


def _compose_grid(height: int, band_width: int, bands: list[np.ndarray]) -> np.ndarray:
    band_count = len(bands)
    width = band_count * band_width + (band_count - 1)
    grid = np.zeros((height, width), dtype=int)
    for idx, band in enumerate(bands):
        c0 = idx * (band_width + 1)
        c1 = c0 + band_width
        grid[:, c0:c1] = band
        if idx < band_count - 1:
            grid[:, c1] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    band_count = random.choice(GEN_BAND_COUNT_CHOICES)
    band_width = random.choice(GEN_BAND_WIDTH_CHOICES)
    total_width = band_count * band_width + (band_count - 1)
    if max(height, total_width) > 30:
        return None

    by_area: dict[int, tuple[np.ndarray, int, str]] = {}
    families: set[str] = set()
    for _ in range(GEN_BAND_ATTEMPTS * band_count * 3):
        generated = _generate_band(height, band_width)
        if generated is None:
            continue
        band, area, family = generated
        if area in by_area:
            continue
        by_area[area] = (band, area, family)
        families.add(family)
        if len(by_area) == band_count:
            break
    if len(by_area) != band_count:
        return None

    bands = _shuffle_non_sorted(list(by_area.values()))
    if band_count >= 4 and len(families) == 1:
        return None

    grid = _compose_grid(height, band_width, [band for band, _area, _family in bands])
    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if len(set(_filled_band_area(grid[:, c0:c1]) for c0, c1 in _band_slices(grid))) != band_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
