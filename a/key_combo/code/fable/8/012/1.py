import random

import numpy as np


BACKGROUND = 0
TERRAIN = 5
FALLING_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_RETRY_LIMIT = 400
GEN_HEIGHT_RANGE = (13, 20)
GEN_PERIOD_RANGE = (2, 6)
GEN_REPETITIONS_RANGE = (3, 5)
GEN_PIXEL_COUNT_RANGE = (2, 5)
GEN_SEGMENT_COUNT_CHOICES = (1, 2, 2, 3)


def _extract_visible_heights(grid: np.ndarray) -> list[int | None]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    visible_heights: list[int | None] = []
    for c in range(width):
        terrain_rows = np.flatnonzero(g[:, c] == TERRAIN)
        if terrain_rows.size == 0:
            visible_heights.append(None)
            continue

        first = int(terrain_rows[0])
        expected = np.arange(first, height)
        assert np.array_equal(terrain_rows, expected), "visible terrain must be a solid column rising from the bottom"
        visible_heights.append(int(terrain_rows.size))
    return visible_heights


def _smallest_consistent_period(visible_heights: list[int | None]) -> int:
    width = len(visible_heights)
    for period in range(1, width + 1):
        residue_height: dict[int, int] = {}
        seen = [False] * period
        ok = True
        for c, value in enumerate(visible_heights):
            if value is None:
                continue
            residue = c % period
            seen[residue] = True
            previous = residue_height.get(residue)
            if previous is None:
                residue_height[residue] = int(value)
            elif previous != int(value):
                ok = False
                break
        if ok and all(seen):
            return period
    raise AssertionError("failed to infer a unique minimal period from the visible columns")


def _reconstruct_heights(visible_heights: list[int | None], period: int) -> list[int]:
    residue_height: dict[int, int] = {}
    for c, value in enumerate(visible_heights):
        if value is not None:
            residue_height[c % period] = int(value)
    assert len(residue_height) == period
    return [residue_height[c % period] for c in range(len(visible_heights))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each colored top pixel onto the visible or inferred skyline terrain beneath its column.

    Input structure:
    - The background is black.
    - Grey terrain columns rise contiguously from the bottom edge, and their heights repeat with one horizontal period.
    - Some contiguous column segments have their entire terrain erased back to black.
    - Additional non-grey foreground cells are isolated colored single pixels placed near the top in distinct columns.

    Rule:
    - Read the visible grey column heights and infer the unique smallest horizontal period consistent with them.
    - Reconstruct the hidden terrain height in every erased column from that period, without repainting the erased terrain itself.
    - Move every colored single pixel straight down within its own column so it lands exactly one cell above that column's visible or inferred terrain top.

    Key insight:
    - Even when a terrain column is fully blacked out, its height is still recoverable from the repeating skyline, so falling pixels can rest on an invisible floor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    visible_heights = _extract_visible_heights(g)
    period = _smallest_consistent_period(visible_heights)
    full_heights = _reconstruct_heights(visible_heights, period)

    falling_coords = np.argwhere((g != BACKGROUND) & (g != TERRAIN))
    falling_cols = [int(c) for _r, c in falling_coords]
    assert len(falling_cols) == len(set(falling_cols)), "falling pixels must occupy distinct columns"

    out = g.copy()
    for r, c in falling_coords:
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        out[rr, cc] = BACKGROUND

        landing_row = g.shape[0] - full_heights[cc] - 1
        assert landing_row >= 0, "terrain must leave room for the landing cell"
        assert out[landing_row, cc] == BACKGROUND, "landing cell must be empty"
        out[landing_row, cc] = color

    return out


def _sample_periodic_heights(grid_height: int, period: int) -> list[int]:
    max_height = min(9, grid_height - 4)
    choices = list(range(2, max_height + 1))
    assert len(choices) >= period
    heights = random.sample(choices, period)
    random.shuffle(heights)
    return heights


def _primitive_period(values: list[int]) -> int:
    width = len(values)
    for period in range(1, width + 1):
        if all(values[i] == values[i % period] for i in range(width)):
            return period
    return width


def _sample_width(period: int) -> int:
    min_reps, max_reps = GEN_REPETITIONS_RANGE
    max_reps = min(max_reps, 30 // period)
    reps = random.randint(min_reps, max_reps)
    extra = random.randint(0, min(period - 1, 30 - reps * period))
    return reps * period + extra


def _sample_hidden_mask(width: int, period: int) -> np.ndarray | None:
    max_segment_len = max(1, min(period + 2, width // 3))
    for _ in range(200):
        hidden = np.zeros(width, dtype=bool)
        segments: list[tuple[int, int]] = []
        segment_count = random.choice([n for n in GEN_SEGMENT_COUNT_CHOICES if n <= max(1, width // 2)])
        ok = True
        for _ in range(segment_count):
            seg_len = random.randint(1, max_segment_len)
            start = random.randint(0, width - seg_len)
            end = start + seg_len - 1
            if any(not (end < s - 1 or start > e + 1) for s, e in segments):
                ok = False
                break
            segments.append((start, end))
        if not ok:
            continue

        for start, end in segments:
            hidden[start : end + 1] = True
        if not np.any(hidden):
            continue
        if np.count_nonzero(hidden) > width // 2:
            continue
        if any(all(hidden[c] for c in range(residue, width, period)) for residue in range(period)):
            continue
        return hidden
    return None


def _pick_pixel_columns(hidden: np.ndarray) -> list[int]:
    width = hidden.size
    pixel_count = random.randint(GEN_PIXEL_COUNT_RANGE[0], min(GEN_PIXEL_COUNT_RANGE[1], width))
    hidden_cols = [c for c in range(width) if hidden[c]]
    visible_cols = [c for c in range(width) if not hidden[c]]

    chosen: list[int] = []
    if hidden_cols:
        chosen.append(random.choice(hidden_cols))
    if visible_cols and len(chosen) < pixel_count:
        visible_choice = random.choice(visible_cols)
        if visible_choice not in chosen:
            chosen.append(visible_choice)

    remaining = [c for c in range(width) if c not in chosen]
    chosen.extend(random.sample(remaining, pixel_count - len(chosen)))
    chosen.sort()
    return chosen


def _build_pair(
    grid_height: int,
    width: int,
    full_heights: list[int],
    hidden: np.ndarray,
    pixel_columns: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros((grid_height, width), dtype=int)
    out = np.zeros((grid_height, width), dtype=int)

    for c, terrain_height in enumerate(full_heights):
        if hidden[c]:
            continue
        inp[grid_height - terrain_height :, c] = TERRAIN
        out[grid_height - terrain_height :, c] = TERRAIN

    colors = random.sample(FALLING_COLORS, len(pixel_columns))
    for c, color in zip(pixel_columns, colors):
        landing_row = grid_height - full_heights[c] - 1
        if landing_row <= 0:
            return None
        start_row = random.randint(0, min(3, landing_row - 1))
        inp[start_row, c] = color
        out[landing_row, c] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    grid_height = random.randint(*GEN_HEIGHT_RANGE)
    period = random.randint(*GEN_PERIOD_RANGE)
    base_pattern = _sample_periodic_heights(grid_height, period)
    if _primitive_period(base_pattern) != period:
        return None

    width = _sample_width(period)
    full_heights = [base_pattern[c % period] for c in range(width)]

    hidden = _sample_hidden_mask(width, period)
    if hidden is None:
        return None

    visible_heights = [None if hidden[c] else full_heights[c] for c in range(width)]
    if _smallest_consistent_period(visible_heights) != period:
        return None

    pixel_columns = _pick_pixel_columns(hidden)
    pair = _build_pair(grid_height, width, full_heights, hidden, pixel_columns)
    if pair is None:
        return None
    inp, out = pair

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for the invisible-floor gravity task")
