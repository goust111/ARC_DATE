from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = (1, 2)
LENGTHS = (3, 4, 5, 6)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _top_runs(row: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    col = 0
    width = int(row.shape[0])
    while col < width:
        color = int(row[col])
        if color == bg:
            col += 1
            continue
        start = col
        while col + 1 < width and int(row[col + 1]) == color:
            col += 1
        runs.append((start, col, color))
        col += 1
    return runs


def _extract_segments(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert not np.any(g[1:] != bg), "all foreground must be on the top row"

    runs = _top_runs(g[0], bg)
    assert len(runs) == 2, "expected exactly two top-row segments"

    segments: list[dict[str, int]] = []
    colors = set()
    for left, right, color in runs:
        length = int(right - left + 1)
        assert 3 <= length <= 6, "segment length must be in [3, 6]"
        segments.append(
            {
                "left": int(left),
                "right": int(right),
                "color": int(color),
                "length": length,
                "radius": int((length - 1) // 2),
            }
        )
        colors.add(int(color))

    assert len(colors) == 2, "the two segments must have distinct colors"
    segments.sort(key=lambda seg: seg["left"])
    assert segments[0]["length"] != segments[1]["length"], "segment lengths must differ"
    gap = int(segments[1]["left"] - segments[0]["right"] - 1)
    assert gap >= 2, "segments must be separated by at least two background cells"
    return g, bg, segments


def _envelope_mask(shape: tuple[int, int], seg: dict[str, int]) -> np.ndarray:
    height, width = map(int, shape)
    left = int(seg["left"])
    right = int(seg["right"])
    radius = int(seg["radius"])

    mask = np.zeros((height, width), dtype=bool)
    for row in range(0, min(height, radius + 1)):
        c0 = max(0, left - row)
        c1 = min(width - 1, right + row)
        mask[row, c0 : c1 + 1] = True
    return mask


def _simulate_growth(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int]], dict[str, object]]:
    g, bg, segments = _extract_segments(grid)
    height, width = map(int, g.shape)
    stop_rows = [min(height // 2, width - 1 - seg["right"]) for seg in segments]
    envelopes = [_envelope_mask(g.shape, seg) for seg in segments]

    cells_by_segment: list[set[tuple[int, int]]] = [set(), set()]
    skipped = [0, 0]
    naive = [0, 0]
    below = [0, 0]

    for idx, seg in enumerate(segments):
        left = int(seg["left"])
        right = int(seg["right"])
        for col in range(left, right + 1):
            cells_by_segment[idx].add((0, col))
        naive[idx] += seg["length"]

        other_env = envelopes[1 - idx]
        for row in range(1, stop_rows[idx] + 1):
            end = min(right + row, width - 1)
            naive[idx] += end - left + 1
            for col in range(left, end + 1):
                if other_env[row, col]:
                    skipped[idx] += 1
                    continue
                cells_by_segment[idx].add((row, col))
                below[idx] += 1

    overlap = cells_by_segment[0] & cells_by_segment[1]

    out = np.full_like(g, bg)
    for idx in sorted(range(2), key=lambda k: segments[k]["left"]):
        color = int(segments[idx]["color"])
        for row, col in cells_by_segment[idx]:
            if int(out[row, col]) == bg:
                out[row, col] = color

    stats: dict[str, object] = {
        "segments": segments,
        "stop_rows": stop_rows,
        "skipped": skipped,
        "naive": naive,
        "below": below,
        "overlap": overlap,
        "envelopes": envelopes,
    }
    return out, segments, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the two top-row color runs downward while each run's diamond envelope trims the other's expansion.

    Input structure:
    - The background is black.
    - Exactly two separated horizontal monochrome segments appear on the top row, with different colors and different lengths.
    - Every other cell is background.

    Rule:
    - Keep both top-row source segments unchanged.
    - For each source segment, define an implicit diamond-like exclusion envelope with radius floor((length - 1) / 2). The visible lower half expands one cell on both sides at each lower row.
    - Independently grow each segment downward, extending only its right end by one cell per row, until row `floor(H / 2)` or until the segment reaches the right edge.
    - When a candidate growth cell falls inside the other segment's envelope, skip that cell; if both surviving growths ever target the same cell, keep the color of the segment whose top-row run starts further left.

    Key insight:
    - The output is not two independent right-growing wedges: each wedge is carved by the other segment's hidden shrinking diamond territory, so relative length and spacing both matter.
    """
    out, _segments, _stats = _simulate_growth(grid)
    return out


def _grid_from_params(
    height: int,
    width: int,
    left_start: int,
    left_length: int,
    gap: int,
    right_length: int,
    left_color: int,
    right_color: int,
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    right_start = left_start + left_length + gap
    grid[0, left_start : left_start + left_length] = int(left_color)
    grid[0, right_start : right_start + right_length] = int(right_color)
    return grid


def _sample_case(mode: str) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for gap in (2, 3):
        for left_length in LENGTHS:
            for right_length in LENGTHS:
                if left_length == right_length:
                    continue
                right_radius = (right_length - 1) // 2
                left_reaches_right_envelope = any(
                    2 * row >= gap + 1 for row in range(1, right_radius + 1)
                )
                if not left_reaches_right_envelope:
                    continue
                if mode == "short_left" and left_length < right_length:
                    candidates.append((left_length, right_length, gap, 0))
                elif mode == "long_left" and left_length > right_length:
                    candidates.append((left_length, right_length, gap, 0))
                elif mode == "boundary":
                    candidates.append((left_length, right_length, gap, 1))
    if not candidates:
        raise RuntimeError(f"no candidates for mode {mode}")
    return random.choice(candidates)


def _valid_instance(grid: np.ndarray) -> bool:
    try:
        out, segments, stats = _simulate_growth(grid)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False
    if stats["skipped"] == [0, 0]:
        return False
    if any(count <= 0 for count in stats["below"]):
        return False
    if len(stats["overlap"]) > 4:
        return False

    stop_rows = stats["stop_rows"]
    if any(row <= 0 for row in stop_rows):
        return False

    height, width = map(int, grid.shape)
    if height > 30 or width > 30:
        return False

    # Reject trivial cases where one envelope has no visible effect on the rival growth.
    naive = stats["naive"]
    skipped = stats["skipped"]
    visible_pixels = [int(np.count_nonzero(out[1:] == seg["color"])) for seg in segments]
    for idx in range(2):
        if visible_pixels[idx] <= 0:
            return False
        if skipped[idx] >= naive[idx] - segments[idx]["length"]:
            return False

    return True


def generate_input() -> np.ndarray:
    modes = ("short_left", "long_left", "boundary")
    for _ in range(MAX_GENERATION_ATTEMPTS):
        mode = random.choice(modes)
        left_length, right_length, gap, boundary_mode = _sample_case(mode)

        left_start = random.randint(1, 4)
        right_start = left_start + left_length + gap
        min_width = right_start + right_length + 1

        if mode == "boundary":
            height = random.randint(12, 18)
        else:
            height = random.randint(11, 18)

        right_radius = (right_length - 1) // 2
        if boundary_mode:
            tail = random.randint(max(2, right_radius), max(3, right_radius + 1))
        else:
            min_tail = max(height // 2 + 2, right_radius)
            max_tail = min(11, 31 - min_width)
            if min_tail > max_tail:
                continue
            tail = random.randint(min_tail, max_tail)
        width = min_width + tail - 1

        if width > 30:
            continue

        colors = list(COLORS)
        random.shuffle(colors)
        grid = _grid_from_params(
            height=height,
            width=width,
            left_start=left_start,
            left_length=left_length,
            gap=gap,
            right_length=right_length,
            left_color=colors[0],
            right_color=colors[1],
        )

        if _valid_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid input")
