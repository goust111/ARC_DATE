import random
from typing import Optional

import numpy as np


_GEN_CALL_COUNT = 0
_GEN_ATTEMPTS = 400
_BG = 0
_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    segments = []

    for c in range(g.shape[1]):
        rows = np.flatnonzero(g[:, c] != bg)
        if rows.size == 0:
            continue

        colors = np.unique(g[rows, c])
        assert colors.size == 1, "each segment must be monochrome"
        assert rows[-1] - rows[0] + 1 == rows.size, "each segment must be contiguous"

        segments.append(
            {
                "col": int(c),
                "top": int(rows[0]),
                "bottom": int(rows[-1]),
                "height": int(rows.size),
                "color": int(colors[0]),
            }
        )

    assert 2 <= len(segments) <= 3, "expected 2 or 3 vertical segments"
    assert len({seg["color"] for seg in segments}) == len(segments), "segment colors must be distinct"
    assert len({seg["height"] for seg in segments}) == len(segments), "segment heights must be distinct"
    assert len({seg["bottom"] for seg in segments}) == 1, "all segments must share one bottom row"

    occupied = np.argwhere(g != bg)
    assert occupied.shape[0] == sum(seg["height"] for seg in segments), "foreground must be exactly the segments"

    for seg in segments:
        if seg["col"] > 0:
            assert not np.any(g[seg["top"] : seg["bottom"] + 1, seg["col"] - 1] != bg), "segments must be width 1"
        if seg["col"] + 1 < g.shape[1]:
            assert not np.any(g[seg["top"] : seg["bottom"] + 1, seg["col"] + 1] != bg), "segments must be width 1"

    segments.sort(key=lambda seg: (seg["col"], seg["height"]))
    return segments


def _pyramid_value_at(seg: dict, r: int, c: int) -> Optional[int]:
    k = seg["bottom"] - int(r)
    if k < 0 or k >= seg["height"]:
        return None
    radius = seg["height"] - 1 - k
    if abs(int(c) - seg["col"]) > radius:
        return None
    return int(seg["color"]) if abs(int(c) - seg["col"]) % 2 == 0 else _BG


def _render_true_output(shape: tuple[int, int], segments: list[dict]) -> np.ndarray:
    out = np.full(shape, _BG, dtype=int)
    for seg in sorted(segments, key=lambda seg: seg["height"]):
        for k in range(seg["height"]):
            r = seg["bottom"] - k
            radius = seg["height"] - 1 - k
            c0 = seg["col"] - radius
            c1 = seg["col"] + radius
            for c in range(c0, c1 + 1):
                if 0 <= r < shape[0] and 0 <= c < shape[1]:
                    out[r, c] = int(seg["color"]) if abs(c - seg["col"]) % 2 == 0 else _BG
    return out


def _render_ignore_black(shape: tuple[int, int], segments: list[dict]) -> np.ndarray:
    out = np.full(shape, _BG, dtype=int)
    for seg in sorted(segments, key=lambda seg: seg["height"]):
        for k in range(seg["height"]):
            r = seg["bottom"] - k
            radius = seg["height"] - 1 - k
            c0 = seg["col"] - radius
            c1 = seg["col"] + radius
            for c in range(c0, c1 + 1):
                if 0 <= r < shape[0] and 0 <= c < shape[1] and abs(c - seg["col"]) % 2 == 0:
                    out[r, c] = int(seg["color"])
    return out


def _render_reverse_priority(shape: tuple[int, int], segments: list[dict]) -> np.ndarray:
    out = np.full(shape, _BG, dtype=int)
    for seg in sorted(segments, key=lambda seg: seg["height"], reverse=True):
        for k in range(seg["height"]):
            r = seg["bottom"] - k
            radius = seg["height"] - 1 - k
            c0 = seg["col"] - radius
            c1 = seg["col"] + radius
            for c in range(c0, c1 + 1):
                if 0 <= r < shape[0] and 0 <= c < shape[1]:
                    out[r, c] = int(seg["color"]) if abs(c - seg["col"]) % 2 == 0 else _BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each vertical segment into an alternating-color pyramid and overlay the pyramids by original height priority.

    Input structure:
    - The grid has a black background and 2-3 non-black vertical line segments.
    - Each segment occupies exactly one column, uses a single color, and all segment bottoms lie on the same row.
    - Segment colors are distinct and segment heights are distinct.

    Rule:
    - For each segment of height h, build a pyramid with the same center column and bottom row.
    - At upward offset k from the bottom, fill the horizontal span from center-(h-1-k) to center+(h-1-k).
    - Inside each pyramid, cells at even horizontal distance from the center keep the segment color and odd distances become black.
    - When multiple pyramids cover one cell, the pyramid from the taller original segment wins, even if its winning value there is black.

    Key insight:
    - Overlap priority is a global property of the source segment height, so a taller pyramid's black stripe can erase a shorter pyramid's visible color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    segments = _extract_segments(g)
    return _render_true_output(g.shape, segments)


def _coverage_stats(shape: tuple[int, int], segments: list[dict]) -> dict:
    overlap_cells = 0
    black_witness = 0
    color_witness = 0

    for r in range(shape[0]):
        for c in range(shape[1]):
            covering = []
            for seg in segments:
                value = _pyramid_value_at(seg, r, c)
                if value is not None:
                    covering.append((seg["height"], value, seg["color"]))
            if len(covering) < 2:
                continue

            overlap_cells += 1
            covering.sort(reverse=True)
            top_value = covering[0][1]
            lower_values = [value for _height, value, _color in covering[1:]]
            if top_value == _BG and any(value != _BG for value in lower_values):
                black_witness += 1
            if top_value != _BG and any(value not in (_BG, top_value) for value in lower_values):
                color_witness += 1

    return {
        "overlap_cells": overlap_cells,
        "black_witness": black_witness,
        "color_witness": color_witness,
    }


def _build_input(shape: tuple[int, int], segments: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for seg in segments:
        grid[seg["top"] : seg["bottom"] + 1, seg["col"]] = int(seg["color"])
    return grid


def _sample_segments(mode: int) -> tuple[tuple[int, int], list[dict]]:
    if mode == 0:
        heights = sorted(random.sample(range(4, 9), 2), reverse=True)
        roles = [0, 1]
        parity_pattern = [1]
        gaps = [random.choice([3, 5])]
    elif mode == 1:
        heights = sorted(random.sample(range(4, 9), 3), reverse=True)
        roles = [1, 0, 2]
        parity_pattern = [1, 0]
        gaps = [
            random.choice([3, 5]),
            random.choice([4, 6]),
        ]
    else:
        heights = sorted(random.sample(range(4, 9), 3), reverse=True)
        roles = [0, 1, 2]
        parity_pattern = [0, 1]
        gaps = [
            random.choice([4, 6]),
            random.choice([3, 5]),
        ]

    n = len(roles)
    colors = random.sample(_COLOR_POOL, n)

    raw_cols = [0]
    for gap in gaps:
        raw_cols.append(raw_cols[-1] + gap)

    segments = []
    for pos, raw_col in enumerate(raw_cols):
        height = heights[roles[pos]]
        segments.append({"col": raw_col, "height": height, "color": colors[pos]})

    span_min = min(seg["col"] - (seg["height"] - 1) for seg in segments)
    span_max = max(seg["col"] + (seg["height"] - 1) for seg in segments)
    required_width = span_max - span_min + 1
    width = random.randint(max(18, required_width + 2), 26)
    shift_lo = 1 - span_min
    shift_hi = (width - 2) - span_max
    shift = random.randint(shift_lo, shift_hi)
    for seg in segments:
        seg["col"] += shift

    max_height = max(seg["height"] for seg in segments)
    height = random.randint(max(18, max_height + 4), 26)
    bottom = random.randint(max_height, height - 2)
    for seg in segments:
        seg["bottom"] = bottom
        seg["top"] = bottom - seg["height"] + 1

    assert all(seg["top"] >= 1 for seg in segments)
    assert all(seg["bottom"] < height - 1 for seg in segments)
    assert len({seg["height"] for seg in segments}) == n
    assert len({seg["color"] for seg in segments}) == n
    assert all(3 <= segments[i + 1]["col"] - segments[i]["col"] <= 6 for i in range(n - 1))
    assert all((segments[i + 1]["col"] - segments[i]["col"]) % 2 == parity_pattern[i] for i in range(n - 1))

    return (height, width), segments


def _is_semantically_good(grid: np.ndarray, mode: int) -> bool:
    segments = _extract_segments(grid)
    out = _render_true_output(grid.shape, segments)
    ignore_black = _render_ignore_black(grid.shape, segments)
    reverse_priority = _render_reverse_priority(grid.shape, segments)
    stats = _coverage_stats(grid.shape, segments)
    covered_cells = sum(
        1
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if any(_pyramid_value_at(seg, r, c) is not None for seg in segments)
    )

    if stats["overlap_cells"] == 0:
        return False
    if stats["black_witness"] == 0:
        return False
    if mode != 0 and stats["color_witness"] == 0:
        return False
    if np.array_equal(out, ignore_black):
        return False
    if np.array_equal(out, reverse_priority):
        return False

    visible_colors = set(np.unique(out)) - {_BG}
    input_colors = {seg["color"] for seg in segments}
    if visible_colors != input_colors:
        return False

    density = np.count_nonzero(out) / out.size
    if not (0.03 <= density <= 0.18):
        return False

    overlap_ratio = stats["overlap_cells"] / max(1, covered_cells)
    if overlap_ratio > 0.48:
        return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    mode = (_GEN_CALL_COUNT - 1) % 3

    for _ in range(_GEN_ATTEMPTS):
        shape, segments = _sample_segments(mode)
        grid = _build_input(shape, segments)
        if _is_semantically_good(grid, mode):
            return grid

    raise RuntimeError("failed to generate a valid pyramid-overlap input")
