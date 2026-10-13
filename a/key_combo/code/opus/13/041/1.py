from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


PALETTE = (2, 3, 1, 4, 6)
BACKGROUND = 0
MIN_SEGMENTS = 3
MAX_SEGMENTS = 5
MIN_LENGTH = 2
MAX_LENGTH = 7
INPUT_ROWS = (13, 17)
INPUT_COLS = (13, 17)
PLACEMENT_TRIES = 300
GENERATION_TRIES = 800
GLOBAL_SIGNATURE_LIMIT = 4000
_USED_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], ...]] = set()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _component_to_segment(comp: dict) -> dict:
    coords = comp["coords"]
    rows = coords[:, 0]
    cols = coords[:, 1]
    color = int(comp["color"])

    same_row = int(np.unique(rows).size) == 1
    same_col = int(np.unique(cols).size) == 1
    assert same_row ^ same_col, "each component must be a horizontal or vertical segment"

    if same_row:
        ordered = np.sort(cols)
        assert np.array_equal(ordered, np.arange(int(ordered[0]), int(ordered[-1]) + 1))
        top_left = (int(rows[0]), int(ordered[0]))
        orientation = "H"
        length = int(ordered.size)
    else:
        ordered = np.sort(rows)
        assert np.array_equal(ordered, np.arange(int(ordered[0]), int(ordered[-1]) + 1))
        top_left = (int(ordered[0]), int(cols[0]))
        orientation = "V"
        length = int(ordered.size)

    return {
        "color": color,
        "length": length,
        "top_left": top_left,
        "orientation": orientation,
    }


def _sorted_segments(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    segments = [_component_to_segment(comp) for comp in _find_components(grid, bg)]
    assert segments, "expected at least one segment"
    lengths = [seg["length"] for seg in segments]
    assert len(lengths) == len(set(lengths)), "segment lengths must be distinct"
    return sorted(segments, key=lambda seg: (-seg["length"], seg["top_left"][0], seg["top_left"][1]))


def _diamond_output(lengths_desc: list[int], colors_desc: list[int]) -> np.ndarray:
    assert lengths_desc and len(lengths_desc) == len(colors_desc)
    assert lengths_desc == sorted(lengths_desc, reverse=True)
    side = 2 * lengths_desc[0] - 1
    center = lengths_desc[0] - 1
    rows = np.arange(side)[:, None]
    cols = np.arange(side)[None, :]
    dist = np.abs(rows - center) + np.abs(cols - center)
    out = np.zeros((side, side), dtype=int)
    for idx, color in enumerate(colors_desc):
        radius = center - idx
        assert radius >= 0
        out[dist == radius] = int(color)
    return out


def _square_output(lengths_desc: list[int], colors_desc: list[int]) -> np.ndarray:
    side = 2 * lengths_desc[0] - 1
    out = np.zeros((side, side), dtype=int)
    for idx, color in enumerate(colors_desc):
        out[idx : side - idx, idx : side - idx] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered segments by length and redraw them as concentric Manhattan-diamond outlines from outside to inside.

    Input structure:
    - The grid has a black background and 3 to 5 isolated monochrome straight segments.
    - Each foreground object is a 4-connected horizontal or vertical bar in one distinct color.
    - Segment lengths are all different, so descending length order is unambiguous.

    Rule:
    - Detect every straight segment and record its color, length, and top-left cell.
    - Sort the segments by decreasing length; if a tie ever occurs, break it by top-left row then column.
    - Let L be the longest length and create a black output grid of shape (2L-1, 2L-1) centered at (L-1, L-1).
    - Paint the k-th sorted segment onto the diamond outline with Manhattan radius L-1-k; if fewer than L segments exist, the remaining inner cells stay black.

    Key insight:
    - The bars are only a legend: their positions and orientations are discarded, while their lengths simultaneously determine both the output order and the diamond stack size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    segments = _sorted_segments(g)
    lengths_desc = [seg["length"] for seg in segments]
    colors_desc = [seg["color"] for seg in segments]
    return _diamond_output(lengths_desc, colors_desc)


def _segment_cells(r0: int, c0: int, length: int, orientation: str) -> list[tuple[int, int]]:
    if orientation == "H":
        return [(r0, c0 + i) for i in range(length)]
    return [(r0 + i, c0) for i in range(length)]


def _mark_forbidden(mask: np.ndarray, cells: list[tuple[int, int]], gap: int = 2) -> None:
    h, w = mask.shape
    for r, c in cells:
        r0 = max(0, r - gap)
        r1 = min(h - 1, r + gap)
        c0 = max(0, c - gap)
        c1 = min(w - 1, c + gap)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _length_plan(rng: random.Random, count: int) -> list[int]:
    # With the task's length lower bound of 2, valid generated inputs always satisfy N < L.
    longest = rng.randint(max(count + 1, 4), MAX_LENGTH)
    others = rng.sample(list(range(MIN_LENGTH, longest)), count - 1)
    return [longest, *sorted(others, reverse=True)]


def _color_plan(rng: random.Random, count: int) -> list[int]:
    while True:
        colors = rng.sample(list(PALETTE), count)
        if colors != sorted(colors) and colors != sorted(colors, reverse=True):
            return colors


def _orientation_plan(rng: random.Random, count: int) -> list[str]:
    orientations = [rng.choice(("H", "V")) for _ in range(count)]
    if len(set(orientations)) == 1:
        flip_idx = rng.randrange(count)
        orientations[flip_idx] = "V" if orientations[flip_idx] == "H" else "H"
    return orientations


def _place_segments(
    rng: random.Random,
    shape: tuple[int, int],
    lengths_desc: list[int],
    colors_desc: list[int],
    orientations: list[str],
) -> list[dict] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    forbidden = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    for length, color, orientation in zip(lengths_desc, colors_desc, orientations):
        if orientation == "H":
            candidates = [(r, c) for r in range(h) for c in range(w - length + 1)]
        else:
            candidates = [(r, c) for r in range(h - length + 1) for c in range(w)]
        rng.shuffle(candidates)

        chosen = None
        for r0, c0 in candidates:
            cells = _segment_cells(r0, c0, length, orientation)
            if any(occupied[r, c] or forbidden[r, c] for r, c in cells):
                continue
            chosen = (r0, c0, cells)
            break
        if chosen is None:
            return None

        r0, c0, cells = chosen
        for r, c in cells:
            occupied[r, c] = True
        _mark_forbidden(forbidden, cells, gap=2)
        placed.append(
            {
                "length": int(length),
                "color": int(color),
                "orientation": orientation,
                "top_left": (int(r0), int(c0)),
                "cells": cells,
            }
        )
    return placed


def _render_input(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for seg in placed:
        for r, c in seg["cells"]:
            grid[r, c] = int(seg["color"])
    return grid


def _minimum_pair_gap(placed: list[dict]) -> int:
    gap = 10**9
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            for r1, c1 in placed[i]["cells"]:
                for r2, c2 in placed[j]["cells"]:
                    cheb = max(abs(r1 - r2), abs(c1 - c2))
                    gap = min(gap, cheb - 1)
    return gap


def _validate_example(grid: np.ndarray, lengths_desc: list[int], colors_desc: list[int]) -> bool:
    try:
        segments = _sorted_segments(grid)
    except Exception:
        return False

    if [seg["length"] for seg in segments] != lengths_desc:
        return False
    if [seg["color"] for seg in segments] != colors_desc:
        return False

    expected = _diamond_output(lengths_desc, colors_desc)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(expected, _square_output(lengths_desc, colors_desc)):
        return False

    row_major_colors = [seg["color"] for seg in sorted(segments, key=lambda seg: seg["top_left"])]
    if row_major_colors == colors_desc:
        return False

    center = lengths_desc[0] - 1
    if len(lengths_desc) == lengths_desc[0]:
        if int(expected[center, center]) != int(colors_desc[-1]):
            return False
    else:
        if int(expected[center, center]) != BACKGROUND:
            return False

    if len(set(seg["orientation"] for seg in segments)) < 2:
        return False

    return True


def _sample_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    lengths_desc = _length_plan(rng, count)
    colors_desc = _color_plan(rng, count)
    signature = tuple(zip(lengths_desc, colors_desc))
    if signature in _USED_OUTPUT_SIGNATURES:
        return None

    orientations = _orientation_plan(rng, count)
    rows = rng.randint(*INPUT_ROWS)
    cols = rng.randint(*INPUT_COLS)
    placed = _place_segments(rng, (rows, cols), lengths_desc, colors_desc, orientations)
    if placed is None:
        return None

    if _minimum_pair_gap(placed) < 2:
        return None

    grid = _render_input((rows, cols), placed)
    if not _validate_example(grid, lengths_desc, colors_desc):
        return None

    _USED_OUTPUT_SIGNATURES.add(signature)
    if len(_USED_OUTPUT_SIGNATURES) > GLOBAL_SIGNATURE_LIMIT:
        _USED_OUTPUT_SIGNATURES.clear()
        _USED_OUTPUT_SIGNATURES.add(signature)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_TRIES):
        grid = _sample_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid non-degenerate example")
