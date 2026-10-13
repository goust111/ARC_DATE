from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(flat, minlength=max(10, int(flat.max(initial=0)) + 1))
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _parse_segment(
    grid: np.ndarray,
    bg: int,
    grey: int,
    comp: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int], int, str]:
    h, w = grid.shape
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    cells = set(comp)

    if len(rows) == 1:
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        assert len(cols) >= 3
        assert cells == {(row, c) for c in range(c0, c1 + 1)}

        options: list[tuple[str, tuple[int, int], int]] = []
        left = (row, c0 - 1) if c0 > 0 else None
        right = (row, c1 + 1) if c1 + 1 < w else None

        if left is not None:
            left_color = int(grid[left])
            assert left_color != grey
            if left_color != bg:
                options.append(("left", left, left_color))
        if right is not None:
            right_color = int(grid[right])
            assert right_color != grey
            if right_color != bg:
                options.append(("right", right, right_color))

        assert len(options) == 1
        side, marker, color = options[0]
        if side == "left":
            ordered = [(row, c) for c in range(c0, c1 + 1)]
        else:
            ordered = [(row, c) for c in range(c1, c0 - 1, -1)]
        return ordered, marker, color, "h"

    if len(cols) == 1:
        col = cols[0]
        r0, r1 = rows[0], rows[-1]
        assert len(rows) >= 3
        assert cells == {(r, col) for r in range(r0, r1 + 1)}

        options = []
        up = (r0 - 1, col) if r0 > 0 else None
        down = (r1 + 1, col) if r1 + 1 < h else None

        if up is not None:
            up_color = int(grid[up])
            assert up_color != grey
            if up_color != bg:
                options.append(("up", up, up_color))
        if down is not None:
            down_color = int(grid[down])
            assert down_color != grey
            if down_color != bg:
                options.append(("down", down, down_color))

        assert len(options) == 1
        side, marker, color = options[0]
        if side == "up":
            ordered = [(r, col) for r in range(r0, r1 + 1)]
        else:
            ordered = [(r, col) for r in range(r1, r0 - 1, -1)]
        return ordered, marker, color, "v"

    raise AssertionError("grey component must be a straight segment")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each grey line segment as a skeleton and grow symmetric feathers from the marked end.

    Input structure
    - The background is the dominant color.
    - The most frequent non-background color is grey and forms one-cell-thick
      horizontal or vertical segments of length at least 3.
    - Exactly one endpoint of each segment has a single colored marker in the
      next cell along the segment's direction.

    Rule
    - Remove each marker.
    - Number skeleton cells starting at the marked endpoint.
    - For the i-th skeleton cell, draw i cells in the marker color on both
      perpendicular sides, clipping at the grid boundary.
    - Keep the grey skeleton unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > 0 and w > 0

    bg = _dominant_color(g)
    non_bg = [int(color) for color in np.unique(g) if int(color) != bg]
    assert non_bg

    grey = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))
    grey_mask = g == grey
    comps = _components(grey_mask)
    assert comps

    segments: list[tuple[list[tuple[int, int]], tuple[int, int], int, str]] = []
    marker_cells: set[tuple[int, int]] = set()
    for comp in comps:
        ordered, marker, color, orientation = _parse_segment(g, bg, grey, comp)
        segments.append((ordered, marker, color, orientation))
        marker_cells.add(marker)

    stray = {
        (int(r), int(c))
        for r, c in np.argwhere((g != bg) & (g != grey))
    }
    assert stray == marker_cells

    out = g.copy()
    for _ordered, marker, _color, _orientation in segments:
        out[marker] = bg

    for ordered, _marker, color, orientation in segments:
        for idx, (r, c) in enumerate(ordered, start=1):
            for step in range(1, idx + 1):
                if orientation == "h":
                    for nr in (r - step, r + step):
                        if 0 <= nr < h:
                            out[nr, c] = color
                else:
                    for nc in (c - step, c + step):
                        if 0 <= nc < w:
                            out[r, nc] = color

    return out


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 26
SEGMENT_COUNT_WEIGHTS = ((1, 2), (2, 4), (3, 3), (4, 1))
MIN_LENGTH = 3
MAX_LENGTH = 6
TRUNCATION_PROB = 0.4
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _buffered(cells: Iterable[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _preferred_indices(span: int, length: int, prefer_truncation: bool) -> list[int]:
    if prefer_truncation:
        idxs = set(range(min(length, span)))
        idxs.update(range(max(0, span - length), span))
        return sorted(idxs) or list(range(span))
    safe = list(range(length, span - length))
    return safe or list(range(span))


def _make_segment(
    shape: tuple[int, int],
    orientation: str,
    length: int,
    marker_on_negative_side: bool,
    fixed: int,
    start: int,
    color: int,
) -> dict:
    h, w = shape
    if orientation == "h":
        skeleton = [(fixed, start + i) for i in range(length)]
        if marker_on_negative_side:
            ordered = skeleton[:]
            marker = (fixed, start - 1)
        else:
            ordered = skeleton[::-1]
            marker = (fixed, start + length)
    else:
        skeleton = [(start + i, fixed) for i in range(length)]
        if marker_on_negative_side:
            ordered = skeleton[:]
            marker = (start - 1, fixed)
        else:
            ordered = skeleton[::-1]
            marker = (start + length, fixed)

    output_cells = set(skeleton)
    truncated = False
    for idx, (r, c) in enumerate(ordered, start=1):
        for step in range(1, idx + 1):
            if orientation == "h":
                for nr in (r - step, r + step):
                    if 0 <= nr < h:
                        output_cells.add((nr, c))
                    else:
                        truncated = True
            else:
                for nc in (c - step, c + step):
                    if 0 <= nc < w:
                        output_cells.add((r, nc))
                    else:
                        truncated = True

    return {
        "orientation": orientation,
        "length": length,
        "skeleton": skeleton,
        "ordered": ordered,
        "marker": marker,
        "color": color,
        "input_cells": set(skeleton) | {marker},
        "output_cells": output_cells,
        "truncated": truncated,
    }


def _sample_segment(
    shape: tuple[int, int],
    marker_colors: list[int],
    input_occupied: set[tuple[int, int]],
    input_blocked: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        orientation = random.choice(("h", "v"))
        prefer_truncation = random.random() < TRUNCATION_PROB
        negative_side = bool(random.getrandbits(1))

        if orientation == "h":
            max_length = min(MAX_LENGTH, w - 1)
            if max_length < MIN_LENGTH:
                continue
            if not prefer_truncation and h >= 2 * MIN_LENGTH + 1:
                max_safe = min(max_length, (h - 1) // 2)
                length = random.randint(MIN_LENGTH, max_safe) if max_safe >= MIN_LENGTH else random.randint(MIN_LENGTH, max_length)
            else:
                length = random.randint(MIN_LENGTH, max_length)

            rows = _preferred_indices(h, length, prefer_truncation)
            if negative_side:
                starts = list(range(1, w - length + 1))
            else:
                starts = list(range(0, w - length))
            if not rows or not starts:
                continue
            fixed = random.choice(rows)
            start = random.choice(starts)
        else:
            max_length = min(MAX_LENGTH, h - 1)
            if max_length < MIN_LENGTH:
                continue
            if not prefer_truncation and w >= 2 * MIN_LENGTH + 1:
                max_safe = min(max_length, (w - 1) // 2)
                length = random.randint(MIN_LENGTH, max_safe) if max_safe >= MIN_LENGTH else random.randint(MIN_LENGTH, max_length)
            else:
                length = random.randint(MIN_LENGTH, max_length)

            cols = _preferred_indices(w, length, prefer_truncation)
            if negative_side:
                starts = list(range(1, h - length + 1))
            else:
                starts = list(range(0, h - length))
            if not cols or not starts:
                continue
            fixed = random.choice(cols)
            start = random.choice(starts)

        segment = _make_segment(
            shape=shape,
            orientation=orientation,
            length=length,
            marker_on_negative_side=negative_side,
            fixed=fixed,
            start=start,
            color=random.choice(marker_colors),
        )

        if prefer_truncation and not segment["truncated"] and random.random() < 0.8:
            continue

        if segment["input_cells"] & input_blocked:
            continue
        if segment["input_cells"] & output_occupied:
            continue
        if segment["output_cells"] & input_occupied:
            continue
        if segment["output_cells"] & output_occupied:
            continue

        return segment
    return None


def _render_expected(
    shape: tuple[int, int],
    bg: int,
    grey: int,
    segments: list[dict],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for segment in segments:
        for r, c in segment["skeleton"]:
            out[r, c] = grey
        color = int(segment["color"])
        orientation = str(segment["orientation"])
        for idx, (r, c) in enumerate(segment["ordered"], start=1):
            for step in range(1, idx + 1):
                if orientation == "h":
                    for nr in (r - step, r + step):
                        if 0 <= nr < shape[0]:
                            out[nr, c] = color
                else:
                    for nc in (c - step, c + step):
                        if 0 <= nc < shape[1]:
                            out[r, nc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)

    bg = random.randint(0, 9)
    remaining = [c for c in range(10) if c != bg]
    grey = random.choice(remaining)
    marker_colors = [c for c in remaining if c != grey]

    area = h * w
    max_segments = 4 if area >= 320 else 3 if area >= 220 else 2
    choices = [count for count, _weight in SEGMENT_COUNT_WEIGHTS if count <= max_segments]
    weights = [weight for count, weight in SEGMENT_COUNT_WEIGHTS if count <= max_segments]
    target_segments = random.choices(choices, weights=weights, k=1)[0]

    grid = np.full((h, w), bg, dtype=int)
    segments: list[dict] = []
    input_occupied: set[tuple[int, int]] = set()
    input_blocked: set[tuple[int, int]] = set()
    output_occupied: set[tuple[int, int]] = set()

    for _ in range(target_segments):
        segment = _sample_segment(
            shape=(h, w),
            marker_colors=marker_colors,
            input_occupied=input_occupied,
            input_blocked=input_blocked,
            output_occupied=output_occupied,
        )
        if segment is None:
            return None

        for r, c in segment["skeleton"]:
            grid[r, c] = grey
        mr, mc = segment["marker"]
        grid[mr, mc] = int(segment["color"])

        segments.append(segment)
        input_occupied.update(segment["input_cells"])
        input_blocked.update(_buffered(segment["input_cells"], (h, w)))
        output_occupied.update(segment["output_cells"])

    expected = _render_expected((h, w), bg, grey, segments)
    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
