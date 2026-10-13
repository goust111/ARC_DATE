from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

COLOR_ORDER = (MAROON, BLUE, GREEN, RED, MAGENTA, ORANGE, GREY)
ORDER_INDEX = {color: i for i, color in enumerate(COLOR_ORDER)}
DIRS = {
    "h": ((0, 1), (0, -1)),
    "v": ((1, 0), (-1, 0)),
}

COUNT_CHOICES = (1, 2, 3, 4, 5, 6)
COUNT_WEIGHTS = (1, 2, 3, 3, 2, 1)
LEN_CAP = 7
GEN_RETRIES = 500
SCATTER_RETRIES = 120


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _component_cells(grid: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    color = int(grid[start])
    stack = [start]
    seen[start] = True
    cells: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        cells.append((int(r), int(c)))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                stack.append((int(nr), int(nc)))
    return cells


def _segment_from_cells(color: int, cells: list[tuple[int, int]]) -> dict[str, int | str]:
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    if len(rows) == 1:
        row = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1))
        assert len(cols) == len(cells) >= 2
        orientation = "h"
    elif len(cols) == 1:
        col = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1))
        assert len(rows) == len(cells) >= 2
        orientation = "v"
    else:
        raise AssertionError(f"component color {color} is not a straight segment")

    return {
        "color": int(color),
        "orientation": orientation,
        "length": int(len(cells)),
    }


def _extract_segments(grid: np.ndarray) -> tuple[tuple[int, int], list[dict[str, int | str]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    start_pixels = np.argwhere(g == YELLOW)
    assert len(start_pixels) == 1, "expected one yellow start marker"
    start = tuple(map(int, start_pixels[0]))

    seen = np.zeros(g.shape, dtype=bool)
    segments: list[dict[str, int | str]] = []
    used_colors: set[int] = set()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color in (BLACK, YELLOW) or seen[r, c]:
                continue
            assert color in ORDER_INDEX, f"unexpected segment color {color}"
            cells = _component_cells(g, (r, c), seen)
            segment = _segment_from_cells(color, cells)
            assert color not in used_colors, "segment colors must be unique"
            used_colors.add(color)
            segments.append(segment)

    assert 1 <= len(segments) <= 6
    segments.sort(key=lambda seg: ORDER_INDEX[int(seg["color"])])
    return start, segments


def _span_from(origin: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = origin
    dr, dc = direction
    if dr == 1:
        return h - r
    if dr == -1:
        return r + 1
    if dc == 1:
        return w - c
    return c + 1


def _choose_direction(
    shape: tuple[int, int],
    origin: tuple[int, int],
    orientation: str,
    length: int,
    *,
    first_segment: bool,
) -> tuple[int, int]:
    pref_dir, rev_dir = DIRS[orientation]
    pref_span = _span_from(origin, pref_dir, shape)
    if first_segment and orientation == "h":
        assert length <= pref_span, "first horizontal segment must extend right from the start"
        return pref_dir
    if length <= pref_span:
        return pref_dir
    rev_span = _span_from(origin, rev_dir, shape)
    assert length <= rev_span, "segment does not fit in either direction"
    return rev_dir


def _trace_segments(
    shape: tuple[int, int],
    start: tuple[int, int],
    segments: list[dict[str, int | str]],
) -> list[dict[str, object]]:
    traced: list[dict[str, object]] = []
    prev_end: tuple[int, int] | None = None
    prev_dir: tuple[int, int] | None = None
    prev_orientation: str | None = None

    for i, seg in enumerate(segments):
        orientation = str(seg["orientation"])
        length = int(seg["length"])
        color = int(seg["color"])

        if i == 0:
            origin = (start[0], start[1] + 1)
        else:
            assert prev_end is not None and prev_dir is not None and prev_orientation is not None
            assert orientation != prev_orientation, "successive segments must be perpendicular"
            origin = (prev_end[0] + prev_dir[0], prev_end[1] + prev_dir[1])

        assert _in_bounds(shape, origin), "corner/start-adjacent anchor leaves the grid"
        direction = _choose_direction(shape, origin, orientation, length, first_segment=(i == 0))
        cells = [
            (origin[0] + direction[0] * k, origin[1] + direction[1] * k)
            for k in range(length)
        ]
        assert all(_in_bounds(shape, cell) for cell in cells)

        traced.append(
            {
                "color": color,
                "orientation": orientation,
                "length": length,
                "dir": direction,
                "cells": cells,
                "origin": origin,
                "end": cells[-1],
            }
        )
        prev_end = cells[-1]
        prev_dir = direction
        prev_orientation = orientation

    return traced


def _render_output(
    shape: tuple[int, int],
    start: tuple[int, int],
    traced: list[dict[str, object]],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    out[start] = YELLOW
    for seg in traced:
        color = int(seg["color"])
        for r, c in seg["cells"]:
            assert int(out[r, c]) == BLACK
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Sort the scattered line segments by brightness and stitch them into one orthogonal polyline from the yellow start."""
    g = np.asarray(grid, dtype=int)
    start, segments = _extract_segments(g)
    traced = _trace_segments(g.shape, start, segments)
    return _render_output(g.shape, start, traced)


def _path_extension_ok(
    path_points: list[tuple[int, int]],
    new_cells: list[tuple[int, int]],
) -> bool:
    combined = path_points + new_cells
    if len(set(combined)) != len(combined):
        return False
    for i, (r0, c0) in enumerate(combined):
        for j in range(i + 1, len(combined)):
            if j == i + 1:
                continue
            r1, c1 = combined[j]
            if abs(r0 - r1) + abs(c0 - c1) == 1:
                return False
    return True


def _candidate_placements(
    shape: tuple[int, int],
    origin: tuple[int, int],
    orientation: str,
    *,
    is_last: bool,
    first_segment: bool,
    rng: random.Random,
) -> list[tuple[tuple[int, int], int]]:
    pref_dir, rev_dir = DIRS[orientation]
    pref_fit = _span_from(origin, pref_dir, shape)
    rev_fit = _span_from(origin, rev_dir, shape)

    pref_upper = pref_fit if is_last else pref_fit - 1
    rev_upper = rev_fit if is_last else rev_fit - 1

    pref: list[tuple[tuple[int, int], int]] = []
    rev: list[tuple[tuple[int, int], int]] = []

    if first_segment and orientation == "h":
        for length in range(2, min(LEN_CAP, pref_upper) + 1):
            pref.append((pref_dir, int(length)))
    else:
        for length in range(2, min(LEN_CAP, pref_upper) + 1):
            pref.append((pref_dir, int(length)))
        rev_lower = max(2, pref_fit + 1)
        for length in range(rev_lower, min(LEN_CAP, rev_upper) + 1):
            rev.append((rev_dir, int(length)))

    rng.shuffle(pref)
    rng.shuffle(rev)
    if pref and rev:
        return rev + pref if rng.random() < 0.35 else pref + rev
    return pref + rev


def _sample_traced_output(
    shape: tuple[int, int],
    start: tuple[int, int],
    colors: list[int],
    orientations: list[str],
    rng: random.Random,
) -> list[dict[str, object]] | None:
    specs: list[dict[str, int | str]] = []
    path_points = [start]
    traced: list[dict[str, object]] = []

    def rec(i: int) -> bool:
        nonlocal path_points, traced, specs
        if i == len(colors):
            return True

        orientation = orientations[i]
        if i == 0:
            origin = (start[0], start[1] + 1)
        else:
            prev = traced[-1]
            origin = (
                int(prev["end"][0]) + int(prev["dir"][0]),
                int(prev["end"][1]) + int(prev["dir"][1]),
            )
        if not _in_bounds(shape, origin):
            return False

        candidates = _candidate_placements(
            shape,
            origin,
            orientation,
            is_last=(i == len(colors) - 1),
            first_segment=(i == 0),
            rng=rng,
        )
        for direction, length in candidates:
            cells = [(origin[0] + direction[0] * k, origin[1] + direction[1] * k) for k in range(length)]
            if not all(_in_bounds(shape, cell) for cell in cells):
                continue
            if not _path_extension_ok(path_points, cells):
                continue

            seg = {
                "color": int(colors[i]),
                "orientation": orientation,
                "length": int(length),
                "dir": direction,
                "cells": cells,
                "origin": origin,
                "end": cells[-1],
            }
            traced.append(seg)
            specs.append(
                {
                    "color": int(colors[i]),
                    "orientation": orientation,
                    "length": int(length),
                }
            )
            old_points = path_points
            path_points = path_points + cells
            if rec(i + 1):
                return True
            path_points = old_points
            specs.pop()
            traced.pop()

        return False

    if rec(0):
        return traced
    return None


def _mark_reserved(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    reserved[nr, nc] = True


def _segment_patch(orientation: str, length: int, color: int) -> np.ndarray:
    if orientation == "h":
        patch = np.full((1, length), color, dtype=int)
    else:
        patch = np.full((length, 1), color, dtype=int)
    return patch


def _scatter_segments(
    shape: tuple[int, int],
    start: tuple[int, int],
    traced: list[dict[str, object]],
    rng: random.Random,
) -> np.ndarray | None:
    h, w = shape
    grid = np.full(shape, BLACK, dtype=int)
    grid[start] = YELLOW
    reserved = np.zeros(shape, dtype=bool)
    _mark_reserved(reserved, [start])

    order = list(range(len(traced)))
    rng.shuffle(order)
    for idx in order:
        seg = traced[idx]
        orientation = str(seg["orientation"])
        length = int(seg["length"])
        color = int(seg["color"])
        patch = _segment_patch(orientation, length, color)
        ph, pw = patch.shape

        candidates: list[tuple[int, int]] = []
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                if np.any(reserved[r : r + ph, c : c + pw]):
                    continue
                cells = [
                    (r + rr, c + cc)
                    for rr in range(ph)
                    for cc in range(pw)
                    if int(patch[rr, cc]) != BLACK
                ]
                halo_ok = True
                for rr, cc in cells:
                    if reserved[rr, cc]:
                        halo_ok = False
                        break
                if halo_ok:
                    candidates.append((int(r), int(c)))
        if not candidates:
            return None

        r0, c0 = rng.choice(candidates)
        mask = patch != BLACK
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        placed_cells = [
            (r0 + rr, c0 + cc)
            for rr in range(ph)
            for cc in range(pw)
            if int(patch[rr, cc]) != BLACK
        ]
        _mark_reserved(reserved, placed_cells)

    return grid


def _sample_start(shape: tuple[int, int], first_orientation: str, count: int, rng: random.Random) -> tuple[int, int] | None:
    h, w = shape
    min_right_room = 2 if count == 1 else 3
    max_c = w - 1 - min_right_room
    if max_c < 0:
        return None

    if first_orientation == "v" and rng.random() < 0.4:
        edge_rows = list(range(min(3, h))) + list(range(max(0, h - 3), h))
        r = rng.choice(edge_rows)
    else:
        r = rng.randrange(h)
    c = rng.randrange(max_c + 1)
    return int(r), int(c)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choices(COUNT_CHOICES, weights=COUNT_WEIGHTS, k=1)[0]
    h = rng.randint(max(10, 9 + count), min(24, 15 + count))
    w = rng.randint(max(10, 9 + count), min(24, 15 + count))

    colors_subset = set(rng.sample(COLOR_ORDER, count))
    colors = [color for color in COLOR_ORDER if color in colors_subset]
    first_orientation = rng.choice(("h", "v"))
    orientations = [first_orientation if i % 2 == 0 else ("v" if first_orientation == "h" else "h") for i in range(count)]

    start = _sample_start((h, w), first_orientation, count, rng)
    if start is None:
        return None

    traced = _sample_traced_output((h, w), start, colors, orientations, rng)
    if traced is None:
        return None

    expected = _render_output((h, w), start, traced)
    for _ in range(SCATTER_RETRIES):
        inp = _scatter_segments((h, w), start, traced, rng)
        if inp is None:
            continue
        try:
            out = main(inp)
        except AssertionError:
            continue
        if np.array_equal(out, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
