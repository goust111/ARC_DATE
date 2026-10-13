from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


Coord = tuple[int, int]
Segment = tuple[str, int, int, int, int]  # axis, fixed, start, end, color

BG_COLOR = 0
WORK_SHAPE = (28, 28)
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _candidate_segments(markers: dict[Coord, int]) -> set[Segment]:
    rows: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    cols: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for (r, c), color in markers.items():
        rows[color][r].append(c)
        cols[color][c].append(r)

    segments: set[Segment] = set()
    for color, row_map in rows.items():
        for r, xs in row_map.items():
            if len(xs) >= 2:
                segments.add(("h", int(r), min(xs), max(xs), int(color)))
    for color, col_map in cols.items():
        for c, ys in col_map.items():
            if len(ys) >= 2:
                segments.add(("v", int(c), min(ys), max(ys), int(color)))
    return segments


def _segment_cells(seg: Segment):
    axis, fixed, start, end, _color = seg
    if axis == "h":
        for c in range(start, end + 1):
            yield fixed, c
    else:
        for r in range(start, end + 1):
            yield r, fixed


def _segment_covers(seg: Segment, pos: Coord) -> bool:
    axis, fixed, start, end, _color = seg
    r, c = pos
    if axis == "h":
        return r == fixed and start <= c <= end
    return c == fixed and start <= r <= end


def _simulate_from_markers(
    shape: tuple[int, int], bg: int, markers: dict[Coord, int], *, with_meta: bool = False
):
    line_layer = np.full(shape, bg, dtype=int)
    seen_segments: set[Segment] = set()
    rounds = 0
    painted_cells = 0
    recolor_count = 0
    line_conflict_cells: set[Coord] = set()
    marker_conflict_cells: set[Coord] = set()

    round_limit = max(20, len(markers) * 6)
    for _ in range(round_limit):
        new_segments = sorted(_candidate_segments(markers) - seen_segments)
        if not new_segments:
            break

        rounds += 1
        seen_segments.update(new_segments)

        proposals: dict[Coord, set[int]] = defaultdict(set)
        crossing_colors: dict[Coord, set[int]] = defaultdict(set)

        for seg in new_segments:
            color = seg[4]
            for cell in _segment_cells(seg):
                if cell in markers:
                    if markers[cell] != color:
                        crossing_colors[cell].add(color)
                    continue
                proposals[cell].add(color)

        for cell, colors in proposals.items():
            r, c = cell
            if len(colors) == 1:
                color = next(iter(colors))
                if line_layer[r, c] == bg:
                    line_layer[r, c] = color
                    painted_cells += 1
            elif line_layer[r, c] == bg:
                line_conflict_cells.add(cell)

        recolors: dict[Coord, int] = {}
        for pos, colors in crossing_colors.items():
            if len(colors) == 1:
                new_color = next(iter(colors))
                if markers[pos] != new_color:
                    recolors[pos] = new_color
            elif len(colors) >= 2:
                marker_conflict_cells.add(pos)

        if recolors:
            recolor_count += len(recolors)
            markers.update(recolors)
    else:
        raise RuntimeError("simulation did not converge")

    out = line_layer.copy()
    for (r, c), color in markers.items():
        out[r, c] = color

    if not with_meta:
        return out

    return out, {
        "rounds": rounds,
        "painted_cells": painted_cells,
        "recolors": recolor_count,
        "segment_count": len(seen_segments),
        "line_conflicts": len(line_conflict_cells),
        "marker_conflicts": len(marker_conflict_cells),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly extend same-color marker spans and recolor markers hit by exactly one new foreign line.

    Input structure:
    - A single background color fills the canvas.
    - Every non-background cell is an isolated colored marker pixel.
    - Multiple marker colors may appear, and future line spans stay inside the same canvas.

    Rule:
    - Track the marker pixels separately from the permanent line layer.
    - In each round, for every color, draw each row span and column span whose endpoints are two or more
      markers of that color; new lines never overwrite an existing different color, so conflicts keep the
      visible cell unchanged.
    - Still within that round, if a marker is covered by exactly one distinct foreign new line color, recolor
      that marker to the line color; markers hit by zero or multiple foreign new colors stay unchanged.
    - Keep all previously created spans, and repeat until no new spans are created.

    Key insight:
    - The decisive state is not just the visible canvas but the evolving marker identities: newly created spans
      can recolor markers, which in turn unlock fresh same-color alignments in later rounds until a fixed point.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    markers = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != bg)
    }
    if not markers:
        return g.copy()
    return _simulate_from_markers(g.shape, bg, markers)


def _normalize_markers(markers: dict[Coord, int]) -> dict[Coord, int]:
    min_r = min(r for r, _c in markers)
    min_c = min(c for _r, c in markers)
    return {(r - min_r, c - min_c): color for (r, c), color in markers.items()}


def _transform_markers(markers: dict[Coord, int]) -> dict[Coord, int]:
    markers = _normalize_markers(markers)
    max_r = max(r for r, _c in markers)
    max_c = max(c for _r, c in markers)
    mode = random.randrange(8)
    transformed: dict[Coord, int] = {}

    for (r, c), color in markers.items():
        if mode == 0:
            nr, nc = r, c
        elif mode == 1:
            nr, nc = max_r - r, c
        elif mode == 2:
            nr, nc = r, max_c - c
        elif mode == 3:
            nr, nc = max_r - r, max_c - c
        elif mode == 4:
            nr, nc = c, r
        elif mode == 5:
            nr, nc = c, max_r - r
        elif mode == 6:
            nr, nc = max_c - c, r
        else:
            nr, nc = max_c - c, max_r - r
        transformed[(nr, nc)] = color

    return _normalize_markers(transformed)


def _bbox_of_markers(markers: dict[Coord, int]) -> tuple[int, int]:
    max_r = max(r for r, _c in markers)
    max_c = max(c for _r, c in markers)
    return max_r + 1, max_c + 1


def _build_chain_gadget(source: int, targets: list[int]) -> dict[Coord, int]:
    markers: dict[Coord, int] = {}

    first_span = random.randint(5, 8)
    first_target = random.randint(2, first_span - 2)
    markers[(0, 0)] = source
    markers[(0, first_span)] = source
    markers[(0, first_target)] = targets[0]

    if random.random() < 0.4:
        candidates = [
            c
            for c in range(1, first_span)
            if min(abs(c - x) for x in (0, first_target, first_span)) >= 2
        ]
        if candidates:
            markers[(0, random.choice(candidates))] = source

    current = (0, first_target)
    vertical = True
    for color in targets[1:]:
        span = random.randint(4, 7)
        offset = random.randint(2, span - 2)
        r, c = current
        if vertical:
            anchor = (r + span, c)
            nxt = (r + offset, c)
        else:
            anchor = (r, c + span)
            nxt = (r, c + offset)
        markers[anchor] = source
        markers[nxt] = color
        current = nxt
        vertical = not vertical

    return markers


def _build_marker_conflict_gadget(color_a: int, color_b: int, marker_color: int) -> dict[Coord, int]:
    return {
        (3, 0): color_a,
        (3, 6): color_a,
        (0, 3): color_b,
        (6, 3): color_b,
        (3, 3): marker_color,
    }


def _build_background_conflict_gadget(color_a: int, color_b: int) -> dict[Coord, int]:
    return {
        (3, 0): color_a,
        (3, 6): color_a,
        (0, 3): color_b,
        (6, 3): color_b,
    }


def _place_gadget(
    canvas_markers: dict[Coord, int],
    gadget: dict[Coord, int],
    reserved_rects: list[tuple[int, int, int, int]],
) -> bool:
    h, w = _bbox_of_markers(gadget)
    tries = 80
    pad = 2

    for _ in range(tries):
        top = random.randint(0, WORK_SHAPE[0] - h)
        left = random.randint(0, WORK_SHAPE[1] - w)
        rect = (top - pad, left - pad, top + h - 1 + pad, left + w - 1 + pad)
        if any(not (rect[2] < r0 or r1 < rect[0] or rect[3] < c0 or c1 < rect[1]) for r0, c0, r1, c1 in reserved_rects):
            continue

        translated = {(r + top, c + left): color for (r, c), color in gadget.items()}
        if any(pos in canvas_markers for pos in translated):
            continue

        canvas_markers.update(translated)
        reserved_rects.append(rect)
        return True

    return False


def _markers_are_isolated(markers: dict[Coord, int]) -> bool:
    coords = list(markers)
    for i, (r1, c1) in enumerate(coords):
        for r2, c2 in coords[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _crop_with_margin(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG_COLOR)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    up = random.randint(1, 3)
    down = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    r0 = max(0, int(r0) - up)
    c0 = max(0, int(c0) - left)
    r1 = min(grid.shape[0] - 1, int(r1) + down)
    c1 = min(grid.shape[1] - 1, int(c1) + right)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _gen_attempt() -> np.ndarray | None:
    canvas_markers: dict[Coord, int] = {}
    reserved_rects: list[tuple[int, int, int, int]] = []

    stages = random.randint(2, 4)
    side_mode = random.choice(["none", "marker_conflict", "background_conflict"])

    needed_colors = 1 + stages
    if side_mode == "marker_conflict":
        needed_colors += 3
    elif side_mode == "background_conflict":
        needed_colors += 2

    palette = random.sample(COLOR_POOL, needed_colors)
    idx = 0
    source = palette[idx]
    idx += 1
    chain_targets = palette[idx : idx + stages]
    idx += stages

    gadgets = [_transform_markers(_build_chain_gadget(source, chain_targets))]

    expected_marker_conflict = False
    expected_line_conflict = False
    if side_mode == "marker_conflict":
        gadgets.append(
            _transform_markers(
                _build_marker_conflict_gadget(palette[idx], palette[idx + 1], palette[idx + 2])
            )
        )
        expected_marker_conflict = True
    elif side_mode == "background_conflict":
        gadgets.append(_transform_markers(_build_background_conflict_gadget(palette[idx], palette[idx + 1])))
        expected_line_conflict = True

    random.shuffle(gadgets)
    for gadget in gadgets:
        if not _place_gadget(canvas_markers, gadget, reserved_rects):
            return None

    if not _markers_are_isolated(canvas_markers):
        return None

    grid = np.full(WORK_SHAPE, BG_COLOR, dtype=int)
    for (r, c), color in canvas_markers.items():
        grid[r, c] = color
    grid = _crop_with_margin(grid)

    out, meta = _simulate_from_markers(
        grid.shape,
        BG_COLOR,
        {(int(r), int(c)): int(grid[r, c]) for r, c in np.argwhere(grid != BG_COLOR)},
        with_meta=True,
    )

    if meta["rounds"] < 2:
        return None
    if meta["recolors"] < stages:
        return None
    if meta["painted_cells"] < 10:
        return None
    if np.array_equal(grid, out):
        return None
    if expected_marker_conflict and meta["marker_conflicts"] == 0:
        return None
    if expected_line_conflict and meta["line_conflicts"] == 0:
        return None
    if out.shape[0] > 30 or out.shape[1] > 30:
        return None
    if np.count_nonzero(out != BG_COLOR) > out.size * 0.45:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
