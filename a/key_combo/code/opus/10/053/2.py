from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
BLUE = 1
PALETTE_COLORS = tuple(range(2, 10))

MIN_GRID = 15
MAX_GRID = 20
MIN_LINE_LENGTH = 1
MAX_LINE_LENGTH = 5
MAX_GENERATION_ATTEMPTS = 500


def _read_palette(grid: np.ndarray) -> list[int]:
    h, w = map(int, grid.shape)
    limit = min(h, w)
    palette: list[int] = []
    for i in range(limit):
        color = int(grid[i, i])
        if color in (BACKGROUND, BLUE):
            break
        palette.append(color)

    assert palette, "expected a non-empty diagonal palette"
    assert len(set(palette)) == len(palette), "palette colors must be unique"

    palette_coords = {(i, i) for i in range(len(palette))}
    for r, c in np.argwhere((grid != BACKGROUND) & (grid != BLUE)):
        coord = (int(r), int(c))
        assert coord in palette_coords, "non-blue non-background cells must be palette markers"
    return palette


def _segment_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        return [(r0, c) for c in range(c0 + 1, c1)]
    return [(r, c0) for r in range(r0 + 1, r1)]


def _extract_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    segments: list[dict] = []
    seen_keys: set[tuple[int, int, int]] = set()

    marker_rows = [int(np.count_nonzero(g[r] == BLUE)) for r in range(h)]
    marker_cols = [int(np.count_nonzero(g[:, c] == BLUE)) for c in range(w)]
    assert all(count <= 2 for count in marker_rows), "each row may contain at most two blue markers"
    assert all(count <= 2 for count in marker_cols), "each column may contain at most two blue markers"

    for r in range(h):
        cols = [int(c) for c in np.flatnonzero(g[r] == BLUE)]
        if len(cols) != 2:
            continue
        c0, c1 = cols
        length = c1 - c0 - 1
        assert length >= 1, "horizontal connections must have positive length"
        assert np.all(g[r, c0 + 1 : c1] == BACKGROUND), "horizontal span must be empty in the input"
        start = (r, c0)
        key = (length, start[0], start[1])
        assert key not in seen_keys, "segment ordering would be ambiguous"
        seen_keys.add(key)
        segments.append(
            {
                "axis": "h",
                "start": start,
                "end": (r, c1),
                "length": length,
                "cells": _segment_cells(start, (r, c1)),
            }
        )

    for c in range(w):
        rows = [int(r) for r in np.flatnonzero(g[:, c] == BLUE)]
        if len(rows) != 2:
            continue
        r0, r1 = rows
        length = r1 - r0 - 1
        assert length >= 1, "vertical connections must have positive length"
        assert np.all(g[r0 + 1 : r1, c] == BACKGROUND), "vertical span must be empty in the input"
        start = (r0, c)
        key = (length, start[0], start[1])
        assert key not in seen_keys, "segment ordering would be ambiguous"
        seen_keys.add(key)
        segments.append(
            {
                "axis": "v",
                "start": start,
                "end": (r1, c),
                "length": length,
                "cells": _segment_cells(start, (r1, c)),
            }
        )

    assert segments, "expected at least one line segment"
    return sorted(segments, key=lambda seg: (seg["length"], seg["start"][0], seg["start"][1]))


def _paint_segments(grid: np.ndarray, palette: list[int], segments: list[dict]) -> np.ndarray:
    assert len(segments) <= len(palette), "palette must be long enough for all segments"
    out = np.asarray(grid, dtype=int).copy()
    for color, segment in zip(palette, segments):
        for r, c in segment["cells"]:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Color aligned blue-marker connections by mapping line-length rank to the diagonal palette.

    Input structure:
    - A black background grid contains a contiguous top-left diagonal palette at
      `(0,0), (1,1), ...`, using unique non-black, non-blue colors.
    - Every other foreground pixel is a blue singleton marker.
    - Some rows or columns contain exactly two blue markers, and their spans are
      otherwise empty.

    Rule:
    - Preserve the diagonal palette pixels and every blue marker.
    - For each row with exactly two blue markers, take the strictly interior
      horizontal cells as one segment; for each column with exactly two blue
      markers, take the strictly interior vertical cells as one segment.
    - Measure segment lengths by their filled interior cell counts, sort segments
      by increasing length, and break equal-length ties by the lexicographically
      smaller endpoint coordinate of the segment.
    - Recolor the cells of the `i`-th segment in that order with the `i`-th
      palette color read from the diagonal.

    Key insight:
    - The diagonal markers are an external ordered palette, while the actual
      color assignment is decided only after globally ranking all row/column
      connections by length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    palette = _read_palette(g)
    segments = _extract_segments(g)
    return _paint_segments(g, palette, segments)


LAYOUT_OPTIONS = [
    (0, 2, 1),
    (0, 1, 2),
    (0, 2, 2),
    (0, 3, 1),
    (0, 1, 3),
    (0, 3, 2),
    (0, 2, 3),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
    (1, 2, 0),
    (1, 0, 2),
    (1, 1, 2),
    (1, 2, 1),
    (2, 0, 0),
]


def _sample_rect_lengths(n_rectangles: int) -> list[tuple[int, int]]:
    lengths: list[tuple[int, int]] = []
    for _ in range(n_rectangles):
        h_len = random.randint(MIN_LINE_LENGTH, MAX_LINE_LENGTH)
        v_choices = [v for v in range(MIN_LINE_LENGTH, MAX_LINE_LENGTH + 1) if v != h_len]
        v_len = random.choice(v_choices)
        lengths.append((h_len, v_len))

    if n_rectangles == 2 and random.random() < 0.5:
        shared = random.choice([0, 1])
        value = lengths[0][shared]
        other = list(lengths[1])
        if shared == 0 and value != other[1]:
            other[0] = value
        elif shared == 1 and value != other[0]:
            other[1] = value
        lengths[1] = tuple(other)
    return lengths


def _sample_pair_lengths(count: int, rect_lengths: Iterable[int]) -> list[int]:
    pair_lengths = [random.randint(MIN_LINE_LENGTH, MAX_LINE_LENGTH) for _ in range(count)]
    rect_lengths = list(rect_lengths)
    if count >= 2 and random.random() < 0.5:
        shared = random.randint(MIN_LINE_LENGTH, MAX_LINE_LENGTH)
        i, j = random.sample(range(count), 2)
        pair_lengths[i] = shared
        pair_lengths[j] = shared
    elif count >= 1 and rect_lengths and random.random() < 0.5:
        pair_lengths[random.randrange(count)] = random.choice(rect_lengths)
    return pair_lengths


def _build_modules() -> tuple[list[dict], int]:
    n_rectangles, n_hpairs, n_vpairs = random.choice(LAYOUT_OPTIONS)
    rect_lengths = _sample_rect_lengths(n_rectangles)
    pair_lengths = _sample_pair_lengths(n_hpairs + n_vpairs, [x for pair in rect_lengths for x in pair])

    modules: list[dict] = []
    for h_len, v_len in rect_lengths:
        modules.append(
            {
                "kind": "rect",
                "h_len": h_len,
                "v_len": v_len,
                "height": v_len + 2,
                "width": h_len + 2,
            }
        )

    split = 0
    for _ in range(n_hpairs):
        length = pair_lengths[split]
        split += 1
        modules.append({"kind": "hpair", "length": length, "height": 1, "width": length + 2})
    for _ in range(n_vpairs):
        length = pair_lengths[split]
        split += 1
        modules.append({"kind": "vpair", "length": length, "height": length + 2, "width": 1})

    n_segments = 4 * n_rectangles + n_hpairs + n_vpairs
    return modules, n_segments


def _module_bbox(module: dict, r0: int, c0: int) -> tuple[int, int, int, int]:
    return (r0, c0, r0 + module["height"] - 1, c0 + module["width"] - 1)


def _module_marker_rows_cols(module: dict, r0: int, c0: int) -> tuple[set[int], set[int]]:
    bbox = _module_bbox(module, r0, c0)
    r1, c1 = bbox[2], bbox[3]
    if module["kind"] == "hpair":
        return {r0}, {c0, c1}
    if module["kind"] == "vpair":
        return {r0, r1}, {c0}
    return {r0, r1}, {c0, c1}


def _bbox_free(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r1 + 2)
    cc1 = min(occupied.shape[1], c1 + 2)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _place_modules(modules: list[dict], height: int, width: int, palette_len: int) -> tuple[list[dict], set[tuple[int, int]]] | None:
    start = palette_len + 1
    occupied = np.zeros((height, width), dtype=bool)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    placed: list[dict] = []
    blue_cells: set[tuple[int, int]] = set()

    for module in sorted(modules, key=lambda item: item["height"] * item["width"], reverse=True):
        row_min = start
        row_max = height - module["height"]
        col_min = start
        col_max = width - module["width"]
        if row_min > row_max or col_min > col_max:
            return None

        placed_here = False
        for _ in range(200):
            r0 = random.randint(row_min, row_max)
            c0 = random.randint(col_min, col_max)
            bbox = _module_bbox(module, r0, c0)
            marker_rows, marker_cols = _module_marker_rows_cols(module, r0, c0)
            if marker_rows & used_rows or marker_cols & used_cols:
                continue
            if not _bbox_free(occupied, bbox):
                continue

            r1, c1 = bbox[2], bbox[3]
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            placed_module = dict(module)
            placed_module["bbox"] = bbox
            placed.append(placed_module)
            used_rows.update(marker_rows)
            used_cols.update(marker_cols)
            if module["kind"] == "hpair":
                blue_cells.update({(r0, c0), (r0, c1)})
            elif module["kind"] == "vpair":
                blue_cells.update({(r0, c0), (r1, c0)})
            else:
                blue_cells.update({(r0, c0), (r0, c1), (r1, c0), (r1, c1)})
            placed_here = True
            break

        if not placed_here:
            return None

    return placed, blue_cells


def _planned_segments(placed_modules: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for module in placed_modules:
        r0, c0, r1, c1 = module["bbox"]
        if module["kind"] == "hpair":
            start = (r0, c0)
            end = (r0, c1)
            segments.append(
                {"axis": "h", "start": start, "end": end, "length": c1 - c0 - 1, "cells": _segment_cells(start, end)}
            )
        elif module["kind"] == "vpair":
            start = (r0, c0)
            end = (r1, c0)
            segments.append(
                {"axis": "v", "start": start, "end": end, "length": r1 - r0 - 1, "cells": _segment_cells(start, end)}
            )
        else:
            top_start = (r0, c0)
            top_end = (r0, c1)
            bottom_start = (r1, c0)
            bottom_end = (r1, c1)
            left_start = (r0, c0)
            left_end = (r1, c0)
            right_start = (r0, c1)
            right_end = (r1, c1)
            segments.extend(
                [
                    {"axis": "h", "start": top_start, "end": top_end, "length": c1 - c0 - 1, "cells": _segment_cells(top_start, top_end)},
                    {"axis": "h", "start": bottom_start, "end": bottom_end, "length": c1 - c0 - 1, "cells": _segment_cells(bottom_start, bottom_end)},
                    {"axis": "v", "start": left_start, "end": left_end, "length": r1 - r0 - 1, "cells": _segment_cells(left_start, left_end)},
                    {"axis": "v", "start": right_start, "end": right_end, "length": r1 - r0 - 1, "cells": _segment_cells(right_start, right_end)},
                ]
            )

    keys = [(seg["length"], seg["start"][0], seg["start"][1]) for seg in segments]
    if len(keys) != len(set(keys)):
        return []

    all_cells = [cell for seg in segments for cell in seg["cells"]]
    if len(all_cells) != len(set(all_cells)):
        return []
    return sorted(segments, key=lambda seg: (seg["length"], seg["start"][0], seg["start"][1]))


def _add_isolated_markers(grid: np.ndarray, occupied: np.ndarray, blue_cells: set[tuple[int, int]], palette_len: int) -> bool:
    used_rows = {r for r, _ in blue_cells}
    used_cols = {c for _, c in blue_cells}
    if not blue_cells:
        return True

    max_isolated = random.randint(0, 2)
    candidates: list[tuple[int, int]] = []
    for r in range(palette_len + 1, grid.shape[0]):
        if r in used_rows:
            continue
        for c in range(palette_len + 1, grid.shape[1]):
            if c in used_cols or occupied[r, c] or grid[r, c] != BACKGROUND:
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    for r, c in candidates[:max_isolated]:
        grid[r, c] = BLUE
        used_rows.add(r)
        used_cols.add(c)
    return True


def _segment_signature(segment: dict) -> tuple:
    return (
        segment["axis"],
        segment["length"],
        segment["start"],
        tuple(segment["cells"]),
    )


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        modules, n_segments = _build_modules()
        palette_len = n_segments + random.randint(0, min(2, len(PALETTE_COLORS) - n_segments))
        height = random.randint(max(MIN_GRID, palette_len + 6), MAX_GRID)
        width = random.randint(max(MIN_GRID, palette_len + 6), MAX_GRID)

        placement = _place_modules(modules, height, width, palette_len)
        if placement is None:
            continue
        placed_modules, blue_cells = placement
        planned = _planned_segments(placed_modules)
        if len(planned) != n_segments:
            continue

        palette = random.sample(PALETTE_COLORS, palette_len)
        grid = np.zeros((height, width), dtype=int)
        for i, color in enumerate(palette):
            grid[i, i] = int(color)
        for r, c in blue_cells:
            grid[r, c] = BLUE

        occupied = np.zeros_like(grid, dtype=bool)
        for module in placed_modules:
            r0, c0, r1, c1 = module["bbox"]
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        _add_isolated_markers(grid, occupied, blue_cells, palette_len)

        expected = _paint_segments(grid, palette, planned)
        if np.array_equal(expected, grid):
            continue

        try:
            parsed_palette = _read_palette(grid)
            parsed_segments = _extract_segments(grid)
            actual = main(grid)
        except AssertionError:
            continue

        if parsed_palette != palette:
            continue
        if [_segment_signature(seg) for seg in parsed_segments] != [_segment_signature(seg) for seg in planned]:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 2")
