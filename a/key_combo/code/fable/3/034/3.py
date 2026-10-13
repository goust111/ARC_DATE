from __future__ import annotations

import random
from typing import Iterable

import numpy as np


GREY = 5
PALETTE = tuple(range(10))

SOURCE_SIZE_RANGE = (5, 8)
SOURCE_BBOX_HEIGHT_RANGE = (3, 5)
SOURCE_BBOX_WIDTH_RANGE = (3, 6)
MARKER_COUNT_RANGE = (3, 4)
GAP_VALUES = (3, 4, 5)
MAX_SOURCE_ATTEMPTS = 200
MAX_LAYOUT_ATTEMPTS = 300
MAX_GENERATION_ATTEMPTS = 500


Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalize(cells: Iterable[Coord]) -> tuple[Coord, ...]:
    cells = list(cells)
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_size(cells: Iterable[Coord]) -> tuple[int, int]:
    cells = list(cells)
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _row_col_counts(cells: Iterable[Coord]) -> tuple[dict[int, int], dict[int, int]]:
    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r, c in cells:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1
    return row_counts, col_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the source object through each unblocked singleton marker and erase the blocked markers.

    Input structure:
    - A dominant background surrounds one multi-pixel source object, several distinct singleton markers, and zero or more grey blocker bars.
    - Each marker lies on exactly one shared row or one shared column with the source, and that row or column contains exactly one aligned source pixel.
    - Some marker-to-source sight lines may contain grey bar cells or other non-background cells strictly between the endpoints.

    Rule:
    - Preserve the source object and every grey blocker bar.
    - Remove all singleton markers from the output, then test each marker against its unique aligned source pixel.
    - If every interior cell on that straight segment is background, reflect the whole source object across the midpoint axis fixed by the pair and recolor the reflected copy with the marker color; otherwise keep the marker erased and add no copy.

    Key insight:
    - A marker is not a command by itself: it only triggers the usual midpoint reflection when it has unobstructed line of sight to its aligned source pixel.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    if not colors:
        return g.copy()

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    blocker_color = GREY if GREY in counts and GREY != bg else None

    source_candidates = [color for color in colors if color != blocker_color and counts[color] > 1]
    if source_candidates:
        source_color = max(source_candidates, key=lambda color: counts[color])
    else:
        return g.copy()

    marker_colors = [
        color
        for color in colors
        if color not in {source_color, blocker_color} and counts[color] == 1
    ]

    source_coords = np.argwhere(g == source_color)
    protected = g == source_color
    if blocker_color is not None:
        protected |= g == blocker_color

    out = g.copy()
    marker_positions: dict[int, Coord] = {}
    for color in marker_colors:
        marker_r, marker_c = map(int, np.argwhere(g == color)[0])
        marker_positions[color] = (marker_r, marker_c)
        out[marker_r, marker_c] = bg

    height, width = g.shape
    for marker_color in marker_colors:
        marker_r, marker_c = marker_positions[marker_color]

        row_anchors = source_coords[source_coords[:, 0] == marker_r]
        col_anchors = source_coords[source_coords[:, 1] == marker_c]
        use_horizontal = len(row_anchors) == 1 and len(col_anchors) != 1
        use_vertical = len(col_anchors) == 1 and len(row_anchors) != 1
        if not (use_horizontal or use_vertical):
            continue

        if use_horizontal:
            anchor_r, anchor_c = map(int, row_anchors[0])
            step = 1 if marker_c > anchor_c else -1
            if any(g[anchor_r, c] != bg for c in range(anchor_c + step, marker_c, step)):
                continue
            col_sum = anchor_c + marker_c
            targets = [(int(r), int(col_sum - c)) for r, c in source_coords]
        else:
            anchor_r, anchor_c = map(int, col_anchors[0])
            step = 1 if marker_r > anchor_r else -1
            if any(g[r, anchor_c] != bg for r in range(anchor_r + step, marker_r, step)):
                continue
            row_sum = anchor_r + marker_r
            targets = [(int(row_sum - r), int(c)) for r, c in source_coords]

        if any(r < 0 or r >= height or c < 0 or c >= width for r, c in targets):
            continue

        for r, c in targets:
            if protected[r, c]:
                continue
            out[r, c] = marker_color

    return out


def _marker_options(source: tuple[Coord, ...]) -> list[dict[str, object]]:
    source_set = set(source)
    row_counts, col_counts = _row_col_counts(source)
    options: list[dict[str, object]] = []

    for anchor_r, anchor_c in source:
        if row_counts[anchor_r] == 1:
            for side in (-1, 1):
                for gap in GAP_VALUES:
                    distance = gap + 1
                    marker = (anchor_r, anchor_c + side * distance)
                    line_cells = tuple(
                        (anchor_r, anchor_c + side * step)
                        for step in range(1, gap + 1)
                    )
                    target = tuple(
                        sorted((r, 2 * anchor_c + side * distance - c) for r, c in source)
                    )
                    target_set = set(target)
                    if target_set & source_set:
                        continue
                    options.append(
                        {
                            "axis": "H",
                            "marker": marker,
                            "line_cells": line_cells,
                            "target": target,
                        }
                    )

        if col_counts[anchor_c] == 1:
            for side in (-1, 1):
                for gap in GAP_VALUES:
                    distance = gap + 1
                    marker = (anchor_r + side * distance, anchor_c)
                    line_cells = tuple(
                        (anchor_r + side * step, anchor_c)
                        for step in range(1, gap + 1)
                    )
                    target = tuple(
                        sorted((2 * anchor_r + side * distance - r, c) for r, c in source)
                    )
                    target_set = set(target)
                    if target_set & source_set:
                        continue
                    options.append(
                        {
                            "axis": "V",
                            "marker": marker,
                            "line_cells": line_cells,
                            "target": target,
                        }
                    )

    return options


def _grow_source_shape() -> tuple[Coord, ...] | None:
    for _ in range(MAX_SOURCE_ATTEMPTS):
        target_size = random.randint(*SOURCE_SIZE_RANGE)
        cells = {(0, 0)}

        while len(cells) < target_size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    candidate = _normalize(cells | {(nr, nc)})
                    h, w = _shape_size(candidate)
                    if h <= SOURCE_BBOX_HEIGHT_RANGE[1] and w <= SOURCE_BBOX_WIDTH_RANGE[1]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_size:
            continue
        source = _normalize(cells)
        height, width = _shape_size(source)
        if not (
            SOURCE_BBOX_HEIGHT_RANGE[0] <= height <= SOURCE_BBOX_HEIGHT_RANGE[1]
            and SOURCE_BBOX_WIDTH_RANGE[0] <= width <= SOURCE_BBOX_WIDTH_RANGE[1]
        ):
            continue

        options = _marker_options(source)
        if len(options) < MARKER_COUNT_RANGE[1]:
            continue
        if {str(option["axis"]) for option in options} != {"H", "V"}:
            continue
        return source

    return None


def _selected_options_feasible(
    source: tuple[Coord, ...],
    selected: list[dict[str, object]],
    visible_indices: set[int],
) -> bool:
    source_set = set(source)
    marker_positions = [tuple(option["marker"]) for option in selected]
    if len(set(marker_positions)) != len(marker_positions):
        return False

    marker_set = set(marker_positions)
    for index, option in enumerate(selected):
        line_set = set(option["line_cells"])
        if line_set & (marker_set - {tuple(option["marker"])}) or line_set & source_set:
            return False
        if index in visible_indices:
            target_set = set(option["target"])
            if target_set & source_set:
                return False
            if target_set & (marker_set - {tuple(option["marker"])}):
                return False

    visible_union: set[Coord] = set()
    for index in sorted(visible_indices):
        target_set = set(selected[index]["target"])
        if target_set & visible_union:
            return False
        visible_union |= target_set
    return True


def _choose_blockers(
    source: tuple[Coord, ...],
    selected: list[dict[str, object]],
    visible_indices: set[int],
) -> dict[int, tuple[Coord, ...]] | None:
    occupied = set(source)
    occupied |= {tuple(option["marker"]) for option in selected}
    for index in visible_indices:
        occupied |= set(selected[index]["target"])

    blocked_indices = [index for index in range(len(selected)) if index not in visible_indices]
    random.shuffle(blocked_indices)
    blockers: dict[int, tuple[Coord, ...]] = {}

    for index in blocked_indices:
        line_cells = list(selected[index]["line_cells"])
        candidates: list[tuple[Coord, ...]] = []
        for length in range(min(3, len(line_cells)), 1, -1):
            for start in range(0, len(line_cells) - length + 1):
                segment = tuple(line_cells[start : start + length])
                if set(segment) & occupied:
                    continue
                candidates.append(segment)
        if not candidates:
            return None
        blockers[index] = random.choice(candidates)
        occupied |= set(blockers[index])

    return blockers


def _choose_marker_layout(
    source: tuple[Coord, ...],
    options: list[dict[str, object]],
) -> tuple[list[dict[str, object]], set[int], dict[int, tuple[Coord, ...]]] | None:
    if len(options) < MARKER_COUNT_RANGE[0]:
        return None

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        marker_count = random.randint(*MARKER_COUNT_RANGE)
        if len(options) < marker_count:
            continue
        selected = random.sample(options, marker_count)
        if {str(option["axis"]) for option in selected} != {"H", "V"}:
            continue

        visible_count = random.randint(2, marker_count - 1)
        visible_indices = set(random.sample(range(marker_count), visible_count))
        if {str(selected[index]["axis"]) for index in visible_indices} != {"H", "V"}:
            continue
        if not _selected_options_feasible(source, selected, visible_indices):
            continue

        blockers = _choose_blockers(source, selected, visible_indices)
        if blockers is None:
            continue
        return selected, visible_indices, blockers

    return None


def _all_cells(
    source: tuple[Coord, ...],
    selected: list[dict[str, object]],
    visible_indices: set[int],
    blockers: dict[int, tuple[Coord, ...]],
) -> set[Coord]:
    cells = set(source)
    for option in selected:
        cells.add(tuple(option["marker"]))
    for segment in blockers.values():
        cells |= set(segment)
    for index in visible_indices:
        cells |= set(selected[index]["target"])
    return cells


def _bbox(cells: Iterable[Coord]) -> tuple[int, int, int, int]:
    cells = list(cells)
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _margin_pair(design_size: int) -> tuple[int, int] | None:
    spare = 30 - design_size
    if spare < 2:
        return None
    first = random.randint(1, min(4, spare - 1))
    second = random.randint(1, min(4, spare - first))
    return first, second


def _choose_colors(marker_count: int) -> tuple[int, int, list[int]]:
    bg = random.choice([color for color in PALETTE if color != GREY])
    source_color = random.choice([color for color in PALETTE if color not in (bg, GREY)])
    marker_colors = random.sample(
        [color for color in PALETTE if color not in (bg, GREY, source_color)],
        marker_count,
    )
    return bg, source_color, marker_colors


def _render_pair(
    source: tuple[Coord, ...],
    selected: list[dict[str, object]],
    visible_indices: set[int],
    blockers: dict[int, tuple[Coord, ...]],
) -> tuple[np.ndarray, np.ndarray] | None:
    design_cells = _all_cells(source, selected, visible_indices, blockers)
    r0, c0, r1, c1 = _bbox(design_cells)
    row_margins = _margin_pair(r1 - r0 + 1)
    col_margins = _margin_pair(c1 - c0 + 1)
    if row_margins is None or col_margins is None:
        return None

    top, bottom = row_margins
    left, right = col_margins
    rows = (r1 - r0 + 1) + top + bottom
    cols = (c1 - c0 + 1) + left + right
    row_shift = top - r0
    col_shift = left - c0

    bg, source_color, marker_colors = _choose_colors(len(selected))
    inp = np.full((rows, cols), bg, dtype=int)

    for r, c in source:
        inp[r + row_shift, c + col_shift] = source_color
    for segment in blockers.values():
        for r, c in segment:
            inp[r + row_shift, c + col_shift] = GREY

    marker_positions = []
    for option, marker_color in zip(selected, marker_colors):
        marker_r, marker_c = tuple(option["marker"])
        rr, cc = marker_r + row_shift, marker_c + col_shift
        if inp[rr, cc] != bg:
            return None
        inp[rr, cc] = marker_color
        marker_positions.append((rr, cc))

    out = inp.copy()
    for rr, cc in marker_positions:
        out[rr, cc] = bg

    protected = (inp == source_color) | (inp == GREY)
    for index, (option, marker_color) in enumerate(zip(selected, marker_colors)):
        if index not in visible_indices:
            continue
        for r, c in option["target"]:
            rr, cc = r + row_shift, c + col_shift
            if protected[rr, cc] or out[rr, cc] not in (bg, marker_color):
                return None
            out[rr, cc] = marker_color

    for index, option in enumerate(selected):
        line_values = [int(inp[r + row_shift, c + col_shift]) for r, c in option["line_cells"]]
        if index in visible_indices:
            if any(value != bg for value in line_values):
                return None
        else:
            if GREY not in line_values or any(value not in (bg, GREY) for value in line_values):
                return None

    if get_bg(inp) != bg:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        source = _grow_source_shape()
        if source is None:
            continue

        options = _marker_options(source)
        layout = _choose_marker_layout(source, options)
        if layout is None:
            continue
        selected, visible_indices, blockers = layout

        rendered = _render_pair(source, selected, visible_indices, blockers)
        if rendered is None:
            continue
        inp, out = rendered

        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, out):
            return inp

    raise RuntimeError("failed to generate a valid input for task 3")
