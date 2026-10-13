from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Generator bounds.
GRID_HEIGHT_MIN = 8
GRID_HEIGHT_MAX = 16
GRID_WIDTH_MIN = 8
GRID_WIDTH_MAX = 16
LINE_COUNT_MIN = 2
LINE_COUNT_MAX = 5
LINE_TRIES = 120
GEN_ATTEMPTS = 400
EDGE_OPTIONS = ("top", "bottom", "left", "right")
ORIENTATION_MODES = ("horizontal_only", "vertical_only", "mixed")


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_hint_bar(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid[1:, :]
    if edge == "bottom":
        return grid[:-1, :]
    if edge == "left":
        return grid[:, 1:]
    if edge == "right":
        return grid[:, :-1]
    raise AssertionError(f"unknown edge {edge}")


def _line_view(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid[0, :]
    if edge == "bottom":
        return grid[-1, :]
    if edge == "left":
        return grid[:, 0]
    if edge == "right":
        return grid[:, -1]
    raise AssertionError(f"unknown edge {edge}")


def _parse_hint_mapping(line: np.ndarray, bg: int) -> dict[int, int] | None:
    n = int(line.size)
    split = n // 2
    if split < 2 or n - split < 2:
        return None

    non_bg = [i for i, value in enumerate(line.tolist()) if int(value) != bg]
    if len(non_bg) != 4:
        return None

    first_half = sorted(i for i in non_bg if i < split)
    second_half = sorted(i for i in non_bg if i >= split)
    if len(first_half) != 2 or len(second_half) != 2:
        return None

    source_colors = [int(line[i]) for i in first_half]
    target_colors = [int(line[i]) for i in second_half]
    if len(set(source_colors)) != 2 or len(set(target_colors)) != 2:
        return None
    if set(source_colors) & set(target_colors):
        return None

    return {
        source_colors[0]: target_colors[0],
        source_colors[1]: target_colors[1],
    }


def _groups_for_color(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"missing anchors for color {color}"

    membership = {tuple(map(int, coord)): 0 for coord in coords}
    groups: list[dict[str, object]] = []

    rows = np.unique(coords[:, 0]).tolist()
    for row in rows:
        row_cols = sorted(int(c) for c in coords[coords[:, 0] == row, 1].tolist())
        if len(row_cols) < 2:
            continue
        assert len(row_cols) <= 4, "each hidden line must have 2-4 anchors"
        cells = [(int(row), c) for c in row_cols]
        groups.append(
            {
                "orientation": "H",
                "color": int(color),
                "fixed": int(row),
                "points": row_cols,
                "cells": cells,
            }
        )
        for cell in cells:
            membership[cell] += 1

    cols = np.unique(coords[:, 1]).tolist()
    for col in cols:
        col_rows = sorted(int(r) for r in coords[coords[:, 1] == col, 0].tolist())
        if len(col_rows) < 2:
            continue
        assert len(col_rows) <= 4, "each hidden line must have 2-4 anchors"
        cells = [(r, int(col)) for r in col_rows]
        groups.append(
            {
                "orientation": "V",
                "color": int(color),
                "fixed": int(col),
                "points": col_rows,
                "cells": cells,
            }
        )
        for cell in cells:
            membership[cell] += 1

    assert groups, f"expected at least one line for color {color}"
    assert all(count == 1 for count in membership.values()), "anchors must belong to exactly one hidden line"
    return groups


def _extract_structure(grid: np.ndarray, mapping: dict[int, int], bg: int) -> list[dict[str, object]]:
    visible_colors = {int(v) for v in np.unique(grid).tolist() if int(v) != bg}
    assert visible_colors == set(mapping), "interior should contain exactly the mapped anchor colors"

    groups: list[dict[str, object]] = []
    for color in sorted(mapping):
        groups.extend(_groups_for_color(grid, color))
    return groups


def _decode_task(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int], list[dict[str, object]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    candidates: list[tuple[np.ndarray, dict[int, int], list[dict[str, object]]]] = []
    for edge in EDGE_OPTIONS:
        mapping = _parse_hint_mapping(_line_view(g, edge), bg)
        if mapping is None:
            continue
        interior = _crop_hint_bar(g, edge)
        try:
            groups = _extract_structure(interior, mapping, bg)
        except AssertionError:
            continue
        candidates.append((interior, mapping, groups))

    assert len(candidates) == 1, "expected exactly one valid hint bar"
    interior, mapping, groups = candidates[0]
    return interior, mapping, groups, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer same-color horizontal/vertical anchor families, then draw each family as a full segment using the hint-bar color mapping.
    The hint bar is removed from the output.
    """
    interior, mapping, groups, bg = _decode_task(grid)
    out = np.full(interior.shape, bg, dtype=int)

    for group in groups:
        anchor_color = int(group["color"])
        draw_color = int(mapping[anchor_color])
        orientation = str(group["orientation"])
        fixed = int(group["fixed"])
        points = [int(v) for v in group["points"]]

        if orientation == "H":
            start, stop = min(points), max(points)
            assert np.all(out[fixed, start : stop + 1] == bg), "hidden lines should not overlap in output"
            out[fixed, start : stop + 1] = draw_color
        else:
            start, stop = min(points), max(points)
            assert np.all(out[start : stop + 1, fixed] == bg), "hidden lines should not overlap in output"
            out[start : stop + 1, fixed] = draw_color

    return out


def _sample_points(start: int, stop: int) -> list[int]:
    assert stop - start >= 2
    choices = list(range(start + 1, stop))
    count = random.randint(2, min(4, stop - start + 1))
    points = {start, stop}
    if count > 2:
        points.update(random.sample(choices, count - 2))
    return sorted(points)


def _sample_line(height: int, width: int, color: int, orientation: str) -> dict[str, object]:
    if orientation == "H":
        row = random.randrange(height)
        min_span = 2
        c0 = random.randint(0, width - min_span - 1)
        c1 = random.randint(c0 + min_span, width - 1)
        cols = _sample_points(c0, c1)
        anchors = [(row, c) for c in cols]
        segment = [(row, c) for c in range(c0, c1 + 1)]
        return {
            "orientation": "H",
            "color": int(color),
            "fixed": int(row),
            "points": cols,
            "anchors": anchors,
            "segment": segment,
        }

    col = random.randrange(width)
    min_span = 2
    r0 = random.randint(0, height - min_span - 1)
    r1 = random.randint(r0 + min_span, height - 1)
    rows = _sample_points(r0, r1)
    anchors = [(r, col) for r in rows]
    segment = [(r, col) for r in range(r0, r1 + 1)]
    return {
        "orientation": "V",
        "color": int(color),
        "fixed": int(col),
        "points": rows,
        "anchors": anchors,
        "segment": segment,
    }


def _place_lines(height: int, width: int, specs: list[tuple[int, str]]) -> list[dict[str, object]] | None:
    used_segments: set[tuple[int, int]] = set()
    used_anchors: set[tuple[int, int]] = set()
    used_rows: dict[int, set[int]] = {}
    used_cols: dict[int, set[int]] = {}
    lines: list[dict[str, object]] = []

    for color, orientation in specs:
        used_rows.setdefault(int(color), set())
        used_cols.setdefault(int(color), set())
        placed = False
        for _ in range(LINE_TRIES):
            line = _sample_line(height, width, color, orientation)
            anchors = [tuple(map(int, cell)) for cell in line["anchors"]]
            segment = [tuple(map(int, cell)) for cell in line["segment"]]
            anchor_rows = {r for r, _c in anchors}
            anchor_cols = {c for _r, c in anchors}

            if used_rows[int(color)] & anchor_rows:
                continue
            if used_cols[int(color)] & anchor_cols:
                continue
            if any(cell in used_anchors for cell in anchors):
                continue
            if any(cell in used_segments for cell in segment):
                continue

            used_rows[int(color)].update(anchor_rows)
            used_cols[int(color)].update(anchor_cols)
            used_anchors.update(anchors)
            used_segments.update(segment)
            lines.append(line)
            placed = True
            break

        if not placed:
            return None

    return lines


def _line_specs(anchor_colors: list[int]) -> list[tuple[int, str]]:
    n_lines = random.randint(LINE_COUNT_MIN, LINE_COUNT_MAX)
    colors = [anchor_colors[0], anchor_colors[1]]
    colors.extend(random.choice(anchor_colors) for _ in range(n_lines - 2))
    random.shuffle(colors)

    mode = random.choice(ORIENTATION_MODES)
    if mode == "horizontal_only":
        orientations = ["H"] * n_lines
    elif mode == "vertical_only":
        orientations = ["V"] * n_lines
    else:
        orientations = ["H", "V"]
        orientations.extend(random.choice(("H", "V")) for _ in range(n_lines - 2))
        random.shuffle(orientations)

    return list(zip(colors, orientations))


def _render_lines(height: int, width: int, lines: list[dict[str, object]], mapping: dict[int, int], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    for line in lines:
        color = int(line["color"])
        for r, c in line["anchors"]:
            inp[int(r), int(c)] = color

        draw_color = int(mapping[color])
        if str(line["orientation"]) == "H":
            row = int(line["fixed"])
            cols = [int(v) for v in line["points"]]
            out[row, min(cols) : max(cols) + 1] = draw_color
        else:
            col = int(line["fixed"])
            rows = [int(v) for v in line["points"]]
            out[min(rows) : max(rows) + 1, col] = draw_color

    return inp, out


def _hint_positions(length: int) -> tuple[list[int], list[int]]:
    split = length // 2
    first = sorted(random.sample(range(split), 2))
    second = sorted(random.sample(range(split, length), 2))
    return first, second


def _attach_hint_bar(grid: np.ndarray, edge: str, anchor_colors: list[int], draw_colors: list[int], bg: int) -> np.ndarray:
    order = [0, 1]
    random.shuffle(order)

    if edge in {"top", "bottom"}:
        height, width = grid.shape
        out = np.full((height + 1, width), bg, dtype=int)
        if edge == "top":
            out[1:, :] = grid
            line = out[0, :]
        else:
            out[:-1, :] = grid
            line = out[-1, :]

        left_slots, right_slots = _hint_positions(width)
        for slot, idx in enumerate(order):
            line[left_slots[slot]] = anchor_colors[idx]
            line[right_slots[slot]] = draw_colors[idx]
        return out

    height, width = grid.shape
    out = np.full((height, width + 1), bg, dtype=int)
    if edge == "left":
        out[:, 1:] = grid
        line = out[:, 0]
    else:
        out[:, :-1] = grid
        line = out[:, -1]

    top_slots, bottom_slots = _hint_positions(height)
    for slot, idx in enumerate(order):
        line[top_slots[slot]] = anchor_colors[idx]
        line[bottom_slots[slot]] = draw_colors[idx]
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_HEIGHT_MIN, GRID_HEIGHT_MAX)
    width = random.randint(GRID_WIDTH_MIN, GRID_WIDTH_MAX)
    bg = BLACK

    anchor_and_draw = random.sample(PALETTE, 4)
    anchor_colors = anchor_and_draw[:2]
    draw_colors = anchor_and_draw[2:]
    mapping = {
        anchor_colors[0]: draw_colors[0],
        anchor_colors[1]: draw_colors[1],
    }

    specs = _line_specs(anchor_colors)
    lines = _place_lines(height, width, specs)
    if lines is None:
        return None

    interior, expected = _render_lines(height, width, lines, mapping, bg)
    edge = random.choice(EDGE_OPTIONS)
    full = _attach_hint_bar(interior, edge, anchor_colors, draw_colors, bg)

    try:
        actual = main(full)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
