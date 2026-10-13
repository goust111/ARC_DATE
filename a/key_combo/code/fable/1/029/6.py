from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


CORNERS = ("tl", "tr", "bl", "br")


def _perimeter_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    out: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        out.append((r0, c))
        out.append((r1, c))
    for r in range(r0 + 1, r1):
        out.append((r, c0))
        out.append((r, c1))
    return out


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], set[tuple[int, int]]]:
    candidates = []
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in components:
        box = tuple(map(int, comp["bbox"]))
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        expected = set(_perimeter_coords(box))
        coords = {tuple(map(int, rc)) for rc in comp["coords"]}
        if coords == expected:
            candidates.append((int(comp["color"]), box, coords))
    assert len(candidates) == 1, "expected exactly one hollow rectangular frame"
    return candidates[0]


def _split_contents(
    grid: np.ndarray,
    bg: int,
    frame_coords: set[tuple[int, int]],
    old_box: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    r0, c0, r1, c1 = old_box
    patterns: list[tuple[int, int, int]] = []
    controls: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(grid != int(bg)):
        rr, cc = int(r), int(c)
        if (rr, cc) in frame_coords:
            continue
        value = int(grid[rr, cc])
        if r0 < rr < r1 and c0 < cc < c1:
            patterns.append((rr, cc, value))
        else:
            controls.append((rr, cc, value))
    assert patterns, "expected interior pattern pixels"
    assert controls, "expected exterior control pixels"
    return patterns, controls


def _control_matches_box(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, box)
    hits = 0
    hits += int(r == r0 - 1 and c0 <= c <= c1)
    hits += int(r == r1 + 1 and c0 <= c <= c1)
    hits += int(c == c0 - 1 and r0 <= r <= r1)
    hits += int(c == c1 + 1 and r0 <= r <= r1)
    return hits == 1


def _find_target_box(
    old_box: tuple[int, int, int, int],
    controls: list[tuple[int, int, int]],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    old_r0, old_c0, old_r1, old_c1 = map(int, old_box)

    r0_candidates = {old_r0}
    c0_candidates = {old_c0}
    r1_candidates = {old_r1}
    c1_candidates = {old_c1}
    for r, c, _color in controls:
        if 0 <= r + 1 <= old_r0:
            r0_candidates.add(int(r + 1))
        if old_r1 <= r - 1 < h:
            r1_candidates.add(int(r - 1))
        if 0 <= c + 1 <= old_c0:
            c0_candidates.add(int(c + 1))
        if old_c1 <= c - 1 < w:
            c1_candidates.add(int(c - 1))

    matches = []
    for r0 in sorted(r0_candidates):
        for c0 in sorted(c0_candidates):
            for r1 in sorted(r1_candidates):
                for c1 in sorted(c1_candidates):
                    if not (0 <= r0 <= old_r0 and 0 <= c0 <= old_c0):
                        continue
                    if not (old_r1 <= r1 < h and old_c1 <= c1 < w):
                        continue
                    box = (int(r0), int(c0), int(r1), int(c1))
                    if all(_control_matches_box(r, c, box) for r, c, _color in controls):
                        matches.append(box)

    unique = sorted(set(matches))
    assert len(unique) == 1, "expected a unique stretched rectangle"
    return unique[0]


def _corner_owner_and_offset(
    box: tuple[int, int, int, int], r: int, c: int
) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    top_gap = int(r - r0)
    bottom_gap = int(r1 - r)
    left_gap = int(c - c0)
    right_gap = int(c1 - c)
    assert top_gap != bottom_gap and left_gap != right_gap, "pattern pixel must have a unique nearest corner"

    if top_gap < bottom_gap and left_gap < right_gap:
        return "tl", (top_gap, left_gap)
    if top_gap < bottom_gap and right_gap < left_gap:
        return "tr", (top_gap, right_gap)
    if bottom_gap < top_gap and left_gap < right_gap:
        return "bl", (bottom_gap, left_gap)
    if bottom_gap < top_gap and right_gap < left_gap:
        return "br", (bottom_gap, right_gap)
    raise AssertionError("unreachable corner classification")


def _move_from_corner(
    new_box: tuple[int, int, int, int], owner: str, offset: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, new_box)
    dr, dc = map(int, offset)
    if owner == "tl":
        return r0 + dr, c0 + dc
    if owner == "tr":
        return r0 + dr, c1 - dc
    if owner == "bl":
        return r1 - dr, c0 + dc
    if owner == "br":
        return r1 - dr, c1 - dc
    raise ValueError(f"unknown corner owner: {owner}")


def _render_output(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    old_box: tuple[int, int, int, int],
    new_box: tuple[int, int, int, int],
    pattern_cells: list[tuple[int, int, int]],
    target_cells: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)

    for r, c in _perimeter_coords(old_box):
        out[r, c] = int(bg)
    for r, c, _color in pattern_cells:
        out[r, c] = int(bg)

    for r, c in _perimeter_coords(new_box):
        out[r, c] = int(frame_color)

    occupied: dict[tuple[int, int], int] = {}
    for r, c, color in target_cells:
        assert (r, c) not in occupied or occupied[(r, c)] == int(color), "pattern collision must agree in color"
        occupied[(r, c)] = int(color)

    for (r, c), color in occupied.items():
        out[r, c] = color
    return out


def _solve_with_parts(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    old_box: tuple[int, int, int, int],
    new_box: tuple[int, int, int, int],
    pattern_cells: list[tuple[int, int, int]],
) -> np.ndarray:
    moved = []
    new_r0, new_c0, new_r1, new_c1 = new_box
    for r, c, color in pattern_cells:
        owner, offset = _corner_owner_and_offset(old_box, r, c)
        nr, nc = _move_from_corner(new_box, owner, offset)
        assert new_r0 < nr < new_r1 and new_c0 < nc < new_c1, "pattern pixels must remain inside the new frame"
        moved.append((nr, nc, color))
    return _render_output(grid, bg, frame_color, old_box, new_box, pattern_cells, moved)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch a hollow frame to the marker-defined rectangle while letting each interior pattern pixel follow its nearest original corner.

    Input structure:
    - A dominant background contains one 1-cell-thick hollow rectangular frame.
    - The frame encloses scattered interior pattern pixels whose colors differ from the frame color.
    - Isolated control pixels lie outside the frame and uniquely determine a larger target rectangle.

    Rule:
    - Find the unique rectangle that contains the original frame and places every control pixel exactly one cell outside one of its sides.
    - Erase the old frame, draw the new frame, and preserve the control pixels.
    - For each interior pattern pixel, identify the unique nearest original corner by row/column distance, keep that pixel's offset from that corner, and redraw it from the corresponding new corner.

    Key insight:
    - The border stretches, but the interior pattern does not scale: it is partitioned by nearest-corner ownership, and each partition is translated by its own corner shift.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frame_color, old_box, frame_coords = _find_frame(g, bg)
    pattern_cells, controls = _split_contents(g, bg, frame_coords, old_box)
    new_box = _find_target_box(old_box, controls, g.shape)
    return _solve_with_parts(g, bg, frame_color, old_box, new_box, pattern_cells)


GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
OLD_HEIGHT_RANGE = (7, 12)
OLD_WIDTH_RANGE = (7, 12)
TOTAL_EXPAND_RANGE = (2, 10)
MARKERS_PER_SIDE_RANGE = (1, 2)
PATTERNS_PER_CORNER_RANGE = (1, 2)
GEN_RETRIES = 500


def _sample_geometry() -> tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)

    old_h_max = min(OLD_HEIGHT_RANGE[1], h - 6)
    old_w_max = min(OLD_WIDTH_RANGE[1], w - 6)
    if old_h_max < OLD_HEIGHT_RANGE[0] or old_w_max < OLD_WIDTH_RANGE[0]:
        return None

    old_h = random.randint(OLD_HEIGHT_RANGE[0], old_h_max)
    old_w = random.randint(OLD_WIDTH_RANGE[0], old_w_max)
    extra_h_max = min(TOTAL_EXPAND_RANGE[1], h - old_h - 4)
    extra_w_max = min(TOTAL_EXPAND_RANGE[1], w - old_w - 4)
    if extra_h_max < TOTAL_EXPAND_RANGE[0] or extra_w_max < TOTAL_EXPAND_RANGE[0]:
        return None

    extra_h = random.randint(TOTAL_EXPAND_RANGE[0], extra_h_max)
    extra_w = random.randint(TOTAL_EXPAND_RANGE[0], extra_w_max)
    top_expand = random.randint(1, extra_h - 1)
    bottom_expand = extra_h - top_expand
    left_expand = random.randint(1, extra_w - 1)
    right_expand = extra_w - left_expand

    new_h = old_h + extra_h
    new_w = old_w + extra_w
    if h - new_h - 2 < 2 or w - new_w - 2 < 2:
        return None

    new_r0 = random.randint(2, h - new_h - 2)
    new_c0 = random.randint(2, w - new_w - 2)
    new_r1 = new_r0 + new_h - 1
    new_c1 = new_c0 + new_w - 1

    old_r0 = new_r0 + top_expand
    old_c0 = new_c0 + left_expand
    old_r1 = new_r1 - bottom_expand
    old_c1 = new_c1 - right_expand

    return h, w, (old_r0, old_c0, old_r1, old_c1), (new_r0, new_c0, new_r1, new_c1)


def _group_candidates(box: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    groups = {corner: [] for corner in CORNERS}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            try:
                owner, _offset = _corner_owner_and_offset(box, r, c)
            except AssertionError:
                continue
            groups[owner].append((r, c))
    return groups


def _far_enough(cell: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen)


def _pick_pattern_positions(box: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    groups = _group_candidates(box)
    chosen: list[tuple[int, int]] = []
    for corner in random.sample(list(CORNERS), len(CORNERS)):
        need = random.randint(*PATTERNS_PER_CORNER_RANGE)
        options = groups[corner][:]
        random.shuffle(options)
        count = 0
        for cell in options:
            if _far_enough(cell, chosen):
                chosen.append(cell)
                count += 1
                if count == need:
                    break
        if count < need:
            return None
    return chosen


def _pick_side_positions(start: int, end: int, count: int) -> list[int] | None:
    options = list(range(start, end + 1))
    random.shuffle(options)
    picked: list[int] = []
    for value in options:
        if all(abs(value - other) > 1 for other in picked):
            picked.append(value)
            if len(picked) == count:
                return sorted(picked)
    return None


def _build_controls(
    new_box: tuple[int, int, int, int], control_color: int
) -> list[tuple[int, int, int]] | None:
    r0, c0, r1, c1 = new_box
    controls: list[tuple[int, int, int]] = []

    top_cols = _pick_side_positions(c0 + 1, c1 - 1, random.randint(*MARKERS_PER_SIDE_RANGE))
    bottom_cols = _pick_side_positions(c0 + 1, c1 - 1, random.randint(*MARKERS_PER_SIDE_RANGE))
    left_rows = _pick_side_positions(r0 + 1, r1 - 1, random.randint(*MARKERS_PER_SIDE_RANGE))
    right_rows = _pick_side_positions(r0 + 1, r1 - 1, random.randint(*MARKERS_PER_SIDE_RANGE))
    if None in (top_cols, bottom_cols, left_rows, right_rows):
        return None

    controls.extend((r0 - 1, c, control_color) for c in top_cols or [])
    controls.extend((r1 + 1, c, control_color) for c in bottom_cols or [])
    controls.extend((r, c0 - 1, control_color) for r in left_rows or [])
    controls.extend((r, c1 + 1, control_color) for r in right_rows or [])
    return controls


def _build_pattern_cells(
    old_box: tuple[int, int, int, int], available_colors: list[int]
) -> list[tuple[int, int, int]] | None:
    positions = _pick_pattern_positions(old_box)
    if positions is None:
        return None

    palette_size = random.randint(2, min(4, len(available_colors)))
    palette = random.sample(available_colors, palette_size)
    random.shuffle(positions)
    pattern_cells = [(r, c, palette[i % palette_size]) for i, (r, c) in enumerate(positions)]
    assert len({color for _r, _c, color in pattern_cells}) >= 2
    return pattern_cells


def _build_input_grid(
    shape: tuple[int, int],
    bg: int,
    frame_color: int,
    control_color: int,
    old_box: tuple[int, int, int, int],
    pattern_cells: list[tuple[int, int, int]],
    controls: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for r, c in _perimeter_coords(old_box):
        grid[r, c] = int(frame_color)
    for r, c, color in pattern_cells:
        grid[r, c] = int(color)
    for r, c, color in controls:
        grid[r, c] = int(color)
    return grid


def _render_static_wrong(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    old_box: tuple[int, int, int, int],
    new_box: tuple[int, int, int, int],
    pattern_cells: list[tuple[int, int, int]],
) -> np.ndarray:
    return _render_output(grid, bg, frame_color, old_box, new_box, pattern_cells, pattern_cells)


def _render_translation_wrong(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    old_box: tuple[int, int, int, int],
    new_box: tuple[int, int, int, int],
    pattern_cells: list[tuple[int, int, int]],
) -> np.ndarray:
    dr = int(new_box[0] - old_box[0])
    dc = int(new_box[1] - old_box[1])
    moved = [(r + dr, c + dc, color) for r, c, color in pattern_cells]
    return _render_output(grid, bg, frame_color, old_box, new_box, pattern_cells, moved)


def _scale_position(
    old_box: tuple[int, int, int, int], new_box: tuple[int, int, int, int], r: int, c: int
) -> tuple[int, int]:
    old_r0, old_c0, old_r1, old_c1 = old_box
    new_r0, new_c0, new_r1, new_c1 = new_box

    old_row_span = old_r1 - old_r0 - 2
    old_col_span = old_c1 - old_c0 - 2
    new_row_span = new_r1 - new_r0 - 2
    new_col_span = new_c1 - new_c0 - 2

    row_index = int(r - (old_r0 + 1))
    col_index = int(c - (old_c0 + 1))
    new_row_index = int(round(row_index * new_row_span / old_row_span))
    new_col_index = int(round(col_index * new_col_span / old_col_span))
    return new_r0 + 1 + new_row_index, new_c0 + 1 + new_col_index


def _render_scaled_wrong(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    old_box: tuple[int, int, int, int],
    new_box: tuple[int, int, int, int],
    pattern_cells: list[tuple[int, int, int]],
) -> np.ndarray:
    moved = [(*_scale_position(old_box, new_box, r, c), color) for r, c, color in pattern_cells]
    return _render_output(grid, bg, frame_color, old_box, new_box, pattern_cells, moved)


def _controls_are_isolated(grid: np.ndarray, controls: list[tuple[int, int, int]], bg: int) -> bool:
    h, w = grid.shape
    control_set = {(r, c) for r, c, _color in controls}
    for r, c, _color in controls:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if (rr, cc) == (r, c):
                    continue
                if (rr, cc) in control_set:
                    return False
                if int(grid[rr, cc]) != int(bg):
                    return False
    return True


def _validate_example(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    frame_color: int,
    old_box: tuple[int, int, int, int],
    new_box: tuple[int, int, int, int],
    pattern_cells: list[tuple[int, int, int]],
    controls: list[tuple[int, int, int]],
) -> bool:
    if not _controls_are_isolated(inp, controls, bg):
        return False

    owners = []
    for r, c, _color in pattern_cells:
        owner, _offset = _corner_owner_and_offset(old_box, r, c)
        owners.append(owner)
    if set(owners) != set(CORNERS):
        return False

    inferred_box = _find_target_box(old_box, controls, inp.shape)
    if inferred_box != new_box:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    if np.array_equal(expected, _render_static_wrong(inp, bg, frame_color, old_box, new_box, pattern_cells)):
        return False
    if np.array_equal(expected, _render_translation_wrong(inp, bg, frame_color, old_box, new_box, pattern_cells)):
        return False
    if np.array_equal(expected, _render_scaled_wrong(inp, bg, frame_color, old_box, new_box, pattern_cells)):
        return False
    return not np.array_equal(inp, expected)


def _gen_attempt() -> np.ndarray | None:
    geometry = _sample_geometry()
    if geometry is None:
        return None
    h, w, old_box, new_box = geometry

    colors = list(range(10))
    bg, frame_color, control_color = random.sample(colors, 3)
    pattern_palette = [color for color in colors if color not in {bg, frame_color, control_color}]
    pattern_cells = _build_pattern_cells(old_box, pattern_palette)
    controls = _build_controls(new_box, control_color)
    if pattern_cells is None or controls is None:
        return None

    inp = _build_input_grid((h, w), bg, frame_color, control_color, old_box, pattern_cells, controls)
    expected = _solve_with_parts(inp, bg, frame_color, old_box, new_box, pattern_cells)
    if not _validate_example(inp, expected, bg, frame_color, old_box, new_box, pattern_cells, controls):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
