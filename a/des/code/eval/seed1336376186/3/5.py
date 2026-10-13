from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
CROSS = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
QUADRANTS = ("ul", "ur", "ll", "lr")
DIRECTION_BY_SIDE = {
    "top": "up",
    "bottom": "down",
    "left": "left",
    "right": "right",
}
MATCHINGS = (
    {"ul": "top", "ur": "right", "ll": "left", "lr": "bottom"},
    {"ul": "left", "ur": "top", "ll": "bottom", "lr": "right"},
)

QUAD_H_RANGE = (4, 8)
QUAD_W_RANGE = (4, 8)
GEN_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).reshape(-1)
    return int(np.bincount(values).argmax())


def _find_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_candidates = []
    for r in range(h):
        if np.all(g[r, :] == int(g[r, 0])) and int(g[r, 0]) != bg:
            row_candidates.append((int(r), int(g[r, 0])))

    col_candidates = []
    for c in range(w):
        if np.all(g[:, c] == int(g[0, c])) and int(g[0, c]) != bg:
            col_candidates.append((int(c), int(g[0, c])))

    matches = [
        (r, c, color_r)
        for r, color_r in row_candidates
        for c, color_c in col_candidates
        if color_r == color_c
    ]
    assert len(matches) == 1, "expected one full grey cross"
    return matches[0]


def _find_single_marker(line: np.ndarray, bg: int, cross: int) -> tuple[int, int]:
    indices = [int(i) for i, value in enumerate(line) if int(value) not in {bg, cross}]
    assert len(indices) == 1, "expected one marker on each border side"
    idx = indices[0]
    return idx, int(line[idx])


def _quadrant_slices(mid_r: int, mid_c: int, h: int, w: int) -> dict[str, tuple[slice, slice]]:
    assert 1 <= mid_r < h - 1 and 1 <= mid_c < w - 1
    return {
        "ul": (slice(1, mid_r), slice(1, mid_c)),
        "ur": (slice(1, mid_r), slice(mid_c + 1, w - 1)),
        "ll": (slice(mid_r + 1, h - 1), slice(1, mid_c)),
        "lr": (slice(mid_r + 1, h - 1), slice(mid_c + 1, w - 1)),
    }


def _parse_structure(grid: np.ndarray) -> tuple[int, int, int, dict[str, tuple[str, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    mid_r, mid_c, cross = _find_cross(g, bg)

    top_idx, top_color = _find_single_marker(g[0, :], bg, cross)
    bottom_idx, bottom_color = _find_single_marker(g[h - 1, :], bg, cross)
    left_idx, left_color = _find_single_marker(g[:, 0], bg, cross)
    right_idx, right_color = _find_single_marker(g[:, w - 1], bg, cross)

    assignments: dict[str, tuple[str, int]] = {}

    def assign(quadrant: str, direction: str, color: int) -> None:
        assert quadrant not in assignments, "quadrant must have exactly one governing marker"
        assignments[quadrant] = (direction, int(color))

    assign("ul" if top_idx < mid_c else "ur", "up", top_color)
    assign("ll" if bottom_idx < mid_c else "lr", "down", bottom_color)
    assign("ul" if left_idx < mid_r else "ll", "left", left_color)
    assign("ur" if right_idx < mid_r else "lr", "right", right_color)

    assert set(assignments) == set(QUADRANTS), "marker-quadrant mapping must be a bijection"
    return bg, mid_r, mid_c, assignments


def _compact(subgrid: np.ndarray, direction: str, bg: int) -> np.ndarray:
    sub = np.asarray(subgrid, dtype=int)
    h, w = sub.shape
    out = np.full((h, w), bg, dtype=int)

    if direction == "up":
        for c in range(w):
            values = [int(sub[r, c]) for r in range(h) if int(sub[r, c]) != bg]
            if values:
                out[: len(values), c] = values
    elif direction == "down":
        for c in range(w):
            values = [int(sub[r, c]) for r in range(h) if int(sub[r, c]) != bg]
            if values:
                out[h - len(values) :, c] = values
    elif direction == "left":
        for r in range(h):
            values = [int(sub[r, c]) for c in range(w) if int(sub[r, c]) != bg]
            if values:
                out[r, : len(values)] = values
    elif direction == "right":
        for r in range(h):
            values = [int(sub[r, c]) for c in range(w) if int(sub[r, c]) != bg]
            if values:
                out[r, w - len(values) :] = values
    else:
        raise AssertionError(direction)

    return out


def _transform_quadrant(subgrid: np.ndarray, direction: str, keep_color: int, bg: int) -> np.ndarray:
    packed = _compact(subgrid, direction, bg)
    packed[(packed != bg) & (packed != int(keep_color))] = bg
    return packed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse each quadrant toward the edge selected by its governing border marker,
    then erase every compacted pixel whose color does not match that marker.
    """
    g = np.asarray(grid, dtype=int)
    bg, mid_r, mid_c, assignments = _parse_structure(g)
    slices = _quadrant_slices(mid_r, mid_c, *g.shape)

    out = g.copy()
    for quadrant, (direction, keep_color) in assignments.items():
        rs, cs = slices[quadrant]
        out[rs, cs] = _transform_quadrant(g[rs, cs], direction, keep_color, bg)
    return out


def _packed_positions(length: int, count: int, direction: str) -> tuple[int, ...]:
    if direction in {"up", "left"}:
        return tuple(range(count))
    if direction in {"down", "right"}:
        return tuple(range(length - count, length))
    raise AssertionError(direction)


def _spaced_choices(length: int, count: int, direction: str) -> list[tuple[int, ...]]:
    packed = _packed_positions(length, count, direction)
    choices = []
    for combo in combinations(range(length), count):
        if combo == packed:
            continue
        if all(combo[i + 1] - combo[i] >= 2 for i in range(len(combo) - 1)):
            choices.append(tuple(map(int, combo)))
    return choices


def _can_place(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    if cell in occupied:
        return False
    for rr, cc in occupied:
        if abs(int(r) - int(rr)) + abs(int(c) - int(cc)) == 1:
            return False
    return True


def _choose_extra_cells(
    height: int,
    width: int,
    occupied: set[tuple[int, int]],
    target_count: int,
) -> list[tuple[int, int]] | None:
    for _ in range(80):
        current = set(occupied)
        picked: list[tuple[int, int]] = []
        cells = [(r, c) for r in range(height) for c in range(width)]
        random.shuffle(cells)
        for cell in cells:
            if _can_place(cell, current):
                picked.append(cell)
                current.add(cell)
                if len(picked) == target_count:
                    return picked
    return None


def _make_quadrant(
    height: int,
    width: int,
    direction: str,
    keep_color: int,
    distractor_palette: list[int],
) -> np.ndarray | None:
    if not distractor_palette:
        return None

    sub = np.full((height, width), BG, dtype=int)
    axis_len = height if direction in {"up", "down"} else width
    ortho_len = width if direction in {"up", "down"} else height

    stack_count = 2
    if axis_len >= 5 and random.random() < 0.5:
        stack_count = 3

    line_positions = _spaced_choices(axis_len, stack_count, direction)
    if not line_positions:
        return None
    positions = random.choice(line_positions)
    line_index = random.randrange(ortho_len)
    lead_distractor = random.choice(distractor_palette)

    stack_colors = [random.choice([keep_color] + distractor_palette) for _ in range(stack_count)]
    if direction in {"up", "left"}:
        stack_colors[0] = lead_distractor
        stack_colors[-1] = keep_color
    else:
        stack_colors[0] = keep_color
        stack_colors[-1] = lead_distractor

    occupied: set[tuple[int, int]] = set()
    for pos, color in zip(positions, stack_colors):
        cell = (int(pos), int(line_index)) if direction in {"up", "down"} else (int(line_index), int(pos))
        occupied.add(cell)
        sub[cell] = int(color)

    max_extra = min(4, (height * width) // 3)
    extra_count = random.randint(2, max(2, max_extra))
    extra_cells = _choose_extra_cells(height, width, occupied, extra_count)
    if extra_cells is None:
        return None

    keep_extras = random.randint(1, extra_count)
    extra_colors = [keep_color] * keep_extras
    extra_colors.extend(random.choice(distractor_palette) for _ in range(extra_count - keep_extras))
    random.shuffle(extra_colors)
    for cell, color in zip(extra_cells, extra_colors):
        occupied.add(cell)
        sub[cell] = int(color)

    keep_count = int(np.count_nonzero(sub == int(keep_color)))
    distractor_count = int(np.count_nonzero((sub != BG) & (sub != int(keep_color))))
    if keep_count < 2 or distractor_count < 2:
        return None

    transformed = _transform_quadrant(sub, direction, keep_color, BG)
    if not np.any(transformed == int(keep_color)):
        return None
    if np.array_equal(sub, transformed):
        return None
    if int(np.count_nonzero(transformed == int(keep_color))) != keep_count:
        return None

    return sub


def _quadrant_offsets(quad_h: int, quad_w: int) -> dict[str, tuple[int, int]]:
    mid_r = quad_h + 1
    mid_c = quad_w + 1
    return {
        "ul": (1, 1),
        "ur": (1, mid_c + 1),
        "ll": (mid_r + 1, 1),
        "lr": (mid_r + 1, mid_c + 1),
    }


def _marker_position_for_quadrant(
    quadrant: str,
    side: str,
    mid_r: int,
    mid_c: int,
    h: int,
    w: int,
) -> tuple[int, int]:
    if side == "top":
        cols = range(1, mid_c) if quadrant == "ul" else range(mid_c + 1, w - 1)
        return 0, int(random.choice(list(cols)))
    if side == "bottom":
        cols = range(1, mid_c) if quadrant == "ll" else range(mid_c + 1, w - 1)
        return h - 1, int(random.choice(list(cols)))
    if side == "left":
        rows = range(1, mid_r) if quadrant == "ul" else range(mid_r + 1, h - 1)
        return int(random.choice(list(rows))), 0
    if side == "right":
        rows = range(1, mid_r) if quadrant == "ur" else range(mid_r + 1, h - 1)
        return int(random.choice(list(rows))), w - 1
    raise AssertionError(side)


def _gen_attempt() -> np.ndarray | None:
    quad_h = random.randint(*QUAD_H_RANGE)
    quad_w = random.randint(*QUAD_W_RANGE)
    h = 2 * quad_h + 3
    w = 2 * quad_w + 3
    mid_r = quad_h + 1
    mid_c = quad_w + 1

    grid = np.full((h, w), BG, dtype=int)
    grid[mid_r, :] = CROSS
    grid[:, mid_c] = CROSS

    quad_to_side = random.choice(MATCHINGS)
    marker_colors = random.sample(PALETTE, 4)
    quad_colors = {quad: int(color) for quad, color in zip(QUADRANTS, marker_colors)}

    offsets = _quadrant_offsets(quad_h, quad_w)
    for quadrant in QUADRANTS:
        side = quad_to_side[quadrant]
        direction = DIRECTION_BY_SIDE[side]
        keep_color = quad_colors[quadrant]
        distractors = [color for color in PALETTE if color != keep_color]
        patch = _make_quadrant(quad_h, quad_w, direction, keep_color, distractors)
        if patch is None:
            return None
        r0, c0 = offsets[quadrant]
        grid[r0 : r0 + quad_h, c0 : c0 + quad_w] = patch

    used_marker_cells: set[tuple[int, int]] = set()
    for quadrant in QUADRANTS:
        side = quad_to_side[quadrant]
        marker_cell = _marker_position_for_quadrant(quadrant, side, mid_r, mid_c, h, w)
        if marker_cell in used_marker_cells:
            return None
        used_marker_cells.add(marker_cell)
        grid[marker_cell] = quad_colors[quadrant]

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    if not np.array_equal(out[mid_r, :], grid[mid_r, :]):
        return None
    if not np.array_equal(out[:, mid_c], grid[:, mid_c]):
        return None

    slices = _quadrant_slices(mid_r, mid_c, h, w)
    for quadrant in QUADRANTS:
        rs, cs = slices[quadrant]
        keep_color = quad_colors[quadrant]
        out_patch = out[rs, cs]
        values = {int(v) for v in np.unique(out_patch)}
        if not values <= {BG, keep_color}:
            return None
        if int(np.count_nonzero(out_patch == keep_color)) == 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
