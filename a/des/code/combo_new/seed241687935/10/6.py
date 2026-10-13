from __future__ import annotations

import random

import numpy as np


BG = 0
MAROON = 9
MAX_GENERATION_ATTEMPTS = 300
RING_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
DIRECTIONS = ("up", "right", "down", "left")
RING_OFFSETS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (2, 0),
    (3, 0),
    (1, 4),
    (2, 4),
    (3, 4),
    (4, 1),
    (4, 2),
    (4, 3),
)
POINTER_OFFSETS = {
    "up": (-1, 2),
    "right": (2, 5),
    "down": (5, 2),
    "left": (2, -1),
}
TOTAL_BBOX = {
    "up": (6, 5),
    "right": (5, 6),
    "down": (6, 5),
    "left": (5, 6),
}
INNER_POSITIONS = tuple((r, c) for r in range(3) for c in range(3))


def _ring_cells(r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + dr, c0 + dc) for dr, dc in RING_OFFSETS]


def _pointer_cell(r0: int, c0: int, direction: str) -> tuple[int, int]:
    dr, dc = POINTER_OFFSETS[direction]
    return r0 + dr, c0 + dc


def _rotate_direction(direction: str, turns: int) -> str:
    idx = DIRECTIONS.index(direction)
    return DIRECTIONS[(idx + turns) % 4]


def _detect_object(grid: np.ndarray, color: int) -> tuple[int, int, str, int]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
    assert cells, f"missing object for color {color}"

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)
    bbox_h = r_max - r_min + 1
    bbox_w = c_max - c_min + 1

    if (bbox_h, bbox_w) == (6, 5):
        candidates = [("up", r_min + 1, c_min), ("down", r_min, c_min)]
    elif (bbox_h, bbox_w) == (5, 6):
        candidates = [("left", r_min, c_min + 1), ("right", r_min, c_min)]
    else:
        raise AssertionError(f"color {color} does not form a valid pointed ring")

    matches: list[tuple[int, int, str, int]] = []
    for direction, r0, c0 in candidates:
        expected = set(_ring_cells(r0, c0))
        expected.add(_pointer_cell(r0, c0, direction))
        if cells != expected:
            continue

        inner = grid[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4]
        if inner.shape != (3, 3):
            continue
        if not np.isin(inner, (BG, MAROON)).all():
            continue

        mark_count = int(np.count_nonzero(inner == MAROON))
        if 1 <= mark_count <= 3:
            matches.append((r0, c0, direction, mark_count))

    assert len(matches) == 1, f"color {color} must match exactly one pointed ring"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each pointed ring clockwise by a number of quarter-turns encoded by its inner maroon marks.

    Input structure:
    - The background is black.
    - Each object is a single-color 5x5 cornerless ring: the 12 edge-midpoint cells of a 5x5 square are filled, while the four corners and the 3x3 interior remain non-ring.
    - Exactly one extra pointer pixel of the same color protrudes outward from the midpoint of one ring side, indicating the current facing direction.
    - The 3x3 interior of each ring contains 1 to 3 maroon marker pixels; different rings use different non-maroon colors and do not overlap.

    Rule:
    - For each ring, count the number K of maroon pixels in its 3x3 interior.
    - Rotate the ring-plus-pointer object clockwise by K quarter-turns about the ring center.
    - Because the ring itself is rotationally symmetric, only the pointer direction changes position.
    - Erase all maroon interior markers to black after the rotation.

    Key insight:
    - Two independent cues must be combined: the pointer says which orientation is being rotated, while the maroon count says how many 90-degree clockwise steps to apply; the count markers are then consumed and disappear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, MAROON)]

    for color in colors:
        r0, c0, direction, mark_count = _detect_object(g, color)
        out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4][g[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] == MAROON] = BG

        old_r, old_c = _pointer_cell(r0, c0, direction)
        new_r, new_c = _pointer_cell(r0, c0, _rotate_direction(direction, mark_count))
        out[old_r, old_c] = BG
        out[new_r, new_c] = color

    return out


def _ring_top_left_from_bbox(bbox_r: int, bbox_c: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return bbox_r + 1, bbox_c
    if direction == "down":
        return bbox_r, bbox_c
    if direction == "left":
        return bbox_r, bbox_c + 1
    return bbox_r, bbox_c


def _draw_object(
    grid: np.ndarray,
    ring_r: int,
    ring_c: int,
    color: int,
    direction: str,
    marks: tuple[tuple[int, int], ...],
    include_marks: bool,
) -> None:
    for r, c in _ring_cells(ring_r, ring_c):
        grid[r, c] = color
    pr, pc = _pointer_cell(ring_r, ring_c, direction)
    grid[pr, pc] = color
    if include_marks:
        for dr, dc in marks:
            grid[ring_r + 1 + dr, ring_c + 1 + dc] = MAROON


def _render_input(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        _draw_object(
            grid,
            int(spec["ring_r"]),
            int(spec["ring_c"]),
            int(spec["color"]),
            str(spec["direction"]),
            tuple(spec["marks"]),
            include_marks=True,
        )
    return grid


def _render_output(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        direction = str(spec["direction"])
        marks = tuple(spec["marks"])
        out_direction = _rotate_direction(direction, len(marks))
        _draw_object(
            grid,
            int(spec["ring_r"]),
            int(spec["ring_c"]),
            int(spec["color"]),
            out_direction,
            marks,
            include_marks=False,
        )
    return grid


def _sample_layout(n_obj: int) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]] | None:
    feasible_cols = [ncols for ncols in range(1, 4) if (n_obj + ncols - 1) // ncols <= 3]
    ncols = random.choice(feasible_cols)
    nrows = (n_obj + ncols - 1) // ncols

    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    row_gap = random.randint(1, 2)
    col_gap = random.randint(1, 2)

    max_slot_h = (30 - top_margin - bottom_margin - (nrows - 1) * row_gap) // nrows
    max_slot_w = (30 - left_margin - right_margin - (ncols - 1) * col_gap) // ncols
    if max_slot_h < 7 or max_slot_w < 7:
        return None

    row_heights = [random.randint(7, min(12, max_slot_h)) for _ in range(nrows)]
    col_widths = [random.randint(7, min(12, max_slot_w)) for _ in range(ncols)]
    height = top_margin + bottom_margin + sum(row_heights) + (nrows - 1) * row_gap
    width = left_margin + right_margin + sum(col_widths) + (ncols - 1) * col_gap
    if height > 30 or width > 30:
        return None

    row_starts: list[int] = []
    cursor = top_margin
    for size in row_heights:
        row_starts.append(cursor)
        cursor += size + row_gap

    col_starts: list[int] = []
    cursor = left_margin
    for size in col_widths:
        col_starts.append(cursor)
        cursor += size + col_gap

    slots = [
        (row_starts[r], col_starts[c], row_heights[r], col_widths[c])
        for r in range(nrows)
        for c in range(ncols)
    ]
    return (height, width), random.sample(slots, n_obj)


def _build_specs(shape: tuple[int, int], slots: list[tuple[int, int, int, int]], colors: list[int]) -> list[dict[str, object]]:
    del shape
    specs: list[dict[str, object]] = []
    for slot, color in zip(slots, colors):
        slot_r, slot_c, slot_h, slot_w = slot
        direction = random.choice(DIRECTIONS)
        bbox_h, bbox_w = TOTAL_BBOX[direction]
        bbox_r = slot_r + random.randint(0, slot_h - bbox_h)
        bbox_c = slot_c + random.randint(0, slot_w - bbox_w)
        ring_r, ring_c = _ring_top_left_from_bbox(bbox_r, bbox_c, direction)

        mark_count = random.randint(1, 3)
        marks = tuple(random.sample(INNER_POSITIONS, mark_count))
        specs.append(
            {
                "ring_r": ring_r,
                "ring_c": ring_c,
                "color": color,
                "direction": direction,
                "marks": marks,
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    n_obj = random.randint(1, 6)
    layout = _sample_layout(n_obj)
    if layout is None:
        return None

    shape, slots = layout
    colors = random.sample(list(RING_COLORS), n_obj)
    specs = _build_specs(shape, slots, colors)

    inp = _render_input(shape, specs)
    expected = _render_output(shape, specs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointed-ring rotation input")
