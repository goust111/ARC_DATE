from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = tuple(range(1, 10))
MAX_GRID = 30

NUM_COLORS_RANGE = (2, 4)
RECT_HEIGHT_RANGE = (2, 4)
RECT_WIDTH_RANGE = (2, 5)
BIG_GAP_RANGE = (2, 3)
PATTERN_MODES = ("row_gaps", "col_gaps", "both_gaps")
MAX_COMPONENTS = 6

SLOT_GAP_RANGE = (1, 2)
OUTER_MARGIN_RANGE = (0, 1)

PATCH_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _describe_color(grid: np.ndarray, color: int) -> dict[str, object]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, "every described color must be present"

    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    patch = grid[np.ix_(rows, cols)]
    assert coords.shape[0] == rows.size * cols.size, (
        "one color must occupy a full Cartesian product of its used rows and columns"
    )
    assert np.all(patch == color), "all used row/column intersections must belong to the same color"

    return {
        "color": int(color),
        "height": int(rows.size),
        "width": int(cols.size),
        "area": int(coords.shape[0]),
        "anchor": (int(rows.min()), int(cols.min())),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse each color's sparse rectangle scaffold into a solid rectangle and stack the recovered rectangles by area.

    Input structure:
    - The background is black.
    - Each non-black color appears as several separated 4-connected fragments.
    - For one color, the occupied cells lie on a sparse row/column scaffold: if the used rows and used columns are compressed to consecutive indices, the color becomes one filled rectangle.

    Rule:
    - For each non-black color, collect its distinct occupied rows and distinct occupied columns.
    - Compress those sorted row and column sets independently so they become `0..h-1` and `0..w-1`.
    - This reconstructs a solid `h x w` rectangle of that color, where `h` is the number of occupied rows and `w` is the number of occupied columns.
    - Sort the reconstructed rectangles by area from largest to smallest and place them top-to-bottom, left-aligned, with one black separator row between neighbors.

    Key insight:
    - The fragments are a sparse coordinate skeleton of a hidden rectangle: the ordered occupied rows and columns matter, while the literal black spacer bands do not.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors, "expected at least one non-black color"

    records = [_describe_color(g, color) for color in colors]
    records.sort(
        key=lambda rec: (
            -int(rec["area"]),
            int(rec["anchor"][0]),
            int(rec["anchor"][1]),
            int(rec["height"]),
            int(rec["width"]),
        )
    )

    out_h = sum(int(rec["height"]) for rec in records) + len(records) - 1
    out_w = max(int(rec["width"]) for rec in records)
    out = np.full((out_h, out_w), BG, dtype=int)

    row = 0
    for rec in records:
        height = int(rec["height"])
        width = int(rec["width"])
        out[row : row + height, :width] = int(rec["color"])
        row += height + 1

    return out


AREA_TO_DIMS: dict[int, list[tuple[int, int]]] = {}
for _h in range(RECT_HEIGHT_RANGE[0], RECT_HEIGHT_RANGE[1] + 1):
    for _w in range(RECT_WIDTH_RANGE[0], RECT_WIDTH_RANGE[1] + 1):
        AREA_TO_DIMS.setdefault(_h * _w, []).append((_h, _w))


def _run_group_count(positions: list[int]) -> int:
    return 1 + sum(positions[i + 1] - positions[i] > 1 for i in range(len(positions) - 1))


def _sample_axis_positions(length: int, gappy: bool) -> list[int]:
    positions = [0]
    if length == 1:
        return positions

    mandatory_big: set[int] = set()
    if gappy:
        mandatory_big.add(random.randrange(length - 1))
        if length > 2 and random.random() < 0.5:
            mandatory_big.add(random.randrange(length - 1))

    for i in range(length - 1):
        make_big = gappy and (i in mandatory_big or random.random() < 0.35)
        step = random.randint(*BIG_GAP_RANGE) if make_big else 1
        positions.append(positions[-1] + step)
    return positions


def _build_sparse_patch(height: int, width: int, mode: str) -> np.ndarray | None:
    row_gappy = mode in {"row_gaps", "both_gaps"}
    col_gappy = mode in {"col_gaps", "both_gaps"}

    for _ in range(PATCH_ATTEMPTS):
        rows = _sample_axis_positions(height, row_gappy)
        cols = _sample_axis_positions(width, col_gappy)
        component_count = _run_group_count(rows) * _run_group_count(cols)
        if not (1 < component_count <= MAX_COMPONENTS):
            continue

        patch = np.zeros((rows[-1] + 1, cols[-1] + 1), dtype=bool)
        patch[np.ix_(rows, cols)] = True
        if len(_components_from_mask(patch)) != component_count:
            continue
        return patch

    return None


def _sample_dimensions(count: int) -> list[tuple[int, int]]:
    areas = random.sample(list(AREA_TO_DIMS), count)
    dims = [random.choice(AREA_TO_DIMS[area]) for area in areas]
    random.shuffle(dims)
    return dims


def _choose_layout(count: int) -> tuple[int, int, list[tuple[int, int]]]:
    if count == 2 and random.random() < 0.5:
        rows, cols = 1, 2
    elif count == 2:
        rows, cols = 2, 1
    else:
        rows, cols = 2, 2

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(slots)
    return rows, cols, slots[:count]


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(patch)
    grid[coords[:, 0] + top, coords[:, 1] + left] = int(color)


def _place_specs(specs: list[dict[str, object]]) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    rows, cols, slots = _choose_layout(len(specs))
    assigned_specs = random.sample(specs, len(specs))
    assignments = list(zip(assigned_specs, slots))

    row_heights = [0] * rows
    col_widths = [0] * cols
    for spec, (slot_r, slot_c) in assignments:
        patch_h, patch_w = np.asarray(spec["patch"], dtype=bool).shape
        row_heights[slot_r] = max(row_heights[slot_r], int(patch_h))
        col_widths[slot_c] = max(col_widths[slot_c], int(patch_w))

    gap = random.randint(*SLOT_GAP_RANGE)
    top_margin = random.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = random.randint(*OUTER_MARGIN_RANGE)
    left_margin = random.randint(*OUTER_MARGIN_RANGE)
    right_margin = random.randint(*OUTER_MARGIN_RANGE)

    height = top_margin + bottom_margin + sum(row_heights) + gap * (rows - 1)
    width = left_margin + right_margin + sum(col_widths) + gap * (cols - 1)
    if height > MAX_GRID or width > MAX_GRID:
        return None

    row_starts: list[int] = []
    cursor = top_margin
    for slot_h in row_heights:
        row_starts.append(cursor)
        cursor += slot_h + gap

    col_starts: list[int] = []
    cursor = left_margin
    for slot_w in col_widths:
        col_starts.append(cursor)
        cursor += slot_w + gap

    grid = np.full((height, width), BG, dtype=int)
    placed: list[dict[str, object]] = []
    for spec, (slot_r, slot_c) in assignments:
        patch = np.asarray(spec["patch"], dtype=bool)
        patch_h, patch_w = patch.shape
        top = row_starts[slot_r] + random.randint(0, row_heights[slot_r] - patch_h)
        left = col_starts[slot_c] + random.randint(0, col_widths[slot_c] - patch_w)
        _paint_patch(grid, patch, top, left, int(spec["color"]))
        placed.append(
            {
                "color": int(spec["color"]),
                "height": int(spec["height"]),
                "width": int(spec["width"]),
                "area": int(spec["area"]),
                "anchor": (int(top), int(left)),
            }
        )

    return grid, placed


def _expected_output(placed: list[dict[str, object]]) -> np.ndarray:
    order = sorted(
        placed,
        key=lambda rec: (
            -int(rec["area"]),
            int(rec["anchor"][0]),
            int(rec["anchor"][1]),
            int(rec["height"]),
            int(rec["width"]),
        ),
    )
    out_h = sum(int(rec["height"]) for rec in order) + len(order) - 1
    out_w = max(int(rec["width"]) for rec in order)
    out = np.full((out_h, out_w), BG, dtype=int)

    row = 0
    for rec in order:
        height = int(rec["height"])
        width = int(rec["width"])
        out[row : row + height, :width] = int(rec["color"])
        row += height + 1
    return out


def _is_valid_case(grid: np.ndarray, placed: list[dict[str, object]]) -> bool:
    colors = [int(spec["color"]) for spec in placed]
    if len(colors) != len(set(colors)):
        return False

    for color in colors:
        if len(_components_from_mask(grid == color)) < 2:
            return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    expected = _expected_output(placed)
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*NUM_COLORS_RANGE)
    colors = random.sample(COLOR_POOL, count)
    dims = _sample_dimensions(count)

    specs: list[dict[str, object]] = []
    for color, (height, width) in zip(colors, dims):
        mode = random.choice(PATTERN_MODES)
        patch = _build_sparse_patch(height, width, mode)
        if patch is None:
            return None
        specs.append(
            {
                "color": int(color),
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
                "patch": patch,
            }
        )

    placed_case = _place_specs(specs)
    if placed_case is None:
        return None
    grid, placed = placed_case
    if not _is_valid_case(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
