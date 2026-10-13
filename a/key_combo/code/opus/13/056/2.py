from __future__ import annotations

import hashlib
import itertools
import random

import numpy as np


BG = 0
RED = 2
GREY = 5

GRID_SIDE_RANGE = (10, 16)
INTERIOR_SIDES = (3, 4)
RED_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 600

_GEN_CALL_COUNT = 0
_USED_OUTPUT_SIGS: set[str] = set()
_USED_INPUT_SIGS: set[str] = set()


def _signature(grid: np.ndarray) -> str:
    arr = np.asarray(grid, dtype=int)
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _local_patterns(rows: int, cols: int, count: int, require_anchor: bool) -> list[np.ndarray]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    valid: list[np.ndarray] = []
    for chosen in itertools.combinations(cells, count):
        chosen_set = set(chosen)
        if require_anchor and (0, 0) not in chosen_set:
            continue
        if (not require_anchor) and (0, 0) in chosen_set:
            continue
        ok = True
        for r, c in chosen:
            for nr, nc in _neighbors8(r, c):
                if (nr, nc) in chosen_set:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        mask = np.zeros((rows, cols), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        valid.append(mask)
    return valid


def _frame_extrema(gray_coords: np.ndarray) -> tuple[int, int, int, int]:
    top_r = int(gray_coords[:, 0].min())
    bottom_r = int(gray_coords[:, 0].max())
    left_c = int(gray_coords[:, 1].min())
    right_c = int(gray_coords[:, 1].max())
    assert len([1 for r, _c in gray_coords if int(r) == top_r]) == 1
    assert len([1 for r, _c in gray_coords if int(r) == bottom_r]) == 1
    assert len([1 for _r, c in gray_coords if int(c) == left_c]) == 1
    assert len([1 for _r, c in gray_coords if int(c) == right_c]) == 1
    return top_r, bottom_r, left_c, right_c


def _expanded_crop(pattern: np.ndarray) -> np.ndarray:
    rows, cols = map(int, pattern.shape)
    canvas = np.zeros((rows * rows, cols * cols), dtype=bool)
    for lr, lc in np.argwhere(pattern):
        rs = int(lr) * rows
        cs = int(lc) * cols
        canvas[rs : rs + rows, cs : cs + cols] |= pattern
    return canvas[:rows, :cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the rectangle defined by four grey controls and replace its interior with the controls' one-step self-similar stamp crop.

    Input structure:
    - The grid is a black background with exactly four grey control pixels and 2-4 red pixels.
    - The grey pixels have unique topmost, bottommost, leftmost, and rightmost coordinates, defining a rectangle.
    - All red pixels lie strictly inside that rectangle, and the interior size is 3x3, 3x4, 4x3, or 4x4.

    Rule:
    - Use the extreme grey coordinates to draw a full grey rectangular frame.
    - Read the rectangle interior as a binary red pattern P in local coordinates.
    - On a virtual canvas of shape (R*R, C*C), stamp P into every block selected by a red cell of P, then crop the top-left R x C region back into the interior.

    Key insight:
    - The same sparse input jointly fixes the frame geometry and supplies the self-referential stamp whose cropped expansion becomes the new interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    gray_coords = np.argwhere(g == GREY)
    red_coords = np.argwhere(g == RED)
    assert len(gray_coords) == 4, "expected exactly four grey control pixels"
    assert 2 <= len(red_coords) <= 4, "expected 2-4 red pixels"
    assert np.all((g == BG) | (g == GREY) | (g == RED)), "unexpected colors"

    top_r, bottom_r, left_c, right_c = _frame_extrema(gray_coords)
    rows = bottom_r - top_r - 1
    cols = right_c - left_c - 1
    assert rows in INTERIOR_SIDES and cols in INTERIOR_SIDES, "unexpected interior size"

    interior = g[top_r + 1 : bottom_r, left_c + 1 : right_c]
    pattern = interior == RED

    for r, c in red_coords:
        rr = int(r)
        cc = int(c)
        assert top_r < rr < bottom_r and left_c < cc < right_c, "red pixels must lie strictly inside the rectangle"

    out = np.full_like(g, BG)
    out[top_r, left_c : right_c + 1] = GREY
    out[bottom_r, left_c : right_c + 1] = GREY
    out[top_r + 1 : bottom_r, left_c] = GREY
    out[top_r + 1 : bottom_r, right_c] = GREY

    cropped = _expanded_crop(pattern)
    out[top_r + 1 : bottom_r, left_c + 1 : right_c][cropped] = RED
    return out


def _build_input(
    grid_rows: int,
    grid_cols: int,
    top_r: int,
    left_c: int,
    pattern: np.ndarray,
    top_c: int,
    bottom_c: int,
    left_r: int,
    right_r: int,
) -> np.ndarray:
    rows, cols = map(int, pattern.shape)
    bottom_r = top_r + rows + 1
    right_c = left_c + cols + 1

    grid = np.zeros((grid_rows, grid_cols), dtype=int)
    grid[top_r, top_c] = GREY
    grid[bottom_r, bottom_c] = GREY
    grid[left_r, left_c] = GREY
    grid[right_r, right_c] = GREY
    grid[top_r + 1 : bottom_r, left_c + 1 : right_c][pattern] = RED
    return grid


def _input_structure_ok(grid: np.ndarray) -> bool:
    gray_coords = np.argwhere(grid == GREY)
    red_coords = np.argwhere(grid == RED)
    if len(gray_coords) != 4 or not (2 <= len(red_coords) <= 4):
        return False
    if not np.all((grid == BG) | (grid == GREY) | (grid == RED)):
        return False

    try:
        top_r, bottom_r, left_c, right_c = _frame_extrema(gray_coords)
    except AssertionError:
        return False

    rows = bottom_r - top_r - 1
    cols = right_c - left_c - 1
    if rows not in INTERIOR_SIDES or cols not in INTERIOR_SIDES:
        return False

    for r, c in red_coords:
        rr = int(r)
        cc = int(c)
        if not (top_r < rr < bottom_r and left_c < cc < right_c):
            return False

    for (r1, c1), (r2, c2) in itertools.combinations([(int(r), int(c)) for r, c in red_coords], 2):
        if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
            return False

    outside = grid.copy()
    outside[top_r + 1 : bottom_r, left_c + 1 : right_c] = BG
    if np.any(outside == RED):
        return False
    return True


def _semantic_oracle(grid: np.ndarray) -> np.ndarray:
    gray_coords = np.argwhere(grid == GREY)
    top_r, bottom_r, left_c, right_c = _frame_extrema(gray_coords)
    interior = grid[top_r + 1 : bottom_r, left_c + 1 : right_c]
    pattern = interior == RED

    expected = np.zeros_like(grid)
    expected[top_r, left_c : right_c + 1] = GREY
    expected[bottom_r, left_c : right_c + 1] = GREY
    expected[top_r + 1 : bottom_r, left_c] = GREY
    expected[top_r + 1 : bottom_r, right_c] = GREY
    expected[top_r + 1 : bottom_r, left_c + 1 : right_c][_expanded_crop(pattern)] = RED
    return expected


def _generator_attempt(require_anchor: bool) -> np.ndarray | None:
    rows = random.choice(INTERIOR_SIDES)
    cols = random.choice(INTERIOR_SIDES)
    count_choices = [
        count
        for count in range(RED_COUNT_RANGE[0], RED_COUNT_RANGE[1] + 1)
        if _local_patterns(rows, cols, count, require_anchor)
    ]
    if not count_choices:
        return None

    count = random.choice(count_choices)
    patterns = _local_patterns(rows, cols, count, require_anchor)
    pattern = random.choice(patterns).copy()

    full_h = rows + 2
    full_w = cols + 2
    grid_rows = random.randint(max(GRID_SIDE_RANGE[0], full_h + 2), GRID_SIDE_RANGE[1])
    grid_cols = random.randint(max(GRID_SIDE_RANGE[0], full_w + 2), GRID_SIDE_RANGE[1])

    top_r = random.randint(1, grid_rows - full_h - 1)
    left_c = random.randint(1, grid_cols - full_w - 1)
    bottom_r = top_r + full_h - 1
    right_c = left_c + full_w - 1

    top_c = random.randint(left_c + 1, right_c - 1)
    bottom_c = random.randint(left_c + 1, right_c - 1)
    left_r = random.randint(top_r + 1, bottom_r - 1)
    right_r = random.randint(top_r + 1, bottom_r - 1)

    grid = _build_input(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        top_r=top_r,
        left_c=left_c,
        pattern=pattern,
        top_c=top_c,
        bottom_c=bottom_c,
        left_r=left_r,
        right_r=right_r,
    )

    if not _input_structure_ok(grid):
        return None

    expected = _semantic_oracle(grid)
    predicted = main(grid)
    if not np.array_equal(predicted, expected):
        return None

    input_sig = _signature(grid)
    output_sig = _signature(expected)
    if input_sig in _USED_INPUT_SIGS or output_sig in _USED_OUTPUT_SIGS:
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    require_anchor = (_GEN_CALL_COUNT % 2) == 0
    alt_anchor = not require_anchor

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generator_attempt(require_anchor=require_anchor)
        if grid is None:
            continue
        _GEN_CALL_COUNT += 1
        _USED_INPUT_SIGS.add(_signature(grid))
        _USED_OUTPUT_SIGS.add(_signature(main(grid)))
        return grid

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generator_attempt(require_anchor=alt_anchor)
        if grid is None:
            continue
        _GEN_CALL_COUNT += 1
        _USED_INPUT_SIGS.add(_signature(grid))
        _USED_OUTPUT_SIGS.add(_signature(main(grid)))
        return grid

    raise RuntimeError("failed to generate a valid input")
