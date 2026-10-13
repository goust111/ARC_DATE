import random

import numpy as np


BACKGROUND = 0
MARK_COLOR = 2
CHECKER_COLOR_POOL = (1, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size, "expected a non-empty mask"
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def _checkerboard(shape: tuple[int, int], colors: tuple[int, int], phase: int) -> np.ndarray:
    rr, cc = np.indices(shape)
    parity = (rr + cc + int(phase)) % 2 == 0
    return np.where(parity, int(colors[0]), int(colors[1])).astype(int)


def _first_two_colors(rect: np.ndarray) -> tuple[int, int]:
    seen: list[int] = []
    for value in rect.flat:
        color = int(value)
        if color not in seen:
            seen.append(color)
            if len(seen) == 2:
                break
    assert len(seen) == 2, "checkerboard must contain exactly two colors"
    return seen[0], seen[1]


def _recover_checker(rect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colors = _first_two_colors(rect)
    cand0 = _checkerboard(rect.shape, colors, phase=0)
    cand1 = _checkerboard(rect.shape, colors, phase=1)
    mism0 = int(np.count_nonzero(rect != cand0))
    mism1 = int(np.count_nonzero(rect != cand1))
    expected = cand0 if mism0 <= mism1 else cand1
    violations = np.argwhere(rect != expected)
    return expected, violations


def _paint_cross(
    out: np.ndarray,
    row: int,
    col: int,
    rect_top: int,
    rect_left: int,
    rect_bottom: int,
    rect_right: int,
) -> None:
    left = out[row, :rect_left]
    left[left == BACKGROUND] = MARK_COLOR
    right = out[row, rect_right + 1 :]
    right[right == BACKGROUND] = MARK_COLOR

    top = out[:rect_top, col]
    top[top == BACKGROUND] = MARK_COLOR
    bottom = out[rect_bottom + 1 :, col]
    bottom[bottom == BACKGROUND] = MARK_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the corrupted checkerboard rectangle and export each violation as a red cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg = g != BACKGROUND
    assert fg.any(), "expected a checkerboard rectangle"
    r0, c0, r1, c1 = _bbox(fg)

    rect = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(rect != BACKGROUND), "checkerboard rectangle must be solid"

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND
    assert np.all(outside == BACKGROUND), "expected only one non-background rectangle"

    colors = np.unique(rect)
    assert colors.size == 2, "checkerboard must use exactly two colors"

    repaired, violations = _recover_checker(rect)

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = repaired
    for local_r, local_c in violations:
        _paint_cross(out, r0 + int(local_r), c0 + int(local_c), r0, c0, r1, c1)
    return out


def _pick_rect_dims(height: int, width: int, rng: random.Random) -> tuple[int, int]:
    max_h = min(height - 2, 10)
    max_w = min(width - 2, 10)
    assert max_h >= 3 and max_w >= 3, "grid must leave a margin around the rectangle"

    for _ in range(100):
        rect_h = rng.randint(3, max_h)
        rect_w = rng.randint(3, max_w)
        if rect_h * rect_w >= 8:
            return rect_h, rect_w
    raise RuntimeError("failed to sample a large-enough checkerboard rectangle")


def _pick_violation_cells(
    rect_h: int, rect_w: int, count: int, rng: random.Random
) -> list[tuple[int, int]]:
    if count == 1:
        return [(rng.randrange(rect_h), rng.randrange(rect_w))]

    modes = ["free"]
    if rect_w >= count:
        modes.append("same_row")
    if rect_h >= count:
        modes.append("same_col")
    mode = rng.choice(modes)

    if mode == "same_row":
        row = rng.randrange(rect_h)
        cols = rng.sample(range(rect_w), count)
        return [(row, col) for col in cols]
    if mode == "same_col":
        col = rng.randrange(rect_w)
        rows = rng.sample(range(rect_h), count)
        return [(row, col) for row in rows]

    cells = rng.sample([(r, c) for r in range(rect_h) for c in range(rect_w)], count)
    return [(int(r), int(c)) for r, c in cells]


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_RETRIES):
        height = rng.randint(9, 18)
        width = rng.randint(9, 18)
        rect_h, rect_w = _pick_rect_dims(height, width, rng)
        top = rng.randint(1, height - rect_h - 1)
        left = rng.randint(1, width - rect_w - 1)

        color_a, color_b = rng.sample(CHECKER_COLOR_POOL, 2)
        phase = rng.randrange(2)
        rect = _checkerboard((rect_h, rect_w), (color_a, color_b), phase)

        violation_count = rng.choice((1, 2, 3))
        violations = _pick_violation_cells(rect_h, rect_w, violation_count, rng)
        corrupted = rect.copy()
        for r, c in violations:
            corrupted[r, c] = color_b if corrupted[r, c] == color_a else color_a

        grid = np.full((height, width), BACKGROUND, dtype=int)
        grid[top : top + rect_h, left : left + rect_w] = corrupted

        repaired, found = _recover_checker(corrupted)
        if int(found.shape[0]) != violation_count:
            continue
        if not np.array_equal(repaired, rect):
            continue
        return grid

    raise RuntimeError("generate_input: exceeded retry budget")
