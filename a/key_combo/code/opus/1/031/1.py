from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
GREY = 5

MAX_COUNTER = 3
MIN_SIZE = 11
MAX_SIZE = 18
GEN_ATTEMPTS = 500


def _parse_input(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = g.shape

    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, BLUE, GREY}, "unexpected colors in input"

    grey_mask = g == GREY
    grey_count = int(np.sum(grey_mask))
    assert 1 <= grey_count <= MAX_COUNTER, "counter must contain 1 to 3 grey pixels"

    expected_counter = np.zeros_like(g, dtype=bool)
    expected_counter[:grey_count, width - 1] = True
    assert np.array_equal(grey_mask, expected_counter), "grey counter must be top-aligned in the last column"

    blue_row_counts = np.sum(g == BLUE, axis=1)
    blue_col_counts = np.sum(g == BLUE, axis=0)
    row_hits = np.flatnonzero(blue_row_counts == width)
    col_hits = np.flatnonzero(blue_col_counts == height)
    assert len(row_hits) == 1 and len(col_hits) == 1, "expected one full blue row and one full blue column"

    row = int(row_hits[0])
    col = int(col_hits[0])
    assert row >= grey_count, "horizontal row cannot overlap the counter"
    assert grey_count <= col < width - 1, "vertical column must stay inside and allow left translation"
    assert row + grey_count < height, "translated row must remain in bounds"

    expected_blue = np.zeros_like(g, dtype=bool)
    expected_blue[row, :] = True
    expected_blue[:, col] = True
    assert np.array_equal(g == BLUE, expected_blue), "blue pixels must form exactly one cross"

    return height, width, row, col, grey_count


def _cross_masks(shape: tuple[int, int], row: int, col: int, shift: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    new_row = row + shift
    new_col = col - shift
    assert 0 <= row < height and 0 <= col < width
    assert 0 <= new_row < height and 0 <= new_col < width

    old_cross = np.zeros(shape, dtype=bool)
    old_cross[row, :] = True
    old_cross[:, col] = True

    new_cross = np.zeros(shape, dtype=bool)
    new_cross[new_row, :] = True
    new_cross[:, new_col] = True

    ghost = old_cross & ~new_cross
    return old_cross, new_cross, ghost


def _even_chebyshev_ring(shape: tuple[int, int], center: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rr, cc = np.indices(shape)
    dist = np.maximum(np.abs(rr - center[0]), np.abs(cc - center[1]))
    ring = (dist > 0) & (dist % 2 == 0)
    return ring, dist


def _render_from_spec(shape: tuple[int, int], row: int, col: int, shift: int) -> np.ndarray:
    _, new_cross, ghost = _cross_masks(shape, row, col, shift)
    center = (row + shift, col - shift)
    ring, _ = _even_chebyshev_ring(shape, center)

    out = np.full(shape, BLACK, dtype=int)
    out[ghost] = GREY
    out[new_cross] = BLUE
    out[ring & (out == BLACK)] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate a counted blue cross, leave a grey ghost trail, and draw even Chebyshev rings around the new center.

    Input structure:
    - The grid has a black background, one full blue row, one full blue column, and a top-aligned grey counter in the last column.
    - The blue row and blue column intersect to form a single cross.
    - The grey counter contains 1 to 3 consecutive pixels and does not overlap the cross.

    Rule:
    - Count the grey pixels N, then erase the counter to background.
    - Move the blue horizontal arm down by N rows and the blue vertical arm left by N columns.
    - Color every old-cross pixel that is not reused by the translated cross grey, keeping reused pixels blue.
    - Using the translated intersection as center, paint every remaining background pixel yellow when its positive Chebyshev distance is even.

    Key insight:
    - The counter fixes the translation amount, and the old cross is not simply erased: its grey ghost blocks parts of the even-distance rings.
    """
    height, width, row, col, shift = _parse_input(grid)
    return _render_from_spec((height, width), row, col, shift)


def _render_wrong_old_center(shape: tuple[int, int], row: int, col: int, shift: int) -> np.ndarray:
    _, new_cross, ghost = _cross_masks(shape, row, col, shift)
    ring, _ = _even_chebyshev_ring(shape, (row, col))
    out = np.full(shape, BLACK, dtype=int)
    out[ghost] = GREY
    out[new_cross] = BLUE
    out[ring & (out == BLACK)] = YELLOW
    return out


def _render_wrong_ignore_ghost(shape: tuple[int, int], row: int, col: int, shift: int) -> np.ndarray:
    _, new_cross, _ = _cross_masks(shape, row, col, shift)
    ring, _ = _even_chebyshev_ring(shape, (row + shift, col - shift))
    out = np.full(shape, BLACK, dtype=int)
    out[new_cross] = BLUE
    out[ring & (out == BLACK)] = YELLOW
    return out


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    height, width, row, col, shift = _parse_input(inp)
    old_cross, new_cross, ghost = _cross_masks((height, width), row, col, shift)
    ring, dist = _even_chebyshev_ring((height, width), (row + shift, col - shift))

    assert out.shape == inp.shape, "output shape must match input shape"
    out = np.asarray(out, dtype=int)
    assert set(map(int, np.unique(out))) <= {BLACK, BLUE, YELLOW, GREY}, "unexpected colors in output"

    blue_mask = out == BLUE
    grey_mask = out == GREY
    yellow_mask = out == YELLOW

    assert np.array_equal(blue_mask, new_cross), "blue pixels must be exactly the translated cross"
    assert np.array_equal(grey_mask, ghost), "grey pixels must be exactly the ghost trail"

    expected_yellow = ring & ~new_cross & ~ghost
    assert np.array_equal(yellow_mask, expected_yellow), "yellow pixels must be the unblocked even Chebyshev ring cells"

    counter_mask = np.zeros((height, width), dtype=bool)
    counter_mask[:shift, width - 1] = True
    assert not np.any(grey_mask & counter_mask), "the original counter must disappear"
    assert np.all((out == BLACK) | blue_mask | grey_mask | yellow_mask), "output must only contain background or rule colors"

    overlap = old_cross & new_cross
    assert int(np.sum(overlap)) == 2, "translated full-grid cross should overlap the old cross at exactly two pixels"
    assert np.all(out[overlap] == BLUE), "overlap pixels must remain blue"
    assert np.all(dist[yellow_mask] % 2 == 0), "yellow pixels must lie on even Chebyshev distances"


def _pick_row(height: int, shift: int, mode: str) -> int:
    lo = shift
    hi = height - 1 - shift
    assert lo <= hi
    if mode == "upper":
        preferred = [shift, shift + 1]
    elif mode == "middle":
        preferred = list(range(shift + 2, height - shift - 2))
    elif mode == "lower":
        preferred = [height - 1 - shift, height - 2 - shift]
    else:
        raise ValueError(f"unknown row mode: {mode}")

    candidates = [value for value in preferred if lo <= value <= hi]
    if not candidates:
        candidates = list(range(lo, hi + 1))
    return int(random.choice(candidates))


def _pick_col(width: int, shift: int, mode: str) -> int:
    lo = shift
    hi = width - 2
    assert lo <= hi
    if mode == "left":
        preferred = [shift, shift + 1]
    elif mode == "middle":
        preferred = list(range(shift + 2, width - 4))
    elif mode == "right":
        preferred = [width - 3, width - 2]
    else:
        raise ValueError(f"unknown col mode: {mode}")

    candidates = [value for value in preferred if lo <= value <= hi]
    if not candidates:
        candidates = list(range(lo, hi + 1))
    return int(random.choice(candidates))


def _build_input(height: int, width: int, row: int, col: int, shift: int) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[row, :] = BLUE
    grid[:, col] = BLUE
    grid[:shift, width - 1] = GREY
    return grid


def _quality_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    height, width, row, col, shift = _parse_input(inp)
    old_cross, new_cross, ghost = _cross_masks((height, width), row, col, shift)
    ring, dist = _even_chebyshev_ring((height, width), (row + shift, col - shift))
    yellow_mask = out == YELLOW

    if int(np.sum(yellow_mask)) < 12:
        return False

    radii = set(map(int, dist[yellow_mask]))
    if 2 not in radii or max(radii) < 4:
        return False

    ideal_without_ghost = ring & ~new_cross
    blocked = ideal_without_ghost & ghost
    if int(np.sum(blocked)) < 4:
        return False

    horizontal_old = np.zeros((height, width), dtype=bool)
    horizontal_old[row, :] = True
    vertical_old = np.zeros((height, width), dtype=bool)
    vertical_old[:, col] = True
    horizontal_ghost = horizontal_old & ~new_cross
    vertical_ghost = vertical_old & ~new_cross

    if not np.any(blocked & horizontal_ghost):
        return False
    if not np.any(blocked & vertical_ghost):
        return False

    counter_positions = [(r, width - 1) for r in range(shift)]
    if not any(out[r, c] == YELLOW for r, c in counter_positions):
        return False

    if np.array_equal(out, _render_wrong_old_center((height, width), row, col, shift)):
        return False
    if np.array_equal(out, _render_wrong_ignore_ghost((height, width), row, col, shift)):
        return False

    if not np.any(old_cross & (out == GREY)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shift = random.randint(1, MAX_COUNTER)
    height = random.randint(max(MIN_SIZE, 2 * shift + 5), MAX_SIZE)
    width = random.randint(max(MIN_SIZE, shift + 6), MAX_SIZE)

    row = _pick_row(height, shift, random.choice(["upper", "middle", "lower"]))
    col = _pick_col(width, shift, random.choice(["left", "middle", "right"]))
    inp = _build_input(height, width, row, col, shift)

    try:
        out = main(inp)
        _validate_output_semantics(inp, out)
        if not _quality_checks(inp, out):
            return None
        return inp
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
