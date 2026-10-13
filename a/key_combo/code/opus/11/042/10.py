import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

KEEP_BY_COUNT = {1: "up", 2: "right", 3: "down", 4: "left"}
ODD_SIZES = tuple(range(11, 30, 2))
GEN_RETRIES = 200

_GEN_CALL_COUNT = 0


def _find_cross_center(grid: np.ndarray) -> tuple[int, int]:
    blue_rows = np.flatnonzero(np.all(grid == BLUE, axis=1))
    blue_cols = np.flatnonzero(np.all(grid == BLUE, axis=0))
    assert len(blue_rows) == 1, "expected exactly one full blue row"
    assert len(blue_cols) == 1, "expected exactly one full blue column"
    return int(blue_rows[0]), int(blue_cols[0])


def _direction_points(shape: tuple[int, int], center: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    r0, c0 = map(int, center)
    pts: list[tuple[int, int]] = []

    if direction == "up":
        for r in range(r0 - 1, -1, -1):
            d = r0 - r
            for c in (c0 - d, c0 + d):
                if 0 <= c < width:
                    pts.append((r, c))
    elif direction == "right":
        for c in range(c0 + 1, width):
            d = c - c0
            for r in (r0 - d, r0 + d):
                if 0 <= r < height:
                    pts.append((r, c))
    elif direction == "down":
        for r in range(r0 + 1, height):
            d = r - r0
            for c in (c0 - d, c0 + d):
                if 0 <= c < width:
                    pts.append((r, c))
    elif direction == "left":
        for c in range(c0 - 1, -1, -1):
            d = c0 - c
            for r in (r0 - d, r0 + d):
                if 0 <= r < height:
                    pts.append((r, c))
    else:
        raise ValueError(f"unknown direction: {direction}")

    return pts


def _render_expected(shape: tuple[int, int], center: tuple[int, int], direction: str) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    r0, c0 = map(int, center)
    out[r0, :] = BLUE
    out[:, c0] = BLUE
    for r, c in _direction_points(shape, center, direction):
        out[r, c] = RED
    return out


def _validate_counter_layout(grid: np.ndarray, center: tuple[int, int]) -> int:
    r0, c0 = map(int, center)
    coords = np.argwhere(grid == GREY)
    count = int(coords.shape[0])
    assert 1 <= count <= 4, "expected 1 to 4 grey counter pixels"
    cols = {int(c) for _r, c in coords}
    assert len(cols) == 1, "grey counter must occupy one column"
    counter_col = next(iter(cols))
    assert counter_col > c0, "grey counter must be to the right of the cross"

    rows = sorted(int(r) for r, _c in coords)
    assert rows[-1] < r0, "grey counter must be above the cross center row"
    assert rows == list(range(rows[0], rows[0] + count)), "grey counter must be contiguous"
    return count


def _validate_generated_pair(inp: np.ndarray, out: np.ndarray, keep_count: int) -> None:
    assert inp.ndim == 2 and out.ndim == 2
    assert inp.shape == out.shape
    assert inp.shape[0] == inp.shape[1] and inp.shape[0] % 2 == 1
    assert max(inp.shape) <= 30

    center = _find_cross_center(inp)
    assert center == _find_cross_center(out)
    observed_count = _validate_counter_layout(inp, center)
    assert observed_count == keep_count
    assert not np.any(out == GREY), "counter must disappear in output"

    expected = _render_expected(inp.shape, center, KEEP_BY_COUNT[keep_count])
    assert np.array_equal(out, expected), "output must match the stated transformation"

    variants = [
        _render_expected(inp.shape, center, KEEP_BY_COUNT[k]).tobytes()
        for k in sorted(KEEP_BY_COUNT)
    ]
    assert len(set(variants)) == 4, "all four counter choices must yield distinct outputs"
    assert np.count_nonzero(out == RED) > 0, "selected staircase must be non-empty"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the blue cross and retain only the counter-selected red distance staircase.

    Input structure:
    - A black odd-sized square grid contains one blue cross made from a full row and a full column.
    - The row and column intersection is the geometric reference center for all distance calculations.
    - One to four grey pixels form a vertical counter column in the upper-right region, away from the cross.

    Rule:
    - Find the unique full blue row R and full blue column C.
    - Count the grey pixels K.
    - A staircase direction is defined by distance from the center:
      for each step d away from the center, place a symmetric red pair at distance d from the center line.
      Up uses rows above R, right uses columns right of C, down uses rows below R, and left uses columns left of C.
    - Keep only the direction chosen by the counter mapping K=1 -> up, K=2 -> right, K=3 -> down, K=4 -> left.
    - Remove the grey counter pixels and preserve the blue cross.

    Key insight:
    - The blue cross fixes the origin of four possible distance-based staircases, and the grey counter chooses which one survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square grid"
    assert g.shape[0] % 2 == 1, "expected an odd side length"

    center = _find_cross_center(g)
    keep_count = _validate_counter_layout(g, center)
    out = g.copy()
    out[out == GREY] = BLACK
    for r, c in _direction_points(g.shape, center, KEEP_BY_COUNT[keep_count]):
        out[r, c] = RED
    return out


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(GEN_RETRIES):
        size = random.choice(ODD_SIZES)
        center = (size // 2, size // 2)

        keep_count = (_GEN_CALL_COUNT % 4) + 1
        counter_col = random.randint(center[1] + 1, size - 1)
        start_row = random.randint(0, center[0] - keep_count)

        inp = np.zeros((size, size), dtype=int)
        inp[center[0], :] = BLUE
        inp[:, center[1]] = BLUE
        inp[start_row : start_row + keep_count, counter_col] = GREY

        expected = _render_expected(inp.shape, center, KEEP_BY_COUNT[keep_count])
        solved = main(inp)
        _validate_generated_pair(inp, solved, keep_count)
        assert np.array_equal(solved, expected)

        _GEN_CALL_COUNT += 1
        return inp

    raise RuntimeError("failed to generate a valid input")
