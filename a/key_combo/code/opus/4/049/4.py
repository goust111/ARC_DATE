import numpy as np


GRID_SIZE = 12
BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
RNG = np.random.default_rng()


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _plus_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, center)
    return [(r + dr, c + dc) for dr, dc in PLUS_OFFSETS]


def _is_valid_center(center: tuple[int, int], *, max_col: int = 9) -> bool:
    r, c = map(int, center)
    return 1 <= r <= GRID_SIZE - 2 and 1 <= c <= max_col


def _paint_plus(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    for r, c in _plus_cells(center):
        grid[r, c] = int(color)


def _count_counter(grid: np.ndarray) -> int:
    col = grid[:, -1]
    n = 0
    while n < GRID_SIZE and int(col[n]) == GREY:
        n += 1
    assert 1 <= n <= 3, "expected 1..3 contiguous grey counter pixels at top of rightmost column"
    assert not np.any(col[n:] == GREY), "grey pixels must form one top-aligned run"
    return n


def _find_plus_center(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.shape == (5, 2), f"expected exactly one 5-pixel plus of color {color}"
    coord_set = {tuple(map(int, rc)) for rc in coords}
    centers = []
    for r, c in coord_set:
        if all((r + dr, c + dc) in coord_set for dr, dc in PLUS_OFFSETS):
            centers.append((r, c))
    assert len(centers) == 1, f"expected a unique plus center for color {color}"
    return centers[0]


def _shift(center: tuple[int, int], dr: int, dc: int) -> tuple[int, int]:
    r, c = map(int, center)
    return r + int(dr), c + int(dc)


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    if abs(dr) != abs(dc):
        return []
    sr = _sign(dr)
    sc = _sign(dc)
    return [(a[0] + k * sr, a[1] + k * sc) for k in range(1, abs(dr))]


def _simulate(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 12x12 grid"

    n = _count_counter(g)
    blue_center = _find_plus_center(g, BLUE)
    green_center = _find_plus_center(g, GREEN)

    blue_new = _shift(blue_center, n, -n)
    green_new = _shift(green_center, -n, n)
    assert _is_valid_center(blue_new, max_col=GRID_SIZE - 2), "shifted blue plus leaves grid"
    assert _is_valid_center(green_new, max_col=GRID_SIZE - 2), "shifted green plus leaves grid"

    out = g.copy()
    erase_mask = np.isin(out, [BLUE, GREEN, GREY])
    out[erase_mask] = BG
    _paint_plus(out, blue_new, BLUE)
    _paint_plus(out, green_new, GREEN)

    path = _path_between(blue_new, green_new)
    if path:
        if all(int(out[r, c]) == BG for r, c in path):
            for r, c in path:
                out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move two counted pluses in opposite directions, then draw a red diagonal only after the move creates it.

    Input structure:
    - A 12x12 black background grid contains one blue 5-pixel plus and one green 5-pixel plus.
    - The rightmost column starts with a contiguous run of 1..3 grey pixels that acts as the counter.
    - The two input plus centers are not initially on the same 45-degree diagonal.

    Rule:
    - Count the grey pixels in the top-aligned run on the rightmost column to get N.
    - Translate the blue plus by (down N, left N) and the green plus by (up N, right N).
    - Remove the grey counter pixels from the output.
    - If the translated plus centers now lie on one 45-degree diagonal, inspect the interior cells on that center-to-center diagonal segment.
    - When every interior cell is black after the move, recolor those interior cells red.

    Key insight:
    - The counter does not label a target directly; it changes both plus locations, and only that translated arrangement reveals whether a diagonal connector exists.
    """
    return _simulate(grid)


def _counter_cells(n: int) -> set[tuple[int, int]]:
    return {(r, GRID_SIZE - 1) for r in range(int(n))}


def _diagonal_after_shift(blue_center: tuple[int, int], green_center: tuple[int, int], n: int) -> bool:
    blue_new = _shift(blue_center, n, -n)
    green_new = _shift(green_center, -n, n)
    if not (_is_valid_center(blue_new, max_col=GRID_SIZE - 2) and _is_valid_center(green_new, max_col=GRID_SIZE - 2)):
        return False
    dr = int(blue_new[0] - green_new[0])
    dc = int(blue_new[1] - green_new[1])
    return abs(dr) == abs(dc)


def _sample_layout() -> tuple[int, tuple[int, int], tuple[int, int]]:
    n = int(RNG.integers(1, 4))
    candidates = []
    max_d = 9 - 2 * n
    for sign in (-1, 1):
        for d in range(2, max_d + 1):
            for blue_new_r in range(1, GRID_SIZE - 1):
                for blue_new_c in range(1, GRID_SIZE - 2):
                    blue_new = (blue_new_r, blue_new_c)
                    green_new = (blue_new_r + sign * d, blue_new_c + sign * d)
                    blue_center = _shift(blue_new, -n, n)
                    green_center = _shift(green_new, n, -n)
                    if not all(_is_valid_center(center) for center in (blue_center, green_center, blue_new, green_new)):
                        continue

                    blue_cells = set(_plus_cells(blue_center))
                    green_cells = set(_plus_cells(green_center))
                    if blue_cells & green_cells:
                        continue
                    if (blue_cells | green_cells) & _counter_cells(n):
                        continue
                    if abs(blue_center[0] - green_center[0]) == abs(blue_center[1] - green_center[1]):
                        continue

                    if sum(_diagonal_after_shift(blue_center, green_center, alt_n) for alt_n in (1, 2, 3)) != 1:
                        continue

                    candidates.append((d, blue_center, green_center))

    assert candidates, "expected at least one feasible layout"
    d_values = sorted({d for d, _, _ in candidates})
    weights = np.array([3 if d == 2 else 2 if d == d_values[-1] else 1 for d in d_values], dtype=float)
    chosen_d = int(RNG.choice(d_values, p=weights / weights.sum()))
    filtered = [(blue_center, green_center) for d, blue_center, green_center in candidates if d == chosen_d]
    blue_center, green_center = filtered[int(RNG.integers(len(filtered)))]
    return n, blue_center, green_center


def _validate_sample(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (GRID_SIZE, GRID_SIZE)
    assert set(map(int, np.unique(grid))) <= {BG, BLUE, GREEN, GREY}

    n = _count_counter(grid)
    blue_center = _find_plus_center(grid, BLUE)
    green_center = _find_plus_center(grid, GREEN)
    assert abs(blue_center[0] - green_center[0]) != abs(blue_center[1] - green_center[1]), "input must not start diagonal"

    true_hits = [alt_n for alt_n in (1, 2, 3) if _diagonal_after_shift(blue_center, green_center, alt_n)]
    assert true_hits == [n], "the visible counter must uniquely determine the diagonal alignment"

    out = _simulate(grid)
    assert not np.any(out[:, -1] == GREY), "counter must disappear"

    blue_new = _shift(blue_center, n, -n)
    green_new = _shift(green_center, -n, n)
    assert _find_plus_center(out, BLUE) == blue_new
    assert _find_plus_center(out, GREEN) == green_new

    path = _path_between(blue_new, green_new)
    assert path, "generator should always create a non-empty connector"
    red_coords = {tuple(map(int, rc)) for rc in np.argwhere(out == RED)}
    assert red_coords == set(path), "red pixels must be exactly the translated diagonal interior"
    assert not np.array_equal(grid, out), "output must differ from input"


def generate_input() -> np.ndarray:
    for _ in range(400):
        n, blue_center, green_center = _sample_layout()
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        _paint_plus(grid, blue_center, BLUE)
        _paint_plus(grid, green_center, GREEN)
        grid[:n, -1] = GREY
        _validate_sample(grid)
        return grid
    raise RuntimeError("failed to generate a valid sample")
