from __future__ import annotations

import random

import numpy as np


BLACK = 0
FOREGROUND_COLORS = tuple(range(1, 10))
GEN_RETRIES = 500
GRID_SIZE_RANGE = (7, 23)
MIN_SEPARATION = 3

_PARITY_BAG: list[int] = []
_MODE_BAG: list[str] = []


def _next_from_bag(bag: list, values: tuple):
    if not bag:
        bag.extend(values)
        random.shuffle(bag)
    return bag.pop()


def _marker_pair(grid: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    coords = np.argwhere(g != BLACK)
    assert coords.shape[0] == 2, "expected exactly two marker pixels"

    markers = []
    seen_colors = set()
    for r, c in coords:
        color = int(g[int(r), int(c)])
        assert color != BLACK, "markers must be non-black"
        assert color not in seen_colors, "the two markers must have different colors"
        seen_colors.add(color)
        markers.append((int(r), int(c), color))
    return markers[0], markers[1]


def _distance_fields(
    shape: tuple[int, int],
    p1: tuple[int, int],
    p2: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    H, W = shape
    rr, cc = np.indices((int(H), int(W)))
    d1 = np.maximum(np.abs(rr - int(p1[0])), np.abs(cc - int(p1[1])))
    d2 = np.maximum(np.abs(rr - int(p2[0])), np.abs(cc - int(p2[1])))
    return d1, d2


def _render_parity_xor(grid: np.ndarray) -> np.ndarray:
    (r1, c1, color1), (r2, c2, color2) = _marker_pair(grid)
    d1, d2 = _distance_fields(grid.shape, (r1, c1), (r2, c2))

    even1 = (d1 % 2) == 0
    even2 = (d2 % 2) == 0

    out = np.zeros(grid.shape, dtype=int)
    out[even1 & ~even2] = int(color1)
    out[even2 & ~even1] = int(color2)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill cells where two Chebyshev parity fields disagree, using the even-distance source color.

    Input structure:
    - The grid has a black background.
    - Exactly two non-black pixels are present.
    - The two marker pixels have different colors and act as the two distance-field sources.

    Rule:
    - Compute each cell's Chebyshev distance to both marker pixels.
    - If only the first marker distance is even, color the cell with the first marker color.
    - If only the second marker distance is even, color the cell with the second marker color.
    - If both distances have the same parity, leave the cell black; marker pixels follow this same test.

    Key insight:
    - The result is not a nearest-source territory split. It is the xor interference pattern of two
      complete even-distance square-ring fields.
    """
    g = np.asarray(grid, dtype=int)
    return _render_parity_xor(g)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _mode_ok(p1: tuple[int, int], p2: tuple[int, int], shape: tuple[int, int], mode: str) -> bool:
    r1, c1 = p1
    r2, c2 = p2
    dr = abs(int(r1) - int(r2))
    dc = abs(int(c1) - int(c2))
    H, W = shape

    if mode == "same_row":
        return dr == 0
    if mode == "same_col":
        return dc == 0
    if mode == "diagonal":
        return dr == dc and dr > 0
    if mode == "edge":
        return (
            r1 in (0, H - 1)
            or r2 in (0, H - 1)
            or c1 in (0, W - 1)
            or c2 in (0, W - 1)
        )
    if mode == "generic":
        return dr > 0 and dc > 0 and dr != dc
    raise AssertionError(mode)


def _random_pair(H: int, W: int, target_parity: int, mode: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    cells = [(r, c) for r in range(int(H)) for c in range(int(W))]
    for _ in range(2000):
        p1, p2 = random.sample(cells, 2)
        sep = _chebyshev(p1, p2)
        if sep < MIN_SEPARATION:
            continue
        if sep % 2 != int(target_parity):
            continue
        if not _mode_ok(p1, p2, (H, W), mode):
            continue
        return p1, p2
    return None


def _has_interrupted_even_ring(out: np.ndarray, d_source: np.ndarray, color: int) -> bool:
    for radius in sorted(set(int(v) for v in np.unique(d_source))):
        if radius == 0 or radius % 2:
            continue
        ring = d_source == radius
        if np.any(ring & (out == int(color))) and np.any(ring & (out == BLACK)):
            return True
    return False


def _semantic_accepts(grid: np.ndarray) -> bool:
    try:
        (r1, c1, color1), (r2, c2, color2) = _marker_pair(grid)
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape or max(out.shape) > 30:
        return False

    expected = _render_parity_xor(grid)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, grid):
        return False

    present = set(int(c) for c in np.unique(out))
    if not {BLACK, int(color1), int(color2)}.issubset(present):
        return False
    if not present.issubset({BLACK, int(color1), int(color2)}):
        return False

    d1, d2 = _distance_fields(grid.shape, (r1, c1), (r2, c2))
    count1 = int(np.count_nonzero(out == int(color1)))
    count2 = int(np.count_nonzero(out == int(color2)))
    if count1 < 4 or count2 < 4 or int(np.count_nonzero(out == BLACK)) < 4:
        return False

    same_even = ((d1 % 2) == 0) & ((d2 % 2) == 0)
    same_odd = ((d1 % 2) == 1) & ((d2 % 2) == 1)
    if not np.any(same_even) or not np.any(same_odd):
        return False

    if not _has_interrupted_even_ring(out, d1, int(color1)):
        return False
    if not _has_interrupted_even_ring(out, d2, int(color2)):
        return False

    far_source1 = (out == int(color1)) & (d1 > d2)
    far_source2 = (out == int(color2)) & (d2 > d1)
    if not np.any(far_source1) or not np.any(far_source2):
        return False

    sep = _chebyshev((r1, c1), (r2, c2))
    marker_values = (int(out[r1, c1]), int(out[r2, c2]))
    if sep % 2 == 0:
        if marker_values != (BLACK, BLACK):
            return False
    else:
        if marker_values != (int(color1), int(color2)):
            return False

    return True


def _gen_attempt(target_parity: int, mode: str) -> np.ndarray | None:
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)

    pair = _random_pair(H, W, target_parity, mode)
    if pair is None:
        return None
    (r1, c1), (r2, c2) = pair

    color1, color2 = random.sample(FOREGROUND_COLORS, 2)
    grid = np.zeros((H, W), dtype=int)
    grid[r1, c1] = int(color1)
    grid[r2, c2] = int(color2)

    if not _semantic_accepts(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        target_parity = _next_from_bag(_PARITY_BAG, (0, 1))
        mode = _next_from_bag(
            _MODE_BAG,
            ("generic", "same_row", "same_col", "diagonal", "edge", "generic"),
        )
        grid = _gen_attempt(int(target_parity), str(mode))
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid parity-xor field after {GEN_RETRIES} attempts")
