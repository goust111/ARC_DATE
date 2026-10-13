from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MAX_GENERATION_ATTEMPTS = 500
HEIGHT_CHOICES = tuple(range(11, 18, 2))
WIDTH_CHOICES = tuple(range(11, 18))


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, mask.shape):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(coords.shape[0])


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors4(r, c, mask.shape):
            if not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _orbit_cells(height: int, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    mr = height - 1 - r
    if mr == r:
        return [(r, c)]
    if r < mr:
        return [(r, c), (mr, c)]
    return [(mr, c), (r, c)]


def _orbit_key(height: int, cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    return min(r, height - 1 - r), c


def _classify_blue(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reflected = mask[::-1, :]
    symmetric = mask & reflected
    asymmetric = mask & ~reflected
    return symmetric, asymmetric


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _oracle_transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    blue = out == BLUE
    symmetric, asymmetric = _classify_blue(blue)
    control_coords = np.argwhere(out == RED)
    assert control_coords.shape[0] == 1, "expected exactly one red control pixel"
    control = tuple(map(int, control_coords[0]))

    out[asymmetric] = GREY
    for r, c in np.argwhere(symmetric):
        d = _chebyshev((int(r), int(c)), control)
        out[int(r), int(c)] = YELLOW if d % 2 == 1 else GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter blue foreground pixels by horizontal-reflection symmetry, then distance-color the surviving subset around the red control pixel.

    Input structure:
    - The grid has black background, blue foreground pixels, and exactly one red control pixel.
    - Some blue pixels have a blue partner at the same column in the opposite row, while others do not.
    - The red control pixel is a separate marker and is not on the horizontal middle row.

    Rule:
    - For each blue pixel at `(r, c)`, test whether `(H - 1 - r, c)` is also blue.
    - Recolor blue pixels that fail this horizontal-reflection test to grey.
    - For blue pixels that pass, compute Chebyshev distance to the red control pixel.
    - Recolor passing pixels to yellow at odd distance and green at even distance, while preserving the red control pixel and black background.

    Key insight:
    - Reflection symmetry acts as a gate: only the symmetric subset is eligible for radial distance coloring, so both the mirror test and the distance parity are essential.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.count_nonzero(g == RED) == 1, "expected exactly one red control pixel"

    control_r, control_c = map(int, np.argwhere(g == RED)[0])
    assert control_r != g.shape[0] // 2, "control pixel must not lie on the horizontal middle row"

    return _oracle_transform(g)


def _grow_symmetric_base(height: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    mid = height // 2

    center_len = int(rng.integers(2, min(6, width - 1)))
    start_min = 1
    start_max = width - center_len - 1
    if start_max < start_min:
        return None
    center_start = int(rng.integers(start_min, start_max + 1))
    mask[mid, center_start : center_start + center_len] = True

    target = int(rng.integers(10, 18))
    for _ in range(200):
        if int(mask.sum()) >= target:
            break
        candidates = _frontier(mask)
        rng.shuffle(candidates)
        added = False
        for cell in candidates:
            orbit = _orbit_cells(height, cell)
            if any(mask[r, c] for r, c in orbit):
                continue
            if int(mask.sum()) + len(orbit) > target + 1:
                continue
            for r, c in orbit:
                mask[r, c] = True
            added = True
            break
        if not added:
            return None

    if int(mask.sum()) < 10:
        return None
    if not _is_connected(mask):
        return None
    return mask


def _add_asymmetric_pixels(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    height, width = mask.shape
    mid = height // 2
    out = mask.copy()
    claimed: set[tuple[int, int]] = set()

    total = int(rng.integers(3, 7))
    top_count = int(rng.integers(1, total))
    bottom_count = total - top_count
    if bottom_count == 0:
        bottom_count = 1
        top_count = total - 1

    plan = ["top"] * top_count + ["bottom"] * bottom_count
    rng.shuffle(plan)

    for side in plan:
        candidates = []
        for r, c in _frontier(out):
            if r == mid:
                continue
            if side == "top" and r > mid:
                continue
            if side == "bottom" and r < mid:
                continue
            mr = height - 1 - r
            if out[mr, c]:
                continue
            key = _orbit_key(height, (r, c))
            if key in claimed:
                continue
            candidates.append((r, c))
        if not candidates:
            return None
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        out[r, c] = True
        claimed.add(_orbit_key(height, (r, c)))

    if not _is_connected(out):
        return None
    if np.array_equal(out, out[::-1, :]):
        return None
    return out


def _pick_embedded_control(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    height, width = mask.shape
    mid = height // 2
    candidates = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if r == mid:
            continue
        if sum(1 for nr, nc in _neighbors4(r, c, mask.shape) if mask[nr, nc]) < 2:
            continue
        test = mask.copy()
        test[r, c] = False
        if _is_connected(test):
            candidates.append((r, c))
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _pick_adjacent_control(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    height, width = mask.shape
    mid = height // 2
    strong = []
    weak = []
    for r in range(height):
        for c in range(width):
            if mask[r, c] or r == mid:
                continue
            neighbors = sum(1 for nr, nc in _neighbors4(r, c, mask.shape) if mask[nr, nc])
            if neighbors >= 2:
                strong.append((r, c))
            elif neighbors == 1:
                weak.append((r, c))
    pool = strong if strong else weak
    if not pool:
        return None
    return pool[int(rng.integers(0, len(pool)))]


def _build_grid(blue_mask: np.ndarray, control: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(blue_mask.shape, dtype=int)
    grid[blue_mask] = BLUE
    grid[control] = RED
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    height, width = grid.shape
    if max(height, width) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BLACK, BLUE, RED}:
        return False

    control_coords = np.argwhere(grid == RED)
    if control_coords.shape[0] != 1:
        return False
    control = tuple(map(int, control_coords[0]))
    mid = height // 2
    if control[0] == mid:
        return False

    blue = grid == BLUE
    if not _is_connected(blue):
        return False
    if sum(1 for nr, nc in _neighbors4(control[0], control[1], grid.shape) if blue[nr, nc]) == 0:
        return False

    rows_used = int(np.count_nonzero(blue.any(axis=1)))
    cols_used = int(np.count_nonzero(blue.any(axis=0)))
    if rows_used < 5 or cols_used < 4:
        return False

    symmetric, asymmetric = _classify_blue(blue)
    sym_coords = [tuple(map(int, rc)) for rc in np.argwhere(symmetric)]
    asym_coords = [tuple(map(int, rc)) for rc in np.argwhere(asymmetric)]
    if len(sym_coords) < 5 or len(asym_coords) < 3:
        return False
    if not np.any(asymmetric[:mid, :]) or not np.any(asymmetric[mid + 1 :, :]):
        return False
    if int(np.count_nonzero(symmetric & (np.arange(height)[:, None] != mid))) < 4:
        return False

    sym_distances = [_chebyshev(cell, control) for cell in sym_coords]
    asym_distances = [_chebyshev(cell, control) for cell in asym_coords]
    if len(set(sym_distances)) < 3:
        return False
    if {d % 2 for d in sym_distances} != {0, 1}:
        return False
    if {d % 2 for d in asym_distances} != {0, 1}:
        return False
    if not (set(sym_distances) & set(asym_distances)):
        return False

    out = main(grid)
    expected = _oracle_transform(grid)
    if not np.array_equal(out, expected):
        return False
    if out[control] != RED:
        return False
    if not np.any(out == GREEN) or not np.any(out == YELLOW) or not np.any(out == GREY):
        return False
    if not np.array_equal(out == GREY, asymmetric):
        return False

    for cell in sym_coords:
        d = _chebyshev(cell, control)
        target = YELLOW if d % 2 == 1 else GREEN
        if out[cell] != target:
            return False

    for cell in asym_coords:
        if out[cell] != GREY:
            return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.choice(HEIGHT_CHOICES))
    width = int(rng.choice(WIDTH_CHOICES))

    symmetric_base = _grow_symmetric_base(height, width, rng)
    if symmetric_base is None:
        return None

    blue_mask = _add_asymmetric_pixels(symmetric_base, rng)
    if blue_mask is None:
        return None

    embedded = _pick_embedded_control(blue_mask, rng)
    adjacent = _pick_adjacent_control(blue_mask, rng)
    if embedded is None and adjacent is None:
        return None

    if embedded is not None and adjacent is not None and rng.random() < 0.5:
        control = embedded
        blue_mask = blue_mask.copy()
        blue_mask[control] = False
    elif adjacent is not None:
        control = adjacent
    else:
        control = embedded
        blue_mask = blue_mask.copy()
        blue_mask[control] = False

    grid = _build_grid(blue_mask, control)
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-filtered distance-coloring input")
