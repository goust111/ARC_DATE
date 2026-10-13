import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
SEED_COLORS = (3, 4, 6, 7, 8, 9)
OPS = ("id", "h", "v")
MAX_GENERATION_ATTEMPTS = 300
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _decode_marker(quadrant: np.ndarray, color: int) -> str:
    coords = np.argwhere(quadrant == color)
    assert len(coords) in (1, 2), "instruction marker must have one or two pixels"
    if len(coords) == 1:
        return "id"

    (r0, c0), (r1, c1) = coords.tolist()
    dr = abs(int(r0) - int(r1))
    dc = abs(int(c0) - int(c1))
    assert dr + dc == 1, "two-pixel marker must be 4-neighbor adjacent"
    if dr == 0:
        return "h"
    return "v"


def _apply_op(arr: np.ndarray, op: str) -> np.ndarray:
    if op == "id":
        return arr.copy()
    if op == "h":
        return np.fliplr(arr)
    if op == "v":
        return np.flipud(arr)
    raise AssertionError(f"unknown operation: {op}")


def _apply_ops(arr: np.ndarray, ops: tuple[str, str] | list[str]) -> np.ndarray:
    out = np.asarray(arr, dtype=int).copy()
    for op in ops:
        out = _apply_op(out, op)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the red-then-blue flip instructions to the seed quadrant and draw the result in the lower-right quadrant.

    Input structure:
    - The grid is an odd-sized square split into four equal quadrants by a grey center row and a grey center column.
    - The upper-left quadrant contains a small seed pattern on a black background.
    - The upper-right quadrant contains exactly one red instruction marker, and the lower-left quadrant contains exactly one blue instruction marker.
    - Each instruction marker is either a single pixel, a horizontal domino, or a vertical domino; the lower-right quadrant is empty black background.

    Rule:
    - Decode each marker shape independently: a single pixel means no flip, a horizontal domino means horizontal flip, and a vertical domino means vertical flip.
    - Treat the full upper-left quadrant as the coordinate system for the seed pattern.
    - First apply the red marker's transform to the seed pattern, then apply the blue marker's transform to that result.
    - Copy the transformed pattern into the lower-right quadrant, preserving the seed color(s) and leaving the rest of the grid unchanged.

    Key insight:
    - The red and blue marker orientations are a compact instruction language for two sequential flips, and the flips act on the seed's coordinates within the whole quadrant rather than only within its tight bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected an odd square grid"
    side = int(g.shape[0])
    assert side % 2 == 1, "grid side length must be odd"

    mid = side // 2
    sep = int(g[mid, mid])
    assert np.all(g[mid, :] == sep) and np.all(g[:, mid] == sep), "expected a center separator cross"

    bg = int(np.bincount(g.ravel()).argmax())
    tl = g[:mid, :mid]
    tr = g[:mid, mid + 1 :]
    bl = g[mid + 1 :, :mid]

    seed_mask = (tl != bg) & (tl != sep)
    assert bool(seed_mask.any()), "upper-left quadrant must contain a seed pattern"
    seed = np.full_like(tl, bg)
    seed[seed_mask] = tl[seed_mask]

    red_op = _decode_marker(tr, RED)
    blue_op = _decode_marker(bl, BLUE)
    transformed = _apply_ops(seed, (red_op, blue_op))

    out = g.copy()
    out[mid + 1 :, mid + 1 :] = transformed
    return out


def _random_connected_patch(height: int, width: int, area: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _random_seed_mask(n: int) -> np.ndarray | None:
    max_h = min(5, n)
    max_w = min(5, n)
    patch_choices = [(h, w) for h in range(1, max_h + 1) for w in range(1, max_w + 1) if h * w >= 2]
    random.shuffle(patch_choices)

    for patch_h, patch_w in patch_choices:
        max_area = min(10, patch_h * patch_w)
        area = random.randint(2, max_area)
        patch_cells = _random_connected_patch(patch_h, patch_w, area)
        if patch_cells is None:
            continue

        r0 = random.randint(0, n - patch_h)
        c0 = random.randint(0, n - patch_w)
        mask = np.zeros((n, n), dtype=bool)
        for r, c in patch_cells:
            mask[r0 + r, c0 + c] = True

        variants = [
            mask,
            np.fliplr(mask),
            np.flipud(mask),
            np.flipud(np.fliplr(mask)),
        ]
        if len({variant.tobytes() for variant in variants}) == 4:
            return mask
    return None


def _marker_cells(n: int, op: str) -> list[tuple[int, int]]:
    if op == "id":
        span_h, span_w = 1, 1
        offsets = [(0, 0)]
    elif op == "h":
        span_h, span_w = 1, 2
        offsets = [(0, 0), (0, 1)]
    elif op == "v":
        span_h, span_w = 2, 1
        offsets = [(0, 0), (1, 0)]
    else:
        raise AssertionError(f"unknown operation: {op}")

    margin = 1 if n >= 5 else 0
    r_min = margin
    c_min = margin
    r_max = n - span_h - margin
    c_max = n - span_w - margin
    if r_max < r_min:
        r_min, r_max = 0, n - span_h
    if c_max < c_min:
        c_min, c_max = 0, n - span_w

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    return [(r0 + dr, c0 + dc) for dr, dc in offsets]


def _build_expected(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mid = g.shape[0] // 2
    sep = int(g[mid, mid])
    bg = BLACK
    tl = g[:mid, :mid]
    tr = g[:mid, mid + 1 :]
    bl = g[mid + 1 :, :mid]

    seed = np.full((mid, mid), bg, dtype=int)
    seed_mask = (tl != bg) & (tl != sep)
    seed[seed_mask] = tl[seed_mask]
    transformed = _apply_ops(seed, (_decode_marker(tr, RED), _decode_marker(bl, BLUE)))

    out = g.copy()
    out[mid + 1 :, mid + 1 :] = transformed
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(4, 14)
    side = 2 * n + 1
    grid = np.full((side, side), BLACK, dtype=int)
    mid = n
    grid[mid, :] = GREY
    grid[:, mid] = GREY

    seed_mask = _random_seed_mask(n)
    if seed_mask is None:
        return None
    seed_color = random.choice(SEED_COLORS)
    grid[:mid, :mid][seed_mask] = seed_color

    red_op = random.choice(OPS)
    blue_op = random.choice(OPS)
    for r, c in _marker_cells(n, red_op):
        grid[r, mid + 1 + c] = RED
    for r, c in _marker_cells(n, blue_op):
        grid[mid + 1 + r, c] = BLUE

    expected = _build_expected(grid)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
