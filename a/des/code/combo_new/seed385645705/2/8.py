from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLOR_IDS = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_SIZE_RANGE = (13, 29)
OBJECT_COUNT_RANGE = (2, 5)
OBJECT_HEIGHT_RANGE = (3, 8)
OBJECT_WIDTH_RANGE = (3, 8)
MIN_HALF_AREA = 2
PLACEMENT_TRIES = 80
GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(coords)
    return comps


def _has_boundary_contact(coords: set[tuple[int, int]], axis: int, threshold: int) -> bool:
    if axis == 0:
        return any(r == threshold and (r + 1, c) in coords for r, c in coords)
    return any(c == threshold and (r, c + 1) in coords for r, c in coords)


def _axis_split_ok(g: np.ndarray, coords: list[tuple[int, int]], colors: tuple[int, int], axis: int) -> bool:
    values = [r if axis == 0 else c for r, c in coords]
    coord_set = set(coords)
    lo, hi = min(values), max(values)
    if lo == hi:
        return False

    for threshold in range(lo, hi):
        if not _has_boundary_contact(coord_set, axis, threshold):
            continue
        for first, second in (colors, (colors[1], colors[0])):
            left_seen = False
            right_seen = False
            valid = True
            for r, c in coords:
                v = r if axis == 0 else c
                if v <= threshold:
                    left_seen = True
                    expected = first
                else:
                    right_seen = True
                    expected = second
                if int(g[r, c]) != expected:
                    valid = False
                    break
            if valid and left_seen and right_seen:
                return True
    return False


def _has_straight_split(g: np.ndarray, coords: list[tuple[int, int]], colors: tuple[int, int]) -> bool:
    return _axis_split_ok(g, coords, colors, axis=0) or _axis_split_ok(g, coords, colors, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the two pure-color halves inside each straight-split foreground object.

    Input structure:
    - The background is black.
    - The non-black cells form several separated 4-connected foreground objects.
    - Each object contains exactly two colors, and its two color regions are
      separated by one hidden horizontal or vertical straight boundary.
    - Different objects may reuse individual colors, but their two-color
      combinations are distinct, so the swap must be read object by object.

    Rule:
    - Find every non-black 4-connected foreground object.
    - For each object, confirm that some horizontal or vertical line separates
      all cells of one color from all cells of the other color.
    - Within that object only, recolor every cell of the first color to the
      second color and every cell of the second color to the first color.
    - Leave black background cells unchanged and keep the grid size unchanged.

    Key insight:
    The visible clue is the local straight split inside each connected object;
    the same color can belong to different pairings in different objects, so a
    single global color replacement is not the rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    out = g.copy()
    comps = _components(g != BLACK)
    assert comps, "expected at least one foreground object"

    seen_pairs: set[frozenset[int]] = set()
    for coords in comps:
        colors = tuple(int(v) for v in np.unique([g[r, c] for r, c in coords]))
        assert len(colors) == 2, "each object must use exactly two colors"
        pair = frozenset(colors)
        assert pair not in seen_pairs, "object color pairs must be distinct"
        seen_pairs.add(pair)
        assert _has_straight_split(g, coords, colors), "object colors must have a straight split"

        a, b = colors
        for r, c in coords:
            out[r, c] = b if int(g[r, c]) == a else a

    return out


def _row_interval(start: int, stop: int, height: int) -> tuple[int, int]:
    start = max(0, min(start, height - 1))
    stop = max(0, min(stop, height - 1))
    if start > stop:
        start, stop = stop, start
    return start, stop


def _sample_overlapping_row_ranges(height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    for _ in range(80):
        a0 = random.randint(0, height - 1)
        a1 = random.randint(a0, height - 1)
        b0 = random.randint(0, height - 1)
        b1 = random.randint(b0, height - 1)
        if min(a1, b1) >= max(a0, b0) and a1 - a0 + 1 >= 2 and b1 - b0 + 1 >= 2:
            return (a0, a1), (b0, b1)

    center = random.randint(1, height - 2)
    return _row_interval(center - 1, height - 1, height), _row_interval(0, center + 1, height)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_like(mask: np.ndarray, arrays: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    r0, c0, r1, c1 = _mask_bbox(mask)
    return tuple(arr[r0 : r1 + 1, c0 : c1 + 1] for arr in arrays)


def _make_vertical_mask(height: int, width: int, cut: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    left_range, right_range = _sample_overlapping_row_ranges(height)
    mask = np.zeros((height, width), dtype=bool)
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)

    for r in range(left_range[0], left_range[1] + 1):
        depth = random.randint(1, cut)
        left[r, cut - depth : cut] = True
    for r in range(right_range[0], right_range[1] + 1):
        depth = random.randint(1, width - cut)
        right[r, cut : cut + depth] = True

    mask[left | right] = True
    if int(left.sum()) < MIN_HALF_AREA or int(right.sum()) < MIN_HALF_AREA:
        return None
    if not np.any(left[:, cut - 1] & right[:, cut]):
        return None
    return _crop_like(mask, (mask, left, right))


def _make_split_mask(orientation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    height = random.randint(*OBJECT_HEIGHT_RANGE)
    width = random.randint(*OBJECT_WIDTH_RANGE)

    if orientation == "vertical":
        cut = random.randint(1, width - 1)
        result = _make_vertical_mask(height, width, cut)
        if result is None:
            return None
        mask, first, second = result
    else:
        cut = random.randint(1, height - 1)
        result = _make_vertical_mask(width, height, cut)
        if result is None:
            return None
        mask, first, second = (arr.T for arr in result)

    if len(_components(mask)) != 1 or len(_components(first)) != 1 or len(_components(second)) != 1:
        return None
    return mask, first, second


def _choose_color_pairs(count: int) -> list[tuple[int, int]]:
    colors = list(COLOR_IDS)
    random.shuffle(colors)
    hub = colors[0]
    others = colors[1:]
    pairs = [(hub, other) for other in others[:count]]
    for i, (a, b) in enumerate(pairs):
        if random.random() < 0.5:
            pairs[i] = (b, a)
    random.shuffle(pairs)
    return pairs


def _padded_region(mask: np.ndarray, r0: int, c0: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    h, w = mask.shape
    gr, gc = shape
    return slice(max(0, r0 - 1), min(gr, r0 + h + 1)), slice(max(0, c0 - 1), min(gc, c0 + w + 1))


def _paint_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    pair: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]] | None:
    h, w = mask.shape
    gr, gc = grid.shape
    if h > gr - 2 or w > gc - 2:
        return None

    positions = []
    for r0 in range(1, gr - h):
        for c0 in range(1, gc - w):
            rs, cs = _padded_region(mask, r0, c0, grid.shape)
            if not occupied[rs, cs].any():
                positions.append((r0, c0))
    if not positions:
        return None

    random.shuffle(positions)
    for r0, c0 in positions[:PLACEMENT_TRIES]:
        patch = np.zeros_like(mask, dtype=int)
        patch[first] = pair[0]
        patch[second] = pair[1]
        view = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any((patch != BLACK) & (view != BLACK)):
            continue
        view[patch != BLACK] = patch[patch != BLACK]
        rs, cs = _padded_region(mask, r0, c0, grid.shape)
        occupied[rs, cs] = True

        expected_patch = np.zeros_like(mask, dtype=int)
        expected_patch[first] = pair[1]
        expected_patch[second] = pair[0]
        return expected_patch, (r0, c0)
    return None


def _orientation_plan(count: int) -> list[str]:
    orientations = ["vertical", "horizontal"]
    while len(orientations) < count:
        orientations.append(random.choice(["vertical", "horizontal"]))
    random.shuffle(orientations)
    return orientations[:count]


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_SIZE_RANGE)
    cols = random.randint(*GRID_SIZE_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)
    grid = np.full((rows, cols), BLACK, dtype=int)
    expected = grid.copy()
    occupied = np.zeros_like(grid, dtype=bool)

    pairs = _choose_color_pairs(count)
    orientations = _orientation_plan(count)

    for pair, orientation in zip(pairs, orientations, strict=False):
        shape = None
        for _ in range(80):
            shape = _make_split_mask(orientation)
            if shape is not None:
                break
        if shape is None:
            return None
        mask, first, second = shape
        placed = _paint_object(grid, occupied, mask, first, second, pair)
        if placed is None:
            return None
        expected_patch, (r0, c0) = placed
        h, w = mask.shape
        expected[r0 : r0 + h, c0 : c0 + w][mask] = expected_patch[mask]

    try:
        result = main(grid)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
