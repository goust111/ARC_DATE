from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste, random_free_location


BLACK = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
PINK = Color.PINK
ORANGE = Color.ORANGE

GEN_RETRIES = 600
GRID_MIN = 16
GRID_MAX = 24
PAIR_COUNT_RANGE = (3, 5)
ACTIVE_COUNT_MAX = 2
ACTIVE_VANISH_PROB = 0.35

SHAPES = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
]

REPEATED_COUNT_PATTERNS = [
    [3],
    [4],
    [5],
    [2, 2],
    [3, 2],
]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    cropped = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return tuple(tuple(int(v) for v in row) for row in cropped.astype(int))


def _component_shape_key(grid: np.ndarray, component: dict) -> tuple[tuple[int, ...], ...]:
    mask = crop_bbox(grid == GREEN, component["bbox"])
    return _mask_key(mask)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return (r0 + r1) // 2, (c0 + c1) // 2


def _rotate_ccw(source: tuple[int, int], pivot: tuple[int, int]) -> tuple[int, int]:
    sr, sc = map(int, source)
    pr, pc = map(int, pivot)
    dr = sr - pr
    dc = sc - pc
    return pr - dc, pc + dr


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _pair_pivot(
    source: tuple[int, int], pivots: list[tuple[int, int]]
) -> tuple[int, int]:
    matches = [pivot for pivot in pivots if pivot[0] == source[0] or pivot[1] == source[1]]
    assert len(matches) == 1, "each source must align with exactly one pivot"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the singleton green shape to gate which source pixels rotate around their blue pivots.

    Input structure:
    - The background is black.
    - Green pixels form several separated 4-connected shapes; exactly one normalized shape appears once and the others repeat.
    - Blue pixels are isolated pivots, and pink pixels are isolated sources; each source shares a row or column with exactly one pivot.

    Rule:
    - Group the green components by their normalized bbox mask and find the unique shape instance.
    - Compute that component's bbox center cell.
    - A blue pivot is activated iff it lies in the same row or the same column as that center.
    - For each pink source, find its unique aligned pivot: if the pivot is activated, rotate the source 90 degrees counterclockwise around the pivot and paint the landing cell orange; otherwise repaint the source in place red.
    - Remove the original pink sources, keep green shapes and blue pivots unchanged, and drop any rotated source that lands outside the grid.

    Key insight:
    - Shape-frequency classification on the green objects decides which pivot-source pairs use geometry and which only recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    green_components = find_components(
        g,
        bg=BLACK,
        colors=[GREEN],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert green_components, "expected at least one green shape"

    shape_keys = [_component_shape_key(g, component) for component in green_components]
    key_counts = Counter(shape_keys)
    singleton_keys = [key for key, count in key_counts.items() if count == 1]
    assert len(singleton_keys) == 1, "expected exactly one singleton green shape"
    unique_component = next(
        component
        for component, key in zip(green_components, shape_keys)
        if key == singleton_keys[0]
    )
    center_r, center_c = _bbox_center(unique_component["bbox"])

    pivots = [tuple(map(int, coord)) for coord in np.argwhere(g == BLUE)]
    sources = [tuple(map(int, coord)) for coord in np.argwhere(g == PINK)]
    assert len(pivots) == len(sources) and pivots, "expected one pivot per source"

    activated = {
        pivot
        for pivot in pivots
        if pivot[0] == center_r or pivot[1] == center_c
    }

    out = g.copy()
    out[g == PINK] = BLACK
    for source in sources:
        pivot = _pair_pivot(source, pivots)
        if pivot in activated:
            target_r, target_c = _rotate_ccw(source, pivot)
            if _in_bounds(out.shape, target_r, target_c):
                out[target_r, target_c] = ORANGE
        else:
            out[source] = RED
    return out


def _point_is_isolated(occupied: np.ndarray, r: int, c: int) -> bool:
    h, w = occupied.shape
    if not _in_bounds((h, w), r, c) or occupied[r, c]:
        return False
    r0 = max(0, r - 1)
    r1 = min(h, r + 2)
    c0 = max(0, c - 1)
    c1 = min(w, c + 2)
    return not bool(occupied[r0:r1, c0:c1].any())


def _paint_shape(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = np.where(mask, int(color), BLACK)
    paste(grid, patch, int(top), int(left), transparent=BLACK)


def _line_clear(blocked: np.ndarray, row: int, c0: int, c1: int) -> bool:
    lo = min(int(c0), int(c1))
    hi = max(int(c0), int(c1))
    return not bool(blocked[int(row), lo : hi + 1].any())


def _mark_segment(blocked: np.ndarray, row: int, c0: int, c1: int) -> None:
    lo = min(int(c0), int(c1))
    hi = max(int(c0), int(c1))
    blocked[int(row), lo : hi + 1] = 1


def _sample_shape_multiset() -> tuple[np.ndarray, list[np.ndarray]]:
    unique_shape = random.choice(SHAPES)
    unique_key = _mask_key(unique_shape)

    pattern = random.choice(REPEATED_COUNT_PATTERNS)
    repeated_candidates = [shape for shape in SHAPES if _mask_key(shape) != unique_key]
    repeated_shapes = random.sample(repeated_candidates, len(pattern))

    placed_shapes: list[np.ndarray] = [unique_shape]
    for shape, count in zip(repeated_shapes, pattern):
        placed_shapes.extend([shape] * count)
    return unique_shape, placed_shapes[1:]


def _sample_center(mask: np.ndarray, h: int, w: int) -> tuple[int, int]:
    mh, mw = mask.shape
    row_margin = mh // 2 + 2
    col_margin = max(mw // 2 + 2, 4)
    center_r = random.randint(row_margin, h - row_margin - 1)
    center_c = random.randint(col_margin, w - col_margin - 1)
    return center_r, center_c


def _place_pairs(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    center_r: int,
    center_c: int,
) -> list[dict] | None:
    h, w = grid.shape
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    active_count = random.randint(1, min(ACTIVE_COUNT_MAX, pair_count - 1))
    vanish_index = 0 if random.random() < ACTIVE_VANISH_PROB else None

    used_rows: set[int] = set()
    used_source_cols: set[int] = set()
    used_pivot_cols: set[int] = set()
    pairs: list[dict] = []

    def add_pair(pair: dict) -> None:
        pr, pc = pair["pivot"]
        sr, sc = pair["source"]
        grid[pr, pc] = BLUE
        grid[sr, sc] = PINK
        occupied[pr, pc] = True
        occupied[sr, sc] = True
        blocked[pr, pc] = 1
        blocked[sr, sc] = 1
        _mark_segment(blocked, sr, sc, pc)
        target = pair["target"]
        if target is not None:
            blocked[target[0], target[1]] = 1
        used_rows.add(int(pr))
        used_pivot_cols.add(int(pc))
        used_source_cols.add(int(sc))
        pairs.append(pair)

    for active_idx in range(active_count):
        want_vanish = active_idx == vanish_index
        candidates: list[dict] = []
        distances = [2, 3, 4, 5]
        random.shuffle(distances)
        for r in random.sample(range(h), h):
            if r in used_rows:
                continue
            if not _point_is_isolated(occupied, r, center_c):
                continue
            for dist in distances:
                for side in random.sample([-1, 1], 2):
                    sc = center_c + side * dist
                    if not _in_bounds((h, w), r, sc):
                        continue
                    if sc in used_pivot_cols or sc in used_source_cols:
                        continue
                    if not _point_is_isolated(occupied, r, sc):
                        continue
                    if not _line_clear(blocked, r, sc, center_c):
                        continue
                    target = _rotate_ccw((r, sc), (r, center_c))
                    inside = _in_bounds((h, w), target[0], target[1])
                    if want_vanish != (not inside):
                        continue
                    if inside:
                        if blocked[target[0], target[1]]:
                            continue
                        if not _point_is_isolated(occupied, target[0], target[1]):
                            continue
                    candidates.append(
                        {
                            "pivot": (r, center_c),
                            "source": (r, sc),
                            "active": True,
                            "target": target if inside else None,
                            "vanishes": not inside,
                        }
                    )
        if not candidates:
            return None
        add_pair(random.choice(candidates))

    for _ in range(pair_count - active_count):
        candidates = []
        distances = [2, 3, 4, 5]
        random.shuffle(distances)
        for r in random.sample(range(h), h):
            if r in used_rows or r == center_r:
                continue
            pivot_cols = list(range(w))
            random.shuffle(pivot_cols)
            for pc in pivot_cols:
                if pc == center_c or pc in used_source_cols:
                    continue
                if not _point_is_isolated(occupied, r, pc):
                    continue
                for dist in distances:
                    for side in random.sample([-1, 1], 2):
                        sc = pc + side * dist
                        if not _in_bounds((h, w), r, sc):
                            continue
                        if sc == center_c or sc in used_pivot_cols or sc in used_source_cols:
                            continue
                        if not _point_is_isolated(occupied, r, sc):
                            continue
                        if not _line_clear(blocked, r, sc, pc):
                            continue
                        candidates.append(
                            {
                                "pivot": (r, pc),
                                "source": (r, sc),
                                "active": False,
                                "target": None,
                                "vanishes": False,
                            }
                        )
        if not candidates:
            return None
        add_pair(random.choice(candidates))

    return pairs


def _place_repeated_shapes(
    grid: np.ndarray,
    blocked: np.ndarray,
    repeated_shapes: list[np.ndarray],
) -> bool:
    for mask in repeated_shapes:
        patch = np.where(mask, GREEN, BLACK)
        try:
            top, left = random_free_location(
                blocked,
                patch,
                transparent=BLACK,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return False
        _paint_shape(grid, mask, top, left, GREEN)
        _paint_shape(blocked, mask, top, left, 1)
    return True


def _validate_input(grid: np.ndarray) -> dict | None:
    green_components = find_components(
        grid,
        bg=BLACK,
        colors=[GREEN],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    if not (4 <= len(green_components) <= 6):
        return None

    keys = [_component_shape_key(grid, component) for component in green_components]
    counts = Counter(keys)
    singleton_keys = [key for key, count in counts.items() if count == 1]
    if len(singleton_keys) != 1:
        return None

    unique_component = next(
        component
        for component, key in zip(green_components, keys)
        if key == singleton_keys[0]
    )
    center_r, center_c = _bbox_center(unique_component["bbox"])

    pivots = [tuple(map(int, coord)) for coord in np.argwhere(grid == BLUE)]
    sources = [tuple(map(int, coord)) for coord in np.argwhere(grid == PINK)]
    if len(pivots) != len(sources) or not (3 <= len(pivots) <= 5):
        return None

    active = 0
    inactive = 0
    vanishing = 0
    for source in sources:
        matches = [pivot for pivot in pivots if pivot[0] == source[0] or pivot[1] == source[1]]
        if len(matches) != 1:
            return None
        pivot = matches[0]
        between = grid[source[0], min(source[1], pivot[1]) : max(source[1], pivot[1]) + 1]
        expected_nonzero = 2
        if int(np.count_nonzero(between)) != expected_nonzero:
            return None
        if pivot[0] == center_r or pivot[1] == center_c:
            active += 1
            target = _rotate_ccw(source, pivot)
            if not _in_bounds(grid.shape, target[0], target[1]):
                vanishing += 1
        else:
            inactive += 1

    if active == 0 or inactive == 0:
        return None

    return {"active": active, "inactive": inactive, "vanishing": vanishing}


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)

    unique_shape, repeated_shapes = _sample_shape_multiset()
    center_r, center_c = _sample_center(unique_shape, h, w)
    top = center_r - unique_shape.shape[0] // 2
    left = center_c - unique_shape.shape[1] // 2

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=int)

    _paint_shape(grid, unique_shape, top, left, GREEN)
    _paint_shape(blocked, unique_shape, top, left, 1)
    occupied[top : top + unique_shape.shape[0], left : left + unique_shape.shape[1]] |= unique_shape

    pairs = _place_pairs(grid, occupied, blocked, center_r, center_c)
    if pairs is None:
        return None

    if not _place_repeated_shapes(grid, blocked, repeated_shapes):
        return None

    stats = _validate_input(grid)
    if stats is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
