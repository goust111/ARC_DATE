from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, paste, random_free_location


Shape = tuple[tuple[int, int], ...]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _normalize_coords(coords) -> Shape:
    pts = np.asarray(list(coords), dtype=int)
    mins = pts.min(axis=0)
    pts = pts - mins
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _shape_dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_to_mask(shape: Shape) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    rr, cc = np.asarray(shape, dtype=int).T
    mask[rr, cc] = True
    return mask


@lru_cache(maxsize=None)
def _shape_variants(shape: Shape) -> tuple[Shape, ...]:
    mask = _shape_to_mask(shape)
    variants: list[Shape] = []
    seen: set[Shape] = set()
    for k in range(4):
        variant = _normalize_coords(np.argwhere(np.rot90(mask, k)))
        if variant not in seen:
            seen.add(variant)
            variants.append(variant)
    return tuple(variants)


def _canonical_shape(shape: Shape) -> Shape:
    return min(_shape_variants(shape))


@lru_cache(maxsize=None)
def _rectangle_sizes(area: int) -> tuple[tuple[int, int], ...]:
    sizes = []
    for h in range(1, int(area ** 0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        sizes.append((h, w))
        if h != w:
            sizes.append((w, h))
    return tuple(sizes)


@lru_cache(maxsize=None)
def _can_form_rectangle(shape_a: Shape, shape_b: Shape) -> bool:
    area_a = len(shape_a)
    area_b = len(shape_b)
    total_area = area_a + area_b
    b_variants = set(_shape_variants(shape_b))

    for variant_a in _shape_variants(shape_a):
        h_a, w_a = _shape_dims(variant_a)
        cells_a = tuple(variant_a)
        for rect_h, rect_w in _rectangle_sizes(total_area):
            if h_a > rect_h or w_a > rect_w:
                continue

            for r0 in range(rect_h - h_a + 1):
                for c0 in range(rect_w - w_a + 1):
                    placed = {(r + r0, c + c0) for r, c in cells_a}
                    remainder = [
                        (r, c)
                        for r in range(rect_h)
                        for c in range(rect_w)
                        if (r, c) not in placed
                    ]
                    if _normalize_coords(remainder) in b_variants:
                        return True
    return False


def _compatibility_matrix(shapes: list[Shape]) -> list[list[bool]]:
    n = len(shapes)
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ok = _can_form_rectangle(shapes[i], shapes[j])
            adj[i][j] = ok
            adj[j][i] = ok
    return adj


def _unique_unmatched_index(adj: list[list[bool]]) -> int:
    n = len(adj)
    used = [False] * n
    solutions: list[tuple[int, tuple[tuple[int, int], ...]]] = []

    def dfs(unmatched: int | None, pairs: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return

        try:
            i = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            if unmatched is not None:
                solutions.append((unmatched, tuple(pairs)))
            return

        if unmatched is None:
            used[i] = True
            dfs(i, pairs)
            used[i] = False

        used[i] = True
        for j in range(i + 1, n):
            if not used[j] and adj[i][j]:
                used[j] = True
                pairs.append((i, j))
                dfs(unmatched, pairs)
                pairs.pop()
                used[j] = False
        used[i] = False

    dfs(None, [])
    assert len(solutions) == 1, "expected a unique near-perfect matching"
    return solutions[0][0]


def _component_shape(component: dict) -> Shape:
    return _normalize_coords(component["coords"])


def _component_crop(grid: np.ndarray, component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = component["coords"]
    out[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the only foreground shape that cannot pair with another shape to complete a solid rectangle, and crop it.

    Input structure:
    - The grid has a dominant background color and an odd number of foreground shapes, each one an 8-connected object.
    - Colors are all distinct, while positions, bounding boxes, and symmetries are distractors.
    - Ignoring color and allowing translation plus rotation, all but one shape can be partitioned into unique pairs whose unions are solid rectangles.

    Rule:
    - Extract every 8-connected foreground object and compare shapes up to rotation.
    - Build all pairings where two shapes can be translated to tile some solid rectangle without overlap or gaps.
    - Find the unique near-perfect matching of those pairings and output the bounding-box crop of the lone unmatched shape, keeping only that shape and background inside the crop.

    Key insight:
    - The target is not the odd color, size, or symmetry class; it is the only fragment whose jagged boundary has no complementary mate that seals into a rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    assert len(components) >= 3 and len(components) % 2 == 1, "expected an odd number of foreground shapes"

    shapes = [_component_shape(component) for component in components]
    unmatched = _unique_unmatched_index(_compatibility_matrix(shapes))
    return _component_crop(g, components[unmatched], bg)


# --------------- generator (o5x) ---------------

RNG = np.random.default_rng()

GRID_SIDE_RANGE = (20, 30)
PAIR_COUNT_CHOICES = (1, 2, 3)
RECT_H_RANGE = (3, 6)
RECT_W_RANGE = (4, 7)
OUTLIER_H_RANGE = (2, 5)
OUTLIER_W_RANGE = (3, 6)
OUTLIER_AREA_RANGE = (4, 10)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
RETRY_LIMIT = 400


def _mask_to_shape(mask: np.ndarray) -> Shape:
    return _normalize_coords(np.argwhere(mask))


def _shape_patch(shape: Shape, color: int, bg: int) -> np.ndarray:
    patch = np.full(_shape_dims(shape), int(bg), dtype=int)
    rr, cc = np.asarray(shape, dtype=int).T
    patch[rr, cc] = int(color)
    return patch


def _random_profile_pair(used_rectangles: set[tuple[int, int]], used_shapes: set[Shape]) -> tuple[Shape, Shape] | None:
    for _ in range(120):
        h = int(RNG.integers(RECT_H_RANGE[0], RECT_H_RANGE[1] + 1))
        w = int(RNG.integers(RECT_W_RANGE[0], RECT_W_RANGE[1] + 1))
        rect_key = tuple(sorted((h, w)))
        if rect_key in used_rectangles:
            continue

        cuts = [int(RNG.integers(1, w - 1)) for _ in range(h)]
        if len(set(cuts)) == 1:
            continue

        left = np.zeros((h, w), dtype=bool)
        for r, cut in enumerate(cuts):
            left[r, : cut + 1] = True
        right = ~left

        shape_a = _mask_to_shape(left)
        shape_b = _mask_to_shape(right)
        if min(len(shape_a), len(shape_b)) < 3:
            continue

        key_a = _canonical_shape(shape_a)
        key_b = _canonical_shape(shape_b)
        if key_a == key_b or key_a in used_shapes or key_b in used_shapes:
            continue

        used_rectangles.add(rect_key)
        used_shapes.add(key_a)
        used_shapes.add(key_b)
        return shape_a, shape_b
    return None


def _random_connected_shape() -> Shape | None:
    for _ in range(160):
        h = int(RNG.integers(OUTLIER_H_RANGE[0], OUTLIER_H_RANGE[1] + 1))
        w = int(RNG.integers(OUTLIER_W_RANGE[0], OUTLIER_W_RANGE[1] + 1))
        max_area = min(h * w - 1, OUTLIER_AREA_RANGE[1])
        min_area = min(max_area, OUTLIER_AREA_RANGE[0])
        if max_area < min_area:
            continue
        target = int(RNG.integers(min_area, max_area + 1))

        start = (int(RNG.integers(h)), int(RNG.integers(w)))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(RNG.integers(len(frontier)))])

        if len(cells) != target:
            continue

        shape = _normalize_coords(cells)
        sh, sw = _shape_dims(shape)
        if sh < 2 or sw < 2:
            continue
        if len(shape) == sh * sw:
            continue
        return shape
    return None


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, bg: int) -> bool:
    footprint = np.ones(patch.shape, dtype=int)
    try:
        r0, c0 = random_free_location(
            occupied,
            footprint,
            transparent=0,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
            padding_connectivity=8,
        )
    except ValueError:
        return False

    paste(grid, patch, r0, c0, transparent=int(bg))
    paste(occupied, footprint, r0, c0)
    return True


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    bg = int(RNG.integers(10))
    side_h = int(RNG.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1))
    side_w = int(RNG.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1))
    pair_count = int(RNG.choice(PAIR_COUNT_CHOICES))

    used_rectangles: set[tuple[int, int]] = set()
    used_shapes: set[Shape] = set()
    shapes: list[Shape] = []

    for _ in range(pair_count):
        pair = _random_profile_pair(used_rectangles, used_shapes)
        if pair is None:
            return None
        shapes.extend(pair)

    outlier = _random_connected_shape()
    if outlier is None:
        return None
    outlier_key = _canonical_shape(outlier)
    if outlier_key in used_shapes:
        return None
    used_shapes.add(outlier_key)
    shapes.append(outlier)

    colors = list(range(10))
    colors.remove(bg)
    if len(shapes) > len(colors):
        return None
    RNG.shuffle(colors)

    order = list(range(len(shapes)))
    RNG.shuffle(order)

    grid = np.full((side_h, side_w), bg, dtype=int)
    occupied = np.zeros((side_h, side_w), dtype=int)
    expected = None

    for idx, shape_index in enumerate(order):
        shape = shapes[shape_index]
        variants = _shape_variants(shape)
        variant = variants[int(RNG.integers(len(variants)))]
        color = colors[idx]
        patch = _shape_patch(variant, color, bg)
        if not _place_patch(grid, occupied, patch, bg):
            return None
        if shape_index == len(shapes) - 1:
            expected = patch

    assert expected is not None
    return grid, expected


def _valid_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        built = _build_candidate()
        if built is None:
            continue
        grid, expected = built
        if _valid_generated(grid, expected):
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
