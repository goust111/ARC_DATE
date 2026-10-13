from __future__ import annotations

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4)
SEPARATOR = 5
QUADRANT_ORDER = ("ul", "ur", "ll", "lr")
DIAGONAL_INDEX = {0: 3, 1: 2, 2: 1, 3: 0}
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _boundary_zero_reachable(zero_mask: np.ndarray) -> np.ndarray:
    h, w = zero_mask.shape
    reachable = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if zero_mask[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if zero_mask[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(zero_mask.shape, nr, nc) and zero_mask[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                stack.append((nr, nc))
    return reachable


def _hole_components(crop: np.ndarray) -> list[np.ndarray]:
    zero_mask = crop == BG
    exterior = _boundary_zero_reachable(zero_mask)
    holes = zero_mask & ~exterior
    return _component_coords(holes)


def _is_solid_square(coords: np.ndarray) -> bool:
    if coords.size == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height != width:
        return False
    mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = True
    return bool(mask.all())


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = np.where(np.all(grid == SEPARATOR, axis=1))[0]
    cols = np.where(np.all(grid == SEPARATOR, axis=0))[0]
    assert len(rows) == 1 and len(cols) == 1
    cross_r = int(rows[0])
    cross_c = int(cols[0])
    assert 0 < cross_r < grid.shape[0] - 1
    assert 0 < cross_c < grid.shape[1] - 1
    return cross_r, cross_c


def _quadrant_regions(grid: np.ndarray, cross_r: int, cross_c: int) -> list[np.ndarray]:
    h, w = grid.shape
    return [
        grid[:cross_r, :cross_c],
        grid[:cross_r, cross_c + 1 : w],
        grid[cross_r + 1 : h, :cross_c],
        grid[cross_r + 1 : h, cross_c + 1 : w],
    ]


def _extract_sprite_crop(region: np.ndarray) -> tuple[np.ndarray, int]:
    colors = sorted(int(v) for v in np.unique(region) if int(v) != BG)
    assert len(colors) == 1
    box = _bbox_from_mask(region != BG)
    assert box is not None
    return _crop(region, box), colors[0]


def _fill_square_holes(crop: np.ndarray, fill_color: int) -> np.ndarray:
    out = crop.copy()
    for hole in _hole_components(crop):
        if _is_solid_square(hole):
            rows, cols = hole[:, 0], hole[:, 1]
            out[rows, cols] = int(fill_color)
    return out


def _pack_montage(crops: list[np.ndarray]) -> np.ndarray:
    top_h = max(crops[0].shape[0], crops[1].shape[0])
    bottom_h = max(crops[2].shape[0], crops[3].shape[0])
    left_w = max(crops[0].shape[1], crops[2].shape[1])
    right_w = max(crops[1].shape[1], crops[3].shape[1])
    out = np.zeros((top_h + bottom_h, left_w + right_w), dtype=int)

    placements = (
        (0, 0, crops[0]),
        (0, left_w, crops[1]),
        (top_h, 0, crops[2]),
        (top_h, left_w, crops[3]),
    )
    for r0, c0, crop in placements:
        h, w = crop.shape
        out[r0 : r0 + h, c0 : c0 + w] = crop
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop four quadrant sprites, fill only square holes using diagonal sprite colors, and pack the results into a 2x2 montage.

    Input structure:
    - The grid has black background with one full gray row and one full gray column forming a separator cross.
    - The cross divides the canvas into four quadrants, each containing exactly one single-color 4-connected sprite.
    - Each sprite has an internal black hole preserved inside its tight bounding box.

    Rule:
    - Remove the separator cross logically by splitting the grid into upper-left, upper-right, lower-left, and lower-right quadrants.
    - Crop the unique sprite in each quadrant to its tight bounding box while preserving black hole pixels.
    - For each cropped sprite, inspect every enclosed black hole: if a hole is a completely filled square block, recolor that hole with the main color of the diagonal quadrant sprite; otherwise leave it black.
    - Pack the four processed crops back into a compact 2x2 montage in quadrant order.

    Key insight:
    - The fill decision depends on local hole geometry, but the fill color comes from a different object: the diagonally opposite quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cross_r, cross_c = _find_separator_cross(g)
    regions = _quadrant_regions(g, cross_r, cross_c)
    crops_and_colors = [_extract_sprite_crop(region) for region in regions]
    colors = [color for _crop_grid, color in crops_and_colors]

    processed = []
    for idx, (crop, _color) in enumerate(crops_and_colors):
        processed.append(_fill_square_holes(crop, colors[DIAGONAL_INDEX[idx]]))
    return _pack_montage(processed)


IRREGULAR_PATTERNS = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
)

ALLOWED_DIMS = {
    (5, 5),
    (5, 6),
    (6, 5),
    (6, 6),
    (7, 5),
    (5, 7),
}

MAX_GENERATION_ATTEMPTS = 500


def _foreground_connected(mask: np.ndarray) -> bool:
    return len(_component_coords(mask)) == 1


def _exact_internal_hole(mask: np.ndarray, expected_hole: np.ndarray) -> bool:
    holes = _hole_components(mask.astype(int))
    if len(holes) != 1:
        return False
    actual = np.zeros_like(mask, dtype=bool)
    rows, cols = holes[0][:, 0], holes[0][:, 1]
    actual[rows, cols] = True
    return bool(np.array_equal(actual, expected_hole))


def _removable_groups(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    groups: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if mask[r, c] and (r in (0, h - 1) or c in (0, w - 1)):
                groups.append([(r, c)])

    for c in range(w - 1):
        if mask[0, c] and mask[0, c + 1]:
            groups.append([(0, c), (0, c + 1)])
        if mask[h - 1, c] and mask[h - 1, c + 1]:
            groups.append([(h - 1, c), (h - 1, c + 1)])
    for r in range(h - 1):
        if mask[r, 0] and mask[r + 1, 0]:
            groups.append([(r, 0), (r + 1, 0)])
        if mask[r, w - 1] and mask[r + 1, w - 1]:
            groups.append([(r, w - 1), (r + 1, w - 1)])
    return groups


def _carve_notches(base_mask: np.ndarray, hole_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    mask = base_mask.copy()
    area = int(mask.sum())
    min_removals = max(2, area - 30)
    max_removals = min(area - 15, 8)
    if min_removals > max_removals:
        return None
    target_removals = int(rng.integers(min_removals, max_removals + 1))
    removed = 0

    for _ in range(80):
        if removed >= target_removals:
            break
        groups = _removable_groups(mask)
        rng.shuffle(groups)
        changed = False
        for group in groups:
            if removed + len(group) > target_removals:
                continue
            candidate = mask.copy()
            if any(not candidate[r, c] for r, c in group):
                continue
            for r, c in group:
                candidate[r, c] = False
            if not _foreground_connected(candidate):
                continue
            if _bbox_from_mask(candidate) != (0, 0, candidate.shape[0] - 1, candidate.shape[1] - 1):
                continue
            if not _exact_internal_hole(candidate, hole_mask):
                continue
            mask = candidate
            removed += len(group)
            changed = True
            break
        if not changed:
            return None

    if removed < 2 or removed != target_removals:
        return None
    if not (15 <= int(mask.sum()) <= 30):
        return None
    return mask


def _square_hole_pattern(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    options = []
    for size in (2, 3):
        if height < size + 2 or width < size + 2:
            continue
        if height * width - size * size < 17:
            continue
        options.append(size)
    assert options
    size = int(rng.choice(options))
    hole = np.zeros((height, width), dtype=bool)
    r0 = int(rng.integers(1, height - size))
    c0 = int(rng.integers(1, width - size))
    hole[r0 : r0 + size, c0 : c0 + size] = True
    return hole


def _irregular_hole_pattern(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    candidates = []
    for pattern in IRREGULAR_PATTERNS:
        ph, pw = pattern.shape
        if height < ph + 2 or width < pw + 2:
            continue
        candidates.append(pattern)
    assert candidates
    pattern = candidates[int(rng.integers(0, len(candidates)))]
    ph, pw = pattern.shape
    hole = np.zeros((height, width), dtype=bool)
    r0 = int(rng.integers(1, height - ph))
    c0 = int(rng.integers(1, width - pw))
    hole[r0 : r0 + ph, c0 : c0 + pw] = pattern
    return hole


def _build_crop(height: int, width: int, color: int, square_hole: bool, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(120):
        hole_mask = (
            _square_hole_pattern(height, width, rng)
            if square_hole
            else _irregular_hole_pattern(height, width, rng)
        )
        base_mask = np.ones((height, width), dtype=bool)
        base_mask[hole_mask] = False
        carved = _carve_notches(base_mask, hole_mask, rng)
        if carved is None:
            continue
        crop = np.zeros((height, width), dtype=int)
        crop[carved] = int(color)
        return crop, hole_mask
    raise RuntimeError("failed to build sprite crop")


def _sample_crop_dims(rng: np.random.Generator) -> tuple[int, int, int, int]:
    while True:
        top_h = int(rng.integers(5, 8))
        bottom_h = int(rng.integers(5, 8))
        left_w = int(rng.integers(5, 8))
        right_w = int(rng.integers(5, 8))
        dims = (
            (top_h, left_w),
            (top_h, right_w),
            (bottom_h, left_w),
            (bottom_h, right_w),
        )
        if all(dim in ALLOWED_DIMS for dim in dims):
            return top_h, bottom_h, left_w, right_w


def _expected_from_specs(crops: list[np.ndarray], colors: list[int], square_flags: list[bool]) -> np.ndarray:
    filled = []
    for idx, crop in enumerate(crops):
        if square_flags[idx]:
            filled.append(_fill_square_holes(crop, colors[DIAGONAL_INDEX[idx]]))
        else:
            filled.append(crop.copy())
    top = np.hstack(filled[:2])
    bottom = np.hstack(filled[2:])
    return np.vstack((top, bottom))


def _quadrant_boxes(grid_shape: tuple[int, int], cross_r: int, cross_c: int):
    h, w = grid_shape
    return [
        (0, 0, cross_r, cross_c),
        (0, cross_c + 1, cross_r, w - cross_c - 1),
        (cross_r + 1, 0, h - cross_r - 1, cross_c),
        (cross_r + 1, cross_c + 1, h - cross_r - 1, w - cross_c - 1),
    ]


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _validate_input_semantics(
    grid: np.ndarray,
    cross_r: int,
    cross_c: int,
    crops: list[np.ndarray],
    colors: list[int],
    square_flags: list[bool],
    expected: np.ndarray,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if len(set(colors)) != 4:
        return False
    if sum(square_flags) != 2:
        return False
    if len(np.where(np.all(grid == SEPARATOR, axis=1))[0]) != 1:
        return False
    if len(np.where(np.all(grid == SEPARATOR, axis=0))[0]) != 1:
        return False

    boxes = _quadrant_boxes(grid.shape, cross_r, cross_c)
    observed_square = []
    for idx, (r0, c0, h, w) in enumerate(boxes):
        region = grid[r0 : r0 + h, c0 : c0 + w]
        present = sorted(int(v) for v in np.unique(region) if int(v) not in (BG, SEPARATOR))
        if present != [colors[idx]]:
            return False
        box = _bbox_from_mask(region != BG)
        if box is None:
            return False
        crop = _crop(region, box)
        if not np.array_equal(crop, crops[idx]):
            return False
        holes = _hole_components(crop)
        if len(holes) != 1:
            return False
        observed_square.append(_is_solid_square(holes[0]))

    if observed_square != square_flags:
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    top_h, bottom_h, left_w, right_w = _sample_crop_dims(rng)
    dims = [
        (top_h, left_w),
        (top_h, right_w),
        (bottom_h, left_w),
        (bottom_h, right_w),
    ]

    color_order = [int(v) for v in rng.permutation(PALETTE)]
    square_flags = [False] * 4
    square_indices = rng.choice(4, size=2, replace=False)
    for idx in square_indices:
        square_flags[int(idx)] = True

    crops: list[np.ndarray] = []
    for idx, (height, width) in enumerate(dims):
        crop, _hole_mask = _build_crop(height, width, color_order[idx], square_flags[idx], rng)
        crops.append(crop)

    expected = _expected_from_specs(crops, color_order, square_flags)
    if max(expected.shape) > 30:
        return None

    top_region_h = top_h + int(rng.integers(2, 5))
    bottom_region_h = bottom_h + int(rng.integers(2, 5))
    left_region_w = left_w + int(rng.integers(2, 5))
    right_region_w = right_w + int(rng.integers(2, 5))
    grid_h = top_region_h + 1 + bottom_region_h
    grid_w = left_region_w + 1 + right_region_w
    if grid_h % 2 == 0 or grid_w % 2 == 0 or max(grid_h, grid_w) > 30:
        return None

    cross_r = top_region_h
    cross_c = left_region_w
    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[cross_r, :] = SEPARATOR
    grid[:, cross_c] = SEPARATOR

    boxes = _quadrant_boxes(grid.shape, cross_r, cross_c)
    for crop, (r0, c0, h, w) in zip(crops, boxes):
        max_r_offset = h - crop.shape[0]
        max_c_offset = w - crop.shape[1]
        if max_r_offset < 2 or max_c_offset < 2:
            return None
        rr = r0 + int(rng.integers(1, max_r_offset))
        cc = c0 + int(rng.integers(1, max_c_offset))
        _paste(grid, crop, rr, cc)

    if not _validate_input_semantics(grid, cross_r, cross_c, crops, color_order, square_flags, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-hole ARC input")
