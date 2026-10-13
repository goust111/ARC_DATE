from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
SPRITE_COLORS = (1, 2, 3, 4)
SHAPE_SIZE_RANGE = range(3, 7)
SHAPE_BOX_SIDE = 3
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _shape_key_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    origin = coords.min(axis=0)
    shifted = coords - origin
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_key_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _shape_key_from_coords(np.argwhere(mask))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_to_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _rotate_shape(shape: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    patch = _shape_to_patch(shape, 1)
    rot = np.rot90(patch, int(turns))
    return _shape_key_from_mask(rot == 1)


def _tile_patch(tile: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = map(int, out_shape)
    tile_h, tile_w = tile.shape
    row_idx = np.arange(out_h) % tile_h
    col_idx = np.arange(out_w) % tile_w
    return tile[row_idx[:, None], col_idx[None, :]]


def _separator_indices(grid: np.ndarray) -> tuple[int, int]:
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(sep_rows) == 1 and len(sep_cols) == 1, "expected one full grey separator row and column"
    return int(sep_rows[0]), int(sep_cols[0])


def _quadrants(grid: np.ndarray) -> list[np.ndarray]:
    sep_r, sep_c = _separator_indices(grid)
    quads = [
        grid[:sep_r, :sep_c],
        grid[:sep_r, sep_c + 1 :],
        grid[sep_r + 1 :, :sep_c],
        grid[sep_r + 1 :, sep_c + 1 :],
    ]
    qshape = quads[0].shape
    assert qshape[0] > 0 and qshape[1] > 0
    assert all(q.shape == qshape for q in quads), "quadrants must have equal size"
    return quads


def _extract_sprite(quadrant: np.ndarray) -> dict[str, object]:
    fg = (quadrant != BLACK) & (quadrant != GREY)
    assert np.any(fg), "each quadrant must contain a sprite"
    colors = np.unique(quadrant[fg])
    assert len(colors) == 1, "sprite must be monochromatic"
    comps = _components4(fg)
    assert len(comps) == 1, "quadrant must contain one 4-connected sprite"
    bb = _bbox(fg)
    assert bb is not None
    patch = _crop_bbox(quadrant, bb)
    patch_mask = patch != BLACK
    shape_key = _shape_key_from_mask(patch_mask)
    return {
        "color": int(colors[0]),
        "patch": patch,
        "mask": patch_mask,
        "shape_key": shape_key,
    }


def _expected_output_from_grid(grid: np.ndarray) -> np.ndarray:
    quads = _quadrants(grid)
    sprites = [_extract_sprite(quad) for quad in quads]
    counts = Counter(sprite["shape_key"] for sprite in sprites)
    assert sorted(counts.values()) == [1, 3], "expected exactly one abnormal sprite"
    odd_key = next(key for key, freq in counts.items() if freq == 1)
    odd_patch = next(sprite["patch"] for sprite in sprites if sprite["shape_key"] == odd_key)
    return _tile_patch(np.asarray(odd_patch, dtype=int), quads[0].shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the uniquely shaped quadrant sprite, crop it, and tile it across a single quadrant canvas.

    Input structure:
    - A black background grid has one full grey separator row and one full grey separator column crossing at the center.
    - Those separators divide the grid into four equal quadrants, each containing exactly one monochromatic 4-connected sprite.
    - The four sprite colors are all different and are non-black, non-grey colors.

    Rule:
    - Remove the separator row and column conceptually and inspect each quadrant independently.
    - Crop each quadrant's sprite to its tight bounding box and compare the normalized binary shapes, ignoring color and position.
    - Three quadrants share one normalized shape and one quadrant has the abnormal shape; crop that abnormal sprite with its original color and tile that cropped patch from the top-left to fill a grid the size of one quadrant.

    Key insight:
    - Color is a distraction: the only reliable way to identify the output seed is to normalize all four quadrant sprites and pick the lone shape outlier before tiling it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    return _expected_output_from_grid(g)


def _enumerate_shapes() -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    shapes: list[tuple[tuple[int, int], ...]] = []
    cells = list(itertools.product(range(SHAPE_BOX_SIDE), repeat=2))
    for size in SHAPE_SIZE_RANGE:
        for combo in itertools.combinations(cells, size):
            mask = np.zeros((SHAPE_BOX_SIDE, SHAPE_BOX_SIDE), dtype=bool)
            for r, c in combo:
                mask[r, c] = True
            if len(_components4(mask)) != 1:
                continue
            key = _shape_key_from_mask(mask)
            if key in seen:
                continue
            seen.add(key)
            shapes.append(key)
    shapes.sort(key=lambda shape: (len(shape), _shape_dims(shape), shape))
    return shapes


ALL_SHAPES = _enumerate_shapes()
SHAPE_SET = set(ALL_SHAPES)


def _eligible_shapes(max_h: int, max_w: int) -> list[tuple[tuple[int, int], ...]]:
    return [shape for shape in ALL_SHAPES if _shape_dims(shape)[0] <= max_h and _shape_dims(shape)[1] <= max_w]


def _choose_abnormal_shape(
    normal_shape: tuple[tuple[int, int], ...],
    eligible: list[tuple[tuple[int, int], ...]],
) -> tuple[tuple[int, int], ...]:
    normal_area = len(normal_shape)
    normal_dims = _shape_dims(normal_shape)

    rotated_candidates = [
        rot
        for turns in (1, 2, 3)
        for rot in [_rotate_shape(normal_shape, turns)]
        if rot != normal_shape and rot in SHAPE_SET and rot in eligible
    ]
    same_area_same_box = [
        shape
        for shape in eligible
        if shape != normal_shape and len(shape) == normal_area and _shape_dims(shape) == normal_dims
    ]
    same_area = [shape for shape in eligible if shape != normal_shape and len(shape) == normal_area]
    different_area = [shape for shape in eligible if len(shape) != normal_area]

    buckets = [
        rotated_candidates,
        same_area_same_box,
        same_area,
        different_area,
    ]
    weights = [2, 3, 3, 2]
    order: list[list[tuple[tuple[int, int], ...]]] = []
    for idx in random.choices(range(len(buckets)), weights=weights, k=len(buckets)):
        bucket = buckets[idx]
        if bucket not in order:
            order.append(bucket)
    for bucket in buckets:
        if bucket not in order:
            order.append(bucket)
    for bucket in order:
        if bucket:
            return random.choice(bucket)
    raise RuntimeError("no abnormal shape candidate found")


def _quadrant_origin(index: int, sep_r: int, sep_c: int) -> tuple[int, int]:
    return (
        0 if index < 2 else sep_r + 1,
        0 if index % 2 == 0 else sep_c + 1,
    )


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    assert view.shape == patch.shape
    assert not np.any((view != BLACK) & (patch != BLACK))
    mask = patch != BLACK
    view[mask] = patch[mask]


def _build_input(
    quadrant_shape: tuple[int, int],
    normal_shape: tuple[tuple[int, int], ...],
    abnormal_shape: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    qh, qw = quadrant_shape
    full = np.zeros((qh * 2 + 1, qw * 2 + 1), dtype=int)
    sep_r, sep_c = qh, qw
    full[sep_r, :] = GREY
    full[:, sep_c] = GREY

    abnormal_idx = random.randrange(4)
    colors = random.sample(list(SPRITE_COLORS), 4)
    abnormal_patch = _shape_to_patch(abnormal_shape, colors[abnormal_idx])

    for idx in range(4):
        shape = abnormal_shape if idx == abnormal_idx else normal_shape
        patch = _shape_to_patch(shape, colors[idx])
        base_r, base_c = _quadrant_origin(idx, sep_r, sep_c)
        h, w = patch.shape
        top = base_r + random.randint(0, qh - h)
        left = base_c + random.randint(0, qw - w)
        _place_patch(full, patch, top, left)

    expected = _tile_patch(abnormal_patch, quadrant_shape)
    return full, expected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        sep_r, sep_c = _separator_indices(inp)
        quads = _quadrants(inp)
        if sep_r * 2 + 1 != inp.shape[0] or sep_c * 2 + 1 != inp.shape[1]:
            return False
        sprites = [_extract_sprite(quad) for quad in quads]
        colors = [sprite["color"] for sprite in sprites]
        if sorted(colors) != [1, 2, 3, 4]:
            return False
        counts = Counter(sprite["shape_key"] for sprite in sprites)
        if sorted(counts.values()) != [1, 3]:
            return False
        odd_key = next(key for key, freq in counts.items() if freq == 1)
        odd_patch = next(sprite["patch"] for sprite in sprites if sprite["shape_key"] == odd_key)
        if odd_patch.shape[0] * 2 > quads[0].shape[0] or odd_patch.shape[1] * 2 > quads[0].shape[1]:
            return False
        if not np.array_equal(expected, _tile_patch(np.asarray(odd_patch, dtype=int), quads[0].shape)):
            return False
        if np.array_equal(expected, np.zeros_like(expected)):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        qh = random.randint(6, 12)
        qw = random.randint(6, 12)
        eligible = _eligible_shapes(qh // 2, qw // 2)
        if len(eligible) < 2:
            continue
        normal_shape = random.choice(eligible)
        abnormal_shape = _choose_abnormal_shape(normal_shape, eligible)
        inp, expected = _build_input((qh, qw), normal_shape, abnormal_shape)
        if not _semantic_checks(inp, expected):
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle input")
