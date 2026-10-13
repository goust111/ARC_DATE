import math
import random
from collections import deque

import numpy as np


_RNG = random.Random()
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_CANVAS_SIDE = 30
_GEN_RETRIES = 400


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in _DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [pt[0] for pt in cells]
            cols = [pt[1] for pt in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in cells:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "area": len(cells),
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return comps


def _downsample_if_uniform(mask: np.ndarray, factor: int) -> np.ndarray | None:
    h, w = mask.shape
    if h % factor != 0 or w % factor != 0:
        return None
    blocks = mask.reshape(h // factor, factor, w // factor, factor)
    reduced = blocks[:, 0, :, 0]
    if np.all(blocks == reduced[:, None, :, None]):
        return reduced.copy()
    return None


def _primitive_mask_and_scale(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    g = math.gcd(h, w)
    for factor in range(g, 0, -1):
        if g % factor != 0:
            continue
        reduced = _downsample_if_uniform(mask, factor)
        if reduced is not None:
            return reduced, factor
    raise AssertionError("every mask should admit factor 1")


def _upsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize same-template scaled shapes to the largest size and overlay them by area order.

    Input structure:
    - The background is black.
    - Several non-black objects appear as monochrome 4-connected components with distinct colors.
    - Every object is the same binary template under integer block scaling, but each uses a different scale.

    Rule:
    - Extract every colored component and recover its primitive binary template by dividing out the largest valid block scale.
    - Verify that all recovered primitive templates match, then take the largest recovered scale as the target size.
    - Re-expand the common template to that target size and draw one copy for each object at the same origin, ordered from largest original area to smallest original area.
    - Later draws overwrite earlier ones, so the final visible color is determined by the smallest original object on the shared normalized shape.

    Key insight:
    - After factoring out block scaling, all objects reduce to the same primitive template; expanding each one to the largest observed scale makes all masks coincide exactly, so only the draw order controls the final color layering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g, bg=0)
    assert len(components) >= 2, "expected multiple colored components"
    assert len({comp["color"] for comp in components}) == len(components), "component colors must be distinct"

    primitive = None
    scales = []
    for comp in components:
        reduced, scale = _primitive_mask_and_scale(comp["mask"])
        if primitive is None:
            primitive = reduced
        else:
            assert primitive.shape == reduced.shape and np.array_equal(primitive, reduced), "all objects must share one template"
        scales.append(scale)

    assert primitive is not None and primitive.any(), "primitive template must be non-empty"

    target_scale = max(scales)
    target_mask = _upsample_mask(primitive, target_scale)
    out = np.zeros(target_mask.shape, dtype=int)
    for comp in sorted(components, key=lambda item: item["area"], reverse=True):
        out[target_mask] = int(comp["color"])
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _transform_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    if rng.random() < 0.5:
        out = np.fliplr(out)
    turns = rng.randrange(4)
    out = np.rot90(out, turns)
    return np.array(out, dtype=bool)


def _make_l_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 5)
    w = rng.randint(2, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _make_t_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 5)
    w = rng.choice([3, 5])
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, w // 2] = True
    return mask


def _make_plus_shape(rng: random.Random) -> np.ndarray:
    h = rng.choice([3, 5])
    w = rng.choice([3, 5])
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2, :] = True
    mask[:, w // 2] = True
    return mask


def _make_frame_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _make_random_polyomino(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 5)
    w = rng.randint(2, 5)
    target = rng.randint(max(3, min(h, w) + 1), min(h * w - 1, h + w + 3))
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in _DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _random_primitive_mask(rng: random.Random) -> np.ndarray:
    builders = [
        _make_l_shape,
        _make_t_shape,
        _make_plus_shape,
        _make_frame_shape,
        _make_random_polyomino,
    ]
    for _ in range(100):
        mask = rng.choice(builders)(rng)
        mask = _crop_mask(mask)
        mask = _transform_mask(mask, rng)
        if min(mask.shape) < 2:
            continue
        if int(mask.sum()) < 3:
            continue
        reduced, scale = _primitive_mask_and_scale(mask)
        if scale != 1:
            continue
        if not np.array_equal(reduced, mask):
            continue
        return mask
    raise RuntimeError("failed to sample a primitive shape")


def _sample_scales(rng: random.Random, base_shape: tuple[int, int]) -> list[int] | None:
    bh, bw = base_shape
    max_scale = min(_CANVAS_SIDE // bh, _CANVAS_SIDE // bw, 6)
    if max_scale < 2:
        return None
    max_count = min(5, max_scale)
    if max_count < 2:
        return None

    for _ in range(80):
        count = rng.randint(2, max_count)
        include_one = rng.random() < 0.5 or max_scale == count
        pool = list(range(1, max_scale + 1)) if include_one else list(range(2, max_scale + 1))
        if len(pool) < count:
            continue
        scales = sorted(rng.sample(pool, count))
        padded_area = sum((bh * s + 2) * (bw * s + 2) for s in scales)
        if padded_area <= int(_CANVAS_SIDE * _CANVAS_SIDE * 0.72):
            return scales
    return None


def _place_boxes(shapes: list[np.ndarray], colors: list[int], rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((_CANVAS_SIDE, _CANVAS_SIDE), dtype=int)
    box_occ = np.zeros((_CANVAS_SIDE, _CANVAS_SIDE), dtype=bool)
    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx].shape[0] * shapes[idx].shape[1], reverse=True)

    for idx in order:
        mask = shapes[idx]
        color = colors[idx]
        h, w = mask.shape
        placed = False
        for _ in range(250):
            r = rng.randint(0, _CANVAS_SIDE - h)
            c = rng.randint(0, _CANVAS_SIDE - w)
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(_CANVAS_SIDE, r + h + 1)
            c1 = min(_CANVAS_SIDE, c + w + 1)
            if box_occ[r0:r1, c0:c1].any():
                continue
            patch = grid[r : r + h, c : c + w]
            patch[mask] = color
            box_occ[r:r + h, c:c + w] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _random_crop(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    rows = np.where(np.any(grid != 0, axis=1))[0]
    cols = np.where(np.any(grid != 0, axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])

    top = rng.randint(0, min(4, r0))
    left = rng.randint(0, min(4, c0))
    bottom = rng.randint(0, min(4, grid.shape[0] - 1 - r1))
    right = rng.randint(0, min(4, grid.shape[1] - 1 - c1))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    primitive = _random_primitive_mask(rng)
    scales = _sample_scales(rng, primitive.shape)
    if scales is None:
        return None

    shapes = [_upsample_mask(primitive, scale) for scale in scales]
    colors = rng.sample(_COLOR_POOL, len(shapes))
    grid = _place_boxes(shapes, colors, rng)
    if grid is None:
        return None

    cropped = _random_crop(grid, rng)
    if cropped.ndim != 2 or max(cropped.shape) > 30:
        return None

    components = _find_components(cropped, bg=0)
    if len(components) != len(shapes):
        return None
    if sorted(comp["area"] for comp in components) != sorted(int(shape.sum()) for shape in shapes):
        return None

    try:
        out = main(cropped)
    except AssertionError:
        return None

    largest_shape = shapes[-1]
    if out.shape != largest_shape.shape:
        return None
    if max(out.shape) > 30:
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
