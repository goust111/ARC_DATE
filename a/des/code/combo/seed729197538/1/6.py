from __future__ import annotations

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
_RNG = np.random.default_rng()

# Solver bounds.
MIN_OBJECTS = 2

# Generator bounds.
MAX_SHAPE_SPAN = 15
MAX_GRID_SIZE = 30
MIN_GRID_SIZE = 14
MAX_OBJECTS = 4
PLACEMENT_CLEARANCE = 1
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 60


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_objects(grid: np.ndarray) -> list[dict]:
    items: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        mask = grid == color
        r0, c0, r1, c1 = _bbox(mask)
        sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        sprite[mask[r0 : r1 + 1, c0 : c1 + 1]] = color
        items.append(
            {
                "color": color,
                "area": int(mask.sum()),
                "sprite": sprite,
            }
        )
    assert len(items) >= MIN_OBJECTS, "expected multiple foreground objects"
    areas = [int(item["area"]) for item in items]
    assert len(areas) == len(set(areas)), "foreground areas must define a unique nesting order"
    return items


def _paste_transparent(dst: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    patch = dst[top : top + h, left : left + w]
    mask = sprite != BG
    patch[mask] = sprite[mask]


def _compose(items: list[dict]) -> np.ndarray:
    ordered = sorted(items, key=lambda item: int(item["area"]), reverse=True)
    out = ordered[0]["sprite"].copy()
    out[out != BG] = int(ordered[0]["color"])
    H, W = out.shape

    for item in ordered[1:]:
        sprite = item["sprite"]
        h, w = sprite.shape
        dr = H - h
        dc = W - w
        assert dr >= 0 and dc >= 0, "smaller objects must fit inside the largest bounding box"
        assert dr % 2 == 0 and dc % 2 == 0, "nested objects must share the same bbox center"
        _paste_transparent(out, sprite, dr // 2, dc // 2)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Center-nest monochrome scaled objects from largest to smallest.

    The input contains several single-color objects on a black background. Their
    shapes are scaled versions of one another, so the only structure needed for
    the transformation is: extract each object's bbox sprite, sort by foreground
    pixel count, and paste every smaller sprite into the center of the largest
    one with black treated as transparent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    return _compose(_extract_objects(g))


def _shape_plus() -> np.ndarray:
    h = int(_RNG.choice([3, 5]))
    w = int(_RNG.choice([3, 5]))
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2, :] = True
    mask[:, w // 2] = True
    return mask


def _shape_ring() -> np.ndarray:
    h = int(_RNG.choice([3, 4, 5]))
    w = int(_RNG.choice([3, 4, 5]))
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _shape_l() -> np.ndarray:
    h = int(_RNG.integers(2, 6))
    w = int(_RNG.integers(2, 6))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _shape_t() -> np.ndarray:
    h = int(_RNG.integers(3, 6))
    w = int(_RNG.choice([3, 5]))
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, w // 2] = True
    return mask


def _shape_u() -> np.ndarray:
    h = int(_RNG.integers(3, 6))
    w = int(_RNG.integers(3, 6))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mask[-1, :] = True
    return mask


def _shape_stair() -> np.ndarray:
    n = int(_RNG.integers(3, 6))
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        mask[i, : i + 1] = True
    return mask


def _random_base_mask() -> np.ndarray:
    families = (
        _shape_plus,
        _shape_ring,
        _shape_l,
        _shape_t,
        _shape_u,
        _shape_stair,
    )
    mask = families[int(_RNG.integers(len(families)))]()
    if int(_RNG.integers(2)):
        mask = np.fliplr(mask)
    mask = np.rot90(mask, int(_RNG.integers(4)))
    mask = _crop_mask(mask)
    assert mask.shape[0] >= 2 and mask.shape[1] >= 2
    assert int(mask.sum()) < int(mask.size), "base pattern must not be a solid rectangle"
    return mask


def _scale_pool(base_h: int, base_w: int) -> list[int]:
    max_scale = min(MAX_SHAPE_SPAN // int(base_h), MAX_SHAPE_SPAN // int(base_w))
    if max_scale < 2:
        return []

    all_scales = list(range(1, max_scale + 1))
    if base_h % 2 == 0 and base_w % 2 == 0:
        return all_scales

    parity_choices = []
    for parity in (0, 1):
        pool = [scale for scale in all_scales if scale % 2 == parity]
        if len(pool) >= MIN_OBJECTS:
            parity_choices.append(pool)
    if not parity_choices:
        return []
    return list(parity_choices[int(_RNG.integers(len(parity_choices)))])


def _build_objects() -> list[dict] | None:
    base = _random_base_mask()
    base_h, base_w = map(int, base.shape)
    pool = _scale_pool(base_h, base_w)
    if len(pool) < MIN_OBJECTS:
        return None

    max_objects = min(MAX_OBJECTS, len(pool))
    n_objects = int(_RNG.integers(MIN_OBJECTS, max_objects + 1))
    scales = sorted(int(x) for x in _RNG.choice(pool, size=n_objects, replace=False))
    colors = [int(x) for x in _RNG.choice(COLORS, size=n_objects, replace=False)]

    items: list[dict] = []
    for scale, color in zip(scales, colors):
        sprite = np.full((base_h * scale, base_w * scale), BG, dtype=int)
        mask = np.kron(base.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)
        sprite[mask] = color
        items.append(
            {
                "color": color,
                "scale": scale,
                "area": int(mask.sum()),
                "sprite": sprite,
            }
        )

    areas = [int(item["area"]) for item in items]
    if len(areas) != len(set(areas)):
        return None
    return items


def _choose_canvas(items: list[dict]) -> tuple[int, int] | None:
    heights = [int(item["sprite"].shape[0]) for item in items]
    widths = [int(item["sprite"].shape[1]) for item in items]
    total_bbox_area = sum(h * w for h, w in zip(heights, widths))

    h_min = max(MIN_GRID_SIZE, max(heights) + 2)
    w_min = max(MIN_GRID_SIZE, max(widths) + 2)

    candidates = [
        (h, w)
        for h in range(h_min, MAX_GRID_SIZE + 1)
        for w in range(w_min, MAX_GRID_SIZE + 1)
        if h * w >= total_bbox_area + max(24, total_bbox_area // 2)
    ]
    if not candidates:
        return None
    return tuple(map(int, candidates[int(_RNG.integers(len(candidates)))]))


def _find_position(occupied: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    candidates = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            r0 = max(0, r - PLACEMENT_CLEARANCE)
            c0 = max(0, c - PLACEMENT_CLEARANCE)
            r1 = min(H, r + h + PLACEMENT_CLEARANCE)
            c1 = min(W, c + w + PLACEMENT_CLEARANCE)
            if not occupied[r0:r1, c0:c1].any():
                candidates.append((r, c))
    if not candidates:
        return None
    return tuple(map(int, candidates[int(_RNG.integers(len(candidates)))]))


def _place_items(items: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    H, W = map(int, shape)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for item in sorted(items, key=lambda item: int(item["sprite"].shape[0] * item["sprite"].shape[1]), reverse=True):
        sprite = item["sprite"]
        h, w = sprite.shape
        pos = None
        for _ in range(PLACEMENT_RETRIES):
            pos = _find_position(occupied, int(h), int(w))
            if pos is not None:
                break
        if pos is None:
            return None
        r, c = pos
        _paste_transparent(grid, sprite, r, c)
        occupied[r : r + h, c : c + w] = True
    return grid


def _generate_one() -> np.ndarray | None:
    items = _build_objects()
    if items is None:
        return None

    canvas = _choose_canvas(items)
    if canvas is None:
        return None

    inp = _place_items(items, canvas)
    if inp is None:
        return None

    out = _compose(items)
    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-scale input")
