from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
CROSS_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_SIZES = (13, 15, 17, 19, 21, 23, 25, 27, 29)
GEN_ATTEMPTS = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = NEIGHBORS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
QUADRANT_ROTATIONS = {
    "tl": 1,  # clockwise quarter-turns
    "tr": 2,
    "bl": 3,
    "br": 0,
}


def _most_common(values: np.ndarray) -> int:
    palette, counts = np.unique(values, return_counts=True)
    return int(palette[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    neighbors = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _tight_crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _is_asymmetric(mask: np.ndarray) -> bool:
    variants = {_mask_signature(_tight_crop_bool(np.rot90(mask, k))) for k in range(4)}
    return len(variants) == 4


def _extract_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[int(r - r0), int(c - c0)] = int(grid[r, c])
    return patch


def _quadrant_of(coords: np.ndarray, mid: int) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 < mid and c1 < mid:
        return "tl"
    if r1 < mid and c0 > mid:
        return "tr"
    if r0 > mid and c1 < mid:
        return "bl"
    if r0 > mid and c0 > mid:
        return "br"
    raise AssertionError("foreground component must lie fully inside one quadrant")


def _target_top_left(quadrant: str, patch_shape: tuple[int, int], mid: int) -> tuple[int, int]:
    ph, pw = patch_shape
    if quadrant == "tl":
        return mid - ph, mid - pw
    if quadrant == "tr":
        return mid - ph, mid + 1
    if quadrant == "bl":
        return mid + 1, mid - pw
    if quadrant == "br":
        return mid + 1, mid + 1
    raise AssertionError(f"unknown quadrant: {quadrant}")


def _paste_patch(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    rows, cols = patch.shape
    for r in range(rows):
        for c in range(cols):
            value = int(patch[r, c])
            if value == int(bg):
                continue
            out[top + r, left + c] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each quadrant object by its quadrant-coded angle, then slide it against the center cross.

    Input structure:
    - The grid is an odd-sized square with a black background.
    - The middle row and middle column form a solid grey cross that splits the board into four quadrants.
    - Each quadrant contains exactly one small monochrome foreground object, and each object stays strictly inside its quadrant.

    Rule:
    - Rotate the top-left object 90 degrees clockwise, the top-right object 180 degrees clockwise, the bottom-left object 270 degrees clockwise, and the bottom-right object not at all.
    - After rotation, translate each object toward the center so its bounding box becomes flush with both center axes while staying off the grey cross.
    - Remove the original object placements and keep the center cross unchanged.

    Key insight:
    - The quadrant itself encodes the rotation amount; the final translation just packs the rotated objects into the four cells of space adjacent to the center cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 1
    n = int(g.shape[0])
    mid = n // 2

    cross_color = int(g[mid, mid])
    assert np.all(g[mid, :] == cross_color)
    assert np.all(g[:, mid] == cross_color)

    off_cross = np.ones_like(g, dtype=bool)
    off_cross[mid, :] = False
    off_cross[:, mid] = False
    bg = _most_common(g[off_cross])

    object_mask = (g != int(bg)) & (g != int(cross_color))
    parts = _components(object_mask, connectivity=8)
    assert len(parts) == 4

    out = np.full_like(g, int(bg))
    out[mid, :] = int(cross_color)
    out[:, mid] = int(cross_color)

    seen_quadrants: set[str] = set()
    for coords in parts:
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1
        quadrant = _quadrant_of(coords, mid)
        assert quadrant not in seen_quadrants
        seen_quadrants.add(quadrant)

        patch = _extract_patch(g, coords, bg)
        quarter_turns = QUADRANT_ROTATIONS[quadrant]
        rotated = np.rot90(patch, -quarter_turns)
        top, left = _target_top_left(quadrant, rotated.shape, mid)
        _paste_patch(out, rotated, top, left, bg)

    assert len(seen_quadrants) == 4
    return out


def _random_connected_shape(rng: random.Random, max_dim: int) -> np.ndarray:
    for _ in range(300):
        height = rng.randint(2, max_dim)
        width = rng.randint(2, max_dim)
        max_area = min(height * width - 1, 8)
        min_area = min(4, max_area)
        if max_area < 4:
            continue
        area = rng.randint(min_area, max_area)

        seed = (rng.randrange(height), rng.randrange(width))
        cells = {seed}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        if len(cells) != area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _tight_crop_bool(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if not _is_asymmetric(mask):
            continue
        return mask
    raise RuntimeError("failed to build a non-degenerate asymmetric shape")


def _place_shape_in_quadrant(
    grid: np.ndarray,
    shape: np.ndarray,
    color: int,
    quadrant: str,
    mid: int,
    rng: random.Random,
) -> None:
    shape = np.rot90(shape, rng.randrange(4))
    height, width = shape.shape
    local_limit_r = mid - height - 1
    local_limit_c = mid - width - 1
    assert local_limit_r >= 0 and local_limit_c >= 0
    offset_r = rng.randint(0, local_limit_r)
    offset_c = rng.randint(0, local_limit_c)

    if quadrant == "tl":
        top, left = offset_r, offset_c
    elif quadrant == "tr":
        top, left = offset_r, mid + 1 + offset_c
    elif quadrant == "bl":
        top, left = mid + 1 + offset_r, offset_c
    elif quadrant == "br":
        top, left = mid + 1 + offset_r, mid + 1 + offset_c
    else:
        raise AssertionError(f"unknown quadrant: {quadrant}")

    for r, c in np.argwhere(shape):
        assert int(grid[top + r, left + c]) == BG_COLOR
        grid[top + r, left + c] = int(color)


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] % 2 != 1 or int(grid.shape[0]) > 30:
        return False
    n = int(grid.shape[0])
    mid = n // 2
    if not np.all(grid[mid, :] == CROSS_COLOR) or not np.all(grid[:, mid] == CROSS_COLOR):
        return False

    object_mask = (grid != BG_COLOR) & (grid != CROSS_COLOR)
    parts = _components(object_mask, connectivity=8)
    if len(parts) != 4:
        return False

    quadrants = set()
    colors = set()
    for coords in parts:
        part_colors = {int(grid[r, c]) for r, c in coords}
        if len(part_colors) != 1:
            return False
        quadrants.add(_quadrant_of(coords, mid))
        colors |= part_colors
        patch = _extract_patch(grid, coords, BG_COLOR) != BG_COLOR
        if not _is_asymmetric(patch):
            return False
    return len(quadrants) == 4 and len(colors) == 4


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        size = rng.choice(GRID_SIZES)
        mid = size // 2
        max_dim = min(5, mid - 1)
        if max_dim < 2:
            continue

        grid = np.full((size, size), BG_COLOR, dtype=int)
        grid[mid, :] = CROSS_COLOR
        grid[:, mid] = CROSS_COLOR

        colors = rng.sample(OBJECT_COLORS, 4)
        quadrant_order = ("tl", "tr", "bl", "br")

        try:
            for quadrant, color in zip(quadrant_order, colors):
                shape = _random_connected_shape(rng, max_dim)
                _place_shape_in_quadrant(grid, shape, color, quadrant, mid, rng)
        except RuntimeError:
            continue

        if not _validate_generated_grid(grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid quadrant-rotation puzzle input")
