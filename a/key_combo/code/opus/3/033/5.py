import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _placeholder_components(grid: np.ndarray) -> list[np.ndarray]:
    return _components_mask(np.asarray(grid, dtype=int) == 5, connectivity=4)


def _normalized_shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox_coords(coords)
    rel = np.asarray(coords, dtype=int) - np.array([r0, c0], dtype=int)
    order = np.lexsort((rel[:, 1], rel[:, 0]))
    rel = rel[order]
    return tuple((int(r), int(c)) for r, c in rel)


def _first_pixel(coords: np.ndarray) -> tuple[int, int]:
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    r, c = coords[order[0]]
    return int(r), int(c)


def _recolor_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    components = _placeholder_components(g)
    family_first: dict[tuple[tuple[int, int], ...], tuple[int, int]] = {}
    component_info: list[tuple[np.ndarray, tuple[tuple[int, int], ...]]] = []

    for coords in components:
        key = _normalized_shape_key(coords)
        first = _first_pixel(coords)
        if key not in family_first or first < family_first[key]:
            family_first[key] = first
        component_info.append((coords, key))

    palette = [1, 2, 3]
    ordered_keys = sorted(family_first, key=lambda key: family_first[key])
    assert 1 <= len(ordered_keys) <= 3, "expected at most three normalized placeholder shapes"
    color_by_key = {key: palette[idx] for idx, key in enumerate(ordered_keys)}

    out = np.zeros_like(g)
    for coords, key in component_info:
        out[coords[:, 0], coords[:, 1]] = color_by_key[key]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor equal placeholder shapes, then place two copies on the main diagonal and expand blank rows into separators.

    Input structure:
    - A black background contains 4-connected grey placeholder objects made of 2 to 4 pixels.
    - There are 2 or 3 translation-normalized placeholder shapes, and equal shapes may appear multiple times.
    - One or two entire rows are completely black; they split the objects into vertical groups.

    Rule:
    - Find each grey connected component and normalize it by translating its bbox to the origin.
    - Sort the distinct normalized shapes by the first place they appear in the input, using row-major order, and recolor shape families to blue, red, then green.
    - Create a black canvas of size 2H x 2W, copy the recolored grid to the top-left and bottom-right quadrants, and keep the other two quadrants black.
    - For every all-black input row, replace the corresponding row in both diagonal copies with a full-width green separator row.

    Key insight:
    - The all-black rows are not just unused space: they survive the recoloring step and then become the structural separators in both diagonal copies.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    recolored = _recolor_grid(g)
    out = np.zeros((2 * h, 2 * w), dtype=int)
    out[:h, :w] = recolored
    out[h:, w:] = recolored

    blank_rows = np.flatnonzero(np.all(g == 0, axis=1))
    for r in blank_rows:
        out[int(r), :] = 3
        out[h + int(r), :] = 3
    return out


_SHAPE_LIBRARY = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]

_FAMILY_COUNT_OPTIONS = {
    2: [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3)],
    3: [(2, 2, 2)],
}

_MAX_GEN_ATTEMPTS = 400


def _bounded_partitions(total: int, parts: int, lo: int, hi: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)] if lo <= total <= hi else []

    out: list[tuple[int, ...]] = []
    remaining_lo = lo * (parts - 1)
    remaining_hi = hi * (parts - 1)
    for first in range(lo, hi + 1):
        rest = total - first
        if rest < remaining_lo or rest > remaining_hi:
            continue
        for tail in _bounded_partitions(rest, parts - 1, lo, hi):
            out.append((first,) + tail)
    return out


def _sample_band_heights(min_heights: list[int], blank_count: int) -> tuple[int, ...] | None:
    upper = 7 if len(min_heights) == 2 else 5
    choices: list[tuple[int, ...]] = []

    def rec(idx: int, cur: list[int]) -> None:
        if idx == len(min_heights):
            if sum(cur) + blank_count <= 15:
                choices.append(tuple(cur))
            return
        for height in range(min_heights[idx], upper + 1):
            cur.append(height)
            rec(idx + 1, cur)
            cur.pop()

    rec(0, [])
    return random.choice(choices) if choices else None


def _shape_key_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    return _normalized_shape_key(coords)


def _build_object_shapes() -> list[np.ndarray]:
    family_count = random.choices([2, 3], weights=[3, 2])[0]
    family_counts = list(random.choice(_FAMILY_COUNT_OPTIONS[family_count]))
    families = random.sample(_SHAPE_LIBRARY, family_count)

    objects: list[np.ndarray] = []
    for family, copies in zip(families, family_counts):
        for _ in range(copies):
            objects.append(family.copy())
    random.shuffle(objects)
    return objects


def _band_ranges(heights: tuple[int, ...]) -> tuple[list[tuple[int, int]], list[int], int]:
    bands: list[tuple[int, int]] = []
    blank_rows: list[int] = []
    cursor = 0
    for idx, height in enumerate(heights):
        bands.append((cursor, cursor + height - 1))
        cursor += height
        if idx < len(heights) - 1:
            blank_rows.append(cursor)
            cursor += 1
    return bands, blank_rows, cursor


def _place_band_objects(
    grid: np.ndarray,
    occupied: np.ndarray,
    band: tuple[int, int],
    objects: list[np.ndarray],
) -> bool:
    h_total, w_total = grid.shape
    r_lo, r_hi = band
    order = sorted(objects, key=lambda mask: (-mask.shape[1], -mask.shape[0], -int(mask.sum())))
    for mask in order:
        h, w = mask.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(r_lo, r_hi - h + 2):
            for c0 in range(0, w_total - w + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(h_total, r0 + h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(w_total, c0 + w + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[mask] = 5
        rr0 = max(0, r0 - 1)
        rr1 = min(h_total, r0 + h + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w_total, c0 + w + 1)
        occupied[rr0:rr1, cc0:cc1] = True
    return True


def _valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 15:
        return False
    if not set(map(int, np.unique(g))).issubset({0, 5}):
        return False

    blank_rows = list(np.flatnonzero(np.all(g == 0, axis=1)))
    if len(blank_rows) not in (1, 2):
        return False
    if blank_rows[0] == 0 or blank_rows[-1] == g.shape[0] - 1:
        return False

    components = _placeholder_components(g)
    if not (4 <= len(components) <= 6):
        return False

    family_counts = Counter(_normalized_shape_key(coords) for coords in components)
    if len(family_counts) not in (2, 3):
        return False
    if min(family_counts.values()) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    objects = _build_object_shapes()
    object_count = len(objects)

    blank_count = random.choices([1, 2], weights=[3, 2])[0]
    band_count = blank_count + 1
    band_object_options = _bounded_partitions(object_count, band_count, 1, 3)
    if not band_object_options:
        return None
    band_object_counts = random.choice(band_object_options)

    band_objects: list[list[np.ndarray]] = []
    cursor = 0
    for count in band_object_counts:
        band_objects.append(objects[cursor : cursor + count])
        cursor += count

    min_heights = [max(3, max(mask.shape[0] for mask in masks)) for masks in band_objects]
    heights = _sample_band_heights(min_heights, blank_count)
    if heights is None:
        return None
    bands, blank_rows, height = _band_ranges(heights)

    min_width = 8
    for masks in band_objects:
        row_fit = sum(mask.shape[1] for mask in masks) + max(0, len(masks) - 1)
        min_width = max(min_width, row_fit)
    if min_width > 15:
        return None
    width = random.randint(min_width, 15)

    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for band, masks in zip(bands, band_objects):
        if not _place_band_objects(grid, occupied, band, masks):
            return None

    if not blank_rows or not all(np.all(grid[r] == 0) for r in blank_rows):
        return None
    if not _valid_input(grid):
        return None

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
