from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))

PAIR_COUNT_RANGE = (2, 4)
HEIGHT_CHOICES = tuple(range(3, 8))
WIDTH_CHOICES = tuple(range(3, 9))
GRID_MIN_SIZE = 16
GRID_MAX_SIZE = 30
PLACEMENT_PADDING = 1
PLACEMENT_TRIES = 1000
MASK_TRIES = 200
GENERATION_TRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)

    return components


def _normalized_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _bbox_shape(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _is_connected(mask: np.ndarray) -> bool:
    return len(_component_coords(mask)) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each same-color fragment pair into a solid rectangle, then stack the
    reconstructed rectangles at the top-left in ascending-area overlay order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rectangles: list[tuple[int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND):
        components = _component_coords(g == color)
        assert len(components) == 2, "each non-background color must form exactly two objects"

        mask_a = _normalized_mask(components[0])
        mask_b = _normalized_mask(components[1])
        assert mask_a.shape == mask_b.shape, "same-color fragments must have matching bbox sizes"

        merged = mask_a | mask_b
        assert bool(merged.all()), "same-color fragments must OR into a full rectangle"

        h, w = merged.shape
        rectangles.append((h * w, h, w, color))

    assert rectangles
    areas = [area for area, _h, _w, _color in rectangles]
    assert len(set(areas)) == len(areas), "reconstructed rectangle areas must be distinct"

    out_h = max(h for _area, h, _w, _color in rectangles)
    out_w = max(w for _area, _h, w, _color in rectangles)
    out = np.full((out_h, out_w), BACKGROUND, dtype=int)

    for _area, h, w, color in sorted(rectangles, reverse=True):
        out[:h, :w] = color

    return out


def _sample_dimensions(n_pairs: int) -> list[tuple[int, int]] | None:
    for _ in range(MASK_TRIES):
        heights = sorted(random.sample(HEIGHT_CHOICES, n_pairs))
        widths = sorted(random.sample(WIDTH_CHOICES, n_pairs))
        dims = list(zip(heights, widths))
        areas = [h * w for h, w in dims]
        if len(set(areas)) == n_pairs:
            return dims
    return None


def _boundary_masks_by_rows(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    boundary = [random.randrange(w) for _ in range(h)]
    top_or_bottom_rows = random.sample(range(h), 2)
    boundary[top_or_bottom_rows[0]] = 0
    boundary[top_or_bottom_rows[1]] = w - 1

    left_mask = np.zeros((h, w), dtype=bool)
    right_mask = np.zeros((h, w), dtype=bool)
    for r, c in enumerate(boundary):
        left_mask[r, : c + 1] = True
        right_mask[r, c:] = True
    return left_mask, right_mask


def _boundary_masks_by_cols(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    boundary = [random.randrange(h) for _ in range(w)]
    left_or_right_cols = random.sample(range(w), 2)
    boundary[left_or_right_cols[0]] = 0
    boundary[left_or_right_cols[1]] = h - 1

    top_mask = np.zeros((h, w), dtype=bool)
    bottom_mask = np.zeros((h, w), dtype=bool)
    for c, r in enumerate(boundary):
        top_mask[: r + 1, c] = True
        bottom_mask[r:, c] = True
    return top_mask, bottom_mask


def _make_fragment_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(MASK_TRIES):
        if random.random() < 0.5:
            mask_a, mask_b = _boundary_masks_by_rows(h, w)
        else:
            mask_a, mask_b = _boundary_masks_by_cols(h, w)

        if _bbox_shape(mask_a) != (h, w) or _bbox_shape(mask_b) != (h, w):
            continue
        if not _is_connected(mask_a) or not _is_connected(mask_b):
            continue
        if not bool((mask_a | mask_b).all()):
            continue
        return mask_a, mask_b
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    grid_h, grid_w = occupied.shape
    rr0 = max(0, r0 - PLACEMENT_PADDING)
    rr1 = min(grid_h, r0 + h + PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    cc1 = min(grid_w, c0 + w + PLACEMENT_PADDING)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_patch(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int) -> bool:
    grid_h, grid_w = grid.shape
    h, w = mask.shape
    for _ in range(PLACEMENT_TRIES):
        r0 = random.randrange(grid_h - h + 1)
        c0 = random.randrange(grid_w - w + 1)
        if not _can_place(occupied, r0, c0, h, w):
            continue

        grid[r0 : r0 + h, c0 : c0 + w][mask] = color
        rr0 = max(0, r0 - PLACEMENT_PADDING)
        rr1 = min(grid_h, r0 + h + PLACEMENT_PADDING)
        cc0 = max(0, c0 - PLACEMENT_PADDING)
        cc1 = min(grid_w, c0 + w + PLACEMENT_PADDING)
        occupied[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _render_expected(rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    out_h = max(h for _area, h, _w, _color in rectangles)
    out_w = max(w for _area, _h, w, _color in rectangles)
    out = np.full((out_h, out_w), BACKGROUND, dtype=int)
    for _area, h, w, color in sorted(rectangles, reverse=True):
        out[:h, :w] = color
    return out


def _generate_once() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    dims = _sample_dimensions(n_pairs)
    if dims is None:
        return None

    total_bbox_area = 2 * sum(h * w for h, w in dims)
    min_side = max(
        GRID_MIN_SIZE,
        max(h for h, _w in dims) + 4,
        max(w for _h, w in dims) + 4,
        int(np.ceil(np.sqrt(total_bbox_area * 2.5))),
    )
    if min_side > GRID_MAX_SIZE:
        return None

    grid_h = random.randint(min_side, GRID_MAX_SIZE)
    grid_w = random.randint(min_side, GRID_MAX_SIZE)
    colors = random.sample(PALETTE, n_pairs)

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    rectangles: list[tuple[int, int, int, int]] = []
    patches: list[tuple[int, np.ndarray, int, int]] = []
    for (h, w), color in zip(dims, colors):
        pair = _make_fragment_pair(h, w)
        if pair is None:
            return None
        mask_a, mask_b = pair
        rectangles.append((h * w, h, w, color))
        patches.append((color, mask_a, h, w))
        patches.append((color, mask_b, h, w))

    random.shuffle(patches)
    patches.sort(key=lambda item: item[2] * item[3], reverse=True)
    for color, mask, _h, _w in patches:
        if not _place_patch(grid, occupied, mask, color):
            return None

    expected = _render_expected(rectangles)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
