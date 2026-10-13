from __future__ import annotations

import random
from collections import deque

import numpy as np


RED = 2
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator configuration.
GEN_RETRIES = 400
OBJECT_RETRIES = 80
PLACEMENT_RETRIES = 160

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (16, 28)
OBJECT_COUNT_RANGE = (2, 4)
SHORT_AXIS_RANGE = (2, 5)
HALF_LONG_EXTRA_RANGE = (0, 3)
BG_CHOICES = (0, 1, 4, 6, 8, 9)
HALO = 1


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )

    return components


def _component_mask(comp: dict[str, object]) -> np.ndarray:
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(grid != bg)
    if pts.size == 0:
        return np.full((1, 1), bg, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _fold_component(comp: dict[str, object], bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    mask = _component_mask(comp)
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    assert h != w, "object bbox must have a unique stretch direction"

    if w > h:
        assert w % 2 == 0, "horizontal object width must be even"
        half = w // 2
        left = mask[:, :half]
        right = np.fliplr(mask[:, half:])
        union = left | right
        overlap = left & right
        patch = np.full((h, half), bg, dtype=int)
        patch[union] = color
        patch[overlap] = RED
        return patch, (int(r0), int(c0))

    assert h % 2 == 0, "vertical object height must be even"
    half = h // 2
    top = mask[:half, :]
    bottom = np.flipud(mask[half:, :])
    union = top | bottom
    overlap = top & bottom
    patch = np.full((half, w), bg, dtype=int)
    patch[union] = color
    patch[overlap] = RED
    return patch, (int(r0), int(c0))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold every non-square foreground object across the midpoint of its longer bbox axis.

    Horizontal objects are folded right-to-left; vertical objects are folded bottom-to-top.
    The folded result is anchored at the original bbox top-left corner, and the final output
    is the tight crop around all folded objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg)
    assert components, "expected at least one foreground object"

    out = np.full_like(g, bg)
    for comp in components:
        patch, (top, left) = _fold_component(comp, bg)
        ph, pw = patch.shape
        write_mask = patch != bg
        out[top : top + ph, left : left + pw][write_mask] = patch[write_mask]

    return _tight_crop(out, bg)


def _row_suffix_mask(height: int, half_width: int) -> np.ndarray:
    starts = [random.randrange(half_width) for _ in range(height)]
    if min(starts) > 0:
        starts[random.randrange(height)] = 0
    if all(start == 0 for start in starts):
        starts[random.randrange(height)] = random.randrange(1, half_width)

    mask = np.zeros((height, half_width), dtype=bool)
    for r, start in enumerate(starts):
        mask[r, start:] = True
    return mask


def _col_suffix_mask(half_height: int, width: int) -> np.ndarray:
    starts = [random.randrange(half_height) for _ in range(width)]
    if min(starts) > 0:
        starts[random.randrange(width)] = 0
    if all(start == 0 for start in starts):
        starts[random.randrange(width)] = random.randrange(1, half_height)

    mask = np.zeros((half_height, width), dtype=bool)
    for c, start in enumerate(starts):
        mask[start:, c] = True
    return mask


def _make_horizontal_object() -> np.ndarray | None:
    for _ in range(OBJECT_RETRIES):
        height = random.randint(*SHORT_AXIS_RANGE)
        min_half = height // 2 + 1
        max_half = min_half + random.randint(*HALF_LONG_EXTRA_RANGE)
        half_width = random.randint(min_half, max_half)

        left = _row_suffix_mask(height, half_width)
        right = _row_suffix_mask(height, half_width)
        if np.array_equal(left, right):
            continue

        mask = np.concatenate([left, np.fliplr(right)], axis=1)
        if np.all(mask):
            continue
        return mask
    return None


def _make_vertical_object() -> np.ndarray | None:
    for _ in range(OBJECT_RETRIES):
        width = random.randint(*SHORT_AXIS_RANGE)
        min_half = width // 2 + 1
        max_half = min_half + random.randint(*HALF_LONG_EXTRA_RANGE)
        half_height = random.randint(min_half, max_half)

        top = _col_suffix_mask(half_height, width)
        bottom = _col_suffix_mask(half_height, width)
        if np.array_equal(top, bottom):
            continue

        mask = np.concatenate([top, np.flipud(bottom)], axis=0)
        if np.all(mask):
            continue
        return mask
    return None


def _orientation_list(count: int) -> list[str]:
    assert count >= 2
    orientations = ["h", "v"]
    for _ in range(count - 2):
        orientations.append(random.choice(("h", "v")))
    random.shuffle(orientations)
    return orientations


def _place_objects(grid: np.ndarray, objects: list[dict[str, object]]) -> bool:
    h, w = grid.shape
    reserved = np.zeros((h, w), dtype=bool)

    for obj in sorted(objects, key=lambda item: np.asarray(item["mask"]).size, reverse=True):
        mask = np.asarray(obj["mask"], dtype=bool)
        color = int(obj["color"])
        oh, ow = mask.shape
        placed = False

        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, h - oh)
            left = random.randint(0, w - ow)

            r0 = max(0, top - HALO)
            c0 = max(0, left - HALO)
            r1 = min(h, top + oh + HALO)
            c1 = min(w, left + ow + HALO)
            if reserved[r0:r1, c0:c1].any():
                continue

            region = grid[top : top + oh, left : left + ow]
            region[mask] = color
            reserved[r0:r1, c0:c1] = True
            placed = True
            break

        if not placed:
            return False

    return True


def _sample_object(orientation: str) -> np.ndarray | None:
    if orientation == "h":
        return _make_horizontal_object()
    return _make_vertical_object()


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    orientations = _orientation_list(count)

    masks: list[np.ndarray] = []
    for orientation in orientations:
        mask = _sample_object(orientation)
        if mask is None:
            return None
        masks.append(mask)

    min_h = max(GRID_H_RANGE[0], max(mask.shape[0] for mask in masks) + 2)
    min_w = max(GRID_W_RANGE[0], max(mask.shape[1] for mask in masks) + 2)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = random.randint(min_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_w, GRID_W_RANGE[1])
    bg = random.choice(BG_CHOICES)
    color_pool = [color for color in range(10) if color not in (bg, RED)]
    if len(color_pool) < count:
        return None
    colors = random.sample(color_pool, count)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    objects = [{"mask": mask, "color": color} for mask, color in zip(masks, colors, strict=False)]
    if not _place_objects(grid, objects):
        return None

    if int(np.count_nonzero(grid == bg)) <= grid.size // 2:
        return None

    components = find_components(grid, bg)
    if len(components) != count:
        return None

    out = main(grid)
    if not np.any(out == RED):
        return None
    if not np.any((out != bg) & (out != RED)):
        return None
    if out.shape == grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
