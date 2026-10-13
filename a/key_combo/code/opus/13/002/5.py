from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BACKGROUND = 0
NESTED_LAYER_COLORS = (1, 2, 3, 4, 5)
X_COLORS = tuple(range(1, 10))
GENERATION_RETRIES = 400

X_MASK = np.zeros((5, 5), dtype=bool)
for _i in range(5):
    X_MASK[_i, _i] = True
    X_MASK[_i, 4 - _i] = True

FOLD_MAP = {
    (1, 1): (0, 0),
    (0, 0): (0, 1),
    (1, 3): (0, 2),
    (0, 4): (1, 2),
    (3, 3): (2, 2),
    (4, 4): (2, 1),
    (3, 1): (2, 0),
    (4, 0): (1, 0),
    (2, 2): (1, 1),
}
READ_ORDER = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
    (1, 1),
]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_x_crop(crop: np.ndarray) -> bool:
    return crop.shape == (5, 5) and np.all(crop[~X_MASK] == BACKGROUND) and np.all(crop[X_MASK] != BACKGROUND)


def _fold_sequence(x_crop: np.ndarray) -> list[int]:
    assert _is_x_crop(x_crop), "expected a 5x5 X crop"
    folded = np.zeros((3, 3), dtype=int)
    for src, dst in FOLD_MAP.items():
        folded[dst] = int(x_crop[src])
    return [int(folded[pos]) for pos in READ_ORDER]


def _boundary_color(rect: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    pieces = [
        rect[top, left:right],
        rect[bottom - 1, left:right],
        rect[top + 1 : bottom - 1, left],
        rect[top + 1 : bottom - 1, right - 1],
    ]
    boundary = np.concatenate([piece.reshape(-1) for piece in pieces if piece.size])
    colors = np.unique(boundary)
    assert colors.size == 1, "each nested boundary must be monochrome"
    return int(colors[0])


def _extract_layer_masks(rect: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    assert rect.ndim == 2 and rect.size > 0
    assert np.all(rect != BACKGROUND), "nested rectangle must fill its bbox"

    top, left = 0, 0
    bottom, right = rect.shape
    masks: list[np.ndarray] = []
    colors: list[int] = []

    while top < bottom and left < right:
        color = _boundary_color(rect, top, bottom, left, right)
        prev = (top, bottom, left, right)

        while top < bottom and np.all(rect[top, left:right] == color):
            top += 1
        while top < bottom and np.all(rect[bottom - 1, left:right] == color):
            bottom -= 1
        while left < right and top < bottom and np.all(rect[top:bottom, left] == color):
            left += 1
        while left < right and top < bottom and np.all(rect[top:bottom, right - 1] == color):
            right -= 1

        assert (top, bottom, left, right) != prev, "peeling must shrink the rectangle"

        outer_mask = np.zeros_like(rect, dtype=bool)
        outer_mask[prev[0] : prev[1], prev[2] : prev[3]] = True
        if top < bottom and left < right:
            outer_mask[top:bottom, left:right] = False
        masks.append(outer_mask)
        colors.append(color)

    assert len(masks) >= 2, "expected multiple nested layers"
    assert len(colors) == len(set(colors)), "nested layer colors must be distinct"
    return masks, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangle layers using the clockwise color code folded out of a 5x5 X.

    Input structure
    - The background is black.
    - One side of the grid contains a filled square made of outward-to-inward nested rectangular layers.
    - Another separate component is exactly one 5x5 X whose nine diagonal pixels are colored and all other cells are black.

    Rule
    - Split the non-black cells into the nested-rectangle component and the 5x5 X component.
    - Fold the X into a 3x3 grid with the fixed diagonal-to-edge mapping, then read the 3x3 from the upper-left clockwise around the perimeter and finally the center.
    - Recolor the nested rectangle from outermost layer inward: layers 1-8 take the first eight read colors, and layer 9 onward uses the center color.
    - Return only the recolored nested rectangle crop.

    Key insight
    - The X is not itself the output object; after folding it becomes an ordered color instruction list for the rectangle's layer depths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g != BACKGROUND, connectivity=8)
    assert len(comps) == 2, "expected exactly two non-black components"

    x_crop: np.ndarray | None = None
    rect_crop: np.ndarray | None = None
    for coords in comps:
        crop = _crop_bbox(g, _bbox_coords(coords))
        if _is_x_crop(crop):
            assert x_crop is None, "expected exactly one X component"
            x_crop = crop
        else:
            assert np.all(crop != BACKGROUND), "nested rectangle must fill its bbox"
            rect_crop = crop

    assert x_crop is not None and rect_crop is not None, "failed to identify the two required objects"

    sequence = _fold_sequence(x_crop)
    layer_masks, _layer_colors = _extract_layer_masks(rect_crop)

    out = rect_crop.copy()
    for idx, mask in enumerate(layer_masks):
        out[mask] = sequence[idx] if idx < 8 else sequence[8]
    return out


def _thickness_configs(side: int, layer_count: int) -> list[list[int]]:
    configs: list[list[int]] = []
    for thicks in itertools.product((1, 2), repeat=layer_count - 1):
        core = side - 2 * sum(thicks)
        if 1 <= core <= 3:
            configs.append(list(thicks))
    return configs


def _build_nested_square(side: int, layer_colors: list[int], thicks: list[int]) -> np.ndarray:
    rect = np.zeros((side, side), dtype=int)
    top = left = 0
    bottom = right = side
    for idx, color in enumerate(layer_colors):
        rect[top:bottom, left:right] = color
        if idx < len(thicks):
            step = thicks[idx]
            top += step
            left += step
            bottom -= step
            right -= step
    return rect


def _choose_prefix(layer_count: int, original_colors: list[int]) -> list[int]:
    for _ in range(200):
        if random.random() < 0.5:
            prefix = random.sample(X_COLORS, layer_count)
        else:
            pool_size = random.randint(2, min(4, layer_count))
            pool = random.sample(X_COLORS, pool_size)
            prefix = [random.choice(pool) for _ in range(layer_count)]
            if len(set(prefix)) < 2:
                continue
        if prefix != original_colors:
            return prefix
    raise RuntimeError("failed to choose a non-degenerate recolor prefix")


def _build_x_from_sequence(sequence: list[int]) -> np.ndarray:
    folded = np.zeros((3, 3), dtype=int)
    for pos, color in zip(READ_ORDER, sequence):
        folded[pos] = color

    x = np.zeros((5, 5), dtype=int)
    for src, dst in FOLD_MAP.items():
        x[src] = int(folded[dst])
    return x


def _gen_attempt() -> np.ndarray:
    layer_count = random.randint(4, 5)
    side = random.randint(9, 11)
    configs = _thickness_configs(side, layer_count)
    assert configs, "chosen side must support the chosen layer count"
    thicks = random.choice(configs)

    layer_colors = random.sample(NESTED_LAYER_COLORS, layer_count)
    nested = _build_nested_square(side, layer_colors, thicks)

    prefix = _choose_prefix(layer_count, layer_colors)
    suffix_palette = random.sample(X_COLORS, random.randint(2, 5))
    sequence = prefix + [random.choice(suffix_palette) for _ in range(9 - layer_count)]
    x_obj = _build_x_from_sequence(sequence)

    left_margin = random.randint(0, 3)
    right_margin = random.randint(0, 3)
    gap = random.randint(2, 4)
    width = left_margin + side + gap + 5 + right_margin

    height = max(side, 5) + random.randint(0, 5)
    nested_r = random.randint(0, height - side)
    x_r = random.randint(0, height - 5)
    nested_c = left_margin
    x_c = left_margin + side + gap

    grid = np.zeros((height, width), dtype=int)
    grid[nested_r : nested_r + side, nested_c : nested_c + side] = nested
    grid[x_r : x_r + 5, x_c : x_c + 5] = x_obj
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        try:
            out = main(grid)
        except Exception:
            continue
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
