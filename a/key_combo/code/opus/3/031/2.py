from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BG = 0
SPRITE_W = 4
SPRITE_H_CHOICES = (3, 4)
OBJECT_COUNT_CHOICES = (2, 3)
GRID_W_RANGE = (16, 26)
UPPER_H_RANGE = (8, 14)
MARKER_GAP_RANGE = (2, 4)
BOTTOM_PAD_RANGE = (0, 3)
PLACEMENT_PADDING = 1
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200


def _is_lr_symmetric(tile: np.ndarray, bg: int) -> bool:
    mask = tile != int(bg)
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _strip_from_tile(tile: np.ndarray, bg: int) -> np.ndarray:
    symmetric = _is_lr_symmetric(tile, bg)
    mirrored = np.fliplr(tile)
    pieces = [
        tile if symmetric or idx % 2 == 0 else mirrored
        for idx in range(5)
    ]
    return np.concatenate(pieces, axis=1)


def _component_sets(grid: np.ndarray) -> tuple[int, dict[int, dict], list[tuple[int, int, int]]]:
    bg = int(get_bg(grid))
    components = find_components(
        np.asarray(grid, dtype=int),
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    sprites: dict[int, dict] = {}
    markers: list[tuple[int, int, int]] = []
    for comp in components:
        color = int(comp["color"])
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            markers.append((r, c, color))
        else:
            if color in sprites:
                raise AssertionError("expected one multi-cell sprite per color")
            sprites[color] = comp

    if len(sprites) < 2:
        raise AssertionError("expected at least two sprites")
    if len(markers) != len(sprites):
        raise AssertionError("expected one marker pixel per sprite")

    marker_rows = {r for r, _c, _color in markers}
    if len(marker_rows) != 1:
        raise AssertionError("marker pixels must lie on one row")
    ordered_markers = sorted(markers, key=lambda item: item[1])
    if {color for _r, _c, color in ordered_markers} != set(sprites):
        raise AssertionError("marker colors must match sprite colors exactly")
    return bg, sprites, ordered_markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect left-right symmetry on each sprite and use it to choose uniform or alternating mirrored tiling.

    Input structure:
    - A black background contains 2 or 3 separated monochrome sprites in the upper area.
    - Each sprite color is unique and appears again as one isolated marker pixel in a lower marker row.
    - Every sprite is treated through its tight bounding box, including internal black cells.

    Rule:
    - For each upper sprite, crop its bounding box and test whether its non-background shape is invariant under horizontal reflection.
    - If the tile is symmetric, concatenate 5 copies in the original orientation; otherwise concatenate 5 copies with original and mirrored orientations alternating.
    - Read the lower marker pixels from left to right and vertically stack the corresponding strips in that order, with no gaps.

    Key insight:
    - The marker row does not change colors or positions itself; it only specifies the stacking order, while symmetry determines whether a sprite uses plain tiling or alternating mirrored tiling.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprites, ordered_markers = _component_sets(g)

    strips: list[np.ndarray] = []
    for _r, _c, color in ordered_markers:
        tile = crop_bbox(g, sprites[color]["bbox"])
        strips.append(_strip_from_tile(tile, bg))

    widths = {strip.shape[1] for strip in strips}
    if len(widths) != 1:
        raise AssertionError("all strips must have the same width")
    return np.concatenate(strips, axis=0)


def _neighbors4(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for cell in cells:
            for nb in _neighbors4(cell, h, w):
                if nb not in cells:
                    frontier.append(nb)
        if not frontier:
            raise ValueError("could not grow connected mask")
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        cell = stack.pop()
        for nb in _neighbors4(cell, h, w):
            if mask[nb] and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == int(mask.sum())


def _has_full_bbox(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _valid_sprite_mask(mask: np.ndarray, symmetric: bool) -> bool:
    area = int(mask.sum())
    if not (4 <= area <= 8):
        return False
    if not _has_full_bbox(mask):
        return False
    if not _is_connected(mask):
        return False
    if _is_lr_symmetric(mask.astype(int), BG) != symmetric:
        return False
    return True


def _enumerate_masks(h: int, symmetric: bool) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for bits in range(1 << (h * SPRITE_W)):
        flat = [(bits >> idx) & 1 for idx in range(h * SPRITE_W)]
        mask = np.array(flat, dtype=bool).reshape(h, SPRITE_W)
        if _valid_sprite_mask(mask, symmetric=symmetric):
            candidates.append(mask)
    return candidates


_SYMMETRIC_MASKS = {
    h: masks
    for h in SPRITE_H_CHOICES
    for masks in [_enumerate_masks(h, symmetric=True)]
    if masks
}
_ASYMMETRIC_MASKS = {
    h: masks
    for h in SPRITE_H_CHOICES
    for masks in [_enumerate_masks(h, symmetric=False)]
    if masks
}


def _generate_symmetric_mask(h: int) -> np.ndarray:
    return random.choice(_SYMMETRIC_MASKS[h]).copy()


def _generate_asymmetric_mask(h: int) -> np.ndarray:
    return random.choice(_ASYMMETRIC_MASKS[h]).copy()


def _render_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(mask.shape, BG, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _place_sprites(
    grid_h: int,
    grid_w: int,
    marker_row: int,
    sprites: list[np.ndarray],
) -> list[tuple[int, int]]:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    locations: list[tuple[int, int]] = []

    for sprite in sprites:
        h, w = sprite.shape
        candidates: list[tuple[int, int]] = []
        max_r = marker_row - 2 - h
        if max_r < 0:
            raise ValueError("upper area too small")
        for r0 in range(max_r + 1):
            for c0 in range(grid_w - w + 1):
                r1 = r0 + h
                c1 = c0 + w
                rr0 = max(0, r0 - PLACEMENT_PADDING)
                cc0 = max(0, c0 - PLACEMENT_PADDING)
                rr1 = min(grid_h, r1 + PLACEMENT_PADDING)
                cc1 = min(grid_w, c1 + PLACEMENT_PADDING)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            raise ValueError("no free placement for sprite")
        loc = random.choice(candidates)
        locations.append(loc)
        r0, c0 = loc
        occupied[r0 : r0 + h, c0 : c0 + w] = True

    return locations


def _build_expected(color_to_sprite: dict[int, np.ndarray], marker_order: list[int]) -> np.ndarray:
    strips = [
        _strip_from_tile(color_to_sprite[color], BG)
        for color in marker_order
    ]
    return np.concatenate(strips, axis=0)


def _generate_case() -> np.ndarray | None:
    obj_count = random.choice(OBJECT_COUNT_CHOICES)
    sym_count = random.randint(1, obj_count - 1)
    symmetric_flags = [True] * sym_count + [False] * (obj_count - sym_count)
    random.shuffle(symmetric_flags)

    colors = random.sample(list(range(1, 10)), obj_count)
    sprites: list[np.ndarray] = []
    color_to_sprite: dict[int, np.ndarray] = {}

    for color, symmetric in zip(colors, symmetric_flags):
        if symmetric:
            h = random.choice(tuple(_SYMMETRIC_MASKS))
        else:
            h = random.choice(tuple(_ASYMMETRIC_MASKS))
        mask = _generate_symmetric_mask(h) if symmetric else _generate_asymmetric_mask(h)
        sprite = _render_sprite(mask, color)
        sprites.append(sprite)
        color_to_sprite[color] = sprite

    marker_span = obj_count * 2 - 1
    for _ in range(PLACEMENT_RETRIES):
        upper_h = random.randint(*UPPER_H_RANGE)
        marker_gap = random.randint(*MARKER_GAP_RANGE)
        bottom_pad = random.randint(*BOTTOM_PAD_RANGE)
        grid_h = upper_h + marker_gap + 1 + bottom_pad
        grid_w = random.randint(
            max(GRID_W_RANGE[0], SPRITE_W + 2, marker_span + 2),
            GRID_W_RANGE[1],
        )
        marker_row = upper_h + marker_gap
        if marker_row >= grid_h:
            continue
        if grid_w < marker_span:
            continue

        try:
            positions = _place_sprites(grid_h, grid_w, marker_row, sprites)
        except ValueError:
            continue

        grid = np.full((grid_h, grid_w), BG, dtype=int)
        for sprite, (r0, c0) in zip(sprites, positions):
            paste(grid, sprite, r0, c0, transparent=BG)

        order = colors[:]
        random.shuffle(order)
        start_c = random.randint(0, grid_w - marker_span)
        for idx, color in enumerate(order):
            grid[marker_row, start_c + 2 * idx] = color

        expected = _build_expected(color_to_sprite, order)
        if expected.shape[0] > 30 or expected.shape[1] > 30:
            continue

        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-tiling input")
