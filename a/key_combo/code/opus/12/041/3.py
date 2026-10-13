from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
ORIGINAL_MARKER = 1
SPRITE_COLOR_POOL = (2, 4, 6, 7, 8, 9)
MIRROR_MARKER = 3
DIVIDER = 5

MAX_GENERATION_ATTEMPTS = 400


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


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _is_distinguishable_sprite(mask: np.ndarray) -> bool:
    return not (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _random_connected_mask(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    while int(mask.sum()) < target_area:
        frontier = []
        for r in range(height):
            for c in range(width):
                if mask[r, c]:
                    continue
                if (
                    (r > 0 and mask[r - 1, c])
                    or (r + 1 < height and mask[r + 1, c])
                    or (c > 0 and mask[r, c - 1])
                    or (c + 1 < width and mask[r, c + 1])
                ):
                    frontier.append((r, c))
        if not frontier:
            break
        mask[random.choice(frontier)] = True
    return mask


def _generate_sprite() -> np.ndarray:
    for _ in range(120):
        height = random.randint(3, 5)
        width = random.randint(3, 5)
        min_area = max(height + width - 1, (height * width + 2) // 3)
        max_area = (height * width * 3) // 4
        area = random.randint(min_area, max_area)
        mask = _random_connected_mask(height, width, area)
        if not _mask_connected(mask):
            continue
        if not _touches_all_sides(mask):
            continue
        if not _is_distinguishable_sprite(mask):
            continue
        color = random.choice(SPRITE_COLOR_POOL)
        sprite = np.zeros((height, width), dtype=int)
        sprite[mask] = color
        return sprite
    raise RuntimeError("failed to generate asymmetric sprite")


def _find_divider_column(grid: np.ndarray) -> int:
    divider_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(divider_cols) == 1, "expected exactly one full-height divider column"
    return int(divider_cols[0])


def _extract_sprite(grid: np.ndarray) -> np.ndarray:
    divider_col = _find_divider_column(grid)
    left = grid[:, :divider_col]
    sprite_bbox = _bbox(left != BACKGROUND)
    assert sprite_bbox is not None, "left half must contain a sprite"
    sprite = _crop_bbox(left, sprite_bbox)
    assert sprite.ndim == 2 and sprite.size > 0
    return sprite


def _decode_template(grid: np.ndarray) -> np.ndarray:
    divider_col = _find_divider_column(grid)
    right = grid[:, divider_col + 1 :]
    assert right.size > 0, "right half must exist"
    height, width = right.shape
    assert height % 3 == 0 and width % 3 == 0, "right half must split into 3x3 equal blocks"
    block_h = height // 3
    block_w = width // 3

    template = np.zeros((3, 3), dtype=int)
    for br in range(3):
        for bc in range(3):
            block = right[br * block_h : (br + 1) * block_h, bc * block_w : (bc + 1) * block_w]
            has_blue = bool(np.any(block == ORIGINAL_MARKER))
            has_green = bool(np.any(block == MIRROR_MARKER))
            assert not (has_blue and has_green), "a block cannot contain both marker colors"
            if has_blue:
                template[br, bc] = ORIGINAL_MARKER
            elif has_green:
                template[br, bc] = MIRROR_MARKER
    return template


def _render_output(sprite: np.ndarray, template: np.ndarray) -> np.ndarray:
    height, width = sprite.shape
    mirrored = np.fliplr(sprite)
    out = np.zeros((3 * height, 3 * width), dtype=int)
    for br in range(3):
        for bc in range(3):
            patch = None
            if template[br, bc] == ORIGINAL_MARKER:
                patch = sprite
            elif template[br, bc] == MIRROR_MARKER:
                patch = mirrored
            if patch is not None:
                r0 = br * height
                c0 = bc * width
                out[r0 : r0 + height, c0 : c0 + width] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the right-side marker blocks into a 3x3 instruction map and tile the left sprite with optional mirroring.

    Input structure:
    - The grid has a black background and one full-height grey vertical divider.
    - The left half contains one small monochrome sprite on black.
    - The right half is partitionable into a 3x3 array of equal blocks; each block is
      either empty, contains only blue marker pixels, or contains only green marker pixels.

    Rule:
    - Crop the left-half sprite to its bounding box.
    - Compress the right half blockwise into a 3x3 template:
      blue means place the sprite unchanged, green means place its left-right mirror,
      and empty means place a black block.
    - Assemble a 3x3 output whose cells are sprite-sized tiles chosen by that template.

    Key insight:
    - The right half is not an image to scale down; it is a coarse instruction grid
      whose marker color selects whether each output tile uses the original sprite or its mirror.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    sprite = _extract_sprite(g)
    template = _decode_template(g)
    return _render_output(sprite, template)


def _sample_template() -> np.ndarray:
    cells = list(range(9))
    blue_count = random.randint(2, 4)
    green_count = random.randint(2, 4)
    if blue_count + green_count > 8:
        green_count = 8 - blue_count
    assert blue_count >= 2 and green_count >= 2
    random.shuffle(cells)
    blue_cells = cells[:blue_count]
    green_cells = cells[blue_count : blue_count + green_count]

    template = np.zeros((3, 3), dtype=int)
    for idx in blue_cells:
        template[idx // 3, idx % 3] = ORIGINAL_MARKER
    for idx in green_cells:
        template[idx // 3, idx % 3] = MIRROR_MARKER
    assert np.count_nonzero(template == 0) >= 1
    return template


def _scatter_marker_block(block_h: int, block_w: int, color: int) -> np.ndarray:
    block = np.zeros((block_h, block_w), dtype=int)
    cell_count = block_h * block_w
    mark_count = random.randint(1, min(4, cell_count))
    positions = random.sample(range(cell_count), k=mark_count)
    for pos in positions:
        r = pos // block_w
        c = pos % block_w
        block[r, c] = color
    return block


def _render_input(sprite: np.ndarray, template: np.ndarray) -> np.ndarray:
    sprite_h, sprite_w = sprite.shape
    grid_h = random.choice([9, 12, 15])
    block_h = grid_h // 3

    block_w = random.choice([2, 3, 4])
    right_w = 3 * block_w
    left_w = random.randint(sprite_w + 2, min(10, sprite_w + 5))
    total_w = left_w + 1 + right_w
    assert total_w <= 30

    grid = np.zeros((grid_h, total_w), dtype=int)
    grid[:, left_w] = DIVIDER

    top = random.randint(0, grid_h - sprite_h)
    left = random.randint(0, left_w - sprite_w)
    grid[top : top + sprite_h, left : left + sprite_w] = sprite

    for br in range(3):
        for bc in range(3):
            marker = int(template[br, bc])
            if marker == 0:
                continue
            block = _scatter_marker_block(block_h, block_w, marker)
            r0 = br * block_h
            c0 = left_w + 1 + bc * block_w
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = block
    return grid


def _validate_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    divider_col = _find_divider_column(grid)
    assert not np.any(grid[:, :divider_col] == DIVIDER)
    assert not np.any(grid[:, divider_col + 1 :] == DIVIDER)

    sprite = _extract_sprite(grid)
    sprite_mask = sprite != BACKGROUND
    assert _mask_connected(sprite_mask)
    assert _touches_all_sides(sprite_mask)
    assert _is_distinguishable_sprite(sprite_mask)
    sprite_colors = {int(v) for v in np.unique(sprite) if int(v) != BACKGROUND}
    assert len(sprite_colors) == 1

    template = _decode_template(grid)
    blue_blocks = int(np.sum(template == ORIGINAL_MARKER))
    green_blocks = int(np.sum(template == MIRROR_MARKER))
    blank_blocks = int(np.sum(template == BACKGROUND))
    assert blue_blocks >= 2
    assert green_blocks >= 2
    assert blank_blocks >= 1

    divider_col = _find_divider_column(grid)
    right = grid[:, divider_col + 1 :]
    block_h = right.shape[0] // 3
    block_w = right.shape[1] // 3
    for br in range(3):
        for bc in range(3):
            block = right[br * block_h : (br + 1) * block_h, bc * block_w : (bc + 1) * block_w]
            colors = {int(v) for v in np.unique(block)} - {BACKGROUND}
            assert colors in (set(), {ORIGINAL_MARKER}, {MIRROR_MARKER})


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sprite = _generate_sprite()
        template = _sample_template()
        grid = _render_input(sprite, template)
        try:
            _validate_input(grid)
            expected = _render_output(sprite, template)
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
