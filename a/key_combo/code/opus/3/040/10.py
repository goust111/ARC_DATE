from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_shared_arc_utils_for_040_10",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg
bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox
paste = _ARC_UTILS.paste

BLOCK_SIZE = 5
BLOCKS_PER_SIDE = 3
GRID_SIZE = BLOCK_SIZE * BLOCKS_PER_SIDE
GEN_ATTEMPTS = 400
ACTIVE_COUNT_OPTIONS = (3, 4, 5, 6, 7, 8)
SPRITE_HEIGHTS = (2, 3)
SPRITE_WIDTHS = (2, 3, 4)
PALETTE = tuple(color for color in range(10) if color != Color.TEAL)


def _block_origin(index: int) -> tuple[int, int]:
    return (index // BLOCKS_PER_SIDE) * BLOCK_SIZE, (index % BLOCKS_PER_SIDE) * BLOCK_SIZE


def _block_view(grid: np.ndarray, index: int) -> np.ndarray:
    r0, c0 = _block_origin(index)
    return grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]


def _centered_top_left(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    return (BLOCK_SIZE - h) // 2, (BLOCK_SIZE - w) // 2


def _render_template_block(sprite: np.ndarray, bg: int) -> np.ndarray:
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), int(bg), dtype=int)
    r0, c0 = _centered_top_left(sprite.shape)
    paste(block, sprite, r0, c0, transparent=int(bg))
    return block


def _extract_sprite(template_block: np.ndarray, bg: int) -> np.ndarray:
    sprite_box = bbox(template_block != int(bg))
    assert sprite_box is not None, "template block must contain a sprite"
    return crop_bbox(template_block, sprite_box)


def _non_bg_color_count(block: np.ndarray, bg: int) -> int:
    return len(np.unique(block[block != int(bg)]))


def _find_template_index(grid: np.ndarray, bg: int) -> int:
    candidates = [
        idx
        for idx in range(BLOCKS_PER_SIDE * BLOCKS_PER_SIDE)
        if _non_bg_color_count(_block_view(grid, idx), bg) >= 2
    ]
    assert len(candidates) == 1, "expected exactly one multicolor template block"
    return candidates[0]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 15x15 grid split into 5x5 blocks"
    bg = int(get_bg(g))
    template_index = _find_template_index(g, bg)
    template_block = _block_view(g, template_index).copy()
    sprite = _extract_sprite(template_block, bg)

    out = np.full_like(g, bg)
    tr0, tc0 = _block_origin(template_index)
    out[tr0 : tr0 + BLOCK_SIZE, tc0 : tc0 + BLOCK_SIZE] = template_block

    sr0, sc0 = _centered_top_left(sprite.shape)
    for idx in range(BLOCKS_PER_SIDE * BLOCKS_PER_SIDE):
        if idx == template_index:
            continue
        block = _block_view(g, idx)
        if np.any(block != bg):
            br0, bc0 = _block_origin(idx)
            paste(out, sprite, br0 + sr0, bc0 + sc0, transparent=bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use a block-level marker map to stamp the reference sprite into selected 5x5 blocks.

    Input structure
    - The 15x15 grid is partitioned into a 3x3 array of 5x5 blocks on a teal background.
    - Exactly one block contains a small multicolor reference sprite, while some other blocks contain only a few single-pixel markers.
    - The remaining blocks are entirely background.

    Rule
    - Identify the template block as the unique block with multiple non-background colors.
    - Extract the template sprite by taking the tight bounding box of its non-background pixels.
    - For every non-template block that contains any non-background pixel, stamp a centered copy of that sprite into the block.
    - Clear every other cell in non-template blocks back to the background color, while leaving the template block unchanged.

    Key insight
    - The scattered marker pixels are only a compressed presence map: each 5x5 block reduces to active or inactive, and that coarse map determines where the reference sprite must be copied.
    """
    return _solve(grid)


def _connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _generate_sprite(bg: int) -> tuple[np.ndarray, tuple[int, ...]]:
    height = random.choice(SPRITE_HEIGHTS)
    width = random.choice(SPRITE_WIDTHS)
    color_count = random.choice((2, 3))
    max_area = min(height * width, 8)
    area = random.randint(max(color_count, 3), max_area)
    mask = _connected_mask(height, width, area)
    assert mask is not None

    colors = tuple(random.sample(PALETTE, color_count))
    sprite = np.full((height, width), int(bg), dtype=int)
    coords = [tuple(coord) for coord in np.argwhere(mask)]
    random.shuffle(coords)
    for idx, (r, c) in enumerate(coords):
        sprite[r, c] = colors[idx] if idx < color_count else random.choice(colors)

    tight = crop_bbox(sprite, bbox(sprite != int(bg)))
    used_colors = tuple(sorted(np.unique(tight[tight != int(bg)]).tolist()))
    assert len(used_colors) == color_count
    return tight, used_colors


def _sample_marker_positions(sprite: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    sprite_h, sprite_w = map(int, sprite.shape)
    sr0, sc0 = _centered_top_left(sprite.shape)
    forbidden = {
        (r, c)
        for r in range(sr0, sr0 + sprite_h)
        for c in range(sc0, sc0 + sprite_w)
    }
    candidates = [
        (r, c)
        for r in range(BLOCK_SIZE)
        for c in range(BLOCK_SIZE)
        if (r, c) not in forbidden
    ]
    random.shuffle(candidates)

    picks: list[tuple[int, int]] = []
    for pos in candidates:
        if any(abs(pos[0] - rr) + abs(pos[1] - cc) <= 1 for rr, cc in picks):
            continue
        picks.append(pos)
        if len(picks) == count:
            return picks
    return None


def _expected_output(template_index: int, active_indices: set[int], sprite: np.ndarray, bg: int) -> np.ndarray:
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    template_block = _render_template_block(sprite, bg)
    tr0, tc0 = _block_origin(template_index)
    out[tr0 : tr0 + BLOCK_SIZE, tc0 : tc0 + BLOCK_SIZE] = template_block

    sr0, sc0 = _centered_top_left(sprite.shape)
    for idx in active_indices:
        br0, bc0 = _block_origin(idx)
        paste(out, sprite, br0 + sr0, bc0 + sc0, transparent=int(bg))
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = Color.TEAL
    sprite, sprite_colors = _generate_sprite(bg)
    remaining_colors = [color for color in PALETTE if color not in sprite_colors]
    marker_color = random.choice(remaining_colors)

    template_index = random.randrange(BLOCKS_PER_SIDE * BLOCKS_PER_SIDE)
    other_indices = [idx for idx in range(BLOCKS_PER_SIDE * BLOCKS_PER_SIDE) if idx != template_index]
    active_count = random.choice(ACTIVE_COUNT_OPTIONS)
    active_indices = set(random.sample(other_indices, active_count))

    grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    template_block = _render_template_block(sprite, bg)
    tr0, tc0 = _block_origin(template_index)
    grid[tr0 : tr0 + BLOCK_SIZE, tc0 : tc0 + BLOCK_SIZE] = template_block

    saw_two_markers = False
    for idx in active_indices:
        marker_count = random.choice((1, 2))
        saw_two_markers |= marker_count == 2
        positions = _sample_marker_positions(sprite, marker_count)
        if positions is None:
            return None
        br0, bc0 = _block_origin(idx)
        for r, c in positions:
            grid[br0 + r, bc0 + c] = marker_color

    if not saw_two_markers and active_indices:
        forced_idx = random.choice(tuple(active_indices))
        positions = _sample_marker_positions(sprite, 2)
        if positions is None:
            return None
        br0, bc0 = _block_origin(forced_idx)
        _block_view(grid, forced_idx)[:, :] = bg
        for r, c in positions:
            grid[br0 + r, bc0 + c] = marker_color

    expected = _expected_output(template_index, active_indices, sprite, bg)
    solved = main(grid)

    if not np.array_equal(solved, expected):
        return None
    if _find_template_index(grid, bg) != template_index:
        return None
    if sum(np.any(_block_view(grid, idx) != bg) for idx in other_indices) != active_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 040/10")
