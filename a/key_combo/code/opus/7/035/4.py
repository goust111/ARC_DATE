from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
SPRITE_SIZE = 5
SEPARATOR_WIDTH = 1
SPRITE_PALETTE = (1, 3, 4, 5, 7)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _separator_columns(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == RED)]


def _split_blocks(grid: np.ndarray) -> list[np.ndarray]:
    starts = [0]
    for c in _separator_columns(grid):
        starts.append(c + 1)

    stops = _separator_columns(grid) + [grid.shape[1]]
    blocks: list[np.ndarray] = []
    for start, stop in zip(starts, stops):
        if stop > start:
            block = grid[:, start:stop].copy()
            assert block.shape == (SPRITE_SIZE, SPRITE_SIZE)
            blocks.append(block)
    assert blocks
    return blocks


def _assemble_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    height = blocks[0].shape[0]
    width = len(blocks) * SPRITE_SIZE + (len(blocks) - 1) * SEPARATOR_WIDTH
    out = np.full((height, width), BLACK, dtype=int)
    col = 0
    for index, block in enumerate(blocks):
        out[:, col : col + SPRITE_SIZE] = block
        col += SPRITE_SIZE
        if index < len(blocks) - 1:
            out[:, col] = RED
            col += 1
    return out


def _center_count(block: np.ndarray) -> int:
    return int(np.count_nonzero(block[:, 2] != BLACK))


def _erase_center_column(block: np.ndarray) -> np.ndarray:
    out = block.copy()
    out[:, 2] = BLACK
    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _render_expected(blocks: list[np.ndarray]) -> np.ndarray:
    ordered = sorted(blocks, key=_center_count)
    return _assemble_blocks([_erase_center_column(block) for block in ordered])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the sprite blocks by center-column occupancy, then erase the center column in every block.

    Input structure:
    - The grid is 5 cells tall and contains several 5x5 monochrome sprites laid out in one horizontal row.
    - Neighboring sprites are separated by full-height red divider columns, while the remaining background is black.
    - Each sprite has a unique non-red foreground color, and its middle column contains a distinct number of foreground pixels.

    Rule:
    - Split the grid at the red separator columns to recover the 5x5 sprite blocks.
    - For each block, count how many non-black cells appear in the block's center column.
    - Reorder the blocks by that count from smallest to largest.
    - After reordering, replace the entire center column of every block with black and keep the red separators between adjacent blocks.

    Key insight:
    - The center column has a dual role: it is the visible sorting key in the input, but it must disappear entirely in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] == SPRITE_SIZE

    blocks = _split_blocks(g)
    return _render_expected(blocks)


def _grow_sprite_mask(center_rows: set[int], target_size: int, rng: random.Random) -> np.ndarray | None:
    mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    for r in center_rows:
        mask[r, 2] = True

    blocked_center = {(r, 2) for r in range(SPRITE_SIZE) if r not in center_rows}

    while int(mask.sum()) < target_size:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if not (0 <= nr < SPRITE_SIZE and 0 <= nc < SPRITE_SIZE):
                    continue
                if mask[nr, nc] or (nr, nc) in blocked_center:
                    continue
                frontier.append((nr, nc))

        if not frontier:
            return None

        left_or_right = [cell for cell in frontier if cell[1] != 2]
        candidates = left_or_right if left_or_right and rng.random() < 0.85 else frontier
        nr, nc = rng.choice(candidates)
        mask[nr, nc] = True

    return mask


def _generate_sprite(color: int, center_count: int, rng: random.Random) -> np.ndarray | None:
    min_size = max(6, center_count + 3)
    for _ in range(200):
        start_row = rng.randint(0, SPRITE_SIZE - center_count)
        center_rows = set(range(start_row, start_row + center_count))
        target_size = rng.randint(min_size, 15)
        mask = _grow_sprite_mask(center_rows, target_size, rng)
        if mask is None:
            continue
        if not _mask_is_connected(mask):
            continue

        non_center_count = int(mask.sum()) - center_count
        if non_center_count < 3:
            continue
        if int(mask[:, :2].sum()) == 0 and int(mask[:, 3:].sum()) == 0:
            continue

        sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
        sprite[mask] = color
        assert _center_count(sprite) == center_count
        return sprite
    return None


def _sample_counts(k: int, rng: random.Random) -> list[int]:
    counts = rng.sample([1, 2, 3, 4, 5], k)
    rng.shuffle(counts)
    return counts


def _non_sorted_order(counts: list[int], rng: random.Random) -> list[int]:
    indices = list(range(len(counts)))
    for _ in range(50):
        rng.shuffle(indices)
        permuted = [counts[i] for i in indices]
        if permuted != sorted(permuted):
            return indices
    return list(reversed(sorted(indices, key=lambda i: counts[i])))


def _valid_input_grid(grid: np.ndarray, blocks: list[np.ndarray], counts: list[int]) -> bool:
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return False
    if sorted(counts) == counts:
        return False
    expected = _render_expected(blocks)
    if np.array_equal(grid, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    for block, count in zip(blocks, counts):
        if _center_count(block) != count:
            return False
        if np.unique(block[block != BLACK]).size != 1:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    k = rng.choice([4, 5])
    colors = rng.sample(list(SPRITE_PALETTE), k)
    counts = _sample_counts(k, rng)

    unsorted_blocks: list[np.ndarray] = []
    unsorted_counts: list[int] = []
    for color, count in zip(colors, counts):
        sprite = _generate_sprite(color, count, rng)
        if sprite is None:
            return None
        unsorted_blocks.append(sprite)
        unsorted_counts.append(count)

    order = _non_sorted_order(unsorted_counts, rng)
    blocks = [unsorted_blocks[i] for i in order]
    counts_in_input = [unsorted_counts[i] for i in order]
    grid = _assemble_blocks(blocks)

    if not _valid_input_grid(grid, blocks, counts_in_input):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
