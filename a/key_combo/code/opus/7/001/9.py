from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
ORANGE = 7
BLOCKS_PER_SIDE = 3

MIN_BLOCK_SIZE = 4
MAX_BLOCK_SIZE = 8
MIN_BLUE_AREA = 3
MAX_BLUE_AREA = 6
MAX_MARKERS = 3

SHAPE_ATTEMPTS = 200
GRID_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        q = deque([(r, c)])
        seen[r, c] = True
        while q:
            rr, cc = q.popleft()
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _mask_signature(mask: np.ndarray) -> bytes:
    tight = _trim_mask(mask.astype(bool))
    return tight.astype(np.uint8).tobytes() + str(tight.shape).encode()


def _is_valid_shape(mask: np.ndarray) -> bool:
    tight = _trim_mask(mask.astype(bool))
    area = int(tight.sum())
    if area < MIN_BLUE_AREA or area > MAX_BLUE_AREA:
        return False
    if _component_count(tight) != 1:
        return False
    if min(tight.shape) < 2:
        return False
    if np.array_equal(tight, np.fliplr(tight)):
        return False
    if np.array_equal(tight, np.flipud(tight)):
        return False
    rotations = {_mask_signature(np.rot90(tight, k=-k)) for k in range(4)}
    return len(rotations) == 4


def _random_shape(max_dim: int) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        canvas = np.zeros((5, 5), dtype=bool)
        target = random.randint(MIN_BLUE_AREA, MAX_BLUE_AREA)
        start = (random.randint(1, 3), random.randint(1, 3))
        cells = {start}
        canvas[start] = True

        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            cells.add((nr, nc))
            canvas[nr, nc] = True

        tight = _trim_mask(canvas)
        if max(tight.shape) > max_dim:
            continue
        if _is_valid_shape(tight):
            return tight.astype(bool)
    raise RuntimeError("failed to generate a valid asymmetric connected shape")


def _build_block(block_size: int, marker_count: int) -> tuple[np.ndarray, np.ndarray]:
    shape = _random_shape(block_size)
    h, w = shape.shape
    placements = [(r0, c0) for r0 in range(block_size - h + 1) for c0 in range(block_size - w + 1)]
    random.shuffle(placements)

    for r0, c0 in placements:
        block = np.zeros((block_size, block_size), dtype=int)
        block[r0 : r0 + h, c0 : c0 + w][shape] = BLUE
        empty = np.argwhere(block == BLACK)
        if len(empty) < marker_count:
            continue
        if marker_count:
            chosen = random.sample([tuple(map(int, cell)) for cell in empty], marker_count)
            for rr, cc in chosen:
                block[rr, cc] = ORANGE
        return block, shape

    raise RuntimeError("failed to place shape and markers inside a block")


def _split_blocks(grid: np.ndarray) -> list[np.ndarray]:
    n = int(grid.shape[0])
    assert grid.ndim == 2 and grid.shape[0] == grid.shape[1]
    assert n % BLOCKS_PER_SIDE == 0
    block_size = n // BLOCKS_PER_SIDE
    blocks = []
    for br in range(BLOCKS_PER_SIDE):
        for bc in range(BLOCKS_PER_SIDE):
            r0 = br * block_size
            c0 = bc * block_size
            blocks.append(grid[r0 : r0 + block_size, c0 : c0 + block_size].copy())
    return blocks


def _assemble_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    assert len(blocks) == BLOCKS_PER_SIDE * BLOCKS_PER_SIDE
    block_size = int(blocks[0].shape[0])
    out = np.zeros((BLOCKS_PER_SIDE * block_size, BLOCKS_PER_SIDE * block_size), dtype=int)
    for idx, block in enumerate(blocks):
        br = idx // BLOCKS_PER_SIDE
        bc = idx % BLOCKS_PER_SIDE
        r0 = br * block_size
        c0 = bc * block_size
        out[r0 : r0 + block_size, c0 : c0 + block_size] = block
    return out


def _rotate_block(block: np.ndarray, turns: int, *, clockwise: bool = True) -> np.ndarray:
    k = int(turns) % 4
    rotated = np.rot90(block, k=-k if clockwise else k)
    out = rotated.copy()
    out[out == ORANGE] = BLACK
    return out


def _expected_output(grid: np.ndarray, counts: list[int] | None = None, *, clockwise: bool = True) -> np.ndarray:
    blocks = _split_blocks(grid)
    if counts is None:
        counts = [int(np.count_nonzero(block == ORANGE)) for block in blocks]
    out_blocks = [_rotate_block(block, turns, clockwise=clockwise) for block, turns in zip(blocks, counts)]
    return _assemble_blocks(out_blocks)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each coarse block by a number of quarter-turns encoded by its orange markers, then erase the markers.

    Input structure:
    - The grid is a square black canvas whose side length is divisible by 3.
    - It is partitioned into a conceptual 3x3 layout of equal square blocks.
    - Each block contains one small 4-connected blue pattern and zero to three orange marker pixels on black background.

    Rule:
    - Split the input into the 3x3 equal blocks.
    - For each block, count its orange pixels as K.
    - Rotate that entire block K times by 90 degrees clockwise.
    - After rotation, erase every orange pixel in the block to black.
    - Reassemble the nine processed blocks into the output grid.

    Key insight:
    - The orange pixels are not content to keep; they are per-block rotation instructions where the local count directly encodes the quarter-turn amount.
    """
    g = np.asarray(grid, dtype=int)
    blocks = _split_blocks(g)
    counts = [int(np.count_nonzero(block == ORANGE)) for block in blocks]
    return _expected_output(g, counts=counts, clockwise=True)


def _validate_generated_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert grid.shape[0] == grid.shape[1]
    assert max(grid.shape) <= 30
    assert grid.shape[0] % BLOCKS_PER_SIDE == 0
    assert set(map(int, np.unique(grid))) <= {BLACK, BLUE, ORANGE}

    blocks = _split_blocks(grid)
    counts = []
    shape_sigs = set()
    for block in blocks:
        blue_mask = block == BLUE
        orange_mask = block == ORANGE
        blue_area = int(blue_mask.sum())
        orange_count = int(orange_mask.sum())
        counts.append(orange_count)

        assert MIN_BLUE_AREA <= blue_area <= MAX_BLUE_AREA
        assert 0 <= orange_count <= MAX_MARKERS
        assert _component_count(blue_mask) == 1
        tight = _trim_mask(blue_mask)
        assert _is_valid_shape(tight)
        assert not np.any(blue_mask & orange_mask)
        shape_sigs.add(_mask_signature(tight))

    assert len(set(counts)) >= 3
    assert {0, 1, 2, 3}.issubset(set(counts))
    assert len(shape_sigs) >= 3

    out = main(grid)
    assert out.shape == grid.shape
    assert set(map(int, np.unique(out))) <= {BLACK, BLUE}
    assert not np.any(out == ORANGE)
    assert not np.array_equal(out, grid)

    correct = _expected_output(grid, clockwise=True)
    no_rotation = _expected_output(grid, counts=[0] * 9, clockwise=True)
    ccw = _expected_output(grid, clockwise=False)
    total_turns = sum(counts) % 4
    uniform_global = _expected_output(grid, counts=[total_turns] * 9, clockwise=True)

    assert np.array_equal(out, correct)
    assert not np.array_equal(out, no_rotation)
    assert not np.array_equal(out, ccw)
    assert not np.array_equal(out, uniform_global)


def generate_input() -> np.ndarray:
    for _ in range(GRID_ATTEMPTS):
        block_size = random.randint(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE)
        counts = [0, 1, 2, 3]
        counts.extend(random.randint(0, 3) for _ in range(5))
        random.shuffle(counts)

        blocks = []
        for count in counts:
            block, _shape = _build_block(block_size, count)
            blocks.append(block)

        grid = _assemble_blocks(blocks)
        try:
            _validate_generated_grid(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
