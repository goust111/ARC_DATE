from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
GREY = 5
VISIBLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _infer_block_shape(grid: np.ndarray) -> tuple[int, int]:
    h_big, w_big = map(int, grid.shape)
    h = math.isqrt(h_big)
    w = math.isqrt(w_big)
    assert h >= 2 and w >= 2, "block shape must be at least 2x2"
    assert h * h == h_big and w * w == w_big, "input must be tiled as pattern-sized square counts"
    return h, w


def _split_blocks(grid: np.ndarray, block_h: int, block_w: int) -> list[list[np.ndarray]]:
    block_rows = []
    for br in range(block_h):
        row = []
        for bc in range(block_w):
            row.append(grid[br * block_h : (br + 1) * block_h, bc * block_w : (bc + 1) * block_w].copy())
        block_rows.append(row)
    return block_rows


def _reconstruct_from_blocks(blocks: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = len(blocks)
    w = len(blocks[0])
    layout = np.zeros((h, w), dtype=bool)
    merged = np.zeros((h, w), dtype=int)

    for br in range(h):
        for bc in range(w):
            block = np.asarray(blocks[br][bc], dtype=int)
            assert block.shape == (h, w), "every block must match the recovered pattern size"
            assert not np.any(block == GREY), "grey may only appear via block layout, never inside a copy"
            nonblack = block != BLACK
            if np.any(nonblack):
                layout[br, bc] = True
            for r, c in np.argwhere(nonblack):
                color = int(block[r, c])
                here = int(merged[int(r), int(c)])
                assert here in (BLACK, color), "partial copies must agree where they overlap"
                merged[int(r), int(c)] = color

    assert int(layout.sum()) >= 2, "need at least two non-empty blocks"
    assert np.any(~layout), "need at least one empty block"
    assert not np.any(merged[layout]), "grey pixels must be absent from every copy"
    assert np.all(layout | (merged != BLACK)), "every non-grey output pixel must appear in some copy"

    for br in range(h):
        for bc in range(w):
            block = blocks[br][bc]
            assert not np.any((block != BLACK) & layout), "copies may not reveal grey-mask locations"
            assert np.all((block == BLACK) | (block == merged)), "copies are only allowed to hide cells"

    out = merged.copy()
    out[layout] = GREY
    return out, layout, merged


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the hidden original tile by merging partial copies and reading grey pixels from the block layout.

    Input structure:
    - The whole grid is partitioned into equal blocks, and each block has the same size as the unknown output pattern.
    - Some block positions contain partial copies of that pattern while the other blocks are all black.
    - Grey never appears inside any copy; the only clue about grey is which block slots are non-empty.

    Rule:
    - Infer the block size from the square tiling, then overlay all non-empty blocks cellwise.
    - Any non-black color seen in at least one copy is a recovered non-grey pixel of the pattern.
    - Treat the macro-layout of non-empty blocks as a grey mask on the recovered pattern and color those cells grey.

    Key insight:
    - The layout is self-descriptive: missing grey pixels are not hidden inside the partial copies at all, they are encoded by where those copies were placed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids"

    block_h, block_w = _infer_block_shape(g)
    blocks = _split_blocks(g, block_h, block_w)
    out, _layout, _merged = _reconstruct_from_blocks(blocks)
    return out


def _build_pattern(height: int, width: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    total = len(cells)
    max_grey = min((total - 1) // 2, total - 4)
    grey_count = random.randint(2, max_grey)

    grey_positions: list[tuple[int, int]]
    for _ in range(80):
        grey_positions = random.sample(cells, grey_count)
        rows = {r for r, _c in grey_positions}
        cols = {c for _r, c in grey_positions}
        if len(rows) >= 2 and len(cols) >= 2:
            break
    else:
        raise RuntimeError("failed to sample a non-degenerate grey layout")

    nongrey_positions = [cell for cell in cells if cell not in grey_positions]
    color_count = random.choice((2, 3))
    colors = random.sample(list(VISIBLE_COLORS), color_count)

    pattern = np.zeros((height, width), dtype=int)
    for r, c in grey_positions:
        pattern[r, c] = GREY

    shuffled = nongrey_positions[:]
    random.shuffle(shuffled)
    for pos, color in zip(shuffled, colors):
        pattern[pos] = color
    for pos in shuffled[color_count:]:
        pattern[pos] = random.choice(colors)

    return pattern, grey_positions, nongrey_positions


def _build_cover_sets(
    nongrey_positions: list[tuple[int, int]],
    copy_count: int,
) -> list[set[tuple[int, int]]]:
    positions = nongrey_positions[:]
    random.shuffle(positions)
    covers = [set() for _ in range(copy_count)]

    for idx, pos in enumerate(positions[:copy_count]):
        covers[idx].add(pos)

    shared_written = False
    for pos in positions[copy_count:]:
        if not shared_written:
            chosen = random.sample(range(copy_count), random.randint(2, min(copy_count, 3)))
            shared_written = True
        else:
            chosen = random.sample(range(copy_count), random.randint(1, min(copy_count, 3)))
        for idx in chosen:
            covers[idx].add(pos)

    for idx in range(copy_count):
        if len(covers[idx]) < 2:
            extras = [pos for pos in positions if pos not in covers[idx]]
            if not extras:
                return []
            covers[idx].add(random.choice(extras))

    essential = positions[:copy_count]
    essential_set = set(essential)
    full_size = len(positions)
    for idx in range(copy_count):
        while len(covers[idx]) == full_size:
            removable = [
                pos
                for pos in covers[idx]
                if pos not in essential_set and sum(pos in other for other in covers) >= 2
            ]
            if not removable:
                return []
            covers[idx].remove(random.choice(removable))

    counts = {pos: sum(pos in cover for cover in covers) for pos in positions}
    if not any(count >= 2 for count in counts.values()):
        return []
    if not all(count >= 1 for count in counts.values()):
        return []
    if not all(any(counts[pos] == 1 for pos in cover) for cover in covers):
        return []
    return covers


def _assemble_input(
    pattern: np.ndarray,
    grey_positions: list[tuple[int, int]],
    covers: list[set[tuple[int, int]]],
) -> np.ndarray:
    h, w = pattern.shape
    grid = np.zeros((h * h, w * w), dtype=int)
    block_slots = grey_positions[:]
    random.shuffle(block_slots)
    for (br, bc), cover in zip(block_slots, covers):
        block = np.zeros((h, w), dtype=int)
        for r, c in cover:
            block[r, c] = int(pattern[r, c])
        grid[br * h : (br + 1) * h, bc * w : (bc + 1) * w] = block
    return grid


def _is_every_copy_necessary(grid: np.ndarray, expected: np.ndarray) -> bool:
    h, w = expected.shape
    for br in range(h):
        for bc in range(w):
            r0, c0 = br * h, bc * w
            block = grid[r0 : r0 + h, c0 : c0 + w]
            if not np.any(block):
                continue
            trial = grid.copy()
            trial[r0 : r0 + h, c0 : c0 + w] = 0
            try:
                got = main(trial)
            except AssertionError:
                continue
            if np.array_equal(got, expected):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(3, 5)
    width = random.randint(3, 5)
    pattern, grey_positions, nongrey_positions = _build_pattern(height, width)
    copy_count = len(grey_positions)
    covers = _build_cover_sets(nongrey_positions, copy_count)
    if not covers:
        return None

    grid = _assemble_input(pattern, grey_positions, covers)

    try:
        got = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(got, pattern):
        return None
    if not _is_every_copy_necessary(grid, pattern):
        return None

    block_h, block_w = _infer_block_shape(grid)
    blocks = _split_blocks(grid, block_h, block_w)
    _out, layout, merged = _reconstruct_from_blocks(blocks)
    if not np.array_equal(layout, pattern == GREY):
        return None
    if np.any(merged[~layout] == BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential block-restoration input")
