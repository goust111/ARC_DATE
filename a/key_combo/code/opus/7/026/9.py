import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
INPUT_SHAPE = (3, 3)
OUTPUT_SHAPE = (15, 15)
BLOCK = 3
BUCKET_ORDER = [(4, 0), (5, 0), (5, 1), (6, 0), (6, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not coords:
        return False
    fg = set(coords)
    seen = {coords[0]}
    stack = [coords[0]]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in fg and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen == fg


def _edge_count(mask: np.ndarray) -> int:
    return sum(
        int(mask[r, c])
        for r in range(3)
        for c in range(3)
        if r in (0, 2) or c in (0, 2)
    )


def _validate_input_mask(mask: np.ndarray) -> None:
    assert mask.shape == INPUT_SHAPE
    fg_count = int(mask.sum())
    assert 4 <= fg_count <= 6
    assert fg_count != 9
    assert _is_connected(mask)
    assert _edge_count(mask) >= 2


def _validate_input_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.shape == INPUT_SHAPE
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE})
    mask = g == BLUE
    _validate_input_mask(mask)
    return g


def _block_origin(r: int, c: int) -> tuple[int, int]:
    return BLOCK * (r + 1), BLOCK * (c + 1)


def _stamp_origins(block_r: int, block_c: int, r: int, c: int):
    if r == 0:
        yield block_r - BLOCK, block_c
    if r == 2:
        yield block_r + BLOCK, block_c
    if c == 0:
        yield block_r, block_c - BLOCK
    if c == 2:
        yield block_r, block_c + BLOCK


def _corner_positions(block_r: int, block_c: int):
    return (
        (block_r - 1, block_c - 1),
        (block_r - 1, block_c + BLOCK),
        (block_r + BLOCK, block_c - 1),
        (block_r + BLOCK, block_c + BLOCK),
    )


def _expected_from_mask(mask: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    _validate_input_mask(mask)

    out = np.full(OUTPUT_SHAPE, BLACK, dtype=int)
    stamp_mask = np.zeros(OUTPUT_SHAPE, dtype=bool)
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]

    for r, c in coords:
        block_r, block_c = _block_origin(r, c)
        out[block_r : block_r + BLOCK, block_c : block_c + BLOCK] = BLUE

    for r, c in coords:
        block_r, block_c = _block_origin(r, c)
        for top, left in _stamp_origins(block_r, block_c, r, c):
            for rr, cc in coords:
                pr = top + rr
                pc = left + cc
                assert _in_bounds(OUTPUT_SHAPE, pr, pc)
                out[pr, pc] = BLUE
                stamp_mask[pr, pc] = True

    red_count = 0
    yellow_count = 0
    skipped_count = 0
    for r, c in coords:
        block_r, block_c = _block_origin(r, c)
        for cr, cc in _corner_positions(block_r, block_c):
            if not _in_bounds(OUTPUT_SHAPE, cr, cc):
                continue
            if stamp_mask[cr, cc]:
                out[cr, cc] = RED
                red_count += 1
            elif out[cr, cc] == BLACK:
                out[cr, cc] = YELLOW
                yellow_count += 1
            else:
                skipped_count += 1

    stats = {
        "red": red_count,
        "yellow": yellow_count,
        "visible": red_count + yellow_count,
        "skipped": skipped_count,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """Expand a 3x3 blue mask into 3x3 blocks, stamp it outward from edge blocks, and color block corners by stamp occupancy.

    Input structure
    - The input is a 3x3 black-background grid with a single blue foreground color.
    - Exactly 4 to 6 blue pixels form one 4-connected shape, and at least two of those pixels lie on the outer edge of the 3x3 grid.

    Rule
    - Create a 15x15 black canvas and map each blue input pixel to a solid blue 3x3 block in the centered 3x3 block lattice.
    - For every blue input pixel on the top, bottom, left, or right edge, stamp the original 3x3 blue mask immediately outward from the corresponding side of that block; corner edge pixels stamp once per touched side.
    - Then inspect each diagonal outside-corner cell around the solid blocks: if that cell is occupied by a stamp, paint it red; if it is empty, paint it yellow; if another solid block already occupies that diagonal position, leave it unchanged.

    Key insight
    - The corner color is not fixed by direction alone: it depends on whether the earlier outward stamps cover that exact corner cell, so the stamping geometry directly controls the later decoration.
    """
    g = _validate_input_grid(grid)
    out, _stats = _expected_from_mask(g == BLUE)
    return out


def _grid_from_mask(mask: np.ndarray) -> np.ndarray:
    grid = np.full(INPUT_SHAPE, BLACK, dtype=int)
    grid[mask] = BLUE
    return grid


def _build_generation_buckets() -> dict[tuple[int, int], list[np.ndarray]]:
    buckets: dict[tuple[int, int], list[np.ndarray]] = {bucket: [] for bucket in BUCKET_ORDER}
    seen_outputs: set[bytes] = set()

    for bits in itertools.product((0, 1), repeat=9):
        mask = np.array(bits, dtype=bool).reshape(INPUT_SHAPE)
        try:
            _validate_input_mask(mask)
        except AssertionError:
            continue

        expected, stats = _expected_from_mask(mask)
        if stats["red"] < 1 or stats["yellow"] < 1:
            continue
        if stats["visible"] < 12:
            continue
        if not (mask[0, :].any() or mask[2, :].any()):
            continue
        if not (mask[:, 0].any() or mask[:, 2].any()):
            continue

        grid = _grid_from_mask(mask)
        actual = main(grid)
        assert np.array_equal(actual, expected)

        sig = expected.tobytes()
        assert sig not in seen_outputs
        seen_outputs.add(sig)

        bucket = (int(mask.sum()), int(mask[1, 1]))
        if bucket in buckets:
            buckets[bucket].append(grid)

    for bucket in BUCKET_ORDER:
        assert buckets[bucket], f"empty generation bucket: {bucket}"
    return buckets


GEN_BUCKETS = _build_generation_buckets()
GEN_QUEUES: dict[tuple[int, int], list[np.ndarray]] = {bucket: [] for bucket in BUCKET_ORDER}
GEN_CALL_COUNT = 0


def generate_input() -> np.ndarray:
    global GEN_CALL_COUNT

    bucket = BUCKET_ORDER[GEN_CALL_COUNT % len(BUCKET_ORDER)]
    GEN_CALL_COUNT += 1

    if not GEN_QUEUES[bucket]:
        shuffled = list(GEN_BUCKETS[bucket])
        random.shuffle(shuffled)
        GEN_QUEUES[bucket] = shuffled

    return GEN_QUEUES[bucket].pop().copy()
