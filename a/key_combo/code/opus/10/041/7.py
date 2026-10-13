from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 9))
LONG_SIDE_OPTIONS = (6, 8, 10)
ORIENTATIONS = ("horizontal", "vertical")
MAX_GENERATION_ATTEMPTS = 400
MISSING_SLOTS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _read_blocks(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0, "input must split into 2x2 blocks"
    assert min(h, w) == 4, "input must be a 4xN or Nx4 strip"

    blocks: list[np.ndarray] = []
    for br in range(h // 2 - 1, -1, -1):
        for bc in range(w // 2 - 1, -1, -1):
            blocks.append(g[2 * br : 2 * br + 2, 2 * bc : 2 * bc + 2].copy())

    assert blocks, "expected at least one 2x2 block"
    return blocks


def _unfold_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    layers = np.asarray([block.reshape(4) for block in blocks], dtype=int)
    layer_count = int(layers.shape[0])
    out = np.zeros((2 * layer_count, 2 * layer_count), dtype=int)

    ul_seq, ur_seq, ll_seq, lr_seq = (layers[:, i] for i in range(4))
    for r in range(layer_count):
        for c in range(layer_count):
            out[r, c] = int(ul_seq[min(r, c)])
            out[r, layer_count + c] = int(ur_seq[min(r, layer_count - 1 - c)])
            out[layer_count + r, c] = int(ll_seq[min(layer_count - 1 - r, c)])
            out[layer_count + r, layer_count + c] = int(
                lr_seq[min(layer_count - 1 - r, layer_count - 1 - c)]
            )
    return out


def _fill_missing_by_priority(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    assert h == w and h % 2 == 0, "unfolded pattern must be an even square"

    for _ in range(out.size):
        zeros = np.argwhere(out == BACKGROUND)
        if len(zeros) == 0:
            return out

        changed = False
        for r_raw, c_raw in zeros:
            r = int(r_raw)
            c = int(c_raw)
            mirrors = (
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            )
            for rr, cc in mirrors:
                color = int(out[rr, cc])
                if color != BACKGROUND:
                    out[r, c] = color
                    changed = True
                    break
        if not changed:
            break

    assert not np.any(out == BACKGROUND), "every unfolded hole must be recoverable from a mirror"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a damaged 4-cell-wide strip of 2x2 blocks into nested quadrants and repair the holes by mirror-priority completion.

    Input structure:
    - The input is a 4xN or Nx4 strip whose cells split exactly into 2x2 blocks.
    - Blocks are read in reverse row-major order: bottom-right first, then leftward, then upward.
    - Most blocks are fully colored; one or two blocks contain exactly one erased pixel shown as black 0.

    Rule:
    - Treat each 2x2 block as one layer of a four-quadrant nested pattern: top-left pixel feeds the upper-left quadrant, top-right feeds upper-right, bottom-left feeds lower-left, and bottom-right feeds lower-right.
    - The first block in reverse row-major order becomes the outermost layer, and later blocks move inward.
    - After unfolding, any black cells are repaired by checking the vertical mirror first, then the horizontal mirror, then the point-symmetric cell, copying the first non-black color found.

    Key insight:
    - The strip itself is not the final pattern; only after unfolding does each erased block pixel become a structured missing layer region whose mirrored counterpart determines how to restore it.
    """
    blocks = _read_blocks(np.asarray(grid, dtype=int))
    unfolded = _unfold_blocks(blocks)
    return _fill_missing_by_priority(unfolded)


def _make_distinct_block() -> np.ndarray:
    colors = random.sample(PALETTE, 4)
    return np.array(colors, dtype=int).reshape(2, 2)


def _render_strip(blocks: list[np.ndarray], orientation: str) -> np.ndarray:
    layer_count = len(blocks)
    if orientation == "horizontal":
        h, w = 4, layer_count
    else:
        assert orientation == "vertical"
        h, w = layer_count, 4

    grid = np.zeros((h, w), dtype=int)
    positions = [
        (br, bc)
        for br in range(h // 2 - 1, -1, -1)
        for bc in range(w // 2 - 1, -1, -1)
    ]
    assert len(positions) == layer_count
    for block, (br, bc) in zip(blocks, positions):
        grid[2 * br : 2 * br + 2, 2 * bc : 2 * bc + 2] = block
    return grid


def _validate_generated_case(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    blocks = _read_blocks(g)
    if not (1 <= np.count_nonzero(g == BACKGROUND) <= 2):
        return False

    damaged = 0
    for block in blocks:
        zeros = np.argwhere(block == BACKGROUND)
        if len(zeros) > 1:
            return False
        visible = [int(v) for v in block.reshape(-1) if int(v) != BACKGROUND]
        if len(zeros) == 1:
            damaged += 1
            if len(visible) != 3 or len(set(visible)) != 3:
                return False
        else:
            if len(set(map(int, block.reshape(-1)))) != 4:
                return False
    if damaged not in {1, 2}:
        return False

    if len(set(map(int, g[g != BACKGROUND].reshape(-1)))) < 5:
        return False

    raw = _unfold_blocks(blocks)
    solved = main(g)
    if solved.shape != raw.shape:
        return False
    if np.any((raw != BACKGROUND) & (solved != raw)):
        return False

    hole_cells = np.argwhere(raw == BACKGROUND)
    if len(hole_cells) == 0:
        return False
    for r_raw, c_raw in hole_cells:
        r = int(r_raw)
        c = int(c_raw)
        mirrors = (
            (r, raw.shape[1] - 1 - c),
            (raw.shape[0] - 1 - r, c),
            (raw.shape[0] - 1 - r, raw.shape[1] - 1 - c),
        )
        chosen = None
        for rr, cc in mirrors:
            color = int(raw[rr, cc])
            if color != BACKGROUND:
                chosen = color
                break
        if chosen is None or int(solved[r, c]) != chosen:
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    layer_count = random.choice(LONG_SIDE_OPTIONS)
    orientation = random.choice(ORIENTATIONS)

    blocks = [_make_distinct_block() for _ in range(layer_count)]
    missing_count = random.choice((1, 2))
    missing_layers = random.sample(range(layer_count), missing_count)
    for layer_index in missing_layers:
        rr, cc = random.choice(MISSING_SLOTS)
        blocks[layer_index][rr, cc] = BACKGROUND

    grid = _render_strip(blocks, orientation)
    if not _validate_generated_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
