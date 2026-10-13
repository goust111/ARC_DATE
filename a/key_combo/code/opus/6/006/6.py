from __future__ import annotations

from collections import Counter
import random

import numpy as np


BBox = tuple[int, int, int, int]

BG = 8
FG = 7
BLOCK = 3
MOTIF = BLOCK * 3
EXPANDED = BLOCK * 5
MAX_RETRIES = 400

# (block_row, block_col, src_local_r, src_local_c, hole_local_r, hole_local_c)
# Side blocks use the midpoint on the center-facing edge; diagonal blocks use the
# inner corner facing the empty center. The decoded bit is written to the
# 180-degree opposite slot inside the reconstructed center patch.
CONTACTS = (
    (0, 0, 2, 2, 0, 0),
    (0, 1, 2, 1, 0, 1),
    (0, 2, 2, 0, 0, 2),
    (1, 0, 1, 2, 1, 0),
    (1, 2, 1, 0, 1, 2),
    (2, 0, 0, 2, 2, 0),
    (2, 1, 0, 1, 2, 1),
    (2, 2, 0, 0, 2, 2),
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _motif_info(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = sorted(set(map(int, np.unique(g))) - {bg})
    assert fg_colors == [FG], "expected exactly one orange foreground color"
    box = bbox(g != bg)
    assert box is not None, "expected a foreground motif"
    r0, c0, r1, c1 = box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (MOTIF, MOTIF), "motif must occupy one 9x9 box"
    assert r0 >= BLOCK and c0 >= BLOCK, "need margin for 5x5 expansion"
    assert r1 + BLOCK < g.shape[0] and c1 + BLOCK < g.shape[1], "need margin for 5x5 expansion"
    return bg, r0, c0, FG


def _decode_center(grid: np.ndarray, r0: int, c0: int, bg: int, fg: int) -> np.ndarray:
    center = np.full((BLOCK, BLOCK), bg, dtype=int)
    center[1, 1] = fg
    hole = grid[r0 + BLOCK : r0 + 2 * BLOCK, c0 + BLOCK : c0 + 2 * BLOCK]
    assert int(hole[1, 1]) == bg, "input center must be empty before reconstruction"

    for br, bc, sr, sc, hr, hc in CONTACTS:
        src_r = r0 + br * BLOCK + sr
        src_c = c0 + bc * BLOCK + sc
        hole_r = r0 + BLOCK + hr
        hole_c = c0 + BLOCK + hc
        src_is_fg = int(grid[src_r, src_c]) == fg
        hole_is_fg = int(grid[hole_r, hole_c]) == fg
        assert src_is_fg == hole_is_fg, "each contact slot must be either a tab or a notch"
        target_r = BLOCK - 1 - hr
        target_c = BLOCK - 1 - hc
        center[target_r, target_c] = fg if hole_is_fg else bg

    return center


def _build_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, r0, c0, fg = _motif_info(g)
    out = g.copy()

    paste(out, _decode_center(g, r0, c0, bg, fg), r0 + BLOCK, c0 + BLOCK)

    expanded_r0 = r0 - BLOCK
    expanded_c0 = c0 - BLOCK
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = crop_bbox(
                g,
                (
                    r0 + br * BLOCK,
                    c0 + bc * BLOCK,
                    r0 + (br + 1) * BLOCK - 1,
                    c0 + (bc + 1) * BLOCK - 1,
                ),
            )
            dst_r = expanded_r0 + (4 - 2 * br) * BLOCK
            dst_c = expanded_c0 + (4 - 2 * bc) * BLOCK
            paste(out, src, dst_r, dst_c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing center block from tab-or-notch contact bits and mirror the surrounding blocks to opposite outer slots.

    Input structure:
    - Teal background surrounds one orange 9x9 motif arranged as a 3x3 grid of 3x3 blocks.
    - The middle 3x3 block is empty teal, while the other eight blocks are orange with one center-facing contact slot each.
    - On side blocks the contact slot is the midpoint of the edge facing the hole; on diagonal blocks it is the inner corner facing the hole.
    - A contact slot is either a tab (the block keeps its slot and also paints the next center-facing pixel) or a notch (both positions are teal).

    Rule:
    - Read the eight contact slots around the empty center: tab = 1 and notch = 0.
    - Write each decoded bit into the 180-degree opposite position of the missing 3x3 center block, and fill the center cell with orange.
    - Keep the original 3x3 ring of blocks in place, then copy each surrounding 3x3 block to the opposite outer slot of the implied 5x5 block grid without rotating the copied block.

    Key insight:
    - The visible tabs and notches are not the answer pixels themselves; they are binary contact cues that must be mapped across the hole by opposite-slot symmetry before the larger 5x5 block layout can be completed.
    """
    return _build_output(grid)


def _random_code_ring() -> np.ndarray:
    while True:
        ring = np.full((BLOCK, BLOCK), -1, dtype=int)
        choices = [0, 1]
        vals = [random.choice(choices) for _ in range(8)]
        ones = sum(vals)
        if not (2 <= ones <= 6):
            continue
        for idx, (_br, _bc, _sr, _sc, hr, hc) in enumerate(CONTACTS):
            ring[hr, hc] = vals[idx]
        ring[1, 1] = 1
        visible = ring.copy()
        target = np.rot90(visible, 2)
        if np.array_equal(visible, target):
            continue
        return ring


def _make_input_from_ring(ring: np.ndarray, h: int, w: int, r0: int, c0: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    motif = np.full((MOTIF, MOTIF), BG, dtype=int)

    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK] = FG

    motif[BLOCK : 2 * BLOCK, BLOCK : 2 * BLOCK] = BG
    for br, bc, sr, sc, hr, hc in CONTACTS:
        if int(ring[hr, hc]) == 1:
            motif[BLOCK + hr, BLOCK + hc] = FG
        else:
            motif[br * BLOCK + sr, bc * BLOCK + sc] = BG

    paste(grid, motif, r0, c0)
    return grid


def _valid_example(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(inp, out):
        return False

    bg, r0, c0, _fg = _motif_info(inp)
    visible = inp[r0 + BLOCK : r0 + 2 * BLOCK, c0 + BLOCK : c0 + 2 * BLOCK].copy()
    target = out[r0 + BLOCK : r0 + 2 * BLOCK, c0 + BLOCK : c0 + 2 * BLOCK].copy()
    if int(visible[1, 1]) != bg:
        return False
    if np.array_equal(visible, target):
        return False
    expected_target = np.rot90(visible, 2).astype(int)
    expected_target[1, 1] = FG
    if not np.array_equal(target, expected_target):
        return False
    if len({
        crop_bbox(
            inp,
            (
                r0 + br * BLOCK,
                c0 + bc * BLOCK,
                r0 + (br + 1) * BLOCK - 1,
                c0 + (bc + 1) * BLOCK - 1,
            ),
        ).tobytes()
        for br in range(3)
        for bc in range(3)
        if (br, bc) != (1, 1)
    }) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    if h < EXPANDED or w < EXPANDED:
        return None

    r0 = random.randint(BLOCK, h - 12)
    c0 = random.randint(BLOCK, w - 12)
    ring = _random_code_ring()
    inp = _make_input_from_ring(ring, h, w, r0, c0)
    out = _build_output(inp)

    if not _valid_example(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
