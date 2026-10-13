from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
RED = 2
GREY = 5
ORANGE = 7

MOTIF_SIZE = 3
OUTPUT_SIZE = MOTIF_SIZE * MOTIF_SIZE
INSTR_CELL_SIZE = 2
INSTR_STRIDE = 3
INSTR_SIZE = INSTR_CELL_SIZE * MOTIF_SIZE + (MOTIF_SIZE - 1)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_instruction_origin(grid: np.ndarray) -> tuple[int, int]:
    grey_rows = [r for r in range(grid.shape[0]) if np.count_nonzero(grid[r] == GREY) >= INSTR_SIZE - 1]
    grey_cols = [c for c in range(grid.shape[1]) if np.count_nonzero(grid[:, c] == GREY) >= INSTR_SIZE - 1]
    assert len(grey_rows) == 2 and len(grey_cols) == 2, "expected one 3x3 grey lattice"
    assert grey_rows[1] - grey_rows[0] == INSTR_STRIDE, "unexpected instruction row spacing"
    assert grey_cols[1] - grey_cols[0] == INSTR_STRIDE, "unexpected instruction col spacing"
    r0 = grey_rows[0] - INSTR_CELL_SIZE
    c0 = grey_cols[0] - INSTR_CELL_SIZE
    assert 0 <= r0 and 0 <= c0, "instruction origin must be in bounds"
    assert r0 + INSTR_SIZE <= grid.shape[0] and c0 + INSTR_SIZE <= grid.shape[1], "instruction region must fit"
    return r0, c0


def _decode_mask(grid: np.ndarray) -> np.ndarray:
    r0, c0 = _find_instruction_origin(grid)
    mask = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=bool)
    for br in range(MOTIF_SIZE):
        for bc in range(MOTIF_SIZE):
            rs = r0 + br * INSTR_STRIDE
            cs = c0 + bc * INSTR_STRIDE
            block = grid[rs : rs + INSTR_CELL_SIZE, cs : cs + INSTR_CELL_SIZE]
            assert block.shape == (INSTR_CELL_SIZE, INSTR_CELL_SIZE)
            if np.all(block == ORANGE):
                mask[br, bc] = True
            else:
                assert np.all(block == BACKGROUND), "instruction cells must be solid orange or black"
    return mask


def _extract_motif_mask(grid: np.ndarray) -> np.ndarray:
    motif_pixels = grid == RED
    box = _bbox(motif_pixels)
    assert box is not None, "expected a red motif"
    r0, c0, r1, c1 = box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (MOTIF_SIZE, MOTIF_SIZE), "expected a 3x3 motif bbox"
    motif = motif_pixels[r0 : r1 + 1, c0 : c1 + 1]
    assert 3 <= int(motif.sum()) <= 7, "unexpected motif density"
    assert _is_connected4(motif), "motif must be 4-connected"
    return motif


def _expand_and_filter(motif: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kept_stamp = motif & mask
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for br, bc in np.argwhere(motif):
        rs = int(br) * MOTIF_SIZE
        cs = int(bc) * MOTIF_SIZE
        out[rs : rs + MOTIF_SIZE, cs : cs + MOTIF_SIZE][kept_stamp] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the red 3x3 motif into a self-similar 9x9 grid, then keep only the positions selected by the decoded orange mask inside every copied block.

    Input structure:
    - A black background contains one grey 3x3 lattice whose 2x2 subcells encode a binary mask with solid orange blocks.
    - A separate compact red motif has tight 3x3 bounding box and 4-connected foreground.
    - The grey instruction lattice and the red motif are spatially separated and are the only non-background structures.

    Rule:
    - Decode the grey lattice into a 3x3 mask: each solid orange 2x2 subcell means keep, each black 2x2 subcell means erase.
    - Crop the red motif from its 3x3 bounding box and perform self-similar nesting: every red motif pixel selects one 3x3 output block that receives a full copy of the motif.
    - Inside each copied block, erase every local position where the decoded mask is off; keep red only where both the copied motif and the decoded mask are on.

    Key insight:
    - The instruction mask does not choose which outer 3x3 blocks exist; it filters the same local coordinates inside every self-similar copy produced by the motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask = _decode_mask(g)
    motif = _extract_motif_mask(g)
    return _expand_and_filter(motif, mask)


def _random_connected_motif() -> np.ndarray:
    while True:
        target = random.randint(3, 7)
        cells = {(random.randrange(MOTIF_SIZE), random.randrange(MOTIF_SIZE))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < MOTIF_SIZE and 0 <= nc < MOTIF_SIZE and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        mask = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        box = _bbox(mask)
        assert box is not None
        if box != (0, 0, MOTIF_SIZE - 1, MOTIF_SIZE - 1):
            continue
        if not _is_connected4(mask):
            continue
        return mask


def _random_filter_mask(motif: np.ndarray) -> np.ndarray:
    motif_count = int(motif.sum())
    while True:
        on_count = random.randint(2, 7)
        flat = np.zeros(MOTIF_SIZE * MOTIF_SIZE, dtype=bool)
        chosen = random.sample(range(MOTIF_SIZE * MOTIF_SIZE), on_count)
        flat[chosen] = True
        mask = flat.reshape(MOTIF_SIZE, MOTIF_SIZE)
        overlap = int((mask & motif).sum())
        if overlap < 2 or overlap >= motif_count:
            continue
        if int((~mask).sum()) < 2:
            continue
        return mask


def _render_instruction(mask: np.ndarray) -> np.ndarray:
    region = np.zeros((INSTR_SIZE, INSTR_SIZE), dtype=int)
    region[INSTR_CELL_SIZE, :] = GREY
    region[INSTR_CELL_SIZE + INSTR_STRIDE, :] = GREY
    region[:, INSTR_CELL_SIZE] = GREY
    region[:, INSTR_CELL_SIZE + INSTR_STRIDE] = GREY
    for br in range(MOTIF_SIZE):
        for bc in range(MOTIF_SIZE):
            if mask[br, bc]:
                rs = br * INSTR_STRIDE
                cs = bc * INSTR_STRIDE
                region[rs : rs + INSTR_CELL_SIZE, cs : cs + INSTR_CELL_SIZE] = ORANGE
    return region


def _render_motif(mask: np.ndarray) -> np.ndarray:
    motif = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=int)
    motif[mask] = RED
    return motif


def _gen_attempt() -> np.ndarray | None:
    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    left_margin = random.randint(0, 1)
    gap = random.randint(1, 2)
    right_margin = random.randint(0, 1)
    motif_top = random.randint(0, 1)

    height = top_margin + INSTR_SIZE + bottom_margin
    width = left_margin + INSTR_SIZE + gap + MOTIF_SIZE + right_margin
    if max(height, width) > 30:
        return None

    motif_mask = _random_connected_motif()
    filter_mask = _random_filter_mask(motif_mask)
    expected = _expand_and_filter(motif_mask, filter_mask)
    if int(np.count_nonzero(expected == RED)) < 6:
        return None

    grid = np.zeros((height, width), dtype=int)
    instr_r0 = top_margin
    instr_c0 = left_margin
    motif_r0 = motif_top
    motif_c0 = left_margin + INSTR_SIZE + gap + random.randint(0, right_margin)

    grid[instr_r0 : instr_r0 + INSTR_SIZE, instr_c0 : instr_c0 + INSTR_SIZE] = _render_instruction(filter_mask)
    motif = _render_motif(motif_mask)
    existing = grid[motif_r0 : motif_r0 + MOTIF_SIZE, motif_c0 : motif_c0 + MOTIF_SIZE]
    if np.any((existing != BACKGROUND) & (motif != BACKGROUND)):
        return None
    grid[motif_r0 : motif_r0 + MOTIF_SIZE, motif_c0 : motif_c0 + MOTIF_SIZE] = motif

    try:
        observed = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
