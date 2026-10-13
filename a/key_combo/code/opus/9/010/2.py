from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
RING_COLORS = (1, 2, 3, 4, 6)

CORNER_POS = ((0, 0), (0, 2), (2, 2), (2, 0))
SIDE_POS = ((0, 1), (1, 2), (2, 1), (1, 0))

INSTR_SIZE = 6
INSTR_CELL = 2
ARRAY_SIZE = 11
RING_SIZE = 3
RING_STEP = 4
GEN_RETRIES = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Small helper kept in the `(r, c)` / inclusive-bbox style used by arc_utils."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _rotate_cycle(values: list[int]) -> list[int]:
    return [values[-1]] + values[:-1]


def _apply_ring_mode(patch: np.ndarray, mode: int) -> np.ndarray:
    out = patch.copy()
    if mode in (1, 2):
        corners = [int(patch[r, c]) for r, c in CORNER_POS]
        for (r, c), value in zip(CORNER_POS, _rotate_cycle(corners)):
            out[r, c] = value
    if mode in (0, 2):
        sides = [int(patch[r, c]) for r, c in SIDE_POS]
        for (r, c), value in zip(SIDE_POS, _rotate_cycle(sides)):
            out[r, c] = value
    return out


def _decode_mask(g: np.ndarray, sep_row: int, instr_col: int) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=int)
    instr_row = sep_row - INSTR_SIZE
    assert instr_row >= 0
    assert instr_col >= 0 and instr_col + INSTR_SIZE <= g.shape[1]
    for mr in range(3):
        for mc in range(3):
            r0 = instr_row + mr * INSTR_CELL
            c0 = instr_col + mc * INSTR_CELL
            block = g[r0 : r0 + INSTR_CELL, c0 : c0 + INSTR_CELL]
            assert block.shape == (2, 2)
            if np.all(block == ORANGE):
                mask[mr, mc] = 1
            else:
                assert np.all(block == BG)
    assert np.count_nonzero(mask == 1) >= 2
    assert np.count_nonzero(mask == 0) >= 2
    return mask


def _ring_layout(g: np.ndarray) -> tuple[int, int, dict[tuple[int, int], tuple[int, int]]]:
    grey_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    assert len(grey_rows) == 1
    sep_row = int(grey_rows[0])
    assert sep_row >= INSTR_SIZE
    assert g.shape[0] - sep_row - 1 >= ARRAY_SIZE

    lower = g[sep_row + 1 :, :]
    fg_mask = (lower != BG) & (lower != GREY)
    comps = _components(fg_mask)
    assert len(comps) == 9

    entries = []
    for coords in comps:
        global_coords = coords + np.array([sep_row + 1, 0], dtype=int)
        r0, c0, r1, c1 = _bbox_coords(global_coords)
        assert r1 == r0 + 2 and c1 == c0 + 2
        assert len(coords) == 8
        assert g[r0 + 1, c0 + 1] == BG
        for dr in range(3):
            for dc in range(3):
                rr, cc = r0 + dr, c0 + dc
                if (dr, dc) == (1, 1):
                    assert g[rr, cc] == BG
                else:
                    assert g[rr, cc] in RING_COLORS
        entries.append((r0, c0))

    rows = sorted({r for r, _c in entries})
    cols = sorted({c for _r, c in entries})
    assert len(rows) == 3 and len(cols) == 3
    assert rows[1] - rows[0] == RING_STEP and rows[2] - rows[1] == RING_STEP
    assert cols[1] - cols[0] == RING_STEP and cols[2] - cols[1] == RING_STEP

    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    expected_positions = set()
    for gr, r0 in enumerate(rows):
        for gc, c0 in enumerate(cols):
            mapping[(gr, gc)] = (r0, c0)
            expected_positions.add((r0, c0))
    assert set(entries) == expected_positions
    return sep_row, cols[0], mapping


def _transform(grid: np.ndarray, selector: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_row, instr_col, mapping = _ring_layout(g)
    mask = _decode_mask(g, sep_row, instr_col)

    out = g.copy()
    for gr in range(3):
        for gc in range(3):
            r0, c0 = mapping[(gr, gc)]
            patch = g[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE]
            if selector == "main":
                mode = 1 if mask[gr, gc] == 1 else 0
            elif selector == "reversed":
                mode = 0 if mask[gr, gc] == 1 else 1
            elif selector == "both":
                mode = 2
            elif selector == "corners":
                mode = 1
            elif selector == "sides":
                mode = 0
            else:
                raise ValueError(selector)
            out[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE] = _apply_ring_mode(patch, mode)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a 3x3 block mask and use it to choose which sub-cycle each ring rotates.

    Input structure:
    - The background is black, and one full grey row separates an upper instruction zone from a lower ring-array zone.
    - In the six rows immediately above the separator, a 3x3 lattice of 2x2 blocks encodes a binary mask: orange means 1 and black means 0.
    - Below the separator are nine 3x3 foreground rings arranged as a 3x3 array with one-cell black gaps; every ring has a black center and colored boundary pixels.

    Rule:
    - Decode the 3x3 instruction mask from the 2x2 orange blocks.
    - For each ring at the matching lattice position, apply exactly one clockwise four-cycle: mask value 1 rotates only the four corner pixels, while mask value 0 rotates only the four side-center pixels.
    - Keep the separator row, the instruction area, the ring centers, and all black background pixels unchanged.

    Key insight:
    - The instruction mask does not select which rings move; it selects which of the two independent four-pixel cycles is activated inside each ring.
    """
    return _transform(grid, "main")


def _random_mask() -> np.ndarray:
    while True:
        ones = random.randint(2, 7)
        bits = [1] * ones + [0] * (9 - ones)
        random.shuffle(bits)
        mask = np.asarray(bits, dtype=int).reshape(3, 3)
        if np.count_nonzero(mask == 1) >= 2 and np.count_nonzero(mask == 0) >= 2:
            return mask


def _random_ring() -> np.ndarray:
    while True:
        unique_count = random.randint(2, 4)
        chosen = random.sample(RING_COLORS, unique_count)
        values = chosen + [random.choice(chosen) for _ in range(8 - unique_count)]
        random.shuffle(values)
        corners = values[:4]
        sides = values[4:]
        if len(set(corners)) < 2 or len(set(sides)) < 2:
            continue
        patch = np.zeros((3, 3), dtype=int)
        for (r, c), value in zip(CORNER_POS, corners):
            patch[r, c] = value
        for (r, c), value in zip(SIDE_POS, sides):
            patch[r, c] = value
        return patch


def _render_input(mask: np.ndarray, rings: list[list[np.ndarray]], top_pad: int, left_pad: int, right_pad: int, bottom_pad: int) -> np.ndarray:
    height = top_pad + INSTR_SIZE + 1 + ARRAY_SIZE + bottom_pad
    width = left_pad + ARRAY_SIZE + right_pad
    g = np.zeros((height, width), dtype=int)

    instr_r0 = top_pad
    instr_c0 = left_pad
    for mr in range(3):
        for mc in range(3):
            if mask[mr, mc] == 1:
                r0 = instr_r0 + mr * INSTR_CELL
                c0 = instr_c0 + mc * INSTR_CELL
                g[r0 : r0 + INSTR_CELL, c0 : c0 + INSTR_CELL] = ORANGE

    sep_row = top_pad + INSTR_SIZE
    g[sep_row, :] = GREY

    lower_r0 = sep_row + 1
    for gr in range(3):
        for gc in range(3):
            r0 = lower_r0 + gr * RING_STEP
            c0 = left_pad + gc * RING_STEP
            g[r0 : r0 + RING_SIZE, c0 : c0 + RING_SIZE] = rings[gr][gc]
    return g


def _semantic_ok(inp: np.ndarray) -> bool:
    expected = _transform(inp, "main")
    if np.array_equal(expected, inp):
        return False

    alternatives = (
        _transform(inp, "reversed"),
        _transform(inp, "both"),
        _transform(inp, "corners"),
        _transform(inp, "sides"),
    )
    if any(np.array_equal(expected, alt) for alt in alternatives):
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    mask = _random_mask()
    rings = [[_random_ring() for _gc in range(3)] for _gr in range(3)]

    top_pad = random.randint(0, 3)
    left_pad = random.randint(0, 4)
    right_pad = random.randint(0, 6)
    bottom_pad = random.randint(0, 4)

    inp = _render_input(mask, rings, top_pad, left_pad, right_pad, bottom_pad)
    if max(inp.shape) > 30:
        return None
    if not _semantic_ok(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
