from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4

OPS = (RED, BLUE, YELLOW)
MOTIF_PALETTE = (3, 5, 6, 7, 8, 9)

SEQ_LEN1 = (
    (RED,),
    (BLUE,),
    (YELLOW,),
)
SEQ_LEN2 = (
    (RED, BLUE),
    (BLUE, RED),
    (RED, YELLOW),
    (YELLOW, RED),
)
SEQ_LEN3 = (
    (RED, RED, BLUE),
    (RED, BLUE, RED),
    (BLUE, RED, RED),
    (RED, RED, YELLOW),
    (RED, YELLOW, RED),
    (YELLOW, RED, RED),
    (RED, BLUE, YELLOW),
    (RED, YELLOW, BLUE),
    (BLUE, RED, YELLOW),
    (YELLOW, RED, BLUE),
    (BLUE, YELLOW, RED),
    (YELLOW, BLUE, RED),
)

MAX_GENERATION_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _apply_op(grid: np.ndarray, op: int) -> np.ndarray:
    if op == RED:
        return np.rot90(grid, -1)
    if op == BLUE:
        return np.fliplr(grid)
    if op == YELLOW:
        return np.flipud(grid)
    raise ValueError(f"unknown instruction color: {op}")


def _apply_sequence(grid: np.ndarray, seq: tuple[int, ...] | list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for op in seq:
        out = _apply_op(out, int(op))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the colored instruction strip from left to right on the square motif region.

    Input structure:
    - The top row contains a contiguous instruction strip of one to three colored cells on
      a black background.
    - The second row is an all-black separator.
    - All remaining rows form a square main region containing a multicolor motif.

    Rule:
    - Decode instruction colors with a fixed meaning: red = rotate the whole main region
      90 degrees clockwise, blue = mirror it left-right, yellow = mirror it top-bottom.
    - Read the strip from left to right and apply every instruction in that exact order.
    - Return only the transformed square main region, dropping the strip and separator.

    Key insight:
    - The colored cells are not part of the motif: they are a tiny program, and only the
      written order of those commands resolves the non-commutative rotate-vs-mirror cases.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3, "expected instruction row, separator, and main region"
    assert np.all(g[1] == BG), "expected a black separator row"

    main_region = g[2:, :]
    assert main_region.shape[0] == main_region.shape[1], "main region must be square"

    cols = np.flatnonzero(g[0] != BG)
    assert 1 <= len(cols) <= 3, "instruction strip must contain one to three cells"
    assert np.all(np.diff(cols) == 1), "instruction strip must be contiguous"

    seq = [int(g[0, c]) for c in cols]
    assert set(seq) <= set(OPS), "unexpected instruction color"
    return _apply_sequence(main_region, seq)


def _grid_sig(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=int)
    return tuple(arr.shape), arr.tobytes()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _grow_connected(h: int, w: int, target_cells: int) -> set[tuple[int, int]]:
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*seed)
    while len(cells) < target_cells and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)
    return cells


def _random_bbox_spanning_component(h: int, w: int, target_cells: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    r = 0
    c = 0
    steps = [(1, 0)] * (h - 1) + [(0, 1)] * (w - 1)
    random.shuffle(steps)
    for dr, dc in steps:
        r += dr
        c += dc
        cells.add((r, c))

    frontier = set()

    def add_frontier(rr: int, cc: int) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    for rr, cc in tuple(cells):
        add_frontier(rr, cc)

    while len(cells) < target_cells and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)

    return cells


def _color_patch(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray | None:
    coords = list(cells)
    if len(coords) < 6:
        return None

    n_colors = random.randint(3, min(5, len(coords)))
    colors = random.sample(MOTIF_PALETTE, n_colors)
    random.shuffle(coords)

    patch = np.full((h, w), BG, dtype=int)
    for color, (r, c) in zip(colors, coords, strict=False):
        patch[r, c] = color

    rest = coords[n_colors:]
    for r, c in rest:
        patch[r, c] = random.choice(colors)

    present = {int(v) for v in patch.flatten() if int(v) != BG}
    if len(present) != n_colors:
        return None
    return patch


def _basic_transform_distinct(grid: np.ndarray) -> bool:
    variants = [
        (),
        (RED,),
        (RED, RED),
        (RED, RED, RED),
        (BLUE,),
        (YELLOW,),
        (RED, BLUE),
        (RED, YELLOW),
    ]
    sigs = {_grid_sig(_apply_sequence(grid, seq)) for seq in variants}
    return len(sigs) == len(variants)


def _sequence_semantics_ok(main_region: np.ndarray, seq: tuple[int, ...]) -> bool:
    states = [main_region]
    cur = main_region
    for op in seq:
        nxt = _apply_op(cur, op)
        if np.array_equal(nxt, cur):
            return False
        states.append(nxt)
        cur = nxt

    final = states[-1]
    for prev in states[:-1]:
        if np.array_equal(final, prev):
            return False

    for i in range(len(seq)):
        shorter = seq[:i] + seq[i + 1 :]
        alt = _apply_sequence(main_region, shorter)
        if np.array_equal(final, alt):
            return False

    if len(seq) > 1:
        seen_alt = False
        for perm in set(itertools.permutations(seq)):
            if perm == seq:
                continue
            seen_alt = True
            alt = _apply_sequence(main_region, perm)
            if np.array_equal(final, alt):
                return False
        if not seen_alt:
            return False

    return True


def _make_patch(ph: int, pw: int) -> np.ndarray | None:
    area = ph * pw
    low = max(6, ph + pw - 1)
    high = min(area - 1, max(low, int(round(area * 0.72))))
    if high < low:
        return None

    for _ in range(120):
        target_cells = random.randint(low, high)
        cells = _random_bbox_spanning_component(ph, pw, target_cells)
        coords = np.array(sorted(cells), dtype=int)
        if coords.shape[0] != target_cells:
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r0 != 0 or c0 != 0 or r1 != ph - 1 or c1 != pw - 1:
            continue

        patch = _color_patch(cells, ph, pw)
        if patch is None:
            continue
        if not _basic_transform_distinct(patch):
            continue
        return patch
    return None


def _sample_sequence() -> tuple[int, ...]:
    length = random.choices((1, 2, 3), weights=(2, 5, 4), k=1)[0]
    if length == 1:
        return random.choice(SEQ_LEN1)
    if length == 2:
        return random.choice(SEQ_LEN2)
    return random.choice(SEQ_LEN3)


def _build_candidate() -> np.ndarray | None:
    side = random.randint(6, 10)
    seq = _sample_sequence()

    ph = random.randint(3, side - 1)
    pw = random.randint(3, side - 1)
    patch = _make_patch(ph, pw)
    if patch is None:
        return None

    placements = [
        (r0, c0)
        for r0 in range(side - ph + 1)
        for c0 in range(side - pw + 1)
        if r0 > 0 or c0 > 0 or r0 + ph < side or c0 + pw < side
    ]
    random.shuffle(placements)

    for r0, c0 in placements:
        main_region = np.full((side, side), BG, dtype=int)
        _paste(main_region, patch, r0, c0)

        motif_coords = np.argwhere(main_region != BG)
        mr0, mc0, mr1, mc1 = _bbox_coords(motif_coords)
        if mr1 - mr0 < 2 or mc1 - mc0 < 2:
            continue
        if not _basic_transform_distinct(main_region):
            continue
        if not _sequence_semantics_ok(main_region, seq):
            continue

        grid = np.full((side + 2, side), BG, dtype=int)
        start = random.randint(0, side - len(seq))
        grid[0, start : start + len(seq)] = np.array(seq, dtype=int)
        grid[2:, :] = main_region
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue

        out = main(grid)
        if out.shape[0] != out.shape[1]:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(out, grid[2:, :]):
            continue
        return grid

    raise RuntimeError("failed to generate a valid instruction-strip sample")
