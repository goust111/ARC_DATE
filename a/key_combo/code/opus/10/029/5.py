from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ALLOWED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300

_GEN_COUNTER = 0


def _find_block(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "ARC colors must be in 0..9"

    coords = np.argwhere(g != BLACK)
    assert coords.shape[0] == 4, "expected exactly four foreground cells"

    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    assert (r1 - r0, c1 - c0) == (1, 1), "foreground must occupy a 2x2 bbox"

    block = g[r0 : r0 + 2, c0 : c0 + 2]
    assert np.all(block != BLACK), "the 2x2 bbox must be fully filled"
    assert GREY not in set(map(int, block.ravel())), "grey is reserved for collision cells"
    assert len(set(map(int, block.ravel()))) == 4, "all four block colors must be distinct"
    return r0, c0, block.copy()


def _compute_output_from_block(block: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    a = int(block[0, 0])
    b = int(block[0, 1])
    c = int(block[1, 0])
    d = int(block[1, 1])

    rr = np.arange(h, dtype=int)[:, None]
    cc = np.arange(w, dtype=int)[None, :]

    d_a = rr + cc
    d_b = rr + (w - 1 - cc)
    d_c = (h - 1 - rr) + cc
    d_d = (h - 1 - rr) + (w - 1 - cc)

    distances = np.stack((d_a, d_b, d_c, d_d), axis=0)
    winners = np.argmin(distances, axis=0)
    min_dist = np.min(distances, axis=0)
    tie_mask = np.sum(distances == min_dist, axis=0) > 1

    palette = np.array((a, b, c, d), dtype=int)
    out = palette[winners]
    out[tie_mask] = GREY
    return out.astype(int, copy=False)


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> None:
    _, _, block = _find_block(inp)
    h, w = inp.shape
    assert out.shape == inp.shape, "output must preserve grid shape"
    assert np.all((0 <= out) & (out <= 9)), "output colors must stay in ARC palette"
    assert np.array_equal(out, _compute_output_from_block(block, (h, w))), "output violates the Voronoi rule"

    corner_colors = (
        int(block[0, 0]),
        int(block[0, 1]),
        int(block[1, 0]),
        int(block[1, 1]),
    )
    assert int(out[0, 0]) == corner_colors[0]
    assert int(out[0, w - 1]) == corner_colors[1]
    assert int(out[h - 1, 0]) == corner_colors[2]
    assert int(out[h - 1, w - 1]) == corner_colors[3]

    for r in range(h):
        for c in range(w):
            distances = (
                r + c,
                r + (w - 1 - c),
                (h - 1 - r) + c,
                (h - 1 - r) + (w - 1 - c),
            )
            best = min(distances)
            winners = [idx for idx, value in enumerate(distances) if value == best]
            cell = int(out[r, c])
            if len(winners) == 1:
                assert cell == corner_colors[winners[0]], "unique nearest corner should determine the color"
            else:
                assert cell == GREY, "simultaneous arrivals must become grey"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Map a 2x2 color block to the grid corners, then expand those corner colors inward by Manhattan distance with grey collision cells.

    Input structure:
    - The grid is black in the background and contains exactly one 2x2 foreground block.
    - The four cells of that block are fully filled, pairwise distinct, and use non-black, non-grey colors.
    - The block may appear anywhere in the grid; its position is only a compact encoding of the corner colors.

    Rule:
    - Read the 2x2 block as top-left=A, top-right=B, bottom-left=C, bottom-right=D.
    - Create a same-size output whose four corners receive A, B, C, D in the corresponding grid corners.
    - For every output cell, compare its Manhattan distances to the four corners: a unique nearest corner paints that cell with its color, while any tie between two or more nearest corners paints that cell grey.

    Key insight:
    - The input block is a compressed description of a four-source Manhattan Voronoi diagram whose sources are fixed at the outer corners.
    """
    g = np.asarray(grid, dtype=int)
    _r0, _c0, block = _find_block(g)
    return _compute_output_from_block(block, g.shape)


def _choose_shape(mode: str) -> tuple[int, int]:
    if mode == "odd_square":
        side = random.choice((9, 11, 13))
        return side, side
    if mode == "even_square":
        side = random.choice((8, 10, 12, 14))
        return side, side
    if mode == "odd_wide":
        options = (
            (9, 11),
            (9, 13),
            (11, 13),
        )
        return random.choice(options)
    if mode == "odd_tall":
        options = (
            (11, 9),
            (13, 9),
            (13, 11),
        )
        return random.choice(options)
    if mode == "row_tie":
        return random.choice(((9, 10), (11, 12), (13, 14)))
    assert mode == "col_tie"
    return random.choice(((10, 9), (12, 11), (14, 13)))


def _choose_position(h: int, w: int, edge_bias: bool) -> tuple[int, int]:
    row_choices = list(range(1, h - 2))
    col_choices = list(range(1, w - 2))
    if edge_bias:
        row_pool = [row_choices[0], row_choices[-1]]
        col_pool = [col_choices[0], col_choices[-1]]
        r0 = random.choice(row_pool if random.random() < 0.7 else row_choices)
        c0 = random.choice(col_pool if random.random() < 0.7 else col_choices)
        return int(r0), int(c0)
    return int(random.choice(row_choices)), int(random.choice(col_choices))


def _build_input(mode: str, edge_bias: bool) -> np.ndarray:
    h, w = _choose_shape(mode)
    grid = np.zeros((h, w), dtype=int)

    colors = random.sample(ALLOWED_COLORS, 4)
    if "square" in mode and random.random() < 0.5:
        colors = colors[::-1]
    elif "tall" in mode and random.random() < 0.5:
        colors = [colors[2], colors[0], colors[3], colors[1]]
    block = np.array(colors, dtype=int).reshape(2, 2)

    r0, c0 = _choose_position(h, w, edge_bias=edge_bias)
    grid[r0 : r0 + 2, c0 : c0 + 2] = block
    return grid


def _accept(inp: np.ndarray, mode: str) -> bool:
    out = main(inp)
    try:
        _semantic_check(inp, out)
    except AssertionError:
        return False

    h, w = inp.shape
    grey_count = int(np.count_nonzero(out == GREY))
    rr = np.arange(h, dtype=int)[:, None]
    cc = np.arange(w, dtype=int)[None, :]
    row_tie_mask = 2 * rr == h - 1
    col_tie_mask = 2 * cc == w - 1

    if mode == "even_square":
        if h != w or grey_count != 0:
            return False
    elif mode == "row_tie":
        if not (h % 2 == 1 and w % 2 == 0):
            return False
        if grey_count != w:
            return False
        if not np.array_equal(out == GREY, np.broadcast_to(row_tie_mask, out.shape)):
            return False
    elif mode == "col_tie":
        if not (h % 2 == 0 and w % 2 == 1):
            return False
        if grey_count != h:
            return False
        if not np.array_equal(out == GREY, np.broadcast_to(col_tie_mask, out.shape)):
            return False
    else:
        if grey_count != h + w - 1:
            return False
        cross_mask = np.broadcast_to(row_tie_mask, out.shape) | np.broadcast_to(col_tie_mask, out.shape)
        if not np.array_equal(out == GREY, cross_mask):
            return False
        if mode == "odd_square" and h != w:
            return False
        if mode == "odd_wide" and not (w > h):
            return False
        if mode == "odd_tall" and not (h > w):
            return False

    # Keep at least one non-grey interior cell so learners can see both "filled" and "collision" behavior.
    if h > 2 and w > 2 and not np.any(out[1:-1, 1:-1] != GREY):
        return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    modes = ("odd_square", "even_square", "row_tie", "col_tie", "odd_wide", "odd_tall")
    mode = modes[_GEN_COUNTER % len(modes)]
    edge_bias = (_GEN_COUNTER // len(modes)) % 2 == 1
    _GEN_COUNTER += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _build_input(mode=mode, edge_bias=edge_bias)
        if _accept(inp, mode):
            return inp
    raise RuntimeError(f"failed to generate a valid sample for mode={mode!r}")
