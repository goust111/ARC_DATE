import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GRID_SIZE = 30
GENERATION_RETRIES = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _bbox(g != BLACK)
    assert frame_bbox is not None, "expected a non-black framed rectangle"
    r0, c0, r1, c1 = frame_bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    assert np.all(g[r0, c0 : c1 + 1] != BLACK), "top border must be present"
    assert np.all(g[r1, c0 : c1 + 1] != BLACK), "bottom border must be present"
    assert np.all(g[r0 : r1 + 1, c0] != BLACK), "left border must be present"
    assert np.all(g[r0 : r1 + 1, c1] != BLACK), "right border must be present"
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BLACK), "input interior must be black"

    corners = [int(g[r0, c0]), int(g[r0, c1]), int(g[r1, c0]), int(g[r1, c1])]
    assert len(set(corners)) == 1 and corners[0] == GREY, "corners must be grey"

    top_seq = g[r0, c0 + 1 : c1].astype(int, copy=False)
    bottom_seq = g[r1, c0 + 1 : c1].astype(int, copy=False)
    left_seq = g[r0 + 1 : r1, c0].astype(int, copy=False)
    right_seq = g[r0 + 1 : r1, c1].astype(int, copy=False)

    assert top_seq.size > 0 and left_seq.size > 0, "frame interior must be non-empty"
    assert np.array_equal(top_seq, bottom_seq), "top and bottom stripe sequences must match"
    assert np.array_equal(left_seq, right_seq), "left and right stripe sequences must match"
    assert np.all((top_seq != BLACK) & (top_seq != GREY)), "top sequence colors must be colored"
    assert np.all((left_seq != BLACK) & (left_seq != GREY)), "left sequence colors must be colored"

    out = g.copy()
    matches = left_seq[:, None] == top_seq[None, :]
    interior = np.where(matches, left_seq[:, None], BLACK)
    out[r0 + 1 : r1, c0 + 1 : c1] = interior
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Weave the border stripe codes into an interior grid by coloring only matching row-column crossings.

    Input structure:
    - The background is black.
    - Somewhere in the grid there is a one-cell-thick rectangular frame.
    - The four corners of that frame are grey, the top and bottom edges carry the
      same left-to-right color stripe sequence, and the left and right edges carry
      the same top-to-bottom color stripe sequence.
    - The frame interior is entirely black.

    Rule:
    - Read the top edge (excluding grey corners) as the horizontal sequence H and
      the left edge (excluding grey corners) as the vertical sequence V.
    - For each interior cell at row r and column c, compare V[r] with H[c].
    - If the two border colors are equal, paint the interior cell with that shared
      color; otherwise leave it black.
    - Keep the full frame, its grey corners, and any outside background unchanged.

    Key insight:
    - The border is not decorative: its horizontal and vertical stripe labels act
      like weaving instructions, and only equal-color crossings become visible.
    """
    return _solve(grid)


def _random_sequence(length: int, colors: list[int]) -> np.ndarray:
    assert length >= 2
    max_segments = min(5, length)
    n_segments = random.randint(2, max_segments)
    cuts = sorted(random.sample(range(1, length), n_segments - 1))
    bounds = [0] + cuts + [length]

    seq = np.empty(length, dtype=int)
    last = None
    for start, end in zip(bounds, bounds[1:]):
        choices = [color for color in colors if color != last]
        color = random.choice(choices)
        seq[start:end] = color
        last = color
    return seq


def _build_input(
    inner_h: int,
    inner_w: int,
    top_margin: int,
    left_margin: int,
    bottom_margin: int,
    right_margin: int,
    top_seq: np.ndarray,
    left_seq: np.ndarray,
) -> np.ndarray:
    rows = top_margin + inner_h + 2 + bottom_margin
    cols = left_margin + inner_w + 2 + right_margin
    grid = np.zeros((rows, cols), dtype=int)

    r0 = top_margin
    c0 = left_margin
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1

    grid[r0, c0] = GREY
    grid[r0, c1] = GREY
    grid[r1, c0] = GREY
    grid[r1, c1] = GREY
    grid[r0, c0 + 1 : c1] = top_seq
    grid[r1, c0 + 1 : c1] = top_seq
    grid[r0 + 1 : r1, c0] = left_seq
    grid[r0 + 1 : r1, c1] = left_seq
    return grid


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(3, 18)
    inner_w = random.randint(3, 18)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if outer_h > MAX_GRID_SIZE or outer_w > MAX_GRID_SIZE:
        return None

    extra_h = random.randint(0, min(8, MAX_GRID_SIZE - outer_h))
    extra_w = random.randint(0, min(8, MAX_GRID_SIZE - outer_w))
    top_margin = random.randint(0, extra_h)
    bottom_margin = extra_h - top_margin
    left_margin = random.randint(0, extra_w)
    right_margin = extra_w - left_margin

    palette_size = random.randint(2, 4)
    colors = random.sample(COLOR_POOL, palette_size)
    top_seq = _random_sequence(inner_w, colors)
    left_seq = _random_sequence(inner_h, colors)

    inp = _build_input(
        inner_h,
        inner_w,
        top_margin,
        left_margin,
        bottom_margin,
        right_margin,
        top_seq,
        left_seq,
    )

    out = _solve(inp)
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    interior = out[r0 + 1 : r1, c0 + 1 : c1]

    if not np.any(interior != BLACK):
        return None
    if not np.any(interior == BLACK):
        return None
    if len(set(map(int, top_seq))) < 2 or len(set(map(int, left_seq))) < 2:
        return None
    if len(set(map(int, top_seq)) | set(map(int, left_seq))) < 2:
        return None

    expected = inp.copy()
    expected[r0 + 1 : r1, c0 + 1 : c1] = interior
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
