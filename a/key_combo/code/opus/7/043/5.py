from __future__ import annotations

import random

import numpy as np


BG = 0
PATH_COLOR = 5
SPRITE_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400

SPRITE_TEMPLATES = (
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop(grid: np.ndarray, color: int) -> np.ndarray:
    mask = grid == int(color)
    assert mask.any(), "expected at least one pixel of the requested color"
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _path_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    c_step = 1 if ec > sc else -1
    r_step = 1 if er > sr else -1
    cells = [(sr, c) for c in range(sc + c_step, ec + c_step, c_step)]
    cells.extend((r, ec) for r in range(sr + r_step, er, r_step))
    return cells


def _rotation_k(start: tuple[int, int], end: tuple[int, int]) -> int:
    sr, sc = start
    er, ec = end
    horiz = 1 if ec > sc else -1
    vert = 1 if er > sr else -1
    if horiz == 1 and vert == 1:
        return 0
    if horiz == 1 and vert == -1:
        return 3
    if horiz == -1 and vert == 1:
        return 1
    return 2


def _stamp_centered(out: np.ndarray, sprite: np.ndarray, center: tuple[int, int], marker_colors: set[int]) -> None:
    h, w = sprite.shape
    top = int(center[0]) - h // 2
    left = int(center[1]) - w // 2
    for r in range(h):
        for c in range(w):
            if int(sprite[r, c]) == BG:
                continue
            rr = top + r
            cc = left + c
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "stamp must stay in bounds"
            if int(out[rr, cc]) in marker_colors:
                continue
            out[rr, cc] = int(sprite[r, c])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw horizontal-first L paths between same-colored marker pairs and stamp the rotated reference sprite at each elbow.

    Input structure:
    - The grid has black background, one gray reference sprite in the top-left corner, and 2-3 same-colored marker pairs elsewhere.
    - The reference sprite is a small asymmetric 4-connected gray shape; every marker color appears exactly twice.
    - Within each pair the two markers are in different rows and different columns, so they define a unique horizontal-first L path.

    Rule:
    - Remove the original top-left reference sprite from the output.
    - For each same-colored pair, take the upper marker as the start and the lower marker as the end, then draw a gray path horizontally to the end column and vertically to the end row without overwriting the markers.
    - Treat the elbow `(start_row, end_col)` as an anchor, rotate the reference sprite according to the turn direction, and stamp the rotated sprite centered on that elbow while preserving marker pixels.

    Key insight:
    - The gray reference sprite is not copied at the marker locations directly; each marker pair first defines an L path, and the elbow of that path determines both where the sprite is stamped and how it is rotated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprite = _tight_crop(g, SPRITE_COLOR)
    assert int(sprite[0, 0]) == SPRITE_COLOR, "reference sprite bbox must start at the top-left corner"

    marker_colors = {int(color) for color in np.unique(g) if int(color) not in (BG, SPRITE_COLOR)}
    assert 2 <= len(marker_colors) <= 3

    out = g.copy()
    out[g == SPRITE_COLOR] = BG

    for color in sorted(marker_colors):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each marker color must appear exactly twice"
        pts = [tuple(map(int, coord)) for coord in coords]
        start, end = sorted(pts)
        assert start[0] != end[0] and start[1] != end[1], "each pair must define a non-degenerate L path"

        for r, c in _path_cells(start, end):
            assert int(out[r, c]) in (BG, PATH_COLOR), "generated paths should only traverse background or prior paths"
            if int(out[r, c]) == BG:
                out[r, c] = PATH_COLOR

        elbow = (start[0], end[1])
        rotated = np.rot90(sprite, _rotation_k(start, end))
        _stamp_centered(out, rotated, elbow, marker_colors)

    return out


def _elbow_margin_cells(elbow: tuple[int, int], radius: int = 2) -> set[tuple[int, int]]:
    er, ec = elbow
    return {
        (r, c)
        for r in range(er - radius, er + radius + 1)
        for c in range(ec - radius, ec + radius + 1)
    }


def _marker_buffer_cells(pt: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    pr, pc = pt
    return {
        (r, c)
        for r in range(pr - radius, pr + radius + 1)
        for c in range(pc - radius, pc + radius + 1)
    }


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _try_place_pair(
    shape: tuple[int, int],
    horiz_sign: int,
    forbidden: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = shape
    row_candidates = list(range(5, h - 4))
    random.shuffle(row_candidates)
    for sr in row_candidates:
        end_rows = list(range(sr + 3, h - 2))
        random.shuffle(end_rows)
        for er in end_rows:
            if horiz_sign > 0:
                start_cols = list(range(5, w - 6))
            else:
                start_cols = list(range(8, w - 2))
            random.shuffle(start_cols)
            for sc in start_cols:
                if horiz_sign > 0:
                    end_cols = list(range(sc + 3, w - 2))
                else:
                    end_cols = list(range(5, sc - 2))
                random.shuffle(end_cols)
                for ec in end_cols:
                    start = (sr, sc)
                    end = (er, ec)
                    elbow = (sr, ec)
                    path = set(_path_cells(start, end))
                    reserved = path | {start, end} | _elbow_margin_cells(elbow)
                    reserved |= _marker_buffer_cells(start)
                    reserved |= _marker_buffer_cells(end)
                    if not all(_in_bounds(shape, cell) for cell in reserved):
                        continue
                    if reserved & forbidden:
                        continue
                    if any(c < 4 for _r, c in reserved if _r < 4):
                        continue
                    return start, end, path, reserved
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 19)
    w = random.randint(15, 19)
    grid = np.zeros((h, w), dtype=int)

    sprite_mask = np.array(random.choice(SPRITE_TEMPLATES), dtype=bool)
    grid[:3, :3][sprite_mask] = SPRITE_COLOR

    pair_count = random.choice((2, 3))
    colors = random.sample(MARKER_COLORS, pair_count)

    signs = [1, -1]
    if pair_count == 3:
        signs.append(random.choice((1, -1)))
    random.shuffle(signs)

    forbidden: set[tuple[int, int]] = {
        (r, c)
        for r in range(0, 5)
        for c in range(0, 5)
        if _in_bounds(grid.shape, (r, c))
    }
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    for color, sign in zip(colors, signs):
        placed = _try_place_pair(grid.shape, sign, forbidden)
        if placed is None:
            return None
        start, end, path, reserved = placed
        grid[start] = color
        grid[end] = color
        pairs.append((color, start, end))
        forbidden |= reserved
        forbidden |= path

    if not any(end[1] > start[1] for _color, start, end in pairs):
        return None
    if not any(end[1] < start[1] for _color, start, end in pairs):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
