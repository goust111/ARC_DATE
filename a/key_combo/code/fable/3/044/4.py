from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
Coord = tuple[int, int]


def _foreground_colors_by_role(grid: np.ndarray) -> tuple[int, int]:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != BLACK)
    assert len(counts) == 2, "expected exactly two non-black foreground colors"

    ranked = counts.most_common()
    assert ranked[0][1] > ranked[1][1], "one foreground color must be a strict majority"
    majority = int(ranked[0][0])
    minority = int(ranked[1][0])
    return majority, minority


def _swap_colors(sprite: np.ndarray, a_color: int, b_color: int) -> np.ndarray:
    out = sprite.copy()
    a_mask = out == int(a_color)
    b_mask = out == int(b_color)
    out[a_mask] = int(b_color)
    out[b_mask] = int(a_color)
    return out


def _inverted_mirror_stamp(sprite: np.ndarray, a_color: int, b_color: int) -> np.ndarray:
    return _swap_colors(np.fliplr(sprite), a_color, b_color)


def _render_with_stamps(grid: np.ndarray, a_stamp: np.ndarray, b_stamp: np.ndarray, a_color: int, b_color: int) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            r0 = r * h
            c0 = c * w
            if value == int(a_color):
                out[r0 : r0 + h, c0 : c0 + w] = a_stamp
            elif value == int(b_color):
                out[r0 : r0 + h, c0 : c0 + w] = b_stamp
            else:
                assert value == BLACK, "only black, A, and B may appear"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold a two-color sprite fractally, using majority-color cells as upright stamps and minority-color cells as inverted mirrored stamps.

    Input structure:
    - The input is an h by w sprite on a black background with exactly two foreground colors A and B.
    - A is the strict foreground majority color, B is the strict foreground minority color.
    - The non-black sprite mask is not symmetric under left-right mirroring.

    Rule:
    - Build an h by w array of output blocks, so the result has shape h*h by w*w.
    - For each black input cell, leave the corresponding output block black.
    - For each A cell, stamp the original sprite into the corresponding block.
    - For each B cell, stamp the left-right mirror of the sprite after swapping A and B inside that copy.

    Key insight:
    - The sprite is both the stencil that chooses where blocks appear and the stamp drawn inside those blocks; the minority color commands a paired geometric mirror and palette inversion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h > 0 and w > 1, "expected a non-empty sprite with a horizontal axis"
    assert h * h <= 30 and w * w <= 30, "fractal expansion must stay within ARC size limits"
    assert BLACK in set(map(int, np.unique(g))), "expected black background cells"

    a_color, b_color = _foreground_colors_by_role(g)
    mask = g != BLACK
    assert np.any(g == BLACK), "expected a visible black background"
    assert not np.array_equal(mask, np.fliplr(mask)), "sprite mask must be left-right asymmetric"

    b_stamp = _inverted_mirror_stamp(g, a_color, b_color)
    return _render_with_stamps(g, g, b_stamp, a_color, b_color)


# --------------- generator ---------------


_MAX_ATTEMPTS = 600
_SIZE_OPTIONS = ((3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4), (5, 5))
_PALETTE = tuple(range(1, 10))
_DIRECTIONS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[Coord]:
    out: list[Coord] = []
    for dr, dc in _DIRECTIONS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], h, w))

    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(cell[0], cell[1], h, w):
            if nb not in cells:
                frontier.add(nb)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_candidate() -> np.ndarray | None:
    h, w = random.choice(_SIZE_OPTIONS)
    total = h * w
    min_area = max(5, total // 3)
    max_area = min(total - 1, max(min_area, (2 * total) // 3))
    if min_area > max_area:
        return None

    area = random.randint(min_area, max_area)
    mask = _random_connected_mask(h, w, area)
    if int(mask.sum()) != area:
        return None
    if np.array_equal(mask, np.fliplr(mask)):
        return None

    a_color, b_color = random.sample(_PALETTE, 2)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)

    b_min = 2 if area >= 7 else 1
    b_max = min((area - 1) // 2, max(b_min, area // 3))
    if b_max < 1:
        return None
    b_count = random.randint(b_min, b_max)

    grid = np.zeros((h, w), dtype=int)
    for r, c in coords:
        grid[r, c] = a_color
    for r, c in coords[:b_count]:
        grid[r, c] = b_color

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        h, w = grid.shape
        if h * h > 30 or w * w > 30:
            return False
        a_color, b_color = _foreground_colors_by_role(grid)
        mask = grid != BLACK
        if not np.any(grid == BLACK):
            return False
        if np.array_equal(mask, np.fliplr(mask)):
            return False

        original = grid.copy()
        mirrored_only = np.fliplr(grid)
        swapped_only = _swap_colors(grid, a_color, b_color)
        inverted = _inverted_mirror_stamp(grid, a_color, b_color)
        if np.array_equal(inverted, original):
            return False
        if np.array_equal(inverted, mirrored_only):
            return False
        if np.array_equal(inverted, swapped_only):
            return False

        expected = _render_with_stamps(grid, original, inverted, a_color, b_color)
        if expected.shape != (h * h, w * w):
            return False
        if not np.array_equal(main(grid), expected):
            return False

        wrong_mirror = _render_with_stamps(grid, original, mirrored_only, a_color, b_color)
        wrong_swap = _render_with_stamps(grid, original, swapped_only, a_color, b_color)
        wrong_plain = _render_with_stamps(grid, original, original, a_color, b_color)
        pixel_upscale = np.kron(grid, np.ones((h, w), dtype=int))
        for wrong in (wrong_mirror, wrong_swap, wrong_plain, pixel_upscale):
            if np.array_equal(wrong, expected):
                return False

        # Each semantic branch must leave a visible witness in the result.
        if not np.any(grid == a_color) or not np.any(grid == b_color):
            return False
        if set(map(int, np.unique(expected))) != {BLACK, a_color, b_color}:
            return False
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _make_candidate()
        if grid is not None and _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid inverted-fractal sprite")
