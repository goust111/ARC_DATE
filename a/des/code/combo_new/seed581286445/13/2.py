from __future__ import annotations

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_dividers(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        (c, int(grid[0, c]))
        for c in range(w)
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]

    matches = [
        (r, c, row_color)
        for r, row_color in row_candidates
        for c, col_color in col_candidates
        if row_color == col_color
    ]
    assert len(matches) == 1, "expected one horizontal/vertical divider pair of the same color"

    cross_r, cross_c, divider_color = matches[0]
    assert 0 < cross_r < h - 1 and 0 < cross_c < w - 1, "divider cross must split four non-empty quadrants"
    return int(cross_r), int(cross_c), int(divider_color)


def _quadrant_slices(cross_r: int, cross_c: int, h: int, w: int) -> dict[str, tuple[slice, slice]]:
    return {
        "tl": (slice(0, cross_r), slice(0, cross_c)),
        "tr": (slice(0, cross_r), slice(cross_c + 1, w)),
        "bl": (slice(cross_r + 1, h), slice(0, cross_c)),
        "br": (slice(cross_r + 1, h), slice(cross_c + 1, w)),
    }


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, int, dict[str, tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _most_common_color(g)
    cross_r, cross_c, divider_color = _find_dividers(g, bg)

    seeds: dict[str, tuple[int, int, int]] = {}
    for quadrant, (row_slice, col_slice) in _quadrant_slices(cross_r, cross_c, h, w).items():
        view = g[row_slice, col_slice]
        coords = np.argwhere(view != bg)
        assert coords.shape[0] == 1, "each quadrant must contain exactly one seed pixel"
        local_r, local_c = map(int, coords[0])
        seeds[quadrant] = (
            int(row_slice.start + local_r),
            int(col_slice.start + local_c),
            int(view[local_r, local_c]),
        )

    seed_colors = [color for _, _, color in seeds.values()]
    assert len(set(seed_colors)) == 4, "the four seeds must use distinct colors"
    return g, bg, cross_r, cross_c, divider_color, seeds


def _target_slices(quadrant: str, cross_r: int, cross_c: int, dh: int, dv: int) -> tuple[slice, slice]:
    row_slice = slice(cross_r + 1, cross_r + 1 + dh) if quadrant in ("tl", "tr") else slice(cross_r - dh, cross_r)
    col_slice = slice(cross_c + 1, cross_c + 1 + dv) if quadrant in ("tl", "bl") else slice(cross_c - dv, cross_c)
    return row_slice, col_slice


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each quadrant seed's distances to the divider cross to draw a same-sized rectangle in the diagonal quadrant.

    Input structure:
    - The background is the most common color.
    - One full horizontal divider line and one full vertical divider line share the same color and split the grid into four quadrants.
    - Each quadrant contains exactly one single-pixel seed color, and the four seed colors are all different.
    - Aside from the divider lines and those four seeds, the grid is background only.

    Rule:
    - Find the divider cross and read each seed's row distance dh to the horizontal divider and column distance dv to the vertical divider.
    - For each seed, paint a solid dh-by-dv rectangle of that seed color in the diagonal quadrant, starting from the inner corner next to the cross and extending outward toward that quadrant's outer corner.
    - Process seeds in quadrant order top-left, top-right, bottom-left, bottom-right.
    - Preserve the original divider lines and original seed pixels.

    Key insight:
    - The seed color tells you which color to paint, while the seed's two distances to the cross encode the rectangle's height and width for the opposite quadrant.
    """
    g, bg, cross_r, cross_c, divider_color, seeds = _parse_input(grid)
    del bg

    out = g.copy()
    for quadrant in ("tl", "tr", "bl", "br"):
        seed_r, seed_c, color = seeds[quadrant]
        dh = abs(seed_r - cross_r)
        dv = abs(seed_c - cross_c)
        assert dh > 0 and dv > 0, "seed pixels must lie strictly inside their quadrants"

        row_slice, col_slice = _target_slices(quadrant, cross_r, cross_c, dh, dv)
        assert 0 <= row_slice.start < row_slice.stop <= g.shape[0]
        assert 0 <= col_slice.start < col_slice.stop <= g.shape[1]
        out[row_slice, col_slice] = color

    out[cross_r, :] = divider_color
    out[:, cross_c] = divider_color
    for seed_r, seed_c, color in seeds.values():
        out[seed_r, seed_c] = color
    return out


_RNG = np.random.default_rng()
_GEN_RETRIES = 400
_SIDE_MIN = 3
_SIDE_MAX = 10
_PAIR_MODES = ("none", "first_covers", "second_covers", "equal")


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[int(_RNG.integers(0, len(seq)))]


def _sample_cover_pair(max_dh: int, max_dv: int, mode: str) -> tuple[int, int, int, int]:
    assert max_dh >= 1 and max_dv >= 1

    if mode == "equal":
        dh = _randint(1, max_dh)
        dv = _randint(1, max_dv)
        return dh, dv, dh, dv

    if mode == "first_covers":
        first_dh = _randint(1, max_dh)
        first_dv = _randint(1, max_dv)
        if first_dh == 1 and first_dv == 1:
            if bool(_RNG.integers(0, 2)):
                first_dh = 2
            else:
                first_dv = 2

        second_dh = _randint(1, first_dh)
        second_dv = _randint(1, first_dv)
        if second_dh == first_dh and second_dv == first_dv:
            if first_dh > 1 and (first_dv == 1 or bool(_RNG.integers(0, 2))):
                second_dh = _randint(1, first_dh - 1)
            else:
                second_dv = _randint(1, first_dv - 1)
        return first_dh, first_dv, second_dh, second_dv

    if mode == "second_covers":
        second_dh, second_dv, first_dh, first_dv = _sample_cover_pair(max_dh, max_dv, "first_covers")
        return first_dh, first_dv, second_dh, second_dv

    assert mode == "none"
    assert max_dh >= 2 and max_dv >= 2
    if bool(_RNG.integers(0, 2)):
        first_dh = _randint(1, max_dh - 1)
        second_dh = _randint(first_dh + 1, max_dh)
        first_dv = _randint(2, max_dv)
        second_dv = _randint(1, first_dv - 1)
    else:
        first_dh = _randint(2, max_dh)
        second_dh = _randint(1, first_dh - 1)
        first_dv = _randint(1, max_dv - 1)
        second_dv = _randint(first_dv + 1, max_dv)
    return first_dh, first_dv, second_dh, second_dv


def _gen_attempt() -> np.ndarray:
    top = _randint(_SIDE_MIN, _SIDE_MAX)
    bottom = _randint(_SIDE_MIN, _SIDE_MAX)
    left = _randint(_SIDE_MIN, _SIDE_MAX)
    right = _randint(_SIDE_MIN, _SIDE_MAX)

    cross_r = top
    cross_c = left
    h = top + 1 + bottom
    w = left + 1 + right

    max_dh = min(top, bottom)
    max_dv = min(left, right)

    colors = list(map(int, _RNG.choice(10, size=6, replace=False)))
    bg, divider_color, tl_color, tr_color, bl_color, br_color = colors

    tl_dh, tl_dv, br_dh, br_dv = _sample_cover_pair(max_dh, max_dv, _choice(_PAIR_MODES))
    tr_dh, tr_dv, bl_dh, bl_dv = _sample_cover_pair(max_dh, max_dv, _choice(_PAIR_MODES))

    grid = np.full((h, w), bg, dtype=int)
    grid[cross_r, :] = divider_color
    grid[:, cross_c] = divider_color

    seeds = {
        "tl": (cross_r - tl_dh, cross_c - tl_dv, tl_color),
        "tr": (cross_r - tr_dh, cross_c + tr_dv, tr_color),
        "bl": (cross_r + bl_dh, cross_c - bl_dv, bl_color),
        "br": (cross_r + br_dh, cross_c + br_dv, br_color),
    }
    for seed_r, seed_c, color in seeds.values():
        grid[seed_r, seed_c] = color

    try:
        parsed_grid, _, parsed_r, parsed_c, _, parsed_seeds = _parse_input(grid)
        assert parsed_grid.shape == (h, w)
        assert parsed_r == cross_r and parsed_c == cross_c
        for quadrant, (seed_r, seed_c, _) in seeds.items():
            dh = abs(seed_r - cross_r)
            dv = abs(seed_c - cross_c)
            row_slice, col_slice = _target_slices(quadrant, cross_r, cross_c, dh, dv)
            assert row_slice.start >= 0 and row_slice.stop <= h
            assert col_slice.start >= 0 and col_slice.stop <= w
        assert parsed_seeds == seeds
    except AssertionError:
        return _gen_attempt()

    areas = [
        tl_dh * tl_dv,
        tr_dh * tr_dv,
        bl_dh * bl_dv,
        br_dh * br_dv,
    ]
    if max(areas) == 1:
        return _gen_attempt()

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid input")
