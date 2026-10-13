from __future__ import annotations

import random

import numpy as np


BG = 4
ORANGE = 7
MAGENTA = 6
AZURE = 8
SEED_COLORS = (0, 1, 2, 3, 5, 9)

GRID_MIN = 9
GRID_MAX = 16
FILL_MIN_GAP = 3
SPACED_RETRIES = 80
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _sample_spaced(candidates: range, count: int, min_gap: int) -> list[int] | None:
    values = list(candidates)
    if len(values) < count:
        return None
    for _ in range(SPACED_RETRIES):
        picks = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _place_seed(grid: np.ndarray, r: int, c: int, color: int) -> None:
    assert grid[r, c] == BG
    grid[r, c] = color


def _collect_seeds(
    grid: np.ndarray,
) -> tuple[dict[int, tuple[int, int, int]], dict[int, tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    horizontal: dict[int, tuple[int, int, int]] = {}
    vertical: dict[int, tuple[int, int, int]] = {}

    for r, c in np.argwhere(g != BG):
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        on_h = rr == 0 or rr == h - 1
        on_v = cc == 0 or cc == w - 1
        assert on_h or on_v, "foreground seeds must lie on the grid edge"
        assert on_h != on_v, "corner seeds are ambiguous and not supported"

        if on_h:
            dr = 1 if rr == 0 else -1
            assert color not in horizontal, "each color needs one horizontal-edge seed"
            horizontal[color] = (rr, cc, dr)
        else:
            dc = 1 if cc == 0 else -1
            assert color not in vertical, "each color needs one vertical-edge seed"
            vertical[color] = (rr, cc, dc)

    colors = set(horizontal) | set(vertical)
    assert colors, "expected at least one seed color"
    assert set(horizontal) == colors == set(vertical), (
        "every color must appear once on a horizontal edge and once on a vertical edge"
    )
    return horizontal, vertical


def _ray_claims(
    grid: np.ndarray,
    horizontal: dict[int, tuple[int, int, int]],
    vertical: dict[int, tuple[int, int, int]],
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    vertical_claims: dict[tuple[int, int], set[int]] = {}
    horizontal_claims: dict[tuple[int, int], set[int]] = {}

    for color, (r0, c0, dr) in horizontal.items():
        r = r0 + dr
        while 0 <= r < h:
            if int(g[r, c0]) == BG:
                vertical_claims.setdefault((int(r), int(c0)), set()).add(int(color))
            r += dr

    for color, (r0, c0, dc) in vertical.items():
        c = c0 + dc
        while 0 <= c < w:
            if int(g[r0, c]) == BG:
                horizontal_claims.setdefault((int(r0), int(c)), set()).add(int(color))
            c += dc

    return vertical_claims, horizontal_claims


def _paint_rays(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    horizontal, vertical = _collect_seeds(g)
    vertical_claims, horizontal_claims = _ray_claims(g, horizontal, vertical)

    for cell in set(vertical_claims) | set(horizontal_claims):
        vset = vertical_claims.get(cell, set())
        hset = horizontal_claims.get(cell, set())
        common = vset & hset

        if common:
            assert len(common) == 1, "a crossing cannot belong to multiple matched colors"
            out[cell] = MAGENTA
            continue

        colors = vset | hset
        assert colors, "claim sets should not be empty"
        if len(colors) == 1:
            out[cell] = next(iter(colors))
        else:
            # When multiple colors try to occupy the same pixel, use the mixed marker.
            out[cell] = ORANGE

    return out


def _fill_magenta_rectangle(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    magenta_pts = [(int(r), int(c)) for r, c in np.argwhere(out == MAGENTA)]
    if len(magenta_pts) != 4:
        return out

    rows = sorted({r for r, _ in magenta_pts})
    cols = sorted({c for _, c in magenta_pts})
    if len(rows) != 2 or len(cols) != 2:
        return out

    corners = {
        (rows[0], cols[0]),
        (rows[0], cols[1]),
        (rows[1], cols[0]),
        (rows[1], cols[1]),
    }
    if set(magenta_pts) != corners:
        return out

    if rows[1] - rows[0] > 1 and cols[1] - cols[0] > 1:
        out[rows[0] + 1 : rows[1], cols[0] + 1 : cols[1]] = AZURE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast inward rays from paired edge seeds, mark mixed-color intersections,
    and fill the rectangle defined by exactly four magenta self-intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "ARC colors must be in 0..9"

    bg = _background_color(g)
    assert bg == BG, "expected yellow background"
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid is too small"
    assert np.all(g[1 : h - 1, 1 : w - 1] == bg), "interior must start as pure background"

    rays = _paint_rays(g)
    return _fill_magenta_rectangle(rays)


def _gen_fill_case() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    rows = _sample_spaced(range(1, h - 1), 2, FILL_MIN_GAP)
    cols = _sample_spaced(range(1, w - 1), 2, FILL_MIN_GAP)
    if rows is None or cols is None:
        return None

    r1, r2 = rows
    c1, c2 = cols
    tl, tr, bl, br = random.sample(SEED_COLORS, 4)

    top_c1, bottom_c1 = random.sample([tl, bl], 2)
    top_c2, bottom_c2 = random.sample([tr, br], 2)
    left_r1, right_r1 = random.sample([tl, tr], 2)
    left_r2, right_r2 = random.sample([bl, br], 2)

    grid = np.full((h, w), BG, dtype=int)
    _place_seed(grid, 0, c1, top_c1)
    _place_seed(grid, h - 1, c1, bottom_c1)
    _place_seed(grid, 0, c2, top_c2)
    _place_seed(grid, h - 1, c2, bottom_c2)

    _place_seed(grid, r1, 0, left_r1)
    _place_seed(grid, r1, w - 1, right_r1)
    _place_seed(grid, r2, 0, left_r2)
    _place_seed(grid, r2, w - 1, right_r2)
    return grid


def _gen_distinct_case(k: int) -> np.ndarray | None:
    min_size = max(GRID_MIN, 2 * k + 1)
    if min_size > GRID_MAX:
        return None

    h = random.randint(min_size, GRID_MAX)
    w = random.randint(min_size, GRID_MAX)
    rows = _sample_spaced(range(1, h - 1), k, 2)
    cols = _sample_spaced(range(1, w - 1), k, 2)
    if rows is None or cols is None:
        return None

    colors = random.sample(SEED_COLORS, k)
    paired_cols = random.sample(cols, k)
    grid = np.full((h, w), BG, dtype=int)

    for color, row, col in zip(colors, rows, paired_cols):
        hr = 0 if random.random() < 0.5 else h - 1
        vc = 0 if random.random() < 0.5 else w - 1
        _place_seed(grid, hr, col, color)
        _place_seed(grid, row, vc, color)

    return grid


def _is_rectangle_corners(points: list[tuple[int, int]]) -> bool:
    if len(points) != 4:
        return False
    rows = sorted({r for r, _ in points})
    cols = sorted({c for _, c in points})
    return len(rows) == 2 and len(cols) == 2 and set(points) == {
        (rows[0], cols[0]),
        (rows[0], cols[1]),
        (rows[1], cols[0]),
        (rows[1], cols[1]),
    }


def _valid_generated(grid: np.ndarray, mode: str) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    magenta_pts = [(int(r), int(c)) for r, c in np.argwhere(out == MAGENTA)]
    orange_count = int(np.count_nonzero(out == ORANGE))
    azure_count = int(np.count_nonzero(out == AZURE))

    if mode == "fill":
        return (
            len(magenta_pts) == 4
            and _is_rectangle_corners(magenta_pts)
            and orange_count > 0
            and azure_count > 0
        )

    if mode == "four_non_rectangle":
        return (
            len(magenta_pts) == 4
            and not _is_rectangle_corners(magenta_pts)
            and orange_count > 0
            and azure_count == 0
        )

    assert mode == "count_not_four"
    return len(magenta_pts) in {3, 5} and orange_count > 0 and azure_count == 0


def _gen_attempt() -> np.ndarray | None:
    branch = random.choice(("fill", "four_non_rectangle", "count_not_four"))
    if branch == "fill":
        grid = _gen_fill_case()
    elif branch == "four_non_rectangle":
        grid = _gen_distinct_case(4)
    else:
        grid = _gen_distinct_case(random.choice((3, 5)))

    if grid is None or not _valid_generated(grid, branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
