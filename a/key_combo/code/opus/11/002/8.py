from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


BACKGROUND = 0
ANCHOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


Coord = tuple[int, int]
Pair = tuple[Coord, Coord]


def _bresenham_line(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    pts: list[Coord] = []

    while True:
        pts.append((r0, c0))
        if r0 == r1 and c0 == c1:
            return pts
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc


def _sorted_unique_coords(coords: Iterable[Coord]) -> list[Coord]:
    return sorted({(int(r), int(c)) for r, c in coords})


def _parse_input(grid: np.ndarray) -> tuple[int, int, Coord, Coord, int, int, list[Coord], list[Coord]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    mid = h // 2
    assert h % 2 == 0, "grid height must be even"

    colored = np.argwhere((g != BACKGROUND) & (g != ANCHOR))
    assert colored.shape[0] == 2, "expected exactly two clue pixels"
    clues = [(int(r), int(c), int(g[r, c])) for r, c in colored]
    upper_clues = [(r, c, col) for r, c, col in clues if r < mid]
    lower_clues = [(r, c, col) for r, c, col in clues if r >= mid]
    assert len(upper_clues) == 1 and len(lower_clues) == 1, "clues must lie in opposite halves"

    top_r, top_c, top_color = upper_clues[0]
    bottom_r, bottom_c, bottom_color = lower_clues[0]
    assert top_c != bottom_c, "clue pixels must not share a column"
    assert top_color != bottom_color, "clue colors must be different"

    upper_anchors = _sorted_unique_coords(map(tuple, np.argwhere(g[:mid, :] == ANCHOR)))
    lower_anchors = _sorted_unique_coords((r + mid, c) for r, c in np.argwhere(g[mid:, :] == ANCHOR))
    assert 2 <= len(upper_anchors) <= 4, "expected 2-4 upper anchors"
    assert len(upper_anchors) == len(lower_anchors), "anchor counts must match"
    assert len(lower_anchors) == len({c for _, c in lower_anchors}), "lower-anchor columns must be unique"
    assert len(upper_anchors) == len({c for _, c in upper_anchors}), "upper-anchor columns must be unique"

    return h, w, (top_r, top_c), (bottom_r, bottom_c), top_color, bottom_color, upper_anchors, lower_anchors


def _greedy_pairs(upper: list[Coord], lower: list[Coord]) -> list[Pair]:
    remaining = lower[:]
    pairs: list[Pair] = []
    for u in sorted(upper, key=lambda p: (p[1], p[0])):
        distances = [abs(u[1] - l[1]) for l in remaining]
        best = min(distances)
        assert distances.count(best) == 1, "pairing must have a unique nearest lower anchor"
        idx = distances.index(best)
        pairs.append((u, remaining.pop(idx)))
    return pairs


def _rank_pairs(upper: list[Coord], lower: list[Coord]) -> list[Pair]:
    return list(
        zip(
            sorted(upper, key=lambda p: (p[1], p[0])),
            sorted(lower, key=lambda p: (p[1], p[0])),
            strict=True,
        )
    )


def _line_pixel_sets(pairs: list[Pair]) -> list[set[Coord]]:
    return [set(_bresenham_line(u, l)) for u, l in pairs]


def _lines_clear_and_disjoint(
    pairs: list[Pair],
    blockers: set[Coord],
    *,
    allow_endpoints: bool,
) -> bool:
    seen: set[Coord] = set()
    endpoints = {pt for pair in pairs for pt in pair}
    for path in _line_pixel_sets(pairs):
        interior = path if allow_endpoints else path - endpoints
        if interior & blockers:
            return False
        if path & seen:
            return False
        seen |= path
    return True


def _paint_border(out: np.ndarray, color: int) -> None:
    out[0, :] = color
    out[-1, :] = color
    out[:, 0] = color
    out[:, -1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair gray anchors across the split and recolor their lines and the frame by half.

    Input structure:
    - The grid has a black background and even height.
    - There is exactly one non-gray clue pixel in the upper half and one different-colored clue pixel in the lower half.
    - There are 2-4 gray anchor pixels in the upper half and the same number in the lower half, with unique columns within each half.

    Rule:
    - Use the upper clue color as the top-half color and the lower clue color as the bottom-half color.
    - Sort upper anchors by column; for each one, greedily pair it with the unused lower anchor having the unique minimum column distance.
    - Draw a gray Bresenham line for every pair, then draw a gray 1-pixel outer border.
    - Recolor every gray pixel by position: rows above the half split become the top color and rows at or below the split become the bottom color.
    - Finally draw full horizontal lines across the two clue rows in their respective clue colors, overwriting anything already there.

    Key insight:
    - The gray anchors are not colored objects to preserve; they define hidden cross-half pairings, and the half split then recolors the resulting geometry by region.
    """
    h, w, top_clue, bottom_clue, top_color, bottom_color, upper, lower = _parse_input(grid)
    mid = h // 2

    pairs = _greedy_pairs(upper, lower)
    out = np.full((h, w), BACKGROUND, dtype=int)

    for u, l in pairs:
        for r, c in _bresenham_line(u, l):
            out[r, c] = ANCHOR

    _paint_border(out, ANCHOR)

    upper_mask = out[:mid, :] == ANCHOR
    lower_mask = out[mid:, :] == ANCHOR
    out[:mid, :][upper_mask] = top_color
    out[mid:, :][lower_mask] = bottom_color

    out[top_clue[0], :] = top_color
    out[bottom_clue[0], :] = bottom_color
    return out


def _sample_spaced_columns(rng: np.random.Generator, width: int, count: int) -> list[int]:
    pool = list(range(1, width - 1))
    for _ in range(80):
        cols = sorted(map(int, rng.choice(pool, size=count, replace=False)))
        if count < 2 or int(np.min(np.diff(cols))) >= 2:
            return cols
    return sorted(map(int, rng.choice(pool, size=count, replace=False)))


def _sample_rows(rng: np.random.Generator, start: int, stop: int, count: int, banned: set[int]) -> list[int]:
    choices = [r for r in range(start, stop) if r not in banned]
    assert len(choices) >= count
    return list(map(int, rng.choice(choices, size=count, replace=False)))


def _make_candidate_anchors(
    rng: np.random.Generator,
    h: int,
    w: int,
    n_pairs: int,
    top_row: int,
    bottom_row: int,
    prefer_alternative: bool,
) -> tuple[list[Coord], list[Coord]] | None:
    mid = h // 2
    for _ in range(200):
        upper_cols = _sample_spaced_columns(rng, w, n_pairs)
        lower_base = upper_cols[:]
        rng.shuffle(lower_base)
        lower_cols = [int(np.clip(c + int(rng.integers(-3, 4)), 1, w - 2)) for c in lower_base]
        if len(set(lower_cols)) < n_pairs:
            continue

        upper_rows = _sample_rows(rng, 1, mid, n_pairs, {top_row})
        lower_rows = _sample_rows(rng, mid, h - 1, n_pairs, {bottom_row})
        upper = sorted(zip(upper_rows, upper_cols), key=lambda p: (p[1], p[0]))
        lower = list(zip(lower_rows, lower_cols))

        try:
            greedy = _greedy_pairs(upper, lower)
        except AssertionError:
            continue

        if any(len(_bresenham_line(u, l)) < 3 for u, l in greedy):
            continue

        clue_blockers: set[Coord] = set()
        if not _lines_clear_and_disjoint(greedy, clue_blockers, allow_endpoints=True):
            continue

        alt = _rank_pairs(upper, lower)
        alt_is_different = {(u, l) for u, l in greedy} != {(u, l) for u, l in alt}
        alt_is_valid = alt_is_different and _lines_clear_and_disjoint(alt, clue_blockers, allow_endpoints=True)
        if prefer_alternative and not alt_is_valid:
            continue
        return upper, lower
    return None


def _build_input(
    h: int,
    w: int,
    top_clue: Coord,
    bottom_clue: Coord,
    top_color: int,
    bottom_color: int,
    upper: list[Coord],
    lower: list[Coord],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[top_clue] = top_color
    grid[bottom_clue] = bottom_color
    for r, c in upper + lower:
        assert grid[r, c] == BACKGROUND
        grid[r, c] = ANCHOR
    return grid


def _candidate_is_semantically_valid(grid: np.ndarray, *, require_alternative: bool) -> bool:
    try:
        h, w, top_clue, bottom_clue, top_color, bottom_color, upper, lower = _parse_input(grid)
        pairs = _greedy_pairs(upper, lower)
    except AssertionError:
        return False

    clue_cells = {top_clue, bottom_clue}
    line_sets = _line_pixel_sets(pairs)
    seen: set[Coord] = set()
    all_endpoints = {pt for pair in pairs for pt in pair}
    for pair, pts in zip(pairs, line_sets, strict=True):
        u, l = pair
        if pts & clue_cells:
            return False
        if (pts - {u, l}) & (all_endpoints - {u, l}):
            return False
        if (pts - {u, l}) & seen:
            return False
        seen |= pts

    rank = _rank_pairs(upper, lower)
    rank_differs = {(u, l) for u, l in rank} != {(u, l) for u, l in pairs}
    rank_valid = rank_differs and not any(
        (set(_bresenham_line(a, b)) & clue_cells) for a, b in rank
    ) and _lines_clear_and_disjoint(rank, set(), allow_endpoints=True)
    if require_alternative and not rank_valid:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False

    top_row = top_clue[0]
    bottom_row = bottom_clue[0]
    if not np.all(out[top_row, :] == top_color):
        return False
    if not np.all(out[bottom_row, :] == bottom_color):
        return False

    mid = h // 2
    skeleton = np.zeros_like(grid, dtype=bool)
    for pts in line_sets:
        for r, c in pts:
            skeleton[r, c] = True
    skeleton[0, :] = True
    skeleton[-1, :] = True
    skeleton[:, 0] = True
    skeleton[:, -1] = True

    for r, c in np.argwhere(skeleton):
        r = int(r)
        c = int(c)
        if r == top_row:
            expected = top_color
        elif r == bottom_row:
            expected = bottom_color
        elif r < mid:
            expected = top_color
        else:
            expected = bottom_color
        if int(out[r, c]) != expected:
            return False

    if np.array_equal(out, grid):
        return False
    if len({hashlib.md5(out.tobytes()).hexdigest(), hashlib.md5(grid.tobytes()).hexdigest()}) != 2:
        return False
    return True


def _generate_once(rng: np.random.Generator, require_alternative: bool) -> np.ndarray | None:
    h = int(rng.choice(np.arange(12, 21, 2)))
    w = int(rng.integers(12, 21))
    mid = h // 2
    n_pairs = int(rng.integers(2, 5))

    top_color, bottom_color = map(int, rng.choice(PALETTE, size=2, replace=False))
    top_row = int(rng.integers(1, mid - 1))
    bottom_row = int(rng.integers(mid + 1, h - 1))
    top_col = int(rng.integers(1, w - 1))
    bottom_col_choices = [c for c in range(1, w - 1) if c != top_col]
    bottom_col = int(rng.choice(bottom_col_choices))

    anchors = _make_candidate_anchors(rng, h, w, n_pairs, top_row, bottom_row, require_alternative)
    if anchors is None:
        return None
    upper, lower = anchors

    top_clue = (top_row, top_col)
    bottom_clue = (bottom_row, bottom_col)
    grid = _build_input(h, w, top_clue, bottom_clue, top_color, bottom_color, upper, lower)
    if not _candidate_is_semantically_valid(grid, require_alternative=require_alternative):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        require_alternative = attempt % 2 == 0
        grid = _generate_once(rng, require_alternative=require_alternative)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
