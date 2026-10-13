from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _dominant_color(values: np.ndarray, excluded: set[int]) -> int:
    counts = Counter(int(v) for v in values.ravel() if int(v) not in excluded)
    assert counts, "expected at least one non-excluded color"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, f"expected a unique dominant color, got {winners}"
    return int(winners[0])


def _find_separator(grid: np.ndarray) -> tuple[str, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int, int]] = []

    for r in range(1, h - 1):
        color = int(grid[r, 0])
        if np.all(grid[r, :] == color):
            before = int(np.sum(grid[:r, :] == color))
            after = int(np.sum(grid[r + 1 :, :] == color))
            if (before > 0) ^ (after > 0):
                ref_sign = -1 if before > 0 else 1
                candidates.append(("h", int(r), color, ref_sign))

    for c in range(1, w - 1):
        color = int(grid[0, c])
        if np.all(grid[:, c] == color):
            before = int(np.sum(grid[:, :c] == color))
            after = int(np.sum(grid[:, c + 1 :] == color))
            if (before > 0) ^ (after > 0):
                ref_sign = -1 if before > 0 else 1
                candidates.append(("v", int(c), color, ref_sign))

    assert len(candidates) == 1, f"expected one framed separator, got {candidates}"
    return candidates[0]


def _side_indices(limit: int, sep: int, sign: int) -> list[int]:
    assert sign in (-1, 1)
    indices = list(range(0, sep) if sign < 0 else range(sep + 1, limit))
    assert indices, "separator must leave cells on each side"
    return indices


def _apply_moves(
    out: np.ndarray,
    moves: list[tuple[int, int, int, int, int]],
    source_bg: int,
) -> None:
    source_cells = {(sr, sc) for sr, sc, _, _, _ in moves}
    dest_cells = {(dr, dc) for _, _, dr, dc, _ in moves}
    assert len(dest_cells) == len(moves), "destinations must be distinct"

    for sr, sc, _, _, _ in moves:
        out[sr, sc] = source_bg

    for sr, sc, dr, dc, color in moves:
        if (dr, dc) != (sr, sc):
            assert (dr, dc) not in source_cells
            assert int(out[dr, dc]) == source_bg
        out[dr, dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare each source pixel to its aligned reference pixel.

    Matching colors move to the cell adjacent to the separator on the source
    side. Non-matching colors move to the far outer edge on the same row or
    column. Source pixels are erased from their original positions; the frame
    and the reference side stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    orient, sep, sep_color, ref_sign = _find_separator(g)
    source_sign = -ref_sign
    out = g.copy()

    if orient == "h":
        ref_rows = _side_indices(h, sep, ref_sign)
        source_rows = _side_indices(h, sep, source_sign)
        ref_bg = _dominant_color(g[ref_rows, :], {sep_color})
        source_bg = _dominant_color(g[source_rows, :], {sep_color})

        ref_pixels = [
            (r, c, int(g[r, c]))
            for r in ref_rows
            for c in range(w)
            if int(g[r, c]) not in {ref_bg, sep_color}
        ]
        source_pixels = [
            (r, c, int(g[r, c]))
            for r in source_rows
            for c in range(w)
            if int(g[r, c]) not in {source_bg, sep_color}
        ]

        assert ref_pixels and source_pixels
        assert len({r for r, _, _ in ref_pixels}) == 1, "reference pixels must occupy one row"
        assert len({r for r, _, _ in source_pixels}) == 1, "source pixels must occupy one row"

        ref_by_col = {c: color for _, c, color in ref_pixels}
        source_by_col = {c: (r, color) for r, c, color in source_pixels}
        assert len(ref_by_col) == len(ref_pixels)
        assert len(source_by_col) == len(source_pixels)
        assert set(ref_by_col) == set(source_by_col), "reference and source columns must align"

        near_row = sep + source_sign
        far_row = 0 if source_sign < 0 else h - 1
        moves: list[tuple[int, int, int, int, int]] = []
        for c, (r, color) in source_by_col.items():
            dst_r = near_row if color == ref_by_col[c] else far_row
            moves.append((r, c, int(dst_r), c, color))
        _apply_moves(out, moves, source_bg)
        return out

    ref_cols = _side_indices(w, sep, ref_sign)
    source_cols = _side_indices(w, sep, source_sign)
    ref_bg = _dominant_color(g[:, ref_cols], {sep_color})
    source_bg = _dominant_color(g[:, source_cols], {sep_color})

    ref_pixels = [
        (r, c, int(g[r, c]))
        for r in range(h)
        for c in ref_cols
        if int(g[r, c]) not in {ref_bg, sep_color}
    ]
    source_pixels = [
        (r, c, int(g[r, c]))
        for r in range(h)
        for c in source_cols
        if int(g[r, c]) not in {source_bg, sep_color}
    ]

    assert ref_pixels and source_pixels
    assert len({c for _, c, _ in ref_pixels}) == 1, "reference pixels must occupy one column"
    assert len({c for _, c, _ in source_pixels}) == 1, "source pixels must occupy one column"

    ref_by_row = {r: color for r, _, color in ref_pixels}
    source_by_row = {r: (c, color) for r, c, color in source_pixels}
    assert len(ref_by_row) == len(ref_pixels)
    assert len(source_by_row) == len(source_pixels)
    assert set(ref_by_row) == set(source_by_row), "reference and source rows must align"

    near_col = sep + source_sign
    far_col = 0 if source_sign < 0 else w - 1
    moves = []
    for r, (c, color) in source_by_row.items():
        dst_c = near_col if color == ref_by_row[r] else far_col
        moves.append((r, c, r, int(dst_c), color))
    _apply_moves(out, moves, source_bg)
    return out


# --------------- generator ---------------

ALL_COLORS = tuple(range(10))
ORIENTATIONS = ("h", "v")
REF_SIGNS = (-1, 1)

GRID_MIN = 8
GRID_MAX = 18
REF_DEPTH_MIN = 2
REF_DEPTH_MAX = 8
SOURCE_DEPTH_MIN = 3
SOURCE_DEPTH_MAX = 13
SEPARATOR_WIDTH = 1

TOKEN_COUNT_MIN = 4
TOKEN_COUNT_MAX = 8
EDGE_MARGIN = 1
MIN_SPAN_RATIO = 0.65

PALETTE_MIN = 3
PALETTE_MAX = 5
SAME_BG_PROB = 0.6
GEN_RETRIES = 300

DEPTH_PAIRS = [
    (ref_depth, source_depth)
    for ref_depth in range(REF_DEPTH_MIN, REF_DEPTH_MAX + 1)
    for source_depth in range(SOURCE_DEPTH_MIN, SOURCE_DEPTH_MAX + 1)
    if GRID_MIN <= ref_depth + source_depth + SEPARATOR_WIDTH <= GRID_MAX
]

Move = tuple[int, int, int, int, int]


def _sample_positions(axis_len: int, count: int) -> list[int]:
    usable = list(range(EDGE_MARGIN, axis_len - EDGE_MARGIN))
    if len(usable) < count:
        raise AssertionError("not enough aligned positions")

    min_span = int(np.ceil((axis_len - 1 - 2 * EDGE_MARGIN) * MIN_SPAN_RATIO))
    for _ in range(100):
        positions = sorted(random.sample(usable, count))
        if positions[-1] - positions[0] >= min_span:
            return positions
    return sorted(random.sample(usable, count))


def _sample_colors(
    count: int,
) -> tuple[int, int, int, list[int], list[int], list[bool]]:
    sep_color = random.choice(ALL_COLORS)
    bg_choices = [c for c in ALL_COLORS if c != sep_color]
    ref_bg = random.choice(bg_choices)
    if random.random() < SAME_BG_PROB:
        source_bg = ref_bg
    else:
        source_bg = random.choice([c for c in bg_choices if c != ref_bg])

    palette_size = random.randint(PALETTE_MIN, min(PALETTE_MAX, count))
    palette = random.sample(
        [c for c in ALL_COLORS if c not in {sep_color, ref_bg, source_bg}],
        palette_size,
    )
    ref_colors = palette[:] + random.choices(palette, k=count - palette_size)
    random.shuffle(ref_colors)

    match_count = random.randint(1, count - 1)
    match_indices = set(random.sample(range(count), match_count))

    source_colors = []
    matches = []
    for idx, ref_color in enumerate(ref_colors):
        matched = idx in match_indices
        matches.append(matched)
        if matched:
            source_colors.append(ref_color)
        else:
            source_colors.append(random.choice([c for c in palette if c != ref_color]))
    return sep_color, ref_bg, source_bg, ref_colors, source_colors, matches


def _sample_reference_distance(ref_depth: int) -> int:
    return random.randint(1, ref_depth)


def _sample_source_distance(ref_bg: int, source_bg: int, source_depth: int) -> int:
    if ref_bg == source_bg:
        return random.randint(1, source_depth - 1)
    return 1


def _render_target(
    orient: str,
    ref_depth: int,
    source_depth: int,
    axis_len: int,
    ref_sign: int,
    sep_color: int,
    ref_bg: int,
    source_bg: int,
    ref_colors: list[int],
    source_colors: list[int],
    matches: list[bool],
    positions: list[int],
    ref_dist: int,
    source_dist: int,
) -> tuple[np.ndarray, list[Move]]:
    normal_len = ref_depth + source_depth + SEPARATOR_WIDTH
    sep = ref_depth if ref_sign < 0 else source_depth
    ref_line = sep + ref_sign * ref_dist
    source_sign = -ref_sign
    source_line = sep + source_sign * source_dist
    near_line = sep + source_sign

    moves: list[Move] = []

    if orient == "h":
        h, w = normal_len, axis_len
        out = np.full((h, w), source_bg, dtype=int)
        ref_rows = slice(0, sep) if ref_sign < 0 else slice(sep + 1, h)
        out[ref_rows, :] = ref_bg
        out[sep, :] = sep_color
        out[ref_rows, 0] = sep_color
        out[ref_rows, w - 1] = sep_color
        out[ref_line, positions] = ref_colors

        far_row = 0 if source_sign < 0 else h - 1
        for c, color, matched in zip(positions, source_colors, matches):
            dst_r = near_line if matched else far_row
            out[dst_r, c] = color
            moves.append((source_line, c, int(dst_r), c, color))
        return out, moves

    h, w = axis_len, normal_len
    out = np.full((h, w), source_bg, dtype=int)
    ref_cols = slice(0, sep) if ref_sign < 0 else slice(sep + 1, w)
    out[:, ref_cols] = ref_bg
    out[:, sep] = sep_color
    out[0, ref_cols] = sep_color
    out[h - 1, ref_cols] = sep_color
    out[positions, ref_line] = ref_colors

    far_col = 0 if source_sign < 0 else w - 1
    for r, color, matched in zip(positions, source_colors, matches):
        dst_c = near_line if matched else far_col
        out[r, dst_c] = color
        moves.append((r, source_line, r, int(dst_c), color))
    return out, moves


def _invert_moves(out: np.ndarray, moves: list[Move], source_bg: int) -> np.ndarray:
    inp = out.copy()
    for sr, sc, dr, dc, _ in moves:
        if (sr, sc) != (dr, dc):
            inp[dr, dc] = source_bg
    for sr, sc, _, _, color in moves:
        inp[sr, sc] = color
    return inp


def _gen_attempt() -> np.ndarray | None:
    orient = random.choice(ORIENTATIONS)
    ref_depth, source_depth = random.choice(DEPTH_PAIRS)
    count = random.randint(TOKEN_COUNT_MIN, TOKEN_COUNT_MAX)
    axis_len = random.randint(max(GRID_MIN, count + 2), GRID_MAX)
    ref_sign = random.choice(REF_SIGNS)

    sep_color, ref_bg, source_bg, ref_colors, source_colors, matches = _sample_colors(count)
    positions = _sample_positions(axis_len, count)
    ref_dist = _sample_reference_distance(ref_depth)
    source_dist = _sample_source_distance(ref_bg, source_bg, source_depth)

    out, moves = _render_target(
        orient=orient,
        ref_depth=ref_depth,
        source_depth=source_depth,
        axis_len=axis_len,
        ref_sign=ref_sign,
        sep_color=sep_color,
        ref_bg=ref_bg,
        source_bg=source_bg,
        ref_colors=ref_colors,
        source_colors=source_colors,
        matches=matches,
        positions=positions,
        ref_dist=ref_dist,
        source_dist=source_dist,
    )
    inp = _invert_moves(out, moves, source_bg)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
