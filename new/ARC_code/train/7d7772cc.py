from __future__ import annotations

from collections import Counter

import numpy as np
import random


def _dominant_color(values: np.ndarray, excluded: set[int]) -> int:
    counts = Counter(int(v) for v in values.ravel() if int(v) not in excluded)
    assert counts, "expected at least one non-separator cell on each side"
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, f"side background must be unique, got {winners}"
    return int(winners[0])


def _find_separator(g: np.ndarray) -> tuple[str, int, int, int]:
    H, W = g.shape
    candidates: list[tuple[str, int, int, int]] = []

    for r in range(1, H - 1):
        color = int(g[r, 0])
        if np.all(g[r, :] == color):
            before = int(np.sum(g[:r, :] == color))
            after = int(np.sum(g[r + 1 :, :] == color))
            if (before > 0) ^ (after > 0):
                ref_sign = -1 if before > 0 else 1
                candidates.append(("h", int(r), color, ref_sign))

    for c in range(1, W - 1):
        color = int(g[0, c])
        if np.all(g[:, c] == color):
            before = int(np.sum(g[:, :c] == color))
            after = int(np.sum(g[:, c + 1 :] == color))
            if (before > 0) ^ (after > 0):
                ref_sign = -1 if before > 0 else 1
                candidates.append(("v", int(c), color, ref_sign))

    assert len(candidates) == 1, f"expected one framed separator line, got {candidates}"
    return candidates[0]


def _side_indices(limit: int, sep: int, sign: int) -> list[int]:
    assert sign in (-1, 1)
    indices = list(range(0, sep) if sign < 0 else range(sep + 1, limit))
    assert indices, "separator must have a nonempty side"
    return indices


def _move_pixels(
    out: np.ndarray,
    moves: list[tuple[int, int, int, int, int]],
    source_bg: int,
) -> None:
    source_cells = {(sr, sc) for sr, sc, _, _, _ in moves}
    dest_cells = {(dr, dc) for _, _, dr, dc, _ in moves}
    assert len(dest_cells) == len(moves), "source pixels must have distinct destinations"

    for sr, sc, _, _, _ in moves:
        out[sr, sc] = source_bg

    for sr, sc, dr, dc, color in moves:
        assert (dr, dc) == (sr, sc) or (dr, dc) not in source_cells
        assert (dr, dc) == (sr, sc) or int(out[dr, dc]) == source_bg
        out[dr, dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each source pixel toward the separator when it matches its aligned reference pixel, otherwise to the far edge.

    Input structure:
    - A single full separator line is part of a rectangular frame; the frame's
      extra cells mark the reference side.
    - On the reference side, one row or column of colored pixels is aligned
      with one row or column of source pixels on the opposite side.
    - Each side has its own dominant background color.

    Rule:
    - Compare each source pixel with the reference pixel on the same row or column.
    - If their colors match, place the source pixel in the cell adjacent to the
      separator on the source side.
    - If their colors differ, place the source pixel on the far outer edge along
      the same row or column.
    - Erase the original source pixels to the source-side background; keep the
      frame and reference pixels unchanged.

    Key insight:
    The extra frame cells identify which side is the reference; equality with the
    aligned reference pixel decides whether each source pixel moves inward or outward.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    orient, sep, sep_color, ref_sign = _find_separator(g)
    source_sign = -ref_sign

    if orient == "h":
        ref_rows = _side_indices(H, sep, ref_sign)
        source_rows = _side_indices(H, sep, source_sign)
        ref_bg = _dominant_color(g[ref_rows, :], {sep_color})
        source_bg = _dominant_color(g[source_rows, :], {sep_color})

        ref_pixels = [
            (r, c, int(g[r, c]))
            for r in ref_rows
            for c in range(W)
            if int(g[r, c]) not in {ref_bg, sep_color}
        ]
        source_pixels = [
            (r, c, int(g[r, c]))
            for r in source_rows
            for c in range(W)
            if int(g[r, c]) not in {source_bg, sep_color}
        ]

        assert len({r for r, _, _ in ref_pixels}) == 1
        assert len({r for r, _, _ in source_pixels}) == 1
        ref_by_col = {c: color for _, c, color in ref_pixels}
        source_by_col = {c: (r, color) for r, c, color in source_pixels}
        assert len(ref_by_col) == len(ref_pixels)
        assert len(source_by_col) == len(source_pixels)
        assert set(ref_by_col) == set(source_by_col)

        moves: list[tuple[int, int, int, int, int]] = []
        for c, (r, color) in source_by_col.items():
            dest_r = sep + source_sign if color == ref_by_col[c] else (0 if source_sign < 0 else H - 1)
            moves.append((r, c, int(dest_r), c, color))
        _move_pixels(out, moves, source_bg)
        return out

    ref_cols = _side_indices(W, sep, ref_sign)
    source_cols = _side_indices(W, sep, source_sign)
    ref_bg = _dominant_color(g[:, ref_cols], {sep_color})
    source_bg = _dominant_color(g[:, source_cols], {sep_color})

    ref_pixels = [
        (r, c, int(g[r, c]))
        for r in range(H)
        for c in ref_cols
        if int(g[r, c]) not in {ref_bg, sep_color}
    ]
    source_pixels = [
        (r, c, int(g[r, c]))
        for r in range(H)
        for c in source_cols
        if int(g[r, c]) not in {source_bg, sep_color}
    ]

    assert len({c for _, c, _ in ref_pixels}) == 1
    assert len({c for _, c, _ in source_pixels}) == 1
    ref_by_row = {r: color for r, _, color in ref_pixels}
    source_by_row = {r: (c, color) for r, c, color in source_pixels}
    assert len(ref_by_row) == len(ref_pixels)
    assert len(source_by_row) == len(source_pixels)
    assert set(ref_by_row) == set(source_by_row)

    moves = []
    for r, (c, color) in source_by_row.items():
        dest_c = sep + source_sign if color == ref_by_row[r] else (0 if source_sign < 0 else W - 1)
        moves.append((r, c, r, int(dest_c), color))
    _move_pixels(out, moves, source_bg)
    return out

# --------------- generator (att2) ---------------

# Color sampling.
ALL_COLORS = tuple(range(10))
SAME_BACKGROUND_PROB = 0.6
TOKEN_PALETTE_MIN, TOKEN_PALETTE_MAX = 3, 5

# Grid, separator, and side depths.
GRID_MIN, GRID_MAX = 8, 16
REF_DEPTH_MIN, REF_DEPTH_MAX = 2, 8
SOURCE_DEPTH_MIN, SOURCE_DEPTH_MAX = 3, 13
SEPARATOR_THICKNESS = 1

# Token count and wide-span placement.
TOKEN_COUNT_MIN, TOKEN_COUNT_MAX = 5, 7
POSITION_EDGE_MARGIN = 1
WIDE_SPAN_RATIO = 0.8

# Source/reference match weighting and retries.
MATCH_WEIGHTS = {1: 2, 2: 4}
DEFAULT_MATCH_WEIGHT = 1
GEN_RETRIES = 300

ORIENTATIONS = ("h", "v")
REF_SIGNS = (-1, 1)
DEPTH_PAIRS = [
    (ref_depth, source_depth)
    for ref_depth in range(REF_DEPTH_MIN, REF_DEPTH_MAX + 1)
    for source_depth in range(SOURCE_DEPTH_MIN, SOURCE_DEPTH_MAX + 1)
    if GRID_MIN <= ref_depth + source_depth + SEPARATOR_THICKNESS <= GRID_MAX
]

SourceMove = tuple[int, int, int, int, int]


def _wide_positions(axis_len: int, count: int) -> list[int]:
    max_span = axis_len - 2 * POSITION_EDGE_MARGIN - 1
    min_span = int(np.ceil(max_span * WIDE_SPAN_RATIO))
    last_pos = axis_len - POSITION_EDGE_MARGIN - 1
    lo = random.randint(POSITION_EDGE_MARGIN, last_pos - min_span)
    hi = random.randint(lo + min_span, last_pos)
    middle = [p for p in range(POSITION_EDGE_MARGIN, axis_len - POSITION_EDGE_MARGIN) if p not in (lo, hi)]
    return sorted([lo, hi] + random.sample(middle, count - 2))


def _colors(count: int) -> tuple[int, int, int, list[int], list[int], list[bool]]:
    sep_color = random.choice(ALL_COLORS)
    bg_choices = [c for c in ALL_COLORS if c != sep_color]
    ref_bg = random.choice(bg_choices)
    if random.random() < SAME_BACKGROUND_PROB:
        source_bg = ref_bg
    else:
        source_bg = random.choice([c for c in bg_choices if c != ref_bg])

    palette_size = random.randint(TOKEN_PALETTE_MIN, min(TOKEN_PALETTE_MAX, count))
    palette = random.sample(
        [c for c in ALL_COLORS if c not in {sep_color, ref_bg, source_bg}],
        palette_size,
    )
    ref_colors = palette[:] + random.choices(palette, k=count - palette_size)
    random.shuffle(ref_colors)

    match_choices = list(range(1, count))
    weights = [MATCH_WEIGHTS.get(k, DEFAULT_MATCH_WEIGHT) for k in match_choices]
    match_count = random.choices(match_choices, weights=weights, k=1)[0]
    match_indices = set(random.sample(range(count), match_count))

    source_colors = []
    matches = []
    for i, ref_color in enumerate(ref_colors):
        is_match = i in match_indices
        matches.append(is_match)
        source_colors.append(ref_color if is_match else random.choice([c for c in palette if c != ref_color]))
    return sep_color, ref_bg, source_bg, ref_colors, source_colors, matches


def _source_distance(ref_bg: int, source_bg: int, source_depth: int) -> int:
    if source_bg == ref_bg:
        return random.randint(SEPARATOR_THICKNESS, source_depth - SEPARATOR_THICKNESS)
    return SEPARATOR_THICKNESS


def _render_output(
    orient: str, ref_depth: int, source_depth: int, axis_len: int, ref_sign: int,
    sep_color: int, ref_bg: int, source_bg: int, ref_colors: list[int],
    source_colors: list[int], matches: list[bool], positions: list[int], source_dist: int,
) -> tuple[np.ndarray, list[SourceMove]]:
    """Build the desired output grid and the source moves needed to invert it."""
    side_len = ref_depth + source_depth + SEPARATOR_THICKNESS
    sep = ref_depth if ref_sign < 0 else source_depth
    source_sign = -ref_sign
    source_line = sep + source_sign * source_dist
    adjacent_line = sep + source_sign
    source_moves = []

    if orient == "h":
        H, W = side_len, axis_len
        out = np.full((H, W), source_bg, dtype=int)
        ref_rows = slice(0, sep) if ref_sign < 0 else slice(sep + SEPARATOR_THICKNESS, H)
        out[ref_rows, :] = ref_bg
        out[sep, :] = sep_color
        out[ref_rows, 0] = sep_color
        out[ref_rows, W - 1] = sep_color
        out[sep + ref_sign, positions] = ref_colors
        far_line = 0 if source_sign < 0 else H - 1
        for c, color, is_match in zip(positions, source_colors, matches):
            dst_r = adjacent_line if is_match else far_line
            out[dst_r, c] = color
            source_moves.append((source_line, c, dst_r, c, color))
        return out, source_moves

    H, W = axis_len, side_len
    out = np.full((H, W), source_bg, dtype=int)
    ref_cols = slice(0, sep) if ref_sign < 0 else slice(sep + SEPARATOR_THICKNESS, W)
    out[:, ref_cols] = ref_bg
    out[:, sep] = sep_color
    out[0, ref_cols] = sep_color
    out[H - 1, ref_cols] = sep_color
    out[positions, sep + ref_sign] = ref_colors
    far_line = 0 if source_sign < 0 else W - 1
    for r, color, is_match in zip(positions, source_colors, matches):
        dst_c = adjacent_line if is_match else far_line
        out[r, dst_c] = color
        source_moves.append((r, source_line, r, dst_c, color))
    return out, source_moves


def _invert_source_moves(out: np.ndarray, source_moves: list[SourceMove], source_bg: int) -> np.ndarray:
    inp = out.copy()
    for sr, sc, dr, dc, _ in source_moves:
        if (sr, sc) != (dr, dc):
            inp[dr, dc] = source_bg
    for sr, sc, _, _, color in source_moves:
        inp[sr, sc] = color
    return inp


def _gen_attempt() -> np.ndarray | None:
    orient = random.choice(ORIENTATIONS)
    ref_depth, source_depth = random.choice(DEPTH_PAIRS)
    count = random.randint(TOKEN_COUNT_MIN, TOKEN_COUNT_MAX)
    axis_len = random.randint(max(GRID_MIN, count + 2), GRID_MAX)
    ref_sign = random.choice(REF_SIGNS)

    sep_color, ref_bg, source_bg, ref_colors, source_colors, matches = _colors(count)
    positions = _wide_positions(axis_len, count)
    source_dist = _source_distance(ref_bg, source_bg, source_depth)

    out, source_moves = _render_output(
        orient, ref_depth, source_depth, axis_len, ref_sign,
        sep_color, ref_bg, source_bg, ref_colors, source_colors, matches, positions, source_dist,
    )
    inp = _invert_source_moves(out, source_moves, source_bg)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 7d7772cc input")
