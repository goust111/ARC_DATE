from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEP = 5
TOKEN_COLORS = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(1, grid.shape[0] - 1) if np.all(grid[r, :] == SEP)]
    assert len(rows) == 1, f"expected one horizontal separator row, got {rows}"
    return int(rows[0])


def _token_columns(row: np.ndarray) -> list[int]:
    return [int(c) for c in np.flatnonzero((row != BG) & (row != SEP))]


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape

    sep = _find_separator_row(g)
    ref_row = sep - 1
    assert ref_row >= 0, "reference row must exist immediately above the separator"

    ref_cols = _token_columns(g[ref_row])
    assert len(ref_cols) >= 3, "expected several reference pixels"

    lower_rows = [
        int(r)
        for r in range(sep + 1, H)
        if np.any((g[r, :] != BG) & (g[r, :] != SEP))
    ]
    assert len(lower_rows) == 1, f"expected one source row below separator, got {lower_rows}"
    source_row = lower_rows[0]
    assert 2 <= source_row - sep <= 3, "source row must be 2-3 rows below the separator"
    assert source_row < H - 1, "source row must leave visible room for downward rays"

    for r in range(sep + 1, H):
        if r == source_row:
            continue
        assert not np.any((g[r, :] != BG) & (g[r, :] != SEP)), "lower area may only contain one source row"

    source_cols = _token_columns(g[source_row])
    assert source_cols == ref_cols, "reference and source pixels must use the same columns"

    for c in ref_cols:
        assert int(g[ref_row, c]) in TOKEN_COLORS, "reference color must be one of the visible token colors"
        assert int(g[source_row, c]) in TOKEN_COLORS, "source color must be one of the visible token colors"

    return {
        "sep": int(sep),
        "ref_row": int(ref_row),
        "source_row": int(source_row),
        "cols": ref_cols,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare each source pixel with the aligned reference pixel and extend a one-way ray in that column.

    Input structure:
    - The background is black, and exactly one full gray horizontal separator line spans the grid.
    - The row immediately above the separator holds several colored reference pixels in distinct columns.
    - One lower row, 2-3 cells below the separator, holds source pixels in exactly the same columns.

    Rule:
    - Keep the separator row and the reference row unchanged.
    - For each source pixel, compare its color with the reference pixel in the same column.
    - If the colors match, fill that column downward from the source row to the bottom with the source color.
    - If the colors differ, fill that column upward from the source row to the cell just below the separator with the source color.

    Key insight:
    - The reference row is only a lookup table: color equality with the aligned reference chooses ray direction, and each column then propagates independently.
    """
    g = np.asarray(grid, dtype=int)
    info = _extract_structure(g)
    out = g.copy()

    ref_row = info["ref_row"]
    source_row = info["source_row"]
    sep = info["sep"]

    for c in info["cols"]:
        ref_color = int(g[ref_row, c])
        src_color = int(g[source_row, c])
        if src_color == ref_color:
            out[source_row:, c] = src_color
        else:
            out[sep + 1 : source_row + 1, c] = src_color

    return out


def _wide_columns(width: int, count: int) -> list[int]:
    usable = list(range(1, width - 1))
    assert len(usable) >= count
    max_span = usable[-1] - usable[0]
    min_span = max(count - 1, int(np.ceil(max_span * 0.6)))
    lo = random.randint(usable[0], usable[-1] - min_span)
    hi = random.randint(lo + min_span, usable[-1])
    middle = [c for c in usable if c not in {lo, hi}]
    return sorted([lo, hi] + random.sample(middle, count - 2))


def _branches_interleaved(matches: list[bool]) -> bool:
    transitions = sum(matches[i] != matches[i - 1] for i in range(1, len(matches)))
    return transitions >= 2


def _color_seen_in_both_branches(colors: list[int], matches: list[bool]) -> bool:
    branches_by_color: dict[int, set[bool]] = {}
    for color, match in zip(colors, matches):
        branches_by_color.setdefault(int(color), set()).add(bool(match))
    return any(len(branches) == 2 for branches in branches_by_color.values())


def _parity_mixes(cols: list[int], matches: list[bool]) -> bool:
    match_parities = {int(c) % 2 for c, match in zip(cols, matches) if match}
    mismatch_parities = {int(c) % 2 for c, match in zip(cols, matches) if not match}
    return match_parities == {0, 1} and mismatch_parities == {0, 1}


def _attempt_input() -> np.ndarray | None:
    width = random.randint(12, 18)
    separator_row = random.randint(3, 5)
    gap = random.randint(2, 3)
    source_row = separator_row + gap
    height = source_row + random.randint(2, 6) + 1
    if height > 30:
        return None

    token_count = random.randint(5, min(7, width - 2))
    cols = _wide_columns(width, token_count)

    for _ in range(200):
        match_count = random.randint(2, token_count - 2)
        match_indices = set(random.sample(range(token_count), match_count))
        matches = [i in match_indices for i in range(token_count)]
        if not _branches_interleaved(matches):
            continue
        if not _parity_mixes(cols, matches):
            continue

        ref_colors = random.choices(TOKEN_COLORS, k=token_count)
        if max(Counter(ref_colors).values()) < 2:
            continue

        src_colors: list[int] = []
        for ref_color, match in zip(ref_colors, matches):
            if match:
                src_colors.append(int(ref_color))
            else:
                src_colors.append(int(random.choice([c for c in TOKEN_COLORS if c != ref_color])))

        if not _color_seen_in_both_branches(ref_colors, matches):
            continue
        if not _color_seen_in_both_branches(src_colors, matches):
            continue

        grid = np.full((height, width), BG, dtype=int)
        grid[separator_row, :] = SEP
        for c, ref_color, src_color in zip(cols, ref_colors, src_colors):
            grid[separator_row - 1, c] = int(ref_color)
            grid[source_row, c] = int(src_color)
        return grid

    return None


def _validate_sample(inp: np.ndarray) -> None:
    info = _extract_structure(inp)
    out = main(inp)

    sep = info["sep"]
    ref_row = info["ref_row"]
    source_row = info["source_row"]
    cols = info["cols"]
    ref_colors = [int(inp[ref_row, c]) for c in cols]
    src_colors = [int(inp[source_row, c]) for c in cols]
    matches = [src == ref for src, ref in zip(src_colors, ref_colors)]

    assert any(matches) and not all(matches), "need at least one matching and one mismatching pair"
    assert _branches_interleaved(matches), "branch directions should not collapse to a left/right split"
    assert _parity_mixes(cols, matches), "both branches should appear on both column parities"
    assert _color_seen_in_both_branches(ref_colors, matches), "reference color alone must not determine direction"
    assert _color_seen_in_both_branches(src_colors, matches), "source color alone must not determine direction"

    assert np.array_equal(out[ref_row, :], inp[ref_row, :]), "reference row must stay unchanged"
    assert np.array_equal(out[sep, :], inp[sep, :]), "separator row must stay unchanged"

    affected = np.zeros_like(inp, dtype=bool)
    for c, src_color, match in zip(cols, src_colors, matches):
        up_version = inp[:, c].copy()
        up_version[sep + 1 : source_row + 1] = src_color
        down_version = inp[:, c].copy()
        down_version[source_row:,] = down_version[source_row:]
        down_version[source_row:] = src_color
        assert not np.array_equal(up_version, down_version), "the two ray directions must be distinguishable"

        expected_col = down_version if match else up_version
        assert np.array_equal(out[:, c], expected_col), "column output does not follow the branch rule"
        if match:
            assert np.all(out[source_row:, c] == src_color)
        else:
            assert np.all(out[sep + 1 : source_row + 1, c] == src_color)
            if source_row + 1 < inp.shape[0]:
                assert np.all(out[source_row + 1 :, c] == inp[source_row + 1 :, c])

        if match:
            affected[source_row:, c] = True
        else:
            affected[sep + 1 : source_row + 1, c] = True

    assert np.array_equal(out[~affected], inp[~affected]), "cells outside the rays must remain unchanged"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is None:
            continue
        _validate_sample(grid)
        return grid
    raise RuntimeError("failed to generate a valid non-degenerate puzzle input")
