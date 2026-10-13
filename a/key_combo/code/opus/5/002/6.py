from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords  # noqa: E402


GLYPH_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
GLYPH_PATTERNS = (
    np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[0, 0, 1], [0, 0, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
)


def _pattern_colset(mask: np.ndarray) -> set[int]:
    return {int(c) for c in np.where(mask.any(axis=0))[0]}


PATTERN_COLSETS = tuple(_pattern_colset(mask) for mask in GLYPH_PATTERNS)


def _border_cycle(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    assert h >= 2 and w >= 2
    coords: list[tuple[int, int]] = []
    for c in range(w):
        coords.append((0, c))
    for r in range(1, h):
        coords.append((r, w - 1))
    for c in range(w - 2, -1, -1):
        coords.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _cyclic_contiguous_run(indices: list[int], period: int) -> tuple[int, int]:
    assert indices
    ordered = sorted(int(i) for i in indices)
    assert len(ordered) == len(set(ordered))
    if len(ordered) == 1:
        return ordered[0], 1
    gaps = [(ordered[(i + 1) % len(ordered)] - ordered[i]) % int(period) for i in range(len(ordered))]
    wide = [i for i, gap in enumerate(gaps) if gap != 1]
    assert len(wide) == 1
    start = ordered[(wide[0] + 1) % len(ordered)]
    return int(start), len(ordered)


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == Color.PINK)]
    assert len(rows) == 1, "expected one full-width pink separator row"
    return int(rows[0])


def _parse_table(top: np.ndarray) -> list[list[dict[str, object]]]:
    fg = np.argwhere(top != Color.TEAL)
    assert fg.size > 0, "table must contain non-background pixels"
    r0, c0, r1, c1 = bbox_coords(fg)
    table = top[r0 : r1 + 1, c0 : c1 + 1]
    assert table.shape[0] == 9, "table must contain exactly three 3-row glyph bands"

    sep_cols = [c for c in range(table.shape[1]) if np.all(table[:, c] == Color.PINK)]
    assert len(sep_cols) == 3, "table must have three full-height vertical separators"

    spans: list[tuple[int, int]] = []
    start = 0
    for sep in sep_cols + [table.shape[1]]:
        if sep > start:
            spans.append((start, sep))
        start = sep + 1
    assert len(spans) == 4 and all(c1 - c0 == 3 for c0, c1 in spans), "expected four 3x3 glyph columns"

    rows: list[list[dict[str, object]]] = []
    for row_idx in range(3):
        row: list[dict[str, object]] = []
        rr0 = row_idx * 3
        for c0, c1 in spans:
            cell = table[rr0 : rr0 + 3, c0:c1]
            colors = {int(v) for v in cell.reshape(-1) if int(v) != int(Color.TEAL)}
            assert len(colors) == 1, "each glyph cell must be monochrome"
            color = colors.pop()
            mask = cell == color
            assert bool(mask.any()), "glyph cell cannot be empty"
            row.append(
                {
                    "color": int(color),
                    "mask": mask.copy(),
                    "cols": _pattern_colset(mask),
                }
            )
        rows.append(row)
    return rows


def _gap_for_row(row: list[dict[str, object]]) -> int:
    common = set.intersection(*(set(cell["cols"]) for cell in row))
    return 1 if common else 2


def _orange_segment(lower: np.ndarray) -> tuple[int, int, list[tuple[int, int]]]:
    border = _border_cycle(lower.shape)
    border_index = {coord: i for i, coord in enumerate(border)}
    orange = [tuple(map(int, rc)) for rc in np.argwhere(lower == Color.ORANGE)]
    assert orange, "expected one orange border segment"
    assert all(coord in border_index for coord in orange), "orange pixels must stay on the lower border"
    assert np.all(np.isin(lower, [Color.BLACK, Color.ORANGE])), "lower area should only use black and orange"
    start, run_len = _cyclic_contiguous_run([border_index[coord] for coord in orange], len(border))
    return int(start), int(run_len), border


def _paint_border(lower: np.ndarray, start: int, colors: list[int], gap: int) -> np.ndarray:
    out = lower.copy()
    out[out == Color.ORANGE] = Color.BLACK
    border = _border_cycle(lower.shape)
    path = border[start:] + border[:start]
    pos = 0
    while pos < len(path):
        for k, color in enumerate(colors):
            idx = pos + k
            if idx >= len(path):
                break
            r, c = path[idx]
            out[r, c] = int(color)
        pos += len(colors) + int(gap)
    return out


def _render_output_from_choice(grid: np.ndarray, row_index: int, gap_override: int | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    rows = _parse_table(g[:sep, :])
    lower = g[sep + 1 :, :]
    start, _seg_len, _border = _orange_segment(lower)
    row = rows[int(row_index)]
    colors = [int(cell["color"]) for cell in row]
    gap = _gap_for_row(row) if gap_override is None else int(gap_override)

    out = g.copy()
    out[sep + 1 :, :] = _paint_border(lower, start, colors, gap)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the orange segment length into a glyph-table row, then cycle that row's colors around the lower border with an alignment-dependent gap.

    Input structure:
    - A full-width pink separator row splits the grid into a teal reference table
      above and a black lower rectangle below.
    - The upper table contains 3 rows x 4 columns of 3x3 monochrome glyph cells,
      with full-height pink separator columns between cell columns.
    - The lower rectangle is otherwise black except for one contiguous orange
      segment on its outer border.

    Rule:
    - Let L be the length of the orange border segment; use row L of the glyph
      table as the lookup row.
    - Read that row's four glyph colors from left to right to obtain the cyclic
      paint sequence.
    - Inspect the same row's glyph masks in their own 3x3 local coordinates: if
      all four glyphs occupy at least one common local column, use gap G = 1;
      otherwise use gap G = 2.
    - Starting from the orange segment's first border cell in clockwise order,
      paint the four colors, leave G border cells black, and repeat until one
      full perimeter lap is exhausted; an incomplete final batch paints only the
      colors that still fit. The upper table stays unchanged.

    Key insight:
    - The border segment length is only an address. It must be decoded into a
      table row before either the paint colors or the paint spacing become known.
    """
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    rows = _parse_table(g[:sep, :])
    lower = g[sep + 1 :, :]
    start, seg_len, _border = _orange_segment(lower)
    assert 1 <= seg_len <= len(rows), "orange segment length must index a table row"

    row = rows[seg_len - 1]
    colors = [int(cell["color"]) for cell in row]
    gap = _gap_for_row(row)

    out = g.copy()
    out[sep + 1 :, :] = _paint_border(lower, start, colors, gap)
    return out


def _edge_runs(shape: tuple[int, int], seg_len: int) -> dict[str, list[list[tuple[int, int]]]]:
    h, w = map(int, shape)
    edges = {
        "top": [(0, c) for c in range(1, w - 1)],
        "right": [(r, w - 1) for r in range(1, h - 1)],
        "bottom": [(h - 1, c) for c in range(w - 2, 0, -1)],
        "left": [(r, 0) for r in range(h - 2, 0, -1)],
    }
    runs: dict[str, list[list[tuple[int, int]]]] = {}
    for edge, coords in edges.items():
        runs[edge] = [coords[i : i + seg_len] for i in range(len(coords) - seg_len + 1)]
    return runs


def _sample_row_patterns(aligned: bool) -> list[np.ndarray]:
    indices = list(range(len(GLYPH_PATTERNS)))
    if aligned:
        common_col = random.randrange(3)
        pool = [i for i, cols in enumerate(PATTERN_COLSETS) if common_col in cols]
        chosen = random.sample(pool, 4)
    else:
        while True:
            chosen = random.sample(indices, 4)
            common = set.intersection(*(set(PATTERN_COLSETS[i]) for i in chosen))
            if not common:
                break
    return [GLYPH_PATTERNS[i] for i in chosen]


def _draw_table(width: int, left_margin: int, row_perms: list[tuple[int, int, int, int]], row_gaps: list[int]) -> np.ndarray:
    top = np.full((9, width), Color.TEAL, dtype=int)
    table_left = int(left_margin)
    for c in (3, 7, 11):
        top[:, table_left + c] = Color.PINK

    col_starts = [0, 4, 8, 12]
    for row_idx in range(3):
        masks = _sample_row_patterns(aligned=row_gaps[row_idx] == 1)
        for col_idx, color in enumerate(row_perms[row_idx]):
            r0 = row_idx * 3
            c0 = table_left + col_starts[col_idx]
            top[r0 : r0 + 3, c0 : c0 + 3][masks[col_idx]] = int(color)
    return top


def _distinct_row_permutations() -> list[tuple[int, int, int, int]]:
    perms = random.sample(list(itertools.permutations(GLYPH_COLORS)), 3)
    random.shuffle(perms)
    return perms


def _choose_row_gaps(target_row: int, target_gap: int) -> list[int]:
    while True:
        row_gaps = [random.choice((1, 2)) for _ in range(3)]
        row_gaps[target_row] = int(target_gap)
        if len(set(row_gaps)) == 2:
            return row_gaps


def _semantic_checks(grid: np.ndarray, row_index: int, gap: int) -> None:
    sep = _separator_row(grid)
    rows = _parse_table(grid[:sep, :])
    lower = grid[sep + 1 :, :]
    start, seg_len, _border = _orange_segment(lower)
    assert seg_len - 1 == row_index
    assert _gap_for_row(rows[row_index]) == gap
    assert len({tuple(int(cell["color"]) for cell in row) for row in rows}) == 3
    assert len({_gap_for_row(row) for row in rows}) == 2
    assert start >= 0

    correct = _render_output_from_choice(grid, row_index)
    for alt_row in range(3):
        if alt_row == row_index:
            continue
        assert not np.array_equal(correct, _render_output_from_choice(grid, alt_row)), "wrong row must change output"
    alt_gap = 1 if gap == 2 else 2
    assert not np.array_equal(correct, _render_output_from_choice(grid, row_index, gap_override=alt_gap)), "wrong gap must change output"
    assert np.array_equal(main(grid), correct)


def _gen_attempt() -> np.ndarray | None:
    left_margin = random.randint(0, 5)
    right_margin = random.randint(0, 5)
    width = 15 + left_margin + right_margin
    lower_h = random.randint(7, 13)
    if max(width, 10 + lower_h) > 30:
        return None

    row_index = random.randrange(3)
    gap = random.choice((1, 2))
    row_perms = _distinct_row_permutations()
    row_gaps = _choose_row_gaps(row_index, gap)

    top = _draw_table(width, left_margin, row_perms, row_gaps)
    sep_row = np.full((1, width), Color.PINK, dtype=int)

    lower = np.full((lower_h, width), Color.BLACK, dtype=int)
    seg_len = row_index + 1
    runs_by_edge = _edge_runs(lower.shape, seg_len)
    valid_edges = [edge for edge, runs in runs_by_edge.items() if runs]
    if not valid_edges:
        return None
    edge = random.choice(valid_edges)
    segment = random.choice(runs_by_edge[edge])
    for r, c in segment:
        lower[r, c] = Color.ORANGE

    grid = np.vstack([top, sep_row, lower])
    try:
        _semantic_checks(grid, row_index, gap)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
