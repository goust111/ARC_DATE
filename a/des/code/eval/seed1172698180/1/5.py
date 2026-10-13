from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
ACTIVE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    return int(rows[0])


def _ordered_path(grid: np.ndarray, sep_row: int) -> list[tuple[int, int]]:
    upper = grid[:sep_row]
    path_mask = upper != BLACK
    coords = [tuple(map(int, rc)) for rc in np.argwhere(path_mask)]
    assert coords, "top region must contain a path"
    path_set = set(coords)

    left = [(r, c) for r, c in coords if c == 0]
    right = [(r, c) for r, c in coords if c == upper.shape[1] - 1]
    assert len(left) == 1 and len(right) == 1, "path must touch left and right edges exactly once"
    start = left[0]
    end = right[0]

    for cell in coords:
        degree = sum((nr, nc) in path_set for nr, nc in _neighbors4(*cell))
        if cell in (start, end):
            assert degree == 1, "path endpoints must have degree 1"
        else:
            assert degree == 2, "path must be a simple non-branching chain"

    ordered = [start]
    prev = None
    cur = start
    while cur != end:
        nxts = [nb for nb in _neighbors4(*cur) if nb in path_set and nb != prev]
        assert len(nxts) == 1, "path walk must have a unique next step"
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    assert len(ordered) == len(coords), "path walk must cover all path cells"
    return ordered


def _parse_palette(grid: np.ndarray, sep_row: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    last_nonempty = -2
    for r in range(sep_row + 1, grid.shape[0]):
        row = grid[r]
        cols = np.flatnonzero(row != BLACK)
        if cols.size == 0:
            continue
        assert r - last_nonempty >= 2, "palette rows must be separated by at least one black row"
        last_nonempty = r
        c0 = int(cols[0])
        c1 = int(cols[-1])
        assert cols.size == c1 - c0 + 1, "palette row must be one contiguous run"
        vals = [int(v) for v in row[c0 : c1 + 1]]
        assert len(vals) >= 2, "palette row must contain index and paint colors"
        index_color = vals[0]
        paint_color = vals[1]
        assert index_color not in mapping, "palette index colors must be unique"
        mapping[index_color] = paint_color

    assert mapping, "palette must not be empty"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the grey path intervals according to marker-indexed palette rows.

    Input structure:
    - black background with a full-width grey separator.
    - above the separator: one simple left-to-right path made of grey cells with
      colored marker cells embedded in the path.
    - below the separator: horizontal palette rows; each row starts with an index
      color followed by one or more paint cells.

    Rule:
    - walk the path from the left endpoint to the right endpoint and read marker
      colors in order.
    - map each marker color to the palette row whose first cell has the same color.
    - use the second cell of that palette row to recolor grey path cells:
      prefix before the first marker uses the first marker's palette row;
      segment after marker k and before marker k+1 uses marker k's palette row;
      suffix after the last marker uses the last marker's palette row.
    - marker cells, separator, and palette stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    sep_row = _find_separator_row(g)
    path = _ordered_path(g, sep_row)
    palette = _parse_palette(g, sep_row)

    markers: list[tuple[int, int]] = []
    for idx, (r, c) in enumerate(path):
        color = int(g[r, c])
        if color not in (BLACK, GREY):
            markers.append((idx, color))

    assert markers, "path must contain at least one colored marker"
    for _, color in markers:
        assert color in palette, "every marker color must appear in the palette"

    out = g.copy()

    def paint_segment(start_idx: int, end_idx: int, color: int) -> None:
        for pos in range(start_idx, end_idx + 1):
            r, c = path[pos]
            if int(out[r, c]) == GREY:
                out[r, c] = int(color)

    first_idx, first_color = markers[0]
    paint_segment(0, first_idx - 1, palette[first_color])

    for (cur_idx, cur_color), (next_idx, _next_color) in zip(markers, markers[1:]):
        paint_segment(cur_idx + 1, next_idx - 1, palette[cur_color])

    last_idx, last_color = markers[-1]
    paint_segment(last_idx + 1, len(path) - 1, palette[last_color])
    return out


def _sample_spaced_positions(
    start: int, stop: int, count: int, min_gap: int, rng: random.Random
) -> list[int] | None:
    if count == 0:
        return []
    values = list(range(start, stop + 1))
    if len(values) < count:
        return None
    for _ in range(200):
        picks = sorted(rng.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _build_path(upper_h: int, width: int, rng: random.Random) -> list[tuple[int, int]] | None:
    rows = list(range(1, upper_h - 1))
    if not rows:
        return None

    max_turns = min(4, (width - 2) // 2)
    turn_count = rng.randint(0, max_turns)
    turn_cols = _sample_spaced_positions(1, width - 2, turn_count, 2, rng)
    if turn_cols is None:
        return None

    cur_row = rng.choice(rows)
    cur_col = 0
    path = [(cur_row, cur_col)]
    for turn_col in turn_cols:
        for c in range(cur_col + 1, turn_col + 1):
            path.append((cur_row, c))
        next_row_choices = [r for r in rows if r != cur_row]
        if not next_row_choices:
            return None
        next_row = rng.choice(next_row_choices)
        step = 1 if next_row > cur_row else -1
        for r in range(cur_row + step, next_row + step, step):
            path.append((r, turn_col))
        cur_row = next_row
        cur_col = turn_col

    for c in range(cur_col + 1, width):
        path.append((cur_row, c))
    return path


def _marker_sequence(
    palette_indices: list[int], max_markers: int, rng: random.Random
) -> list[int] | None:
    used_count = rng.randint(1, len(palette_indices))
    used_colors = rng.sample(palette_indices, used_count)
    marker_count = rng.randint(used_count, max_markers)
    seq = list(used_colors)
    while len(seq) < marker_count:
        seq.append(rng.choice(used_colors))
    rng.shuffle(seq)
    return seq


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_count = rng.randint(2, 4)
    upper_h = rng.randint(7, 11)
    lower_h = 2 * palette_count + rng.randint(1, 4)
    width = rng.randint(12, 20)
    height = upper_h + 1 + lower_h

    grid = np.full((height, width), BLACK, dtype=int)
    sep_row = upper_h
    grid[sep_row, :] = GREY

    path = _build_path(upper_h, width, rng)
    if path is None:
        return None
    for r, c in path:
        grid[r, c] = GREY

    max_markers = min(5, max(1, (len(path) - 1) // 2))
    palette_indices = rng.sample(list(ACTIVE_COLORS), palette_count)
    remaining = [c for c in ACTIVE_COLORS if c not in palette_indices]
    paint_colors = rng.sample(remaining, palette_count)
    palette = dict(zip(palette_indices, paint_colors))

    marker_colors = _marker_sequence(palette_indices, max_markers, rng)
    if marker_colors is None:
        return None
    marker_positions = _sample_spaced_positions(1, len(path) - 2, len(marker_colors), 2, rng)
    if marker_positions is None:
        return None
    for pos, color in zip(marker_positions, marker_colors):
        r, c = path[pos]
        grid[r, c] = color

    palette_rows = _sample_spaced_positions(0, lower_h - 1, palette_count, 2, rng)
    if palette_rows is None:
        return None
    for offset_r, index_color, paint_color in zip(palette_rows, palette_indices, paint_colors):
        row = sep_row + 1 + offset_r
        row_len = rng.randint(2, min(5, width - 2))
        start_col = rng.randint(1, width - row_len - 1)
        grid[row, start_col] = index_color
        grid[row, start_col + 1 : start_col + row_len] = paint_color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
