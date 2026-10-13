from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
Interval = tuple[int, int]


def _runs(indices: list[int]) -> list[Interval]:
    if not indices:
        return []
    ordered = sorted(set(int(i) for i in indices))
    runs: list[Interval] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _content_intervals(length: int, separator_indices: list[int]) -> list[Interval]:
    intervals: list[Interval] = []
    start = 0
    for sep0, sep1 in _runs(separator_indices):
        if start <= sep0 - 1:
            intervals.append((start, sep0 - 1))
        start = sep1 + 1
    if start <= length - 1:
        intervals.append((start, length - 1))
    assert intervals, "expected content between separator lines"
    sizes = {end - begin + 1 for begin, end in intervals}
    assert len(sizes) == 1, "all array cells must have equal size"
    return intervals


def _background_and_intervals(g: np.ndarray) -> tuple[int, list[Interval], list[Interval]]:
    height, width = map(int, g.shape)
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in sorted(set(map(int, g.ravel()))):
        full_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        full_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if full_rows and full_cols:
            candidates.append((int(color), full_rows, full_cols))

    assert len(candidates) == 1, "expected one color to form all separator rows and columns"
    bg, full_rows, full_cols = candidates[0]
    row_intervals = _content_intervals(height, full_rows)
    col_intervals = _content_intervals(width, full_cols)
    assert len(row_intervals) >= 2 and len(col_intervals) >= 2, "expected a block array"
    return bg, row_intervals, col_intervals


def _patch(g: np.ndarray, row_interval: Interval, col_interval: Interval) -> np.ndarray:
    r0, r1 = row_interval
    c0, c1 = col_interval
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _is_center_cell(row: int, col: int, rows: int, cols: int) -> bool:
    return rows % 2 == 1 and cols % 2 == 1 and row == rows // 2 and col == cols // 2


def _parse_scene(g: np.ndarray) -> dict[str, object]:
    bg, row_intervals, col_intervals = _background_and_intervals(g)
    rows = len(row_intervals)
    cols = len(col_intervals)
    cell_h = row_intervals[0][1] - row_intervals[0][0] + 1
    cell_w = col_intervals[0][1] - col_intervals[0][0] + 1
    assert (cell_h, cell_w) == (rows, cols), "each non-solid block must fit one unscaled minimap"

    solid_colors = np.full((rows, cols), -1, dtype=int)
    non_solid: list[tuple[int, int, np.ndarray]] = []
    for rr, row_interval in enumerate(row_intervals):
        for cc, col_interval in enumerate(col_intervals):
            block = _patch(g, row_interval, col_interval)
            values = set(map(int, block.ravel()))
            if len(values) == 1:
                color = next(iter(values))
                assert color != bg, "solid blocks must not be separator background"
                solid_colors[rr, cc] = int(color)
            else:
                non_solid.append((rr, cc, block.copy()))

    assert len(non_solid) == 1, "expected exactly one minimap block"
    mini_r, mini_c, minimap = non_solid[0]
    assert not _is_center_cell(mini_r, mini_c, rows, cols), "minimap cell must not be the array center"
    assert minimap.shape == (rows, cols), "minimap dimensions must match the block array"
    assert len(set(map(int, minimap.ravel()))) > 1, "minimap block must not be solid"

    for rr in range(rows):
        for cc in range(cols):
            if (rr, cc) == (mini_r, mini_c):
                continue
            assert int(solid_colors[rr, cc]) >= 0, "all non-minimap cells must be solid"
            assert int(minimap[rr, cc]) == int(
                solid_colors[rr, cc]
            ), "minimap must match every other array position"

    target_color = int(minimap[mini_r, mini_c])
    assert target_color != bg, "minimap cannot claim separator background as a block color"

    return {
        "bg": bg,
        "rows": rows,
        "cols": cols,
        "row_intervals": row_intervals,
        "col_intervals": col_intervals,
        "cell": (mini_r, mini_c),
        "minimap": minimap,
        "target": target_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the one self-referential minimap cell by filling it with the color it assigns to its own array position.

    Input structure:
    - Equal-size solid color blocks form a row/column array separated by full background
      rows and columns.
    - Exactly one block is not solid; its pixels form a minimap with the same row and
      column count as the surrounding block array.
    - The minimap block is not the array center, and repeated block colors are allowed.

    Rule:
    - Compare each minimap pixel to the corresponding array block color.
    - All positions except the minimap's own position must match the visible solid
      blocks, so the remaining minimap pixel is the missing color claim.
    - Replace the entire non-solid block by that claimed color and leave everything
      else unchanged.

    Key insight:
    - The odd block is not noise: it is a map of the array that stores its own missing
      color only at the coordinate where the map itself sits.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    meta = _parse_scene(g)
    mini_r, mini_c = meta["cell"]  # type: ignore[assignment]
    row_intervals = meta["row_intervals"]  # type: ignore[assignment]
    col_intervals = meta["col_intervals"]  # type: ignore[assignment]
    target = int(meta["target"])

    r0, r1 = row_intervals[mini_r]  # type: ignore[index]
    c0, c1 = col_intervals[mini_c]  # type: ignore[index]
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = target
    return out


# --------------- generator ---------------

_COLORS = tuple(range(10))
_GEN_RETRIES = 400


def _mode_count(values: list[int]) -> int:
    counts = Counter(int(v) for v in values)
    return max(counts.values())


def _neighbors4(row: int, col: int, rows: int, cols: int) -> list[Coord]:
    out: list[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            out.append((rr, cc))
    return out


def _sample_layout() -> tuple[int, int, int, int]:
    options: list[tuple[int, int, int, int]] = []
    for rows in range(3, 6):
        for cols in range(3, 6):
            for gap in (1, 2):
                for border in (0, 1):
                    height = rows * rows + gap * (rows - 1) + border * 2
                    width = cols * cols + gap * (cols - 1) + border * 2
                    if max(height, width) <= 30:
                        options.append((rows, cols, gap, border))
    return random.choice(options)


def _sample_color_map(rows: int, cols: int, bg: int) -> tuple[np.ndarray, Coord] | None:
    palette_choices = [color for color in _COLORS if color != bg]
    palette_size = random.randint(3, min(6, len(palette_choices)))
    palette = random.sample(palette_choices, palette_size)

    for _ in range(500):
        values = [random.choice(palette) for _ in range(rows * cols)]
        color_map = np.array(values, dtype=int).reshape((rows, cols))
        flat = [int(v) for v in color_map.ravel()]
        counts = Counter(flat)
        if len(counts) < 3 or max(counts.values()) < 3:
            continue

        modal_count = max(counts.values())
        candidates: list[Coord] = []
        for rr in range(rows):
            for cc in range(cols):
                if (rr, cc) == (0, 0):
                    continue
                if _is_center_cell(rr, cc, rows, cols):
                    continue
                target = int(color_map[rr, cc])
                if counts[target] < 2:
                    continue
                if counts[target] == modal_count:
                    continue
                if target == int(color_map[0, 0]):
                    continue
                if any(int(color_map[nr, nc]) == target for nr, nc in _neighbors4(rr, cc, rows, cols)):
                    continue
                candidates.append((rr, cc))

        if candidates:
            return color_map, random.choice(candidates)
    return None


def _cell_bounds(index: int, cell_size: int, gap: int, border: int) -> Interval:
    start = border + index * (cell_size + gap)
    return start, start + cell_size - 1


def _render(color_map: np.ndarray, anomaly: Coord, bg: int, gap: int, border: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = map(int, color_map.shape)
    height = rows * rows + gap * (rows - 1) + border * 2
    width = cols * cols + gap * (cols - 1) + border * 2
    inp = np.full((height, width), int(bg), dtype=int)
    out = np.full((height, width), int(bg), dtype=int)

    for rr in range(rows):
        r0, r1 = _cell_bounds(rr, rows, gap, border)
        for cc in range(cols):
            c0, c1 = _cell_bounds(cc, cols, gap, border)
            color = int(color_map[rr, cc])
            out[r0 : r1 + 1, c0 : c1 + 1] = color
            if (rr, cc) == anomaly:
                inp[r0 : r1 + 1, c0 : c1 + 1] = color_map
            else:
                inp[r0 : r1 + 1, c0 : c1 + 1] = color
    return inp, out


def _single_block_repair(grid: np.ndarray, color: int) -> np.ndarray:
    meta = _parse_scene(grid)
    rr, cc = meta["cell"]  # type: ignore[assignment]
    row_intervals = meta["row_intervals"]  # type: ignore[assignment]
    col_intervals = meta["col_intervals"]  # type: ignore[assignment]
    r0, r1 = row_intervals[rr]  # type: ignore[index]
    c0, c1 = col_intervals[cc]  # type: ignore[index]
    out = grid.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, color_map: np.ndarray, anomaly: Coord) -> bool:
    try:
        solved = main(inp)
        meta = _parse_scene(inp)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    rr, cc = anomaly
    minimap = meta["minimap"]  # type: ignore[assignment]
    target = int(color_map[rr, cc])
    if int(minimap[rr, cc]) != target:
        return False

    other_values = [
        int(color_map[r, c])
        for r in range(color_map.shape[0])
        for c in range(color_map.shape[1])
        if (r, c) != anomaly
    ]
    if other_values.count(target) == 0:
        return False
    if other_values.count(target) == _mode_count(other_values):
        return False
    if target == int(minimap[0, 0]):
        return False
    if any(int(color_map[nr, nc]) == target for nr, nc in _neighbors4(rr, cc, *color_map.shape)):
        return False

    mode_candidates = [color for color, count in Counter(other_values).items() if count == _mode_count(other_values)]
    for shortcut_color in mode_candidates + [int(minimap[0, 0])]:
        if np.array_equal(_single_block_repair(inp, shortcut_color), expected):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows, cols, gap, border = _sample_layout()
    bg = random.choice(_COLORS)
    sampled = _sample_color_map(rows, cols, bg)
    if sampled is None:
        return None
    color_map, anomaly = sampled
    inp, expected = _render(color_map, anomaly, bg, gap, border)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if _semantic_checks(inp, expected, color_map, anomaly):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential minimap input")
