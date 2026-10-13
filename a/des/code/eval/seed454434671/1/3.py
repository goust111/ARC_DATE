import random
from typing import Dict, List, Optional, Tuple

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 200


def _scan_corner_legend(
    grid: np.ndarray,
    *,
    top: bool,
    left: bool,
) -> Optional[Tuple[Dict[frozenset[int], int], np.ndarray]]:
    h, w = grid.shape
    if w < 3:
        return None

    cols = list(range(0, 3)) if left else list(range(w - 3, w))
    row_order = range(h) if top else range(h - 1, -1, -1)

    entries: List[Tuple[int, int, int]] = []
    entry_rows: List[int] = []
    expect_entry = True

    for r in row_order:
        vals = [int(grid[r, c]) for c in cols]
        if expect_entry:
            if all(v != BG for v in vals):
                entries.append(tuple(vals))
                entry_rows.append(int(r))
                expect_entry = False
                continue
            break
        if all(v == BG for v in vals):
            expect_entry = True
            continue
        break

    if len(entries) < 2:
        return None

    legend: Dict[frozenset[int], int] = {}
    mask = np.zeros(grid.shape, dtype=bool)
    for r, (a, b, c) in zip(entry_rows, entries, strict=False):
        key = frozenset((int(a), int(b)))
        assert len(key) == 2, "legend path colors must be distinct"
        assert int(c) != BG, "legend result color must be non-background"
        assert key not in legend, "legend pairs must be unique"
        legend[key] = int(c)
        for c0 in cols:
            mask[r, c0] = True

    return legend, mask


def _find_legend(grid: np.ndarray) -> Tuple[Dict[frozenset[int], int], np.ndarray]:
    candidates = []
    for top, left in ((True, True), (True, False), (False, True), (False, False)):
        candidate = _scan_corner_legend(grid, top=top, left=left)
        if candidate is not None:
            candidates.append(candidate)
    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every path crossing by looking up the mixed color in the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    legend, legend_mask = _find_legend(g)

    out = g.copy()
    crossing_count = 0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if legend_mask[r, c]:
                continue

            left = int(g[r, c - 1])
            right = int(g[r, c + 1])
            up = int(g[r - 1, c])
            down = int(g[r + 1, c])
            if left == right != BG and up == down != BG and left != up:
                center = int(g[r, c])
                assert center in (left, up), "crossing cell must show one of the two path colors"
                key = frozenset((left, up))
                assert key in legend, "every crossing pair must appear in the legend"
                out[r, c] = legend[key]
                crossing_count += 1

    assert crossing_count > 0, "expected at least one crossing"
    return out


def _legend_rows(total_h: int, pair_count: int, top: bool) -> List[int]:
    legend_h = 2 * pair_count - 1
    if top:
        start = 0
    else:
        start = total_h - legend_h
    return [start + 2 * k for k in range(pair_count)]


def _gen_attempt() -> Optional[np.ndarray]:
    h_count, v_count = random.choice(((1, 2), (2, 1), (2, 2)))
    pair_count = h_count * v_count
    legend_h = 2 * pair_count - 1
    corner = random.choice(("tl", "tr", "bl", "br"))
    top = corner[0] == "t"
    left = corner[1] == "l"

    height = random.randint(legend_h + 7, 18)
    width = random.randint(10, 18)

    row_low = legend_h + 1 if top else 1
    row_high = height - legend_h - 2 if not top else height - 2
    col_low = 4 if left else 1
    col_high = width - 5 if not left else width - 2
    if row_high - row_low + 1 < 5 or col_high - col_low + 1 < 5:
        return None

    box_h = random.randint(5, row_high - row_low + 1)
    box_w = random.randint(5, col_high - col_low + 1)
    r0 = random.randint(row_low, row_high - box_h + 1)
    c0 = random.randint(col_low, col_high - box_w + 1)
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1

    if box_h - 2 < h_count or box_w - 2 < v_count:
        return None

    colors = random.sample(PALETTE, h_count + v_count + pair_count)
    h_colors = colors[:h_count]
    v_colors = colors[h_count : h_count + v_count]
    mix_colors = colors[h_count + v_count :]

    grid = np.zeros((height, width), dtype=int)

    path_rows = sorted(random.sample(range(r0 + 1, r1), h_count))
    path_cols = sorted(random.sample(range(c0 + 1, c1), v_count))

    pair_to_mix: Dict[frozenset[int], int] = {}
    pair_rows: List[Tuple[int, int, int]] = []
    mix_index = 0
    for h_color in h_colors:
        for v_color in v_colors:
            mix_color = int(mix_colors[mix_index])
            mix_index += 1
            key = frozenset((int(h_color), int(v_color)))
            pair_to_mix[key] = mix_color
            if random.random() < 0.5:
                pair_rows.append((int(h_color), int(v_color), mix_color))
            else:
                pair_rows.append((int(v_color), int(h_color), mix_color))
    random.shuffle(pair_rows)

    legend_rows = _legend_rows(height, pair_count, top)
    legend_cols = list(range(0, 3)) if left else list(range(width - 3, width))
    for row, (a, b, c) in zip(legend_rows, pair_rows, strict=False):
        grid[row, legend_cols[0]] = a
        grid[row, legend_cols[1]] = b
        grid[row, legend_cols[2]] = c

    for row, color in zip(path_rows, h_colors, strict=False):
        grid[row, c0 : c1 + 1] = int(color)
    for col, color in zip(path_cols, v_colors, strict=False):
        grid[r0 : r1 + 1, col] = int(color)

    crossings = []
    for row, h_color in zip(path_rows, h_colors, strict=False):
        for col, v_color in zip(path_cols, v_colors, strict=False):
            visible = int(random.choice((h_color, v_color)))
            grid[row, col] = visible
            crossings.append((row, col, int(h_color), int(v_color)))

    try:
        out = main(grid)
    except AssertionError:
        return None

    if int(np.count_nonzero(out != grid)) != len(crossings):
        return None

    for row, col, h_color, v_color in crossings:
        expected = pair_to_mix[frozenset((h_color, v_color))]
        if int(out[row, col]) != expected:
            return None

    return grid


def generate_input() -> np.ndarray:
    """
    Generate a random line-network input whose crossings are fully specified by a corner legend.
    """
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
