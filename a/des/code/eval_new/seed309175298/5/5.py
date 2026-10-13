from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Generator bounds.
GEN_ATTEMPTS = 400
COLOR_COUNT_RANGE = (3, 5)
EXTRA_RELATION_PROB = 0.6
MAX_EXTRA_RELATIONS = 3
POSITION_RETRIES = 120
ROW_COL_MIN_DELTA = 2
LEGEND_COL_MIN_DELTA = 2
LEGEND_ROW_MIN_DELTA = 3


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _background(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _column_pairs(line: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], np.ndarray] | None:
    pairs: list[tuple[int, int]] = []
    used = np.zeros(line.shape[0], dtype=bool)
    for start, end in _runs(line != bg):
        length = end - start + 1
        if length < 2 or length % 2 != 0:
            return None
        for r in range(start, end + 1, 2):
            top = int(line[r])
            bottom = int(line[r + 1])
            if top == bg or bottom == bg or top == bottom:
                return None
            pairs.append((top, bottom))
            used[r : r + 2] = True
    if len(pairs) < 2:
        return None
    return pairs, used


def _find_legend(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, w = g.shape
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]], np.ndarray]] = []

    if h >= 2:
        cols = [c for c in range(w) if int(g[0, c]) != bg and int(g[1, c]) != bg and int(g[0, c]) != int(g[1, c])]
        if len(cols) >= 2 and (h == 2 or all(int(g[2, c]) == bg for c in cols)):
            mask = np.zeros_like(g, dtype=bool)
            mask[0, cols] = True
            mask[1, cols] = True
            pairs = [(int(g[0, c]), int(g[1, c])) for c in cols]
            colors = {color for pair in pairs for color in pair}
            candidates.append(((len(pairs), len(colors)), pairs, mask))

    if h >= 2:
        cols = [
            c
            for c in range(w)
            if int(g[h - 2, c]) != bg and int(g[h - 1, c]) != bg and int(g[h - 2, c]) != int(g[h - 1, c])
        ]
        if len(cols) >= 2 and (h == 2 or all(int(g[h - 3, c]) == bg for c in cols)):
            mask = np.zeros_like(g, dtype=bool)
            mask[h - 2, cols] = True
            mask[h - 1, cols] = True
            pairs = [(int(g[h - 2, c]), int(g[h - 1, c])) for c in cols]
            colors = {color for pair in pairs for color in pair}
            candidates.append(((len(pairs), len(colors)), pairs, mask))

    if w >= 1:
        parsed = _column_pairs(g[:, 0], bg)
        if parsed is not None and (w == 1 or bool(np.all(g[:, 1] == bg))):
            pairs, used = parsed
            mask = np.zeros_like(g, dtype=bool)
            mask[used, 0] = True
            colors = {color for pair in pairs for color in pair}
            candidates.append(((len(pairs), len(colors)), pairs, mask))

    if w >= 1:
        parsed = _column_pairs(g[:, w - 1], bg)
        if parsed is not None and (w == 1 or bool(np.all(g[:, w - 2] == bg))):
            pairs, used = parsed
            mask = np.zeros_like(g, dtype=bool)
            mask[used, w - 1] = True
            colors = {color for pair in pairs for color in pair}
            candidates.append(((len(pairs), len(colors)), pairs, mask))

    assert candidates, "expected an edge legend of vertical color pairs"
    score, pairs, mask = max(candidates, key=lambda item: item[0])
    assert score[0] >= 2
    return pairs, mask


def _topological_order(pairs: list[tuple[int, int]]) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for upper, lower in pairs:
        if upper not in seen:
            seen.add(upper)
            colors.append(upper)
        if lower not in seen:
            seen.add(lower)
            colors.append(lower)

    outgoing: dict[int, set[int]] = defaultdict(set)
    indeg = {color: 0 for color in colors}
    for upper, lower in pairs:
        if lower not in outgoing[upper]:
            outgoing[upper].add(lower)
            indeg[lower] += 1

    available = [color for color in colors if indeg[color] == 0]
    order: list[int] = []
    while available:
        color = available.pop(0)
        order.append(color)
        for nxt in colors:
            if nxt not in outgoing[color]:
                continue
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                available.append(nxt)

    assert len(order) == len(colors), "legend relations must define an acyclic order"
    return order


def _infer_stripes(
    g: np.ndarray,
    legend_mask: np.ndarray,
    colors: list[int],
    bg: int,
) -> tuple[dict[int, tuple[int, int, int]], dict[int, tuple[int, int, int]]]:
    horizontal: dict[int, tuple[int, int, int]] = {}
    vertical: dict[int, tuple[int, int, int]] = {}

    for color in colors:
        coords = np.argwhere((g == color) & (~legend_mask))
        assert coords.size > 0, f"missing stripe cells for color {color}"
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1:
            row = int(rows[0])
            active = np.where((g[row, :] != bg) & (~legend_mask[row, :]))[0]
            assert active.size > 0
            horizontal[color] = (row, int(active.min()), int(active.max()))
        elif len(cols) == 1:
            col = int(cols[0])
            active = np.where((g[:, col] != bg) & (~legend_mask[:, col]))[0]
            assert active.size > 0
            vertical[color] = (col, int(active.min()), int(active.max()))
        else:
            raise AssertionError("each stripe color must occupy exactly one row or one column")

    assert horizontal and vertical, "expected both horizontal and vertical stripes"
    return horizontal, vertical


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair stripe crossing colors by reading an edge legend of pairwise over-under relations.

    Input structure:
    - Black is the background.
    - Several long 1-cell-thick horizontal and vertical colored stripes cross each other, and each stripe color is unique.
    - One outer edge contains a legend made of 2x1 vertical color pairs; the upper cell of a pair names the higher layer and the lower cell names the lower layer.

    Rule:
    - Read every legend pair as "upper color is above lower color" and combine all pairs transitively into one global stripe priority order.
    - Erase the legend from the output.
    - For every horizontal/vertical crossing, repaint the crossing cell with whichever stripe color is higher in the inferred order.
    - Leave background cells and all non-crossing stripe cells unchanged.

    Key insight:
    - The visible color at a crossing is not reliable; the true occlusion order is encoded only by the edge legend and must be enforced consistently across all intersections.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    pairs, legend_mask = _find_legend(g, bg)
    order = _topological_order(pairs)
    rank = {color: idx for idx, color in enumerate(order)}
    horizontal, vertical = _infer_stripes(g, legend_mask, order, bg)

    out = g.copy()
    out[legend_mask] = bg
    for h_color, (row, c0, c1) in horizontal.items():
        for v_color, (col, r0, r1) in vertical.items():
            if c0 <= col <= c1 and r0 <= row <= r1:
                out[row, col] = h_color if rank[h_color] < rank[v_color] else v_color
    return out


def _sample_spaced_positions(start: int, end: int, count: int, min_delta: int) -> list[int] | None:
    values = list(range(start, end + 1))
    if len(values) < count:
        return None
    for _ in range(POSITION_RETRIES):
        chosen = sorted(random.sample(values, count))
        if all(b - a >= min_delta for a, b in zip(chosen, chosen[1:], strict=False)):
            return chosen
    return None


def _make_relations(order: list[int]) -> list[tuple[int, int]]:
    pairs = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    optional = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 2, len(order))]
    random.shuffle(optional)
    if optional and random.random() < EXTRA_RELATION_PROB:
        extra_count = random.randint(1, min(len(optional), MAX_EXTRA_RELATIONS))
        pairs.extend(optional[:extra_count])
    random.shuffle(pairs)
    return pairs


def _draw_specs(
    shape: tuple[int, int],
    order: list[int],
    horizontal: dict[int, tuple[int, int, int]],
    vertical: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in reversed(order):
        if color in horizontal:
            row, c0, c1 = horizontal[color]
            out[row, c0 : c1 + 1] = color
        else:
            col, r0, r1 = vertical[color]
            out[r0 : r1 + 1, col] = color
    return out


def _place_legend(grid: np.ndarray, side: str, pairs: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    if side in {"top", "bottom"}:
        cols = _sample_spaced_positions(0, w - 1, len(pairs), LEGEND_COL_MIN_DELTA)
        if cols is None:
            return False
        top_row = 0 if side == "top" else h - 2
        bottom_row = 1 if side == "top" else h - 1
        for col, (upper, lower) in zip(cols, pairs, strict=False):
            grid[top_row, col] = upper
            grid[bottom_row, col] = lower
        return True

    rows = _sample_spaced_positions(0, h - 2, len(pairs), LEGEND_ROW_MIN_DELTA)
    if rows is None:
        return False
    col = 0 if side == "left" else w - 1
    for row, (upper, lower) in zip(rows, pairs, strict=False):
        grid[row, col] = upper
        grid[row + 1, col] = lower
    return True


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(PALETTE, color_count)
    order = random.sample(colors, len(colors))
    h_count = random.randint(1, color_count - 1)
    h_colors = set(random.sample(colors, h_count))
    v_colors = [color for color in colors if color not in h_colors]
    pairs = _make_relations(order)
    side = random.choice(["top", "bottom", "left", "right"])

    top_pad = 3 if side == "top" else 1
    bottom_pad = 3 if side == "bottom" else 1
    left_pad = 2 if side == "left" else 1
    right_pad = 2 if side == "right" else 1

    min_inner_h = max(6, 2 * h_count + 2)
    min_inner_w = max(7, 2 * len(v_colors) + 4)
    min_h = top_pad + min_inner_h + bottom_pad
    min_w = left_pad + min_inner_w + right_pad
    if side in {"top", "bottom"}:
        min_w = max(min_w, 2 * len(pairs) - 1)
    else:
        min_h = max(min_h, 3 * len(pairs) - 1)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, min(30, min_h + 8))
    w = random.randint(min_w, min(30, min_w + 8))
    row_start = top_pad
    row_end = h - bottom_pad - 1
    col_start = left_pad
    col_end = w - right_pad - 1

    rows = _sample_spaced_positions(row_start, row_end, h_count, ROW_COL_MIN_DELTA)
    cols = _sample_spaced_positions(col_start, col_end, len(v_colors), ROW_COL_MIN_DELTA)
    if rows is None or cols is None:
        return None

    horizontal: dict[int, tuple[int, int, int]] = {}
    vertical: dict[int, tuple[int, int, int]] = {}
    min_col = min(cols)
    max_col = max(cols)
    min_row = min(rows)
    max_row = max(rows)

    shuffled_h = random.sample(list(h_colors), len(h_colors))
    shuffled_v = random.sample(v_colors, len(v_colors))
    for color, row in zip(shuffled_h, rows, strict=False):
        c0 = random.randint(col_start, min_col)
        c1 = random.randint(max_col, col_end)
        if c0 == min_col and c1 == max_col:
            if col_start < min_col and (col_end == max_col or random.random() < 0.5):
                c0 = random.randint(col_start, min_col - 1)
            elif max_col < col_end:
                c1 = random.randint(max_col + 1, col_end)
            else:
                return None
        horizontal[color] = (row, c0, c1)
    for color, col in zip(shuffled_v, cols, strict=False):
        r0 = random.randint(row_start, min_row)
        r1 = random.randint(max_row, row_end)
        if r0 == min_row and r1 == max_row:
            if row_start < min_row and (row_end == max_row or random.random() < 0.5):
                r0 = random.randint(row_start, min_row - 1)
            elif max_row < row_end:
                r1 = random.randint(max_row + 1, row_end)
            else:
                return None
        vertical[color] = (col, r0, r1)

    expected = _draw_specs((h, w), order, horizontal, vertical)
    inp = expected.copy()
    rank = {color: idx for idx, color in enumerate(order)}
    intersections: list[tuple[int, int, int, int]] = []
    for h_color, (row, c0, c1) in horizontal.items():
        for v_color, (col, r0, r1) in vertical.items():
            if c0 <= col <= c1 and r0 <= row <= r1:
                intersections.append((row, col, h_color, v_color))
    if len(intersections) < 2:
        return None

    wrong = {idx for idx in range(len(intersections)) if random.random() < 0.5}
    if not wrong:
        wrong.add(random.randrange(len(intersections)))
    if len(wrong) == len(intersections):
        wrong.remove(random.choice(tuple(wrong)))
    for idx in wrong:
        row, col, h_color, v_color = intersections[idx]
        top = h_color if rank[h_color] < rank[v_color] else v_color
        low = v_color if top == h_color else h_color
        inp[row, col] = low

    if not _place_legend(inp, side, pairs):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-layer sample")
