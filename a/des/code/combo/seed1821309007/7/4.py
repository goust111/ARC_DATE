from __future__ import annotations

import random

import numpy as np


BG = 7  # orange
PALETTE = [c for c in range(10) if c != BG]
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_reference_columns(g: np.ndarray, bg: int) -> tuple[int, int]:
    non_bg_cols = np.any(g != bg, axis=0)
    groups: list[tuple[int, int]] = []
    c = 0
    while c < g.shape[1]:
        if not non_bg_cols[c]:
            c += 1
            continue
        c0 = c
        while c + 1 < g.shape[1] and non_bg_cols[c + 1]:
            c += 1
        groups.append((c0, c))
        c += 1
    assert groups, "expected at least one non-background column group"
    widths = [c1 - c0 + 1 for c0, c1 in groups]
    widest = max(widths)
    assert widths.count(widest) == 1, "reference pattern must be the unique widest column group"
    return groups[widths.index(widest)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read right-side 2x2 color instructions as row/column range pointers into a left-side reference pattern.

    The first block in each instruction row selects the row range spanned by that color in the reference pattern.
    The second block selects the column range spanned by its color in the reference pattern.
    Multiple instruction rows intersect their selected row and column ranges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(np.bincount(g.ravel()).argmax())

    ref_c0, ref_c1 = _find_reference_columns(g, bg)
    ref_rows = np.argwhere(np.any(g[:, ref_c0 : ref_c1 + 1] != bg, axis=1)).ravel()
    assert ref_rows.size > 0, "reference pattern must be present"
    ref_r0, ref_r1 = int(ref_rows[0]), int(ref_rows[-1])
    ref = g[ref_r0 : ref_r1 + 1, ref_c0 : ref_c1 + 1]

    color_ranges: dict[int, tuple[int, int, int, int]] = {}
    for color in np.unique(ref):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(ref == color)
        if coords.size == 0:
            continue
        color_ranges[color] = _bbox(coords)
    assert color_ranges, "reference pattern must contain non-background colors"

    right = g[:, ref_c1 + 1 :]
    comps = []
    for comp in _components(right != bg):
        coords = np.array(comp, dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        patch = right[r0 : r1 + 1, c0 : c1 + 1]
        if patch.shape == (2, 2) and np.unique(patch).size == 1:
            comps.append((r0, c0, r1, c1, int(patch[0, 0])))
    assert comps, "expected 2x2 instruction blocks to the right of the reference pattern"

    by_row: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = {}
    for item in comps:
        key = (item[0], item[2])
        by_row.setdefault(key, []).append(item)

    instr_rows: list[tuple[int, int]] = []
    for key in sorted(by_row):
        blocks = sorted(by_row[key], key=lambda x: x[1])
        assert len(blocks) == 2, "each instruction row must contain exactly two 2x2 blocks"
        row_color = blocks[0][4]
        col_color = blocks[1][4]
        assert row_color in color_ranges and col_color in color_ranges
        instr_rows.append((row_color, col_color))
    assert instr_rows

    out_r0, out_r1 = 0, ref.shape[0] - 1
    out_c0, out_c1 = 0, ref.shape[1] - 1
    for row_color, col_color in instr_rows:
        rr0, _, rr1, _ = color_ranges[row_color]
        _, cc0, _, cc1 = color_ranges[col_color]
        out_r0 = max(out_r0, rr0)
        out_r1 = min(out_r1, rr1)
        out_c0 = max(out_c0, cc0)
        out_c1 = min(out_c1, cc1)

    assert out_r0 <= out_r1 and out_c0 <= out_c1, "instruction ranges must have a non-empty intersection"
    return ref[out_r0 : out_r1 + 1, out_c0 : out_c1 + 1]


def _rand_interval(length: int, min_size: int, max_size: int) -> tuple[int, int]:
    size = random.randint(min_size, min(max_size, length))
    start = random.randint(0, length - size)
    return start, start + size - 1


def _super_interval(base: tuple[int, int], limit: int) -> tuple[int, int]:
    a, b = base
    left = random.randint(0, a)
    right = random.randint(b, limit - 1)
    return left, right


def _build_intervals(target: tuple[int, int], limit: int, count: int) -> list[tuple[int, int]]:
    intervals = [target]
    while len(intervals) < count:
        mode = random.randint(0, 2)
        if mode == 0:
            cand = (random.randint(0, target[0]), target[1])
        elif mode == 1:
            cand = (target[0], random.randint(target[1], limit - 1))
        else:
            cand = _super_interval(target, limit)
        if cand not in intervals:
            intervals.append(cand)
    random.shuffle(intervals)
    return intervals


def _paint_reference(
    h: int,
    w: int,
    target_rows: tuple[int, int],
    target_cols: tuple[int, int],
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
) -> tuple[np.ndarray, list[int], list[int]]:
    grid = np.empty((h, w), dtype=int)
    filler = random.sample(PALETTE, k=random.randint(2, min(4, len(PALETTE))))
    for r in range(h):
        for c in range(w):
            grid[r, c] = random.choice(filler)

    needed = len(row_intervals) + len(col_intervals)
    colors = random.sample(PALETTE, k=needed + 1)
    target_color = colors[0]
    row_colors = colors[1 : 1 + len(row_intervals)]
    col_colors = colors[1 + len(row_intervals) :]

    tr0, tr1 = target_rows
    tc0, tc1 = target_cols
    grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = target_color

    available_cols = list(range(w))
    random.shuffle(available_cols)
    if len(available_cols) < len(row_intervals):
        raise RuntimeError("not enough reference columns for row markers")
    for color, (r0, r1) in zip(row_colors, row_intervals):
        c = available_cols.pop()
        grid[r0 : r1 + 1, c] = color
        if w >= 2 and random.random() < 0.5:
            c2 = min(w - 1, c + 1 if c + 1 not in available_cols else c + 1)
            grid[r0 : r1 + 1, c2] = color

    available_rows = list(range(h))
    random.shuffle(available_rows)
    if len(available_rows) < len(col_intervals):
        raise RuntimeError("not enough reference rows for column markers")
    for color, (c0, c1) in zip(col_colors, col_intervals):
        r = available_rows.pop()
        grid[r, c0 : c1 + 1] = color
        if h >= 2 and random.random() < 0.5:
            r2 = min(h - 1, r + 1 if r + 1 not in available_rows else r + 1)
            grid[r2, c0 : c1 + 1] = color

    return grid, row_colors, col_colors


def _place_instruction_rows(
    canvas: np.ndarray,
    top: int,
    left: int,
    pairs: list[tuple[int, int]],
) -> None:
    r = top
    for row_color, col_color in pairs:
        canvas[r : r + 2, left : left + 2] = row_color
        gap = random.randint(1, 3)
        c2 = left + 2 + gap
        canvas[r : r + 2, c2 : c2 + 2] = col_color
        r += 2 + random.randint(1, 3)


def _make_case() -> np.ndarray:
    ref_h = random.randint(8, 15)
    ref_w = random.randint(8, 15)
    target_rows = _rand_interval(ref_h, 2, max(2, ref_h // 2 + 1))
    target_cols = _rand_interval(ref_w, 2, max(2, ref_w // 2 + 1))

    instr_count = random.randint(1, 4)
    row_intervals = _build_intervals(target_rows, ref_h, instr_count)
    col_intervals = _build_intervals(target_cols, ref_w, instr_count)

    ref, row_colors, col_colors = _paint_reference(
        ref_h, ref_w, target_rows, target_cols, row_intervals, col_intervals
    )

    row_order = list(range(instr_count))
    random.shuffle(row_order)
    pairs = [(row_colors[i], col_colors[j]) for i, j in zip(row_order, reversed(row_order))]

    instr_h = 2 * instr_count + sum(random.randint(1, 3) for _ in range(instr_count - 1)) if instr_count > 1 else 2
    instr_w = 2 + random.randint(1, 3) + 2
    top_margin = random.randint(0, 3)
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 3)
    sep = random.randint(2, 5)

    height = max(ref_h + top_margin + bottom_margin, instr_h + top_margin + bottom_margin)
    width = left_margin + ref_w + sep + instr_w + right_margin
    canvas = np.full((height, width), BG, dtype=int)

    ref_top = random.randint(0, height - ref_h)
    ref_left = left_margin
    canvas[ref_top : ref_top + ref_h, ref_left : ref_left + ref_w] = ref

    instr_top = random.randint(0, height - instr_h)
    instr_left = ref_left + ref_w + sep
    _place_instruction_rows(canvas, instr_top, instr_left, pairs)

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _make_case()
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.size == 0:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
