from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _parse_legend_row(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int]]]:
    h, w = grid.shape
    best: tuple[int, list[tuple[int, int, int]]] | None = None

    for r in range(h):
        c = 0
        entries: list[tuple[int, int, int]] = []
        valid = True
        saw_grey = False
        while c < w and int(grid[r, c]) != BG:
            if int(grid[r, c]) == GREY:
                valid = False
                break
            if c + 1 >= w:
                valid = False
                break
            a = int(grid[r, c])
            b = int(grid[r, c + 1])
            if a in (BG, GREY) or b in (BG, GREY):
                valid = False
                break
            entries.append((c, a, b))
            c += 2
            if c < w and int(grid[r, c]) == GREY:
                saw_grey = True
                c += 1
            elif c < w and int(grid[r, c]) != BG:
                valid = False
                break
        if valid and entries and (saw_grey or len(entries) == 1):
            if best is None or len(entries) > len(best[1]) or (len(entries) == len(best[1]) and r > best[0]):
                best = (r, entries)

    assert best is not None, "expected a grey-separated legend row"
    return best


def _is_path_color(v: int, legend_colors: set[int]) -> bool:
    return v not in (BG, GREY) and v in legend_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat colored 1-pixel paths as woven bands and flip every over/under relation.

    The bottom legend row lists crossing orders as adjacent color pairs separated by grey.
    Each pair (A, B) means that A is currently shown on top of B at every A/B crossing.
    The output reverses those relations: every crossing pixel switches to the hidden color,
    and every legend pair is reversed as well.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 4 and w >= 4

    legend_row, entries = _parse_legend_row(g)
    legend_colors = {int(a) for _, a, _ in entries} | {int(b) for _, _, b in entries}

    out = g.copy()

    for c0, a, b in entries:
        out[legend_row, c0] = b
        out[legend_row, c0 + 1] = a

    for r in range(h):
        if r == legend_row:
            continue
        for c in range(w):
            cur = int(g[r, c])
            if not _is_path_color(cur, legend_colors):
                continue

            horiz = None
            vert = None
            if 0 < c < w - 1:
                left = int(g[r, c - 1])
                right = int(g[r, c + 1])
                if left == right and _is_path_color(left, legend_colors):
                    horiz = left
            if 0 < r < h - 1 and r - 1 != legend_row and r + 1 != legend_row:
                up = int(g[r - 1, c])
                down = int(g[r + 1, c])
                if up == down and _is_path_color(up, legend_colors):
                    vert = up

            if horiz is not None and vert is not None and horiz != vert:
                if cur == horiz:
                    out[r, c] = vert
                elif cur == vert:
                    out[r, c] = horiz

    return out


def _sample_positions(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    vals = list(range(low, high + 1))
    random.shuffle(vals)
    chosen: list[int] = []
    for v in vals:
        if all(abs(v - x) >= min_gap for x in chosen):
            chosen.append(v)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _choose_counts(h: int, w: int) -> tuple[int, int] | None:
    options = []
    for nh in range(1, 4):
        for nv in range(1, 4):
            pair_count = nh * nv
            legend_len = 3 * pair_count - 1
            if h >= 8 and w >= legend_len + nv + 5 and nh <= (h - 4 + 1) // 2:
                options.append((nh, nv))
    return random.choice(options) if options else None


def _build_input() -> np.ndarray | None:
    h = random.randint(12, 20)
    w = random.randint(16, 28)
    counts = _choose_counts(h, w)
    if counts is None:
        return None
    nh, nv = counts
    pair_count = nh * nv
    legend_len = 3 * pair_count - 1
    legend_end = legend_len - 1

    colors = random.sample(PALETTE, nh + nv)
    h_colors = colors[:nh]
    v_colors = colors[nh:]

    rows = _sample_positions(2, h - 3, nh, 2)
    cols = _sample_positions(legend_end + 2, w - 2, nv, 2)
    if rows is None or cols is None:
        return None

    grid = np.full((h, w), BG, dtype=int)

    for row, color in zip(rows, h_colors):
        grid[row, :] = color
    for col, color in zip(cols, v_colors):
        grid[:, col] = color

    legend_entries: list[tuple[int, int]] = []
    for h_color in h_colors:
        for v_color in v_colors:
            if random.random() < 0.5:
                top, bottom = h_color, v_color
            else:
                top, bottom = v_color, h_color
            legend_entries.append((top, bottom))
            rr = rows[h_colors.index(h_color)]
            cc = cols[v_colors.index(v_color)]
            grid[rr, cc] = top

    random.shuffle(legend_entries)
    c = 0
    for i, (a, b) in enumerate(legend_entries):
        grid[h - 1, c] = a
        grid[h - 1, c + 1] = b
        c += 2
        if i != len(legend_entries) - 1:
            grid[h - 1, c] = GREY
            c += 1

    # Keep the legend row isolated from vertical paths.
    for col in cols:
        assert col > legend_end

    # Avoid a monochrome legend by ensuring both top-order branches appear when there is more than one crossing.
    if len(legend_entries) > 1:
        tops = {a for a, _ in legend_entries}
        if len(tops) == 1:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_input()
        if grid is None:
            continue
        out = main(grid)
        if not np.array_equal(out, grid):
            return grid
    raise RuntimeError("failed to generate a valid woven-path input")
