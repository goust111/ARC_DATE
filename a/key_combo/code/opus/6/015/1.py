from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 7
PARTICLE = 6
_DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
_PERP = {
    "right": ("up", "down"),
    "left": ("up", "down"),
    "up": ("left", "right"),
    "down": ("left", "right"),
}


def _mode_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str, bg: int) -> bool:
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if int(grid[r, c]) not in {bg, PARTICLE}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return False
        a = int(grid[r - 1, c])
        b = int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return False
        a = int(grid[r, c - 1])
        b = int(grid[r, c + 1])
    return a == b and a not in {bg, PARTICLE}


def _find_legend(grid: np.ndarray, bg: int) -> list[int]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int]]] = []
    for c in range(w):
        vals: list[int] = []
        ok = True
        for r in range(h):
            v = int(grid[r, c])
            if v in {bg, PARTICLE}:
                break
            if c > 0 and int(grid[r, c - 1]) != bg:
                ok = False
                break
            if c + 1 < w and int(grid[r, c + 1]) != bg:
                ok = False
                break
            vals.append(v)
        if not ok:
            continue
        k = len(vals)
        if 2 <= k <= 4 and len(set(vals)) == k:
            if k < h and int(grid[k, c]) != bg:
                continue
            candidates.append((c, vals))
    assert candidates, "legend not found"
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _trace_path(grid: np.ndarray, start_r: int, start_c: int, bg: int) -> tuple[list[tuple[int, int]], int]:
    h, w = grid.shape
    r, c = int(start_r), int(start_c)
    direction = "right"
    path = [(r, c)]
    turns = 0
    for _ in range(h * w * 4):
        dr, dc = _DIRS[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(grid[nr, nc]) in {bg, PARTICLE}:
            r, c = nr, nc
            path.append((r, c))
            continue
        options: list[tuple[int, int, str]] = []
        for turn_dir in _PERP[direction]:
            tr = r + _DIRS[turn_dir][0]
            tc = c + _DIRS[turn_dir][1]
            if _has_corridor(grid, tr, tc, turn_dir, bg):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn"
        if not options:
            break
        r, c, direction = options[0]
        path.append((r, c))
        turns += 1
    return path, turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each magenta path by indexing the top-right legend with its turn count.

    Input structure:
    - Background is orange-like and the movers are magenta pixels in column 0.
    - Colored horizontal strips on alternating rows form corridor walls; a short vertical legend of single pixels sits near the top-right.
    - Each particle starts on a bg cell row and the legend colors are stacked from top to bottom.

    Rule:
    - Trace every particle from left to right through bg cells.
    - When the next cell is blocked, turn 90 degrees into the unique perpendicular bg cell whose two side walls share one color; stop at an edge or dead end.
    - Count the number of turns t, look up the t-th legend color from the top, and repaint every visited cell of that particle with the legend color.

    Key insight:
    - The corridor geometry determines a path's turn count, and that turn count is used as a zero-based index into the isolated legend column.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    legend = _find_legend(g, bg)
    out = g.copy()

    particle_rows = sorted(int(r) for r in range(g.shape[0]) if int(g[r, 0]) == PARTICLE)
    for r in particle_rows:
        path, turns = _trace_path(g, r, 0, bg)
        assert 0 <= turns < len(legend), "turn count exceeds legend"
        color = int(legend[turns])
        for rr, cc in path:
            out[rr, cc] = color
    return out


def _paint(row: np.ndarray, c0: int, c1: int, color: int) -> None:
    if c0 <= c1:
        row[c0 : c1 + 1] = int(color)


def _build_upper_band(width: int, color: int, turns: int, horiz_dir: int, x1: int, x2: int) -> np.ndarray:
    patch = np.full((5, width), BG, dtype=int)
    top, upper, mid, lower, bottom = 0, 1, 2, 3, 4
    patch[upper, 0] = PARTICLE

    if turns == 0:
        end = x1
        _paint(patch[top], 1, end + 1, color)
        _paint(patch[mid], 1, end + 1, color)
        patch[upper, end + 1] = color
        return patch

    _paint(patch[top], 1, x1 + 1, color)

    if turns == 1:
        _paint(patch[mid], 1, x1 + 1, color)
        patch[mid, x1] = BG
        patch[lower, x1] = color
        return patch

    if horiz_dir > 0:
        support_lo = x1 + 1
        support_hi = x2 + 1
        block_col = x2 + 1
        patch[lower, x1 - 1] = color
    else:
        support_lo = x2 - 1
        support_hi = x1 - 1
        block_col = x2 - 1
        patch[lower, x1 + 1] = color

    _paint(patch[mid], 1, x1 + 1, color)
    _paint(patch[mid], support_lo, support_hi, color)
    _paint(patch[bottom], min(x1, support_lo), max(x1, support_hi), color)
    patch[mid, x1] = BG
    patch[lower, block_col] = color

    if turns == 2:
        return patch

    patch[upper, x2] = color
    patch[mid, x2] = BG
    return patch


def _build_band(width: int, color: int, turns: int, start_lower: bool, horiz_dir: int, x1: int, x2: int) -> np.ndarray:
    patch = _build_upper_band(width, color, turns, horiz_dir, x1, x2)
    if start_lower:
        patch = np.flipud(patch)
    return patch


def _choose_turn_profile(width: int, turns: int) -> tuple[bool, int, int, int]:
    start_lower = bool(random.getrandbits(1))
    if turns == 0:
        end = random.randint(4, width - 5)
        return start_lower, 1, end, end
    if turns == 1:
        x1 = random.randint(4, width - 6)
        return start_lower, 1, x1, x1

    horiz_dir = random.choice((-1, 1))
    if horiz_dir > 0:
        x1 = random.randint(4, width - 9)
        x2 = random.randint(x1 + 3, width - 5)
    else:
        x1 = random.randint(6, width - 6)
        x2 = random.randint(2, x1 - 3)
    return start_lower, horiz_dir, x1, x2


def _available_colors(used: set[int], count: int) -> list[int]:
    pool = [c for c in range(10) if c not in used]
    random.shuffle(pool)
    return pool[:count]


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.choice([2, 3, 4])
    n_particles = random.randint(max(3, legend_len), 5)
    height = 6 * n_particles - 1
    width = random.randint(18, 25)
    grid = np.full((height, width), BG, dtype=int)

    used = {BG, PARTICLE}
    legend_colors = _available_colors(used, legend_len)
    if len(legend_colors) != legend_len:
        return None
    used.update(legend_colors)

    corridor_palette = [c for c in range(10) if c not in used]
    random.shuffle(corridor_palette)
    if len(corridor_palette) < 2:
        return None

    turn_counts = list(range(legend_len))
    while len(turn_counts) < n_particles:
        turn_counts.append(random.randrange(legend_len))
    random.shuffle(turn_counts)

    prev_color = None
    for idx, turns in enumerate(turn_counts):
        base = 6 * idx
        choices = [c for c in corridor_palette if c != prev_color]
        color = random.choice(choices or corridor_palette)
        prev_color = color
        start_lower, horiz_dir, x1, x2 = _choose_turn_profile(width, turns)
        band = _build_band(width, color, turns, start_lower, horiz_dir, x1, x2)
        grid[base : base + 5] = np.where(band != BG, band, grid[base : base + 5])

    legend_col = width - 1
    grid[:legend_len, legend_col] = np.array(legend_colors, dtype=int)
    grid[: legend_len + 1, width - 2] = BG
    if width - 3 >= 0:
        grid[: legend_len + 1, width - 3] = BG

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if len({tuple(row) for row in out}) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
