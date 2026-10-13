from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _find_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Find the isolated vertical multicolor legend and return (priority, mask)."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best: tuple[tuple[int, int, int, int, int], int, int, int] | None = None

    for c in range(w):
        r = 0
        while r < h:
            if int(g[r, c]) == BG:
                r += 1
                continue
            r0 = r
            while r < h and int(g[r, c]) != BG:
                r += 1
            r1 = r - 1
            length = int(r1 - r0 + 1)
            if length < 2:
                continue

            colors = [int(g[rr, c]) for rr in range(r0, r1 + 1)]
            if len(set(colors)) != length:
                continue
            if r0 > 0 and int(g[r0 - 1, c]) != BG:
                continue
            if r1 + 1 < h and int(g[r1 + 1, c]) != BG:
                continue

            isolated = True
            for rr in range(r0, r1 + 1):
                if c > 0 and int(g[rr, c - 1]) != BG:
                    isolated = False
                    break
                if c + 1 < w and int(g[rr, c + 1]) != BG:
                    isolated = False
                    break
            if not isolated:
                continue

            supported = 0
            for rr, color in enumerate(colors, start=r0):
                found_elsewhere = False
                for r2 in range(h):
                    for c2 in range(w):
                        if r2 == rr and c2 == c:
                            continue
                        if int(g[r2, c2]) == color:
                            found_elsewhere = True
                            break
                    if found_elsewhere:
                        break
                if found_elsewhere:
                    supported += 1

            edge_touch = int(c == w - 1 or r1 == h - 1)
            score = (supported, edge_touch, length, c, r1)
            if best is None or score > best[0]:
                best = (score, r0, r1, c)

    assert best is not None, "expected one isolated legend strip"
    _score, r0, r1, c = best
    mask = np.zeros_like(g, dtype=bool)
    mask[r0 : r1 + 1, c] = True
    priority = [int(g[r, c]) for r in range(r0, r1 + 1)]
    return priority, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Weave straight colored paths by applying the legend priority at crossings.

    The final visible pixel at a crossing is the higher-priority path color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    priority, legend_mask = _find_legend(g)
    rank = {color: idx for idx, color in enumerate(priority)}
    h, w = g.shape

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if legend_mask[r, c]:
                continue
            if int(g[r, c]) == BG:
                continue

            left = int(g[r, c - 1])
            right = int(g[r, c + 1])
            up = int(g[r - 1, c])
            down = int(g[r + 1, c])

            if left == BG or right == BG or up == BG or down == BG:
                continue
            if legend_mask[r, c - 1] or legend_mask[r, c + 1] or legend_mask[r - 1, c] or legend_mask[r + 1, c]:
                continue
            if left != right or up != down or left == up:
                continue
            if left not in rank or up not in rank:
                continue

            out[r, c] = left if rank[left] < rank[up] else up

    return out


def _sample_spaced_positions(start: int, end: int, count: int) -> list[int]:
    positions = list(range(start, end + 1))
    valid: list[list[int]] = []

    def backtrack(idx: int, chosen: list[int]) -> None:
        if len(chosen) == count:
            valid.append(chosen.copy())
            return
        if idx >= len(positions):
            return
        remaining = count - len(chosen)
        for j in range(idx, len(positions)):
            pos = positions[j]
            if chosen and pos - chosen[-1] < 2:
                continue
            slots_left = len(positions) - j - 1
            if slots_left + 1 < remaining:
                break
            chosen.append(pos)
            backtrack(j + 1, chosen)
            chosen.pop()

    backtrack(0, [])
    assert valid, "no spaced positions available"
    return random.choice(valid)


def _place_legend(
    grid: np.ndarray,
    priority: list[int],
    side: str,
    top: int,
    left: int,
    draw_h: int,
    draw_w: int,
    vertical_cols: list[int],
) -> None:
    h, w = grid.shape
    length = len(priority)
    if side == "right":
        col = w - 1
        start_row = random.randint(0, h - length)
        grid[start_row : start_row + length, col] = np.array(priority, dtype=int)
        return

    start_row = h - length
    blocked = set(vertical_cols)
    candidate_cols = [
        c
        for c in range(1, w - 1)
        if c not in blocked
        and (left + draw_w + 1 > w or c < left or c >= left + draw_w)
    ]
    if not candidate_cols:
        candidate_cols = [c for c in range(1, w - 1) if c not in blocked]
    assert candidate_cols, "expected at least one bottom-legend column"
    col = random.choice(candidate_cols)
    grid[start_row:, col] = np.array(priority, dtype=int)


def _render_input(
    grid_h: int,
    grid_w: int,
    top: int,
    left: int,
    draw_h: int,
    draw_w: int,
    h_rows: list[int],
    v_cols: list[int],
    h_colors: list[int],
    v_colors: list[int],
    priority: list[int],
    legend_side: str,
) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)

    for row, color in zip(h_rows, h_colors, strict=False):
        grid[row, left : left + draw_w] = int(color)
    for col, color in zip(v_cols, v_colors, strict=False):
        grid[top : top + draw_h, col] = int(color)

    crossings: list[tuple[int, int, int, int]] = []
    for row, h_color in zip(h_rows, h_colors, strict=False):
        for col, v_color in zip(v_cols, v_colors, strict=False):
            crossings.append((row, col, h_color, v_color))

    if crossings:
        if len(crossings) >= 2:
            forced = [0, 1]
            random.shuffle(forced)
            for idx, choice_idx in zip(range(2), forced, strict=False):
                row, col, h_color, v_color = crossings[idx]
                grid[row, col] = int((h_color, v_color)[choice_idx])
            start_idx = 2
        else:
            start_idx = 0
        for row, col, h_color, v_color in crossings[start_idx:]:
            grid[row, col] = int(random.choice((h_color, v_color)))

    _place_legend(grid, priority, legend_side, top, left, draw_h, draw_w, v_cols)
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_h = random.randint(2, 3)
    n_v = random.randint(2, 3)
    total_paths = n_h + n_v
    colors = random.sample(COLOR_POOL, total_paths)
    random.shuffle(colors)
    h_colors = colors[:n_h]
    v_colors = colors[n_h:]
    priority = random.sample(colors, len(colors))
    legend_side = random.choice(("right", "bottom"))

    draw_h = random.randint(max(6, 2 * n_h + 2), 10)
    draw_w = random.randint(max(6, 2 * n_v + 2), 10)
    top = random.randint(1, 3)
    left = random.randint(1, 3)

    if legend_side == "right":
        gap = random.randint(2, 4)
        grid_h = top + draw_h + random.randint(1, 3)
        grid_w = left + draw_w + gap + 1
    else:
        gap = random.randint(2, 4)
        grid_h = top + draw_h + gap + total_paths
        grid_w = left + draw_w + random.randint(1, 3)

    if grid_h > 30 or grid_w > 30:
        return None

    row_start = top + 1
    row_end = top + draw_h - 2
    col_start = left + 1
    col_end = left + draw_w - 2
    if row_end - row_start + 1 < 2 * n_h - 1 or col_end - col_start + 1 < 2 * n_v - 1:
        return None

    h_rows = _sample_spaced_positions(row_start, row_end, n_h)
    v_cols = _sample_spaced_positions(col_start, col_end, n_v)

    grid = _render_input(
        grid_h,
        grid_w,
        top,
        left,
        draw_h,
        draw_w,
        h_rows,
        v_cols,
        h_colors,
        v_colors,
        priority,
        legend_side,
    )

    legend_priority, legend_mask = _find_legend(grid)
    if legend_priority != priority:
        return None
    if np.any(grid[legend_mask] == BG):
        return None

    out = main(grid)
    expected = grid.copy()
    rank = {color: idx for idx, color in enumerate(priority)}
    for row, h_color in zip(h_rows, h_colors, strict=False):
        for col, v_color in zip(v_cols, v_colors, strict=False):
            expected[row, col] = h_color if rank[h_color] < rank[v_color] else v_color

    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
