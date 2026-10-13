from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 300


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _read_palette(grid: np.ndarray) -> tuple[int, int, int, list[int]]:
    h, w = grid.shape
    row = grid[h - 1]
    cols = np.flatnonzero(row != BG)
    assert cols.size >= 6, "palette must contain at least three 2-cell bars"

    c0 = int(cols[0])
    c1 = int(cols[-1])
    assert np.array_equal(cols, np.arange(c0, c1 + 1)), "palette must be a contiguous run"

    span = c1 - c0 + 1
    assert span % 2 == 0, "palette width must be divisible by 2"

    colors: list[int] = []
    for c in range(c0, c1 + 1, 2):
        pair = row[c : c + 2]
        color = int(pair[0])
        assert color != BG and int(pair[1]) == color, "each palette bar must be a 2-pixel solid color"
        colors.append(color)
    assert len(colors) == len(set(colors)), "palette colors must be distinct"

    palette_height = 1
    if h >= 2:
        row2 = grid[h - 2]
        same_strip = np.array_equal(row2[c0 : c1 + 1], row[c0 : c1 + 1])
        clean_margin = bool((row2[:c0] == BG).all() and (row2[c1 + 1 :] == BG).all())
        if same_strip and clean_margin:
            palette_height = 2

    palette_top = h - palette_height
    return palette_top, c0, c1, colors


def _frame_specs(grid: np.ndarray, palette_top: int) -> list[tuple[int, int, int, int]]:
    search = grid[:palette_top, :]
    comps = _components4(search != BG)
    frames: list[tuple[int, int, int, int]] = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if inner_h < 2 or inner_w < 1:
            continue

        color = int(search[comp[0]])
        if any(int(search[r, c]) != color for r, c in comp):
            continue

        patch = search[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(patch[border] == color):
            continue
        if np.any(patch[~border] != BG):
            continue
        if len(comp) != int(border.sum()):
            continue
        frames.append((r0, c0, r1, c1))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame using a bottom palette indexed by inner width.

    Input structure:
    - The background is black.
    - The bottom one or two rows contain a palette made of contiguous 2-pixel-wide horizontal bars with distinct colors.
    - Above the palette are several one-pixel-thick hollow rectangular frames of a single color.
    - Each frame has a unique inner width between 1 and the number of palette bars, and inner height at least 2.

    Rule:
    - Keep every existing frame pixel and every palette pixel unchanged.
    - For each frame, measure its inner width W and select the W-th palette bar from the left.
    - If the frame's inner height is odd, fill the entire middle interior row with that palette color.
    - If the frame's inner height is even, fill the interior leftmost and rightmost columns with that palette color.
    - Leave all remaining interior cells black.

    Key insight:
    - The palette is not matched by position: the frame's inner width chooses the color, while the frame's inner-height parity chooses the fill pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    palette_top, _c0, _c1, palette_colors = _read_palette(g)
    frames = _frame_specs(g, palette_top)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert 1 <= inner_w <= len(palette_colors)
        fill_color = int(palette_colors[inner_w - 1])
        if inner_h % 2 == 1:
            mid_r = r0 + 1 + inner_h // 2
            out[mid_r, c0 + 1 : c1] = fill_color
        else:
            out[r0 + 1 : r1, c0 + 1] = fill_color
            out[r0 + 1 : r1, c1 - 1] = fill_color
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, color: int) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _expected_output(
    grid: np.ndarray,
    placements: list[tuple[int, int]],
    inner_heights: list[int],
    inner_widths: list[int],
    palette_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for (r0, c0), inner_h, inner_w in zip(placements, inner_heights, inner_widths):
        fill_color = int(palette_colors[inner_w - 1])
        if inner_h % 2 == 1:
            mid_r = r0 + 1 + inner_h // 2
            out[mid_r, c0 + 1 : c0 + 1 + inner_w] = fill_color
        else:
            out[r0 + 1 : r0 + 1 + inner_h, c0 + 1] = fill_color
            out[r0 + 1 : r0 + 1 + inner_h, c0 + inner_w] = fill_color
    return out


def _sample_inner_heights(count: int, rng: random.Random) -> list[int]:
    odd_choices = [3, 5]
    even_choices = [2, 4, 6]
    if count % 2 == 0:
        odd_count = count // 2
    else:
        odd_count = count // 2 + rng.randint(0, 1)
    parities = [1] * odd_count + [0] * (count - odd_count)
    rng.shuffle(parities)
    heights: list[int] = []
    for is_odd in parities:
        heights.append(rng.choice(odd_choices if is_odd else even_choices))
    return heights


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(3, 5)
    palette_height = rng.choice([1, 2])

    palette_colors = rng.sample(list(range(1, 10)), n)
    frame_color = rng.choice([c for c in range(1, 10) if c not in palette_colors])

    inner_widths = list(range(1, n + 1))
    inner_heights = _sample_inner_heights(n, rng)

    max_inner_w = max(inner_widths)
    max_inner_h = max(inner_heights)
    min_w = max(12, 2 * n + 2, max_inner_w + 4)
    min_h = max(10, palette_height + max_inner_h + 4)
    max_w = min(30, max(min_w, min_w + 10))
    max_h = min(30, max(min_h, min_h + 10))
    if min_w > 30 or min_h > 30:
        return None

    grid_w = rng.randint(min_w, max_w)
    grid_h = rng.randint(min_h, max_h)
    palette_left = rng.randint(0, grid_w - 2 * n)
    palette_top = grid_h - palette_height
    top_limit = palette_top - 1
    if top_limit <= 0:
        return None

    specs = list(zip(inner_widths, inner_heights))
    order = sorted(range(n), key=lambda i: (specs[i][0] + 2) * (specs[i][1] + 2), reverse=True)

    occupied = np.zeros((top_limit, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * n
    for idx in order:
        inner_w, inner_h = specs[idx]
        outer_w = inner_w + 2
        outer_h = inner_h + 2

        row_min = 1 if top_limit - outer_h >= 2 else 0
        row_max = top_limit - outer_h - 1 if top_limit - outer_h >= 2 else top_limit - outer_h
        col_min = 1 if grid_w - outer_w >= 2 else 0
        col_max = grid_w - outer_w - 1 if grid_w - outer_w >= 2 else grid_w - outer_w
        if row_max < row_min or col_max < col_min:
            return None

        placed = False
        for _ in range(200):
            r0 = rng.randint(row_min, row_max)
            c0 = rng.randint(col_min, col_max)
            rr0 = max(0, r0 - 1)
            rr1 = min(top_limit - 1, r0 + outer_h)
            cc0 = max(0, c0 - 1)
            cc1 = min(grid_w - 1, c0 + outer_w)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    assert all(p is not None for p in placements)
    final_placements = [(int(r), int(c)) for r, c in placements if r is not None and c is not None]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for (r0, c0), inner_w, inner_h in zip(final_placements, inner_widths, inner_heights):
        _draw_frame(grid, r0, c0, inner_h, inner_w, frame_color)

    for dr in range(palette_height):
        cc = palette_left
        for color in palette_colors:
            grid[palette_top + dr, cc : cc + 2] = color
            cc += 2

    expected = _expected_output(grid, final_placements, inner_heights, inner_widths, palette_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
