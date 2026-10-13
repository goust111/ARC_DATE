from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GREY = 5
BG = 0
MAX_GENERATION_ATTEMPTS = 400
NON_BG_NON_GREY = [1, 2, 3, 4, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _parse_palette_column(grid: np.ndarray, bg: int) -> tuple[list[int], int]:
    col = np.asarray(grid[:, -1], dtype=int)
    assert col[0] != bg, "palette column must start with a non-background color"

    grey_rows = np.flatnonzero(col == GREY)
    assert grey_rows.size > 0, "expected a grey marker segment in the last column"
    assert np.array_equal(grey_rows, np.arange(grey_rows[0], grey_rows[-1] + 1)), "grey marker must be contiguous"

    marker_start = int(grey_rows[0])
    marker_end = int(grey_rows[-1])
    palette = [int(v) for v in col[:marker_start]]
    assert palette, "expected at least one palette color"
    assert len(palette) == len(set(palette)), "palette colors must be distinct"
    assert all(color not in (bg, GREY) for color in palette), "palette colors must exclude bg and grey"

    prefix = np.flatnonzero(col != bg)
    assert prefix.size == marker_end + 1 and np.array_equal(prefix, np.arange(marker_end + 1)), "last column must be palette, then grey, then background"
    assert np.all(col[marker_end + 1 :] == bg), "grey segment must be followed only by background"
    return palette, int(grey_rows.size)


def _find_chambers(body: np.ndarray, bg: int, wall: int) -> list[list[tuple[int, int]]]:
    h, w = body.shape
    seen = np.zeros((h, w), dtype=bool)
    chambers: list[list[tuple[int, int]]] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(body[sr, sc]) != bg:
                continue

            comp: list[tuple[int, int]] = []
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            while queue:
                r, c = queue.popleft()
                comp.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(body[nr, nc]) != wall:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            chambers.append(comp)

    chambers.sort(key=lambda comp: min(comp))
    return chambers


def main(grid: np.ndarray) -> np.ndarray:
    """Color maze chambers in reading-order groups whose run length is given by the grey marker.

    Input structure:
    - The grid has a dominant background color and one wall color forming a maze-like chamber layout.
    - The last column is separated from the maze body by one full background column.
    - The last column contains distinct palette pixels at the top, then a contiguous grey marker segment,
      then only background pixels.

    Rule:
    - Read the grey marker height as k, the number of consecutive chambers assigned to each palette color.
    - Sort all 4-connected background chambers in the maze body by the reading order of each chamber's
      top-left cell.
    - Recolor chamber i with palette color floor(i / k), cycling when the palette is exhausted.
    - Erase the palette column markings back to the background color and keep the walls unchanged.

    Key insight:
    - The grey segment is not copied; it encodes the chamber-group rhythm, while the chamber order is
      implicit in the reading order of each chamber's anchor cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 3, "expected a 2D grid with a body, separator, and palette column"

    bg = _get_bg(g)
    palette, group_len = _parse_palette_column(g, bg)

    separator = g[:, -2]
    assert np.all(separator == bg), "expected a background separator column before the palette"

    body = g[:, :-2]
    body_colors = sorted(set(map(int, np.unique(body))) - {bg})
    assert len(body_colors) == 1, "maze body must contain exactly one wall color besides background"
    wall = body_colors[0]

    chambers = _find_chambers(body, bg, wall)
    assert chambers, "expected at least one chamber"

    out = g.copy()
    out[:, -1] = bg
    for idx, chamber in enumerate(chambers):
        color = palette[(idx // group_len) % len(palette)]
        for r, c in chamber:
            out[r, c] = color
    return out


Rect = tuple[int, int, int, int]


def _rect_h(rect: Rect) -> int:
    return rect[2] - rect[0] + 1


def _rect_w(rect: Rect) -> int:
    return rect[3] - rect[1] + 1


def _make_partitioned_body(height: int, width: int, wall: int, target: int) -> tuple[np.ndarray, list[Rect]] | None:
    grid = np.full((height, width), BG, dtype=int)
    grid[0, :] = wall
    grid[-1, :] = wall
    grid[:, 0] = wall
    grid[:, -1] = wall

    chambers: list[Rect] = [(1, 1, height - 2, width - 2)]
    for _ in range(target * 6):
        if len(chambers) >= target:
            break

        choices: list[tuple[int, str]] = []
        for idx, rect in enumerate(chambers):
            if _rect_h(rect) >= 5:
                choices.append((idx, "h"))
            if _rect_w(rect) >= 5:
                choices.append((idx, "v"))
        if not choices:
            break

        idx, orient = random.choice(choices)
        r0, c0, r1, c1 = chambers.pop(idx)
        if orient == "h":
            split = random.randint(r0 + 2, r1 - 2)
            grid[split, c0 : c1 + 1] = wall
            chambers.extend([(r0, c0, split - 1, c1), (split + 1, c0, r1, c1)])
        else:
            split = random.randint(c0 + 2, c1 - 2)
            grid[r0 : r1 + 1, split] = wall
            chambers.extend([(r0, c0, r1, split - 1), (r0, split + 1, r1, c1)])

    if len(chambers) < target:
        return None
    chambers.sort()
    return grid, chambers


def _sample_structure() -> tuple[int, int, int, bool]:
    palette_len = random.randint(2, 4)
    group_len = random.randint(1, 4)
    can_cycle = palette_len * group_len < 12
    use_cycle = can_cycle and random.random() < 0.6

    if use_cycle:
        chamber_count = random.randint(palette_len * group_len + 1, 12)
    else:
        chamber_count = random.randint(max(3, group_len + 1), max(4, min(12, palette_len * group_len)))
    return palette_len, group_len, chamber_count, use_cycle


def _build_expected(inp: np.ndarray, chambers: list[Rect], palette: list[int], group_len: int) -> np.ndarray:
    out = inp.copy()
    out[:, -1] = BG
    for idx, rect in enumerate(sorted(chambers)):
        color = palette[(idx // group_len) % len(palette)]
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    palette_len, group_len, chamber_count, _use_cycle = _sample_structure()
    wall = random.choice(NON_BG_NON_GREY)
    palette_pool = [color for color in NON_BG_NON_GREY if color != wall]
    palette = random.sample(palette_pool, palette_len)

    min_height = max(9, palette_len + group_len + 2)
    body_height = random.randint(min_height, 24)
    body_width = random.randint(max(8, 5 + chamber_count // 2), 28)
    total_width = body_width + 2
    if total_width > 30:
        return None

    built = _make_partitioned_body(body_height, body_width, wall, chamber_count)
    if built is None:
        return None
    body, chambers = built

    grid = np.full((body_height, total_width), BG, dtype=int)
    grid[:, :body_width] = body
    for row, color in enumerate(palette):
        grid[row, -1] = color
    grid[palette_len : palette_len + group_len, -1] = GREY

    expected = _build_expected(grid, chambers, palette, group_len)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chamber-rhythm input")
