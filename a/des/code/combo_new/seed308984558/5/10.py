import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MIN_BAND_HEIGHT = 5
MAX_BAND_HEIGHT = 7
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _scan_color(grid: np.ndarray, r: int, c: int, dr: int, dc: int, bg: int, grey: int) -> int | None:
    h, w = grid.shape
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        color = int(grid[rr, cc])
        if color == bg:
            return None
        if color != grey:
            return color
        rr += dr
        cc += dc
    return None


def _horizontal_run(grid: np.ndarray, r: int, c: int, color: int) -> tuple[int, int]:
    h, w = grid.shape
    assert 0 <= r < h and 0 <= c < w
    assert c - 1 >= 0 and c + 1 < w
    assert int(grid[r, c - 1]) == color and int(grid[r, c + 1]) == color

    left = c - 1
    right = c + 1
    while left - 1 >= 0 and int(grid[r, left - 1]) == color:
        left -= 1
    while right + 1 < w and int(grid[r, right + 1]) == color:
        right += 1
    return left, right


def _detour_points(r: int, c: int, side_col: int) -> list[tuple[int, int]]:
    if side_col < c:
        points = [(r - 1, cc) for cc in range(c - 1, side_col - 1, -1)]
        points.append((r, side_col))
        points.append((r + 1, side_col))
        points.extend((r + 1, cc) for cc in range(side_col + 1, c))
        return points

    points = [(r - 1, cc) for cc in range(c + 1, side_col + 1)]
    points.append((r, side_col))
    points.append((r + 1, side_col))
    points.extend((r + 1, cc) for cc in range(side_col - 1, c, -1))
    return points


def _cells_clear(grid: np.ndarray, cells: Iterable[tuple[int, int]], bg: int, color: int) -> bool:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    for r, c in cells:
        if (r, c) in seen:
            continue
        seen.add((r, c))
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) not in (bg, color):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign each grey crossing so the horizontal path keeps the center while the vertical path detours around it.

    Input structure:
    - The background is Black.
    - Colored paths are one-pixel-wide orthogonal traces; each crossing conflict is marked by a Grey pixel.
    - At every Grey pixel, the left and right approaches belong to one color and the up and down approaches belong to another color.

    Rule:
    - Recolor each Grey crossing with the color that approaches from both left and right, so the horizontal path passes straight through.
    - Keep the horizontal path unchanged elsewhere.
    - Reconnect the vertical color by drawing a same-color side detour around the horizontal run, using the only clear side when one side is blocked.
    - Leave all unrelated path pixels unchanged.

    Key insight:
    - The Grey pixel is not a third path color; it is a directional conflict marker whose ownership is decided by orientation, with horizontal traffic taking priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    grey = GREY if np.any(g == GREY) else None
    if grey is None:
        return g.copy()

    out = g.copy()
    for r, c in sorted(map(tuple, np.argwhere(g == grey))):
        left = _scan_color(g, r, c, 0, -1, bg, grey)
        right = _scan_color(g, r, c, 0, 1, bg, grey)
        up = _scan_color(g, r, c, -1, 0, bg, grey)
        down = _scan_color(g, r, c, 1, 0, bg, grey)

        assert left is not None and right is not None and left == right, "grey cell needs one horizontal color"
        assert up is not None and down is not None and up == down, "grey cell needs one vertical color"
        h_color = int(left)
        v_color = int(up)
        assert h_color != v_color, "crossing colors must differ"

        left_bound, right_bound = _horizontal_run(g, r, c, h_color)
        left_path = None
        right_path = None

        if left_bound - 1 >= 0:
            candidate = _detour_points(r, c, left_bound - 1)
            if _cells_clear(out, candidate, bg, v_color):
                left_path = candidate
        if right_bound + 1 < g.shape[1]:
            candidate = _detour_points(r, c, right_bound + 1)
            if _cells_clear(out, candidate, bg, v_color):
                right_path = candidate

        assert left_path is not None or right_path is not None, "vertical path needs at least one clear detour"
        if left_path is None:
            chosen = right_path
        elif right_path is None:
            chosen = left_path
        elif len(left_path) <= len(right_path):
            chosen = left_path
        else:
            chosen = right_path

        out[r, c] = h_color
        for rr, cc in chosen:
            out[rr, cc] = v_color

    return out


def _band_layout(count: int) -> tuple[int, list[tuple[int, int]]] | None:
    heights = [random.randint(MIN_BAND_HEIGHT, MAX_BAND_HEIGHT) for _ in range(count)]
    baseline = sum(heights) + (count - 1)
    if baseline > 30:
        return None

    height = random.randint(baseline, min(30, baseline + 5))
    slack = height - baseline
    pads = [0] * (count + 1)
    for _ in range(slack):
        pads[random.randrange(count + 1)] += 1

    bands: list[tuple[int, int]] = []
    row = pads[0]
    for i, band_h in enumerate(heights):
        bands.append((row, band_h))
        row += band_h
        if i + 1 < count:
            row += 1 + pads[i + 1]
    return height, bands


def _paint_crossing(
    grid: np.ndarray,
    top: int,
    band_h: int,
    h_color: int,
    v_color: int,
    side: str,
) -> None:
    h, w = grid.shape
    bottom = top + band_h - 1
    r = random.randint(top + 1, bottom - 1)
    c = random.randint(2, w - 3)

    v_top = top + random.randint(0, min(1, r - top - 1))
    v_bottom = bottom - random.randint(0, min(1, bottom - r - 1))

    if side == "left":
        h_start = 0
        h_end = random.randint(max(c + 1, 4), w - 2)
    else:
        h_start = random.randint(1, min(c - 1, w - 5))
        h_end = w - 1

    grid[r, h_start : h_end + 1] = h_color
    grid[v_top : v_bottom + 1, c] = v_color
    grid[r, c] = GREY


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(1, 4)
    if 2 * count > len(PATH_COLORS):
        return None

    layout = _band_layout(count)
    if layout is None:
        return None
    height, bands = layout

    width = random.randint(12, 24)
    grid = np.full((height, width), BG, dtype=int)
    colors = random.sample(PATH_COLORS, 2 * count)
    sides = [random.choice(["left", "right"]) for _ in range(count)]

    for i, ((top, band_h), side) in enumerate(zip(bands, sides)):
        h_color = colors[2 * i]
        v_color = colors[2 * i + 1]
        _paint_crossing(grid, top, band_h, h_color, v_color, side)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
