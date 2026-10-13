from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np


BG_COLOR = 0
SEPARATOR_COLOR = 5
PALETTE = tuple(color for color in range(10) if color not in (BG_COLOR, SEPARATOR_COLOR))
DIRECTIONS = ("top", "bottom", "left", "right")

PANEL_ROWS_RANGE = (2, 4)
PANEL_COLS_RANGE = (2, 4)
PANEL_HEIGHT_RANGE = (7, 10)
PANEL_WIDTH_RANGE = (7, 10)
MIN_OBJECT_CELLS = 2
MAX_OBJECT_CELLS = 8
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield (r - 1, c)
    if r + 1 < h:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < w:
        yield (r, c + 1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _uniform_color(values: np.ndarray) -> int:
    uniq = np.unique(np.asarray(values, dtype=int))
    assert len(uniq) == 1, f"expected a uniform edge color, got {uniq.tolist()}"
    return int(uniq[0])


def _find_separator_color(grid: np.ndarray) -> int | None:
    best_color = None
    best_score = 0
    for color in map(int, np.unique(grid)):
        full_rows = int(np.all(grid == color, axis=1).sum())
        full_cols = int(np.all(grid == color, axis=0).sum())
        score = full_rows + full_cols
        if score > best_score:
            best_score = score
            best_color = color
    return best_color if best_score > 0 else None


def _find_background_color(grid: np.ndarray, separator: int | None) -> int:
    values, counts = np.unique(grid, return_counts=True)
    pairs = [
        (int(count), int(value))
        for value, count in zip(values, counts)
        if separator is None or int(value) != int(separator)
    ]
    assert pairs
    pairs.sort()
    return pairs[-1][1]


def _segments_from_separator_flags(flags: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    n = len(flags)
    i = 0
    while i < n:
        while i < n and flags[i]:
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not flags[j]:
            j += 1
        segments.append((i, j))
        i = j
    return segments


def _panel_ranges(grid: np.ndarray, separator: int | None) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if separator is None:
        return [(0, grid.shape[0])], [(0, grid.shape[1])]
    row_flags = np.all(grid == int(separator), axis=1)
    col_flags = np.all(grid == int(separator), axis=0)
    row_segments = _segments_from_separator_flags(row_flags)
    col_segments = _segments_from_separator_flags(col_flags)
    assert row_segments and col_segments, "expected at least one panel"
    row_sizes = {end - start for start, end in row_segments}
    col_sizes = {end - start for start, end in col_segments}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "panels must be equal-sized"
    return row_segments, col_segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each interior object toward the border side that shares its color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator = _find_separator_color(g)
    bg = _find_background_color(g, separator)
    row_segments, col_segments = _panel_ranges(g, separator)

    out = g.copy()
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            panel = g[r0:r1, c0:c1]
            ph, pw = panel.shape
            assert ph >= 3 and pw >= 3

            top_color = _uniform_color(panel[0, 1:-1])
            bottom_color = _uniform_color(panel[-1, 1:-1])
            left_color = _uniform_color(panel[1:-1, 0])
            right_color = _uniform_color(panel[1:-1, -1])
            side_colors = {
                "top": top_color,
                "bottom": bottom_color,
                "left": left_color,
                "right": right_color,
            }

            inner = panel[1:-1, 1:-1]
            object_mask = inner != int(bg)
            comps = _components(object_mask)
            assert len(comps) == 1, "each panel must contain one 4-connected object"
            coords = comps[0]

            obj_colors = np.unique(inner[coords[:, 0], coords[:, 1]])
            assert len(obj_colors) == 1, "object must be monochrome"
            obj_color = int(obj_colors[0])

            matched_sides = [side for side, color in side_colors.items() if int(color) == int(obj_color)]
            assert len(matched_sides) == 1, "object color must match exactly one border side"
            side = matched_sides[0]

            rr = coords[:, 0]
            cc = coords[:, 1]
            top = int(rr.min())
            bottom = int(rr.max())
            left = int(cc.min())
            right = int(cc.max())

            dr = 0
            dc = 0
            if side == "top":
                dr = -top
            elif side == "bottom":
                dr = (inner.shape[0] - 1) - bottom
            elif side == "left":
                dc = -left
            else:
                dc = (inner.shape[1] - 1) - right

            panel_out = out[r0:r1, c0:c1]
            inner_out = panel_out[1:-1, 1:-1]
            inner_out[rr, cc] = int(bg)
            inner_out[rr + dr, cc + dc] = int(obj_color)

    return out


def _balanced_directions(count: int, rng: random.Random) -> list[str]:
    dirs = list(DIRECTIONS) * (count // len(DIRECTIONS))
    remainder = list(DIRECTIONS)
    rng.shuffle(remainder)
    dirs.extend(remainder[: count % len(DIRECTIONS)])
    rng.shuffle(dirs)
    return dirs


def _random_connected_mask(
    rng: random.Random,
    max_h: int,
    max_w: int,
    min_cells: int = MIN_OBJECT_CELLS,
    max_cells: int = MAX_OBJECT_CELLS,
) -> np.ndarray:
    assert max_h >= 1 and max_w >= 1
    for _ in range(200):
        h = rng.randint(1, max_h)
        w = rng.randint(1, max_w)
        upper = min(max_cells, h * w)
        lower = min(min_cells, upper)
        target = rng.randint(lower, upper)

        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        cells = {start}

        while len(cells) < target:
            frontier = set()
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choice = rng.choice(tuple(frontier))
            mask[choice] = True
            cells.add(choice)

        if int(mask.sum()) != target:
            continue

        cropped = _crop_mask(mask)
        if cropped.shape[0] > max_h or cropped.shape[1] > max_w:
            continue
        return cropped

    raise RuntimeError("failed to sample a connected object mask")


def _draw_panel_border(
    panel: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    rng: random.Random,
) -> None:
    panel[0, :] = int(top)
    panel[-1, :] = int(bottom)
    panel[:, 0] = int(left)
    panel[:, -1] = int(right)

    panel[0, 0] = int(rng.choice((top, left)))
    panel[0, -1] = int(rng.choice((top, right)))
    panel[-1, 0] = int(rng.choice((bottom, left)))
    panel[-1, -1] = int(rng.choice((bottom, right)))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_rows = rng.randint(*PANEL_ROWS_RANGE)
    panel_cols = rng.randint(*PANEL_COLS_RANGE)
    panel_h = rng.randint(*PANEL_HEIGHT_RANGE)
    panel_w = rng.randint(*PANEL_WIDTH_RANGE)

    inner_h = panel_h - 2
    inner_w = panel_w - 2
    if inner_h < 3 or inner_w < 3:
        return None

    grid_h = panel_rows * panel_h + (panel_rows - 1)
    grid_w = panel_cols * panel_w + (panel_cols - 1)
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

    for r in range(panel_h, grid_h, panel_h + 1):
        grid[r, :] = SEPARATOR_COLOR
    for c in range(panel_w, grid_w, panel_w + 1):
        grid[:, c] = SEPARATOR_COLOR

    directions = _balanced_directions(panel_rows * panel_cols, rng)

    for idx, direction in enumerate(directions):
        pr = idx // panel_cols
        pc = idx % panel_cols
        r0 = pr * (panel_h + 1)
        c0 = pc * (panel_w + 1)
        panel = grid[r0 : r0 + panel_h, c0 : c0 + panel_w]

        top_color, bottom_color, left_color, right_color = rng.sample(PALETTE, 4)
        _draw_panel_border(panel, top_color, bottom_color, left_color, right_color, rng)

        max_shape_h = inner_h - 2
        max_shape_w = inner_w - 2
        if max_shape_h < 1 or max_shape_w < 1:
            return None
        shape = _random_connected_mask(rng, max_shape_h, max_shape_w)
        sh, sw = shape.shape

        start_r = rng.randint(1, inner_h - sh - 1)
        start_c = rng.randint(1, inner_w - sw - 1)

        color_by_side = {
            "top": top_color,
            "bottom": bottom_color,
            "left": left_color,
            "right": right_color,
        }
        obj_color = int(color_by_side[direction])

        inner = panel[1:-1, 1:-1]
        coords = np.argwhere(shape)
        inner[start_r + coords[:, 0], start_c + coords[:, 1]] = obj_color

    try:
        transformed = main(grid)
    except AssertionError:
        return None
    if np.array_equal(transformed, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
