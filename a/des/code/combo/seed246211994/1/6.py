import random
from collections import deque

import numpy as np


GEN_ATTEMPTS = 300
GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)
TILE_H_RANGE = (2, 5)
TILE_W_RANGE = (2, 5)
TILE_COLOR_COUNT_RANGE = (3, 4)
RECT_COUNT_RANGE = (1, 4)
RECT_SIDE_MIN = 3
RECT_COVER_RATIO_MAX = 0.38
SEQUENCE_ATTEMPTS = 120
TILE_ATTEMPTS = 120
RECT_PLACEMENT_ATTEMPTS = 100


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_solid_rectangles(grid):
    g = np.asarray(grid, dtype=int)
    rects = []
    for color in np.unique(g):
        for coords in _components(g == int(color)):
            if coords.shape[0] < 4:
                continue
            (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if height >= 2 and width >= 2 and coords.shape[0] == height * width:
                rects.append((int(r0), int(c0), int(r1), int(c1), int(color)))
    rects.sort()
    return rects


def _infer_tile(grid, blocked):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visible = np.argwhere(~blocked)
    values = g[~blocked]
    if len(values) == 0:
        raise AssertionError("at least one background cell must stay visible")

    best_tile = None
    best_score = None
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            tile = np.full((ph, pw), -1, dtype=int)
            seen = np.zeros((ph, pw), dtype=bool)
            ok = True
            for idx, (r, c) in enumerate(visible):
                rr = int(r % ph)
                cc = int(c % pw)
                value = int(values[idx])
                if seen[rr, cc]:
                    if int(tile[rr, cc]) != value:
                        ok = False
                        break
                else:
                    tile[rr, cc] = value
                    seen[rr, cc] = True
            if ok and bool(seen.all()):
                score = (ph * pw, ph, pw)
                if best_score is None or score < best_score:
                    best_score = score
                    best_tile = tile.copy()

    if best_tile is None:
        raise AssertionError("failed to infer a periodic background tile")
    return best_tile


def _tile_to_grid(tile, shape):
    h, w = shape
    rr = np.arange(h)[:, None] % tile.shape[0]
    cc = np.arange(w)[None, :] % tile.shape[1]
    return tile[rr, cc]


def _draw_borders(background, rects):
    out = np.asarray(background, dtype=int).copy()
    for r0, c0, r1, c1, color in rects:
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the periodic background hidden under solid rectangles, then keep each
    rectangle only as a one-pixel hollow border in its original color.
    """
    g = np.asarray(grid, dtype=int)
    rects = _find_solid_rectangles(g)

    blocked = np.zeros(g.shape, dtype=bool)
    for r0, c0, r1, c1, _ in rects:
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    tile = _infer_tile(g, blocked)
    restored = _tile_to_grid(tile, g.shape)
    return _draw_borders(restored, rects)


def _minimal_cycle_period(seq):
    n = len(seq)
    for p in range(1, n + 1):
        if all(seq[i] == seq[(i + p) % n] for i in range(n)):
            return p
    return n


def _make_primitive_cycle(length, color_count, rng):
    for _ in range(SEQUENCE_ATTEMPTS):
        seq = [rng.randrange(color_count) for _ in range(length)]
        if len(set(seq)) < 2:
            continue
        if any(seq[i] == seq[(i + 1) % length] for i in range(length)):
            continue
        if _minimal_cycle_period(seq) != length:
            continue
        return seq
    return None


def _make_tile(tile_h, tile_w, palette, rng):
    color_count = len(palette)
    palette_arr = np.asarray(palette, dtype=int)
    for _ in range(TILE_ATTEMPTS):
        row_cycle = _make_primitive_cycle(tile_h, color_count, rng)
        col_cycle = _make_primitive_cycle(tile_w, color_count, rng)
        if row_cycle is None or col_cycle is None:
            continue
        idx = (np.asarray(row_cycle)[:, None] + np.asarray(col_cycle)[None, :]) % color_count
        tile = palette_arr[idx]
        if _infer_tile(tile, np.zeros(tile.shape, dtype=bool)).shape == (tile_h, tile_w):
            return tile
    return None


def _overlay_rectangles(background, rects):
    grid = np.asarray(background, dtype=int).copy()
    for r0, c0, r1, c1, color in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _place_rectangles(h, w, colors, count, rng):
    rects = []
    pad_occupied = np.zeros((h, w), dtype=bool)
    covered = 0
    area_cap = max(1, int(h * w * RECT_COVER_RATIO_MAX))
    max_rect_h = max(RECT_SIDE_MIN, min(h, int(round(h * 0.55))))
    max_rect_w = max(RECT_SIDE_MIN, min(w, int(round(w * 0.55))))

    for color in colors[:count]:
        placed = False
        for _ in range(RECT_PLACEMENT_ATTEMPTS):
            rect_h = rng.randint(RECT_SIDE_MIN, max_rect_h)
            rect_w = rng.randint(RECT_SIDE_MIN, max_rect_w)
            r0 = rng.randint(0, h - rect_h)
            c0 = rng.randint(0, w - rect_w)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(h - 1, r1 + 1)
            pc1 = min(w - 1, c1 + 1)
            if pad_occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue
            if covered + rect_h * rect_w > area_cap:
                continue
            rects.append((r0, c0, r1, c1, int(color)))
            pad_occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            covered += rect_h * rect_w
            placed = True
            break
        if not placed:
            return None
    return rects


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    tile_h = rng.randint(TILE_H_RANGE[0], min(TILE_H_RANGE[1], h))
    tile_w = rng.randint(TILE_W_RANGE[0], min(TILE_W_RANGE[1], w))
    tile_color_count = rng.randint(*TILE_COLOR_COUNT_RANGE)
    if tile_color_count > 9:
        return None

    tile_palette = rng.sample(list(range(10)), tile_color_count)
    tile = _make_tile(tile_h, tile_w, tile_palette, rng)
    if tile is None:
        return None

    row_shift = rng.randrange(tile_h)
    col_shift = rng.randrange(tile_w)
    rr = (np.arange(h)[:, None] + row_shift) % tile_h
    cc = (np.arange(w)[None, :] + col_shift) % tile_w
    background = tile[rr, cc]

    if _infer_tile(background, np.zeros((h, w), dtype=bool)).shape != (tile_h, tile_w):
        return None

    remaining_colors = [color for color in range(10) if color not in tile_palette]
    max_rects = min(RECT_COUNT_RANGE[1], len(remaining_colors))
    if max_rects < RECT_COUNT_RANGE[0]:
        return None
    rect_count = rng.randint(RECT_COUNT_RANGE[0], max_rects)
    rng.shuffle(remaining_colors)
    rects = _place_rectangles(h, w, remaining_colors, rect_count, rng)
    if rects is None:
        return None

    inp = _overlay_rectangles(background, rects)
    expected = _draw_borders(background, rects)

    if len(_find_solid_rectangles(inp)) != rect_count:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
