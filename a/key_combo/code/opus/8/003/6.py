from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RECT = 1
ORANGE = 7
MAROON = 9
SPRITE_COLORS = (2, 3, 4, 5, 6)
SPECIAL_COLORS = (BG, RECT, ORANGE, MAROON)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _sort_key(cells: list[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, _, _ = _bbox(cells)
    return r0, c0


def _anchor_pixel(cells: list[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    return min(cells, key=lambda p: ((p[0] - cr) ** 2 + (p[1] - cc) ** 2, p[0], p[1]))


def _largest_blue_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(grid == RECT)
    assert comps, "expected a blue rectangle"
    largest = max(comps, key=len)
    return _bbox(largest)


def _relative_offsets(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    anchor = _anchor_pixel(cells)
    offsets = [(r - anchor[0], c - anchor[1]) for r, c in cells]
    return anchor, offsets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the outside sprites, keep every other one, stamp them onto the blue rectangle, and crop stamped pixels on side-marked rows and columns.

    Input structure:
    - The grid has a black background, one central blue filled rectangle, 2 or more orange singleton markers inside that rectangle, several small external sprites, and maroon singleton markers in the rectangle's side bands.
    - Each external sprite is a 4-connected monochrome component outside the rectangle; the maroon markers lie above/below the rectangle on rectangle columns or left/right of the rectangle on rectangle rows.

    Rule:
    - Find all external sprites and sort them by the top-left corner of their bounding boxes, row first then column.
    - Keep ranks 1, 3, 5, ... from that order as active sprites, pair them in order with the orange internal markers sorted row first then column, and ignore any extra active sprites beyond the marker count.
    - For each paired sprite, use the sprite pixel nearest its bounding-box center as the anchor, copy every sprite pixel by the same relative offsets onto the paired orange marker, and then erase any stamped pixel whose row or column is selected by a maroon side-band marker back to blue while preserving the orange markers themselves.
    - Erase all original external sprites from their source locations and keep the rectangle, orange markers, and maroon side markers in place.

    Key insight:
    - The transformation is a three-stage composition: positional sorting decides which sprites participate, marker order decides where each chosen sprite is stamped, and the maroon side bands crop rows and columns out of the stamped result.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, c1 = _largest_blue_rectangle(g)

    markers = sorted(
        [
            (r, c)
            for r, c in map(tuple, np.argwhere(g == ORANGE))
            if r0 <= r <= r1 and c0 <= c <= c1
        ]
    )
    assert markers, "expected orange markers inside the rectangle"

    excluded_rows = {
        r
        for r, c in map(tuple, np.argwhere(g == MAROON))
        if r0 <= r <= r1 and (c < c0 or c > c1)
    }
    excluded_cols = {
        c
        for r, c in map(tuple, np.argwhere(g == MAROON))
        if c0 <= c <= c1 and (r < r0 or r > r1)
    }

    sprite_mask = ~np.isin(g, SPECIAL_COLORS)
    sprite_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    sprites = _components(sprite_mask)
    sprites.sort(key=_sort_key)

    out = g.copy()
    for cells in sprites:
        for r, c in cells:
            out[r, c] = BG

    active = sprites[::2]
    for cells, marker in zip(active, markers):
        color = int(g[cells[0][0], cells[0][1]])
        _, offsets = _relative_offsets(cells)
        for dr, dc in offsets:
            tr = marker[0] + dr
            tc = marker[1] + dc
            if not (r0 <= tr <= r1 and c0 <= tc <= c1):
                continue
            if tr in excluded_rows or tc in excluded_cols:
                continue
            if out[tr, tc] == ORANGE:
                continue
            out[tr, tc] = color

    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _generate_shape(size: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            random.shuffle(frontier)
            placed = False
            for cand in frontier:
                trial = set(cells)
                trial.add(cand)
                rs = [r for r, _ in trial]
                cs = [c for _, c in trial]
                if max(rs) - min(rs) + 1 > 4 or max(cs) - min(cs) + 1 > 4:
                    continue
                cells = trial
                placed = True
                break
            if not placed:
                break
        if len(cells) != size:
            continue
        norm = _normalize(cells)
        r0, c0, r1, c1 = _bbox(norm)
        if (r1 - r0 + 1) == 1 or (c1 - c0 + 1) == 1:
            if random.random() < 0.75:
                continue
        return norm
    return None


def _can_place(grid: np.ndarray, shape: list[tuple[int, int]], top: int, left: int) -> bool:
    h, w = grid.shape
    for dr, dc in shape:
        r = top + dr
        c = left + dc
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != BG:
                    return False
    return True


def _place_shape(grid: np.ndarray, color: int, shape: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    shp_h = max(r for r, _ in shape) + 1
    shp_w = max(c for _, c in shape) + 1
    candidates = [
        (r, c)
        for r in range(h - shp_h + 1)
        for c in range(w - shp_w + 1)
        if _can_place(grid, shape, r, c)
    ]
    if not candidates:
        return None
    top, left = random.choice(candidates)
    abs_cells = []
    for dr, dc in shape:
        rr = top + dr
        cc = left + dc
        grid[rr, cc] = color
        abs_cells.append((rr, cc))
    return abs_cells


def _fit_positions(
    rect_bbox: tuple[int, int, int, int],
    offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect_bbox
    min_dr = min(dr for dr, _ in offsets)
    max_dr = max(dr for dr, _ in offsets)
    min_dc = min(dc for _, dc in offsets)
    max_dc = max(dc for _, dc in offsets)
    rows = range(max(r0 + 1, r0 - min_dr), min(r1 - 1, r1 - max_dr) + 1)
    cols = range(max(c0 + 1, c0 - min_dc), min(c1 - 1, c1 - max_dc) + 1)
    return [(r, c) for r in rows for c in cols]


def _choose_markers_and_cuts(
    rect_bbox: tuple[int, int, int, int],
    paired_sprites: list[list[tuple[int, int]]],
) -> tuple[list[tuple[int, int]], set[int], set[int]] | None:
    r0, c0, r1, c1 = rect_bbox
    if len(paired_sprites) != 2:
        return None

    sprite_offsets = [_relative_offsets(cells)[1] for cells in paired_sprites]
    fits = [_fit_positions(rect_bbox, offsets) for offsets in sprite_offsets]
    if not fits[0] or not fits[1]:
        return None

    rect_rows = list(range(r0, r1 + 1))
    rect_cols = list(range(c0, c1 + 1))
    for _ in range(400):
        m1 = random.choice(fits[0])
        later = [p for p in fits[1] if p != m1 and p >= m1]
        if not later:
            continue
        m2 = random.choice(later)
        markers = [m1, m2]
        targets = []
        ok = True
        occupied: set[tuple[int, int]] = set()
        for marker, offsets in zip(markers, sprite_offsets):
            cells = {(marker[0] + dr, marker[1] + dc) for dr, dc in offsets}
            if any(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in cells):
                ok = False
                break
            if occupied & cells:
                ok = False
                break
            occupied |= cells
            targets.append(cells)
        if not ok:
            continue

        for _ in range(300):
            n_rows = random.randint(1, min(2, len(rect_rows) - 1))
            n_cols = random.randint(1, min(2, len(rect_cols) - 1))
            excluded_rows = set(random.sample(rect_rows, n_rows))
            excluded_cols = set(random.sample(rect_cols, n_cols))

            cropped_any = False
            visible_ok = True
            for marker, cells in zip(markers, targets):
                visible = [
                    cell
                    for cell in cells
                    if cell[0] not in excluded_rows and cell[1] not in excluded_cols and cell != marker
                ]
                cropped = any(
                    cell[0] in excluded_rows or cell[1] in excluded_cols for cell in cells if cell != marker
                )
                cropped_any |= cropped
                if not visible:
                    visible_ok = False
                    break
            if visible_ok and cropped_any:
                return markers, excluded_rows, excluded_cols
    return None


def _build_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)

    rect_h = random.randint(8, min(10, h - 6))
    rect_w = random.randint(8, min(10, w - 6))
    r0 = random.randint(3, h - rect_h - 3)
    c_min = max(4, w // 2 - 2)
    c_max = w - rect_w - 3
    if c_min > c_max:
        c_min = 3
    c0 = random.randint(c_min, c_max)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = RECT

    sprite_count = random.choice([4, 5])
    colors = random.sample(list(SPRITE_COLORS), sprite_count)
    placed_sprites: list[list[tuple[int, int]]] = []
    for color in colors:
        shape = _generate_shape(random.randint(3, 6))
        if shape is None:
            return None
        cells = _place_shape(grid, color, shape)
        if cells is None:
            return None
        placed_sprites.append(cells)

    sprites = sorted(placed_sprites, key=_sort_key)
    active = sprites[::2]
    paired = active[:2]
    if len(paired) != 2:
        return None

    chosen = _choose_markers_and_cuts((r0, c0, r1, c1), paired)
    if chosen is None:
        return None
    markers, excluded_rows, excluded_cols = chosen

    for mr, mc in markers:
        grid[mr, mc] = ORANGE

    for row in excluded_rows:
        side_c = c0 - 1 if random.random() < 0.5 else c1 + 1
        if grid[row, side_c] != BG:
            return None
        grid[row, side_c] = MAROON
    for col in excluded_cols:
        side_r = r0 - 1 if random.random() < 0.5 else r1 + 1
        if grid[side_r, col] != BG:
            return None
        grid[side_r, col] = MAROON

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(np.isin(out[r0 : r1 + 1, c0 : c1 + 1], SPRITE_COLORS)) < 2:
        return None
    outside = out.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = 0
    if np.any(np.isin(outside, SPRITE_COLORS)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
