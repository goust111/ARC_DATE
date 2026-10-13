from __future__ import annotations

import random

import numpy as np


BLACK = 0
TILE_COLORS = (1, 2, 3, 4)
MARKER_COLORS = (5, 6, 7)
SIDES = ("top", "bottom", "left", "right")

GEN_ATTEMPTS = 600


def _marker_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int] | None:
    h, w = shape
    if r == 0 and 0 < c < w - 1:
        return 1, 0
    if r == h - 1 and 0 < c < w - 1:
        return -1, 0
    if c == 0 and 0 < r < h - 1:
        return 0, 1
    if c == w - 1 and 0 < r < h - 1:
        return 0, -1
    return None


def _trace_repair_channels(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    ray_mask = np.zeros((h, w), dtype=bool)
    markers: list[tuple[int, int]] = []

    edge_coords = (
        [(0, c) for c in range(1, w - 1)]
        + [(h - 1, c) for c in range(1, w - 1)]
        + [(r, 0) for r in range(1, h - 1)]
        + [(r, w - 1) for r in range(1, h - 1)]
    )
    for r, c in edge_coords:
        if int(grid[r, c]) not in MARKER_COLORS:
            continue
        direction = _marker_direction((h, w), r, c)
        if direction is None:
            continue
        markers.append((r, c))
        dr, dc = direction
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BLACK:
            ray_mask[rr, cc] = True
            rr += dr
            cc += dc

    return ray_mask, markers


def _infer_periodic_tile(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    observed = np.isin(g, TILE_COLORS)
    assert observed.any(), "expected visible wallpaper pixels"

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for row_period in range(1, h):
        for col_period in range(1, w):
            tile = np.full((row_period, col_period), -1, dtype=int)
            conflict = False
            for r, c in np.argwhere(observed):
                color = int(g[r, c])
                tr, tc = int(r % row_period), int(c % col_period)
                known = int(tile[tr, tc])
                if known == -1:
                    tile[tr, tc] = color
                elif known != color:
                    conflict = True
                    break
            if not conflict and np.all(tile != -1):
                candidates.append((row_period * col_period, row_period, col_period, tile))

    assert candidates, "expected an inferable periodic wallpaper"
    min_area = min(area for area, _, _, _ in candidates)
    smallest = [item for item in candidates if item[0] == min_area]
    assert len(smallest) == 1, "expected a unique smallest periodic wallpaper"
    return smallest[0][3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair only the black wallpaper holes that lie inside border-projected repair channels.

    Input structure:
    - A black-backed grid contains a repeated rectangular wallpaper tile using colors 1-4.
    - Some wallpaper cells are damaged into black holes, including border-adjacent corridors.
    - Single-pixel border markers in colors 5-7 sit on the outer edge but not on corners.

    Rule:
    - For each border marker, project inward along its row or column from the adjacent inner cell.
    - The repair channel is the maximal run of black cells until the first non-black cell, which blocks the ray.
    - Infer the unique smallest wallpaper row/column periods from the remaining visible wallpaper pixels.
    - Fill only black cells inside at least one repair channel with their wallpaper phase color, keep other black holes black, and erase all border markers to black.

    Key insight:
    - The markers decide where repair is allowed, while the wallpaper periodicity decides what color each allowed repair cell must receive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    ray_mask, markers = _trace_repair_channels(g)
    tile = _infer_periodic_tile(g)

    out = g.copy()
    for r, c in markers:
        out[r, c] = BLACK

    for r, c in np.argwhere(ray_mask & (g == BLACK)):
        out[r, c] = int(tile[int(r % tile.shape[0]), int(c % tile.shape[1])])

    return out


def _row_period(tile: np.ndarray) -> int:
    h, _ = map(int, tile.shape)
    for period in range(1, h + 1):
        ok = True
        for r in range(h):
            if not np.array_equal(tile[r], tile[r % period]):
                ok = False
                break
        if ok:
            return period
    return h


def _col_period(tile: np.ndarray) -> int:
    _, w = map(int, tile.shape)
    for period in range(1, w + 1):
        ok = True
        for c in range(w):
            if not np.array_equal(tile[:, c], tile[:, c % period]):
                ok = False
                break
        if ok:
            return period
    return w


def _generate_base_tile(period_h: int, period_w: int) -> np.ndarray:
    for _ in range(400):
        palette = random.sample(TILE_COLORS, random.randint(2, 3))
        tile = np.array(
            random.choices(palette, k=period_h * period_w),
            dtype=int,
        ).reshape(period_h, period_w)
        if len(set(map(int, tile.ravel()))) < len(palette):
            continue
        if _row_period(tile) != period_h or _col_period(tile) != period_w:
            continue
        if all(len(set(map(int, tile[r, :]))) == 1 for r in range(period_h)):
            continue
        if all(len(set(map(int, tile[:, c]))) == 1 for c in range(period_w)):
            continue
        return tile
    raise RuntimeError("failed to construct a non-degenerate periodic tile")


def _expand_chebyshev(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _make_region(side: str, index: int, h: int, w: int) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    vertical = side in ("top", "bottom")
    max_trunk = (h - 3) if vertical else (w - 3)
    if max_trunk < 2:
        return None

    trunk_len = random.randint(2, min(5, max_trunk))
    cells: set[tuple[int, int]] = set()

    if side == "top":
        marker = (0, index)
        trunk = [(r, index) for r in range(1, trunk_len + 1)]
        side_choices = [dc for dc in (-1, 1) if 1 <= index + dc <= w - 2]
        if not side_choices:
            return None
        branch_dc = random.choice(side_choices)
        branch_len = random.randint(1, min(3, trunk_len))
        start = random.randint(1, trunk_len - branch_len + 1)
        branch = [(r, index + branch_dc) for r in range(start, start + branch_len)]
    elif side == "bottom":
        marker = (h - 1, index)
        trunk = [(h - 1 - r, index) for r in range(1, trunk_len + 1)]
        side_choices = [dc for dc in (-1, 1) if 1 <= index + dc <= w - 2]
        if not side_choices:
            return None
        branch_dc = random.choice(side_choices)
        branch_len = random.randint(1, min(3, trunk_len))
        start = random.randint(1, trunk_len - branch_len + 1)
        branch = [(h - 1 - r, index + branch_dc) for r in range(start, start + branch_len)]
    elif side == "left":
        marker = (index, 0)
        trunk = [(index, c) for c in range(1, trunk_len + 1)]
        side_choices = [dr for dr in (-1, 1) if 1 <= index + dr <= h - 2]
        if not side_choices:
            return None
        branch_dr = random.choice(side_choices)
        branch_len = random.randint(1, min(3, trunk_len))
        start = random.randint(1, trunk_len - branch_len + 1)
        branch = [(index + branch_dr, c) for c in range(start, start + branch_len)]
    else:
        marker = (index, w - 1)
        trunk = [(index, w - 1 - c) for c in range(1, trunk_len + 1)]
        side_choices = [dr for dr in (-1, 1) if 1 <= index + dr <= h - 2]
        if not side_choices:
            return None
        branch_dr = random.choice(side_choices)
        branch_len = random.randint(1, min(3, trunk_len))
        start = random.randint(1, trunk_len - branch_len + 1)
        branch = [(index + branch_dr, w - 1 - c) for c in range(start, start + branch_len)]

    cells.update(trunk)
    cells.update(branch)

    if len(cells) < 3 or len(cells) > 8:
        return None
    if any(r <= 0 or r >= h - 1 or c <= 0 or c >= w - 1 for r, c in cells):
        return None
    return cells, marker


def _visible_phases(inp: np.ndarray, period_h: int, period_w: int) -> bool:
    visible = np.zeros((period_h, period_w), dtype=bool)
    for r, c in np.argwhere(np.isin(inp, TILE_COLORS)):
        visible[int(r % period_h), int(c % period_w)] = True
    return bool(visible.all())


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(17, 22)
    w = random.randint(17, 22)
    period_h = random.choice((3, 4))
    period_w = random.choice((3, 4))

    tile = _generate_base_tile(period_h, period_w)
    wallpaper = np.tile(tile, (h // period_h + 1, w // period_w + 1))[:h, :w]

    marker_count = random.randint(2, 4)
    sides = random.sample(SIDES, marker_count)
    random.shuffle(sides)

    hole_cells: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    markers: list[tuple[int, int, int]] = []

    for side in sides:
        indices = list(range(1, w - 1)) if side in ("top", "bottom") else list(range(1, h - 1))
        random.shuffle(indices)
        placed = False
        for index in indices:
            for _ in range(12):
                region = _make_region(side, index, h, w)
                if region is None:
                    continue
                cells, marker = region
                if any(cell in blocked for cell in cells):
                    continue
                if any(abs(marker[0] - mr) <= 1 and abs(marker[1] - mc) <= 1 for mr, mc, _ in markers):
                    continue
                hole_cells |= cells
                blocked |= _expand_chebyshev(cells, h, w)
                markers.append((marker[0], marker[1], random.choice(MARKER_COLORS)))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    inp = wallpaper.copy()
    hole_mask = np.zeros((h, w), dtype=bool)
    for r, c in hole_cells:
        hole_mask[r, c] = True
    inp[hole_mask] = BLACK
    for r, c, color in markers:
        inp[r, c] = color

    if not _visible_phases(inp, period_h, period_w):
        return None

    ray_mask, found_markers = _trace_repair_channels(inp)
    if len(found_markers) != marker_count:
        return None

    repaired = hole_mask & ray_mask
    unrepaired = hole_mask & ~ray_mask
    if not repaired.any() or not unrepaired.any():
        return None

    expected = wallpaper.copy()
    expected[unrepaired] = BLACK
    for r, c, _ in markers:
        expected[r, c] = BLACK

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
