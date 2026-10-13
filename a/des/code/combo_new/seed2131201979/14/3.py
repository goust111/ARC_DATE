from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GRID = 30
GEN_RETRIES = 400


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _ring_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = {(int(r0), int(c)) for c in range(c0, c1 + 1)}
    cells.update((int(r1), int(c)) for c in range(c0, c1 + 1))
    cells.update((int(r), int(c0)) for r in range(r0, r1 + 1))
    cells.update((int(r), int(c1)) for r in range(r0, r1 + 1))
    return cells


def _find_rooms(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rooms: list[tuple[int, int, int, int]] = []
    for comp in _components4(grid == WALL):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if set(comp) != _ring_cells(r0, c0, r1, c1):
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or bool(np.any(interior == WALL)):
            continue
        rooms.append((int(r0), int(c0), int(r1), int(c1)))
    rooms.sort()
    return rooms


def _tile_from_markers(
    interior_shape: tuple[int, int], marker_coords: list[tuple[int, int]], color: int
) -> np.ndarray:
    h, w = map(int, interior_shape)
    tiled = np.zeros((h, w), dtype=int)
    if not marker_coords:
        return tiled

    coords = np.array(marker_coords, dtype=int)
    min_r = int(coords[:, 0].min())
    max_r = int(coords[:, 0].max())
    min_c = int(coords[:, 1].min())
    max_c = int(coords[:, 1].max())
    unit_h = int(max_r - min_r + 1)
    unit_w = int(max_c - min_c + 1)

    pattern = np.zeros((unit_h, unit_w), dtype=bool)
    pattern[coords[:, 0] - min_r, coords[:, 1] - min_c] = True

    row_idx = (np.arange(h, dtype=int) - min_r) % unit_h
    col_idx = (np.arange(w, dtype=int) - min_c) % unit_w
    tiled[pattern[row_idx[:, None], col_idx[None, :]]] = int(color)
    return tiled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-walled room by tiling the sparse marker pattern across its floor.

    Input structure:
    - The background is black.
    - One or more grey 1-pixel-thick rectangular wall loops form closed rooms.
    - Inside each room, 2 to 4 marker pixels share one non-black, non-grey color and all other interior cells are black.

    Rule:
    - For every room, collect the marker coordinates relative to the room interior.
    - Use the smallest marker bounding box as the repeating unit, with the marker pixels occupying the same relative offsets inside that unit.
    - Repeat that unit periodically across the whole room interior, preserving the wall and leaving non-hit floor cells black.

    Key insight:
    - The markers are not the final drawing; their bounding box defines the tile period, and the marker offsets inside that box define which cells are colored in every repeated copy.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _find_rooms(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = (interior != BG) & (interior != WALL)
        if not bool(marker_mask.any()):
            continue
        colors = np.unique(interior[marker_mask])
        if colors.size != 1:
            continue
        color = int(colors[0])
        marker_coords = [(int(r), int(c)) for r, c in np.argwhere(marker_mask)]
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_from_markers(interior.shape, marker_coords, color)

    return out


def _sample_pattern_cells(
    rng: np.random.Generator, unit_h: int, unit_w: int, count: int
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(unit_h) for c in range(unit_w)]
    for _ in range(200):
        idx = rng.choice(len(cells), size=count, replace=False)
        pts = [cells[int(i)] for i in idx]
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        if min(rows) == 0 and max(rows) == unit_h - 1 and min(cols) == 0 and max(cols) == unit_w - 1:
            return pts
    raise RuntimeError("failed to sample a minimal pattern")


def _sample_interior_extent(rng: np.random.Generator, unit: int) -> int:
    if unit == 1:
        return int(rng.integers(2, 7))
    candidates = sorted(
        {
            size
            for reps in range(1, 4)
            for extra in range(unit)
            for size in [reps * unit + extra]
            if unit <= size <= 8
        }
    )
    return int(candidates[int(rng.integers(0, len(candidates)))])


def _design_room(rng: np.random.Generator, color: int) -> dict[str, object]:
    unit_options = [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 1),
        (3, 1),
        (4, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (2, 4),
        (4, 2),
        (3, 3),
    ]
    unit_h, unit_w = unit_options[int(rng.integers(0, len(unit_options)))]
    count = int(rng.integers(2, min(4, unit_h * unit_w) + 1))
    pattern = _sample_pattern_cells(rng, unit_h, unit_w, count)

    interior_h = _sample_interior_extent(rng, unit_h)
    interior_w = _sample_interior_extent(rng, unit_w)
    anchor_r = int(rng.integers(0, interior_h - unit_h + 1))
    anchor_c = int(rng.integers(0, interior_w - unit_w + 1))
    markers = [(int(anchor_r + r), int(anchor_c + c)) for r, c in pattern]

    return {
        "color": int(color),
        "interior_h": int(interior_h),
        "interior_w": int(interior_w),
        "outer_h": int(interior_h + 2),
        "outer_w": int(interior_w + 2),
        "markers": markers,
    }


def _can_place(used: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(MAX_GRID, r0 + h + 1)
    cc1 = min(MAX_GRID, c0 + w + 1)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _draw_room(
    inp: np.ndarray, out: np.ndarray, room: dict[str, object], top: int, left: int
) -> None:
    interior_h = int(room["interior_h"])
    interior_w = int(room["interior_w"])
    color = int(room["color"])
    markers = list(room["markers"])

    bottom = int(top + interior_h + 1)
    right = int(left + interior_w + 1)

    inp[top : bottom + 1, left] = WALL
    inp[top : bottom + 1, right] = WALL
    inp[top, left : right + 1] = WALL
    inp[bottom, left : right + 1] = WALL

    out[top : bottom + 1, left] = WALL
    out[top : bottom + 1, right] = WALL
    out[top, left : right + 1] = WALL
    out[bottom, left : right + 1] = WALL

    for mr, mc in markers:
        inp[top + 1 + int(mr), left + 1 + int(mc)] = int(color)

    out[top + 1 : bottom, left + 1 : right] = _tile_from_markers(
        (interior_h, interior_w), markers, color
    )


def _crop_canvas(
    rng: np.random.Generator, inp: np.ndarray, out: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(inp != BG)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    top_pad = int(rng.integers(0, min(4, int(r0)) + 1))
    left_pad = int(rng.integers(0, min(4, int(c0)) + 1))
    bottom_pad = int(rng.integers(0, min(4, int(inp.shape[0] - 1 - r1)) + 1))
    right_pad = int(rng.integers(0, min(4, int(inp.shape[1] - 1 - c1)) + 1))
    rs = slice(int(r0 - top_pad), int(r1 + bottom_pad + 1))
    cs = slice(int(c0 - left_pad), int(c1 + right_pad + 1))
    return inp[rs, cs].copy(), out[rs, cs].copy()


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    room_count = int(rng.choice(np.array([1, 2, 2, 3, 3, 4], dtype=int)))
    colors = [int(x) for x in rng.choice(np.array(MARKER_COLORS, dtype=int), size=room_count, replace=False)]
    rooms = [_design_room(rng, color) for color in colors]
    rooms.sort(key=lambda room: int(room["outer_h"]) * int(room["outer_w"]), reverse=True)

    used = np.zeros((MAX_GRID, MAX_GRID), dtype=bool)
    placements: list[tuple[dict[str, object], int, int]] = []
    for room in rooms:
        placed = False
        outer_h = int(room["outer_h"])
        outer_w = int(room["outer_w"])
        for _ in range(300):
            top = int(rng.integers(0, MAX_GRID - outer_h + 1))
            left = int(rng.integers(0, MAX_GRID - outer_w + 1))
            if not _can_place(used, top, left, outer_h, outer_w):
                continue
            used[top : top + outer_h, left : left + outer_w] = True
            placements.append((room, top, left))
            placed = True
            break
        if not placed:
            return None

    inp = np.zeros((MAX_GRID, MAX_GRID), dtype=int)
    out = np.zeros((MAX_GRID, MAX_GRID), dtype=int)
    for room, top, left in placements:
        _draw_room(inp, out, room, top, left)

    cropped_inp, cropped_out = _crop_canvas(rng, inp, out)
    if np.array_equal(cropped_inp, cropped_out):
        return None
    if len(_find_rooms(cropped_inp)) != room_count:
        return None
    if not np.array_equal(main(cropped_inp), cropped_out):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tiled-room puzzle")
