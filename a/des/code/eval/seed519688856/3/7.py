from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 8
WALL_COLOR = 9
SEED_COLORS = (1, 2, 3)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _corner_bg(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return Counter(corners).most_common(1)[0][0]


def _dominant_non_bg(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.ravel()))
    non_bg = [(count, color) for color, count in counts.items() if color != bg]
    assert non_bg, "expected a wall color"
    return max(non_bg)[1]


def _find_gap_candidates(grid: np.ndarray, bg: int, wall: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    gaps: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != bg:
                continue
            vertical = (
                int(grid[r - 1, c]) == wall
                and int(grid[r + 1, c]) == wall
                and int(grid[r, c - 1]) != wall
                and int(grid[r, c + 1]) != wall
            )
            horizontal = (
                int(grid[r, c - 1]) == wall
                and int(grid[r, c + 1]) == wall
                and int(grid[r - 1, c]) != wall
                and int(grid[r + 1, c]) != wall
            )
            if vertical ^ horizontal:
                gaps.append((r, c))
    return gaps


def _analyze_rooms(grid: np.ndarray) -> tuple[list[dict], int, int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _corner_bg(g)
    wall = _dominant_non_bg(g, bg)

    gap_cells = _find_gap_candidates(g, bg, wall)
    closed = g.copy()
    for r, c in gap_cells:
        closed[r, c] = wall

    rooms: list[dict] = []
    for comp in _components(closed != wall):
        if _touches_edge(comp, h, w):
            continue
        seeds = [(r, c) for r, c in comp if int(g[r, c]) not in (bg, wall)]
        if len(seeds) != 1:
            continue
        comp_set = set(comp)
        gap_count = 0
        for gr, gc in gap_cells:
            if any((gr + dr, gc + dc) in comp_set for dr, dc in DIR4):
                gap_count += 1
        if not (1 <= gap_count <= 3):
            continue
        seed = seeds[0]
        rooms.append(
            {
                "cells": comp,
                "cell_set": comp_set,
                "seed": seed,
                "color": int(g[seed]),
                "gap_count": gap_count,
                "top_row": min(r for r, _ in comp),
            }
        )

    rooms.sort(key=lambda room: room["seed"])
    return rooms, bg, wall, gap_cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rooms, bg, _wall, _gap_cells = _analyze_rooms(g)

    for room in rooms:
        seed_r, _seed_c = room["seed"]
        fill_color = room["color"]
        gap_count = room["gap_count"]
        top_row = room["top_row"]

        for r, c in room["cells"]:
            if int(out[r, c]) != bg:
                continue
            if gap_count == 1:
                fill_here = r == seed_r
            elif gap_count == 2:
                fill_here = (r - top_row) % 2 == 0
            else:
                fill_here = True
            if fill_here:
                out[r, c] = fill_color

    return out


def _erode4(mask: np.ndarray) -> np.ndarray:
    eroded = np.zeros_like(mask, dtype=bool)
    eroded[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return eroded


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _border_and_interior(area: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    interior = _erode4(area)
    border = area & ~interior
    return border, interior


def _mask_get(mask: np.ndarray, r: int, c: int) -> bool:
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _gap_positions(area: np.ndarray, border: np.ndarray) -> list[tuple[int, int]]:
    h, w = area.shape
    positions: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not border[r, c]:
                continue
            horizontal = (
                _mask_get(border, r, c - 1)
                and _mask_get(border, r, c + 1)
                and _mask_get(area, r - 1, c) != _mask_get(area, r + 1, c)
            )
            vertical = (
                _mask_get(border, r - 1, c)
                and _mask_get(border, r + 1, c)
                and _mask_get(area, r, c - 1) != _mask_get(area, r, c + 1)
            )
            if horizontal ^ vertical:
                positions.append((r, c))
    return positions


def _choose_spaced(points: list[tuple[int, int]], count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    pts = list(points)
    for _ in range(60):
        rng.shuffle(pts)
        chosen: list[tuple[int, int]] = []
        for point in pts:
            if all(abs(point[0] - other[0]) + abs(point[1] - other[1]) >= 3 for other in chosen):
                chosen.append(point)
                if len(chosen) == count:
                    return chosen
    return None


def _make_area_mask(rng: random.Random) -> np.ndarray | None:
    canvas = np.zeros((18, 18), dtype=bool)
    base_h = rng.randint(6, 10)
    base_w = rng.randint(6, 10)
    r0 = rng.randint(4, 18 - base_h - 4)
    c0 = rng.randint(4, 18 - base_w - 4)
    canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True

    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    protrusion_count = rng.choices([0, 1, 2], weights=[1, 3, 2])[0]
    for side in sides[:protrusion_count]:
        if side in ("top", "bottom"):
            depth = rng.randint(2, 3)
            span = rng.randint(3, base_w - 2)
            start = rng.randint(c0 + 1, c0 + base_w - span - 1)
            if side == "top":
                canvas[r0 - depth : r0, start : start + span] = True
            else:
                canvas[r0 + base_h : r0 + base_h + depth, start : start + span] = True
        else:
            depth = rng.randint(2, 3)
            span = rng.randint(3, base_h - 2)
            start = rng.randint(r0 + 1, r0 + base_h - span - 1)
            if side == "left":
                canvas[start : start + span, c0 - depth : c0] = True
            else:
                canvas[start : start + span, c0 + base_w : c0 + base_w + depth] = True

    area = _crop_mask(canvas)
    border, interior = _border_and_interior(area)
    if int(interior.sum()) < 10:
        return None
    if len(_components(interior)) != 1 or len(_components(border)) != 1:
        return None
    row_counts = [int(interior[r].sum()) for r in range(interior.shape[0]) if interior[r].any()]
    if min(row_counts) < 2:
        return None
    return area


def _room_fill_cells(interior: np.ndarray, seed: tuple[int, int], entrance_count: int) -> set[tuple[int, int]]:
    cells = {(int(r), int(c)) for r, c in np.argwhere(interior)}
    top_row = min(r for r, _ in cells)
    if entrance_count == 1:
        return {(r, c) for r, c in cells if r == seed[0]}
    if entrance_count == 2:
        return {(r, c) for r, c in cells if (r - top_row) % 2 == 0}
    return cells


def _build_room(rng: random.Random, entrance_count: int) -> dict | None:
    for _ in range(120):
        area = _make_area_mask(rng)
        if area is None:
            continue
        border, interior = _border_and_interior(area)
        gap_choices = _gap_positions(area, border)
        gaps = _choose_spaced(gap_choices, entrance_count, rng)
        if gaps is None:
            continue

        row_options: dict[int, list[tuple[int, int]]] = {}
        for r, c in np.argwhere(interior):
            row_options.setdefault(int(r), []).append((int(r), int(c)))
        if entrance_count == 1:
            good_rows = [r for r, cells in row_options.items() if len(cells) >= 3]
            if not good_rows:
                continue
            seed_row = rng.choice(good_rows)
        else:
            seed_row = rng.choice(list(row_options))
        seed = rng.choice(row_options[seed_row])
        seed_color = rng.choice(SEED_COLORS)

        patch = np.full(area.shape, BG_COLOR, dtype=int)
        patch[border] = WALL_COLOR
        for gr, gc in gaps:
            patch[gr, gc] = BG_COLOR
        patch[seed] = seed_color

        fill_cells = _room_fill_cells(interior, seed, entrance_count)
        framed = np.full((patch.shape[0] + 4, patch.shape[1] + 4), BG_COLOR, dtype=int)
        framed[2 : 2 + patch.shape[0], 2 : 2 + patch.shape[1]] = patch

        expected = framed.copy()
        for r, c in fill_cells:
            expected[2 + r, 2 + c] = seed_color

        rooms, _bg, _wall, gap_cells = _analyze_rooms(framed)
        if len(rooms) != 1:
            continue
        room = rooms[0]
        if room["gap_count"] != entrance_count or room["seed"] != (2 + seed[0], 2 + seed[1]):
            continue
        if room["color"] != seed_color or len(gap_cells) != entrance_count:
            continue
        if not np.array_equal(main(framed), expected):
            continue

        return {
            "patch": patch,
            "fill_cells": fill_cells,
            "seed_color": seed_color,
            "entrance_count": entrance_count,
        }
    return None


def _place_rooms(grid: np.ndarray, rooms: list[dict], rng: random.Random) -> tuple[list[tuple[dict, int, int]], np.ndarray] | None:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[dict, int, int]] = []

    for room in rooms:
        patch = room["patch"]
        ph, pw = patch.shape
        placed = False
        for _ in range(250):
            r0 = rng.randint(2, h - ph - 2)
            c0 = rng.randint(2, w - pw - 2)
            r1, r2 = max(0, r0 - 2), min(h, r0 + ph + 2)
            c1, c2 = max(0, c0 - 2), min(w, c0 + pw + 2)
            if occupied[r1:r2, c1:c2].any():
                continue
            occupied[r1:r2, c1:c2] = True
            placements.append((room, r0, c0))
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for room, r0, c0 in placements:
        patch = room["patch"]
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        expected[r0 : r0 + ph, c0 : c0 + pw] = patch
        for rr, cc in room["fill_cells"]:
            expected[r0 + rr, c0 + cc] = room["seed_color"]

    return placements, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.choice([3, 4])
    entrance_counts = [1, 2, 3]
    if room_count == 4:
        entrance_counts.append(rng.choice([1, 2, 3]))
    rng.shuffle(entrance_counts)

    rooms = []
    for entrance_count in entrance_counts:
        room = _build_room(rng, entrance_count)
        if room is None:
            return None
        rooms.append(room)

    grid_h = rng.randint(24, 34)
    grid_w = rng.randint(24, 34)
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

    placed = _place_rooms(grid, rooms, rng)
    if placed is None:
        return None
    _placements, expected = placed

    found_rooms, bg, wall, gap_cells = _analyze_rooms(grid)
    if bg != BG_COLOR or wall != WALL_COLOR:
        return None
    if len(found_rooms) != room_count:
        return None
    if len(gap_cells) != sum(entrance_counts):
        return None
    if sorted(room["gap_count"] for room in found_rooms) != sorted(entrance_counts):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-entrance puzzle")
