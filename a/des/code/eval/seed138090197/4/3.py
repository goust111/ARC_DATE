from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
SEED_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_GRID_SIZE = 30
ROOM_SIZE_MIN = 6
GEN_ATTEMPTS = 300
ROOM_ATTEMPTS = 120
EXTRA_ATTEMPTS = 40

BASE_CUP = [(-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
BASE_BAR = [(-1, 1), (0, 1), (1, 1)]
BASE_CORNER = [(-1, 1), (0, 1), (1, 1), (1, 0)]
BASE_RING = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]
EXTRA_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
]


def _rotate_offsets(offsets: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    pts = list(offsets)
    for _ in range(turns % 4):
        pts = [(c, -r) for r, c in pts]
    return pts


def _normalize_shape(offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in offsets)
    min_c = min(c for _, c in offsets)
    return sorted((r - min_r, c - min_c) for r, c in offsets)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _segments(full_lines: np.ndarray, size: int) -> list[tuple[int, int]]:
    bounds = [-1] + [int(x) for x in full_lines] + [size]
    return [(a + 1, b) for a, b in zip(bounds, bounds[1:]) if b - a > 1]


def _solve_room(room: np.ndarray, return_meta: bool = False):
    local = np.asarray(room, dtype=int)
    h, w = local.shape
    seed_mask = (local != BLACK) & (local != GREY) & (local != AZURE)
    seeds = np.argwhere(seed_mask)
    assert seeds.shape[0] == 1, "each room must contain exactly one seed"
    sr, sc = map(int, seeds[0])
    seed_color = int(local[sr, sc])

    stage1 = local.copy()
    painted = 0
    for dr, dc in DIRS:
        r, c = sr + dr, sc + dc
        while 0 <= r < h and 0 <= c < w and int(local[r, c]) != GREY and int(local[r, c]) != AZURE:
            if int(stage1[r, c]) == BLACK:
                stage1[r, c] = seed_color
                painted += 1
            r += dr
            c += dc

    out = stage1.copy()
    removed_components = 0
    kept_components = 0
    removed_cells = 0

    for comp in _components(local == GREY):
        touched = set()
        for r, c in comp:
            if r > 0 and int(stage1[r - 1, c]) == seed_color:
                touched.add("up")
            if r + 1 < h and int(stage1[r + 1, c]) == seed_color:
                touched.add("down")
            if c > 0 and int(stage1[r, c - 1]) == seed_color:
                touched.add("left")
            if c + 1 < w and int(stage1[r, c + 1]) == seed_color:
                touched.add("right")
        if len(touched) >= 3:
            removed_components += 1
            removed_cells += len(comp)
            for r, c in comp:
                out[r, c] = seed_color
        else:
            kept_components += 1

    if not return_meta:
        return out
    return out, {
        "painted": painted,
        "removed_components": removed_components,
        "kept_components": kept_components,
        "removed_cells": removed_cells,
    }


def _solve_grid(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    full_rows = np.flatnonzero(np.all(g == AZURE, axis=1))
    full_cols = np.flatnonzero(np.all(g == AZURE, axis=0))
    row_spans = _segments(full_rows, g.shape[0])
    col_spans = _segments(full_cols, g.shape[1])
    assert row_spans and col_spans, "expected azure row and column separators"

    out = g.copy()
    total_painted = 0
    total_removed_components = 0
    total_kept_components = 0
    total_removed_cells = 0

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            solved_room, meta = _solve_room(g[r0:r1, c0:c1], return_meta=True)
            out[r0:r1, c0:c1] = solved_room
            total_painted += int(meta["painted"])
            total_removed_components += int(meta["removed_components"])
            total_kept_components += int(meta["kept_components"])
            total_removed_cells += int(meta["removed_cells"])

    if not return_meta:
        return out
    return out, {
        "painted": total_painted,
        "removed_components": total_removed_components,
        "kept_components": total_kept_components,
        "removed_cells": total_removed_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint seed rays inside each azure-delimited room and remove walls touched from at least three sides.
    """
    return _solve_grid(grid)


def _sample_axis(room_count: int) -> list[int] | None:
    max_room = min(9, (MAX_GRID_SIZE - (room_count + 1)) // room_count)
    if max_room < ROOM_SIZE_MIN:
        return None
    return [random.randint(ROOM_SIZE_MIN, max_room) for _ in range(room_count)]


def _build_grid(row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    height = sum(row_sizes) + len(row_sizes) + 1
    width = sum(col_sizes) + len(col_sizes) + 1
    grid = np.full((height, width), BLACK, dtype=int)

    row_lines = [0]
    pos = 0
    for size in row_sizes:
        pos += size + 1
        row_lines.append(pos)

    col_lines = [0]
    pos = 0
    for size in col_sizes:
        pos += size + 1
        col_lines.append(pos)

    grid[row_lines, :] = AZURE
    grid[:, col_lines] = AZURE

    boxes: list[tuple[int, int, int, int]] = []
    r = 1
    for rh in row_sizes:
        c = 1
        for cw in col_sizes:
            boxes.append((r, r + rh, c, c + cw))
            c += cw + 1
        r += rh + 1
    return grid, boxes


def _can_place_extra(
    shape: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    seed: tuple[int, int],
    height: int,
    width: int,
) -> bool:
    for r, c in shape:
        if not (1 <= r < height - 1 and 1 <= c < width - 1):
            return False
        if (r, c) == seed or (r, c) in walls:
            return False
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) == seed:
                return False
            if (nr, nc) in walls and (nr, nc) not in shape:
                return False
    return True


def _try_add_extras(
    walls: set[tuple[int, int]],
    seed: tuple[int, int],
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    wall_cells = set(walls)
    max_extra = 2 if min(height, width) >= 7 else 1
    extra_count = random.randint(0, max_extra)
    for _ in range(extra_count):
        placed = False
        for _ in range(EXTRA_ATTEMPTS):
            base = _normalize_shape(_rotate_offsets(random.choice(EXTRA_SHAPES), random.randint(0, 3)))
            shape_h = max(r for r, _ in base) + 1
            shape_w = max(c for _, c in base) + 1
            top = random.randint(1, height - shape_h - 1)
            left = random.randint(1, width - shape_w - 1)
            shape = {(top + r, left + c) for r, c in base}
            if _can_place_extra(shape, wall_cells, seed, height, width):
                wall_cells |= shape
                placed = True
                break
        if not placed:
            break
    return wall_cells


def _make_room(height: int, width: int, want_removed: bool) -> np.ndarray | None:
    for _ in range(ROOM_ATTEMPTS):
        room = np.full((height, width), BLACK, dtype=int)
        seed = (
            random.randint(2, height - 3),
            random.randint(2, width - 3),
        )
        seed_color = random.choice(SEED_COLORS)
        room[seed] = seed_color

        if want_removed:
            if random.random() < 0.2:
                primary = BASE_RING
                turns = 0
            else:
                primary = BASE_CUP
                turns = random.randint(0, 3)
        else:
            primary = random.choice([BASE_BAR, BASE_CORNER])
            turns = random.randint(0, 3)

        offsets = _rotate_offsets(primary, turns)
        wall_cells = {(seed[0] + dr, seed[1] + dc) for dr, dc in offsets}
        if any(not (1 <= r < height - 1 and 1 <= c < width - 1) for r, c in wall_cells):
            continue

        wall_cells = _try_add_extras(wall_cells, seed, height, width)
        for r, c in wall_cells:
            room[r, c] = GREY

        solved, meta = _solve_room(room, return_meta=True)
        changed = int(np.count_nonzero(room != solved))
        if changed == 0:
            continue
        if want_removed and int(meta["removed_components"]) < 1:
            continue
        if not want_removed and int(meta["removed_components"]) != 0:
            continue
        if not want_removed and int(meta["painted"]) < 2:
            continue
        if int(meta["kept_components"]) < 1:
            continue
        return room
    return None


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(2, 4)
    room_cols = random.randint(2, 4)
    row_sizes = _sample_axis(room_rows)
    col_sizes = _sample_axis(room_cols)
    if row_sizes is None or col_sizes is None:
        return None

    grid, boxes = _build_grid(row_sizes, col_sizes)
    total_rooms = len(boxes)
    removable_rooms = random.randint(1, total_rooms - 1)
    room_types = [True] * removable_rooms + [False] * (total_rooms - removable_rooms)
    random.shuffle(room_types)

    used_seed_colors: list[int] = []
    for (r0, r1, c0, c1), want_removed in zip(boxes, room_types):
        room = _make_room(r1 - r0, c1 - c0, want_removed)
        if room is None:
            return None
        used_seed_colors.extend(
            int(v) for v in np.unique(room[(room != BLACK) & (room != GREY) & (room != AZURE)])
        )
        grid[r0:r1, c0:c1] = room

    if len(set(used_seed_colors)) < 2:
        return None

    out, meta = _solve_grid(grid, return_meta=True)
    if np.array_equal(out, grid):
        return None
    if int(meta["removed_components"]) < 1 or int(meta["kept_components"]) < 1:
        return None
    if int(meta["painted"]) < total_rooms:
        return None
    if int(meta["removed_cells"]) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
