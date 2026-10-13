from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg, in_bounds, neighbors4


BG = Color.BLACK
WALL = Color.GREY
SOURCE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_GEN_ATTEMPTS = 400
SIDE_ROTATIONS = {"top": 0, "left": 1, "bottom": 2, "right": 3}
SIDE_PAIRS = [
    ("top", "bottom"),
    ("left", "right"),
]
LOCAL_WIDTH = 13


def _wall_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != int(bg)]
    assert colors, "expected at least one non-background color"
    return int(max(colors, key=lambda c: int(np.count_nonzero(grid == c))))


def _source_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    if c == w - 1:
        return 0, -1
    raise AssertionError("source must lie on the border")


def _wall_openings(grid: np.ndarray, wall_color: int, hit_r: int, hit_c: int, dr: int, dc: int) -> list[tuple[int, int]]:
    if dr != 0:
        c0 = int(hit_c)
        c1 = int(hit_c)
        while c0 - 1 >= 0 and int(grid[hit_r, c0 - 1]) == int(wall_color):
            c0 -= 1
        while c1 + 1 < grid.shape[1] and int(grid[hit_r, c1 + 1]) == int(wall_color):
            c1 += 1
        return [(int(hit_r), int(c0 - 1)), (int(hit_r), int(c1 + 1))]

    r0 = int(hit_r)
    r1 = int(hit_r)
    while r0 - 1 >= 0 and int(grid[r0 - 1, hit_c]) == int(wall_color):
        r0 -= 1
    while r1 + 1 < grid.shape[0] and int(grid[r1 + 1, hit_c]) == int(wall_color):
        r1 += 1
    return [(int(r0 - 1), int(hit_c)), (int(r1 + 1), int(hit_c))]


def _trace_rays(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    wall = _wall_color(g, bg)
    h, w = g.shape
    out = g.copy()

    stack: list[tuple[int, int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            value = int(g[r, c])
            if value in (int(bg), int(wall)):
                continue
            dr, dc = _source_direction(g.shape, int(r), int(c))
            stack.append((int(r), int(c), int(dr), int(dc), value))

    seen: set[tuple[int, int, int, int, int]] = set()
    while stack:
        r, c, dr, dc, color = stack.pop()
        state = (int(r), int(c), int(dr), int(dc), int(color))
        if state in seen:
            continue
        seen.add(state)

        cur_r = int(r)
        cur_c = int(c)
        while True:
            nr = cur_r + dr
            nc = cur_c + dc
            if not in_bounds(g.shape, nr, nc):
                break

            cell = int(g[nr, nc])
            if cell == int(wall):
                for br, bc in _wall_openings(g, wall, nr, nc, dr, dc):
                    if not in_bounds(g.shape, br, bc):
                        continue
                    if int(g[br, bc]) != int(bg):
                        continue
                    out[br, bc] = int(color)
                    stack.append((int(br), int(bc), int(dr), int(dc), int(color)))
                break

            if cell != int(bg):
                break

            out[nr, nc] = int(color)
            cur_r, cur_c = int(nr), int(nc)

    return out


def _fill_labeled_cavities(traced: np.ndarray, bg: int, wall: int) -> np.ndarray:
    out = traced.copy()
    h, w = traced.shape
    for comp in components_mask(traced == int(bg), connectivity=4):
        if comp.size == 0:
            continue
        if any(int(r) in (0, h - 1) or int(c) in (0, w - 1) for r, c in comp):
            continue

        adjacent_colors: set[int] = set()
        for r, c in comp:
            for nr, nc in neighbors4(int(r), int(c)):
                if not in_bounds(traced.shape, nr, nc):
                    continue
                value = int(traced[nr, nc])
                if value not in (int(bg), int(wall)):
                    adjacent_colors.add(value)

        if len(adjacent_colors) == 1:
            color = next(iter(adjacent_colors))
            out[comp[:, 0], comp[:, 1]] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace edge light rays through breakable wall openings, then fill the sealed cavities they label.

    Input structure:
    - Black background with several single-pixel colored light sources on the border.
    - Interior walls are all one shared color and form short horizontal or vertical segments.
    - Some background pockets are almost enclosed by walls and become fully sealed only after nearby rays are painted.

    Rule:
    - Each border source advances straight inward, coloring every background cell it traverses.
    - If the next cell is a wall segment perpendicular to the ray, that ray stops and branches from the open cells just beyond the two segment ends, continuing in the same inward direction.
    - After all rays are traced, every background component that no longer touches the border and is 4-neighbor-adjacent to exactly one ray color is filled with that color.
    - Wall cells and source pixels remain unchanged.

    Key insight:
    - The cavity color is not visible locally: it is determined by first simulating the full branching ray network, then using those painted rays as labels for the newly sealed background chambers.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    wall = _wall_color(g, bg)
    traced = _trace_rays(g)
    return _fill_labeled_cavities(traced, bg, wall)


def _rotate_for_side(grid: np.ndarray, side: str) -> np.ndarray:
    return np.rot90(grid, SIDE_ROTATIONS[side])


def _transform_coord(local_shape: tuple[int, int], side: str, r: int, c: int) -> tuple[int, int]:
    h, w = map(int, local_shape)
    if side == "top":
        return int(r), int(c)
    if side == "left":
        return int(w - 1 - c), int(r)
    if side == "bottom":
        return int(h - 1 - r), int(w - 1 - c)
    if side == "right":
        return int(c), int(h - 1 - r)
    raise AssertionError(f"unknown side: {side}")


def _paint_room_boundary(
    inp: np.ndarray,
    out: np.ndarray,
    cavity_cells: set[tuple[int, int]],
    corridor_cell: tuple[int, int],
) -> None:
    for r, c in cavity_cells:
        out[r, c] = out[corridor_cell]
        for nr, nc in neighbors4(int(r), int(c)):
            if not in_bounds(inp.shape, nr, nc):
                continue
            if (int(nr), int(nc)) in cavity_cells or (int(nr), int(nc)) == corridor_cell:
                continue
            inp[nr, nc] = WALL
            out[nr, nc] = WALL


def _add_room(
    inp: np.ndarray,
    out: np.ndarray,
    *,
    source_color: int,
    corridor_col: int,
    room_cols: tuple[int, int],
    doorway_col: int,
    top_row: int,
    height: int,
) -> dict:
    bottom_row = int(top_row + height - 1)
    door_row = random.randint(int(top_row), int(bottom_row))
    c0, c1 = map(int, room_cols)
    corridor_cell = (int(door_row), int(corridor_col))

    cavity_cells = {(int(r), int(c)) for r in range(top_row, bottom_row + 1) for c in range(c0, c1 + 1)}
    cavity_cells.add((int(door_row), int(doorway_col)))

    out[corridor_cell] = int(source_color)
    _paint_room_boundary(inp, out, cavity_cells, corridor_cell)
    for r, c in cavity_cells:
        out[r, c] = int(source_color)

    return {
        "cells": cavity_cells,
        "color": int(source_color),
        "corridor_cell": corridor_cell,
    }


def _local_layout() -> dict:
    if random.random() < 0.6:
        return {
            "left_col": 4,
            "right_col": 8,
            "wall_span": (5, 7),
            "room_plans": [
                {"branch": "left", "corridor_col": 4, "room_cols": (1, 2), "doorway_col": 3},
                {"branch": "right", "corridor_col": 8, "room_cols": (10, 11), "doorway_col": 9},
            ],
            "room_counts": [1, 2],
        }
    return {
        "left_col": 3,
        "right_col": 9,
        "wall_span": (4, 8),
        "room_plans": [
            {"branch": "left", "corridor_col": 3, "room_cols": (5, 6), "doorway_col": 4},
            {"branch": "right", "corridor_col": 9, "room_cols": (6, 7), "doorway_col": 8},
        ],
        "room_counts": [1],
    }


def _build_local_case(source_color: int) -> tuple[np.ndarray, np.ndarray, dict]:
    h = random.randint(10, 11)
    w = LOCAL_WIDTH
    split_row = random.randint(3, 4)
    root_col = w // 2
    layout = _local_layout()

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    inp[0, root_col] = int(source_color)
    out[0, root_col] = int(source_color)

    wall_c0, wall_c1 = layout["wall_span"]
    inp[split_row, wall_c0 : wall_c1 + 1] = WALL
    out[split_row, wall_c0 : wall_c1 + 1] = WALL

    for r in range(1, split_row):
        out[r, root_col] = int(source_color)

    leaf_cols = [int(layout["left_col"]), int(layout["right_col"])]
    for leaf_col in leaf_cols:
        out[split_row, leaf_col] = int(source_color)
        for r in range(split_row + 1, h):
            out[r, leaf_col] = int(source_color)

    room_plans = random.sample(layout["room_plans"], random.choice(layout["room_counts"]))
    room_specs = []
    for plan in room_plans:
        max_room_h = h - split_row - 4
        room_h = random.randint(2, max(2, max_room_h))
        room_top = random.randint(split_row + 2, h - room_h - 2)
        room_specs.append(
            _add_room(
                inp,
                out,
                source_color=int(source_color),
                corridor_col=int(plan["corridor_col"]),
                room_cols=tuple(plan["room_cols"]),
                doorway_col=int(plan["doorway_col"]),
                top_row=int(room_top),
                height=int(room_h),
            )
        )

    return inp, out, {"room_specs": room_specs, "split_row": int(split_row), "layout": layout, "root_col": int(root_col)}


def _placement_ranges(
    canvas_shape: tuple[int, int],
    sprite_shape: tuple[int, int],
    side: str,
    band_sizes: dict[str, tuple[int, int]],
) -> tuple[range, range]:
    h, w = map(int, canvas_shape)
    sprite_h, sprite_w = map(int, sprite_shape)
    top_h, _ = band_sizes.get("top", (0, 0))
    bottom_h, _ = band_sizes.get("bottom", (0, 0))
    _, left_w = band_sizes.get("left", (0, 0))
    _, right_w = band_sizes.get("right", (0, 0))

    if side == "top":
        return range(0, 1), range(left_w, w - right_w - sprite_w + 1)
    if side == "bottom":
        return range(h - sprite_h, h - sprite_h + 1), range(left_w, w - right_w - sprite_w + 1)
    if side == "left":
        return range(top_h, h - bottom_h - sprite_h + 1), range(0, 1)
    if side == "right":
        return range(top_h, h - bottom_h - sprite_h + 1), range(w - sprite_w, w - sprite_w + 1)
    raise AssertionError(f"unknown side: {side}")


def _paste(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int) -> bool:
    h, w = sprite.shape
    window = canvas[top : top + h, left : left + w]
    if window.shape != sprite.shape:
        return False
    if np.any((window != BG) & (sprite != BG)):
        return False
    mask = sprite != BG
    window[mask] = sprite[mask]
    return True


def _line_positions(side: str, local_shape: tuple[int, int], top: int, left: int, meta: dict) -> set[int]:
    local_cols = [int(meta["root_col"]), int(meta["layout"]["left_col"]), int(meta["layout"]["right_col"])]
    positions = set()
    for c in local_cols:
        rr, cc = _transform_coord(local_shape, side, 0, c)
        if side in ("top", "bottom"):
            positions.add(int(left + cc))
        else:
            positions.add(int(top + rr))
    return positions


def _extend_constructed_rays(out: np.ndarray, inp: np.ndarray, side: str, top: int, left: int, local_shape: tuple[int, int], meta: dict, color: int) -> None:
    src_r, src_c = _transform_coord(local_shape, side, 0, meta["root_col"])
    dr, dc = _source_direction(out.shape, int(top + src_r), int(left + src_c))
    for leaf_col in (int(meta["layout"]["left_col"]), int(meta["layout"]["right_col"])):
        end_r, end_c = _transform_coord(local_shape, side, local_shape[0] - 1, leaf_col)
        cur_r = int(top + end_r)
        cur_c = int(left + end_c)
        nr = cur_r + dr
        nc = cur_c + dc
        while in_bounds(out.shape, nr, nc) and int(inp[nr, nc]) == BG:
            out[nr, nc] = int(color)
            nr += dr
            nc += dc


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    sides = random.choice(SIDE_PAIRS)
    colors = random.sample(SOURCE_COLORS, len(sides))
    local_cases = {}
    rotated = {}

    for side, color in zip(sides, colors):
        inp_local, out_local, meta_local = _build_local_case(int(color))
        local_cases[side] = (inp_local, out_local, meta_local, int(color))
        rotated[side] = (_rotate_for_side(inp_local, side), _rotate_for_side(out_local, side))

    band_sizes = {}
    for side in ("top", "bottom", "left", "right"):
        if side in rotated:
            band_sizes[side] = rotated[side][0].shape
        else:
            band_sizes[side] = (0, 0)

    top_h = band_sizes["top"][0]
    bottom_h = band_sizes["bottom"][0]
    left_h, left_w = band_sizes["left"]
    right_h, right_w = band_sizes["right"]
    h_min = top_h + bottom_h + max(1, left_h, right_h)
    w_min = left_w + right_w + max(1, band_sizes["top"][1], band_sizes["bottom"][1])
    if sides == ("top", "bottom"):
        w_min = max(w_min, 24)
    if sides == ("left", "right"):
        h_min = max(h_min, 24)
    h = random.randint(h_min, min(30, h_min + 4))
    w = random.randint(w_min, min(30, w_min + 4))

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    placements = {}

    order = [side for side in ("top", "bottom", "left", "right") if side in rotated]
    for _ in range(100):
        inp.fill(BG)
        out.fill(BG)
        placements.clear()
        line_sets: dict[str, set[int]] = {}
        ok = True
        for side in order:
            inp_zone, out_zone = rotated[side]
            inp_local, _out_local, meta_local, color = local_cases[side]
            row_range, col_range = _placement_ranges(inp.shape, inp_zone.shape, side, band_sizes)
            top = random.choice(list(row_range))
            left = random.choice(list(col_range))
            lines = _line_positions(side, inp_local.shape, top, left, meta_local)
            if line_sets and any(lines & other for other in line_sets.values()):
                ok = False
                break
            if not _paste(inp, inp_zone, top, left):
                ok = False
                break
            if not _paste(out, out_zone, top, left):
                ok = False
                break
            placements[side] = (int(top), int(left), inp_zone.shape, int(color))
            line_sets[side] = lines
        if ok:
            break
    else:
        raise RuntimeError("failed to place non-overlapping opposite-side scenes")

    for side in order:
        top, left, _shape, color = placements[side]
        inp_local, _out_local, meta_local, _color = local_cases[side]
        _extend_constructed_rays(out, inp, side, top, left, inp_local.shape, meta_local, color)

    return inp, out, {"placements": placements, "local_cases": local_cases}


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if get_bg(inp) != BG:
        return False

    h, w = inp.shape
    wall = _wall_color(inp, BG)
    source_cells = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            value = int(inp[r, c])
            if value not in (BG, wall):
                source_cells.append((int(r), int(c), value))
    if len(source_cells) != 2 or len({cell[2] for cell in source_cells}) != 2:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    traced = _trace_rays(inp)
    cavities = 0
    for comp in components_mask(traced == BG, connectivity=4):
        if comp.size == 0:
            continue
        if any(int(r) in (0, h - 1) or int(c) in (0, w - 1) for r, c in comp):
            continue
        adjacent_colors = set()
        for r, c in comp:
            for nr, nc in neighbors4(int(r), int(c)):
                if not in_bounds(traced.shape, nr, nc):
                    continue
                value = int(traced[nr, nc])
                if value not in (BG, wall):
                    adjacent_colors.add(value)
        if len(adjacent_colors) != 1:
            return False
        color = next(iter(adjacent_colors))
        if not np.all(solved[comp[:, 0], comp[:, 1]] == int(color)):
            return False
        cavities += 1

    return cavities >= 2


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        inp, expected, _meta = _build_case()
        if _semantic_checks(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid light-cavity ARC input")
