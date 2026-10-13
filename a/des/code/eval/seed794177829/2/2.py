from __future__ import annotations

import random

import numpy as np


BG = 8
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
ROT_STEPS = {"right": 0, "down": 1, "left": 2, "up": 3}
ALL_COLORS = tuple(range(10))
CANONICAL_VARIANT = {direction: (-1, 0) for direction in DIRS}

GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
ARROW_COUNT_RANGE = (2, 4)
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 400


def _majority_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _rotate_cw(point: tuple[int, int], steps: int) -> tuple[int, int]:
    r, c = map(int, point)
    for _ in range(int(steps) % 4):
        r, c = c, -r
    return int(r), int(c)


def _raw_arrow_relative(
    direction: str,
    tail_len: int,
    skew: int,
    anchor_row: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Arrow cells relative to the tip at (0, 0)."""
    assert direction in DIRS
    assert tail_len in (1, 2, 3, 4)
    assert skew in (-1, 1)
    assert anchor_row in (0, skew)

    head = [(0, 0), (0, -1), (skew, -1)]
    tail = [(anchor_row, -1 - i) for i in range(1, tail_len + 1)]
    steps = ROT_STEPS[direction]
    coords = [_rotate_cw(cell, steps) for cell in head + tail]
    tail_coords = [_rotate_cw(cell, steps) for cell in tail]
    return coords, tail_coords


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    norm = tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in pts))
    return norm, (int(min_r), int(min_c))


def _build_pattern_table() -> dict[tuple[tuple[int, int], ...], dict]:
    table: dict[tuple[tuple[int, int], ...], dict] = {}
    for direction in DIRS:
        for tail_len in (1, 2, 3, 4):
            skew, anchor_row = CANONICAL_VARIANT[direction]
            coords, tail = _raw_arrow_relative(direction, tail_len, skew, anchor_row)
            norm, (min_r, min_c) = _normalize(coords)
            tip_norm = (int(-min_r), int(-min_c))
            tail_norm = tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in tail))
            head_norm = tuple(sorted(cell for cell in norm if cell not in set(tail_norm)))
            info = {
                "direction": direction,
                "tail_len": int(tail_len),
                "coords_norm": norm,
                "tip_norm": tip_norm,
                "tail_norm": tail_norm,
                "head_norm": head_norm,
                "skew": int(skew),
                "anchor_row": int(anchor_row),
            }
            assert norm not in table, "canonical arrow patterns must be unique"
            table[norm] = info
    return table


PATTERNS = _build_pattern_table()


def _scan_edge_runs(column: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    runs: list[tuple[int, int, int, int]] = []
    r = 0
    h = int(column.shape[0])
    while r < h:
        color = int(column[r])
        if color == bg:
            r += 1
            continue
        r0 = int(r)
        while r + 1 < h and int(column[r + 1]) == color:
            r += 1
        r1 = int(r)
        runs.append((r0, r1, color, int(r1 - r0 + 1)))
        r += 1
    return runs


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, dict[int, int], set[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, dict[int, int], set[int]]] = []
    for edge_col in (0, int(w - 1)):
        runs = _scan_edge_runs(grid[:, edge_col], bg)
        if len(runs) != 4:
            continue
        heights = [int(run[3]) for run in runs]
        colors = [int(run[2]) for run in runs]
        if heights != [1, 2, 3, 4]:
            continue
        if len(set(colors)) != 4:
            continue

        ok = True
        for r0, r1, color, height in runs:
            coords = np.argwhere(grid == int(color))
            if coords.shape[0] != int(height):
                ok = False
                break
            if not np.all(coords[:, 1] == int(edge_col)):
                ok = False
                break
            if int(coords[:, 0].min()) != int(r0) or int(coords[:, 0].max()) != int(r1):
                ok = False
                break
        if ok:
            mapping = {int(height): int(color) for _r0, _r1, color, height in runs}
            candidates.append((int(edge_col), mapping, set(colors)))

    assert len(candidates) == 1, "expected exactly one edge legend column"
    return candidates[0]


def _match_arrow(coords: list[tuple[int, int]]) -> dict:
    norm, (min_r, min_c) = _normalize(coords)
    assert norm in PATTERNS, "foreground object is not a valid arrow"
    pattern = PATTERNS[norm]
    tip_r = int(min_r + pattern["tip_norm"][0])
    tip_c = int(min_c + pattern["tip_norm"][1])
    tail = {
        (int(min_r + r), int(min_c + c))
        for r, c in pattern["tail_norm"]
    }
    head = {
        (int(min_r + r), int(min_c + c))
        for r, c in pattern["head_norm"]
    }
    return {
        "direction": pattern["direction"],
        "tip": (tip_r, tip_c),
        "tail": tail,
        "head": head,
        "cells": set((int(r), int(c)) for r, c in coords),
        "tail_len": int(pattern["tail_len"]),
    }


def _parse_scene(grid: np.ndarray) -> tuple[int, int, dict[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _majority_color(g)
    legend_col, legend_map, legend_colors = _find_legend(g, bg)

    arrow_colors = [
        int(color)
        for color in np.unique(g)
        if int(color) != int(bg) and int(color) not in legend_colors
    ]
    assert arrow_colors, "expected at least one arrow"

    arrows: list[dict] = []
    for color in arrow_colors:
        coords = [tuple(map(int, cell)) for cell in np.argwhere(g == int(color))]
        assert coords, "foreground arrow colors must be present"
        arrow = _match_arrow(coords)
        arrow["color"] = int(color)
        arrow["recolor"] = None
        arrows.append(arrow)

    tail_owner: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow["tail"]:
            assert cell not in tail_owner, "tail pixels must be unique"
            tail_owner[cell] = int(idx)

    for idx, arrow in enumerate(arrows):
        r, c = map(int, arrow["tip"])
        dr, dc = DIRS[arrow["direction"]]
        length = 0
        stop = "boundary"
        while True:
            r += int(dr)
            c += int(dc)
            if not (0 <= r < h and 0 <= c < w):
                break
            length += 1
            owner = tail_owner.get((int(r), int(c)))
            if owner is not None and int(owner) != int(idx):
                stop = "tail"
                break

        assert length in legend_map, "ray length must be encoded by legend"
        arrow["length"] = int(length)
        arrow["stop"] = stop
        arrow["recolor"] = int(legend_map[int(length)])

    return int(bg), int(legend_col), legend_map, arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each arrow by ray length, where the legend maps lengths 1..4 to colors.
    """
    g = np.asarray(grid, dtype=int)
    _bg, _legend_col, _legend_map, arrows = _parse_scene(g)
    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[int(r), int(c)] = int(arrow["recolor"])
    return out


def _arrow_from_tip(
    tip: tuple[int, int],
    direction: str,
    tail_len: int,
    skew: int,
    anchor_row: int,
    color: int,
) -> dict:
    tip_r, tip_c = map(int, tip)
    rel_coords, rel_tail = _raw_arrow_relative(direction, tail_len, skew, anchor_row)
    cells = {(int(tip_r + r), int(tip_c + c)) for r, c in rel_coords}
    tail = {(int(tip_r + r), int(tip_c + c)) for r, c in rel_tail}
    return {
        "color": int(color),
        "direction": direction,
        "tip": (int(tip_r), int(tip_c)),
        "tail_len": int(tail_len),
        "cells": cells,
        "tail": tail,
        "head": set(cells) - set(tail),
        "skew": int(skew),
        "anchor_row": int(anchor_row),
    }


def _ray_cells(tip: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = map(int, tip)
    path = []
    for _ in range(int(length)):
        r += int(dr)
        c += int(dc)
        path.append((int(r), int(c)))
    return path


def _fits_arrow(arrow: dict, shape: tuple[int, int], legend_col: int, occupied: set[tuple[int, int]], blocked: set[tuple[int, int]]) -> bool:
    h, w = map(int, shape)
    for r, c in arrow["cells"]:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(c) == int(legend_col):
            return False
        if (int(r), int(c)) in occupied:
            return False
        if (int(r), int(c)) in blocked:
            return False
    return True


def _place_boundary_arrow(
    shape: tuple[int, int],
    legend_col: int,
    color: int,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> tuple[dict, list[tuple[int, int]]] | None:
    h, w = map(int, shape)
    for _ in range(PLACEMENT_RETRIES):
        direction = random.choice(list(DIRS))
        length = random.randint(1, 4)
        tail_len = random.randint(1, 4)
        skew, anchor_row = CANONICAL_VARIANT[direction]

        if direction == "right":
            tip_c = int(w - 1 - length)
            tip_r = random.randint(0, h - 1)
        elif direction == "left":
            tip_c = int(length)
            tip_r = random.randint(0, h - 1)
        elif direction == "down":
            tip_r = int(h - 1 - length)
            tip_c = random.randint(0, w - 1)
        else:
            tip_r = int(length)
            tip_c = random.randint(0, w - 1)

        arrow = _arrow_from_tip((tip_r, tip_c), direction, tail_len, skew, anchor_row, color)
        path = _ray_cells(arrow["tip"], direction, length)
        if not path:
            continue
        if not _fits_arrow(arrow, shape, legend_col, occupied, blocked):
            continue
        if any(cell in occupied for cell in path):
            continue
        if any(not (0 <= cell[0] < h and 0 <= cell[1] < w) for cell in path):
            continue
        if tuple(path[-1]) not in {
            (0, path[-1][1]),
            (h - 1, path[-1][1]),
            (path[-1][0], 0),
            (path[-1][0], w - 1),
        }:
            continue
        return arrow, path
    return None


def _place_target_arrow(
    shape: tuple[int, int],
    legend_col: int,
    color: int,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    targets: list[dict],
) -> tuple[dict, list[tuple[int, int]]] | None:
    h, w = map(int, shape)
    choices = []
    for target in targets:
        for tail_cell in target["tail"]:
            choices.append((target, tail_cell))
    random.shuffle(choices)

    for target, tail_cell in choices:
        tail_r, tail_c = map(int, tail_cell)
        for _ in range(PLACEMENT_RETRIES):
            direction = random.choice(list(DIRS))
            length = random.randint(1, 4)
            dr, dc = DIRS[direction]
            tip = (int(tail_r - dr * length), int(tail_c - dc * length))
            tail_len = random.randint(1, 4)
            skew, anchor_row = CANONICAL_VARIANT[direction]
            arrow = _arrow_from_tip(tip, direction, tail_len, skew, anchor_row, color)
            path = _ray_cells(arrow["tip"], direction, length)
            if not path or tuple(path[-1]) != (tail_r, tail_c):
                continue
            if not _fits_arrow(arrow, shape, legend_col, occupied, blocked):
                continue
            if any(cell in occupied for cell in path[:-1]):
                continue
            if any(not (0 <= cell[0] < h and 0 <= cell[1] < w) for cell in path):
                continue
            return arrow, path
    return None


def _paint_legend(grid: np.ndarray, legend_col: int, legend_map: dict[int, int]) -> None:
    gaps = [1, 1, 1]
    slack = int(grid.shape[0] - 13)
    extra_count = random.randint(0, min(3, slack))
    for idx in random.sample(range(3), k=extra_count):
        gaps[idx] += 1
    used = 10 + sum(gaps)
    cursor = random.randint(0, int(grid.shape[0] - used))
    for height in (1, 2, 3, 4):
        color = int(legend_map[height])
        grid[cursor : cursor + height, legend_col] = color
        if height < 4:
            cursor += height + gaps[height - 1]


def _render_scene(
    shape: tuple[int, int],
    legend_col: int,
    legend_map: dict[int, int],
    arrows: list[dict],
) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), BG, dtype=int)
    _paint_legend(grid, int(legend_col), legend_map)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[int(r), int(c)] = int(arrow["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    legend_col = random.choice((0, w - 1))

    total_arrows = random.randint(*ARROW_COUNT_RANGE)
    boundary_count = random.randint(1, total_arrows - 1)
    target_count = int(total_arrows - boundary_count)

    colors = [color for color in ALL_COLORS if color != BG]
    random.shuffle(colors)
    legend_colors = colors[:4]
    arrow_colors = colors[4 : 4 + total_arrows]
    if len(arrow_colors) != total_arrows:
        return None

    legend_map = {height: int(color) for height, color in zip((1, 2, 3, 4), legend_colors, strict=False)}

    arrows: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    blocked_by_old_rays: set[tuple[int, int]] = set()

    for color in arrow_colors[:boundary_count]:
        placed = _place_boundary_arrow((h, w), legend_col, int(color), occupied, blocked_by_old_rays)
        if placed is None:
            return None
        arrow, ray = placed
        arrows.append(arrow)
        occupied.update(arrow["cells"])
        blocked_by_old_rays.update(ray)

    for color in arrow_colors[boundary_count:]:
        placed = _place_target_arrow((h, w), legend_col, int(color), occupied, blocked_by_old_rays, arrows)
        if placed is None:
            return None
        arrow, ray = placed
        arrows.append(arrow)
        occupied.update(arrow["cells"])
        blocked_by_old_rays.update(ray)

    grid = _render_scene((h, w), legend_col, legend_map, arrows)

    try:
        bg, parsed_legend_col, parsed_legend_map, parsed_arrows = _parse_scene(grid)
    except AssertionError:
        return None

    if int(bg) != BG or int(parsed_legend_col) != int(legend_col) or parsed_legend_map != legend_map:
        return None
    if len(parsed_arrows) != total_arrows:
        return None
    if not any(arrow["stop"] == "boundary" for arrow in parsed_arrows):
        return None
    if not any(arrow["stop"] == "tail" for arrow in parsed_arrows):
        return None

    output = main(grid)
    if output.shape != grid.shape:
        return None
    if np.array_equal(output, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GENERATION_RETRIES} attempts")
