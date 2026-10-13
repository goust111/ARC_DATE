from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

PRIMARY_COLORS = (RED, BLUE, YELLOW)
SECONDARY_COLORS = (GREEN, MAGENTA, ORANGE)
LEGEND_TABLE = np.array(
    [
        [RED, MAGENTA, ORANGE],
        [MAGENTA, BLUE, GREEN],
        [ORANGE, GREEN, YELLOW],
    ],
    dtype=int,
)

GEN_RETRIES = 400
SIZE_RANGE = (13, 20)
MIN_WALL_GAP = 4


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _legend_bounds(shape: tuple[int, int]) -> tuple[int, int]:
    return shape[0] - 3, shape[1] - 3


def _legend_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0 = _legend_bounds(shape)
    mask[r0:, c0:] = True
    return mask


def _parse_legend(grid: np.ndarray) -> tuple[tuple[int, int, int], dict[frozenset[int], int]]:
    h, w = grid.shape
    assert h >= 3 and w >= 3, "grid too small for a 3x3 legend"
    legend = np.asarray(grid[h - 3 :, w - 3 :], dtype=int)
    diagonal = tuple(int(legend[i, i]) for i in range(3))
    assert len(set(diagonal)) == 3, "legend diagonal must list the three primary colors"

    mix_map: dict[frozenset[int], int] = {}
    for i in range(3):
        for j in range(3):
            assert int(legend[i, j]) == int(legend[j, i]), "legend must be symmetric"
            mix_map[frozenset((diagonal[i], diagonal[j]))] = int(legend[i, j])
    return diagonal, mix_map


def _components_for_color(grid: np.ndarray, color: int, blocked: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if blocked[r, c] or seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not blocked[nr, nc]
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ray_length(grid: np.ndarray, head: tuple[int, int], direction: tuple[int, int]) -> int:
    dr, dc = direction
    r, c = head[0] + dr, head[1] + dc
    length = 0
    while _in_bounds(grid.shape, r, c) and int(grid[r, c]) == BLACK:
        length += 1
        r += dr
        c += dc
    return length


def _infer_arrow(
    grid: np.ndarray, color: int, cells: list[tuple[int, int]]
) -> dict[str, object]:
    assert len(cells) == 2, "each arrow must occupy exactly two cells"
    a, b = cells
    assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "arrow cells must be orthogonally adjacent"

    candidates = []
    for head, tail in ((a, b), (b, a)):
        direction = (head[0] - tail[0], head[1] - tail[1])
        forward = (head[0] + direction[0], head[1] + direction[1])
        backward = (tail[0] - direction[0], tail[1] - direction[1])
        immediate_open = _in_bounds(grid.shape, *forward) and int(grid[forward]) == BLACK
        backed = (not _in_bounds(grid.shape, *backward)) or int(grid[backward]) != BLACK
        length = _ray_length(grid, head, direction)
        score = (int(immediate_open), int(backed), int(length))
        candidates.append((score, head, tail, direction, length))

    candidates.sort(reverse=True)
    best, second = candidates[0], candidates[1]
    assert best[0] > second[0], "arrow orientation must be uniquely identifiable"
    assert best[4] > 0, "every arrow must have at least one cell to propagate into"

    return {
        "color": color,
        "head": best[1],
        "tail": best[2],
        "dir": best[3],
        "cells": {best[1], best[2]},
        "path_len": best[4],
    }


def _find_arrows(grid: np.ndarray, primary_colors: Iterable[int]) -> list[dict[str, object]]:
    blocked = _legend_mask(grid.shape)
    arrows: list[dict[str, object]] = []
    for color in primary_colors:
        for comp in _components_for_color(grid, int(color), blocked):
            arrows.append(_infer_arrow(grid, int(color), comp))
    assert arrows, "expected at least one arrow"
    return arrows


def _collect_coverage(
    grid: np.ndarray, arrows: list[dict[str, object]]
) -> dict[tuple[int, int], set[int]]:
    coverage: dict[tuple[int, int], set[int]] = {}
    for arrow in arrows:
        color = int(arrow["color"])
        head = arrow["head"]
        dr, dc = arrow["dir"]
        r, c = head[0] + dr, head[1] + dc
        while _in_bounds(grid.shape, r, c) and int(grid[r, c]) == BLACK:
            coverage.setdefault((r, c), set()).add(color)
            r += dr
            c += dc
    return coverage


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    primary_order, mix_map = _parse_legend(grid)
    arrows = _find_arrows(grid, primary_order)
    coverage = _collect_coverage(grid, arrows)
    return {"primary_order": primary_order, "mix_map": mix_map, "arrows": arrows, "coverage": coverage}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each arrow's color in its facing direction until a wall, boundary, or
    another non-background structure blocks the ray. If two different rays cover the
    same background cell, recolor that cell using the bottom-right 3x3 mixing legend.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    details = _analyze_grid(g)
    mix_map: dict[frozenset[int], int] = details["mix_map"]
    coverage: dict[tuple[int, int], set[int]] = details["coverage"]

    for (r, c), colors in coverage.items():
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        else:
            assert len(colors) == 2, "generator guarantees at most pairwise intersections"
            out[r, c] = mix_map[frozenset(colors)]
    return out


def _pick_spaced_positions(start: int, end: int, max_count: int) -> list[int]:
    if start > end or max_count <= 0:
        return []
    target = random.randint(0, max_count)
    pool = list(range(start, end + 1))
    random.shuffle(pool)
    picked: list[int] = []
    for value in pool:
        if all(abs(value - other) >= MIN_WALL_GAP for other in picked):
            picked.append(value)
            if len(picked) == target:
                break
    return sorted(picked)


def _open_intervals(limit: int, walls: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(walls) + [limit]
    intervals = []
    for left, right in zip(bounds, bounds[1:], strict=False):
        start = left + 1
        end = right - 1
        if start <= end:
            intervals.append((start, end))
    return intervals


def _rooms(split_r: int, split_c: int, h_walls: list[int], v_walls: list[int]) -> list[tuple[int, int, int, int]]:
    rooms: list[tuple[int, int, int, int]] = []
    for r0, r1 in _open_intervals(split_r, h_walls):
        for c0, c1 in _open_intervals(split_c, v_walls):
            if r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3:
                rooms.append((r0, r1, c0, c1))
    return rooms


def _make_horizontal(room: tuple[int, int, int, int], row: int, sign: int, color: int) -> dict[str, object]:
    r0, r1, c0, c1 = room
    assert r0 <= row <= r1
    assert sign in (-1, 1)
    if sign == 1:
        tail = (row, c0)
        head = (row, c0 + 1)
        path = {(row, c) for c in range(c0 + 2, c1 + 1)}
    else:
        tail = (row, c1)
        head = (row, c1 - 1)
        path = {(row, c) for c in range(c0, c1 - 1)}
    assert path
    return {
        "color": int(color),
        "head": head,
        "tail": tail,
        "dir": (0, sign),
        "cells": {head, tail},
        "path": path,
        "orientation": "h",
    }


def _make_vertical(room: tuple[int, int, int, int], col: int, sign: int, color: int) -> dict[str, object]:
    r0, r1, c0, c1 = room
    assert c0 <= col <= c1
    assert sign in (-1, 1)
    if sign == 1:
        tail = (r0, col)
        head = (r0 + 1, col)
        path = {(r, col) for r in range(r0 + 2, r1 + 1)}
    else:
        tail = (r1, col)
        head = (r1 - 1, col)
        path = {(r, col) for r in range(r0, r1 - 1)}
    assert path
    return {
        "color": int(color),
        "head": head,
        "tail": tail,
        "dir": (sign, 0),
        "cells": {head, tail},
        "path": path,
        "orientation": "v",
    }


def _cross_pair(room: tuple[int, int, int, int], colors: tuple[int, int]) -> tuple[dict[str, object], dict[str, object]]:
    r0, r1, c0, c1 = room
    h_sign = random.choice((-1, 1))
    v_sign = random.choice((-1, 1))

    if v_sign == 1:
        row_choices = list(range(r0 + 2, r1 + 1))
    else:
        row_choices = list(range(r0, r1 - 1))
    if h_sign == 1:
        col_choices = list(range(c0 + 2, c1 + 1))
    else:
        col_choices = list(range(c0, c1 - 1))

    row = random.choice(row_choices)
    col = random.choice(col_choices)
    horiz = _make_horizontal(room, row, h_sign, colors[0])
    vert = _make_vertical(room, col, v_sign, colors[1])
    assert horiz["path"] & vert["path"] == {(row, col)}
    assert not (horiz["cells"] & vert["cells"])
    return horiz, vert


def _parallel_arrows(room: tuple[int, int, int, int], orientation: str, count: int) -> list[dict[str, object]]:
    r0, r1, c0, c1 = room
    assert orientation in ("h", "v")
    assert 2 <= count <= 3

    color_pool = list(PRIMARY_COLORS)
    random.shuffle(color_pool)
    if random.random() < 0.35:
        colors = [random.choice(PRIMARY_COLORS) for _ in range(count)]
    else:
        colors = (color_pool * 2)[:count]

    arrows: list[dict[str, object]] = []
    if orientation == "h":
        rows = random.sample(list(range(r0, r1 + 1)), count)
        for row, color in zip(sorted(rows), colors, strict=False):
            sign = random.choice((-1, 1))
            arrows.append(_make_horizontal(room, row, sign, color))
    else:
        cols = random.sample(list(range(c0, c1 + 1)), count)
        for col, color in zip(sorted(cols), colors, strict=False):
            sign = random.choice((-1, 1))
            arrows.append(_make_vertical(room, col, sign, color))
    return arrows


def _paint_walls(grid: np.ndarray, split_r: int, split_c: int, h_walls: list[int], v_walls: list[int]) -> None:
    grid[split_r, :] = GREY
    grid[:, split_c] = GREY
    for row in h_walls:
        grid[row, :] = GREY
    for col in v_walls:
        grid[:, col] = GREY


def _paint_legend(grid: np.ndarray) -> None:
    r0, c0 = _legend_bounds(grid.shape)
    grid[r0:, c0:] = LEGEND_TABLE


def _paint_arrows(grid: np.ndarray, arrows: list[dict[str, object]]) -> None:
    for arrow in arrows:
        color = int(arrow["color"])
        for r, c in arrow["cells"]:
            assert int(grid[r, c]) == BLACK
            grid[r, c] = color


def _build_candidate(require_mix: bool) -> tuple[np.ndarray, list[dict[str, object]]]:
    h = random.randint(*SIZE_RANGE)
    w = random.randint(*SIZE_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    split_r = h - 4
    split_c = w - 4
    max_h_walls = 2 if split_r >= 11 else 1
    max_v_walls = 2 if split_c >= 11 else 1
    h_walls = _pick_spaced_positions(3, split_r - 4, max_h_walls)
    v_walls = _pick_spaced_positions(3, split_c - 4, max_v_walls)
    rooms = _rooms(split_r, split_c, h_walls, v_walls)
    assert rooms, "expected at least one usable room"

    room = random.choice(rooms)
    arrows: list[dict[str, object]]
    if require_mix:
        base_colors = tuple(random.sample(PRIMARY_COLORS, 2))
        horiz, vert = _cross_pair(room, base_colors)
        arrows = [horiz, vert]
        if random.random() < 0.6:
            remaining = [c for c in PRIMARY_COLORS if c not in base_colors]
            extra_color = remaining[0] if remaining and random.random() < 0.7 else random.choice(PRIMARY_COLORS)
            if random.random() < 0.5:
                row_choices = [r for r in range(room[0], room[1] + 1) if r != horiz["head"][0]]
                extra_row = random.choice(row_choices)
                arrows.append(_make_horizontal(room, extra_row, horiz["dir"][1], extra_color))
            else:
                col_choices = [c for c in range(room[2], room[3] + 1) if c != vert["head"][1]]
                extra_col = random.choice(col_choices)
                arrows.append(_make_vertical(room, extra_col, vert["dir"][0], extra_color))
    else:
        orientation = random.choice(("h", "v"))
        count = random.randint(2, 3)
        arrows = _parallel_arrows(room, orientation, count)

    _paint_walls(grid, split_r, split_c, h_walls, v_walls)
    _paint_legend(grid)
    _paint_arrows(grid, arrows)
    return grid, arrows


def _validate_candidate(grid: np.ndarray, expected: list[dict[str, object]], require_mix: bool) -> bool:
    try:
        details = _analyze_grid(grid)
    except AssertionError:
        return False

    actual_arrows: list[dict[str, object]] = details["arrows"]
    actual_signature = {(arrow["head"], arrow["dir"], arrow["color"]) for arrow in actual_arrows}
    expected_signature = {(arrow["head"], arrow["dir"], arrow["color"]) for arrow in expected}
    if actual_signature != expected_signature:
        return False

    coverage: dict[tuple[int, int], set[int]] = details["coverage"]
    if any(len(colors) > 2 for colors in coverage.values()):
        return False
    mixed = [cell for cell, colors in coverage.items() if len(colors) == 2]
    if require_mix != bool(mixed):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    legend = _legend_mask(grid.shape)
    if not np.array_equal(out[legend], grid[legend]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        require_mix = random.random() < 0.5
        grid, arrows = _build_candidate(require_mix)
        if _validate_candidate(grid, arrows, require_mix):
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
