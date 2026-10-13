from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import Color, bbox_coords, colors_present, components_mask, get_bg, neighbors4, paste, random_free_location


BG = int(Color.BLACK)
GREY = int(Color.GREY)
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
LEFT_TURN = {
    "U": "L",
    "L": "D",
    "D": "R",
    "R": "U",
}
RIGHT_TURN = {v: k for k, v in LEFT_TURN.items()}
OPPOSITE = {
    "U": "D",
    "D": "U",
    "L": "R",
    "R": "L",
}
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500


def _rotate_ccw_point(h: int, w: int, point: tuple[int, int]) -> tuple[int, int]:
    r, c = point
    return w - 1 - c, r


def _rot90_points(
    patch: np.ndarray,
    tip: tuple[int, int],
    content_cells: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    if direction == "U":
        return patch.copy(), tip, list(content_cells)

    if direction == "D":
        h, w = patch.shape
        rotated = np.flipud(patch).copy()
        new_tip = (h - 1 - tip[0], tip[1])
        new_cells = [(h - 1 - r, c) for r, c in content_cells]
        return rotated, new_tip, new_cells

    if direction == "L":
        h, w = patch.shape
        rotated = np.rot90(patch, k=1).copy()
        new_tip = _rotate_ccw_point(h, w, tip)
        new_cells = [_rotate_ccw_point(h, w, cell) for cell in content_cells]
        return rotated, new_tip, new_cells

    if direction == "R":
        h, w = patch.shape
        rotated = np.rot90(patch, k=-1).copy()
        new_tip = (tip[1], h - 1 - tip[0])
        new_cells = [(c, h - 1 - r) for r, c in content_cells]
        return rotated, new_tip, new_cells

    raise ValueError(f"unknown direction: {direction}")


def _build_cup_patch(
    border_color: int,
    content_color: int,
    direction: str,
    tail_len: int,
    body_h: int,
    width: int,
    content_count: int,
) -> dict:
    patch = np.full((tail_len + body_h, width), BG, dtype=int)
    mid = width // 2
    patch[:tail_len, mid] = border_color
    patch[tail_len, :] = border_color
    patch[tail_len + 1 :, 0] = border_color
    patch[tail_len + 1 :, -1] = border_color

    interior = [
        (r, c)
        for r in range(tail_len + 1, tail_len + body_h)
        for c in range(1, width - 1)
    ]
    chosen = random.sample(interior, content_count)
    for r, c in chosen:
        patch[r, c] = content_color

    rotated, tip, rotated_cells = _rot90_points(
        patch,
        (0, mid),
        chosen,
        direction,
    )
    return {
        "patch": rotated,
        "tip_local": tip,
        "content_local": rotated_cells,
        "direction": direction,
        "border_color": int(border_color),
        "content_color": int(content_color),
        "content_count": int(content_count),
    }


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _polyline_cells(
    start: tuple[int, int],
    dirs: list[str],
    lengths: list[int],
) -> list[tuple[int, int]]:
    path = [start]
    r, c = start
    for idx, direction in enumerate(dirs):
        dr, dc = DIRS[direction]
        steps = lengths[idx] - 1 if idx == 0 else lengths[idx]
        for _ in range(steps):
            r += dr
            c += dc
            path.append((r, c))
    return path


def _path_has_visible_turn(path: list[tuple[int, int]], visible_len: int) -> bool:
    upto = min(len(path), visible_len)
    for i in range(1, upto - 1):
        d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d1 != d2:
            return True
    return False


def _adjacent_to_existing_grey(
    path: list[tuple[int, int]],
    existing_grey: set[tuple[int, int]],
) -> bool:
    for cell in path:
        for nbr in neighbors4(*cell):
            if nbr in existing_grey:
                return True
    return False


def _build_path(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: str,
    total_len: int,
    visible_len: int,
    existing_grey: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    patterns = []
    for turn in random.sample([LEFT_TURN[direction], RIGHT_TURN[direction]], 2):
        patterns.append([direction, turn])
        patterns.append([direction, turn, direction])

    random.shuffle(patterns)
    for dirs in patterns:
        candidates: list[list[int]] = []
        if len(dirs) == 2:
            max_l1 = min(visible_len - 1, total_len - 1)
            for l1 in range(1, max_l1 + 1):
                candidates.append([l1, total_len - l1])
        else:
            max_l1 = min(visible_len - 1, total_len - 2)
            for l1 in range(1, max_l1 + 1):
                for l2 in range(1, total_len - l1):
                    l3 = total_len - l1 - l2
                    if l3 >= 1:
                        candidates.append([l1, l2, l3])
        random.shuffle(candidates)

        for lengths in candidates:
            path = _polyline_cells(start, dirs, lengths)
            if len(path) != total_len:
                continue
            if len(set(path)) != len(path):
                continue
            if not all(_in_bounds(grid.shape, cell) for cell in path):
                continue
            if any(int(grid[r, c]) != BG for r, c in path):
                continue
            if _adjacent_to_existing_grey(path, existing_grey):
                continue
            if not _path_has_visible_turn(path, visible_len):
                continue
            return path
    return None


def _trace_path(grid: np.ndarray, start: tuple[int, int], grey: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = set()
    current = start

    while True:
        r, c = current
        assert int(grid[r, c]) == int(grey)
        path.append(current)
        visited.add(current)
        next_cells = []
        for nr, nc in neighbors4(r, c):
            if not _in_bounds(grid.shape, (nr, nc)):
                continue
            if int(grid[nr, nc]) != int(grey):
                continue
            if (nr, nc) in visited:
                continue
            next_cells.append((nr, nc))
        assert len(next_cells) <= 1, "guide path must be an unbranched chain"
        if not next_cells:
            return path
        current = next_cells[0]


def _detect_cups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    cups: list[dict] = []

    for color in colors_present(g, exclude={bg, GREY}):
        comps = components_mask(g == int(color), connectivity=4)
        for coords in comps:
            area = int(coords.shape[0])
            if area < 8:
                continue

            r0, c0, r1, c1 = map(int, bbox_coords(coords))
            cells = {tuple(map(int, cell)) for cell in coords}
            edges = {
                "U": [cell for cell in cells if cell[0] == r0],
                "D": [cell for cell in cells if cell[0] == r1],
                "L": [cell for cell in cells if cell[1] == c0],
                "R": [cell for cell in cells if cell[1] == c1],
            }
            candidates = [(d, edge_cells[0]) for d, edge_cells in edges.items() if len(edge_cells) == 1]
            if len(candidates) != 1:
                continue

            direction, tip = candidates[0]
            tip_r, tip_c = tip
            grey_neighbors = []
            for nr, nc in neighbors4(tip_r, tip_c):
                if _in_bounds(g.shape, (nr, nc)) and int(g[nr, nc]) == GREY:
                    grey_neighbors.append((nr, nc))
            assert len(grey_neighbors) == 1, "each tail tip must touch exactly one grey entry cell"

            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            content_mask = (sub != bg) & (sub != GREY) & (sub != int(color))
            content_colors = sorted(set(map(int, sub[content_mask])))
            assert len(content_colors) == 1, "cup bbox must contain exactly one content color"
            content_color = content_colors[0]
            content_cells = np.argwhere(content_mask)
            assert 2 <= content_cells.shape[0] <= 5

            path = _trace_path(g, grey_neighbors[0], GREY)
            cups.append(
                {
                    "border_color": int(color),
                    "direction": direction,
                    "tip": tip,
                    "bbox": (r0, c0, r1, c1),
                    "content_color": content_color,
                    "content_count": int(content_cells.shape[0]),
                    "content_cells": [(r0 + int(r), c0 + int(c)) for r, c in content_cells],
                    "path": path,
                }
            )

    assert 2 <= len(cups) <= 3, "expected 2-3 cups"
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each cup's contents and repaint the front of its attached grey guide path with the same number of pixels.

    Input structure:
    - Black is the background and grey pixels form 4-connected unbranched guide paths.
    - Two or three cup-shaped border components use distinct non-grey colors; each cup is open on one side and has a single-pixel tail tip on the opposite bbox edge.
    - Each cup bbox contains 2-5 pixels of one unique content color, and exactly one grey path cell is 4-neighbor adjacent to the tail tip.

    Rule:
    - For every cup, count its content pixels and remember their shared color.
    - Erase those content pixels from the cup interior.
    - Starting from the grey cell touching the tail tip, follow the unique unvisited grey 4-neighbor chain.
    - Recolor the first N path pixels with the cup's content color, where N is the erased content count, and erase the rest of that path to black.

    Key insight:
    - The tail no longer defines a straight ray; it only selects the entrance to a specific grey track, and the content count determines how far recoloring proceeds along that curved track.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    cups = _detect_cups(g)

    seen_path_cells: set[tuple[int, int]] = set()
    for cup in cups:
        for cell in cup["path"]:
            assert cell not in seen_path_cells, "guide paths must be disjoint"
            seen_path_cells.add(cell)

        for r, c in cup["content_cells"]:
            out[r, c] = BG

        for idx, (r, c) in enumerate(cup["path"]):
            out[r, c] = int(cup["content_color"]) if idx < cup["content_count"] else BG

    return out


def _expected_output(grid: np.ndarray, cups: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for cup in cups:
        for r, c in cup["content_global"]:
            out[r, c] = BG
        for idx, (r, c) in enumerate(cup["path"]):
            out[r, c] = cup["content_color"] if idx < cup["content_count"] else BG
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    num_cups = random.choice([2, 3])
    directions = random.sample(list(DIRS), num_cups)
    colors = random.sample(PALETTE, 2 * num_cups)

    grid = np.full((height, width), BG, dtype=int)
    cups: list[dict] = []
    occupied = grid.copy()

    for idx in range(num_cups):
        border_color = colors[2 * idx]
        content_color = colors[2 * idx + 1]
        tail_len = random.choice([1, 2])
        body_h = random.randint(4, 6)
        width0 = random.randint(4, 6)
        capacity = (body_h - 1) * (width0 - 2)
        if capacity < 2:
            return None
        content_count = random.randint(2, min(5, capacity))
        cup = _build_cup_patch(
            border_color=border_color,
            content_color=content_color,
            direction=directions[idx],
            tail_len=tail_len,
            body_h=body_h,
            width=width0,
            content_count=content_count,
        )

        try:
            r0, c0 = random_free_location(
                occupied,
                cup["patch"],
                transparent=BG,
                border=2,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        tip_global = (r0 + cup["tip_local"][0], c0 + cup["tip_local"][1])
        dr, dc = DIRS[cup["direction"]]
        start = (tip_global[0] + dr, tip_global[1] + dc)
        if not _in_bounds(grid.shape, start) or int(grid[start]) != BG:
            return None

        paste(grid, cup["patch"], r0, c0, transparent=BG)
        paste(occupied, cup["patch"], r0, c0, transparent=BG)
        cup["origin"] = (r0, c0)
        cup["tip_global"] = tip_global
        cup["content_global"] = [(r0 + r, c0 + c) for r, c in cup["content_local"]]
        cup["start"] = start
        cups.append(cup)

    random.shuffle(cups)
    existing_grey: set[tuple[int, int]] = set()
    for cup in cups:
        extra = random.randint(1, 4)
        total_len = cup["content_count"] + extra
        path = _build_path(
            grid,
            cup["start"],
            cup["direction"],
            total_len,
            cup["content_count"],
            existing_grey,
        )
        if path is None:
            return None
        for r, c in path:
            grid[r, c] = GREY
            existing_grey.add((r, c))
        cup["path"] = path

    if not any(_path_has_visible_turn(cup["path"], cup["content_count"]) for cup in cups):
        return None
    if len({cup["direction"] for cup in cups}) < 2:
        return None

    expected = _expected_output(grid, cups)

    try:
        parsed = _detect_cups(grid)
    except AssertionError:
        return None
    if len(parsed) != num_cups:
        return None
    for cup in parsed:
        if len(cup["path"]) <= cup["content_count"]:
            return None
        if not _path_has_visible_turn(cup["path"], cup["content_count"]):
            return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid guide-path ejection puzzle")
