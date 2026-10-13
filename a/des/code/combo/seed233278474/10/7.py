import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PATH_COLORS = (3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_marker(grid: np.ndarray, color: int) -> tuple[int, int]:
    h, w = grid.shape
    matches: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if bool(np.all(grid[r : r + 2, c : c + 2] == color)):
                matches.append((r, c))
    assert len(matches) == 1, f"expected exactly one {color} 2x2 marker"
    return matches[0]


def _marker_mask(shape: tuple[int, int], top_left: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0 = top_left
    mask[r0 : r0 + 2, c0 : c0 + 2] = True
    return mask


def _adjacent_path_pixels(path_mask: np.ndarray, marker_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = path_mask.shape
    out: set[tuple[int, int]] = set()
    for r, c in np.argwhere(path_mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc, h, w):
            if bool(marker_mask[nr, nc]):
                out.add((rr, cc))
                break
    return sorted(out)


def _follow_branchless(
    path_mask: np.ndarray,
    start: tuple[int, int],
    ends: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = path_mask.shape
    path = [start]
    seen = {start}
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        if cur in ends:
            return path

        choices: list[tuple[int, int]] = []
        for nr, nc in _neighbors4(cur[0], cur[1], h, w):
            nxt = (nr, nc)
            if not bool(path_mask[nr, nc]):
                continue
            if nxt == prev:
                continue
            if nxt in seen:
                return None
            choices.append(nxt)

        if len(choices) != 1:
            return None

        prev = cur
        cur = choices[0]
        path.append(cur)
        seen.add(cur)


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    start_tl = _find_marker(g, RED)
    end_tl = _find_marker(g, BLUE)

    start_mask = _marker_mask(g.shape, start_tl)
    end_mask = _marker_mask(g.shape, end_tl)
    path_mask = (g != BLACK) & (~start_mask) & (~end_mask)

    starts = _adjacent_path_pixels(path_mask, start_mask)
    ends = _adjacent_path_pixels(path_mask, end_mask)
    assert starts, "start marker must touch at least one path pixel"
    assert ends, "end marker must touch at least one path pixel"

    successes: list[tuple[tuple[int, int], ...]] = []
    end_set = set(ends)
    for start in starts:
        path = _follow_branchless(path_mask, start, end_set)
        if path is not None:
            successes.append(tuple(path))

    unique = list(dict.fromkeys(successes))
    assert len(unique) == 1, "expected a unique branchless route from red to blue"
    route = list(unique[0])

    out = g.copy()
    for r, c in route:
        out[r, c] = BLUE
    return out, route


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique branchless route between a red 2x2 start marker and a blue 2x2 end marker.

    Input structure
    - Black background.
    - A multicolored single-pixel path network whose colors are irrelevant noise.
    - One red 2x2 block and one blue 2x2 block, each touching a path endpoint.

    Rule
    - Start from the path pixel adjacent to the red marker.
    - Keep moving only while there is exactly one forward path neighbor besides the pixel you came from.
    - Reject branches and dead ends; the unique successful route reaches the blue marker.
    - Repaint that route blue and keep everything else unchanged.
    """
    return _solve(grid)[0]


def _marker_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}


def _random_marker(h: int, w: int) -> dict:
    side = random.choice(("up", "down", "left", "right"))
    offset = random.randint(0, 1)

    if side == "up":
        r0 = random.randint(1, h - 2)
        c0 = random.randint(0, w - 2)
        endpoint = (r0 - 1, c0 + offset)
    elif side == "down":
        r0 = random.randint(0, h - 3)
        c0 = random.randint(0, w - 2)
        endpoint = (r0 + 2, c0 + offset)
    elif side == "left":
        r0 = random.randint(0, h - 2)
        c0 = random.randint(1, w - 2)
        endpoint = (r0 + offset, c0 - 1)
    else:
        r0 = random.randint(0, h - 2)
        c0 = random.randint(0, w - 3)
        endpoint = (r0 + offset, c0 + 2)

    top_left = (r0, c0)
    cells = _marker_cells(top_left)
    reserved: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) not in cells:
                reserved.add((nr, nc))
    reserved.discard(endpoint)
    return {
        "top_left": top_left,
        "cells": cells,
        "endpoint": endpoint,
        "reserved": reserved,
        "footprint": cells | reserved | {endpoint},
    }


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _sample_marker_pair(h: int, w: int) -> tuple[dict, dict]:
    for _ in range(300):
        start = _random_marker(h, w)
        end = _random_marker(h, w)
        if start["footprint"] & end["footprint"]:
            continue
        if _manhattan(start["endpoint"], end["endpoint"]) < max(6, min(h, w) // 2):
            continue
        return start, end
    raise AssertionError("failed to place non-overlapping markers")


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d1 != d2:
            turns += 1
    return turns


def _build_main_path(
    h: int,
    w: int,
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: set[tuple[int, int]],
    min_nodes: int,
    max_nodes: int,
) -> list[tuple[int, int]] | None:
    path = [start]
    used = {start}
    budget = {"calls": 0}

    def valid_step(cur: tuple[int, int], nxt: tuple[int, int]) -> bool:
        if nxt in blocked or nxt in used:
            return False
        for nr, nc in _neighbors4(nxt[0], nxt[1], h, w):
            if (nr, nc) in used and (nr, nc) != cur:
                return False
        return True

    def dfs(cur: tuple[int, int]) -> bool:
        budget["calls"] += 1
        if budget["calls"] > 50000:
            return False

        dist = _manhattan(cur, end)
        if len(path) + dist > max_nodes:
            return False
        if len(path) > max_nodes:
            return False
        if cur == end:
            return len(path) >= min_nodes

        candidates: list[tuple[int, int]] = []
        for nr, nc in _neighbors4(cur[0], cur[1], h, w):
            nxt = (nr, nc)
            if valid_step(cur, nxt):
                candidates.append(nxt)
        random.shuffle(candidates)
        if len(path) < min_nodes:
            candidates.sort(key=lambda cell: _manhattan(cell, end), reverse=True)
        else:
            candidates.sort(key=lambda cell: _manhattan(cell, end))

        for nxt in candidates:
            path.append(nxt)
            used.add(nxt)
            if dfs(nxt):
                return True
            used.remove(nxt)
            path.pop()
        return False

    if dfs(start):
        return path
    return None


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _rotate(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_r = max(r for r, _ in cells)
    return {(c, max_r - r) for r, c in cells}


def _random_decoy_shape() -> set[tuple[int, int]]:
    kind = random.choices(
        ["plus", "tee", "branch", "zigzag", "line"],
        weights=[3, 3, 3, 2, 1],
        k=1,
    )[0]

    cells: set[tuple[int, int]]
    if kind == "plus":
        up = random.randint(1, 3)
        down = random.randint(1, 3)
        left = random.randint(1, 3)
        right = random.randint(1, 3)
        center = (up, left)
        cells = {center}
        for i in range(1, up + 1):
            cells.add((up - i, left))
        for i in range(1, down + 1):
            cells.add((up + i, left))
        for i in range(1, left + 1):
            cells.add((up, left - i))
        for i in range(1, right + 1):
            cells.add((up, left + i))
    elif kind == "tee":
        stem = random.randint(2, 4)
        left = random.randint(1, 3)
        right = random.randint(1, 3)
        cells = {(r, 0) for r in range(stem + 1)}
        center_r = random.randint(1, stem - 1)
        for i in range(1, left + 1):
            cells.add((center_r, -i))
        for i in range(1, right + 1):
            cells.add((center_r, i))
    elif kind == "branch":
        trunk = random.randint(4, 7)
        branch_at = random.randint(1, trunk - 2)
        branch_len = random.randint(1, 3)
        cells = {(r, 0) for r in range(trunk)}
        for i in range(1, branch_len + 1):
            cells.add((branch_at, i))
    elif kind == "line":
        length = random.randint(4, 8)
        cells = {(0, c) for c in range(length)}
    else:
        row = 0
        col = 0
        cells = {(row, col)}
        horiz = random.choice([-1, 1])
        segments = random.randint(3, 5)
        for seg in range(segments):
            run = random.randint(2, 4)
            if seg % 2 == 0:
                for _ in range(run):
                    col += horiz
                    cells.add((row, col))
            else:
                step = random.choice([-1, 1])
                for _ in range(run):
                    row += step
                    cells.add((row, col))
                horiz *= -1

    cells = _normalize(cells)
    for _ in range(random.randint(0, 3)):
        cells = _normalize(_rotate(cells))
    return cells


def _can_place_component(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied or (r, c) in blocked:
            return False
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in occupied and (nr, nc) not in cells:
                return False
    return True


def _place_decoys(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    scene_colors: list[int],
) -> None:
    h, w = grid.shape
    target = random.randint(1, 4)
    placed = 0
    for _ in range(120):
        if placed >= target:
            break
        shape = _random_decoy_shape()
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1
        if shape_h >= h or shape_w >= w:
            continue
        top = random.randint(0, h - shape_h)
        left = random.randint(0, w - shape_w)
        cells = {(top + r, left + c) for r, c in shape}
        if not _can_place_component(cells, occupied, blocked, h, w):
            continue
        for r, c in cells:
            grid[r, c] = random.choice(scene_colors)
        occupied.update(cells)
        placed += 1


def _paint_marker(grid: np.ndarray, marker: dict, color: int) -> None:
    r0, c0 = marker["top_left"]
    grid[r0 : r0 + 2, c0 : c0 + 2] = color


def _gen_attempt() -> np.ndarray:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    grid = np.full((h, w), BLACK, dtype=int)

    start_marker, end_marker = _sample_marker_pair(h, w)
    blocked = (
        start_marker["cells"]
        | end_marker["cells"]
        | start_marker["reserved"]
        | end_marker["reserved"]
    )

    start = start_marker["endpoint"]
    end = end_marker["endpoint"]
    min_nodes = _manhattan(start, end) + random.randint(2, max(5, (h + w) // 4))
    max_nodes = min(h * w // 2, min_nodes + random.randint(4, 10))
    path = _build_main_path(h, w, start, end, blocked, min_nodes, max_nodes)
    assert path is not None, "failed to build a valid main route"
    assert _count_turns(path) >= random.randint(1, 4), "route is too straight"

    scene_colors = random.sample(PATH_COLORS, random.randint(3, 5))
    for r, c in path:
        grid[r, c] = random.choice(scene_colors)

    _paint_marker(grid, start_marker, RED)
    _paint_marker(grid, end_marker, BLUE)

    occupied = set(path) | start_marker["cells"] | end_marker["cells"]
    _place_decoys(grid, occupied, blocked, scene_colors)

    expected = grid.copy()
    for r, c in path:
        expected[r, c] = BLUE

    solved, solved_path = _solve(grid)
    assert solved_path == path, "solver must recover the constructed route exactly"
    assert np.array_equal(solved, expected), "solver output must match the intended output"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle input")
