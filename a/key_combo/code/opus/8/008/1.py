from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]

BG = 0
WALL = 1
MARKER = 6
PATH_PALETTE = [2, 3, 4, 7, 8, 9]
MANDATORY_PATH_COLORS = [2, 3, 4, 7]

MAX_ATTEMPTS = 400
MAX_DECOR_ATTEMPTS = 40

BASE_TEMPLATES = [
    [
        "600011111100000",
        "000000000100000",
        "000000000100000",
        "000000000100000",
        "111111110100000",
        "100000010100000",
        "100000110100000",
        "100000010100000",
        "111111110100000",
        "000000000100000",
        "992474323000000",
        "000000009000000",
        "000000007000000",
        "000000002000000",
        "000000004249400",
    ],
    [
        "60001111110000000",
        "72340000010000000",
        "20000002010000000",
        "34347397010000000",
        "00000000010000000",
        "11111111111100000",
        "10000000010100000",
        "10000000010110000",
        "10000000010100000",
        "10000100010100000",
        "10000000010100000",
        "10000000010100000",
        "11111111111100000",
        "00000000010000000",
        "00000000010000000",
        "00000000010000000",
        "00000000000000000",
    ],
    [
        "60001111000000",
        "79270001000000",
        "20000201000000",
        "44883401000000",
        "00000001000000",
        "11111111100000",
        "00000011000000",
        "00000001000000",
        "00000001000000",
        "00000001000000",
        "00000001000000",
        "00000001000000",
        "00000000000000",
        "00000000000000",
    ],
]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            labels[r, c] = comp_id
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
    return labels, comp_id


def _ordered_path(path_mask: np.ndarray) -> list[Coord]:
    cells = {tuple(map(int, coord)) for coord in np.argwhere(path_mask)}
    assert cells, "expected a path"

    degree = {
        cell: sum(neighbor in cells for neighbor in _neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, deg in degree.items() if deg == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degree.values()), "path must not branch"

    order: list[Coord] = []
    prev = None
    cur = endpoints[0]
    while True:
        order.append(cur)
        nxt = [p for p in _neighbors4(*cur) if p in cells and p != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unique"
        prev, cur = cur, nxt[0]

    assert len(order) == len(cells), "path traversal must visit every path cell"
    return order


def _clean_walls(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    wall_labels, n_wall = _label_mask(g == WALL)
    h, w = g.shape
    structural = np.zeros_like(g, dtype=bool)
    cleaned = g.copy()

    for comp_id in range(1, n_wall + 1):
        coords = np.argwhere(wall_labels == comp_id)
        touches_border = any(
            r == 0 or c == 0 or r == h - 1 or c == w - 1
            for r, c in coords
        )
        if touches_border:
            structural[wall_labels == comp_id] = True
        else:
            cleaned[wall_labels == comp_id] = BG
    return cleaned, structural


def _reachable_background(cleaned: np.ndarray) -> np.ndarray:
    h, w = cleaned.shape
    marker_coords = np.argwhere(cleaned == MARKER)
    assert len(marker_coords) == 1, "expected exactly one marker"
    start = tuple(map(int, marker_coords[0]))

    reach = np.zeros((h, w), dtype=bool)
    q = deque([start])
    reach[start] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and cleaned[nr, nc] in (BG, MARKER):
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _path_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BG) & (grid != WALL) & (grid != MARKER)


def _reachable_flags(grid: np.ndarray) -> tuple[list[Coord], list[bool]]:
    cleaned, _structural = _clean_walls(grid)
    reach = _reachable_background(cleaned)
    path = _ordered_path(_path_mask(grid))
    h, w = grid.shape

    flags = []
    for r, c in path:
        flag = any(
            0 <= nr < h and 0 <= nc < w and cleaned[nr, nc] == BG and reach[nr, nc]
            for nr, nc in _neighbors4(r, c)
        )
        flags.append(flag)
    return path, flags


def _flag_runs(flags: list[bool]) -> list[tuple[bool, int]]:
    runs: list[tuple[bool, int]] = []
    current = flags[0]
    count = 0
    for flag in flags:
        if flag == current:
            count += 1
        else:
            runs.append((current, count))
            current = flag
            count = 1
    runs.append((current, count))
    return runs


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reverse the color order only on path pixels that border corner-reachable background.

    Input structure:
    - The background is black, blue pixels form one or more wall components, and a single magenta marker sits on a corner.
    - One multicolor 4-connected one-pixel-wide path uses non-blue, non-black colors and has exactly two endpoints.
    - Some blue wall components touch the grid border while others are interior decorative fragments.

    Rule:
    - Split blue wall pixels into 4-connected components and erase every component that does not touch a grid edge.
    - Flood-fill from the corner marker through black background cells, using only the border-touching blue components as barriers.
    - Mark each path pixel as reachable when one of its 4-neighbors is a reachable background cell.
    - Traverse the path from the lexicographically smaller endpoint to the other endpoint and reverse, in place, the color subsequence drawn on the reachable path pixels only.
    - Leave unreachable path pixels, structural walls, background, and the marker unchanged.

    Key insight:
    - Border-connected blue components are the real spatial walls; once they determine which background cells the corner marker can reach, that reachability gates exactly which path positions participate in the color reversal.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC grid"

    out, _structural = _clean_walls(g)
    path, flags = _reachable_flags(g)
    assert any(flags) and not all(flags), "path must contain both reachable and unreachable cells"

    reachable_colors = [int(g[r, c]) for (r, c), flag in zip(path, flags) if flag]
    reversed_colors = list(reversed(reachable_colors))

    idx = 0
    for (r, c), flag in zip(path, flags):
        if flag:
            out[r, c] = reversed_colors[idx]
            idx += 1
        else:
            out[r, c] = int(g[r, c])

    return out


def _parse_template(lines: list[str]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in lines], dtype=int)


def _transform_grid(grid: np.ndarray, op: int) -> np.ndarray:
    g = grid
    rot = op % 4
    if op >= 4:
        g = np.fliplr(g)
    return np.rot90(g, rot)


def _random_path_colors(path_len: int, reachable_flags: list[bool]) -> list[int]:
    while True:
        seq = list(MANDATORY_PATH_COLORS)
        while len(seq) < path_len:
            seq.append(random.choice(PATH_PALETTE))
        random.shuffle(seq)
        reachable_colors = [color for color, flag in zip(seq, reachable_flags) if flag]
        if reachable_colors != list(reversed(reachable_colors)):
            return seq


def _safe_for_decor(grid: np.ndarray, cells: list[Coord]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if grid[r, c] != BG:
            return False
        for nr, nc in _neighbors4(r, c):
            if grid[nr, nc] in (WALL, MARKER) or grid[nr, nc] not in (BG, WALL, MARKER):
                return False
    return True


def _add_decorations(grid: np.ndarray) -> None:
    h, w = grid.shape
    shapes = [
        [(0, 0)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
    ]
    target = random.choices([0, 1, 2, 3], weights=[1, 3, 3, 2], k=1)[0]
    placed = 0
    for _ in range(MAX_DECOR_ATTEMPTS):
        if placed >= target:
            return
        shape = random.choice(shapes)
        base_r = random.randint(1, h - 2)
        base_c = random.randint(1, w - 2)
        cells = [(base_r + dr, base_c + dc) for dr, dc in shape]
        if _safe_for_decor(grid, cells):
            for r, c in cells:
                grid[r, c] = WALL
            placed += 1


def _build_from_template() -> np.ndarray:
    skeleton = _parse_template(random.choice(BASE_TEMPLATES))

    pad_bottom = random.randint(0, 3)
    pad_right = random.randint(0, 3)
    if pad_bottom or pad_right:
        skeleton = np.pad(skeleton, ((0, pad_bottom), (0, pad_right)), constant_values=BG)

    skeleton = _transform_grid(skeleton, random.randint(0, 7))

    path, flags = _reachable_flags(np.where(_path_mask(skeleton), 2, skeleton))
    runs = _flag_runs(flags)
    assert len(runs) == 2 and min(length for _flag, length in runs) >= 3

    grid = skeleton.copy()
    for r, c in path:
        grid[r, c] = BG
    colors = _random_path_colors(len(path), flags)
    for (r, c), color in zip(path, colors):
        grid[r, c] = color

    _add_decorations(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_from_template()
        try:
            path, flags = _reachable_flags(grid)
            runs = _flag_runs(flags)
            assert len(runs) == 2 and min(length for _flag, length in runs) >= 3
            assert path
            assert len(set(int(grid[r, c]) for r, c in path) & set(MANDATORY_PATH_COLORS)) == len(MANDATORY_PATH_COLORS)
            out = main(grid)
            assert out.shape == grid.shape
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
