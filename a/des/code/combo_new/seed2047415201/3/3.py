from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _touches_edge(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _palette_from_edge(g: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = g.shape
    palette_items: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    palette_mask = np.zeros((h, w), dtype=bool)

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BLACK, GREY)):
        for comp in _components(g == color):
            if not _touches_edge(comp, h, w):
                continue
            min_r = min(r for r, _ in comp)
            min_c = min(c for _, c in comp)
            palette_items.append((int(min_c), int(min_r), int(color), comp))

    palette_items.sort(key=lambda x: (x[0], x[1]))
    colors: list[int] = []
    for _, _, color, comp in palette_items:
        colors.append(int(color))
        for r, c in comp:
            palette_mask[r, c] = True
    return colors, palette_mask


def _fill_dead_ends(out: np.ndarray, comp: list[tuple[int, int]], palette: list[int]) -> None:
    comp_set = set(comp)
    degree: dict[tuple[int, int], int] = {}
    for r, c in comp:
        degree[(r, c)] = sum((nr, nc) in comp_set for nr, nc in _neighbors4(r, c))

    painted: set[tuple[int, int]] = set()
    endpoints = [p for p, d in degree.items() if d == 1]
    for end in endpoints:
        if end in painted:
            continue
        path = [end]
        prev: tuple[int, int] | None = None
        cur = end
        while True:
            nexts = [p for p in _neighbors4(*cur) if p in comp_set and p != prev]
            if len(nexts) != 1:
                break
            nxt = nexts[0]
            nxt_degree = degree[nxt]
            if nxt_degree > 2:
                depth = len(path)
                if 1 <= depth <= len(palette):
                    color = int(palette[depth - 1])
                    for r, c in path:
                        out[r, c] = color
                        painted.add((r, c))
                break
            if nxt_degree != 2:
                break
            path.append(nxt)
            prev, cur = cur, nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill maze dead-end corridors according to their depth using the edge palette.

    Input structure:
    - Black passage pixels sit inside a maze of one-pixel-wide Grey wall pixels.
    - The maze contains straight and turning corridors; some corridor branches terminate as dead ends.
    - A small horizontal legend of non-Black, non-Grey color squares lies on a grid edge and is read left-to-right as depth 1, depth 2, depth 3, and so on.

    Rule:
    - Read the edge legend from left to right to obtain the depth color palette.
    - In each enclosed Black passage component, find pixels with exactly one 4-adjacent passage neighbor.
    - Trace from each such endpoint through degree-two corridor pixels until reaching a branch pixel with more than two passage neighbors; the number of traced Black pixels is the dead-end depth.
    - Recolor all traced dead-end pixels with the palette color for that depth, leaving branch pixels, non-dead corridors, walls, and the legend unchanged.

    Key insight: Dead ends are topological leaves of the passage graph, and the ordered edge legend converts the counted leaf-to-branch depth into a visible fill color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    palette, _palette_mask = _palette_from_edge(g)
    assert palette, "expected a visible edge palette"

    black_components = _components(g == BLACK)
    maze_components = [comp for comp in black_components if not _touches_edge(comp, h, w)]
    if not maze_components:
        maze_components = black_components

    out = g.copy()
    for comp in maze_components:
        _fill_dead_ends(out, comp, palette)
    return out


def _carve_loop(grid: np.ndarray, top: int, left: int, bottom: int, right: int) -> set[tuple[int, int]]:
    black: set[tuple[int, int]] = set()
    for c in range(left, right + 1):
        grid[top, c] = BLACK
        grid[bottom, c] = BLACK
        black.add((top, c))
        black.add((bottom, c))
    for r in range(top, bottom + 1):
        grid[r, left] = BLACK
        grid[r, right] = BLACK
        black.add((r, left))
        black.add((r, right))
    return black


def _candidate_root(side: str, top: int, left: int, bottom: int, right: int) -> tuple[int, int, tuple[int, int]]:
    if side == "top":
        return top, random.randint(left + 1, right - 1), (-1, 0)
    if side == "bottom":
        return bottom, random.randint(left + 1, right - 1), (1, 0)
    if side == "left":
        return random.randint(top + 1, bottom - 1), left, (0, -1)
    return random.randint(top + 1, bottom - 1), right, (0, 1)


def _try_add_branch(
    grid: np.ndarray,
    black: set[tuple[int, int]],
    maze_box: tuple[int, int, int, int],
    loop_box: tuple[int, int, int, int],
    depth: int,
) -> list[tuple[int, int]] | None:
    maze_r0, maze_c0, maze_r1, maze_c1 = maze_box
    top, left, bottom, right = loop_box
    sides = ["top", "bottom", "left", "right"]
    for _ in range(120):
        side = random.choice(sides)
        rr, cc, (dr, dc) = _candidate_root(side, top, left, bottom, right)
        path = [(rr + dr * i, cc + dc * i) for i in range(1, depth + 1)]
        ok = True
        for i, (r, c) in enumerate(path):
            if not (maze_r0 < r < maze_r1 - 1 and maze_c0 < c < maze_c1 - 1):
                ok = False
                break
            if (r, c) in black:
                ok = False
                break
            allowed = {(rr, cc)} if i == 0 else {path[i - 1]}
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in black and (nr, nc) not in allowed:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for r, c in path:
            grid[r, c] = BLACK
            black.add((r, c))
        return path
    return None


def _place_palette(grid: np.ndarray, colors: list[int]) -> None:
    h, w = grid.shape
    block = random.choice([1, 2])
    side = random.choice(["top", "bottom"])
    width = block * len(colors)
    start = random.randint(0, w - width)
    r0 = 0 if side == "top" else h - block
    for i, color in enumerate(colors):
        c0 = start + i * block
        grid[r0 : r0 + block, c0 : c0 + block] = int(color)


def _gen_attempt() -> np.ndarray | None:
    n_depths = random.randint(3, 5)
    max_depth = n_depths
    legend_block = 2
    min_maze_dim = 2 * (max_depth + 1) + 6
    min_grid_dim = min_maze_dim + legend_block + 3
    if min_grid_dim > 30:
        return None

    h = random.randint(min_grid_dim, 30)
    w = random.randint(min_grid_dim, 30)
    grid = np.zeros((h, w), dtype=int)

    top_reserve = random.randint(3, 4)
    bottom_reserve = random.randint(2, 4)
    left_reserve = random.randint(1, 3)
    right_reserve = random.randint(1, 3)
    if h - top_reserve - bottom_reserve < min_maze_dim or w - left_reserve - right_reserve < min_maze_dim:
        return None

    maze_r0 = top_reserve
    maze_r1 = h - bottom_reserve
    maze_c0 = left_reserve
    maze_c1 = w - right_reserve
    grid[maze_r0:maze_r1, maze_c0:maze_c1] = GREY

    maze_h = maze_r1 - maze_r0
    maze_w = maze_c1 - maze_c0
    available_h = maze_h - 2 * (max_depth + 1)
    available_w = maze_w - 2 * (max_depth + 1)
    if available_h < 5 or available_w < 5:
        return None

    loop_h = random.randint(5, available_h)
    loop_w = random.randint(5, available_w)
    top = maze_r0 + max_depth + 1 + random.randint(0, available_h - loop_h)
    left = maze_c0 + max_depth + 1 + random.randint(0, available_w - loop_w)
    bottom = top + loop_h - 1
    right = left + loop_w - 1
    black = _carve_loop(grid, top, left, bottom, right)

    depths = list(range(1, n_depths + 1))
    depths.extend(random.randint(1, n_depths) for _ in range(random.randint(1, 4)))
    random.shuffle(depths)
    branches: list[tuple[int, list[tuple[int, int]]]] = []
    maze_box = (maze_r0, maze_c0, maze_r1, maze_c1)
    loop_box = (top, left, bottom, right)
    for depth in depths:
        path = _try_add_branch(grid, black, maze_box, loop_box, int(depth))
        if path is None:
            return None
        branches.append((int(depth), path))

    palette = random.sample(COLORS, n_depths)
    _place_palette(grid, palette)

    expected = grid.copy()
    for depth, path in branches:
        fill = int(palette[depth - 1])
        for r, c in path:
            expected[r, c] = fill

    try:
        pred = main(grid)
    except Exception:
        return None
    if not np.array_equal(pred, expected):
        return None
    for color in palette:
        if not bool(((pred == int(color)) & (grid == BLACK)).any()):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid maze dead-end input")
