import random
from collections import deque

import numpy as np

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(grid: np.ndarray, mask: np.ndarray | None = None, connectivity: int = 4) -> list[np.ndarray]:
    h, w = grid.shape
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    vis = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp: list[tuple[int, int]] = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in nbrs:
                    if _in_bounds((h, w), nr + cr, nc + cc) and mask[nr + cr, nc + cc] and not vis[nr + cr, nc + cc]:
                        vis[nr + cr, nc + cc] = True
                        q.append((nr + cr, nc + cc))
                        comp.append((nr + cr, nc + cc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _find_diagonal_segments(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    bg = GREY
    visited = np.zeros((h, w), dtype=bool)
    out: list[tuple[int, list[tuple[int, int]]]] = []
    dirs = [(1, 1), (1, -1)]
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            for dr, dc in dirs:
                pr, pc = r - dr, c - dc
                if _in_bounds((h, w), pr, pc) and int(grid[pr, pc]) == color:
                    continue
                cur = []
                rr, cc = r, c
                while _in_bounds((h, w), rr, cc) and int(grid[rr, cc]) == color:
                    visited[rr, cc] = True
                    cur.append((rr, cc))
                    rr += dr
                    cc += dc
                if len(cur) >= 3:
                    out.append((color, cur))
    # dedupe by coordinate sets
    uniq = []
    seen = set()
    for color, seg in out:
        fs = frozenset(seg)
        if fs in seen:
            continue
        seen.add(fs)
        uniq.append((color, seg))
    return uniq


def _find_sprites(grid: np.ndarray) -> dict[int, tuple[list[tuple[int, int]], int, tuple[int, int], int, int]]:
    h, w = grid.shape
    bg = GREY
    comps = _components(grid, grid != bg)
    sprites: dict[int, tuple[list[tuple[int, int]], int, tuple[int, int], int, int]] = {}
    for comp in comps:
        if comp.shape[0] != 9:
            continue
        r0 = int(comp[:, 0].min())
        r1 = int(comp[:, 0].max())
        c0 = int(comp[:, 1].min())
        c1 = int(comp[:, 1].max())
        if (r1 - r0 + 1) != 3 or (c1 - c0 + 1) != 3:
            continue
        border_cells = set((int(rr), int(cc)) for rr, cc in comp if (rr, cc) not in {(r0 + 1, c0 + 1)})
        border_colors = {int(grid[r, c]) for r, c in border_cells}
        if len(border_colors) != 1:
            continue
        border_col = int(border_colors.pop())
        center_col = int(grid[r0 + 1, c0 + 1])
        if center_col == border_col:
            continue
        sprites[border_col] = (
            [(int(r), int(c)) for r, c in comp],
            border_col,
            (r0 + 1, c0 + 1),
            3,  # size
            center_col,
        )
    return sprites


def _find_2x2_squares(grid: np.ndarray) -> dict[int, list[tuple[int, int, int]]]:
    h, w = grid.shape
    by_color: dict[int, list[tuple[int, int, int]]] = {}
    for r in range(h - 1):
        for c in range(w - 1):
            vals = {grid[r + i, c + j] for i in (0, 1) for j in (0, 1)}
            if len(vals) == 1:
                col = int(next(iter(vals)))
                if col != BLACK and col != GREY:
                    by_color.setdefault(col, []).append((r, c, col))
    return by_color


def _direction_from_sprite_to_square(
    sprite_center: tuple[int, int], square_tl: tuple[int, int], direction: int
) -> tuple[int, int]:
    r, c = sprite_center
    sq_r, sq_c = square_tl
    sq_center_r = sq_r + 0.5
    sq_center_c = sq_c + 0.5
    dr = sq_center_r - r
    dc = sq_center_c - c
    if abs(dr) > abs(dc):
        return (-1, 0) if dr < 0 else (1, 0)
    if abs(dc) > abs(dr):
        return (0, -1) if dc < 0 else (0, 1)
    return (0, 1) if direction == 0 else (1, 0)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int, copy=True)
    bg = GREY
    segs = _find_diagonal_segments(grid)
    if not segs:
        return grid

    segs_sorted = sorted(segs, key=lambda x: len(x[1]), reverse=True)
    n = len(segs_sorted)

    sprites = _find_sprites(grid)
    squares_by_color = _find_2x2_squares(grid)

    # build mapping from segment border color -> sprite + movement direction + steps
    mapping: dict[int, tuple[list[tuple[int, int]], int, tuple[int, int], int]] = {}
    for rank, (color, _coords) in enumerate(segs_sorted, start=1):
        if color not in sprites:
            continue
        sprite_cells, border_col, sc, _size, center_col = sprites[color]
        dirs = [sq for sq in squares_by_color.get(center_col, [])]
        if not dirs:
            continue
        sq_r, sq_c, _ = dirs[0]
        # choose a square not necessarily the nearest; it is guaranteed to be one-direction from sprite
        dr, dc = _direction_from_sprite_to_square(sc, (sq_r, sq_c), 0)
        steps = n + 1 - rank
        mapping[color] = (sprite_cells, dr, dc, steps)

    out = grid.copy()
    for _color, (sprite_cells, dr, dc, steps) in mapping.items():
        r0 = min(r for r, _ in sprite_cells)
        c0 = min(c for _, c in sprite_cells)
        # normalized sprite offsets
        offsets = [(r - r0, c - c0) for r, c in sprite_cells]
        for step in range(1, steps + 1):
            rr_off = dr * 3 * step
            cc_off = dc * 3 * step
            for rr, cc in offsets:
                r = r0 + rr + rr_off
                c = c0 + cc + cc_off
                if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                    out[r, c] = color

    return out


def generate_input() -> np.ndarray:
    h = 15
    w = 15
    colors = [BLUE, GREEN, ORANGE, MAROON]
    seg_colors = random.sample(colors, random.randint(3, 4))
    lengths = random.sample(range(3, 9), len(seg_colors))

    while True:
        g = np.full((h, w), GREY, dtype=int)
        segments: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        for color, length in zip(seg_colors, lengths):
            orientation = random.choice([(1, 1), (1, -1)])
            if orientation == (1, 1):
                r0 = random.randint(1, h - length - 1)
                c0 = random.randint(1, w - length - 1)
            else:
                r0 = random.randint(1, h - length - 1)
                c0 = random.randint(length - 1, w - 2)
            coords = [(r0 + i * orientation[0], c0 + i * orientation[1]) for i in range(length)]
            for rr, cc in coords:
                g[rr, cc] = color
            segments.append(((r0, c0), orientation, color))

        # sprites near border
        occupied = set(zip(*np.where(g != GREY)))
        for idx, (_, _, color) in enumerate(segments):
            center_col = random.choice([3, 6, 9, 7])
            while (center_col in seg_colors) and len(seg_colors) < 4:
                center_col = random.choice([3, 6, 9, 7])
            sprite_color = center_col
            border_color = color

            # 3x3 near edge
            for _attempt in range(200):
                if random.random() < 0.5:
                    sr = 0
                else:
                    sr = random.randint(0, 1)
                if random.random() < 0.5:
                    sc = 0
                else:
                    sc = random.randint(0, 1)
                if random.random() < 0.5:
                    sr = random.choice([0, h - 3])
                else:
                    sr = random.choice([0, h - 3])
                if random.random() < 0.5:
                    sc = random.choice([0, w - 3])
                else:
                    sc = random.choice([0, w - 3])
                cells = [(sr + i, sc + j) for i in (0, 1, 2) for j in (0, 1, 2)]
                if any((r, c) in occupied for r, c in cells):
                    continue
                if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                    continue
                # border
                for rr, cc in cells:
                    if rr == sr or rr == sr + 2 or cc == sc or cc == sc + 2:
                        g[rr, cc] = border_color
                    else:
                        g[rr, cc] = sprite_color
                occupied.update(cells)

                # 2x2 square in a cardinal direction
                dir_choices = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                dr, dc = random.choice(dir_choices)
                dist = random.randint(2, 4)
                sq_r = sr + 1 + dr * dist
                sq_c = sc + 1 + dc * dist
                if 0 <= sq_r <= h - 2 and 0 <= sq_c <= w - 2:
                    g[sq_r : sq_r + 2, sq_c : sq_c + 2] = sprite_color
                    occupied.update([(sq_r, sq_c), (sq_r + 1, sq_c), (sq_r, sq_c + 1), (sq_r + 1, sq_c + 1)])
                break
        return g


_generate_input_single = generate_input


def generate_inputs() -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    attempts = 0
    while len(outputs) < 20 and attempts < 200:
        attempts += 1
        g = _generate_input_single()
        arr = np.array(g, dtype=int, copy=True)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2 or arr.shape[0] > 30 or arr.shape[1] > 30:
            continue
        outputs.append(arr)

    while len(outputs) < 20:
        if outputs:
            outputs.append(outputs[-1].copy())
        else:
            outputs.append(np.zeros((1, 1), dtype=int))
    return outputs


    random.seed(0)
    print(main(generate_input()).shape)
