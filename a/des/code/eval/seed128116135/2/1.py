import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
COLORS = tuple(range(10))
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    return counts.most_common(1)[0][0]


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
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _corridor_component(
    grid: np.ndarray,
    bg: int,
    bounds: tuple[int, int, int, int],
    start: tuple[int, int],
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int] | None]]:
    r0, c0, r1, c1 = bounds
    mask = np.zeros_like(grid, dtype=bool)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    mask[start] = True
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if grid[nr, nc] != bg or mask[nr, nc]:
                continue
            mask[nr, nc] = True
            parent[(nr, nc)] = (r, c)
            q.append((nr, nc))
    return mask, parent


def _dead_ends(corridor: np.ndarray, entry: tuple[int, int]) -> list[tuple[int, int]]:
    leaves: list[tuple[int, int]] = []
    h, w = corridor.shape
    for r, c in map(tuple, np.argwhere(corridor)):
        degree = 0
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and corridor[nr, nc]:
                degree += 1
        if degree == 1 and (r, c) != entry:
            leaves.append((r, c))
    leaves.sort()
    return leaves


def _path_to_root(
    node: tuple[int, int], parent: dict[tuple[int, int], tuple[int, int] | None]
) -> list[tuple[int, int]]:
    path = [node]
    cur = node
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull matching outside blocks into dead-end cells of a colored maze.

    The maze is the central rectangular wall structure. The unique perimeter gap is
    the entrance. Each dead-end is paired with the color of the last wall touched
    along the entrance-to-dead-end corridor path. If an outside monochrome block has
    that color, the block is removed and the dead-end cell is filled with it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)

    non_bg = g != bg
    comps = _components(non_bg)
    if not comps:
        return g.copy()

    maze_seed = max(comps, key=lambda comp: ((lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1))(_bbox(comp)), len(comp)))
    r0, c0, r1, c1 = _bbox(maze_seed)
    maze_mask = np.zeros_like(g, dtype=bool)
    maze_mask[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1] != bg

    entry_candidates = [(r, c) for r, c in _perimeter_cells(r0, c0, r1, c1) if g[r, c] == bg]
    assert len(entry_candidates) == 1
    entry = entry_candidates[0]

    corridor, parent = _corridor_component(g, bg, (r0, c0, r1, c1), entry)
    leaves = _dead_ends(corridor, entry)

    outside_mask = non_bg.copy()
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    block_by_color: dict[int, list[tuple[int, int]]] = {}
    for comp in _components(outside_mask):
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        block_by_color[color] = comp

    out = g.copy()
    used_colors: set[int] = set()
    for leaf in leaves:
        last_color: int | None = None
        for r, c in _path_to_root(leaf, parent):
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if r0 <= nr <= r1 and c0 <= nc <= c1 and maze_mask[nr, nc]:
                    last_color = int(g[nr, nc])
        if last_color is None or last_color not in block_by_color or last_color in used_colors:
            continue
        for r, c in block_by_color[last_color]:
            out[r, c] = bg
        out[leaf] = last_color
        used_colors.add(last_color)

    return out


def _generate_tree_maze(
    rng: random.Random, rows: int, cols: int
) -> tuple[np.ndarray, tuple[int, int]]:
    visited = [[False] * cols for _ in range(rows)]
    stack = [(rng.randrange(rows), rng.randrange(cols))]
    visited[stack[0][0]][stack[0][1]] = True
    passages: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            stack.pop()
            continue
        nr, nc = rng.choice(neighbors)
        visited[nr][nc] = True
        a = (r, c)
        b = (nr, nc)
        if a > b:
            a, b = b, a
        passages.add((a, b))
        stack.append((nr, nc))

    h, w = 2 * rows + 1, 2 * cols + 1
    corridor = np.zeros((h, w), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            corridor[2 * r + 1, 2 * c + 1] = True
    for (r0, c0), (r1, c1) in passages:
        corridor[r0 + r1 + 1, c0 + c1 + 1] = True

    side = rng.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"}:
        cell_c = rng.randrange(cols)
        c = 2 * cell_c + 1
        if side == "top":
            corridor[0, c] = True
            entry = (0, c)
        else:
            corridor[h - 1, c] = True
            entry = (h - 1, c)
    else:
        cell_r = rng.randrange(rows)
        r = 2 * cell_r + 1
        if side == "left":
            corridor[r, 0] = True
            entry = (r, 0)
        else:
            corridor[r, w - 1] = True
            entry = (r, w - 1)
    return corridor, entry


def _wall_colors_for_maze(
    rng: random.Random,
    corridor: np.ndarray,
    bg: int,
    base_colors: list[int],
) -> np.ndarray:
    h, w = corridor.shape
    local = np.full((h, w), bg, dtype=int)
    wall_mask = ~corridor
    for r in range(h):
        run_start = None
        for c in range(w + 1):
            is_wall = c < w and wall_mask[r, c]
            if is_wall and run_start is None:
                run_start = c
            if run_start is not None and not is_wall:
                color = rng.choice(base_colors)
                local[r, run_start:c] = color
                run_start = None
    for c in range(w):
        run_start = None
        for r in range(h + 1):
            is_wall = r < h and wall_mask[r, c]
            if is_wall and run_start is None:
                run_start = r
            if run_start is not None and not is_wall:
                if (c % 2 == 0) or rng.random() < 0.35:
                    color = rng.choice(base_colors)
                    local[run_start:r, c] = color
                run_start = None
    local[corridor] = bg
    return local


def _place_blocks(
    rng: random.Random,
    grid: np.ndarray,
    bg: int,
    colors: list[int],
    maze_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = maze_bbox
    occupied = grid != bg

    for color in colors:
        placed = False
        for _ in range(120):
            bh = rng.choice([1, 1, 2, 2])
            bw = rng.choice([1, 1, 2, 2])
            rr = rng.randrange(h - bh + 1)
            cc = rng.randrange(w - bw + 1)
            if not (rr + bh - 1 < r0 - 1 or rr > r1 + 1 or cc + bw - 1 < c0 - 1 or cc > c1 + 1):
                continue
            rr0 = max(0, rr - 1)
            rr1 = min(h, rr + bh + 1)
            cc0 = max(0, cc - 1)
            cc1 = min(w, cc + bw + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            grid[rr : rr + bh, cc : cc + bw] = color
            occupied[rr : rr + bh, cc : cc + bw] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([0, 8])
    non_bg = [color for color in COLORS if color != bg]

    rows = rng.randint(3, 5)
    cols = rng.randint(3, 5)
    corridor, entry_local = _generate_tree_maze(rng, rows, cols)
    local_leaves = _dead_ends(corridor, entry_local)
    if not (2 <= len(local_leaves) <= 5):
        return None

    needed_colors = len(local_leaves) + 2
    if needed_colors > len(non_bg):
        return None
    chosen = rng.sample(non_bg, needed_colors)
    leaf_colors = chosen[: len(local_leaves)]
    base_colors = chosen[len(local_leaves) :]

    local = _wall_colors_for_maze(rng, corridor, bg, base_colors + leaf_colors)
    for leaf, color in zip(local_leaves, leaf_colors):
        lr, lc = leaf
        for dr, dc in DIR4:
            nr, nc = lr + dr, lc + dc
            if 0 <= nr < local.shape[0] and 0 <= nc < local.shape[1] and not corridor[nr, nc]:
                local[nr, nc] = color

    pad_top = rng.randint(3, 6)
    pad_bottom = rng.randint(3, 6)
    pad_left = rng.randint(3, 6)
    pad_right = rng.randint(3, 6)
    h = local.shape[0] + pad_top + pad_bottom
    w = local.shape[1] + pad_left + pad_right
    grid = np.full((h, w), bg, dtype=int)
    grid[pad_top : pad_top + local.shape[0], pad_left : pad_left + local.shape[1]] = local
    maze_bbox = (pad_top, pad_left, pad_top + local.shape[0] - 1, pad_left + local.shape[1] - 1)

    want_missing = rng.random() < 0.5
    if want_missing:
        matched_count = rng.randint(1, len(leaf_colors) - 1)
    else:
        matched_count = len(leaf_colors)
    block_colors = rng.sample(leaf_colors, matched_count)
    if not _place_blocks(rng, grid, bg, block_colors, maze_bbox):
        return None

    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid != bg):
        return None

    transformed = main(grid)
    filled = sum(int(transformed[r + pad_top, c + pad_left] != bg) for r, c in local_leaves)
    if filled != matched_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid maze instance")
