from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


SEED_COLOR = 4
MIRROR_COLOR = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG_CHOICES = [0, 1, 2, 3, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _flood(seed_cells: list[tuple[int, int]], passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    reached = np.zeros((h, w), dtype=bool)
    q = deque()
    for r, c in seed_cells:
        if not passable[r, c] or reached[r, c]:
            continue
        reached[r, c] = True
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                q.append((nr, nc))
    return reached


def _trace_path(
    grid: np.ndarray,
    seed_mask: np.ndarray,
    trav_mask: np.ndarray,
    start: tuple[int, int],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = set()
    prev: tuple[int, int] | None = None
    cur = start
    h, w = grid.shape

    while True:
        if cur in visited:
            break
        visited.add(cur)
        path.append(cur)

        nxts: list[tuple[int, int]] = []
        for dr, dc in DIR4:
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < h and 0 <= nc < w and trav_mask[nr, nc] and (nr, nc) != prev:
                nxts.append((nr, nc))
        if not nxts:
            break

        if len(nxts) == 1:
            nxt = nxts[0]
        else:
            if prev is None:
                away = [p for p in nxts if not any(seed_mask[p[0] + dr, p[1] + dc]
                                                  for dr, dc in DIR4
                                                  if 0 <= p[0] + dr < h and 0 <= p[1] + dc < w)]
                nxt = away[0] if away else nxts[0]
            else:
                dr0 = cur[0] - prev[0]
                dc0 = cur[1] - prev[1]
                straight = [p for p in nxts if (p[0] - cur[0], p[1] - cur[1]) == (dr0, dc0)]
                unused = [p for p in nxts if p not in visited]
                if straight:
                    nxt = straight[0]
                elif unused:
                    nxt = unused[0]
                else:
                    nxt = nxts[0]
        if nxt in visited:
            break
        prev, cur = cur, nxt

    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the activated corridor cells by tracing the seeded path inside each walled cavity.

    Yellow marks the seed region only. The actual fill color is inferred from the
    surrounding wall color of that cavity. Grey mirror cells are preserved.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = g.copy()

    seed_comps = _components(g == SEED_COLOR)
    passable = (g == bg) | (g == SEED_COLOR) | (g == MIRROR_COLOR)

    for comp in seed_comps:
        seed_mask = np.zeros((h, w), dtype=bool)
        for r, c in comp:
            seed_mask[r, c] = True

        cavity = _flood(comp, passable)

        wall_colors = Counter()
        for r, c in np.argwhere(cavity):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not cavity[nr, nc]:
                    v = int(g[nr, nc])
                    if v not in (bg, SEED_COLOR, MIRROR_COLOR):
                        wall_colors[v] += 1
        if not wall_colors:
            continue
        wall_color = int(wall_colors.most_common(1)[0][0])

        trav_mask = cavity & ((g == bg) | (g == MIRROR_COLOR))
        start_candidates: list[tuple[int, int, int, int]] = []
        for r, c in np.argwhere(trav_mask):
            rr = int(r)
            cc = int(c)
            seed_adj = 0
            deg = 0
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if seed_mask[nr, nc]:
                    seed_adj += 1
                if trav_mask[nr, nc]:
                    deg += 1
            if seed_adj:
                start_candidates.append((deg, -seed_adj, rr, cc))
        if not start_candidates:
            continue
        start_candidates.sort()
        start = (start_candidates[0][2], start_candidates[0][3])

        for r, c in _trace_path(g, seed_mask, trav_mask, start):
            if int(g[r, c]) == bg:
                out[r, c] = wall_color

    out[g == SEED_COLOR] = bg
    return out


def _turn(direction: tuple[int, int], sign: int) -> tuple[int, int]:
    dr, dc = direction
    if sign == -1:
        return (-dc, dr)
    return (dc, -dr)


def _path_open_neighbors(cells: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in cells for dr, dc in DIR4)


def _path_is_simple(cells: set[tuple[int, int]]) -> bool:
    degrees = [_path_open_neighbors(cells, cell) for cell in cells]
    if any(deg not in (1, 2) for deg in degrees):
        return False
    return sum(deg == 1 for deg in degrees) == 2


def _build_polyline(
    start_dir: tuple[int, int],
    lengths: list[int],
    turns: list[int],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    path = [(0, 0)]
    path_set = {(0, 0)}
    mirrors: set[tuple[int, int]] = set()
    cur = (0, 0)
    direction = start_dir

    for idx, seg_len in enumerate(lengths):
        for _ in range(seg_len):
            cur = (cur[0] + direction[0], cur[1] + direction[1])
            if cur in path_set:
                return None
            path.append(cur)
            path_set.add(cur)
        if idx < len(turns):
            mirrors.add(cur)
            direction = _turn(direction, turns[idx])

    for i in range(len(path)):
        for j in range(i + 1, len(path)):
            dist = max(abs(path[i][0] - path[j][0]), abs(path[i][1] - path[j][1]))
            if j == i + 1:
                continue
            if j > i + 2 and dist <= 1:
                return None
    return path, mirrors


SEED_SHAPES = [
    [(1, 0), (2, 0), (3, 0)],
    [(1, -1), (1, 0), (1, 1)],
    [(1, 0), (2, 0), (1, 1), (2, 1)],
    [(1, 0), (2, 0), (1, -1), (2, -1)],
]


def _seed_points(
    start: tuple[int, int],
    direction: tuple[int, int],
    coeffs: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    dr, dc = direction
    back = (-dr, -dc)
    side = (-dc, dr)
    pts = set()
    for a, b in coeffs:
        rr = start[0] + a * back[0] + b * side[0]
        cc = start[1] + a * back[1] + b * side[1]
        pts.add((rr, cc))
    return pts


def _make_module_grid(
    open_cells: set[tuple[int, int]],
    wall_color: int,
    bg: int,
    seed_cells: set[tuple[int, int]] | None = None,
    mirror_cells: set[tuple[int, int]] | None = None,
) -> dict | None:
    seed_cells = seed_cells or set()
    mirror_cells = mirror_cells or set()

    if open_cells & seed_cells:
        return None

    region = open_cells | seed_cells
    walls: set[tuple[int, int]] = set()
    for r, c in region:
        for dr, dc in DIR4:
            nbr = (r + dr, c + dc)
            if nbr not in region:
                walls.add(nbr)

    all_cells = region | walls
    min_r = min(r for r, _ in all_cells)
    max_r = max(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_c = max(c for _, c in all_cells)
    shift_r = -min_r
    shift_c = -min_c

    h = max_r - min_r + 1
    w = max_c - min_c + 1
    grid = np.full((h, w), bg, dtype=int)

    shifted_walls = {(r + shift_r, c + shift_c) for r, c in walls}
    shifted_open = {(r + shift_r, c + shift_c) for r, c in open_cells}
    shifted_seed = {(r + shift_r, c + shift_c) for r, c in seed_cells}
    shifted_mirrors = {(r + shift_r, c + shift_c) for r, c in mirror_cells}

    for r, c in shifted_walls:
        grid[r, c] = wall_color
    for r, c in shifted_seed:
        grid[r, c] = SEED_COLOR
    for r, c in shifted_mirrors:
        grid[r, c] = MIRROR_COLOR

    return {
        "grid": grid,
        "open_cells": shifted_open,
        "seed_cells": shifted_seed,
        "mirror_cells": shifted_mirrors,
        "wall_color": wall_color,
    }


def _build_seeded_module(rng: random.Random, wall_color: int, bg: int) -> dict | None:
    start_dir = rng.choice(DIR4)
    bends = rng.choice([0, 1, 2])
    turns = [rng.choice([-1, 1]) for _ in range(bends)]

    if bends == 0:
        lengths = [rng.randint(4, 8)]
    elif bends == 1:
        lengths = [rng.randint(3, 7), rng.randint(3, 7)]
    else:
        lengths = [rng.randint(3, 6), rng.randint(4, 7), rng.randint(3, 6)]

    polyline = _build_polyline(start_dir, lengths, turns)
    if polyline is None:
        return None
    path, mirror_cells = polyline
    open_cells = set(path)
    if not _path_is_simple(open_cells):
        return None

    seed_cells = _seed_points(path[0], start_dir, rng.choice(SEED_SHAPES))
    if len([cell for cell in open_cells if _path_open_neighbors(open_cells, cell) == 1]) != 2:
        return None
    if sum(any(abs(sr - r) + abs(sc - c) == 1 for sr, sc in seed_cells) for r, c in open_cells) != 1:
        return None

    module = _make_module_grid(open_cells, wall_color, bg, seed_cells, mirror_cells)
    if module is None:
        return None

    fill_cells = sorted(module["open_cells"] - module["mirror_cells"])
    if not fill_cells:
        return None
    module["seeded"] = True
    module["fill_cells"] = fill_cells
    return module


def _rotate_points(points: set[tuple[int, int]], orientation: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    for r, c in points:
        if orientation == (1, 0):
            rr, cc = r, c
        elif orientation == (-1, 0):
            rr, cc = -r, -c
        elif orientation == (0, 1):
            rr, cc = c, -r
        else:
            rr, cc = -c, r
        out.add((rr, cc))
    return out


def _build_unseeded_module(rng: random.Random, wall_color: int, bg: int) -> dict | None:
    kind = rng.choice(["straight", "elbow", "tee"])
    mirror_cells: set[tuple[int, int]] = set()

    if kind == "straight":
        direction = rng.choice(DIR4)
        length = rng.randint(4, 8)
        polyline = _build_polyline(direction, [length], [])
        if polyline is None:
            return None
        open_cells = set(polyline[0])
    elif kind == "elbow":
        direction = rng.choice(DIR4)
        lengths = [rng.randint(3, 6), rng.randint(3, 6)]
        turn = [rng.choice([-1, 1])]
        polyline = _build_polyline(direction, lengths, turn)
        if polyline is None:
            return None
        open_cells = set(polyline[0])
        if rng.random() < 0.7:
            mirror_cells = set(polyline[1])
    else:
        stem = rng.randint(4, 6)
        half_bar = rng.randint(2, 3)
        base = {(r, 0) for r in range(stem)}
        branch_row = 0 if rng.random() < 0.5 else rng.randint(1, stem - 2)
        base |= {(branch_row, c) for c in range(-half_bar, half_bar + 1)}
        open_cells = _rotate_points(base, rng.choice(DIR4))

    module = _make_module_grid(open_cells, wall_color, bg, mirror_cells=mirror_cells)
    if module is None:
        return None
    module["seeded"] = False
    module["fill_cells"] = []
    return module


def _place_module(
    reserved: np.ndarray,
    module_shape: tuple[int, int],
    rng: random.Random,
    gap: int = 1,
) -> tuple[int, int] | None:
    h, w = reserved.shape
    mh, mw = module_shape
    if mh > h or mw > w:
        return None

    slots = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    rng.shuffle(slots)
    for r0, c0 in slots:
        r1 = r0 + mh
        c1 = c0 + mw
        rr0 = max(0, r0 - gap)
        cc0 = max(0, c0 - gap)
        rr1 = min(h, r1 + gap)
        cc1 = min(w, c1 + gap)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        reserved[r0:r1, c0:c1] = True
        return r0, c0
    return None


def _stamp(grid: np.ndarray, module: dict, top: int, left: int, bg: int) -> None:
    patch = module["grid"]
    mask = patch != bg
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    sub[mask] = patch[mask]


def _apply_expected_output(inp: np.ndarray, placed: list[tuple[dict, int, int]], bg: int) -> np.ndarray:
    out = inp.copy()
    for module, top, left in placed:
        if not module["seeded"]:
            continue
        for r, c in module["seed_cells"]:
            out[top + r, left + c] = bg
        for r, c in module["fill_cells"]:
            out[top + r, left + c] = module["wall_color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(BG_CHOICES)
    h = rng.randint(20, 30)
    w = rng.randint(20, 30)
    inp = np.full((h, w), bg, dtype=int)

    n_seeded = rng.choice([1, 2])
    n_unseeded = rng.choice([0, 1, 2])
    total = n_seeded + n_unseeded

    wall_colors = [c for c in range(10) if c not in (bg, SEED_COLOR, MIRROR_COLOR)]
    rng.shuffle(wall_colors)
    if total > len(wall_colors):
        return None

    modules: list[dict] = []
    for _ in range(n_seeded):
        module = None
        for _ in range(40):
            module = _build_seeded_module(rng, wall_colors[len(modules)], bg)
            if module is not None:
                break
        if module is None:
            return None
        modules.append(module)
    for _ in range(n_unseeded):
        module = None
        for _ in range(40):
            module = _build_unseeded_module(rng, wall_colors[len(modules)], bg)
            if module is not None:
                break
        if module is None:
            return None
        modules.append(module)

    modules.sort(key=lambda m: m["grid"].shape[0] * m["grid"].shape[1], reverse=True)
    reserved = np.zeros((h, w), dtype=bool)
    placed: list[tuple[dict, int, int]] = []
    for module in modules:
        loc = _place_module(reserved, module["grid"].shape, rng, gap=1)
        if loc is None:
            return None
        top, left = loc
        _stamp(inp, module, top, left, bg)
        placed.append((module, top, left))

    if _get_bg(inp) != bg:
        return None

    out = _apply_expected_output(inp, placed, bg)
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
