import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SEED = 6
GATE_COLORS = [1, 2, 3, 4, 7, 8, 9]

GEN_RETRIES = 5000
_RNG = random.Random()


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _boundary_mask(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    out = np.zeros_like(region, dtype=bool)
    for r, c in zip(*np.where(region)):
        for nr, nc in neighbors4(int(r), int(c)):
            if not (0 <= nr < h and 0 <= nc < w) or not region[nr, nc]:
                out[r, c] = True
                break
    return out


def _ordered_cycle(contour: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    neigh = {}
    for cell in contour:
        ns = [p for p in neighbors4(*cell) if p in contour]
        if len(ns) != 2:
            raise AssertionError("contour must be a simple 4-connected cycle")
        neigh[cell] = ns

    def trace(first_next: tuple[int, int]) -> list[tuple[int, int]] | None:
        order = [start]
        prev = start
        cur = first_next
        while cur != start:
            order.append(cur)
            nxts = [p for p in neigh[cur] if p != prev]
            if len(nxts) != 1:
                return None
            prev, cur = cur, nxts[0]
            if len(order) > len(contour):
                return None
        return order if len(order) == len(contour) else None

    order = trace(neigh[start][0])
    if order is None:
        order = trace(neigh[start][1])
    if order is None:
        raise AssertionError("failed to trace contour cycle")

    area2 = 0
    for (r1, c1), (r2, c2) in zip(order, order[1:] + order[:1]):
        area2 += c1 * r2 - c2 * r1
    if area2 < 0:
        order = [start] + list(reversed(order[1:]))
    return order


def _inside_region(grid: np.ndarray, seed: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    sr, sc = seed
    seen = {(sr, sc)}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) in (BG, SEED):
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    seeds = np.argwhere(g == SEED)
    assert len(seeds) == 1, "expected exactly one magenta seed"
    seed = tuple(int(x) for x in seeds[0])

    inside = _inside_region(g, seed)

    contour_mask = (g != BG) & (g != SEED)
    best_comp = None
    best_key = None
    for comp in _components(contour_mask):
        adj = 0
        for r, c in comp:
            if any((nr, nc) in inside for nr, nc in neighbors4(r, c)):
                adj += 1
        if adj == 0:
            continue
        key = (adj, len(comp))
        if best_key is None or key > best_key:
            best_key = key
            best_comp = comp
    assert best_comp is not None, "failed to find the main contour"

    contour = set(best_comp)
    start = min(
        contour,
        key=lambda p: (abs(p[0] - seed[0]) + abs(p[1] - seed[1]), p[0], p[1]),
    )
    order = _ordered_cycle(contour, start)

    gate_indices = [i for i, (r, c) in enumerate(order) if int(g[r, c]) != GREY]
    assert gate_indices, "expected at least one gate color on the contour"
    first_gate = gate_indices[0]
    rotated = order[first_gate:] + order[:first_gate]

    source_info: dict[tuple[int, int], tuple[int, int]] = {}
    current_color = int(g[rotated[0]])
    for step, (r, c) in enumerate(rotated):
        val = int(g[r, c])
        if val != GREY:
            current_color = val
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in inside:
                source_info[(nr, nc)] = (step, current_color)
    assert source_info, "contour must touch the enclosed interior"

    out = g.copy()
    queue = deque()
    painted: dict[tuple[int, int], int] = {}
    for cell, (_, color) in sorted(source_info.items(), key=lambda kv: kv[1][0]):
        painted[cell] = color
        out[cell] = color
        queue.append(cell)

    while queue:
        r, c = queue.popleft()
        color = painted[(r, c)]
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in inside and (nr, nc) not in painted:
                painted[(nr, nc)] = color
                out[nr, nc] = color
                queue.append((nr, nc))

    return out


def _crop(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _pick_segment(rng: random.Random, length: int) -> tuple[int, int] | None:
    if length < 4:
        return None
    max_span = max(2, length - 2)
    span = rng.randint(2, max_span)
    max_start = length - span - 1
    if max_start < 1:
        return None
    start = rng.randint(1, max_start)
    return start, span


def _make_shape(rng: random.Random) -> np.ndarray | None:
    ph = rng.randint(10, 17)
    pw = rng.randint(10, 17)
    mt = rng.randint(1, 3)
    mb = rng.randint(1, 3)
    ml = rng.randint(1, 3)
    mr = rng.randint(1, 3)

    max_rh = ph - mt - mb
    max_rw = pw - ml - mr
    if max_rh < 6 or max_rw < 6:
        return None

    rh = rng.randint(6, max_rh)
    rw = rng.randint(6, max_rw)
    top = mt
    left = ml

    mask = np.zeros((ph, pw), dtype=bool)
    mask[top : top + rh, left : left + rw] = True

    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides:
        if side in ("top", "bottom"):
            seg = _pick_segment(rng, rw)
            if seg is None:
                continue
            offset, span = seg
            add_depth = top if side == "top" else ph - (top + rh)
            cut_depth = rh - 3
            modes = ["none"]
            if add_depth >= 1:
                modes.append("bump")
            if cut_depth >= 1:
                modes.append("notch")
            mode = rng.choice(modes)
            if mode == "bump":
                depth = rng.randint(1, add_depth)
                cols = slice(left + offset, left + offset + span)
                if side == "top":
                    mask[top - depth : top, cols] = True
                else:
                    mask[top + rh : top + rh + depth, cols] = True
            elif mode == "notch":
                depth = rng.randint(1, min(3, cut_depth))
                cols = slice(left + offset, left + offset + span)
                if side == "top":
                    mask[top : top + depth, cols] = False
                else:
                    mask[top + rh - depth : top + rh, cols] = False
        else:
            seg = _pick_segment(rng, rh)
            if seg is None:
                continue
            offset, span = seg
            add_depth = left if side == "left" else pw - (left + rw)
            cut_depth = rw - 3
            modes = ["none"]
            if add_depth >= 1:
                modes.append("bump")
            if cut_depth >= 1:
                modes.append("notch")
            mode = rng.choice(modes)
            if mode == "bump":
                depth = rng.randint(1, add_depth)
                rows = slice(top + offset, top + offset + span)
                if side == "left":
                    mask[rows, left - depth : left] = True
                else:
                    mask[rows, left + rw : left + rw + depth] = True
            elif mode == "notch":
                depth = rng.randint(1, min(3, cut_depth))
                rows = slice(top + offset, top + offset + span)
                if side == "left":
                    mask[rows, left : left + depth] = False
                else:
                    mask[rows, left + rw - depth : left + rw] = False

    if not mask.any():
        return None
    mask = _crop(mask)

    if len(_components(mask)) != 1:
        return None
    boundary = _boundary_mask(mask)
    interior = mask & ~boundary
    if interior.sum() < 6:
        return None
    if len(_components(interior)) != 1:
        return None

    boundary_comps = _components(boundary)
    if len(boundary_comps) != 1:
        return None
    boundary_set = set(boundary_comps[0])
    for r, c in boundary_set:
        deg = sum((nr, nc) in boundary_set for nr, nc in neighbors4(r, c))
        if deg != 2:
            return None

    return mask


def _sample_gate_positions(
    rng: random.Random,
    order: list[tuple[int, int]],
    contour: set[tuple[int, int]],
    gate_count: int,
) -> list[int] | None:
    straight = []
    for idx, cell in enumerate(order):
        neigh = [p for p in neighbors4(*cell) if p in contour]
        if len(neigh) != 2:
            continue
        if neigh[0][0] == neigh[1][0] or neigh[0][1] == neigh[1][1]:
            straight.append(idx)
    if len(straight) < gate_count:
        return None

    min_gap = max(3, len(order) // (gate_count * 3))
    for _ in range(200):
        picks = sorted(rng.sample(straight, gate_count))
        gaps = [
            (picks[(i + 1) % gate_count] - picks[i]) % len(order)
            for i in range(gate_count)
        ]
        if min(gaps) >= min_gap:
            return picks
    return None


def _fragment_shapes() -> list[list[tuple[int, int]]]:
    return [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
    ]


def _place_fragments(
    grid: np.ndarray,
    region_mask: np.ndarray,
    contour_cells: set[tuple[int, int]],
    rng: random.Random,
):
    h, w = grid.shape
    current_grey = set(contour_cells)
    for _ in range(rng.randint(0, 4)):
        shape = rng.choice(_fragment_shapes())
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        placed = False
        for _ in range(120):
            r0 = rng.randint(0, h - sh)
            c0 = rng.randint(0, w - sw)
            pts = [(r0 + r, c0 + c) for r, c in shape]
            if any(region_mask[r, c] or int(grid[r, c]) != BG for r, c in pts):
                continue
            bad = False
            for r, c in pts:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) in current_grey:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            for r, c in pts:
                grid[r, c] = GREY
                current_grey.add((r, c))
            placed = True
            break
        if not placed:
            continue


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region = _make_shape(rng)
    if region is None:
        return None

    boundary = _boundary_mask(region)
    contour_local = set(_components(boundary)[0])
    start = min(contour_local)
    order_local = _ordered_cycle(contour_local, start)

    max_gates = min(5, len(order_local) // 6, len(GATE_COLORS))
    if max_gates < 2:
        return None
    gate_count = rng.randint(2, max_gates)
    gate_positions = _sample_gate_positions(rng, order_local, contour_local, gate_count)
    if gate_positions is None:
        return None
    gate_colors = rng.sample(GATE_COLORS, gate_count)

    rh, rw = region.shape
    min_h = max(12, rh + 4)
    min_w = max(12, rw + 4)
    max_h = min(28, rh + 10)
    max_w = min(28, rw + 10)
    if min_h > max_h or min_w > max_w:
        return None

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    top = rng.randint(1, h - rh - 1)
    left = rng.randint(1, w - rw - 1)

    grid = np.zeros((h, w), dtype=int)
    region_mask = np.zeros((h, w), dtype=bool)
    region_mask[top : top + rh, left : left + rw] = region

    contour_cells: set[tuple[int, int]] = set()
    for r, c in zip(*np.where(boundary)):
        gr, gc = top + int(r), left + int(c)
        grid[gr, gc] = GREY
        contour_cells.add((gr, gc))

    global_order = [(top + r, left + c) for r, c in order_local]
    for idx, color in zip(gate_positions, gate_colors):
        r, c = global_order[idx]
        grid[r, c] = color

    interior_cells = [(top + int(r), left + int(c)) for r, c in zip(*np.where(region & ~boundary))]
    if not interior_cells:
        return None

    if rng.random() < 0.5:
        boundary_cells = list(contour_cells)
        depths = []
        for cell in interior_cells:
            depth = min(abs(cell[0] - br) + abs(cell[1] - bc) for br, bc in boundary_cells)
            depths.append(depth)
        max_depth = max(depths)
        candidates = [cell for cell, depth in zip(interior_cells, depths) if depth == max_depth]
        seed = rng.choice(candidates)
    else:
        seed = rng.choice(interior_cells)
    grid[seed] = SEED

    _place_fragments(grid, region_mask, contour_cells, rng)

    out = main(grid)
    if out[seed] == SEED:
        return None
    for color in gate_colors:
        if int(np.count_nonzero(out == color)) <= int(np.count_nonzero(grid == color)):
            return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
