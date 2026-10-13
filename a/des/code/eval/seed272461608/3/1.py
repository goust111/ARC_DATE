from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[Coord], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _outside_bg_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if int(grid[r, c]) == bg and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(grid[r, c]) == bg and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == bg and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _wall_color(grid: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(grid[grid != bg], return_counts=True)
    assert len(vals) > 0, "expected at least one non-background color"
    return int(vals[int(np.argmax(counts))])


def _segment(a: Coord, b: Coord) -> list[Coord]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("segment endpoints must be orthogonally aligned")


def _candidate_paths(seed: Coord, gap: Coord) -> list[list[Coord]]:
    sr, sc = seed
    gr, gc = gap
    candidates: list[list[Coord]] = []
    bends = [(sr, gc), (gr, sc)]
    for bend in bends:
        if bend != seed:
            try:
                first = _segment(seed, bend)
                second = _segment(bend, gap)
            except ValueError:
                continue
            candidates.append(first + second[1:])
        else:
            try:
                candidates.append(_segment(seed, gap))
            except ValueError:
                pass

    unique: list[list[Coord]] = []
    seen: set[tuple[Coord, ...]] = set()
    for path in candidates:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _turn_count(path: list[Coord]) -> int:
    dirs: list[Coord] = []
    for (r0, c0), (r1, c1) in zip(path, path[1:], strict=False):
        dirs.append((r1 - r0, c1 - c0))
    return sum(d0 != d1 for d0, d1 in zip(dirs, dirs[1:], strict=False))


def _valid_path(
    out: np.ndarray,
    bg: int,
    region: set[Coord],
    gap: Coord,
    path: list[Coord],
) -> bool:
    if path[-1] != gap:
        return False
    for cell in path[1:-1]:
        if cell not in region:
            return False
        if int(out[cell]) != bg:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each interior seed to the boundary gap of the same color and draw the shortest
    orthogonal path with at most one turn through the room interior.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    wall = _wall_color(g, bg)
    outside_bg = _outside_bg_mask(g, bg)

    gap_mask = np.zeros((h, w), dtype=bool)
    seed_mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (bg, wall):
                continue
            if any(0 <= nr < h and 0 <= nc < w and outside_bg[nr, nc] for nr, nc in _neighbors4(r, c)):
                gap_mask[r, c] = True
            else:
                seed_mask[r, c] = True

    passable = (g == bg) | seed_mask
    out = g.copy()

    for cells in _components(passable):
        if _touches_border(cells, g.shape):
            continue

        region = set(cells)
        seeds = [(r, c) for r, c in cells if seed_mask[r, c]]
        if len(seeds) != 1:
            continue
        seed = seeds[0]
        seed_color = int(g[seed])

        matching_gaps: list[Coord] = []
        for r in range(h):
            for c in range(w):
                if not gap_mask[r, c] or int(g[r, c]) != seed_color:
                    continue
                if any((nr, nc) in region for nr, nc in _neighbors4(r, c)):
                    matching_gaps.append((r, c))

        best: tuple[tuple[object, ...], list[Coord]] | None = None
        for gap in matching_gaps:
            for path in _candidate_paths(seed, gap):
                if not _valid_path(out, bg, region, gap, path):
                    continue
                metric = (
                    len(path) - 1,
                    _turn_count(path),
                    gap[0],
                    gap[1],
                    tuple(path),
                )
                if best is None or metric < best[0]:
                    best = (metric, path)

        assert best is not None, "each region should have one reachable matching gap"
        for r, c in best[1][1:-1]:
            out[r, c] = seed_color

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
REGION_COUNT_RANGE = (1, 3)
ROOM_H_RANGE = (5, 9)
ROOM_W_RANGE = (5, 10)
EXTRA_GAP_RANGE = (0, 2)
MAX_NOTCHES = 3
PLACEMENT_TRIES = 120
COLORS = tuple(range(10))
BRANCHES = ("straight_h", "straight_v", "turn_hv", "turn_vh")


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _boundary_cells(room: np.ndarray) -> list[Coord]:
    h, w = room.shape
    boundary: set[Coord] = set()
    for r, c in map(tuple, np.argwhere(room)):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not room[nr, nc]:
                boundary.add((nr, nc))
    return sorted(boundary)


def _room_neighbor_count(room: np.ndarray, cell: Coord) -> int:
    r, c = cell
    return sum(0 <= nr < room.shape[0] and 0 <= nc < room.shape[1] and room[nr, nc] for nr, nc in _neighbors4(r, c))


def _pick_branch_spec(ih: int, iw: int, branch: str) -> tuple[Coord, Coord, list[Coord]]:
    top = 2
    bottom = ih + 1
    left = 2
    right = iw + 1
    inner_rows = list(range(top + 1, bottom))
    inner_cols = list(range(left + 1, right))

    if branch == "straight_h":
        row = random.choice(inner_rows)
        if random.random() < 0.5:
            gap = (row, 1)
            seed = (row, random.choice(list(range(left + 2, right))))
        else:
            gap = (row, right + 1)
            seed = (row, random.choice(list(range(left + 1, right - 1))))
        path = _segment(seed, gap)
        return seed, gap, path

    if branch == "straight_v":
        col = random.choice(inner_cols)
        if random.random() < 0.5:
            gap = (1, col)
            seed = (random.choice(list(range(top + 2, bottom))), col)
        else:
            gap = (bottom + 1, col)
            seed = (random.choice(list(range(top + 1, bottom - 1))), col)
        path = _segment(seed, gap)
        return seed, gap, path

    if branch == "turn_hv":
        col = random.choice(inner_cols)
        seed_col_choices = [c for c in inner_cols if c != col]
        assert seed_col_choices
        if random.random() < 0.5:
            gap = (1, col)
            seed = (random.choice(inner_rows), random.choice(seed_col_choices))
        else:
            gap = (bottom + 1, col)
            seed = (random.choice(inner_rows), random.choice(seed_col_choices))
        bend = (seed[0], gap[1])
        path = _segment(seed, bend) + _segment(bend, gap)[1:]
        return seed, gap, path

    row = random.choice(inner_rows)
    seed_row_choices = [r for r in inner_rows if r != row]
    assert seed_row_choices
    if random.random() < 0.5:
        gap = (row, 1)
        seed = (random.choice(seed_row_choices), random.choice(inner_cols))
    else:
        gap = (row, right + 1)
        seed = (random.choice(seed_row_choices), random.choice(inner_cols))
    bend = (gap[0], seed[1])
    path = _segment(seed, bend) + _segment(bend, gap)[1:]
    return seed, gap, path


def _try_carve_notches(room: np.ndarray, protected: np.ndarray) -> np.ndarray:
    ih = room.shape[0] - 4
    iw = room.shape[1] - 4
    top = 2
    bottom = ih + 1
    left = 2
    right = iw + 1

    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    target = random.randint(0, MAX_NOTCHES)
    carved = room.copy()

    for side in sides:
        if target <= 0:
            break
        for _ in range(20):
            depth = random.randint(1, min(3, ih - 2 if side in ("top", "bottom") else iw - 2))
            span_len = random.randint(1, min(4, iw - 2 if side in ("top", "bottom") else ih - 2))
            if side == "top":
                c0 = random.randint(left, right - span_len + 1)
                rr = slice(top, top + depth)
                cc = slice(c0, c0 + span_len)
            elif side == "bottom":
                c0 = random.randint(left, right - span_len + 1)
                rr = slice(bottom - depth + 1, bottom + 1)
                cc = slice(c0, c0 + span_len)
            elif side == "left":
                r0 = random.randint(top, bottom - span_len + 1)
                rr = slice(r0, r0 + span_len)
                cc = slice(left, left + depth)
            else:
                r0 = random.randint(top, bottom - span_len + 1)
                rr = slice(r0, r0 + span_len)
                cc = slice(right - depth + 1, right + 1)

            removed = carved.copy()
            if not np.any(removed[rr, cc]):
                continue
            if np.any(protected[rr, cc]):
                continue
            removed[rr, cc] = False
            if removed.sum() < 10 or not _mask_connected(removed):
                continue
            carved = removed
            target -= 1
            break

    return carved


def _spaced_subset(candidates: list[Coord], avoid: list[Coord], count: int) -> list[Coord]:
    chosen: list[Coord] = []
    items = candidates[:]
    random.shuffle(items)
    for cell in items:
        if any(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) <= 1 for other in avoid + chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            break
    return chosen


def _render_region(
    room: np.ndarray,
    wall: int,
    seed: Coord,
    seed_color: int,
    gap: Coord,
    extra_gaps: list[tuple[Coord, int]],
    path: list[Coord],
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(room.shape, bg, dtype=int)
    out = np.full(room.shape, bg, dtype=int)

    gap_colors = {gap: seed_color}
    for cell, color in extra_gaps:
        gap_colors[cell] = color

    for cell in _boundary_cells(room):
        if cell in gap_colors:
            inp[cell] = gap_colors[cell]
            out[cell] = gap_colors[cell]
        else:
            inp[cell] = wall
            out[cell] = wall

    inp[seed] = seed_color
    out[seed] = seed_color
    for r, c in path[1:-1]:
        out[r, c] = seed_color
    return inp, out


def _build_region_patch(bg: int, wall: int, palette: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    ih = random.randint(*ROOM_H_RANGE)
    iw = random.randint(*ROOM_W_RANGE)
    room = np.zeros((ih + 4, iw + 4), dtype=bool)
    room[2 : ih + 2, 2 : iw + 2] = True

    branch = random.choice(BRANCHES)
    seed, gap, path = _pick_branch_spec(ih, iw, branch)

    protected = np.zeros_like(room, dtype=bool)
    for cell in path[:-1]:
        protected[cell] = True

    room = _try_carve_notches(room, protected)
    if not all(room[cell] for cell in path[:-1]):
        return None

    without_seed = room.copy()
    without_seed[seed] = False
    if not _mask_connected(without_seed):
        return None

    boundary = _boundary_cells(room)
    if gap not in boundary or _room_neighbor_count(room, gap) != 1:
        return None

    seed_color = random.choice(palette)
    extra_color_choices = [c for c in palette if c != seed_color]

    simple_boundary = [cell for cell in boundary if cell != gap and _room_neighbor_count(room, cell) == 1]
    extra_count = random.randint(EXTRA_GAP_RANGE[0], min(EXTRA_GAP_RANGE[1], len(simple_boundary)))
    extra_cells = _spaced_subset(simple_boundary, [gap], extra_count)
    if len(extra_cells) != extra_count:
        return None
    extra_colors = random.sample(extra_color_choices, extra_count)
    extra_gaps = list(zip(extra_cells, extra_colors, strict=False))

    inp, out = _render_region(room, wall, seed, seed_color, gap, extra_gaps, path, bg)

    for r, c in path[1:-1]:
        if int(inp[r, c]) != bg:
            return None
    return inp, out


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(patch != bg):
        r = int(r0 + dr)
        c = int(c0 + dc)
        r1 = max(0, r - 1)
        r2 = min(H, r + 2)
        c1 = max(0, c - 1)
        c2 = min(W, c + 2)
        if np.any(canvas[r1:r2, c1:c2] != bg):
            return False
    return True


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    region = patch != bg
    target = canvas[r0 : r0 + h, c0 : c0 + w]
    assert not np.any((target != bg) & region)
    target[region] = patch[region]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg, wall = random.sample(COLORS, 2)
    palette = [c for c in COLORS if c not in (bg, wall)]

    n_regions = random.randint(*REGION_COUNT_RANGE)
    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_regions):
        patch = _build_region_patch(bg, wall, palette)
        if patch is None:
            return None
        patches.append(patch)

    patches.sort(key=lambda item: int(np.count_nonzero(item[0] != bg)), reverse=True)
    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)

    for in_patch, out_patch in patches:
        ph, pw = in_patch.shape
        placed = False
        positions = [
            (r0, c0)
            for r0 in range(0, H - ph + 1)
            for c0 in range(0, W - pw + 1)
        ]
        random.shuffle(positions)
        for r0, c0 in positions[:PLACEMENT_TRIES]:
            if _can_place(inp, in_patch, r0, c0, bg):
                _paste_patch(inp, in_patch, r0, c0, bg)
                _paste_patch(out, out_patch, r0, c0, bg)
                placed = True
                break
        if not placed:
            return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
