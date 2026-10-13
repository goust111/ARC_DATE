from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 300
GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
ROOM_COUNT_MAX = 4
ROOM_PAD = 2
ROOM_H_RANGE = (7, 13)
ROOM_W_RANGE = (7, 13)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _wall_outline(solid: np.ndarray) -> np.ndarray:
    h, w = solid.shape
    wall = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not solid[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not solid[nr, nc]:
                    wall[r, c] = True
                    break
    return wall


def _interior_from_wall(wall: np.ndarray) -> np.ndarray | None:
    free = ~wall
    comps = [comp for comp in _components(free) if not _touches_edge(comp, *wall.shape)]
    if len(comps) != 1:
        return None
    interior = np.zeros_like(wall, dtype=bool)
    for r, c in comps[0]:
        interior[r, c] = True
    return interior


def _boundary_depth(interior: np.ndarray, wall: np.ndarray) -> np.ndarray | None:
    h, w = interior.shape
    depth = np.full((h, w), -1, dtype=int)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if not interior[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and wall[nr, nc] for nr, nc in _neighbors4(r, c)):
                depth[r, c] = 0
                queue.append((r, c))
    if not queue:
        return None
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and interior[nr, nc] and depth[nr, nc] < 0:
                depth[nr, nc] = depth[r, c] + 1
                queue.append((nr, nc))
    return depth


def _seed_priority_distance(grid: np.ndarray, region_mask: np.ndarray, seed: tuple[int, int]) -> int:
    h, w = grid.shape
    sr, sc = seed
    if any(0 <= nr < h and 0 <= nc < w and grid[nr, nc] == GREY for nr, nc in _neighbors4(sr, sc)):
        return 0

    black_mask = region_mask & (grid == BLACK)
    depth = np.full((h, w), -1, dtype=int)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if not black_mask[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and grid[nr, nc] == GREY for nr, nc in _neighbors4(r, c)):
                depth[r, c] = 0
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and black_mask[nr, nc] and depth[nr, nc] < 0:
                depth[nr, nc] = depth[r, c] + 1
                queue.append((nr, nc))

    best: int | None = None
    for nr, nc in _neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w and black_mask[nr, nc] and depth[nr, nc] >= 0:
            cand = depth[nr, nc] + 1
            if best is None or cand < best:
                best = cand
    assert best is not None, "seed must connect to interior black cells or touch the wall"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand territory from colored seed pixels inside grey-walled rooms.

    For each enclosed non-grey region that does not touch the grid boundary:
    - collect the seed pixels (colors other than black/grey),
    - rank them by shortest 4-connected black-only distance to the wall,
    - break equal distances by row, then column,
    - run a multi-source flood fill whose initial queue follows that priority.

    Queue order gives earlier seeds ownership of cells reached at the same step,
    which produces the intended boundary shape while keeping the wall unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    region_mask = g != GREY
    for comp in _components(region_mask):
        if _touches_edge(comp, h, w):
            continue

        room = np.zeros((h, w), dtype=bool)
        seeds: list[tuple[int, int, int]] = []
        for r, c in comp:
            room[r, c] = True
            val = int(g[r, c])
            if val not in (BLACK, GREY):
                seeds.append((r, c, val))

        assert 2 <= len(seeds) <= 4, "each enclosed room must contain 2-4 seeds"

        ordered = []
        for r, c, color in seeds:
            d = _seed_priority_distance(g, room, (r, c))
            ordered.append((d, r, c, color))
        ordered.sort()

        queue: deque[tuple[int, int, int]] = deque((r, c, color) for _, r, c, color in ordered)
        while queue:
            r, c, color = queue.popleft()
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and room[nr, nc] and out[nr, nc] == BLACK:
                    out[nr, nc] = color
                    queue.append((nr, nc, color))

    return out


def _random_solid(h: int, w: int, rng: random.Random) -> np.ndarray:
    solid = np.ones((h, w), dtype=bool)
    notch_count = rng.randint(0, 3)
    for _ in range(notch_count):
        side = rng.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            if h < 7 or w < 5:
                continue
            max_depth = min(max(1, h // 3), h - 4)
            if max_depth < 1:
                continue
            depth = rng.randint(1, max_depth)
            max_span = w - 2
            if max_span < 2:
                continue
            span = rng.randint(2, max_span)
            if w - span - 1 < 1:
                continue
            c0 = rng.randint(1, w - span - 1)
            rows = slice(0, depth) if side == "top" else slice(h - depth, h)
            solid[rows, c0 : c0 + span] = False
        else:
            if w < 7 or h < 5:
                continue
            max_depth = min(max(1, w // 3), w - 4)
            if max_depth < 1:
                continue
            depth = rng.randint(1, max_depth)
            max_span = h - 2
            if max_span < 2:
                continue
            span = rng.randint(2, max_span)
            if h - span - 1 < 1:
                continue
            r0 = rng.randint(1, h - span - 1)
            cols = slice(0, depth) if side == "left" else slice(w - depth, w)
            solid[r0 : r0 + span, cols] = False
    return solid


def _choose_seed_positions(
    interior: np.ndarray,
    depth: np.ndarray,
    rng: random.Random,
    force_tie: bool,
    force_mixed: bool,
) -> list[tuple[int, int]] | None:
    coords = [(int(r), int(c)) for r, c in np.argwhere(interior)]
    groups: dict[int, list[tuple[int, int]]] = {}
    for r, c in coords:
        groups.setdefault(int(depth[r, c]), []).append((r, c))

    if force_tie and not any(len(v) >= 2 for v in groups.values()):
        return None
    if force_mixed and len(groups) < 2:
        return None

    area = len(coords)
    max_seeds = 2 + int(area >= 12) + int(area >= 20)
    max_seeds = min(4, max_seeds)
    depths_all = sorted(groups)

    for min_sep in (3, 2, 1):
        for _ in range(400):
            count = rng.randint(2, max_seeds)
            target_depths: list[int] = []

            if force_tie:
                tie_depths = [d for d, cells in groups.items() if len(cells) >= 2]
                tie_depth = rng.choice(tie_depths)
                target_depths.extend([tie_depth, tie_depth])

            if force_mixed:
                if not target_depths:
                    a, b = rng.sample(depths_all, 2)
                    target_depths.extend([a, b])
                elif len(set(target_depths)) == 1:
                    others = [d for d in depths_all if d != target_depths[0]]
                    if not others:
                        continue
                    target_depths.append(rng.choice(others))

            while len(target_depths) < count:
                if len(depths_all) >= 2 and rng.random() < 0.55:
                    target_depths.append(depths_all[0] if rng.random() < 0.5 else depths_all[-1])
                else:
                    target_depths.append(rng.choice(depths_all))

            if len(target_depths) > max_seeds:
                continue
            count = len(target_depths)
            rng.shuffle(target_depths)

            chosen: list[tuple[int, int]] = []
            ok = True
            for d in target_depths:
                pool = groups[d][:]
                rng.shuffle(pool)
                pick = None
                for cell in pool:
                    if cell in chosen:
                        continue
                    if all(abs(cell[0] - rr) + abs(cell[1] - cc) >= min_sep for rr, cc in chosen):
                        pick = cell
                        break
                if pick is None:
                    ok = False
                    break
                chosen.append(pick)

            if not ok:
                continue

            chosen_depths = [int(depth[r, c]) for r, c in chosen]
            if force_tie and len(set(chosen_depths)) == len(chosen_depths):
                continue
            if force_mixed and len(set(chosen_depths)) < 2:
                continue
            return chosen

    return None


def _room_attempt(
    h: int,
    w: int,
    rng: random.Random,
    force_tie: bool,
    force_mixed: bool,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[int]] | None:
    for _ in range(120):
        bh = rng.randint(ROOM_H_RANGE[0], min(ROOM_H_RANGE[1], h - 2))
        bw = rng.randint(ROOM_W_RANGE[0], min(ROOM_W_RANGE[1], w - 2))
        solid = _random_solid(bh, bw, rng)
        wall = _wall_outline(solid)
        interior = _interior_from_wall(wall)
        if interior is None or int(interior.sum()) < 6:
            continue

        depth = _boundary_depth(interior, wall)
        if depth is None:
            continue

        local_force_tie = force_tie
        if not local_force_tie and any(len(np.argwhere(depth == d)) >= 2 for d in np.unique(depth[interior])) and rng.random() < 0.3:
            local_force_tie = True

        local_force_mixed = force_mixed or (len(set(int(depth[r, c]) for r, c in np.argwhere(interior))) >= 2 and rng.random() < 0.7)
        seeds = _choose_seed_positions(interior, depth, rng, local_force_tie, local_force_mixed)
        if seeds is None:
            continue

        seed_depths = [int(depth[r, c]) for r, c in seeds]
        return wall, interior, seeds, seed_depths
    return None


def _placeable(used: np.ndarray, r0: int, c0: int, wall: np.ndarray) -> bool:
    h, w = used.shape
    bh, bw = wall.shape
    rs = max(0, r0 - ROOM_PAD)
    re = min(h, r0 + bh + ROOM_PAD)
    cs = max(0, c0 - ROOM_PAD)
    ce = min(w, c0 + bw + ROOM_PAD)
    return not used[rs:re, cs:ce].any()


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(*GRID_H_RANGE)
        w = rng.randint(*GRID_W_RANGE)
        grid = np.full((h, w), BLACK, dtype=int)
        used = np.zeros((h, w), dtype=bool)

        max_rooms = min(ROOM_COUNT_MAX, max(1, (h * w) // 170))
        target_rooms = rng.randint(1, max_rooms)
        need_tie = rng.random() < 0.45
        have_tie = False
        have_mixed = False
        placed_rooms = 0

        for room_idx in range(target_rooms):
            remaining = target_rooms - room_idx
            force_tie = need_tie and not have_tie and remaining == 1
            force_mixed = not have_mixed and remaining == 1
            placed = False

            for _ in range(120):
                room = _room_attempt(h, w, rng, force_tie, force_mixed)
                if room is None:
                    continue
                wall, _interior, seeds, seed_depths = room
                bh, bw = wall.shape
                if bh >= h - 1 or bw >= w - 1:
                    continue
                r0 = rng.randint(1, h - bh - 1)
                c0 = rng.randint(1, w - bw - 1)
                if not _placeable(used, r0, c0, wall):
                    continue

                grid[r0 : r0 + bh, c0 : c0 + bw][wall] = GREY
                colors = rng.sample(SEED_COLORS, len(seeds))
                for color, (sr, sc) in zip(colors, seeds):
                    grid[r0 + sr, c0 + sc] = color

                rs = max(0, r0 - ROOM_PAD)
                re = min(h, r0 + bh + ROOM_PAD)
                cs = max(0, c0 - ROOM_PAD)
                ce = min(w, c0 + bw + ROOM_PAD)
                used[rs:re, cs:ce] = True

                placed_rooms += 1
                have_tie |= len(set(seed_depths)) < len(seed_depths)
                have_mixed |= len(set(seed_depths)) >= 2
                placed = True
                break

            if not placed:
                break

        if placed_rooms != target_rooms or not have_mixed or (need_tie and not have_tie):
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue

        free = grid != GREY
        for comp in _components(free):
            if _touches_edge(comp, h, w):
                continue
            if any(out[r, c] == BLACK for r, c in comp):
                break
        else:
            return grid

    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
