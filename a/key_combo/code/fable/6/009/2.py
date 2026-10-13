from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATE_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def bfs_mask(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = start
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def detect_wall_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != bg)
    assert counts, "expected at least one non-background color"
    return int(max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0])


def enclosed_regions(grid: np.ndarray, wall_color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    passable = g != int(wall_color)
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return components_mask(passable & ~outside)


def solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    wall_color = detect_wall_color(g, bg)
    out = g.copy()

    for comp in enclosed_regions(g, wall_color):
        comp_mask = np.zeros_like(g, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        marker_coords = np.argwhere(comp_mask & (g != bg) & (g != wall_color))
        assert len(marker_coords) == 1, "each enclosed room must contain exactly one marker"
        mr, mc = map(int, marker_coords[0])
        marker_color = int(g[mr, mc])
        dist = bfs_mask(comp_mask, (mr, mc))
        assert np.all(dist[comp_mask] >= 0), "room interior must be fully reachable from its marker"
        out[comp_mask & (dist % 2 == 0)] = marker_color
        out[comp_mask & (dist % 2 == 1)] = GREY

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each enclosed room into alternating shortest-path layers from its marker.

    Input structure:
    - Black background surrounds several closed rooms made from one shared wall color.
    - Each room contains exactly one non-wall marker pixel, and at least one room also contains wall-colored internal obstacle segments.
    - Room interiors otherwise contain background cells that are separated from the outside by the wall loops.

    Rule:
    - For every enclosed room, treat wall-colored cells as blocked and compute four-neighbor shortest-path distances from that room's marker to every passable cell in the same room.
    - Recolor even-distance cells, including the marker itself at distance 0, with that room's marker color.
    - Recolor odd-distance cells with grey, while leaving all wall cells and the outside background unchanged.

    Key insight:
    - The fill domain is not the whole grid but each enclosed room separately, so the marker must be matched to its own wall-bounded component before the alternating geodesic layers are painted.
    """
    return solve(grid)


def random_path(
    ih: int,
    iw: int,
    rng: random.Random,
    used: set[tuple[int, int]] | None = None,
    min_len: int = 2,
    max_len: int = 5,
) -> set[tuple[int, int]] | None:
    used = set() if used is None else set(used)
    rows = range(1, max(2, ih - 1))
    cols = range(1, max(2, iw - 1))
    starts = [(r, c) for r in rows for c in cols if (r, c) not in used]
    rng.shuffle(starts)
    for start in starts:
        path = [start]
        blocked = used | {start}
        target_len = rng.randint(min_len, max_len)
        while len(path) < target_len:
            r, c = path[-1]
            candidates = []
            for nr, nc in neighbors4(r, c):
                if not (1 <= nr < ih - 1 and 1 <= nc < iw - 1):
                    continue
                if (nr, nc) in blocked:
                    continue
                if any((nr, nc) == prev for prev in path[:-1]):
                    continue
                ok = True
                for ar, ac in neighbors4(nr, nc):
                    if (ar, ac) in used:
                        ok = False
                        break
                if ok:
                    candidates.append((nr, nc))
            if not candidates:
                break
            nxt = rng.choice(candidates)
            path.append(nxt)
            blocked.add(nxt)
        if len(path) >= min_len:
            return set(path)
    return None


def long_barrier(ih: int, iw: int, rng: random.Random) -> set[tuple[int, int]] | None:
    if ih < 5 or iw < 5:
        return None
    vertical_ok = iw >= 6
    horizontal_ok = ih >= 6
    if not vertical_ok and not horizontal_ok:
        return None
    orient = rng.choice(
        [o for o, ok in (("v", vertical_ok), ("h", horizontal_ok)) if ok]
    )
    cells: set[tuple[int, int]] = set()
    if orient == "v":
        col_lo = 2 if iw >= 7 else 1
        col_hi = iw - 3 if iw >= 7 else iw - 2
        if col_lo > col_hi:
            return None
        col = rng.randint(col_lo, col_hi)
        top = rng.choice([1, 2] if ih >= 7 else [1])
        bottom = ih - 1 - rng.choice([1, 2] if ih >= 7 else [1])
        if bottom - top + 1 < max(3, ih - 4):
            top = 1
            bottom = ih - 2
        for r in range(top, bottom + 1):
            cells.add((r, col))
    else:
        row_lo = 2 if ih >= 7 else 1
        row_hi = ih - 3 if ih >= 7 else ih - 2
        if row_lo > row_hi:
            return None
        row = rng.randint(row_lo, row_hi)
        left = rng.choice([1, 2] if iw >= 7 else [1])
        right = iw - 1 - rng.choice([1, 2] if iw >= 7 else [1])
        if right - left + 1 < max(3, iw - 4):
            left = 1
            right = iw - 2
        for c in range(left, right + 1):
            cells.add((row, c))
    return cells


def dogleg_barrier(ih: int, iw: int, rng: random.Random) -> set[tuple[int, int]] | None:
    base = long_barrier(ih, iw, rng)
    if not base:
        return None
    cells = set(base)
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(cols) == 1:
        col = next(iter(cols))
        end_r = min(rows) if rng.random() < 0.5 else max(rows)
        dirs = []
        if col - 2 >= 1:
            dirs.append(-1)
        if col + 2 < iw - 1:
            dirs.append(1)
        if not dirs:
            return cells
        dc = rng.choice(dirs)
        arm_len = rng.randint(2, max(2, min(4, iw - 2)))
        for k in range(1, arm_len + 1):
            nc = col + dc * k
            if not (1 <= nc < iw - 1):
                break
            cells.add((end_r, nc))
    else:
        row = next(iter(rows))
        end_c = min(cols) if rng.random() < 0.5 else max(cols)
        dirs = []
        if row - 2 >= 1:
            dirs.append(-1)
        if row + 2 < ih - 1:
            dirs.append(1)
        if not dirs:
            return cells
        dr = rng.choice(dirs)
        arm_len = rng.randint(2, max(2, min(4, ih - 2)))
        for k in range(1, arm_len + 1):
            nr = row + dr * k
            if not (1 <= nr < ih - 1):
                break
            cells.add((nr, end_c))
    return cells


def build_obstacles(ih: int, iw: int, rng: random.Random, want_obstacle: bool) -> set[tuple[int, int]] | None:
    if not want_obstacle:
        return set()

    makers = [long_barrier, dogleg_barrier]
    rng.shuffle(makers)
    for maker in makers:
        cells = maker(ih, iw, rng)
        if cells:
            break
    else:
        return None

    if rng.random() < 0.45:
        extra = random_path(ih, iw, rng, used=cells, min_len=2, max_len=4)
        if extra:
            cells |= extra
    return cells


def detour_cell_count(passable: np.ndarray, marker: tuple[int, int]) -> int:
    dist = bfs_mask(passable, marker)
    mr, mc = marker
    detours = 0
    for r, c in np.argwhere(passable):
        manhattan = abs(int(r) - mr) + abs(int(c) - mc)
        if dist[r, c] > manhattan:
            detours += 1
    return detours


def room_patch(
    ih: int,
    iw: int,
    wall_color: int,
    marker_color: int,
    want_obstacle: bool,
    rng: random.Random,
) -> np.ndarray | None:
    for _ in range(80):
        obstacles = build_obstacles(ih, iw, rng, want_obstacle)
        if obstacles is None:
            continue
        passable = np.ones((ih, iw), dtype=bool)
        for r, c in obstacles:
            passable[r, c] = False
        comps = components_mask(passable)
        if len(comps) != 1 or int(passable.sum()) < 12:
            continue

        coords = [tuple(map(int, x)) for x in np.argwhere(passable)]
        rng.shuffle(coords)
        chosen_marker = None
        detours = 0
        for marker in coords:
            detours = detour_cell_count(passable, marker)
            if want_obstacle and detours < 3:
                continue
            chosen_marker = marker
            break
        if chosen_marker is None:
            continue

        patch = np.full((ih + 2, iw + 2), BG, dtype=int)
        patch[0, :] = wall_color
        patch[-1, :] = wall_color
        patch[:, 0] = wall_color
        patch[:, -1] = wall_color
        for r, c in obstacles:
            patch[r + 1, c + 1] = wall_color
        mr, mc = chosen_marker
        patch[mr + 1, mc + 1] = marker_color
        if want_obstacle and detours < 3:
            continue
        return patch
    return None


def can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    h0, w0 = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h0, top + h + pad)
    c1 = min(w0, left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(np.sum(grid == GREY)) != 0:
        return False
    bg = get_bg(grid)
    if bg != BG:
        return False
    wall_color = detect_wall_color(grid, bg)
    rooms = enclosed_regions(grid, wall_color)
    if not rooms:
        return False

    marker_colors = set()
    obstacle_room_found = False
    detour_room_found = False
    for comp in rooms:
        comp_mask = np.zeros_like(grid, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        markers = np.argwhere(comp_mask & (grid != bg) & (grid != wall_color))
        if len(markers) != 1:
            return False
        mr, mc = map(int, markers[0])
        marker_color = int(grid[mr, mc])
        if marker_color in marker_colors or marker_color in (BG, GREY, wall_color):
            return False
        marker_colors.add(marker_color)

        rs = comp[:, 0]
        cs = comp[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        inner_wall = grid[r0 : r1 + 1, c0 : c1 + 1] == wall_color
        if np.any(inner_wall):
            obstacle_room_found = True
            dist = bfs_mask(comp_mask, (mr, mc))
            manhattan = np.abs(rs - mr) + np.abs(cs - mc)
            if np.any(dist[rs, cs] > manhattan):
                detour_room_found = True

    if not obstacle_room_found or not detour_room_found:
        return False

    return np.array_equal(main(grid), solve(grid))


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_ATTEMPTS):
        h = rng.randint(18, 30)
        w = rng.randint(18, 30)
        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        room_count = rng.choice([1, 2, 2, 3])
        wall_color = rng.choice(PALETTE)
        marker_colors = rng.sample([c for c in PALETTE if c != wall_color], room_count)

        obstacle_flags = [False] * room_count
        for idx in rng.sample(range(room_count), rng.randint(1, room_count)):
            obstacle_flags[idx] = True
        if room_count > 1 and all(obstacle_flags):
            obstacle_flags[rng.randrange(room_count)] = False
        if not any(obstacle_flags):
            obstacle_flags[rng.randrange(room_count)] = True

        placed = 0
        for i in range(room_count):
            max_side = 11 if room_count == 1 else 9 if room_count == 2 else 8
            min_side = 5
            patch = None
            for _room_try in range(40):
                ih = rng.randint(min_side, max_side)
                iw = rng.randint(min_side, max_side)
                patch = room_patch(ih, iw, wall_color, marker_colors[i], obstacle_flags[i], rng)
                if patch is not None:
                    break
            if patch is None:
                break

            ph, pw = patch.shape
            positions = [
                (top, left)
                for top in range(0, h - ph + 1)
                for left in range(0, w - pw + 1)
                if can_place(occupied, top, left, ph, pw, pad=1)
            ]
            if not positions:
                break
            top, left = rng.choice(positions)
            place_patch(grid, patch, top, left)
            occupied[max(0, top - 1) : min(h, top + ph + 1), max(0, left - 1) : min(w, left + pw + 1)] = True
            placed += 1

        if placed != room_count:
            continue
        if validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid input grid")
