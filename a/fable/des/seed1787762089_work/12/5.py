import random
from collections import deque

import numpy as np


BG = 0
END = 1
GREEN = 3
YELLOW = 4
FRAME = 5
MAGENTA = 6
OBSTACLE = 7
TRACK = 8
START = 9

STATION_COLORS = (GREEN, YELLOW, MAGENTA)
TRACK_GRAPH_COLORS = (END, GREEN, YELLOW, MAGENTA, TRACK, START)
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray, connectivity: int = 4):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _readout_slots(grid: np.ndarray):
    candidates = []
    for comp in _components(grid == FRAME, connectivity=4):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 != 2 or c1 - c0 < 2:
            continue
        if len(comp) != 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4:
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                border = r in (r0, r1) or c in (c0, c1)
                if border and grid[r, c] != FRAME:
                    ok = False
                    break
                if not border and grid[r, c] == FRAME:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((r0, c0, c1))
    if not candidates:
        raise ValueError("readout frame not found")
    r0, c0, c1 = max(candidates, key=lambda item: (item[0], item[2] - item[1]))
    return [(r0 + 1, c) for c in range(c0 + 1, c1)]


def _extract_track_nodes(grid: np.ndarray):
    track_mask = np.isin(grid, TRACK_GRAPH_COLORS)
    return {(int(r), int(c)) for r, c in np.argwhere(track_mask)}


def _forbidden_nodes(grid: np.ndarray, track_nodes: set[tuple[int, int]]):
    h, w = grid.shape
    forbidden = set()
    for r, c in np.argwhere(grid == OBSTACLE):
        rr = int(r)
        cc = int(c)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                if (nr, nc) in track_nodes:
                    forbidden.add((nr, nc))
    return forbidden


def _count_paths(nodes: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int], cap: int = 2):
    if start not in nodes or end not in nodes:
        return 0, []
    adj = {}
    for node in nodes:
        r, c = node
        nbrs = [nbr for nbr in _neighbors4(r, c) if nbr in nodes]
        adj[node] = sorted(nbrs)

    found_path = []
    count = 0
    path = [start]
    visited = {start}

    def dfs(cur: tuple[int, int]):
        nonlocal count, found_path
        if count >= cap:
            return
        if cur == end:
            count += 1
            if count == 1:
                found_path = path.copy()
            return
        for nxt in adj[cur]:
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(nxt)
            dfs(nxt)
            path.pop()
            visited.remove(nxt)
            if count >= cap:
                return

    dfs(start)
    return count, found_path


def _solve_path(grid: np.ndarray):
    starts = np.argwhere(grid == START)
    ends = np.argwhere(grid == END)
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("grid must contain exactly one start and one end")
    start = tuple(map(int, starts[0]))
    end = tuple(map(int, ends[0]))

    track_nodes = _extract_track_nodes(grid)
    forbidden = _forbidden_nodes(grid, track_nodes)
    passable = track_nodes - forbidden
    count, path = _count_paths(passable, start, end, cap=2)
    return count, path, forbidden


def _station_sequence(grid: np.ndarray, path: list[tuple[int, int]]):
    seq = []
    for r, c in path:
        val = int(grid[r, c])
        if val in STATION_COLORS:
            seq.append(val)
    return seq


def main(grid: np.ndarray) -> np.ndarray:
    """Read station colors along the obstacle-pruned unique route and write them into the bottom bar."""
    g = np.asarray(grid, dtype=int)
    count, path, _ = _solve_path(g)
    if count != 1:
        raise ValueError("expected exactly one valid start-to-end path after obstacle pruning")
    seq = _station_sequence(g, path)
    slots = _readout_slots(g)
    out = g.copy()
    for (r, c), color in zip(slots, seq):
        out[r, c] = color
    return out


def _translate(points, dr: int, dc: int):
    return [(r + dr, c + dc) for r, c in points]


def _transform_points(points, h: int, w: int, mode: str):
    if mode == "id":
        return [(r, c) for r, c in points]
    if mode == "h":
        return [(r, w - 1 - c) for r, c in points]
    if mode == "v":
        return [(h - 1 - r, c) for r, c in points]
    if mode == "hv":
        return [(h - 1 - r, w - 1 - c) for r, c in points]
    raise ValueError(mode)


def _sample_breakpoints(rng: random.Random, width: int, turns: int):
    for _ in range(100):
        mids = sorted(rng.sample(range(3, width - 3), turns))
        pts = [0] + mids + [width - 1]
        if min(b - a for a, b in zip(pts, pts[1:])) >= 4:
            return mids
    return None


def _build_backbone(rng: random.Random, height: int, width: int):
    for _ in range(120):
        turns = rng.choice([2, 3])
        mids = _sample_breakpoints(rng, width, turns)
        if mids is None:
            continue
        rows = [rng.randint(1, height - 2)]
        ok = True
        for _ in range(turns):
            choices = [r for r in range(1, height - 1) if abs(r - rows[-1]) >= 2]
            if not choices:
                ok = False
                break
            rows.append(rng.choice(choices))
        if not ok:
            continue

        path = [(rows[0], 0)]
        occupied = {path[0]}
        hsegments = []
        cur_r, cur_c = rows[0], 0
        all_cols = mids + [width - 1]
        valid = True
        for i, next_c in enumerate(all_cols):
            for c in range(cur_c + 1, next_c + 1):
                cell = (cur_r, c)
                if cell in occupied:
                    valid = False
                    break
                path.append(cell)
                occupied.add(cell)
            if not valid:
                break
            hsegments.append((cur_r, cur_c, next_c))
            cur_c = next_c
            if i < turns:
                target_r = rows[i + 1]
                step = 1 if target_r > cur_r else -1
                for r in range(cur_r + step, target_r + step, step):
                    cell = (r, cur_c)
                    if cell in occupied:
                        valid = False
                        break
                    path.append(cell)
                    occupied.add(cell)
                if not valid:
                    break
                cur_r = target_r
        if valid and any(c1 - c0 >= 6 for _, c0, c1 in hsegments):
            return path, hsegments
    return None, None


def _detour_cells(row: int, a: int, b: int, detour_row: int):
    cells = []
    step = 1 if detour_row > row else -1
    for r in range(row + step, detour_row + step, step):
        cells.append((r, a))
    for c in range(a + 1, b + 1):
        cells.append((detour_row, c))
    lo = min(row, detour_row)
    hi = max(row, detour_row)
    for r in range(lo + 1, hi):
        cells.append((r, b))
    # Remove duplicates while preserving order.
    seen = set()
    out = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return out


def _obstacle_candidates(cells, track_nodes: set[tuple[int, int]], protected: set[tuple[int, int]], height: int, width: int):
    detour_set = set(cells)
    cands = []
    for r, c in cells:
        for nr in range(max(0, r - 1), min(height, r + 2)):
            for nc in range(max(0, c - 1), min(width, c + 2)):
                pos = (nr, nc)
                if pos in track_nodes:
                    continue
                forbidden = {
                    cell for cell in track_nodes
                    if abs(cell[0] - nr) <= 1 and abs(cell[1] - nc) <= 1
                }
                if not forbidden:
                    continue
                if forbidden & protected:
                    continue
                if not forbidden <= detour_set:
                    continue
                cands.append(pos)
    return sorted(set(cands))


def _add_loop(rng: random.Random, height: int, width: int, backbone: list[tuple[int, int]], hsegments):
    occupied = set(backbone)
    segments = hsegments[:]
    rng.shuffle(segments)
    for row, c0, c1 in segments:
        if c1 - c0 < 6:
            continue
        offsets = [-3, -2, 2, 3]
        rng.shuffle(offsets)
        for offset in offsets:
            detour_row = row + offset
            if not (0 <= detour_row < height):
                continue
            for _ in range(20):
                a = rng.randint(c0 + 1, c1 - 4)
                b = rng.randint(a + 3, c1 - 1)
                detour = _detour_cells(row, a, b, detour_row)
                if any(cell in occupied for cell in detour):
                    continue
                track_nodes = occupied | set(detour)
                cands = _obstacle_candidates(detour, track_nodes, set(backbone), height, width)
                if not cands:
                    continue
                return detour, rng.choice(cands)
    return None, None


def _add_branch(rng: random.Random, height: int, width: int, backbone: list[tuple[int, int]], occupied: set[tuple[int, int]], obstacle: tuple[int, int]):
    candidates = backbone[2:-2]
    rng.shuffle(candidates)
    for attach in candidates:
        ar, ac = attach
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            length_choices = [2, 3, 4]
            rng.shuffle(length_choices)
            for length in length_choices:
                cells = []
                ok = True
                for step in range(1, length + 1):
                    nr = ar + dr * step
                    nc = ac + dc * step
                    cell = (nr, nc)
                    if not (0 <= nr < height and 0 <= nc < width):
                        ok = False
                        break
                    if cell in occupied:
                        ok = False
                        break
                    if abs(nr - obstacle[0]) <= 1 and abs(nc - obstacle[1]) <= 1:
                        ok = False
                        break
                    cells.append(cell)
                if ok:
                    return cells
    return None


def _extra_obstacle_candidates(track_nodes: set[tuple[int, int]], protected: set[tuple[int, int]], existing: set[tuple[int, int]], height: int, width: int):
    cands = []
    for r, c in track_nodes - protected:
        for nr in range(max(0, r - 1), min(height, r + 2)):
            for nc in range(max(0, c - 1), min(width, c + 2)):
                pos = (nr, nc)
                if pos in track_nodes or pos in existing:
                    continue
                forbidden = {
                    cell for cell in track_nodes
                    if abs(cell[0] - nr) <= 1 and abs(cell[1] - nc) <= 1
                }
                if not forbidden:
                    continue
                if forbidden & protected:
                    continue
                cands.append(pos)
    return sorted(set(cands))


def _paint_frame(grid: np.ndarray, left: int, width: int):
    row0 = grid.shape[0] - 3
    grid[row0, left:left + width] = FRAME
    grid[row0 + 2, left:left + width] = FRAME
    grid[row0 + 1, left] = FRAME
    grid[row0 + 1, left + width - 1] = FRAME


def _expected_output(grid: np.ndarray):
    out = grid.copy()
    count, path, _ = _solve_path(grid)
    if count != 1:
        return None
    seq = _station_sequence(grid, path)
    slots = _readout_slots(grid)
    if len(slots) != len(seq):
        return None
    for (r, c), color in zip(slots, seq):
        out[r, c] = color
    return out


def _gen_attempt(rng: random.Random):
    total_height = rng.randint(14, 19)
    total_width = rng.randint(16, 24)
    track_height = total_height - 4

    pattern_height = rng.randint(8, track_height)
    pattern_width = rng.randint(12, min(18, total_width - 2))

    backbone, hsegments = _build_backbone(rng, pattern_height, pattern_width)
    if backbone is None:
        return None

    detour, obstacle0 = _add_loop(rng, pattern_height, pattern_width, backbone, hsegments)
    if detour is None:
        return None

    occupied = set(backbone) | set(detour)
    branch = _add_branch(rng, pattern_height, pattern_width, backbone, occupied, obstacle0)
    if branch is None:
        return None
    occupied |= set(branch)

    obstacles = [obstacle0]
    if rng.random() < 0.5:
        protected = set(backbone)
        cands = _extra_obstacle_candidates(occupied, protected, set(obstacles), pattern_height, pattern_width)
        if cands:
            obstacles.append(rng.choice(cands))

    mode = rng.choice(["id", "h", "v", "hv"])
    backbone = _transform_points(backbone, pattern_height, pattern_width, mode)
    detour = _transform_points(detour, pattern_height, pattern_width, mode)
    branch = _transform_points(branch, pattern_height, pattern_width, mode)
    obstacles = _transform_points(obstacles, pattern_height, pattern_width, mode)

    xoff = rng.randint(1, total_width - pattern_width - 1)
    yoff = rng.randint(0, track_height - pattern_height)
    backbone = _translate(backbone, yoff, xoff)
    detour = _translate(detour, yoff, xoff)
    branch = _translate(branch, yoff, xoff)
    obstacles = _translate(obstacles, yoff, xoff)

    backbone_set = set(backbone)
    detour_set = set(detour)
    branch_set = set(branch)
    track_nodes = backbone_set | detour_set | branch_set

    path_station_count = rng.choice([1, 2])
    path_candidates = backbone[2:-2]
    if len(path_candidates) < path_station_count:
        return None

    path_station_cells = rng.sample(path_candidates, path_station_count)
    branch_station = rng.choice(branch)
    extra_offpath = []
    if path_station_count == 1:
        detour_choices = [cell for cell in detour if cell not in path_station_cells]
        if detour_choices:
            extra_offpath = [rng.choice(detour_choices)]
        else:
            branch_choices = [cell for cell in branch if cell != branch_station]
            if not branch_choices:
                return None
            extra_offpath = [rng.choice(branch_choices)]
    station_cells = path_station_cells + [branch_station] + extra_offpath
    if len(set(station_cells)) != 3:
        return None

    colors = list(STATION_COLORS)
    rng.shuffle(colors)
    station_by_cell = dict(zip(station_cells, colors))

    grid = np.full((total_height, total_width), BG, dtype=int)
    for r, c in track_nodes:
        grid[r, c] = TRACK
    for r, c in detour_set | branch_set:
        grid[r, c] = TRACK
    for (r, c), color in station_by_cell.items():
        grid[r, c] = color

    start = backbone[0]
    end = backbone[-1]
    if start in station_by_cell or end in station_by_cell:
        return None
    grid[start] = START
    grid[end] = END
    for r, c in obstacles:
        if grid[r, c] != BG:
            return None
        grid[r, c] = OBSTACLE

    full_count, _ = _count_paths(_extract_track_nodes(grid), start, end, cap=2)
    if full_count < 2:
        return None

    count, path, forbidden = _solve_path(grid)
    if count != 1:
        return None

    if any(cell in forbidden for cell in backbone_set):
        return None
    if not (detour_set & forbidden):
        return None

    seq = _station_sequence(grid, path)
    if len(seq) != path_station_count:
        return None

    readout_width = path_station_count + 2
    left = rng.randint(1, total_width - readout_width - 1)
    _paint_frame(grid, left, readout_width)

    expected = _expected_output(grid)
    if expected is None:
        return None
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
