from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GREY = 5
USABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
NUM_PATHS_RANGE = (2, 4)
TERMINAL_SIZE_RANGE = (4, 6)
MAX_STATIONS_PER_PATH = 3
MAX_GENERATE_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _expand8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(coords)
    return out


def _detect_stations(grid: np.ndarray) -> dict[tuple[int, int], dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    stations: dict[tuple[int, int], dict] = {}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            mids = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            if any(int(g[rr, cc]) != GREY for rr, cc in mids):
                continue

            corner_coords = [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
            corner_vals = [int(g[rr, cc]) for rr, cc in corner_coords]
            out_color = int(corner_vals[0])
            if out_color in (BG, GREY) or any(v != out_color for v in corner_vals[1:]):
                continue

            center_val = int(g[r, c])
            if center_val in (BG, GREY):
                continue

            used_mids: list[tuple[int, int]] = []
            for rr, cc in mids:
                dr, dc = int(rr - r), int(cc - c)
                er, ec = int(rr + dr), int(cc + dc)
                if 0 <= er < h and 0 <= ec < w and int(g[er, ec]) != BG:
                    used_mids.append((int(rr), int(cc)))

            if len(used_mids) != 2:
                continue

            stations[(int(r), int(c))] = {
                "output_color": int(out_color),
                "corners": set((int(rr), int(cc)) for rr, cc in corner_coords),
                "mids": set((int(rr), int(cc)) for rr, cc in mids),
            }

    return stations


def _detect_terminals(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    candidates: list[dict] = []
    max_size = min(h, w)

    for size in range(3, max_size + 1):
        for r0 in range(0, h - size + 1):
            for c0 in range(0, w - size + 1):
                patch = g[r0 : r0 + size, c0 : c0 + size]
                if not bool((patch[0, :] == GREY).all()):
                    continue
                if not bool((patch[-1, :] == GREY).all()):
                    continue
                if not bool((patch[:, 0] == GREY).all()):
                    continue
                if not bool((patch[:, -1] == GREY).all()):
                    continue
                if size > 2 and not bool((patch[1:-1, 1:-1] == BG).all()):
                    continue

                border: set[tuple[int, int]] = set()
                for cc in range(c0, c0 + size):
                    border.add((int(r0), int(cc)))
                    border.add((int(r0 + size - 1), int(cc)))
                for rr in range(r0 + 1, r0 + size - 1):
                    border.add((int(rr), int(c0)))
                    border.add((int(rr), int(c0 + size - 1)))

                interior = [
                    (int(rr), int(cc))
                    for rr in range(r0 + 1, r0 + size - 1)
                    for cc in range(c0 + 1, c0 + size - 1)
                ]
                candidates.append(
                    {
                        "bbox": (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)),
                        "border": border,
                        "interior": interior,
                        "area": int(size * size),
                    }
                )

    candidates.sort(key=lambda item: item["area"], reverse=True)
    chosen: list[dict] = []
    used_cells: set[tuple[int, int]] = set()
    for cand in candidates:
        if cand["border"] & used_cells:
            continue
        chosen.append(cand)
        used_cells |= cand["border"]

    chosen.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return chosen


def _shortest_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    q = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        r, c = q.popleft()
        if (r, c) == end:
            break
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not bool(mask[nr, nc]) or (nr, nc) in parent:
                continue
            parent[(int(nr), int(nc))] = (int(r), int(c))
            q.append((int(nr), int(nc)))

    if end not in parent:
        return []

    route: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        route.append(cur)
        cur = parent[cur]
    route.reverse()
    return route


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a color signal along each boundary-starting path, overwrite that
    signal at every converter station, and fill the paired terminal square with
    the final color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    out = g.copy()

    terminals = _detect_terminals(g)
    if not terminals:
        return out

    stations = _detect_stations(g)
    station_corner_cells = {cell for station in stations.values() for cell in station["corners"]}
    terminal_border_cells = {cell for terminal in terminals for cell in terminal["border"]}

    traversable = (g != BG).copy()
    for r, c in station_corner_cells | terminal_border_cells:
        traversable[r, c] = False

    cell_to_terminal: dict[tuple[int, int], set[int]] = {}
    for tid, terminal in enumerate(terminals):
        for br, bc in terminal["border"]:
            for nr, nc in _neighbors4(br, bc):
                if 0 <= nr < h and 0 <= nc < w and bool(traversable[nr, nc]):
                    cell_to_terminal.setdefault((int(nr), int(nc)), set()).add(int(tid))

    for comp in _components(traversable):
        comp_set = set(comp)
        if not comp_set:
            continue

        boundary_cells = [
            (r, c)
            for r, c in comp
            if r == 0 or c == 0 or r == h - 1 or c == w - 1
        ]
        if not boundary_cells:
            continue

        degrees = {
            (r, c): sum((nr, nc) in comp_set for nr, nc in _neighbors4(r, c))
            for r, c in comp
        }

        leaf_boundary = [cell for cell in boundary_cells if degrees[cell] <= 1]
        start = min(leaf_boundary or boundary_cells)

        terminal_hits = {
            tid
            for cell in comp
            for tid in cell_to_terminal.get(cell, set())
        }
        if len(terminal_hits) != 1:
            continue
        tid = next(iter(terminal_hits))

        end_candidates = [cell for cell in comp if tid in cell_to_terminal.get(cell, set())]
        leaf_ends = [cell for cell in end_candidates if degrees[cell] <= 1]
        end = max(leaf_ends or end_candidates, key=lambda cell: abs(cell[0] - start[0]) + abs(cell[1] - start[1]))

        comp_mask = np.zeros_like(traversable, dtype=bool)
        for r, c in comp:
            comp_mask[r, c] = True
        route = _shortest_path(comp_mask, start, end)
        if not route:
            continue

        signal = int(g[start])
        for cell in route[1:]:
            if cell in stations:
                signal = int(stations[cell]["output_color"])

        for r, c in terminals[tid]["interior"]:
            out[r, c] = int(signal)

    return out


def _frame_cells(r0: int, c0: int, size: int) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    border: set[tuple[int, int]] = set()
    for c in range(c0, c0 + size):
        border.add((int(r0), int(c)))
        border.add((int(r0 + size - 1), int(c)))
    for r in range(r0 + 1, r0 + size - 1):
        border.add((int(r), int(c0)))
        border.add((int(r), int(c0 + size - 1)))
    interior = [
        (int(r), int(c))
        for r in range(r0 + 1, r0 + size - 1)
        for c in range(c0 + 1, c0 + size - 1)
    ]
    return border, interior


def _frame_outside_neighbors(border: set[tuple[int, int]], bbox: tuple[int, int, int, int], h: int, w: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out: set[tuple[int, int]] = set()
    for r, c in border:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if r0 <= nr <= r1 and c0 <= nc <= c1:
                continue
            out.add((int(nr), int(nc)))
    return out


def _boundary_candidates(h: int, w: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    return [(r, w - 1) for r in range(1, h - 1)]


def _boundary_blocked(h: int, w: int, keep: tuple[int, int]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if (r, c) != keep:
                    blocked.add((int(r), int(c)))
    return blocked


def _random_point(
    rng: np.random.Generator,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
    avoid: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if (r, c) in forbidden:
                continue
            too_close = False
            for ar, ac in avoid:
                if max(abs(r - ar), abs(c - ac)) <= 3:
                    too_close = True
                    break
            if not too_close:
                candidates.append((int(r), int(c)))
    if not candidates:
        return None
    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


def _randomized_bfs(
    h: int,
    w: int,
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    if start in blocked or end in blocked:
        return None

    q = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        r, c = q.popleft()
        if (r, c) == end:
            break
        order = list(DIRS4)
        rng.shuffle(order)
        for dr, dc in order:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (int(nr), int(nc))
            if nxt in blocked or nxt in parent:
                continue
            parent[nxt] = (int(r), int(c))
            q.append(nxt)

    if end not in parent:
        return None

    route: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        route.append(cur)
        cur = parent[cur]
    route.reverse()
    return route


def _route_with_waypoints(
    h: int,
    w: int,
    waypoints: list[tuple[int, int]],
    base_blocked: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    path_cells: set[tuple[int, int]] = set()
    halo_cells: set[tuple[int, int]] = set()
    full_route: list[tuple[int, int]] = []

    for idx in range(len(waypoints) - 1):
        start = waypoints[idx]
        end = waypoints[idx + 1]

        relax: set[tuple[int, int]] = set()
        for base in (start, end):
            br, bc = base
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = int(br + dr), int(bc + dc)
                    if 0 <= nr < h and 0 <= nc < w:
                        relax.add((nr, nc))

        blocked = set(base_blocked) | set(path_cells) | {cell for cell in halo_cells if cell not in relax}
        blocked.discard(start)
        blocked.discard(end)

        segment = _randomized_bfs(h, w, start, end, blocked, rng)
        if not segment:
            return None

        if idx > 0:
            segment = segment[1:]
        full_route.extend(segment)
        path_cells |= set(segment)
        halo_cells |= _expand8(set(segment), h, w)

    return full_route


def _eligible_station_centers(
    route: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    route_set = set(route)
    eligible: list[tuple[int, int]] = []

    for idx in range(1, len(route) - 1):
        center = route[idx]
        r, c = center
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            continue

        prev_cell = route[idx - 1]
        next_cell = route[idx + 1]
        allowed = {center, prev_cell, next_cell}
        ok = True
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                cell = (int(rr), int(cc))
                if cell in allowed:
                    continue
                if cell in occupied:
                    ok = False
                    break
                if cell in route_set:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            eligible.append(center)

    return eligible


def _select_station_centers(
    rng: np.random.Generator,
    eligible: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    pool = list(eligible)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for cell in pool:
        if all(max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) >= 3 for other in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _station_mid_and_corner_cells(center: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    r, c = center
    mids = {(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))}
    corners = {
        (int(r - 1), int(c - 1)),
        (int(r - 1), int(c + 1)),
        (int(r + 1), int(c - 1)),
        (int(r + 1), int(c + 1)),
    }
    return mids, corners


def _place_terminals(
    rng: np.random.Generator,
    h: int,
    w: int,
    n_paths: int,
) -> list[dict] | None:
    terminals: list[dict] = []
    reserved: set[tuple[int, int]] = set()

    for _ in range(n_paths):
        placed = False
        for _attempt in range(300):
            size = int(rng.integers(TERMINAL_SIZE_RANGE[0], TERMINAL_SIZE_RANGE[1] + 1))
            if h - size - 4 <= 1 or w - size - 4 <= 1:
                return None
            r0 = int(rng.integers(2, h - size - 1))
            c0 = int(rng.integers(2, w - size - 1))
            r1, c1 = int(r0 + size - 1), int(c0 + size - 1)

            footprint = {
                (rr, cc)
                for rr in range(max(0, r0 - 3), min(h, r1 + 4))
                for cc in range(max(0, c0 - 3), min(w, c1 + 4))
            }
            if footprint & reserved:
                continue

            side = ["top", "bottom", "left", "right"][int(rng.integers(0, 4))]
            offset = int(rng.integers(1, size - 1))
            if side == "top":
                border_cell = (int(r0), int(c0 + offset))
                endpoint = (int(r0 - 1), int(c0 + offset))
            elif side == "bottom":
                border_cell = (int(r1), int(c0 + offset))
                endpoint = (int(r1 + 1), int(c0 + offset))
            elif side == "left":
                border_cell = (int(r0 + offset), int(c0))
                endpoint = (int(r0 + offset), int(c0 - 1))
            else:
                border_cell = (int(r0 + offset), int(c1))
                endpoint = (int(r0 + offset), int(c1 + 1))

            if not _in_bounds(h, w, endpoint[0], endpoint[1]):
                continue
            if endpoint in reserved:
                continue

            border, interior = _frame_cells(r0, c0, size)
            terminals.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "border": border,
                    "interior": interior,
                    "endpoint": endpoint,
                    "border_cell": border_cell,
                }
            )
            reserved |= footprint
            placed = True
            break
        if not placed:
            return None

    return terminals


def _draw_terminal(grid: np.ndarray, terminal: dict) -> None:
    for r, c in terminal["border"]:
        grid[r, c] = GREY


def _station_count_vector(rng: np.random.Generator, n_paths: int) -> np.ndarray:
    for _ in range(100):
        counts = rng.integers(0, MAX_STATIONS_PER_PATH + 1, size=n_paths)
        if 0 not in counts:
            continue
        if not bool((counts > 0).any()):
            continue
        if n_paths >= 3 and not bool((counts >= 2).any()):
            continue
        return counts
    return rng.integers(0, MAX_STATIONS_PER_PATH + 1, size=n_paths)


def _build_expected_output(grid: np.ndarray, fills: list[tuple[list[tuple[int, int]], int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for interior, color in fills:
        for r, c in interior:
            out[r, c] = int(color)
    return out


def _split_sizes(total: int, parts: int, minimum: int, rng: np.random.Generator) -> list[int]:
    sizes = [int(minimum)] * parts
    extra = int(total - parts * minimum)
    if extra <= 0:
        return sizes
    cuts = sorted(int(x) for x in rng.integers(0, extra + 1, size=parts - 1))
    prev = 0
    for idx, cut in enumerate(cuts + [extra]):
        sizes[idx] += int(cut - prev)
        prev = int(cut)
    return sizes


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    route: list[tuple[int, int]] = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        assert r0 == r1 or c0 == c1
        dr = (r1 > r0) - (r1 < r0)
        dc = (c1 > c0) - (c1 < c0)
        rr, cc = int(r0), int(c0)
        while (rr, cc) != (r1, c1):
            rr += int(dr)
            cc += int(dc)
            route.append((int(rr), int(cc)))
    return route


def _canonical_layout(
    rng: np.random.Generator,
    h: int,
    w: int,
    n_paths: int,
    path_colors: list[int],
    station_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    margin = 1
    gap = 1
    min_lane = 6
    total_lane_space = int(h - 2 * margin - gap * (n_paths - 1))
    if total_lane_space < n_paths * min_lane:
        return None

    lane_sizes = _split_sizes(total_lane_space, n_paths, min_lane, rng)
    lanes: list[tuple[int, int]] = []
    cursor = int(margin)
    for size in lane_sizes:
        lanes.append((int(cursor), int(cursor + size - 1)))
        cursor += int(size + gap)

    grid = np.full((h, w), BG, dtype=int)
    fills: list[tuple[list[tuple[int, int]], int]] = []
    occupied: set[tuple[int, int]] = set()

    for lane_idx, ((lane_r0, lane_r1), path_color, station_count) in enumerate(zip(lanes, path_colors, station_counts)):
        lane_h = int(lane_r1 - lane_r0 + 1)
        size_hi = min(6, lane_h - 2)
        if size_hi < 4:
            return None

        start_left = bool(rng.integers(0, 2) == 0)
        size = int(rng.integers(4, size_hi + 1))
        term_r0 = int(rng.integers(lane_r0 + 1, lane_r1 - size + 2))
        target_row = int(term_r0 + rng.integers(1, size - 1))

        if start_left:
            c0_lo = max(9, w // 2)
            c0_hi = w - size - 2
        else:
            c0_lo = 2
            c0_hi = min(w // 2 - size, w - size - 2)
        if c0_lo > c0_hi:
            return None
        term_c0 = int(rng.integers(c0_lo, c0_hi + 1))

        border, interior = _frame_cells(term_r0, term_c0, size)
        for r, c in border:
            grid[r, c] = GREY
        occupied |= border

        if start_left:
            endpoint = (int(target_row), int(term_c0 - 1))
            start_col = 0
            inner_far = int(term_c0 - 2)
            x1_lo, x1_hi = 2, max(2, inner_far - 5)
        else:
            endpoint = (int(target_row), int(term_c0 + size))
            start_col = int(w - 1)
            inner_far = int(term_c0 + size + 1)
            x1_lo, x1_hi = min(w - 3, inner_far + 5), w - 3

        if not (1 <= endpoint[1] <= w - 2):
            return None

        available_rows = [r for r in range(lane_r0, lane_r1 + 1) if r != target_row]
        if len(available_rows) < 2:
            return None
        rng.shuffle(available_rows)
        start_row = int(available_rows[0])
        mid_row = int(available_rows[1])

        template_b = bool(station_count >= 2 or rng.integers(0, 2) == 0)
        if start_left:
            if x1_lo > x1_hi:
                return None
            x1 = int(rng.integers(x1_lo, x1_hi + 1))
            if template_b:
                x2_lo = x1 + 3
                x2_hi = inner_far - 2
                if x2_lo > x2_hi:
                    return None
                x2 = int(rng.integers(x2_lo, x2_hi + 1))
                points = [
                    (int(start_row), int(start_col)),
                    (int(start_row), int(x1)),
                    (int(mid_row), int(x1)),
                    (int(mid_row), int(x2)),
                    (int(target_row), int(x2)),
                    endpoint,
                ]
            else:
                points = [
                    (int(start_row), int(start_col)),
                    (int(start_row), int(x1)),
                    (int(target_row), int(x1)),
                    endpoint,
                ]
        else:
            if x1_lo > x1_hi:
                return None
            x1 = int(rng.integers(x1_lo, x1_hi + 1))
            if template_b:
                x2_lo = inner_far + 1
                x2_hi = x1 - 3
                if x2_lo > x2_hi:
                    return None
                x2 = int(rng.integers(x2_lo, x2_hi + 1))
                points = [
                    (int(start_row), int(start_col)),
                    (int(start_row), int(x1)),
                    (int(mid_row), int(x1)),
                    (int(mid_row), int(x2)),
                    (int(target_row), int(x2)),
                    endpoint,
                ]
            else:
                points = [
                    (int(start_row), int(start_col)),
                    (int(start_row), int(x1)),
                    (int(target_row), int(x1)),
                    endpoint,
                ]

        route = _polyline(points)
        route_set = set(route)
        if len(route_set) != len(route):
            return None
        if route_set & occupied:
            return None

        eligible = _eligible_station_centers(route, occupied, h, w)
        centers = _select_station_centers(rng, eligible, int(station_count))
        if station_count > 0 and centers is None:
            return None
        chosen_centers = centers or []

        signal = int(path_color)
        route_index = {cell: i for i, cell in enumerate(route)}
        stations_for_path: list[dict] = []
        for center in sorted(chosen_centers, key=lambda cell: route_index[cell]):
            output_choices = [color for color in USABLE_COLORS if color != signal]
            output_color = int(output_choices[int(rng.integers(0, len(output_choices)))])
            mids, corners = _station_mid_and_corner_cells(center)
            if corners & occupied:
                return None
            if any(cell in route_set and cell not in {center, route[route_index[center] - 1], route[route_index[center] + 1]} for cell in mids | corners):
                return None
            stations_for_path.append(
                {
                    "center": center,
                    "mids": mids,
                    "corners": corners,
                    "output_color": output_color,
                }
            )
            signal = int(output_color)

        for r, c in route:
            grid[r, c] = int(path_color)
        new_occupied = set(route)
        for station in stations_for_path:
            for r, c in station["mids"]:
                grid[r, c] = GREY
            for r, c in station["corners"]:
                grid[r, c] = int(station["output_color"])
            new_occupied |= station["mids"]
            new_occupied |= station["corners"]
        occupied |= new_occupied
        fills.append((interior, int(signal)))

        if lane_idx == 0 and len(route) < 8:
            return None

    expected = _build_expected_output(grid, fills)
    return grid, expected


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    max_paths = 1
    for candidate in range(NUM_PATHS_RANGE[1], NUM_PATHS_RANGE[0] - 1, -1):
        if h - 2 - (candidate - 1) >= candidate * 6:
            max_paths = candidate
            break
    if max_paths < NUM_PATHS_RANGE[0]:
        return None

    n_paths = int(rng.integers(NUM_PATHS_RANGE[0], max_paths + 1))
    path_colors = list(rng.choice(np.array(USABLE_COLORS, dtype=int), size=n_paths, replace=False))
    station_counts = _station_count_vector(rng, n_paths)

    built = _canonical_layout(rng, h, w, n_paths, path_colors, station_counts)
    if built is None:
        return None
    grid, expected = built

    k = int(rng.integers(0, 4))
    if k:
        grid = np.rot90(grid, k)
        expected = np.rot90(expected, k)

    if not np.array_equal(main(grid), expected):
        return None
    if len(_detect_terminals(grid)) != n_paths:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATE_ATTEMPTS} attempts")
