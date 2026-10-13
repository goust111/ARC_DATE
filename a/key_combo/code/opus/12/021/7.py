from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
HALO = 1
MARKER = 5
WALL = 6
TOKEN = 9

ORTH_DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}

SIDE_CELLS = {
    "N": [(-1, -1), (-1, 0), (-1, 1)],
    "S": [(1, -1), (1, 0), (1, 1)],
    "W": [(-1, -1), (0, -1), (1, -1)],
    "E": [(-1, 1), (0, 1), (1, 1)],
}

EXTRA_BY_DIR = {
    "N": [(-1, -1), (-1, 1)],
    "S": [(1, -1), (1, 1)],
    "W": [(-1, -1), (1, -1)],
    "E": [(-1, 1), (1, 1)],
}

GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _marker_neighbors(marker: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = marker
    cells = []
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) != marker and _in_bounds(shape, rr, cc):
                cells.append((rr, cc))
    return cells


def _paint_halos(grid: np.ndarray, markers: list[tuple[int, int]]):
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    for mr, mc in markers:
        for rr in range(mr - 1, mr + 2):
            for cc in range(mc - 1, mc + 2):
                if not _in_bounds((h, w), rr, cc):
                    continue
                if out[rr, cc] == BG:
                    out[rr, cc] = HALO
    slots_by_marker: list[list[tuple[int, int]]] = []
    for marker in markers:
        slots = [cell for cell in _marker_neighbors(marker, out.shape) if out[cell] == HALO]
        slots_by_marker.append(sorted(slots))
    return out, slots_by_marker


def _distances_from_marker(grid_with_halo: np.ndarray, marker: tuple[int, int]) -> np.ndarray:
    h, w = grid_with_halo.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque([marker])
    dist[marker] = 0
    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((h, w), nr, nc):
                continue
            if dist[nr, nc] != -1:
                continue
            if grid_with_halo[nr, nc] not in (BG, TOKEN):
                continue
            dist[nr, nc] = nd
            q.append((nr, nc))
    return dist


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    markers = sorted((int(r), int(c)) for r, c in np.argwhere(g == MARKER))
    halo_grid, slots_by_marker = _paint_halos(g, markers)
    distances = [_distances_from_marker(halo_grid, marker) for marker in markers]

    maroons = sorted((int(r), int(c)) for r, c in np.argwhere(g == TOKEN))
    assigned: list[list[tuple[int, int, int]]] = [[] for _ in markers]
    maroon_info: dict[tuple[int, int], dict] = {}

    for cell in maroons:
        reachable = []
        for idx, marker in enumerate(markers):
            d = int(distances[idx][cell])
            if d >= 0:
                reachable.append((d, marker[0], marker[1], idx))
        if not reachable:
            maroon_info[cell] = {
                "cell": cell,
                "assigned": None,
                "dist": None,
                "reachable": (),
                "best_tied": False,
            }
            continue
        reachable.sort()
        best = reachable[0]
        best_tied = len(reachable) >= 2 and reachable[1][0] == best[0]
        assigned[best[3]].append((best[0], cell[0], cell[1]))
        maroon_info[cell] = {
            "cell": cell,
            "assigned": best[3],
            "dist": best[0],
            "reachable": tuple(reachable),
            "best_tied": best_tied,
        }

    moved_by_marker: list[list[tuple[int, int, int]]] = []
    moved_sources: set[tuple[int, int]] = set()
    capacities = [len(slots) for slots in slots_by_marker]
    for idx, items in enumerate(assigned):
        items.sort()
        chosen = items[: capacities[idx]]
        moved_by_marker.append(chosen)
        moved_sources.update((r, c) for _d, r, c in chosen)

    out = halo_grid.copy()
    for r, c in moved_sources:
        out[r, c] = BG
    for idx, slots in enumerate(slots_by_marker):
        for slot, (_d, sr, sc) in zip(slots, moved_by_marker[idx]):
            _ = (sr, sc)
            out[slot] = TOKEN

    return {
        "markers": markers,
        "halo_grid": halo_grid,
        "slots_by_marker": slots_by_marker,
        "capacities": capacities,
        "distances": distances,
        "maroons": maroons,
        "maroon_info": maroon_info,
        "assigned": assigned,
        "moved_by_marker": moved_by_marker,
        "moved_sources": moved_sources,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather reachable maroon pixels into the blue halo cells generated around each gray marker.

    Input structure:
    - The grid is mostly black background.
    - Two or three gray (5) marker pixels are spaced apart; magenta (6) wall segments and maroon (9) pixels are scattered around them.
    - Markers may lie near the border, so their 3x3 neighborhoods can be clipped by the grid edge.

    Rule:
    - First paint every background cell in each marker's clipped 3x3 neighborhood blue (1), while leaving the gray markers, walls, and maroon pixels unchanged.
    - Then, for every maroon pixel, compute its shortest 4-connected path length to each gray marker using only background and maroon cells; blue halo cells, walls, and gray markers cannot be traversed.
    - Assign each reachable maroon to its nearest reachable marker, breaking equal-distance ties by smaller marker row, then smaller marker column; for each marker, sort its assigned maroons by distance and place them into that marker's blue halo cells in row-major order, erasing only the maroons that were successfully moved.

    Key insight:
    - The destination regions are not fixed in advance: each marker first creates its own halo, and that halo's blocked cells and clipped shape determine both the reachable entrances and the fill capacity.
    """
    return _analyze(np.asarray(grid, dtype=int))["output"]


def _side_positions(marker: tuple[int, int], side: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = marker
    cells = []
    for dr, dc in SIDE_CELLS[side]:
        rr, cc = r + dr, c + dc
        if _in_bounds(shape, rr, cc):
            cells.append((rr, cc))
    return cells


def _line_segment(base_cells: list[tuple[int, int]], shape: tuple[int, int], forbidden: set[tuple[int, int]]):
    if len(base_cells) < 2:
        return None
    cells = sorted(base_cells)
    horizontal = len({r for r, _c in cells}) == 1
    if horizontal:
        cells = sorted(cells, key=lambda p: p[1])
        step = (0, 1)
    else:
        cells = sorted(cells, key=lambda p: p[0])
        step = (1, 0)

    target = random.randint(max(3, len(cells)), 5)
    cellset = set(cells)
    while len(cells) < target:
        first = cells[0]
        last = cells[-1]
        before = (first[0] - step[0], first[1] - step[1])
        after = (last[0] + step[0], last[1] + step[1])
        options = []
        for pos, add_front in ((before, True), (after, False)):
            if _in_bounds(shape, pos[0], pos[1]) and pos not in forbidden and pos not in cellset:
                options.append((pos, add_front))
        if not options:
            break
        pos, add_front = random.choice(options)
        cellset.add(pos)
        if add_front:
            cells.insert(0, pos)
        else:
            cells.append(pos)
    if len(cells) < 3:
        return None
    return cells


def _local_pattern(
    marker: tuple[int, int],
    shape: tuple[int, int],
    forced_dir: str | None = None,
    allowed_wall_sides: tuple[str, ...] | None = None,
):
    dirs = [forced_dir] if forced_dir is not None else random.sample(list(ORTH_DIRS), k=4)
    halo_neighbors = set(_marker_neighbors(marker, shape))

    for direction in dirs:
        dr, dc = ORTH_DIRS[direction]
        primary = (marker[0] + dr, marker[1] + dc)
        if primary not in halo_neighbors:
            continue
        extras = []
        for edr, edc in EXTRA_BY_DIR[direction]:
            extra = (marker[0] + edr, marker[1] + edc)
            if extra in halo_neighbors:
                extras.append(extra)
        random.shuffle(extras)
        if not extras:
            continue

        wall_candidates = [side for side in ("N", "S", "W", "E") if side != direction]
        if allowed_wall_sides is not None:
            wall_candidates = [side for side in wall_candidates if side in allowed_wall_sides]
        random.shuffle(wall_candidates)

        for extra in extras:
            maroons = {primary, extra}
            for side in wall_candidates:
                base = _side_positions(marker, side, shape)
                if len(base) < 2:
                    continue
                if primary in base or extra in base:
                    continue
                capacity = len(halo_neighbors - maroons - set(base))
                if 1 <= capacity <= 3:
                    return {
                        "primary_dir": direction,
                        "local_maroons": sorted(maroons),
                        "wall_side": side,
                        "wall_base": base,
                        "capacity": capacity,
                    }

    return None


def _pairwise_far_enough(cells: list[tuple[int, int]]) -> bool:
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 6:
                return False
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 2:
                return False
    return True


def _pick_markers(h: int, w: int, n: int, tie_branch: bool, edge_branch: bool):
    if tie_branch:
        row = random.randint(2, h - 3)
        separations = [sep for sep in (6, 8, 10) if sep <= w - 5 and sep % 2 == 0]
        if not separations:
            return None
        sep = random.choice(separations)
        left = random.randint(2, w - sep - 3)
        markers = [(row, left), (row, left + sep)]
        return markers if _pairwise_far_enough(markers) else None

    for _ in range(100):
        markers = []
        if edge_branch:
            side = random.choice(["top", "bottom", "left", "right"])
            if side == "top":
                markers.append((0, random.randint(2, w - 3)))
            elif side == "bottom":
                markers.append((h - 1, random.randint(2, w - 3)))
            elif side == "left":
                markers.append((random.randint(2, h - 3), 0))
            else:
                markers.append((random.randint(2, h - 3), w - 1))

        while len(markers) < n:
            cell = (random.randint(0, h - 1), random.randint(0, w - 1))
            if cell in markers:
                continue
            trial = markers + [cell]
            if _pairwise_far_enough(trial):
                markers.append(cell)

        if _pairwise_far_enough(markers):
            return sorted(markers)
    return None


def _forced_dir_for_edge(marker: tuple[int, int], shape: tuple[int, int]) -> str | None:
    h, w = shape
    r, c = marker
    if r == 0:
        return "S"
    if r == h - 1:
        return "N"
    if c == 0:
        return "E"
    if c == w - 1:
        return "W"
    return None


def _marker_regions(markers: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    covered = set()
    for marker in markers:
        covered.add(marker)
        covered.update(_marker_neighbors(marker, shape))
    return covered


def _place_unreachable_box(grid: np.ndarray, blocked: set[tuple[int, int]]) -> tuple[bool, tuple[int, int] | None]:
    h, w = grid.shape
    sizes = [size for size in (4, 5) if size <= h and size <= w]
    random.shuffle(sizes)
    for size in sizes:
        top_choices = list(range(0, h - size + 1))
        left_choices = list(range(0, w - size + 1))
        random.shuffle(top_choices)
        random.shuffle(left_choices)
        for top in top_choices:
            for left in left_choices:
                border = set()
                interior = []
                ok = True
                for r in range(top, top + size):
                    for c in range(left, left + size):
                        if r in (top, top + size - 1) or c in (left, left + size - 1):
                            border.add((r, c))
                        else:
                            interior.append((r, c))
                zone = border | set(interior)
                for cell in zone:
                    if cell in blocked or grid[cell] != BG:
                        ok = False
                        break
                if not ok or not interior:
                    continue
                for r, c in border:
                    grid[r, c] = WALL
                token_cell = random.choice(interior)
                grid[token_cell] = TOKEN
                return True, token_cell
    return False, None


def _try_add_token_for_marker(
    grid: np.ndarray,
    marker_idx: int,
    excluded_cells: set[tuple[int, int]],
    min_dist: int = 2,
):
    markers = sorted((int(r), int(c)) for r, c in np.argwhere(grid == MARKER))
    target = markers[marker_idx]
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG or (r, c) in excluded_cells:
                continue
            md = abs(r - target[0]) + abs(c - target[1])
            if 2 <= md <= max(h, w):
                candidates.append((r, c))
    random.shuffle(candidates)
    for cell in candidates:
        grid[cell] = TOKEN
        analysis = _analyze(grid)
        info = analysis["maroon_info"][cell]
        if info["assigned"] == marker_idx and info["dist"] is not None and info["dist"] >= min_dist:
            return True
        grid[cell] = BG
    return False


def _try_add_tie_token(grid: np.ndarray, left_idx: int, right_idx: int, excluded_cells: set[tuple[int, int]]):
    markers = sorted((int(r), int(c)) for r, c in np.argwhere(grid == MARKER))
    left = markers[left_idx]
    right = markers[right_idx]
    if left[0] != right[0]:
        return False
    mid = (left[1] + right[1]) // 2
    cell = (left[0], mid)
    if (left[1] + right[1]) % 2 != 0:
        return False
    if cell in excluded_cells or grid[cell] != BG:
        return False
    grid[cell] = TOKEN
    analysis = _analyze(grid)
    info = analysis["maroon_info"][cell]
    if info["assigned"] != left_idx or not info["best_tied"]:
        grid[cell] = BG
        return False
    return True


def _final_checks(grid: np.ndarray, tie_branch: bool, unreachable_branch: bool, overflow_branch: bool) -> bool:
    analysis = _analyze(grid)
    markers = analysis["markers"]
    if len(markers) not in (2, 3):
        return False
    if not _pairwise_far_enough(markers):
        return False

    maroon_count = len(analysis["maroons"])
    if not (6 <= maroon_count <= 12):
        return False

    capacities = analysis["capacities"]
    if sum(capacities) >= maroon_count:
        return False
    if max(capacities) <= 0:
        return False

    for idx, marker in enumerate(markers):
        reachable = [item for item in analysis["assigned"][idx]]
        if len(reachable) < 2:
            return False
        slots = analysis["slots_by_marker"][idx]
        if slots != sorted(slots):
            return False
        if any(analysis["halo_grid"][cell] != HALO for cell in slots):
            return False
        _ = marker

    for idx, items in enumerate(analysis["assigned"]):
        items = sorted(items)
        cap = capacities[idx]
        if len(items) > cap and items[cap - 1][0] >= items[cap][0]:
            return False

    if tie_branch:
        if not any(info["best_tied"] for info in analysis["maroon_info"].values() if info["assigned"] is not None):
            return False

    unreachable_count = sum(1 for info in analysis["maroon_info"].values() if info["assigned"] is None)
    if unreachable_branch and unreachable_count == 0:
        return False
    if overflow_branch and not any(len(items) > cap for items, cap in zip(analysis["assigned"], capacities)):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out, analysis["output"]):
        return False

    moved = analysis["moved_sources"]
    if not moved:
        return False
    for cell in moved:
        if out[cell] != BG:
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    tie_branch = random.random() < 0.35
    edge_branch = (not tie_branch) and random.random() < 0.35
    unreachable_branch = random.random() < 0.55
    overflow_branch = random.random() < 0.70
    n_markers = 2 if tie_branch else random.choice([2, 3])

    markers = _pick_markers(h, w, n_markers, tie_branch, edge_branch)
    if markers is None:
        return None

    shape = (h, w)
    plans = []
    for idx, marker in enumerate(markers):
        forced_dir = _forced_dir_for_edge(marker, shape)
        allowed_sides = None
        if tie_branch and idx == 0:
            forced_dir = "E"
            allowed_sides = ("N", "S")
        elif tie_branch and idx == 1:
            forced_dir = "W"
            allowed_sides = ("N", "S")
        plan = _local_pattern(marker, shape, forced_dir=forced_dir, allowed_wall_sides=allowed_sides)
        if plan is None:
            return None
        plans.append(plan)

    local_maroons = {cell for plan in plans for cell in plan["local_maroons"]}
    forbidden = set(markers) | local_maroons
    wall_segments = []
    for marker, plan in zip(markers, plans):
        segment = _line_segment(plan["wall_base"], shape, forbidden)
        if segment is None:
            return None
        if any(cell in local_maroons or cell in markers for cell in segment):
            return None
        wall_segments.append(segment)
        forbidden.update(segment)
        _ = marker

    grid = np.zeros(shape, dtype=int)
    for marker in markers:
        grid[marker] = MARKER
    for cell in local_maroons:
        grid[cell] = TOKEN
    for segment in wall_segments:
        for cell in segment:
            if grid[cell] != BG:
                return None
            grid[cell] = WALL

    blocked = _marker_regions(markers, shape) | set().union(*[set(seg) for seg in wall_segments])
    if unreachable_branch:
        ok, _token = _place_unreachable_box(grid, blocked)
        if not ok:
            return None
        blocked = blocked | {tuple(cell) for cell in np.argwhere(grid == WALL)} | {tuple(cell) for cell in np.argwhere(grid == TOKEN)}

    analysis = _analyze(grid)
    capacities = analysis["capacities"]
    if sum(capacities) > 10:
        return None

    current_count = len(analysis["maroons"])
    if current_count > 12:
        return None

    if tie_branch:
        excluded = _marker_regions(markers, shape) | {tuple(cell) for cell in np.argwhere(grid == WALL)}
        if not _try_add_tie_token(grid, 0, 1, excluded):
            return None

    analysis = _analyze(grid)
    current_count = len(analysis["maroons"])
    assigned_counts = [len(items) for items in analysis["assigned"]]
    capacities = analysis["capacities"]

    focus = min(range(n_markers), key=lambda idx: capacities[idx])
    overflow_need = 0
    if overflow_branch:
        overflow_need = max(0, capacities[focus] - assigned_counts[focus] + 1)

    minimum_total = max(6, sum(capacities) + 1, current_count + overflow_need)
    if minimum_total > 12:
        return None
    target_total = random.randint(minimum_total, 12)

    excluded = _marker_regions(markers, shape) | {tuple(cell) for cell in np.argwhere(grid == WALL)}
    while len(_analyze(grid)["maroons"]) < current_count + overflow_need:
        if not _try_add_token_for_marker(grid, focus, excluded):
            return None

    while len(_analyze(grid)["maroons"]) < target_total:
        order = list(range(n_markers))
        random.shuffle(order)
        added = False
        for idx in order:
            if _try_add_token_for_marker(grid, idx, excluded):
                added = True
                break
        if not added:
            return None

    if not _final_checks(grid, tie_branch, unreachable_branch, overflow_branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
