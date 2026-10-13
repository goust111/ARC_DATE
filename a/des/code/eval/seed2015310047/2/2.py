import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _path_neighbors(path_mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = path_mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr = int(r + dr)
        nc = int(c + dc)
        if _in_bounds(h, w, nr, nc) and bool(path_mask[nr, nc]):
            out.append((nr, nc))
    return out


def _path_degrees(path_mask: np.ndarray) -> np.ndarray:
    degrees = np.zeros(path_mask.shape, dtype=int)
    for r, c in np.argwhere(path_mask):
        degrees[int(r), int(c)] = len(_path_neighbors(path_mask, int(r), int(c)))
    return degrees


def _find_path_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.reshape(-1) if int(v) != bg)
    if not counts:
        raise ValueError("grid must contain a maze color")

    max_count = max(counts.values())
    candidates = [int(color) for color, count in counts.items() if count == max_count]
    if len(candidates) == 1:
        return candidates[0]

    best_color = candidates[0]
    best_key = None
    for color in candidates:
        mask = grid == int(color)
        degrees = _path_degrees(mask)
        key = (int(np.sum(degrees >= 3)), int(np.sum(degrees == 1)), int(mask.sum()))
        if best_key is None or key > best_key:
            best_key = key
            best_color = int(color)
    return best_color


def _label_regions(grid: np.ndarray, bg: int, path_color: int) -> tuple[np.ndarray, dict[int, int | None]]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    owners: dict[int, int | None] = {}
    region_id = 0

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == path_color or int(labels[r, c]) != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = int(region_id)
            seed_colors: set[int] = set()

            while q:
                cr, cc = q.popleft()
                value = int(grid[cr, cc])
                if value not in (bg, path_color):
                    seed_colors.add(value)
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not _in_bounds(h, w, nr, nc):
                        continue
                    if int(grid[nr, nc]) == path_color or int(labels[nr, nc]) != -1:
                        continue
                    labels[nr, nc] = int(region_id)
                    q.append((nr, nc))

            owners[int(region_id)] = next(iter(seed_colors)) if len(seed_colors) == 1 else None
            region_id += 1

    return labels, owners


def _dead_end_depth(path_mask: np.ndarray, degrees: np.ndarray, start: tuple[int, int]) -> int:
    r, c = map(int, start)
    neighbors = _path_neighbors(path_mask, r, c)
    if len(neighbors) != 1:
        return 0

    depth = 1
    prev = (r, c)
    cur = neighbors[0]
    seen = {prev}

    while True:
        cr, cc = cur
        if cur in seen:
            break
        if int(degrees[cr, cc]) != 2:
            break
        seen.add(cur)
        next_steps = [p for p in _path_neighbors(path_mask, cr, cc) if p != prev]
        if len(next_steps) != 1:
            break
        depth += 1
        prev, cur = cur, next_steps[0]

    return int(depth)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _get_bg(g)
    path_color = _find_path_color(g, bg)
    path_mask = g == int(path_color)
    degrees = _path_degrees(path_mask)
    labels, owners = _label_regions(g, bg, path_color)

    dead_ends = []
    for r, c in sorted(map(tuple, np.argwhere(path_mask))):
        r = int(r)
        c = int(c)
        if int(degrees[r, c]) != 1:
            continue
        neighbors = _path_neighbors(path_mask, r, c)
        if len(neighbors) != 1:
            continue
        nr, nc = neighbors[0]
        dr = int(r - nr)
        dc = int(c - nc)
        depth = _dead_end_depth(path_mask, degrees, (r, c))

        ahead_r = int(r + dr)
        ahead_c = int(c + dc)
        owner = None
        region_id = -1
        if _in_bounds(g.shape[0], g.shape[1], ahead_r, ahead_c) and not bool(path_mask[ahead_r, ahead_c]):
            region_id = int(labels[ahead_r, ahead_c])
            owner = owners.get(region_id)

        dead_ends.append(
            {
                "terminal": (r, c),
                "direction": (dr, dc),
                "depth": int(depth),
                "ahead": (ahead_r, ahead_c),
                "region_id": int(region_id),
                "owner": None if owner is None else int(owner),
            }
        )

    return {
        "bg": int(bg),
        "path_color": int(path_color),
        "path_mask": path_mask,
        "degrees": degrees,
        "labels": labels,
        "owners": owners,
        "dead_ends": dead_ends,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored branches from maze dead ends.

    1. The dominant color is the background.
    2. The largest non-background color is the one-pixel-wide orthogonal maze.
    3. Non-maze cells are partitioned into regions; a region's unique seed color,
       if present, determines that region's branch color.
    4. Each dead end grows straight outward, away from its only path neighbor,
       for a length equal to its dead-end depth, painting only background cells.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    bg = int(info["bg"])
    out = g.copy()

    for dead_end in info["dead_ends"]:
        owner = dead_end["owner"]
        if owner is None:
            continue
        r, c = dead_end["terminal"]
        dr, dc = dead_end["direction"]
        depth = int(dead_end["depth"])
        for step in range(1, depth + 1):
            nr = int(r + dr * step)
            nc = int(c + dc * step)
            if not _in_bounds(g.shape[0], g.shape[1], nr, nc):
                break
            if int(out[nr, nc]) != bg:
                break
            out[nr, nc] = int(owner)

    return out


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
ROOM_H_RANGE = (6, 10)
ROOM_W_RANGE = (6, 10)
ROOM_GAP = 2
ROOM_BUILD_TRIES = 60
ROOM_PLACE_TRIES = 120
GENERATION_ATTEMPTS = 400


def _room_border(h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(w):
        cells.add((0, c))
        cells.add((h - 1, c))
    for r in range(1, h - 1):
        cells.add((r, 0))
        cells.add((r, w - 1))
    return cells


def _spur_footprint(h: int, w: int, side: str, offset: int, depth: int) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    if side == "top":
        branchpoint = (0, offset)
        path = [(r, offset) for r in range(1, depth + 1)]
        growth = [(r, offset) for r in range(depth + 1, 2 * depth + 1)]
    elif side == "bottom":
        branchpoint = (h - 1, offset)
        path = [(h - 1 - r, offset) for r in range(1, depth + 1)]
        growth = [(h - 1 - r, offset) for r in range(depth + 1, 2 * depth + 1)]
    elif side == "left":
        branchpoint = (offset, 0)
        path = [(offset, c) for c in range(1, depth + 1)]
        growth = [(offset, c) for c in range(depth + 1, 2 * depth + 1)]
    elif side == "right":
        branchpoint = (offset, w - 1)
        path = [(offset, w - 1 - c) for c in range(1, depth + 1)]
        growth = [(offset, w - 1 - c) for c in range(depth + 1, 2 * depth + 1)]
    else:
        raise ValueError(f"bad side: {side}")
    return branchpoint, path, growth


def _zone_touches(zone: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in zone:
        if (r, c) in occupied:
            return True
        for dr, dc in DIR4:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _seed_candidates(h: int, w: int, reserved: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    sizes = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]

    for sh, sw in sizes:
        if sh > h - 2 or sw > w - 2:
            continue

        for c0 in range(1, w - sw):
            cells = [(r, c) for r in range(1, 1 + sh) for c in range(c0, c0 + sw)]
            if not any(cell in reserved for cell in cells):
                candidates.append(cells)
        for c0 in range(1, w - sw):
            cells = [(r, c) for r in range(h - 1 - sh, h - 1) for c in range(c0, c0 + sw)]
            if not any(cell in reserved for cell in cells):
                candidates.append(cells)
        for r0 in range(1, h - sh):
            cells = [(r, c) for r in range(r0, r0 + sh) for c in range(1, 1 + sw)]
            if not any(cell in reserved for cell in cells):
                candidates.append(cells)
        for r0 in range(1, h - sh):
            cells = [(r, c) for r in range(r0, r0 + sh) for c in range(w - 1 - sw, w - 1)]
            if not any(cell in reserved for cell in cells):
                candidates.append(cells)

    return candidates


def _build_room_spec(h: int, w: int, rng: random.Random) -> dict | None:
    border = _room_border(h, w)

    for _ in range(ROOM_BUILD_TRIES):
        spur_count = 2 if min(h, w) >= 8 and rng.random() < 0.5 else 1
        spurs = []
        reserved: set[tuple[int, int]] = set()
        used_sides: set[str] = set()
        used_branchpoints: set[tuple[int, int]] = set()
        spur_path_cells: set[tuple[int, int]] = set()

        ok = True
        for _spur_idx in range(spur_count):
            candidates = []
            side_order = list(SIDES)
            rng.shuffle(side_order)
            for side in side_order:
                if side in used_sides:
                    continue
                inner_axis = h - 2 if side in ("top", "bottom") else w - 2
                along_len = w - 2 if side in ("top", "bottom") else h - 2
                max_depth = inner_axis // 2
                if max_depth < 1:
                    continue
                depths = list(range(1, max_depth + 1))
                offsets = list(range(1, along_len + 1))
                rng.shuffle(depths)
                rng.shuffle(offsets)
                for depth in depths:
                    for offset in offsets:
                        branchpoint, path_cells, growth_cells = _spur_footprint(h, w, side, offset, depth)
                        zone = set(path_cells) | set(growth_cells)
                        if branchpoint in used_branchpoints:
                            continue
                        if _zone_touches(zone, reserved):
                            continue
                        if _zone_touches(set(path_cells), spur_path_cells):
                            continue
                        candidates.append(
                            {
                                "side": side,
                                "depth": int(depth),
                                "branchpoint": branchpoint,
                                "path_cells": path_cells,
                                "growth_cells": growth_cells,
                            }
                        )
            if not candidates:
                ok = False
                break
            choice = candidates[rng.randrange(len(candidates))]
            spurs.append(choice)
            used_sides.add(str(choice["side"]))
            used_branchpoints.add(tuple(choice["branchpoint"]))
            reserved.update(map(tuple, choice["path_cells"]))
            reserved.update(map(tuple, choice["growth_cells"]))
            spur_path_cells.update(map(tuple, choice["path_cells"]))

        if not ok or not spurs:
            continue

        seeds = _seed_candidates(h, w, reserved)
        if not seeds:
            continue
        seed_cells = seeds[rng.randrange(len(seeds))]

        local_path = set(border)
        for spur in spurs:
            local_path.update(map(tuple, spur["path_cells"]))

        return {
            "h": int(h),
            "w": int(w),
            "path_cells": sorted(local_path),
            "seed_cells": sorted(map(tuple, seed_cells)),
            "spur_count": int(len(spurs)),
        }

    return None


def _boxes_clear(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in placed:
        if not (r1 + ROOM_GAP < or0 or or1 + ROOM_GAP < r0 or c1 + ROOM_GAP < oc0 or oc1 + ROOM_GAP < c0):
            return False
    return True


def _draw_room(grid: np.ndarray, top: int, left: int, spec: dict, path_color: int, seed_color: int) -> None:
    for r, c in spec["path_cells"]:
        grid[int(top + r), int(left + c)] = int(path_color)
    for r, c in spec["seed_cells"]:
        grid[int(top + r), int(left + c)] = int(seed_color)


def _choose_room_count(area: int, rng: random.Random) -> int:
    if area < 440:
        return rng.choice([2, 3])
    if area < 620:
        return rng.choice([2, 3, 4])
    return rng.choice([3, 4, 5])


def _free_growth_cells(grid: np.ndarray, bg: int, dead_end: dict) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    r, c = dead_end["terminal"]
    dr, dc = dead_end["direction"]
    depth = int(dead_end["depth"])
    cells: list[tuple[int, int]] = []
    for step in range(1, depth + 1):
        nr = int(r + dr * step)
        nc = int(c + dc * step)
        if not _in_bounds(h, w, nr, nc):
            return None
        if int(grid[nr, nc]) != bg:
            return None
        cells.append((nr, nc))
    return cells


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg, path_color = rng.sample(PALETTE, 2)
    grid = np.full((h, w), int(bg), dtype=int)

    room_count = _choose_room_count(h * w, rng)
    seed_colors = [color for color in PALETTE if color not in (bg, path_color)]
    rng.shuffle(seed_colors)
    seed_colors = seed_colors[:room_count]
    if len(seed_colors) != room_count:
        return None

    room_specs = []
    for _ in range(room_count):
        hh = rng.randint(*ROOM_H_RANGE)
        ww = rng.randint(*ROOM_W_RANGE)
        spec = _build_room_spec(hh, ww, rng)
        if spec is None:
            return None
        room_specs.append(spec)

    room_specs.sort(key=lambda spec: int(spec["h"]) * int(spec["w"]), reverse=True)

    placed_boxes: list[tuple[int, int, int, int]] = []
    expected_dead_ends = 0
    for spec, seed_color in zip(room_specs, seed_colors):
        placed = False
        for _ in range(ROOM_PLACE_TRIES):
            top = rng.randint(1, h - int(spec["h"]) - 1)
            left = rng.randint(1, w - int(spec["w"]) - 1)
            box = (top, left, top + int(spec["h"]) - 1, left + int(spec["w"]) - 1)
            if not _boxes_clear(box, placed_boxes):
                continue
            _draw_room(grid, top, left, spec, path_color, seed_color)
            placed_boxes.append(box)
            expected_dead_ends += int(spec["spur_count"])
            placed = True
            break
        if not placed:
            return None

    counts = Counter(map(int, grid.reshape(-1)))
    non_bg_counts = [count for color, count in counts.items() if int(color) != int(bg)]
    if not non_bg_counts or counts[int(bg)] <= max(non_bg_counts):
        return None
    if counts[int(path_color)] != max(non_bg_counts):
        return None

    try:
        info = _analyze_grid(grid)
    except Exception:
        return None

    dead_ends = info["dead_ends"]
    if len(dead_ends) != expected_dead_ends or len(dead_ends) < 2:
        return None

    planned: set[tuple[int, int]] = set()
    for dead_end in dead_ends:
        if dead_end["owner"] is None:
            return None
        cells = _free_growth_cells(grid, int(info["bg"]), dead_end)
        if cells is None:
            return None
        if any(cell in planned for cell in cells):
            return None
        planned.update(cells)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out != grid)) != len(planned):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
