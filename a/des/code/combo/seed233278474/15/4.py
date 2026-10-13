from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
GEN_RETRIES = 400
MAX_SIDE = 30


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _find_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rooms: list[dict] = []
    for coords in _components(g != GREY):
        if _touches_edge(coords, g.shape):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if np.any(g[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(g[r0 - 1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(g[r1 + 1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(g[r0 : r1 + 1, c0 - 1] == GREY):
            continue
        if not np.all(g[r0 : r1 + 1, c1 + 1] == GREY):
            continue
        seeds = [(int(r), int(c)) for r, c in coords if int(g[r, c]) not in (BLACK, GREY)]
        if len(seeds) > 1:
            continue
        seed = seeds[0] if seeds else None
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
                "seed": seed,
                "seed_color": None if seed is None else int(g[seed]),
            }
        )
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _room_distances(room: dict) -> dict[str, int]:
    (r0, c0, r1, c1) = room["bbox"]
    sr, sc = room["seed"]
    return {
        "up": sr - r0,
        "down": r1 - sr,
        "left": sc - c0,
        "right": c1 - sc,
    }


def _trace_room_ray(room: dict, shape: tuple[int, int]) -> dict | None:
    if room["seed"] is None:
        return None
    distances = _room_distances(room)
    if len(set(distances.values())) < 4:
        return None
    nearest = min(distances, key=distances.get)
    direction = OPPOSITE[nearest]
    span = room["height"] if direction in ("up", "down") else room["width"]
    dr, dc = DIRS[direction]
    sr, sc = room["seed"]
    path: list[tuple[int, int]] = []
    for step in range(1, span + 1):
        rr = sr + dr * step
        cc = sc + dc * step
        if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
            break
        path.append((rr, cc))
    return {
        "seed": (sr, sc),
        "color": int(room["seed_color"]),
        "direction": direction,
        "span": span,
        "path": path,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit a colored ray away from the nearest wall in each seeded rectangular room.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rooms = _find_rooms(g)
    seed_cells = {room["seed"] for room in rooms if room["seed"] is not None}
    for room in rooms:
        ray = _trace_room_ray(room, g.shape)
        if ray is None:
            continue
        color = int(ray["color"])
        for rr, cc in ray["path"]:
            if (rr, cc) in seed_cells:
                continue
            out[rr, cc] = color
    return out


def _build_grid(rows: int, cols: int, row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, list[dict]]:
    structure_h = sum(row_sizes) + rows + 1
    structure_w = sum(col_sizes) + cols + 1
    extra_h = MAX_SIDE - structure_h
    extra_w = MAX_SIDE - structure_w
    if extra_h < 2 or extra_w < 2:
        raise ValueError("layout too large")

    top = random.randint(1, extra_h - 1)
    bottom = random.randint(1, extra_h - top)
    left = random.randint(1, extra_w - 1)
    right = random.randint(1, extra_w - left)
    height = structure_h + top + bottom
    width = structure_w + left + right

    grid = np.full((height, width), BLACK, dtype=int)

    wall_rows = [top]
    cur = top
    for size in row_sizes:
        cur += size + 1
        wall_rows.append(cur)

    wall_cols = [left]
    cur = left
    for size in col_sizes:
        cur += size + 1
        wall_cols.append(cur)

    for r in wall_rows:
        grid[r, wall_cols[0] : wall_cols[-1] + 1] = GREY
    for c in wall_cols:
        grid[wall_rows[0] : wall_rows[-1] + 1, c] = GREY

    rooms: list[dict] = []
    for i in range(rows):
        for j in range(cols):
            r0 = wall_rows[i] + 1
            r1 = wall_rows[i + 1] - 1
            c0 = wall_cols[j] + 1
            c1 = wall_cols[j + 1] - 1
            rooms.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )
    return grid, rooms


def _candidate_rays(room: dict, shape: tuple[int, int]) -> dict[str, list[dict]]:
    r0, c0, r1, c1 = room["bbox"]
    out = {name: [] for name in DIRS}
    for sr in range(r0, r1 + 1):
        for sc in range(c0, c1 + 1):
            distances = {
                "up": sr - r0,
                "down": r1 - sr,
                "left": sc - c0,
                "right": c1 - sc,
            }
            if min(distances.values()) < 1:
                continue
            if len(set(distances.values())) < 4:
                continue
            nearest = min(distances, key=distances.get)
            direction = OPPOSITE[nearest]
            span = room["height"] if direction in ("up", "down") else room["width"]
            dr, dc = DIRS[direction]
            path: list[tuple[int, int]] = []
            ok = True
            for step in range(1, span + 1):
                rr = sr + dr * step
                cc = sc + dc * step
                if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
                    ok = False
                    break
                path.append((rr, cc))
            if ok:
                out[direction].append({"seed": (sr, sc), "path": path, "span": span})
    return out


def _search_seed_plans(
    rooms: list[dict],
    directions: list[str],
) -> list[dict] | None:
    demand_order = list(range(len(directions)))
    demand_order.sort(
        key=lambda idx: sum(1 for room in rooms if room["candidates"][directions[idx]]),
    )

    def backtrack(step: int, blocked: set[tuple[int, int]], used_rooms: set[int], plans: list[dict]) -> list[dict] | None:
        if step == len(demand_order):
            return plans.copy()

        direction = directions[demand_order[step]]
        room_indices = [idx for idx in range(len(rooms)) if idx not in used_rooms and rooms[idx]["candidates"][direction]]
        random.shuffle(room_indices)
        for room_idx in room_indices:
            choices = rooms[room_idx]["candidates"][direction][:]
            random.shuffle(choices)
            for choice in choices:
                footprint = set(choice["path"])
                footprint.add(choice["seed"])
                if footprint & blocked:
                    continue
                plans.append(
                    {
                        "room_idx": room_idx,
                        "direction": direction,
                        "seed": choice["seed"],
                        "path": choice["path"],
                        "span": choice["span"],
                    }
                )
                answer = backtrack(step + 1, blocked | footprint, used_rooms | {room_idx}, plans)
                if answer is not None:
                    return answer
                plans.pop()
        return None

    return backtrack(0, set(), set(), [])


def _expected_output(grid: np.ndarray, plans: list[dict]) -> np.ndarray:
    out = grid.copy()
    seeds = {plan["seed"] for plan in plans}
    for plan in plans:
        color = int(plan["color"])
        for rr, cc in plan["path"]:
            if (rr, cc) in seeds:
                continue
            out[rr, cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(2, 3)
    cols = random.randint(2, 3)
    row_sizes = [random.choice((6, 7)) for _ in range(rows)]
    col_sizes = [random.choice((6, 7)) for _ in range(cols)]

    try:
        grid, rooms = _build_grid(rows, cols, row_sizes, col_sizes)
    except ValueError:
        return None

    for room in rooms:
        room["candidates"] = _candidate_rays(room, grid.shape)

    available_dirs = [name for name in DIRS if any(room["candidates"][name] for room in rooms)]
    if not available_dirs:
        return None

    max_seed_count = min(4, len(rooms) - 1)
    if max_seed_count < 1:
        return None
    seed_count = random.randint(1, max_seed_count)

    shuffled_dirs = list(DIRS.keys())
    random.shuffle(shuffled_dirs)
    directions = shuffled_dirs[: min(seed_count, len(available_dirs))]
    directions = [d for d in directions if d in available_dirs]
    while len(directions) < seed_count:
        directions.append(random.choice(available_dirs))
    random.shuffle(directions)

    plans = _search_seed_plans(rooms, directions)
    if plans is None:
        return None

    colors = random.sample(SEED_COLORS, len(plans))
    for color, plan in zip(colors, plans):
        sr, sc = plan["seed"]
        grid[sr, sc] = color
        plan["color"] = color

    found = _find_rooms(grid)
    if len(found) != len(rooms):
        return None
    if sum(room["seed"] is not None for room in found) != len(plans):
        return None

    expected = _expected_output(grid, plans)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
