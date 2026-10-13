import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
COLOR_NAME_RANK = {
    1: 1,  # blue
    8: 2,  # cyan
    3: 3,  # green
    6: 4,  # magenta
    7: 5,  # orange
    2: 6,  # red
    4: 7,  # yellow
    9: 8,  # brown
}

GEN_RETRIES = 400


def _mode(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=int).ravel()
    return int(Counter(vals.tolist()).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _seal_walls(grid: np.ndarray, bg: int, wall: int) -> np.ndarray:
    h, w = grid.shape
    sealed = grid == wall
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != bg:
                continue
            horiz = bool(sealed[r, c - 1] and sealed[r, c + 1])
            vert = bool(sealed[r - 1, c] and sealed[r + 1, c])
            if horiz or vert:
                sealed[r, c] = True
    return sealed


def _room_data(grid: np.ndarray, bg: int, wall: int):
    sealed = _seal_walls(grid, bg, wall)
    mask = ~sealed
    label = -np.ones(grid.shape, dtype=int)
    rooms = []
    for idx, pts in enumerate(_components(mask)):
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        for r, c in pts:
            label[r, c] = idx
        rooms.append(
            {
                "bbox": (r0, r1, c0, c1),
                "cells": pts,
                "set": set(pts),
            }
        )
    return rooms, label


def _direction_from_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, r1, c0, c1 = bbox
    left = c - c0
    right = c1 - c
    up = r - r0
    down = r1 - r
    if min(left, right) <= min(up, down):
        return (0, -1) if left <= right else (0, 1)
    return (-1, 0) if up <= down else (1, 0)


def _vine_path(grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int], wall: int):
    h, w = grid.shape
    r, c = start
    dr, dc = direction
    t = 0
    out = []
    while True:
        r += dr
        c += dc
        t += 1
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(grid[r, c]) == wall:
            break
        out.append((r, c, t))
    return out


def _apply_vines(grid: np.ndarray, bg: int, wall: int) -> np.ndarray:
    rooms, labels = _room_data(grid, bg, wall)
    seeds = [tuple(map(int, rc)) for rc in np.argwhere((grid != bg) & (grid != wall))]
    best: dict[tuple[int, int], tuple[int, int, int]] = {}
    for sr, sc in seeds:
        room_idx = int(labels[sr, sc])
        if room_idx < 0:
            continue
        color = int(grid[sr, sc])
        direction = _direction_from_bbox(sr, sc, rooms[room_idx]["bbox"])
        for r, c, t in _vine_path(grid, (sr, sc), direction, wall):
            if int(grid[r, c]) != bg:
                continue
            rank = COLOR_NAME_RANK.get(color, 999)
            prev = best.get((r, c))
            if prev is None or (t, rank, color) < prev:
                best[(r, c)] = (t, rank, color)
    out = np.array(grid, copy=True)
    for (r, c), (_, _, color) in best.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow straight colored vines across rectangular rooms by reading each seed's nearest wall and using wall gaps as doorways.

    Input structure:
    - The background is black.
    - One-pixel-thick grey walls partition the canvas into enclosed rectangular rooms, with some wall cells removed as one-pixel gaps.
    - Some rooms contain a single colored seed pixel; seed colors are non-black and non-grey, and each seed color is unique.
    - Gap cells are black in the input and connect adjacent rooms through the shared wall.

    Rule:
    - For each seed, identify its containing room and compare its distances to the room's left, right, top, and bottom sides.
    - The seed grows in the direction of the nearest side; if a horizontal and vertical side tie, choose the horizontal direction.
    - Extend a straight vine of the seed's color through black cells until a grey wall blocks it; when the blocking wall position is a gap, the vine passes through that black gap cell and continues in the same direction inside the next room.
    - Seeds act independently and do not repaint grey walls or other pre-existing colored seeds.

    Key insight:
    - The seed's offset inside its rectangular room encodes the growth direction, while the wall gaps determine whether that straight trajectory stops or propagates into later rooms.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    bg = _mode(g)
    others = g[g != bg]
    if others.size == 0:
        return g.copy()
    wall = _mode(others)
    return _apply_vines(g, bg, wall)


def _room_boxes(row_heights: list[int], col_widths: list[int]):
    row_boxes = []
    r = 1
    for h in row_heights:
        row_boxes.append((r, r + h - 1))
        r += h + 1
    col_boxes = []
    c = 1
    for w in col_widths:
        col_boxes.append((c, c + w - 1))
        c += w + 1
    return row_boxes, col_boxes


def _path_rooms(n_room_rows: int, n_room_cols: int) -> list[tuple[int, int]]:
    rooms = []
    for rr in range(n_room_rows):
        cols = range(n_room_cols) if rr % 2 == 0 else range(n_room_cols - 1, -1, -1)
        for cc in cols:
            rooms.append((rr, cc))
    return rooms


def _room_doors(
    row_boxes: list[tuple[int, int]],
    col_boxes: list[tuple[int, int]],
    rng: random.Random,
):
    n_room_rows = len(row_boxes)
    n_room_cols = len(col_boxes)
    horiz_rows = []
    for r0, r1 in row_boxes:
        if r1 - r0 + 1 >= 3:
            horiz_rows.append(rng.randint(r0 + 1, r1 - 1))
        else:
            horiz_rows.append(rng.randint(r0, r1))
    vert_cols = []
    for c0, c1 in col_boxes:
        if c1 - c0 + 1 >= 3:
            vert_cols.append(rng.randint(c0 + 1, c1 - 1))
        else:
            vert_cols.append(rng.randint(c0, c1))

    doors = {(rr, cc): {} for rr in range(n_room_rows) for cc in range(n_room_cols)}
    gaps: list[tuple[int, int]] = []
    path = _path_rooms(n_room_rows, n_room_cols)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r1 == r2:
            rr = r1
            left = min(c1, c2)
            wall_c = col_boxes[left][1] + 1
            gap_r = horiz_rows[rr]
            gaps.append((gap_r, wall_c))
            doors[(rr, left)]["right"] = gap_r
            doors[(rr, left + 1)]["left"] = gap_r
        else:
            cc = c1
            top = min(r1, r2)
            wall_r = row_boxes[top][1] + 1
            gap_c = vert_cols[cc]
            gaps.append((wall_r, gap_c))
            doors[(top, cc)]["down"] = gap_c
            doors[(top + 1, cc)]["up"] = gap_c
    return doors, gaps


def _local_direction(local_r: int, local_c: int, h: int, w: int) -> str:
    left = local_c
    right = w - 1 - local_c
    up = local_r
    down = h - 1 - local_r
    if min(left, right) <= min(up, down):
        return "left" if left <= right else "right"
    return "up" if up <= down else "down"


def _is_horizontal_tie(local_r: int, local_c: int, h: int, w: int, direction: str) -> bool:
    if direction not in ("left", "right"):
        return False
    left = local_c
    right = w - 1 - local_c
    up = local_r
    down = h - 1 - local_r
    return min(left, right) == min(up, down)


def _seed_candidates(
    bbox: tuple[int, int, int, int],
    door_map: dict[str, int],
    direction: str,
    want_pass: bool,
    want_tie: bool,
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    out = []
    target_coord = door_map.get(direction)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            lr, lc = r - r0, c - c0
            if _local_direction(lr, lc, h, w) != direction:
                continue
            if want_tie and not _is_horizontal_tie(lr, lc, h, w, direction):
                continue
            if direction in ("left", "right"):
                aligned = target_coord is not None and r == target_coord
            else:
                aligned = target_coord is not None and c == target_coord
            if want_pass:
                if not aligned:
                    continue
            elif target_coord is not None and aligned:
                continue
            out.append((r, c))
    return out


def _scenario_pool(room_infos, doors):
    scenarios = []
    for key, info in room_infos.items():
        bbox = info["bbox"]
        for direction in ("left", "right", "up", "down"):
            for want_pass in (False, True):
                tie_options = (False, True) if direction in ("left", "right") else (False,)
                for want_tie in tie_options:
                    cells = _seed_candidates(bbox, doors[key], direction, want_pass, want_tie)
                    if cells:
                        scenarios.append(
                            {
                                "room": key,
                                "direction": direction,
                                "want_pass": want_pass,
                                "want_tie": want_tie,
                                "cells": cells,
                            }
                        )
    return scenarios


DIR_MAP = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}


def _option_pool(base_grid, scenarios, gaps):
    gap_set = set(gaps)
    options = []
    for sc in scenarios:
        dr, dc = DIR_MAP[sc["direction"]]
        for start in sc["cells"]:
            path = [(r, c) for r, c, _ in _vine_path(base_grid, start, (dr, dc), WALL) if int(base_grid[r, c]) == BG]
            if not path:
                continue
            options.append(
                {
                    "room": sc["room"],
                    "direction": sc["direction"],
                    "want_tie": sc["want_tie"],
                    "start": start,
                    "path": path,
                    "path_set": set(path),
                    "crosses_gap": any(cell in gap_set for cell in path),
                }
            )
    return options


def _disjoint_option(option, chosen):
    for prev in chosen:
        if option["room"] == prev["room"]:
            return False
        if option["start"] in prev["path_set"] or prev["start"] in option["path_set"]:
            return False
        if option["path_set"] & prev["path_set"]:
            return False
    return True


def _choose_options(options, num_rooms: int, rng: random.Random):
    if not options:
        return []
    seed_count = rng.randint(2, min(4, num_rooms))
    needs = [
        lambda op: op["crosses_gap"],
        lambda op: not op["crosses_gap"],
        lambda op: op["direction"] in ("left", "right"),
        lambda op: op["direction"] in ("up", "down"),
    ]
    if rng.random() < 0.5:
        needs.append(lambda op: op["want_tie"])

    chosen = []
    for pred in needs:
        candidates = [op for op in options if pred(op) and _disjoint_option(op, chosen)]
        if not candidates:
            continue
        chosen.append(rng.choice(candidates))

    remaining = options[:]
    rng.shuffle(remaining)
    for op in remaining:
        if len(chosen) >= seed_count:
            break
        if _disjoint_option(op, chosen):
            chosen.append(op)
    return chosen


def _trace_cells(
    grid: np.ndarray,
    seeds: list[tuple[int, int]],
    bg: int,
    wall: int,
):
    rooms, labels = _room_data(grid, bg, wall)
    traced = {}
    for sr, sc in seeds:
        direction = _direction_from_bbox(sr, sc, rooms[int(labels[sr, sc])]["bbox"])
        color = int(grid[sr, sc])
        cells = [(r, c) for r, c, _ in _vine_path(grid, (sr, sc), direction, wall) if int(grid[r, c]) == bg]
        traced[color] = cells
    return traced


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    n_room_rows = rng.randint(2, 4)
    n_room_cols = rng.randint(2, 4)
    row_heights = [rng.randint(3, 6) for _ in range(n_room_rows)]
    col_widths = [rng.randint(3, 6) for _ in range(n_room_cols)]
    h = sum(row_heights) + n_room_rows + 1
    w = sum(col_widths) + n_room_cols + 1
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    row_boxes, col_boxes = _room_boxes(row_heights, col_widths)

    wall_rows = [0] + [r1 + 1 for _, r1 in row_boxes]
    wall_cols = [0] + [c1 + 1 for _, c1 in col_boxes]
    for rr in wall_rows:
        grid[rr, :] = WALL
    for cc in wall_cols:
        grid[:, cc] = WALL

    doors, gaps = _room_doors(row_boxes, col_boxes, rng)
    for gr, gc in gaps:
        grid[gr, gc] = BG

    room_infos = {}
    for rr, (r0, r1) in enumerate(row_boxes):
        for cc, (c0, c1) in enumerate(col_boxes):
            room_infos[(rr, cc)] = {"bbox": (r0, r1, c0, c1)}

    base_grid = np.array(grid, copy=True)
    scenarios = _scenario_pool(room_infos, doors)
    options = _option_pool(base_grid, scenarios, gaps)
    if not options:
        return None
    chosen = _choose_options(options, len(room_infos), rng)
    if len(chosen) < 2:
        return None

    colors = list(SEED_COLORS)
    rng.shuffle(colors)
    seeds = []
    for color, option in zip(colors, chosen):
        grid[option["start"]] = color
        seeds.append(option["start"])

    traces = _trace_cells(grid, seeds, BG, WALL)
    all_cells = []
    for cells in traces.values():
        all_cells.extend(cells)
    if len(all_cells) != len(set(all_cells)):
        return None

    lengths = [len(cells) for cells in traces.values()]
    if min(lengths) < 1:
        return None

    crosses_gap = 0
    for cells in traces.values():
        if any(cell in gaps for cell in cells):
            crosses_gap += 1
    if crosses_gap == 0:
        return None
    if crosses_gap == len(seeds):
        return None

    tie_present = any(option["want_tie"] for option in chosen)
    pass_present = any(option["crosses_gap"] for option in chosen)
    stop_present = any(not option["crosses_gap"] for option in chosen)
    rooms, labels = _room_data(grid, BG, WALL)
    for sr, sc in seeds:
        room_idx = int(labels[sr, sc])
        bbox = rooms[room_idx]["bbox"]
        direction = _direction_from_bbox(sr, sc, bbox)
        left = sc - bbox[2]
        right = bbox[3] - sc
        up = sr - bbox[0]
        down = bbox[1] - sr
        if min(left, right) == min(up, down) and direction[0] == 0:
            tie_present = True
    if not pass_present or not stop_present:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after retries")
