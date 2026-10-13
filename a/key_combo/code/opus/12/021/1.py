from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
WALL = 5
SEED_COLORS = (BLUE, RED, GREEN)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bool_component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _seed_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in SEED_COLORS:
        for coords in _bool_component_coords(grid == color):
            coord_list = [tuple(map(int, rc)) for rc in coords]
            bbox = _bbox_from_coords(coord_list)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "top_row": int(bbox[0]),
                    "left_col": int(bbox[1]),
                }
            )
    comps.sort(key=lambda comp: (comp["top_row"], comp["left_col"], comp["color"]))
    return comps


def _mask_from_component(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _symmetry_type(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    mask = _mask_from_component(coords, bbox)
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    if horizontal and not vertical:
        return "H"
    if vertical and not horizontal:
        return "V"
    return "N"


def _door_map(grid: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], str]]:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    orient: dict[tuple[int, int], str] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BACKGROUND:
                continue
            up = int(grid[r - 1, c]) == WALL
            down = int(grid[r + 1, c]) == WALL
            left = int(grid[r, c - 1]) == WALL
            right = int(grid[r, c + 1]) == WALL
            if up and down and not left and not right:
                mask[r, c] = True
                orient[(r, c)] = "H"
            elif left and right and not up and not down:
                mask[r, c] = True
                orient[(r, c)] = "V"
    return mask, orient


def _room_partition(grid: np.ndarray, door_mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    roomable = (grid != WALL) & ~door_mask
    room_id = np.full((h, w), -1, dtype=int)
    rooms: list[dict] = []
    next_id = 0
    for coords in _bool_component_coords(roomable):
        coord_list = [tuple(map(int, rc)) for rc in coords]
        for r, c in coord_list:
            room_id[r, c] = next_id
        background = [(r, c) for r, c in coord_list if int(grid[r, c]) == BACKGROUND]
        rooms.append(
            {
                "id": next_id,
                "coords": coord_list,
                "background": background,
                "bbox": _bbox_from_coords(coord_list),
            }
        )
        next_id += 1
    return room_id, rooms


def _room_graph(
    grid: np.ndarray,
    door_mask: np.ndarray,
    door_orient: dict[tuple[int, int], str],
    room_id: np.ndarray,
    room_count: int,
) -> tuple[list[list[tuple[int, str, tuple[int, int]]]], dict[tuple[int, int], tuple[int, int]]]:
    adj = [[] for _ in range(room_count)]
    door_rooms: dict[tuple[int, int], tuple[int, int]] = {}
    for (r, c), orient in door_orient.items():
        if orient == "H":
            a = int(room_id[r, c - 1])
            b = int(room_id[r, c + 1])
        else:
            a = int(room_id[r - 1, c])
            b = int(room_id[r + 1, c])
        if a < 0 or b < 0 or a == b:
            continue
        door_rooms[(r, c)] = (a, b)
        adj[a].append((b, orient, (r, c)))
        adj[b].append((a, orient, (r, c)))
    return adj, door_rooms


def _allowed_orientations(symmetry: str) -> set[str]:
    if symmetry == "H":
        return {"H"}
    if symmetry == "V":
        return {"V"}
    return {"H", "V"}


def _solve_with_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    door_mask, door_orient = _door_map(g)
    room_id, rooms = _room_partition(g, door_mask)
    seeds = _seed_components(g)
    for seed in seeds:
        seed["symmetry"] = _symmetry_type(seed["coords"], seed["bbox"])
        sr, sc = map(int, seed["coords"][0])
        seed["room"] = int(room_id[sr, sc])

    adj, door_rooms = _room_graph(g, door_mask, door_orient, room_id, len(rooms))

    room_candidates: dict[int, list[tuple[tuple[int, int, int, int], int]]] = {
        room["id"]: [] for room in rooms
    }
    door_candidates: dict[tuple[int, int], list[tuple[tuple[int, int, int, int], int]]] = {
        door: [] for door in door_rooms
    }

    for idx, seed in enumerate(seeds):
        start = int(seed["room"])
        allowed = _allowed_orientations(seed["symmetry"])
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nxt, orient, _door in adj[cur]:
                if orient not in allowed or nxt in dist:
                    continue
                dist[nxt] = dist[cur] + 1
                q.append(nxt)

        tie = (seed["top_row"], seed["left_col"], idx)
        for rid, d in dist.items():
            room_candidates[rid].append(((d, *tie), idx))
        for door, (a, b) in door_rooms.items():
            if door_orient[door] not in allowed:
                continue
            if a in dist:
                door_candidates[door].append(((dist[a], *tie), idx))
            if b in dist:
                door_candidates[door].append(((dist[b], *tie), idx))

    room_winner: dict[int, int] = {}
    for rid, candidates in room_candidates.items():
        if candidates:
            room_winner[rid] = min(candidates, key=lambda item: item[0])[1]

    door_winner: dict[tuple[int, int], int] = {}
    for door, candidates in door_candidates.items():
        if candidates:
            door_winner[door] = min(candidates, key=lambda item: item[0])[1]

    out = g.copy()
    for room in rooms:
        winner = room_winner.get(room["id"])
        if winner is None:
            continue
        color = int(seeds[winner]["color"])
        for r, c in room["background"]:
            out[r, c] = color
    for (r, c), winner in door_winner.items():
        out[r, c] = int(seeds[winner]["color"])

    return {
        "output": out,
        "rooms": rooms,
        "room_id": room_id,
        "door_mask": door_mask,
        "door_orient": door_orient,
        "door_rooms": door_rooms,
        "seeds": seeds,
        "room_candidates": room_candidates,
        "door_candidates": door_candidates,
        "room_winner": room_winner,
        "door_winner": door_winner,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood each room through only those door orientations permitted by its seed's reflection symmetry.

    Input structure:
    - The grid has black background cells, grey wall cells, and one small colored seed object in each seeded room.
    - Single-cell gaps in the walls act as doors; a gap in a vertical wall is a horizontal door, and a gap in a horizontal wall is a vertical door.
    - Seed objects use blue, red, or green and are 4-connected shapes whose geometry encodes horizontal symmetry, vertical symmetry, or neither.

    Rule:
    - Detect each seed's reflection type from its tight mask.
    - Horizontally symmetric seeds may traverse only horizontal doors, vertically symmetric seeds only vertical doors, and non-symmetric seeds either kind.
    - Starting from every seed, flood all reachable background cells with that seed's color while keeping seed pixels and grey walls unchanged.
    - If multiple seeds can claim the same chamber, prefer the one that reaches it through fewer doors; if tied, prefer the seed whose bbox starts on the smaller row.

    Key insight:
    - Symmetry does not map directly to an output color; it filters which wall gaps count as traversable, so geometry changes the reachable topology before the repaint happens.
    """
    return _solve_with_details(np.asarray(grid, dtype=int))["output"]


H_SHAPES = [
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=int),
]

V_SHAPES = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=int),
]

N_SHAPES = [
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=int),
]


def _room_ranges(lengths: list[int]) -> list[tuple[int, int]]:
    ranges = []
    pos = 1
    for size in lengths:
        ranges.append((pos, pos + size - 1))
        pos += size + 1
    return ranges


def _build_base_scene() -> tuple[np.ndarray, list[dict], int]:
    row_sizes = [random.randint(6, 8), random.randint(6, 8)]
    col_sizes = [random.randint(5, 7), random.randint(5, 7), random.randint(5, 7)]
    h = sum(row_sizes) + 3
    w = sum(col_sizes) + 4
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    row_wall = 1 + row_sizes[0]
    col_wall_1 = 1 + col_sizes[0]
    col_wall_2 = 2 + col_sizes[0] + col_sizes[1]
    grid[row_wall, :] = WALL
    grid[:, col_wall_1] = WALL
    grid[:, col_wall_2] = WALL

    row_ranges = _room_ranges(row_sizes)
    col_ranges = _room_ranges(col_sizes)
    rooms: list[dict] = []
    rid = 0
    for rr, (r0, r1) in enumerate(row_ranges):
        for cc, (c0, c1) in enumerate(col_ranges):
            rooms.append(
                {
                    "id": rid,
                    "row": rr,
                    "col": cc,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                }
            )
            rid += 1

    # One door on every shared wall keeps each seeded room exposed to both door orientations.
    for rr, (r0, r1) in enumerate(row_ranges):
        for wall_c in (col_wall_1, col_wall_2):
            door_r = random.randint(r0 + 1, r1 - 1)
            grid[door_r, wall_c] = BACKGROUND
    for cc, (c0, c1) in enumerate(col_ranges):
        door_c = random.randint(c0 + 1, c1 - 1)
        grid[row_wall, door_c] = BACKGROUND

    # Small inward wall stubs make some rooms non-rectangular without creating new rooms.
    stub_count = random.randint(1, 3)
    for _ in range(stub_count):
        room = random.choice(rooms)
        r0, r1, c0, c1 = room["r0"], room["r1"], room["c0"], room["c1"]
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        side = random.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            c = random.randint(c0 + 1, c1 - 1)
            length = random.randint(1, 2)
            rows = range(r0, min(r0 + length, r1)) if side == "top" else range(max(r1 - length + 1, r0), r1 + 1)
            for r in rows:
                if grid[r, c] == BACKGROUND:
                    grid[r, c] = WALL
        else:
            r = random.randint(r0 + 1, r1 - 1)
            length = random.randint(1, 2)
            cols = range(c0, min(c0 + length, c1)) if side == "left" else range(max(c1 - length + 1, c0), c1 + 1)
            for c in cols:
                if grid[r, c] == BACKGROUND:
                    grid[r, c] = WALL

    empty_room_id = 1 if random.random() < 0.5 else 4
    return grid, rooms, empty_room_id


def _pick_shape(kind: str) -> np.ndarray:
    if kind == "H":
        return random.choice(H_SHAPES)
    if kind == "V":
        return random.choice(V_SHAPES)
    return random.choice(N_SHAPES)


def _place_shape(grid: np.ndarray, room_mask: np.ndarray, shape: np.ndarray, color: int, top_bias: str | None = None) -> tuple[int, int] | None:
    h, w = grid.shape
    sh, sw = shape.shape
    candidates: list[tuple[int, int]] = []
    rows, cols = np.where(room_mask)
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    for r0 in range(rmin, rmax - sh + 2):
        for c0 in range(cmin, cmax - sw + 2):
            ok = True
            for rr in range(sh):
                for cc in range(sw):
                    if shape[rr, cc] and (not room_mask[r0 + rr, c0 + cc] or int(grid[r0 + rr, c0 + cc]) != BACKGROUND):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r0, c0))
    if not candidates:
        return None
    candidates.sort()
    if top_bias == "high":
        chosen = candidates[0]
    elif top_bias == "low":
        chosen = candidates[-1]
    else:
        chosen = random.choice(candidates)
    r0, c0 = chosen
    for rr in range(sh):
        for cc in range(sw):
            if shape[rr, cc]:
                grid[r0 + rr, c0 + cc] = color
    return chosen


def _place_shape_of_kind(
    grid: np.ndarray,
    room_mask: np.ndarray,
    kind: str,
    color: int,
    top_bias: str | None = None,
) -> tuple[int, int] | None:
    if kind == "H":
        shapes = H_SHAPES[:]
    elif kind == "V":
        shapes = V_SHAPES[:]
    else:
        shapes = N_SHAPES[:]
    random.shuffle(shapes)
    for shape in shapes:
        pos = _place_shape(grid, room_mask, shape, color, top_bias=top_bias)
        if pos is not None:
            return pos
    return None


def _rotate_layout_cell(rc: tuple[int, int], rotations: int, nrows: int = 2, ncols: int = 3) -> tuple[int, int]:
    r, c = rc
    cur_r, cur_c, cur_nr, cur_nc = r, c, nrows, ncols
    for _ in range(rotations % 4):
        cur_r, cur_c = cur_c, cur_nr - 1 - cur_r
        cur_nr, cur_nc = cur_nc, cur_nr
    return cur_r, cur_c


def _room_by_cell(rooms: list[dict], row: int, col: int) -> dict:
    for room in rooms:
        if room["row"] == row and room["col"] == col:
            return room
    raise KeyError((row, col))


def _generate_attempt() -> np.ndarray | None:
    grid, layout_rooms, empty_room_id = _build_base_scene()

    door_mask, _ = _door_map(grid)
    room_id, rooms = _room_partition(grid, door_mask)

    empty_layout = next(room for room in layout_rooms if room["id"] == empty_room_id)
    empty_neighbors: list[tuple[int, int]] = []
    er, ec = empty_layout["row"], empty_layout["col"]
    for dr, dc in DIR4:
        nr, nc = er + dr, ec + dc
        if 0 <= nr < 2 and 0 <= nc < 3:
            empty_neighbors.append((nr, nc))

    type_by_room: dict[int, str] = {}
    color_by_room: dict[int, int] = {}
    top_bias: dict[int, str | None] = {}

    side_neighbors = [room for room in empty_neighbors if room[0] == er]
    vertical_neighbor = [room for room in empty_neighbors if room[0] != er]

    # Two same-row neighbors enter the empty chamber through horizontal doors and force a row-based tie.
    for i, cell in enumerate(side_neighbors):
        room = _room_by_cell(layout_rooms, *cell)
        type_by_room[room["id"]] = "H"
        top_bias[room["id"]] = "high" if i == 0 else "low"

    if vertical_neighbor:
        room = _room_by_cell(layout_rooms, *vertical_neighbor[0])
        type_by_room[room["id"]] = "V"
        top_bias[room["id"]] = random.choice((None, "high"))

    remaining_seed_rooms = [room for room in layout_rooms if room["id"] != empty_room_id and room["id"] not in type_by_room]
    forced_remaining = ["N", random.choice(("H", "V", "N"))]
    for idx, room in enumerate(remaining_seed_rooms):
        type_by_room[room["id"]] = forced_remaining[idx] if idx < len(forced_remaining) else random.choice(("H", "V", "N"))
        top_bias[room["id"]] = None

    # Randomize visible color/symmetry pairing each time.
    palette = [BLUE, RED, GREEN]
    for room in layout_rooms:
        if room["id"] == empty_room_id:
            continue
        color_by_room[room["id"]] = random.choice(palette)

    placed = 0
    for layout_room in layout_rooms:
        rid = layout_room["id"]
        if rid == empty_room_id:
            continue
        room_info = rooms[rid]
        room_mask = room_id == rid
        pos = _place_shape_of_kind(grid, room_mask, type_by_room[rid], color_by_room[rid], top_bias=top_bias[rid])
        if pos is None:
            return None
        placed += 1
    if placed < 4:
        return None

    details = _solve_with_details(grid)
    out = details["output"]
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out != grid)) < max(8, grid.size // 10):
        return None

    seeded_rooms = {seed["room"] for seed in details["seeds"]}
    empty_rooms = [room["id"] for room in details["rooms"] if room["id"] not in seeded_rooms]
    if not empty_rooms:
        return None

    # Ensure the empty chamber is claimed competitively and that the row tie-break matters.
    competitive = False
    row_tie = False
    for rid in empty_rooms:
        candidates = details["room_candidates"][rid]
        if len({idx for _score, idx in candidates}) >= 2:
            competitive = True
        best = sorted(candidates, key=lambda item: item[0])
        if len(best) >= 2 and best[0][0][0] == best[1][0][0] and best[0][0][1] != best[1][0][1]:
            row_tie = True
    if not competitive or not row_tie:
        return None

    symmetries = {seed["symmetry"] for seed in details["seeds"]}
    if len(symmetries) < 3:
        return None

    # Require both door orientations to matter somewhere in the scene.
    useful_orients = set()
    for candidates in details["room_candidates"].values():
        for _score, seed_idx in candidates:
            useful_orients.update(_allowed_orientations(details["seeds"][seed_idx]["symmetry"]))
    if useful_orients != {"H", "V"}:
        return None

    # Rotate the whole puzzle for extra variety while keeping the solver generic.
    rotations = random.randint(0, 3)
    if rotations:
        grid = np.rot90(grid, rotations)
        details = _solve_with_details(grid)
        out = details["output"]
        if np.array_equal(out, grid):
            return None
        seeded_rooms = {seed["room"] for seed in details["seeds"]}
        empty_rooms = [room["id"] for room in details["rooms"] if room["id"] not in seeded_rooms]
        if not empty_rooms:
            return None
        row_tie = False
        for rid in empty_rooms:
            best = sorted(details["room_candidates"][rid], key=lambda item: item[0])
            if len(best) >= 2 and best[0][0][0] == best[1][0][0] and best[0][0][1] != best[1][0][1]:
                row_tie = True
        if not row_tie:
            return None

    return grid.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None and grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-gated flood puzzle")
