import random
from collections import Counter

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _label_components(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    label = 0
    for comp in _components(mask):
        label += 1
        labels[tuple(comp.T)] = label
    return labels


def _find_doors(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    doors: list[dict] = []
    seen = set()

    for r in range(1, h - 1):
        for c in range(w - 3):
            col = int(g[r, c])
            if col == bg or int(g[r, c + 3]) != col:
                continue
            if int(g[r, c + 1]) != bg or int(g[r, c + 2]) != bg:
                continue
            if not (((g[r - 1, c + 1] != bg) or (g[r + 1, c + 1] != bg)) and ((g[r - 1, c + 2] != bg) or (g[r + 1, c + 2] != bg))):
                continue
            key = ("h", r, c)
            if key not in seen:
                seen.add(key)
                doors.append(
                    {
                        "orientation": "h",
                        "line": r,
                        "start": c,
                        "color": col,
                        "frames": [(r, c), (r, c + 3)],
                        "gap": [(r, c + 1), (r, c + 2)],
                    }
                )

    for r in range(h - 3):
        for c in range(1, w - 1):
            col = int(g[r, c])
            if col == bg or int(g[r + 3, c]) != col:
                continue
            if int(g[r + 1, c]) != bg or int(g[r + 2, c]) != bg:
                continue
            if not (((g[r + 1, c - 1] != bg) or (g[r + 1, c + 1] != bg)) and ((g[r + 2, c - 1] != bg) or (g[r + 2, c + 1] != bg))):
                continue
            key = ("v", r, c)
            if key not in seen:
                seen.add(key)
                doors.append(
                    {
                        "orientation": "v",
                        "line": c,
                        "start": r,
                        "color": col,
                        "frames": [(r, c), (r + 3, c)],
                        "gap": [(r + 1, c), (r + 2, c)],
                    }
                )

    return doors


def _room_masks(passable: np.ndarray, door: dict) -> tuple[np.ndarray, np.ndarray] | None:
    blocked = passable.copy()
    for r, c in door["gap"]:
        blocked[r, c] = False
    labels = _label_components(blocked)

    side_a = []
    side_b = []
    if door["orientation"] == "h":
        r = door["line"]
        c0 = door["start"] + 1
        for c in (c0, c0 + 1):
            if r - 1 >= 0 and blocked[r - 1, c]:
                side_a.append((r - 1, c))
            if r + 1 < blocked.shape[0] and blocked[r + 1, c]:
                side_b.append((r + 1, c))
    else:
        c = door["line"]
        r0 = door["start"] + 1
        for r in (r0, r0 + 1):
            if c - 1 >= 0 and blocked[r, c - 1]:
                side_a.append((r, c - 1))
            if c + 1 < blocked.shape[1] and blocked[r, c + 1]:
                side_b.append((r, c + 1))

    if not side_a or not side_b:
        return None
    id_a = int(labels[side_a[0]])
    id_b = int(labels[side_b[0]])
    if id_a == 0 or id_b == 0 or id_a == id_b:
        return None
    return labels == id_a, labels == id_b


def _mirror_point(door: dict, r: int, c: int) -> tuple[int, int]:
    if door["orientation"] == "h":
        return 2 * door["line"] - r, c
    return r, 2 * door["line"] - c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat paired same-color pixels around a 2-cell wall gap as a doorway.
    If exactly one side contains a pattern, mirror-copy that pattern across
    the wall line into the empty neighboring region.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    doors = _find_doors(g, bg)
    if not doors:
        return g.copy()

    door_mask = np.zeros_like(g, dtype=bool)
    for door in doors:
        for r, c in door["frames"]:
            door_mask[r, c] = True

    wall_candidates = [int(g[r, c]) for r in range(g.shape[0]) for c in range(g.shape[1]) if int(g[r, c]) != bg and not door_mask[r, c]]
    if not wall_candidates:
        return g.copy()
    wall = Counter(wall_candidates).most_common(1)[0][0]

    passable = (g != wall) & (~door_mask)
    pattern_mask = (g != bg) & (g != wall) & (~door_mask)
    out = g.copy()

    for door in doors:
        room_pair = _room_masks(passable, door)
        if room_pair is None:
            continue
        room_a, room_b = room_pair
        pat_a = room_a & pattern_mask
        pat_b = room_b & pattern_mask
        has_a = bool(np.any(pat_a))
        has_b = bool(np.any(pat_b))
        if has_a == has_b:
            continue

        src_mask = pat_a if has_a else pat_b
        dst_room = room_b if has_a else room_a
        for r, c in np.argwhere(src_mask):
            tr, tc = _mirror_point(door, int(r), int(c))
            if 0 <= tr < g.shape[0] and 0 <= tc < g.shape[1] and dst_room[tr, tc] and int(g[tr, tc]) == bg:
                out[tr, tc] = int(g[r, c])

    return out


_MAX_ATTEMPTS = 400
_COLORS = list(range(10))


def _draw_rect_walls(grid: np.ndarray, row_edges: list[int], col_edges: list[int], wall: int) -> None:
    for r in row_edges:
        grid[r, col_edges[0] : col_edges[-1] + 1] = wall
    for c in col_edges:
        grid[row_edges[0] : row_edges[-1] + 1, c] = wall


def _room_bbox(row_edges: list[int], col_edges: list[int], i: int, j: int) -> tuple[int, int, int, int]:
    r0 = row_edges[i] + 1
    r1 = row_edges[i + 1] - 1
    c0 = col_edges[j] + 1
    c1 = col_edges[j + 1] - 1
    return r0, r1, c0, c1


def _place_connected_pattern(grid: np.ndarray, bbox: tuple[int, int, int, int], colors: list[int], rng: random.Random) -> np.ndarray | None:
    r0, r1, c0, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2:
        return None

    target = rng.randint(2, min(7, h * w))
    pts = [(rng.randint(r0, r1), rng.randint(c0, c1))]
    used = {pts[0]}
    while len(pts) < target:
        base_r, base_c = rng.choice(pts)
        nbrs = [(nr, nc) for nr, nc in _neighbors4(base_r, base_c) if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in used]
        if nbrs:
            nxt = rng.choice(nbrs)
            used.add(nxt)
            pts.append(nxt)
        elif len(used) == h * w:
            break
        else:
            free = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in used]
            nxt = rng.choice(free)
            used.add(nxt)
            pts.append(nxt)

    pts_arr = np.asarray(pts, dtype=int)
    for r, c in pts:
        grid[r, c] = rng.choice(colors)
    return pts_arr


def _mirror_points(door: dict, pts: np.ndarray) -> np.ndarray:
    res = []
    for r, c in pts:
        res.append(_mirror_point(door, int(r), int(c)))
    return np.asarray(res, dtype=int)


def _valid_copy_for_door(grid: np.ndarray, door: dict, bg: int, wall: int) -> bool:
    doors = _find_doors(grid, bg)
    door_mask = np.zeros_like(grid, dtype=bool)
    for d in doors:
        for r, c in d["frames"]:
            door_mask[r, c] = True
    passable = (grid != wall) & (~door_mask)
    room_pair = _room_masks(passable, door)
    if room_pair is None:
        return False
    room_a, room_b = room_pair
    pattern_mask = (grid != bg) & (grid != wall) & (~door_mask)
    pat_a = room_a & pattern_mask
    pat_b = room_b & pattern_mask
    if bool(np.any(pat_a)) == bool(np.any(pat_b)):
        return False
    src = np.argwhere(pat_a if np.any(pat_a) else pat_b)
    dst_room = room_b if np.any(pat_a) else room_a
    mirrored = _mirror_points(door, src)
    return any(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and dst_room[r, c] and int(grid[r, c]) == bg for r, c in mirrored)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    bg, wall = rng.sample(_COLORS, 2)
    remaining = [c for c in _COLORS if c not in (bg, wall)]
    rng.shuffle(remaining)
    n_door_colors = rng.randint(1, 3)
    door_colors = remaining[:n_door_colors]
    pattern_colors = remaining[n_door_colors:]
    if len(pattern_colors) < 2:
        return None

    n_rows = rng.choice([2, 3])
    n_cols = rng.choice([2, 3])
    room_heights = [rng.randint(4, 8) for _ in range(n_rows)]
    room_widths = [rng.randint(4, 8) for _ in range(n_cols)]
    h = sum(room_heights) + n_rows + 1
    w = sum(room_widths) + n_cols + 1
    if not (14 <= h <= 28 and 14 <= w <= 28):
        return None

    row_edges = [0]
    for size in room_heights:
        row_edges.append(row_edges[-1] + size + 1)
    col_edges = [0]
    for size in room_widths:
        col_edges.append(col_edges[-1] + size + 1)

    grid = np.full((h, w), bg, dtype=int)
    _draw_rect_walls(grid, row_edges, col_edges, wall)

    candidates = []
    for i in range(n_rows - 1):
        for j in range(n_cols):
            if room_widths[j] >= 4:
                candidates.append(("h", i, j))
    for i in range(n_rows):
        for j in range(n_cols - 1):
            if room_heights[i] >= 4:
                candidates.append(("v", i, j))
    rng.shuffle(candidates)

    selected = []
    used_rooms = set()
    target_doors = rng.randint(2, min(4, len(candidates)))
    for cand in candidates:
        ori, i, j = cand
        rooms = {(i, j), (i + 1, j)} if ori == "h" else {(i, j), (i, j + 1)}
        if used_rooms & rooms:
            continue
        selected.append(cand)
        used_rooms |= rooms
        if len(selected) >= target_doors:
            break
    if len(selected) < 2:
        return None

    door_info = []
    for idx, (ori, i, j) in enumerate(selected):
        color = door_colors[idx % len(door_colors)]
        if ori == "h":
            wall_row = row_edges[i + 1]
            left = col_edges[j] + 1
            right = col_edges[j + 1] - 1
            start = rng.randint(left, right - 3)
            grid[wall_row, start] = color
            grid[wall_row, start + 1] = bg
            grid[wall_row, start + 2] = bg
            grid[wall_row, start + 3] = color
            door = {
                "orientation": "h",
                "line": wall_row,
                "start": start,
                "color": color,
                "rooms": ((i, j), (i + 1, j)),
                "frames": [(wall_row, start), (wall_row, start + 3)],
                "gap": [(wall_row, start + 1), (wall_row, start + 2)],
            }
        else:
            wall_col = col_edges[j + 1]
            top = row_edges[i] + 1
            bottom = row_edges[i + 1] - 1
            start = rng.randint(top, bottom - 3)
            grid[start, wall_col] = color
            grid[start + 1, wall_col] = bg
            grid[start + 2, wall_col] = bg
            grid[start + 3, wall_col] = color
            door = {
                "orientation": "v",
                "line": wall_col,
                "start": start,
                "color": color,
                "rooms": ((i, j), (i, j + 1)),
                "frames": [(start, wall_col), (start + 3, wall_col)],
                "gap": [(start + 1, wall_col), (start + 2, wall_col)],
            }
        door_info.append(door)

    states = ["a_only", "b_only", "both_empty", "both_patterned"]
    rng.shuffle(states)
    while len(states) < len(door_info):
        states.append(rng.choice(["a_only", "b_only", "both_empty", "both_patterned"]))

    occupied_rooms = set()
    active_copies = 0
    for door, state in zip(door_info, states):
        room_a, room_b = door["rooms"]
        bbox_a = _room_bbox(row_edges, col_edges, *room_a)
        bbox_b = _room_bbox(row_edges, col_edges, *room_b)

        pts_a = pts_b = None
        if state in ("a_only", "both_patterned"):
            pts_a = _place_connected_pattern(grid, bbox_a, pattern_colors, rng)
            if pts_a is None:
                return None
            occupied_rooms.add(room_a)
        if state in ("b_only", "both_patterned"):
            pts_b = _place_connected_pattern(grid, bbox_b, pattern_colors, rng)
            if pts_b is None:
                return None
            occupied_rooms.add(room_b)
        if state in ("a_only", "b_only"):
            if not _valid_copy_for_door(grid, door, bg, wall):
                return None
            active_copies += 1

    extra_rooms = [(i, j) for i in range(n_rows) for j in range(n_cols) if (i, j) not in occupied_rooms and rng.random() < 0.25]
    for room in extra_rooms:
        if _place_connected_pattern(grid, _room_bbox(row_edges, col_edges, *room), pattern_colors, rng) is None:
            return None

    if active_copies == 0:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
