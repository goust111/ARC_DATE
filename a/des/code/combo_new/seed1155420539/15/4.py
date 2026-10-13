import random
from collections import deque

import numpy as np


BG = 0
WALL = 7
KEY_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
SIDES = ("top", "bottom", "left", "right")
LAYOUTS = {
    1: [(1, 1)],
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1), (2, 2)],
    4: [(2, 2)],
}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    pref = mask.astype(int).cumsum(axis=0).cumsum(axis=1)
    return np.pad(pref, ((1, 0), (1, 0)), constant_values=0)


def _rect_sum(pref: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(pref[r1, c1] - pref[r0, c1] - pref[r1, c0] + pref[r0, c0])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_4_connected(coords: list[tuple[int, int]]) -> bool:
    pts = set(coords)
    start = next(iter(pts))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nbr in _neighbors4(r, c):
            if nbr in pts and nbr not in seen:
                seen.add(nbr)
                q.append(nbr)
    return len(seen) == len(pts)


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return norm, h, w


def _rotation_matches(key_coords: list[tuple[int, int]], gap_coords: list[tuple[int, int]]) -> bool:
    key_norm, h, w = _normalize_coords(key_coords)
    gap_norm, _gh, _gw = _normalize_coords(gap_coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in key_norm:
        mask[r, c] = True
    for k in range(4):
        rot = np.rot90(mask, k)
        rot_coords = tuple(map(tuple, np.argwhere(rot)))
        if rot_coords == gap_norm:
            return True
    return False


def _classify_side(vals: np.ndarray):
    non_wall = np.flatnonzero(vals != WALL)
    if non_wall.size == 0:
        return "wall", None
    if np.any(vals[non_wall] != BG):
        return "invalid", None
    if int(non_wall[-1] - non_wall[0] + 1) != int(non_wall.size):
        return "invalid", None
    if non_wall.size > 3:
        return "invalid", None
    return "gap", (int(non_wall[0]), int(non_wall[-1]))


def _gap_details(bbox: tuple[int, int, int, int], side: str, start: int, end: int):
    r0, c0, r1, c1 = bbox
    if side == "top":
        coords = [(r0, c0 + 1 + d) for d in range(start, end + 1)]
        outside_step = (-1, 0)
    elif side == "bottom":
        coords = [(r1, c0 + 1 + d) for d in range(start, end + 1)]
        outside_step = (1, 0)
    elif side == "left":
        coords = [(r0 + 1 + d, c0) for d in range(start, end + 1)]
        outside_step = (0, -1)
    else:
        coords = [(r0 + 1 + d, c1) for d in range(start, end + 1)]
        outside_step = (0, 1)
    return coords, outside_step


def _find_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall_pref = _prefix_sum(g == WALL)
    key_mask = (g != BG) & (g != WALL)
    key_pref = _prefix_sum(key_mask)
    key_components = _components4(key_mask)
    assert key_components, "expected at least one key component"
    key_sets = {frozenset(comp) for comp in key_components}
    candidates_by_key = {ks: [] for ks in key_sets}

    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            inner_r0, inner_r1 = r0 + 1, r1
            for c0 in range(w - 2):
                for c1 in range(c0 + 2, w):
                    key_count = _rect_sum(key_pref, inner_r0, c0 + 1, inner_r1, c1)
                    if not (1 <= key_count <= 3):
                        continue
                    if _rect_sum(wall_pref, inner_r0, c0 + 1, inner_r1, c1) != 0:
                        continue
                    if g[r0, c0] != WALL or g[r0, c1] != WALL or g[r1, c0] != WALL or g[r1, c1] != WALL:
                        continue

                    side_specs = [
                        ("top", g[r0, c0 + 1 : c1]),
                        ("bottom", g[r1, c0 + 1 : c1]),
                        ("left", g[r0 + 1 : r1, c0]),
                        ("right", g[r0 + 1 : r1, c1]),
                    ]
                    gap_side = None
                    gap_span = None
                    valid = True
                    for side_name, vals in side_specs:
                        status, span = _classify_side(vals)
                        if status == "invalid":
                            valid = False
                            break
                        if status == "gap":
                            if gap_side is not None:
                                valid = False
                                break
                            gap_side = side_name
                            gap_span = span
                    if not valid or gap_side is None:
                        continue

                    inner = g[r0 + 1 : r1, c0 + 1 : c1]
                    rel_key_coords = [tuple(map(int, rc)) for rc in np.argwhere((inner != BG) & (inner != WALL))]
                    if len(rel_key_coords) != key_count or not _is_4_connected(rel_key_coords):
                        continue
                    key_colors = {int(inner[r, c]) for r, c in rel_key_coords}
                    if len(key_colors) != 1:
                        continue

                    abs_key_coords = [(r + r0 + 1, c + c0 + 1) for r, c in rel_key_coords]
                    key_set = frozenset(abs_key_coords)
                    if key_set not in candidates_by_key:
                        continue

                    gap_coords, outside_step = _gap_details((r0, c0, r1, c1), gap_side, gap_span[0], gap_span[1])
                    if len(gap_coords) != len(abs_key_coords):
                        continue
                    if not _rotation_matches(rel_key_coords, gap_coords):
                        continue

                    candidates_by_key[key_set].append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "gap_side": gap_side,
                            "gap_coords": gap_coords,
                            "outside_step": outside_step,
                            "key_coords": abs_key_coords,
                            "key_color": key_colors.pop(),
                        }
                    )

    rooms = []
    for key_set in sorted(key_sets, key=lambda ks: tuple(sorted(ks))):
        candidates = candidates_by_key[key_set]
        assert candidates, "failed to identify a room for a key"
        best = min(candidates, key=lambda room: ((room["bbox"][2] - room["bbox"][0]) * (room["bbox"][3] - room["bbox"][1]), room["bbox"]))
        rooms.append(best)
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _flood_black(out: np.ndarray, starts: list[tuple[int, int]], color: int) -> None:
    h, w = out.shape
    q = deque()
    seen = set()
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and out[r, c] == BG and (r, c) not in seen:
            seen.add((r, c))
            q.append((r, c))
    while q:
        r, c = q.popleft()
        out[r, c] = color
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == BG and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each room's key into its wall gap, then flood the connected corridor outside that gap with the key color.

    Input structure:
    - Background is black, representing corridor cells and room interiors.
    - Orange one-cell-thick rectangular walls form rooms; each room has exactly one side gap of length 1-3 away from the corners.
    - Each room interior contains one 4-connected non-black, non-orange key fragment of 1-3 pixels, and different rooms may share wall segments.

    Rule:
    - Detect every rectangular room by finding a mostly orange perimeter with exactly one contiguous wall gap.
    - Read the unique key inside the room, verify that some rotation matches the gap's straight arrangement, erase the original key, and paint the gap cells with the key color.
    - Starting from the cells immediately outside the gap, flood-fill the 4-connected black corridor region with the same key color.
    - Keep all remaining orange wall cells unchanged and leave the rest of each room interior black.

    Key insight:
    - The wall geometry identifies each room and its lockhole, while the key fragment's rotated shape determines which missing wall segment it plugs before its color leaks into the connected outer corridor.
    """
    g = np.asarray(grid, dtype=int)
    rooms = _find_rooms(g)

    out = g.copy()
    for room in rooms:
        color = int(room["key_color"])
        for r, c in room["key_coords"]:
            out[r, c] = BG
        for r, c in room["gap_coords"]:
            out[r, c] = color
        dr, dc = room["outside_step"]
        starts = [(r + dr, c + dc) for r, c in room["gap_coords"]]
        _flood_black(out, starts, color)
    return out


def _sample_sizes(rng: random.Random, parts: int) -> list[int] | None:
    if parts == 1:
        min_size, max_size = 10, 16
    elif parts == 2:
        min_size, max_size = 8, 12
    else:
        min_size, max_size = 7, 8
    total_cap = 29 - parts
    if parts * min_size > total_cap:
        return None
    total = rng.randint(parts * min_size, min(parts * max_size, total_cap))
    sizes = [min_size] * parts
    slack = [max_size - min_size] * parts
    extra = total - parts * min_size
    while extra > 0:
        choices = [i for i, rem in enumerate(slack) if rem > 0]
        if not choices:
            break
        idx = rng.choice(choices)
        add = rng.randint(1, min(slack[idx], extra))
        sizes[idx] += add
        slack[idx] -= add
        extra -= add
    rng.shuffle(sizes)
    return sizes


def _cell_rects(heights: list[int], widths: list[int]) -> tuple[int, int, list[tuple[int, int, int, int]], list[int], list[int]]:
    r_lines = [0]
    cur = 0
    for h in heights:
        cur += h + 1
        r_lines.append(cur)
    h_total = cur + 1

    c_lines = [0]
    cur = 0
    for w in widths:
        cur += w + 1
        c_lines.append(cur)
    w_total = cur + 1

    cells = []
    for i in range(len(heights)):
        for j in range(len(widths)):
            cells.append((r_lines[i] + 1, r_lines[i + 1] - 1, c_lines[j] + 1, c_lines[j + 1] - 1))
    return h_total, w_total, cells, r_lines, c_lines


def _enumerate_room_bboxes(cell: tuple[int, int, int, int], anchors: set[str], gap_side: str, gap_len: int) -> list[tuple[int, int, int, int]]:
    cr0, cr1, cc0, cc1 = cell
    top_wall = cr0 - 1
    bottom_wall = cr1 + 1
    left_wall = cc0 - 1
    right_wall = cc1 + 1

    tops = [top_wall] if "top" in anchors else list(range(cr0 + 1, cr1))
    bottoms = [bottom_wall] if "bottom" in anchors else list(range(cr0 + 2, cr1))
    lefts = [left_wall] if "left" in anchors else list(range(cc0 + 1, cc1))
    rights = [right_wall] if "right" in anchors else list(range(cc0 + 2, cc1))

    boxes = []
    for r0 in tops:
        for r1 in bottoms:
            if r1 - r0 + 1 < 5:
                continue
            inner_h = r1 - r0 - 1
            if inner_h < 3:
                continue
            for c0 in lefts:
                for c1 in rights:
                    if c1 - c0 + 1 < 5:
                        continue
                    inner_w = c1 - c0 - 1
                    if inner_w < 3:
                        continue
                    span = inner_w if gap_side in ("top", "bottom") else inner_h
                    if span < gap_len:
                        continue
                    boxes.append((r0, c0, r1, c1))
    return boxes


def _gap_candidates(bbox: tuple[int, int, int, int], side: str, gap_len: int):
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        span = c1 - c0 - 1
    else:
        span = r1 - r0 - 1
    gaps = []
    for start in range(span - gap_len + 1):
        end = start + gap_len - 1
        gaps.append(_gap_details(bbox, side, start, end)[0])
    return gaps


def _line_margin(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    vals = []
    for r, c in coords:
        vals.append(min(r - (r0 + 1), (r1 - 1) - r, c - (c0 + 1), (c1 - 1) - c))
    return min(vals)


def _key_candidates(bbox: tuple[int, int, int, int], gap_len: int, axis: str) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    coords_list: list[list[tuple[int, int]]] = []
    if axis == "h":
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1 - gap_len + 1):
                coords_list.append([(r, c + d) for d in range(gap_len)])
    else:
        for r in range(r0 + 1, r1 - gap_len + 1):
            for c in range(c0 + 1, c1):
                coords_list.append([(r + d, c) for d in range(gap_len)])
    good = [coords for coords in coords_list if _line_margin(coords, bbox) >= 1]
    return good if good else coords_list


def _attempt_build(rng: random.Random) -> np.ndarray | None:
    room_count = rng.randint(1, 4)
    layout = rng.choice(LAYOUTS[room_count])
    rows, cols = layout
    heights = _sample_sizes(rng, rows)
    widths = _sample_sizes(rng, cols)
    if heights is None or widths is None:
        return None

    h, w, cells, r_lines, c_lines = _cell_rects(heights, widths)
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for r in r_lines:
        grid[r, :] = WALL
    for c in c_lines:
        grid[:, c] = WALL

    chosen_cells = rng.sample(range(len(cells)), room_count)
    colors = rng.sample(KEY_COLORS, room_count)

    room_specs = []
    for cell_idx, color in zip(chosen_cells, colors):
        cell = cells[cell_idx]
        spec = None
        for _ in range(80):
            gap_side = rng.choice(SIDES)
            gap_len = rng.randint(1, 3)
            anchors = {side for side in SIDES if side != gap_side and rng.random() < 0.35}
            boxes = _enumerate_room_bboxes(cell, anchors, gap_side, gap_len)
            if not boxes:
                continue
            bbox = rng.choice(boxes)
            gap_coords = rng.choice(_gap_candidates(bbox, gap_side, gap_len))

            gap_axis = "h" if gap_side in ("top", "bottom") else "v"
            key_axis = gap_axis if gap_len == 1 else rng.choice(("h", "v"))
            keys = _key_candidates(bbox, gap_len, key_axis)
            if not keys:
                continue
            key_coords = rng.choice(keys)
            spec = {
                "bbox": bbox,
                "gap_coords": gap_coords,
                "gap_side": gap_side,
                "key_coords": key_coords,
                "color": color,
            }
            break
        if spec is None:
            return None
        room_specs.append(spec)

    for spec in room_specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        for r, c in spec["gap_coords"]:
            grid[r, c] = BG
        for r, c in spec["key_coords"]:
            grid[r, c] = spec["color"]

    try:
        rooms = _find_rooms(grid)
        if len(rooms) != room_count:
            return None
        out = main(grid)
    except AssertionError:
        return None

    for spec in room_specs:
        color = int(spec["color"])
        if int(np.count_nonzero(out == color)) <= len(spec["gap_coords"]):
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _attempt_build(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-key corridor puzzle")
