import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
WALL = 6
ORANGE = 7
TEAL = 8
MARKER = 9

FLOOR_COLORS = (BLUE, GREEN, YELLOW, ORANGE)

# A compact T-like stamp whose four rotations are distinct. With the required
# anchor placements, only the right/down copies overlap, so the generator keeps
# those two source regions the same color to make the result order-independent.
BASE_TEMPLATE = np.array(
    [
        [TEAL, BG],
        [TEAL, TEAL],
        [TEAL, BG],
    ],
    dtype=int,
)


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords):
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        comp = []
        q = deque([(r, c)])
        seen[r, c] = True
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(np.array(comp, dtype=int))
    return out


def _crop_component_mask(mask):
    comps = _components(mask)
    assert len(comps) == 1, "expected exactly one template component"
    r0, c0, r1, c1 = _bbox(comps[0])
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rot_cw(arr, quarter_turns):
    k = int(quarter_turns) % 4
    return np.rot90(arr, -k)


def _primary_direction(marker_rc, target_rc):
    mr, mc = marker_rc
    tr, tc = target_rc
    dr = int(mr - tr)
    dc = int(mc - tc)
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def _manhattan(a, b):
    return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def _flood_reachable(grid, target_rc):
    g = np.asarray(grid, dtype=int)
    visited = np.zeros_like(g, dtype=bool)
    tr, tc = target_rc
    q = deque([(int(tr), int(tc))])
    visited[tr, tc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(g.shape, nr, nc) or visited[nr, nc]:
                continue
            value = int(g[nr, nc])
            if value in (BG, RED, WALL, TEAL):
                continue
            visited[nr, nc] = True
            q.append((nr, nc))
    return visited


def _infer_floor_color(grid, marker_rc):
    g = np.asarray(grid, dtype=int)
    r, c = marker_rc
    local = []
    for nr, nc in _neighbors4(int(r), int(c)):
        if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) in FLOOR_COLORS:
            local.append(int(g[nr, nc]))
    if not local:
        for radius in (1, 2, 3):
            for nr in range(max(0, r - radius), min(g.shape[0], r + radius + 1)):
                for nc in range(max(0, c - radius), min(g.shape[1], c + radius + 1)):
                    if int(g[nr, nc]) in FLOOR_COLORS:
                        local.append(int(g[nr, nc]))
            if local:
                break
    assert local, "marker must touch some floor color"
    return int(Counter(local).most_common(1)[0][0])


def _analyze_grid(grid):
    g = np.asarray(grid, dtype=int)
    target_cells = np.argwhere(g == RED)
    assert len(target_cells) == 1, "expected exactly one target center"
    target_rc = tuple(map(int, target_cells[0]))

    template_mask = _crop_component_mask(g == TEAL)
    reachable_mask = _flood_reachable(g, target_rc)

    markers = []
    for rc in np.argwhere(g == MARKER):
        pos = tuple(map(int, rc))
        markers.append(
            {
                "pos": pos,
                "color": _infer_floor_color(g, pos),
                "dist": _manhattan(pos, target_rc),
                "dir": _primary_direction(pos, target_rc),
                "reachable": bool(reachable_mask[pos]),
            }
        )

    reachable_markers = [m for m in markers if m["reachable"]]
    reachable_markers.sort(key=lambda m: m["dist"])
    selected = reachable_markers[:4]
    selected_by_dir = {m["dir"]: m for m in selected}
    return {
        "target": target_rc,
        "template_mask": template_mask,
        "markers": markers,
        "reachable_markers": reachable_markers,
        "selected": selected,
        "selected_by_dir": selected_by_dir,
    }


def _place_patch(dst, patch, anchor_rc):
    ar, ac = anchor_rc
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            color = int(patch[r, c])
            if color == BG:
                continue
            rr = int(ar + r)
            cc = int(ac + c)
            if _in_bounds(dst.shape, rr, cc):
                dst[rr, cc] = color


def _render_output(grid, analysis):
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    out[g == TEAL] = BG
    for entry in analysis["selected"]:
        r, c = entry["pos"]
        out[r, c] = entry["color"]

    tr, tc = analysis["target"]
    anchors = {
        "left": (tr, tc - 1),
        "up": (tr - 1, tc),
        "right": (tr, tc + 1),
        "down": (tr + 1, tc),
    }
    rotations = {"left": 0, "up": 1, "right": 2, "down": 3}
    for direction in ("left", "up", "right", "down"):
        entry = analysis["selected_by_dir"][direction]
        mask = _rot_cw(analysis["template_mask"], rotations[direction])
        patch = np.where(mask, entry["color"], BG)
        _place_patch(out, patch, anchors[direction])

    out[tr, tc] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather the reachable markers around the target and stamp rotated template copies that preserve arrival direction.

    Input structure:
    - The grid has black background, one red target center, one teal template component, several maroon marker pixels, and magenta wall segments.
    - Non-wall floor cells use room colors chosen from blue/green/yellow/orange; markers sit on top of those floor regions.
    - At least four markers are flood-reachable from the target through floor cells and marker cells, while some others can be blocked by walls.

    Rule:
    - Flood from the red target with 4-connectivity through floor colors and maroon markers, but not through black background, magenta walls, the teal template, or the target itself.
    - Sort the reachable markers by Manhattan distance to the target and keep the nearest four.
    - For each of those four markers, derive its main direction relative to the target: compare vertical vs horizontal offset, and on ties use the row axis first.
    - Erase the four chosen markers back to their local floor color, erase the teal template to black, then stamp four recolored copies of the template at the target's left/up/right/down neighbor cells.
    - Use the original template orientation for the marker coming from the left, rotate it clockwise for the marker from above, rotate it 180 degrees for the marker from the right, and rotate it counterclockwise for the marker from below.
    - Keep the red target pixel unchanged; clip any out-of-bounds stamp cells away.

    Key insight:
    - The intermediate gather step does not discard provenance: each chosen marker contributes two independent signals, namely which side it approached from and which floor region it came from.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid, dtype=int), analysis)


def _room_interior(bbox):
    r0, c0, r1, c1 = bbox
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _deep_cells(bbox):
    r0, c0, r1, c1 = _room_interior(bbox)
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cells.append((r, c))
    return cells


def _paint_room(grid, bbox, color, door_side=None, door_pos=None):
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    if door_side is not None:
        rr, cc = door_pos
        grid[int(rr), int(cc)] = int(color)


def _room_candidates(bbox, target, direction, tie_only=False):
    out = []
    for cell in _deep_cells(bbox):
        if _primary_direction(cell, target) != direction:
            continue
        dr = abs(cell[0] - target[0])
        dc = abs(cell[1] - target[1])
        if tie_only and dr != dc:
            continue
        if not tie_only and min(dr, dc) == 0 and max(dr, dc) < 2:
            continue
        out.append(cell)
    return out


def _all_accessible_floor_cells(grid):
    cells = []
    for r, c in np.argwhere(np.isin(grid, FLOOR_COLORS)):
        cells.append((int(r), int(c)))
    return cells


def _stamp_template_in_corner(grid):
    h, w = BASE_TEMPLATE.shape
    H, W = grid.shape
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        anchor = (0, 0)
    elif corner == "tr":
        anchor = (0, W - w)
    elif corner == "bl":
        anchor = (H - h, 0)
    else:
        anchor = (H - h, W - w)
    _place_patch(grid, BASE_TEMPLATE, anchor)
    return {"corner": corner, "anchor": anchor}


def _disjoint(bboxes):
    for i, (ar0, ac0, ar1, ac1) in enumerate(bboxes):
        for br0, bc0, br1, bc1 in bboxes[i + 1 :]:
            if not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0):
                return False
    return True


def _choose_selected(room_bboxes, room_colors, target):
    directions = ("left", "up", "right", "down")
    tie_dir = random.choice(("up", "down")) if random.random() < 0.45 else None
    candidate_map = {}
    for direction in directions:
        bbox = room_bboxes[direction]
        tie_candidates = _room_candidates(bbox, target, direction, tie_only=True) if direction == tie_dir else []
        normal_candidates = _room_candidates(bbox, target, direction, tie_only=False)
        if tie_candidates:
            candidate_map[direction] = tie_candidates + [c for c in normal_candidates if c not in tie_candidates]
        else:
            candidate_map[direction] = normal_candidates
        if not candidate_map[direction]:
            return None

    for _ in range(250):
        used_dists = set()
        chosen = {}
        ok = True
        for direction in directions:
            candidates = candidate_map[direction][:]
            random.shuffle(candidates)
            pick = None
            for cell in candidates:
                dist = _manhattan(cell, target)
                if dist in used_dists:
                    continue
                pick = cell
                used_dists.add(dist)
                break
            if pick is None:
                ok = False
                break
            chosen[direction] = {
                "pos": pick,
                "color": room_colors[direction],
                "dist": _manhattan(pick, target),
                "dir": direction,
            }
        if ok:
            return chosen
    return None


def _choose_extra_reachable(grid, target, selected_positions, min_distance, count):
    occupied = set(selected_positions)
    temp = np.array(grid, dtype=int, copy=True)
    temp[target] = RED
    reachable_mask = _flood_reachable(temp, target)
    cells = []
    for cell in _all_accessible_floor_cells(grid):
        if cell in occupied:
            continue
        if not reachable_mask[cell]:
            continue
        if _manhattan(cell, target) <= min_distance:
            continue
        if not _infer_floor_color(np.where(grid == MARKER, BG, grid), cell) in FLOOR_COLORS:
            continue
        cells.append(cell)

    random.shuffle(cells)
    picked = []
    used_dists = {_manhattan(pos, target) for pos in selected_positions}
    for cell in cells:
        dist = _manhattan(cell, target)
        if dist in used_dists:
            continue
        if any(abs(cell[0] - pr) <= 1 and abs(cell[1] - pc) <= 1 for pr, pc in picked):
            continue
        picked.append(cell)
        used_dists.add(dist)
        if len(picked) == count:
            return picked
    return None


def _choose_unreachable_markers(bbox, target, selected_dists, count):
    cells = _deep_cells(bbox)
    cells.sort(key=lambda rc: _manhattan(rc, target))
    picked = []
    used_dists = set(selected_dists)

    first = None
    max_selected = max(selected_dists)
    for cell in cells:
        dist = _manhattan(cell, target)
        if dist >= max_selected or dist in used_dists:
            continue
        first = cell
        picked.append(cell)
        used_dists.add(dist)
        break
    if first is None:
        return None

    remaining = [cell for cell in cells if cell not in picked]
    random.shuffle(remaining)
    for cell in remaining:
        dist = _manhattan(cell, target)
        if dist in used_dists:
            continue
        picked.append(cell)
        used_dists.add(dist)
        if len(picked) == count:
            return picked
    return picked if len(picked) == count else None


def _expected_from_meta(grid, meta):
    analysis = {
        "target": meta["target"],
        "template_mask": BASE_TEMPLATE == TEAL,
        "selected": [
            {
                "pos": meta["selected"][direction]["pos"],
                "color": meta["selected"][direction]["color"],
                "dist": meta["selected"][direction]["dist"],
                "dir": direction,
            }
            for direction in ("left", "up", "right", "down")
        ],
        "selected_by_dir": {
            direction: {
                "pos": meta["selected"][direction]["pos"],
                "color": meta["selected"][direction]["color"],
                "dist": meta["selected"][direction]["dist"],
                "dir": direction,
            }
            for direction in ("left", "up", "right", "down")
        },
    }
    return _render_output(grid, analysis)


def _validate_attempt(grid, meta):
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    analysis = _analyze_grid(g)
    if len(analysis["markers"]) < 6 or len(analysis["markers"]) > 8:
        return False
    if len(analysis["reachable_markers"]) < 5:
        return False

    selected = analysis["selected"]
    if len(selected) != 4:
        return False
    if {m["dir"] for m in selected} != {"left", "up", "right", "down"}:
        return False
    if len({m["dist"] for m in analysis["reachable_markers"]}) != len(analysis["reachable_markers"]):
        return False

    for direction in ("left", "up", "right", "down"):
        got = analysis["selected_by_dir"][direction]
        want = meta["selected"][direction]
        if got["pos"] != want["pos"] or got["color"] != want["color"] or got["dist"] != want["dist"]:
            return False

    reachable_sorted = analysis["reachable_markers"]
    if reachable_sorted[4]["dist"] <= max(m["dist"] for m in selected):
        return False

    unreachable = [m for m in analysis["markers"] if not m["reachable"]]
    if not unreachable:
        return False
    if min(m["dist"] for m in unreachable) >= max(m["dist"] for m in selected):
        return False

    if analysis["selected_by_dir"]["right"]["color"] != analysis["selected_by_dir"]["down"]["color"]:
        return False

    expected = _expected_from_meta(g, meta)
    actual = main(g)
    if not np.array_equal(expected, actual):
        return False
    if np.array_equal(actual, g):
        return False
    if np.any(actual == TEAL):
        return False
    return True


def _gen_attempt():
    H = random.randint(27, 30)
    W = random.randint(27, 30)
    grid = np.zeros((H, W), dtype=int)

    arena = (3, 3, H - 4, W - 4)
    ar0, ac0, ar1, ac1 = arena

    palette = list(FLOOR_COLORS)
    random.shuffle(palette)
    hub_color = palette[0]
    left_color = palette[1]
    up_color = palette[2]
    right_down_color = palette[3]
    room_colors = {
        "left": left_color,
        "up": up_color,
        "right": right_down_color,
        "down": right_down_color,
    }

    grid[ar0 + 1 : ar1, ac0 + 1 : ac1] = hub_color
    grid[ar0, ac0 : ac1 + 1] = WALL
    grid[ar1, ac0 : ac1 + 1] = WALL
    grid[ar0 : ar1 + 1, ac0] = WALL
    grid[ar0 : ar1 + 1, ac1] = WALL

    if ar1 - ar0 < 20 or ac1 - ac0 < 20:
        return None

    tr = random.randint(ar0 + 10, ar1 - 8)
    tc = random.randint(ac0 + 10, ac1 - 8)

    left_bbox = (tr - 2, tc - 7, tr + 2, tc - 3)
    right_bbox = (tr - 2, tc + 3, tr + 2, tc + 7)
    up_bbox = (tr - 7, tc - 2, tr - 3, tc + 2)
    down_bbox = (tr + 3, tc - 2, tr + 7, tc + 2)
    trap_bbox = (tr - 2, tc - 3, tr, tc - 1)

    bboxes = [left_bbox, up_bbox, right_bbox, down_bbox]
    inside = all(
        ar0 + 1 <= r0 and ac0 + 1 <= c0 and r1 <= ar1 - 1 and c1 <= ac1 - 1
        for r0, c0, r1, c1 in bboxes
    )
    if not inside or not _disjoint(bboxes):
        return None

    room_bboxes = {"left": left_bbox, "up": up_bbox, "right": right_bbox, "down": down_bbox}

    _paint_room(grid, left_bbox, left_color, "east", (tr, left_bbox[3]))
    _paint_room(grid, right_bbox, right_down_color, "west", (tr, right_bbox[1]))
    _paint_room(grid, up_bbox, up_color, "south", (up_bbox[2], tc))
    _paint_room(grid, down_bbox, right_down_color, "north", (down_bbox[0], tc))

    # Open a second door on each selected room so the reachable area extends
    # beyond the four near rooms; this makes distance-sorting among reachable
    # markers necessary instead of vacuous.
    grid[tr, left_bbox[1]] = left_color
    grid[tr, right_bbox[3]] = right_down_color
    grid[up_bbox[0], tc] = up_color
    grid[down_bbox[2], tc] = right_down_color

    # A small sealed pocket near the target provides a closer-but-unreachable marker.
    _paint_room(grid, trap_bbox, hub_color, None, None)

    selected = _choose_selected(room_bboxes, room_colors, (tr, tc))
    if selected is None:
        return None

    selected_positions = [entry["pos"] for entry in selected.values()]
    selected_dists = [entry["dist"] for entry in selected.values()]

    extra_reach_count = random.randint(1, 3)
    extra_reachable = _choose_extra_reachable(grid, (tr, tc), selected_positions, max(selected_dists), extra_reach_count)
    if extra_reachable is None:
        return None

    trap_count = 1
    trap_markers = _choose_unreachable_markers(trap_bbox, (tr, tc), selected_dists + [_manhattan(cell, (tr, tc)) for cell in extra_reachable], trap_count)
    if trap_markers is None:
        return None

    for entry in selected.values():
        r, c = entry["pos"]
        grid[r, c] = MARKER

    for cell in extra_reachable:
        grid[cell] = MARKER

    for cell in trap_markers:
        grid[cell] = MARKER

    grid[tr, tc] = RED
    _stamp_template_in_corner(grid)

    meta = {
        "target": (tr, tc),
        "selected": selected,
        "extra_reachable": extra_reachable,
        "trap_markers": trap_markers,
    }
    if _validate_attempt(grid, meta):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid puzzle after many attempts")
