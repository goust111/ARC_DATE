from __future__ import annotations

from collections import deque
import random

import numpy as np


TOP = "top"
BOTTOM = "bottom"
LEFT = "left"
RIGHT = "right"
SIDES = (TOP, BOTTOM, LEFT, RIGHT)
SIDE_TO_VEC = {
    TOP: (-1, 0),
    BOTTOM: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
SIDE_TO_COLOR = {TOP: 1, BOTTOM: 2, LEFT: 3, RIGHT: 4}
BG = 0
OBJ = 5


BASE_SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
]


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _transform(shape: tuple[tuple[int, int], ...], rot: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = [(r, -c) if flip else (r, c) for r, c in shape]
    for _ in range(rot % 4):
        pts = [(-c, r) for r, c in pts]
    return _normalize(pts)


def _build_shape_pool() -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    out = []
    for base in BASE_SHAPES:
        for rot in range(4):
            for flip in (False, True):
                shp = _transform(base, rot, flip)
                if shp not in seen:
                    seen.add(shp)
                    out.append(shp)
    return out


SHAPES = _build_shape_pool()


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _place(shape: tuple[tuple[int, int], ...], top: int, left: int) -> np.ndarray:
    return np.array([(top + r, left + c) for r, c in shape], dtype=int)


def _move(coords: np.ndarray, dr: int, dc: int, steps: int = 1) -> np.ndarray:
    return coords + np.array([dr * steps, dc * steps], dtype=int)


def _inside(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.all(coords[:, 0] >= 0)
        and np.all(coords[:, 0] < h)
        and np.all(coords[:, 1] >= 0)
        and np.all(coords[:, 1] < w)
    )


def _coords_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _neighbors8_of_coords(coords: np.ndarray) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((rr + dr, cc + dc))
    return out


def _touching_ids(coords: np.ndarray, occ: np.ndarray) -> set[int]:
    ids: set[int] = set()
    own = {tuple(map(int, rc)) for rc in coords}
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = rr + dr
            nc = cc + dc
            if (nr, nc) in own:
                continue
            if 0 <= nr < occ.shape[0] and 0 <= nc < occ.shape[1] and occ[nr, nc] >= 0:
                ids.add(int(occ[nr, nc]))
    return ids


def _side_distances(coords: np.ndarray, inner_top: int, inner_bottom: int, inner_left: int, inner_right: int) -> dict[str, float]:
    r0, c0, r1, c1 = _bbox(coords)
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    return {
        TOP: cr - inner_top,
        BOTTOM: inner_bottom - cr,
        LEFT: cc - inner_left,
        RIGHT: inner_right - cc,
    }


def _nearest_side(coords: np.ndarray, inner_top: int, inner_bottom: int, inner_left: int, inner_right: int) -> tuple[str, float, dict[str, float]] | None:
    dists = _side_distances(coords, inner_top, inner_bottom, inner_left, inner_right)
    best = min(dists.values())
    winners = [side for side, dist in dists.items() if abs(dist - best) < 1e-9]
    if len(winners) != 1:
        return None
    return winners[0], best, dists


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _detect_frame(grid: np.ndarray) -> dict:
    mask = (grid != BG) & (grid != OBJ)
    coords = np.argwhere(mask)
    assert coords.size > 0, "frame not found"
    fr0, fc0, fr1, fc1 = _bbox(coords)
    assert fr1 - fr0 >= 3 and fc1 - fc0 >= 3, "frame too small"

    top_vals = set(map(int, grid[fr0, fc0 + 1 : fc1]))
    bottom_vals = set(map(int, grid[fr1, fc0 + 1 : fc1]))
    left_vals = set(map(int, grid[fr0 + 1 : fr1, fc0]))
    right_vals = set(map(int, grid[fr0 + 1 : fr1, fc1]))
    assert len(top_vals) == len(bottom_vals) == len(left_vals) == len(right_vals) == 1
    assert int(grid[fr0, fc0]) == BG and int(grid[fr0, fc1]) == BG
    assert int(grid[fr1, fc0]) == BG and int(grid[fr1, fc1]) == BG

    return {
        "bbox": (fr0, fc0, fr1, fc1),
        "inner": (fr0 + 1, fr1 - 1, fc0 + 1, fc1 - 1),
        "colors": {
            TOP: next(iter(top_vals)),
            BOTTOM: next(iter(bottom_vals)),
            LEFT: next(iter(left_vals)),
            RIGHT: next(iter(right_vals)),
        },
        "mask": mask,
    }


def _solve(grid: np.ndarray, trace: bool = False):
    g = np.asarray(grid, dtype=int)
    frame = _detect_frame(g)
    fr0, fc0, fr1, fc1 = frame["bbox"]
    inner_top, inner_bottom, inner_left, inner_right = frame["inner"]
    comps = _components_of_color(g, OBJ)
    objects = []
    for comp in comps:
        near = _nearest_side(comp, inner_top, inner_bottom, inner_left, inner_right)
        assert near is not None, "each object must have a unique nearest side"
        side, dist, dists = near
        objects.append(
            {
                "coords": comp.copy(),
                "bbox": _bbox(comp),
                "side": side,
                "dist": dist,
                "dists": dists,
                "color": int(frame["colors"][side]),
            }
        )

    objects.sort(
        key=lambda obj: (
            int(round(obj["dist"] * 2)),
            SIDES.index(obj["side"]),
            obj["bbox"][0],
            obj["bbox"][1],
            obj["bbox"][2],
            obj["bbox"][3],
        )
    )

    out = np.full_like(g, BG)
    out[frame["mask"]] = g[frame["mask"]]
    occ_kind = np.full(g.shape, -2, dtype=int)
    occ_kind[frame["mask"]] = -1
    settled = []

    for obj_idx, obj in enumerate(objects):
        dr, dc = SIDE_TO_VEC[obj["side"]]
        coords = obj["coords"].copy()
        stop_reason = "wall"
        blocker_ids: set[int] = set()
        while True:
            nxt = _move(coords, dr, dc)
            in_bounds = _inside(nxt, g.shape[0], g.shape[1])
            if not in_bounds:
                stop_reason = "wall"
                blocker_ids = set()
                break
            hits = occ_kind[nxt[:, 0], nxt[:, 1]]
            if np.any(hits != -2):
                block_vals = {int(v) for v in hits if int(v) != -2}
                if -1 in block_vals:
                    stop_reason = "wall"
                    blocker_ids = set()
                else:
                    stop_reason = "object"
                    blocker_ids = block_vals
                break
            coords = nxt

        out[coords[:, 0], coords[:, 1]] = obj["color"]
        occ_kind[coords[:, 0], coords[:, 1]] = obj_idx
        record = dict(obj)
        record["final_coords"] = coords
        record["stop_reason"] = stop_reason
        record["blocker_ids"] = blocker_ids
        settled.append(record)

    cropped = out[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()
    if not trace:
        return cropped
    return cropped, {"frame": frame, "objects": settled}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each gray shape toward its nearest frame side, stack it there, recolor it with that side's color, and crop to the frame.

    Input structure:
    - The background is black.
    - A hollow rectangular frame sits inside the grid; its four non-corner sides are distinct colors.
    - Several disconnected gray 4-connected shapes lie inside the frame.

    Rule:
    - Detect the frame and its four side colors.
    - For each gray shape, compute the Manhattan distance from its bbox center to the frame's inner top, bottom, left, and right sides; the unique nearest side is that shape's attraction side.
    - Settle shapes in increasing attraction-side distance. Each shape slides rigidly toward its own attraction side until the next step would hit the frame wall or an already settled shape.
    - Recolor each settled shape with the color of its attraction side, keep the frame unchanged, and crop the result to the frame bbox.

    Key insight:
    - The nearest side determines both movement direction and final repaint color, while near-wall-first settling makes previously moved shapes part of later collision geometry.
    """
    return _solve(grid, trace=False)


def _draw_frame(frame_h: int, frame_w: int) -> np.ndarray:
    g = np.zeros((frame_h, frame_w), dtype=int)
    g[0, 1:-1] = SIDE_TO_COLOR[TOP]
    g[-1, 1:-1] = SIDE_TO_COLOR[BOTTOM]
    g[1:-1, 0] = SIDE_TO_COLOR[LEFT]
    g[1:-1, -1] = SIDE_TO_COLOR[RIGHT]
    return g


def _candidate_wall_positions(shape: tuple[tuple[int, int], ...], side: str, h: int, w: int, occ: np.ndarray) -> list[np.ndarray]:
    shp_h = max(r for r, _ in shape) + 1
    shp_w = max(c for _, c in shape) + 1
    if side == TOP:
        tops = [0]
        lefts = range(0, w - shp_w + 1)
    elif side == BOTTOM:
        tops = [h - shp_h]
        lefts = range(0, w - shp_w + 1)
    elif side == LEFT:
        tops = range(0, h - shp_h + 1)
        lefts = [0]
    else:
        tops = range(0, h - shp_h + 1)
        lefts = [w - shp_w]

    out = []
    for top in tops:
        for left in lefts:
            coords = _place(shape, top, left)
            if np.any(occ[coords[:, 0], coords[:, 1]] >= 0):
                continue
            if _touching_ids(coords, occ):
                continue
            near = _nearest_side(coords, 0, h - 1, 0, w - 1)
            if near is None or near[0] != side:
                continue
            out.append(coords)
    random.shuffle(out)
    return out


def _candidate_blocked_positions(
    shape: tuple[tuple[int, int], ...],
    side: str,
    h: int,
    w: int,
    occ: np.ndarray,
    blocker_id: int,
) -> list[np.ndarray]:
    shp_h = max(r for r, _ in shape) + 1
    shp_w = max(c for _, c in shape) + 1
    dr, dc = SIDE_TO_VEC[side]
    out = []
    for top in range(0, h - shp_h + 1):
        for left in range(0, w - shp_w + 1):
            coords = _place(shape, top, left)
            if np.any(occ[coords[:, 0], coords[:, 1]] >= 0):
                continue
            near = _nearest_side(coords, 0, h - 1, 0, w - 1)
            if near is None or near[0] != side:
                continue
            nxt = _move(coords, dr, dc)
            if not _inside(nxt, h, w):
                continue
            hits = {int(v) for v in occ[nxt[:, 0], nxt[:, 1]] if int(v) >= 0}
            if hits != {blocker_id}:
                continue
            touching = _touching_ids(coords, occ)
            if touching - {blocker_id}:
                continue
            out.append(coords)
    random.shuffle(out)
    return out


def _all_shift_options(obj: dict, h: int, w: int) -> list[dict]:
    side = obj["side"]
    dr, dc = SIDE_TO_VEC[side]
    out = []
    for steps in range(1, max(h, w) + 1):
        coords = _move(obj["final_coords"], -dr, -dc, steps)
        if not _inside(coords, h, w):
            break
        near = _nearest_side(coords, 0, h - 1, 0, w - 1)
        if near is None or near[0] != side:
            continue
        dist = near[1]
        out.append(
            {
                "steps": steps,
                "coords": coords,
                "dist_key": int(round(dist * 2)),
                "dist": dist,
            }
        )
    random.shuffle(out)
    out.sort(key=lambda item: (item["dist_key"], item["steps"]))
    return out


def _choose_initial_positions(objects: list[dict], h: int, w: int, anchor_id: int, dependent_id: int) -> dict[int, dict] | None:
    options = {obj["id"]: _all_shift_options(obj, h, w) for obj in objects}
    if any(not vals for vals in options.values()):
        return None

    search_order = sorted(objects, key=lambda obj: (len(options[obj["id"]]), random.random()))
    chosen: dict[int, dict] = {}
    used_dist: set[int] = set()
    blocked8: set[tuple[int, int]] = set()

    def dfs(pos: int) -> bool:
        if pos == len(search_order):
            return chosen[anchor_id]["dist_key"] < chosen[dependent_id]["dist_key"]
        obj = search_order[pos]
        oid = obj["id"]
        for opt in options[oid]:
            if opt["dist_key"] in used_dist:
                continue
            coords = opt["coords"]
            if any((int(r), int(c)) in blocked8 for r, c in coords):
                continue
            chosen[oid] = opt
            used_dist.add(opt["dist_key"])
            added = _neighbors8_of_coords(coords)
            blocked8.update(added)
            if dfs(pos + 1):
                return True
            for cell in added:
                if cell in blocked8:
                    blocked8.remove(cell)
            used_dist.remove(opt["dist_key"])
            del chosen[oid]
            for prev_opt in chosen.values():
                blocked8.update(_neighbors8_of_coords(prev_opt["coords"]))
        return False

    if dfs(0):
        return chosen
    return None


def _build_side_list(count: int, anchor_side: str, dependent_side: str) -> list[str]:
    distinct_target = 4 if count >= 5 and random.random() < 0.7 else 3
    sides = [anchor_side, dependent_side]
    remaining_distinct = [side for side in SIDES if side not in sides]
    random.shuffle(remaining_distinct)
    while len(set(sides)) < distinct_target:
        sides.append(remaining_distinct.pop())
    while len(sides) < count:
        sides.append(random.choice(list(set(sides))))
    tail = sides[2:]
    random.shuffle(tail)
    return sides[:2] + tail


def _moves_toward_side(coords: np.ndarray, side: str, h: int, w: int) -> bool:
    dr, dc = SIDE_TO_VEC[side]
    return _inside(_move(coords, dr, dc), h, w)


def _random_object_for_side(
    side: str,
    h: int,
    w: int,
    blocked8: set[tuple[int, int]],
    *,
    require_move: bool = True,
) -> np.ndarray | None:
    for _ in range(220):
        shape = random.choice(SHAPES)
        shp_h = max(r for r, _ in shape) + 1
        shp_w = max(c for _, c in shape) + 1
        top = random.randint(0, h - shp_h)
        left = random.randint(0, w - shp_w)
        coords = _place(shape, top, left)
        if any((int(r), int(c)) in blocked8 for r, c in coords):
            continue
        near = _nearest_side(coords, 0, h - 1, 0, w - 1)
        if near is None or near[0] != side:
            continue
        if require_move and not _moves_toward_side(coords, side, h, w):
            continue
        return coords
    return None


def _assemble_input(inner_h: int, inner_w: int, objects: list[np.ndarray]) -> np.ndarray:
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    max_top = 30 - frame_h - 1
    max_left = 30 - frame_w - 1
    top_margin = random.randint(1, max_top)
    left_margin = random.randint(1, max_left)
    bottom_margin = random.randint(1, 30 - frame_h - top_margin)
    right_margin = random.randint(1, 30 - frame_w - left_margin)

    inp = np.zeros((top_margin + frame_h + bottom_margin, left_margin + frame_w + right_margin), dtype=int)
    frame = _draw_frame(frame_h, frame_w)
    inp[top_margin : top_margin + frame_h, left_margin : left_margin + frame_w] = frame
    for coords in objects:
        shifted = coords + np.array([top_margin + 1, left_margin + 1], dtype=int)
        inp[shifted[:, 0], shifted[:, 1]] = OBJ
    return inp


def _spans_overlap(a: np.ndarray, b: np.ndarray, axis: int) -> bool:
    a0 = int(a[:, axis].min())
    a1 = int(a[:, axis].max())
    b0 = int(b[:, axis].min())
    b1 = int(b[:, axis].max())
    return max(a0, b0) <= min(a1, b1)


def _canonical_collision_pair(anchor_side: str, dependent_side: str, h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    if anchor_side == TOP and dependent_side == LEFT:
        return (
            np.array([(0, 2), (1, 2), (2, 2)], dtype=int),
            np.array([(2, 3), (3, 3), (4, 3), (5, 3)], dtype=int),
        )
    if anchor_side == TOP and dependent_side == RIGHT:
        return (
            np.array([(0, w - 3), (1, w - 3), (2, w - 3)], dtype=int),
            np.array([(2, w - 4), (3, w - 4), (4, w - 4), (5, w - 4)], dtype=int),
        )
    if anchor_side == BOTTOM and dependent_side == LEFT:
        return (
            np.array([(h - 3, 2), (h - 2, 2), (h - 1, 2)], dtype=int),
            np.array([(h - 6, 3), (h - 5, 3), (h - 4, 3), (h - 3, 3)], dtype=int),
        )
    if anchor_side == BOTTOM and dependent_side == RIGHT:
        return (
            np.array([(h - 3, w - 3), (h - 2, w - 3), (h - 1, w - 3)], dtype=int),
            np.array([(h - 6, w - 4), (h - 5, w - 4), (h - 4, w - 4), (h - 3, w - 4)], dtype=int),
        )
    if anchor_side == LEFT and dependent_side == TOP:
        return (
            np.array([(2, 0), (2, 1), (2, 2)], dtype=int),
            np.array([(3, 2), (3, 3), (3, 4), (3, 5)], dtype=int),
        )
    if anchor_side == LEFT and dependent_side == BOTTOM:
        return (
            np.array([(h - 3, 0), (h - 3, 1), (h - 3, 2)], dtype=int),
            np.array([(h - 4, 2), (h - 4, 3), (h - 4, 4), (h - 4, 5)], dtype=int),
        )
    if anchor_side == RIGHT and dependent_side == TOP:
        return (
            np.array([(2, w - 3), (2, w - 2), (2, w - 1)], dtype=int),
            np.array([(3, w - 6), (3, w - 5), (3, w - 4), (3, w - 3)], dtype=int),
        )
    if anchor_side == RIGHT and dependent_side == BOTTOM:
        return (
            np.array([(h - 3, w - 3), (h - 3, w - 2), (h - 3, w - 1)], dtype=int),
            np.array([(h - 4, w - 6), (h - 4, w - 5), (h - 4, w - 4), (h - 4, w - 3)], dtype=int),
        )
    return None


def _make_layout() -> tuple[np.ndarray, np.ndarray] | None:
    inner_h = random.randint(13, 16)
    inner_w = random.randint(13, 16)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    if frame_h > 24 or frame_w > 24:
        return None

    stack_side = random.choice(SIDES)
    count = random.randint(4, 7)
    other_sides = [side for side in SIDES if side != stack_side]
    random.shuffle(other_sides)
    distinct_target = 4 if count >= 5 and random.random() < 0.7 else 3
    side_list = [stack_side, stack_side]
    while len(set(side_list)) < distinct_target:
        side_list.append(other_sides.pop())
    while len(side_list) < count:
        side_list.append(random.choice(list(set(side_list))))
    tail = side_list[2:]
    random.shuffle(tail)
    side_list = side_list[:2] + tail

    if stack_side == TOP:
        mid = random.randint(6, inner_w - 7)
        anchor_coords = np.array([(2, mid - 1), (2, mid), (2, mid + 1)], dtype=int)
        dependent_coords = np.array([(4, mid), (5, mid), (6, mid), (7, mid)], dtype=int)
    elif stack_side == BOTTOM:
        mid = random.randint(6, inner_w - 7)
        anchor_coords = np.array([(inner_h - 3, mid - 1), (inner_h - 3, mid), (inner_h - 3, mid + 1)], dtype=int)
        dependent_coords = np.array(
            [(inner_h - 8, mid), (inner_h - 7, mid), (inner_h - 6, mid), (inner_h - 5, mid)],
            dtype=int,
        )
    elif stack_side == LEFT:
        mid = random.randint(6, inner_h - 7)
        anchor_coords = np.array([(mid - 1, 2), (mid, 2), (mid + 1, 2)], dtype=int)
        dependent_coords = np.array([(mid, 4), (mid, 5), (mid, 6), (mid, 7)], dtype=int)
    else:
        mid = random.randint(6, inner_h - 7)
        anchor_coords = np.array([(mid - 1, inner_w - 3), (mid, inner_w - 3), (mid + 1, inner_w - 3)], dtype=int)
        dependent_coords = np.array(
            [(mid, inner_w - 8), (mid, inner_w - 7), (mid, inner_w - 6), (mid, inner_w - 5)],
            dtype=int,
        )

    for coords, side in ((anchor_coords, stack_side), (dependent_coords, stack_side)):
        near = _nearest_side(coords, 0, inner_h - 1, 0, inner_w - 1)
        if near is None or near[0] != side:
            return None
        if not _moves_toward_side(coords, side, inner_h, inner_w):
            return None

    objects = [anchor_coords, dependent_coords]
    blocked8 = set()
    for coords in objects:
        blocked8.update(_neighbors8_of_coords(coords))

    for side in side_list[2:]:
        coords = _random_object_for_side(side, inner_h, inner_w, blocked8, require_move=True)
        if coords is None:
            return None
        objects.append(coords)
        blocked8.update(_neighbors8_of_coords(coords))

    inp = _assemble_input(inner_h, inner_w, objects)
    out, meta = _solve(inp, trace=True)
    if len(meta["objects"]) != count:
        return None

    dkeys = [int(round(obj["dist"] * 2)) for obj in meta["objects"]]
    if len(set(dkeys)) != len(dkeys):
        return None
    side_set = {obj["side"] for obj in meta["objects"]}
    if len(side_set) < 3:
        return None
    blocker_ok = False
    for obj in meta["objects"]:
        if obj["stop_reason"] != "object":
            continue
        if obj["blocker_ids"]:
            blocker_ok = True
            break
    if not blocker_ok:
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(600):
        sample = _make_layout()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
