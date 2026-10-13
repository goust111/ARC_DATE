from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("up", "right", "down", "left")
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
OPPOSITE = {
    "up": "down",
    "right": "left",
    "down": "up",
    "left": "right",
}
SIDE_FROM_RAY = {
    "up": "down",
    "right": "left",
    "down": "up",
    "left": "right",
}
BASE_CENTERS = [2, 8, 14, 20, 26]
SIZE_OPTIONS = [3, 5]
GEN_RETRIES = 400
_GEN_COUNTER = 0


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int = BLACK,
    colors: list[int] | None = None,
) -> list[dict]:
    """Monochromatic 4-connected components using arc_utils' (r,c) / bbox conventions."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if colors is None:
        colors = sorted(int(v) for v in np.unique(g) if int(v) != int(bg))
    out: list[dict] = []
    for color in colors:
        seen = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if seen[r, c] or int(g[r, c]) != int(color):
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while q:
                    rr, cc = q.popleft()
                    coords.append((rr, cc))
                    for nr, nc in neighbors4(rr, cc):
                        if not in_bounds((h, w), nr, nc):
                            continue
                        if seen[nr, nc] or int(g[nr, nc]) != int(color):
                            continue
                        seen[nr, nc] = True
                        q.append((nr, nc))
                arr = np.array(coords, dtype=int)
                out.append(
                    {
                        "color": int(color),
                        "coords": arr,
                        "bbox": bbox_coords(arr),
                        "area": int(arr.shape[0]),
                    }
                )
    return out


def _is_hollow_rectangle(grid: np.ndarray, comp: dict) -> bool:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        return False
    expected = set()
    for rr in range(r0, r1 + 1):
        expected.add((rr, c0))
        expected.add((rr, c1))
    for cc in range(c0, c1 + 1):
        expected.add((r0, cc))
        expected.add((r1, cc))
    actual = {tuple(map(int, p)) for p in coords}
    if actual != expected:
        return False
    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK))


def _rect_midpoint(rect: dict, side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = rect["bbox"]
    if side == "up":
        return r0, (c0 + c1) // 2
    if side == "down":
        return r1, (c0 + c1) // 2
    if side == "left":
        return (r0 + r1) // 2, c0
    return (r0 + r1) // 2, c1


def _first_hit(rects: list[dict], origin: tuple[int, int], ray_side: str) -> int | None:
    r, c = origin
    best_idx = None
    best_dist = None
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        dist = None
        if ray_side == "right" and r0 <= r <= r1 and c < c0:
            dist = c0 - c
        elif ray_side == "left" and r0 <= r <= r1 and c > c1:
            dist = c - c1
        elif ray_side == "down" and c0 <= c <= c1 and r < r0:
            dist = r0 - r
        elif ray_side == "up" and c0 <= c <= c1 and r > r1:
            dist = r - r1
        if dist is None:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _simulate_rays(rects: list[dict], seed: tuple[int, int], seed_ray_side: str, seed_color: int) -> dict[int, tuple[int, str | None, str]]:
    infections: dict[int, tuple[int, str | None, str]] = {}
    q = deque([(seed, seed_ray_side, int(seed_color), None)])
    while q:
        origin, ray_side, ray_color, parent_id = q.popleft()
        hit_idx = _first_hit(rects, origin, ray_side)
        if hit_idx is None:
            continue
        if hit_idx in infections:
            continue
        incoming_side = SIDE_FROM_RAY[ray_side]
        infections[hit_idx] = (int(ray_color), parent_id, incoming_side)
        rect = rects[hit_idx]
        for side in SIDES:
            if side == incoming_side:
                continue
            q.append((_rect_midpoint(rect, side), side, int(rect["color"]), hit_idx))
    return infections


def _parse_objects(grid: np.ndarray) -> tuple[list[dict], tuple[int, int], tuple[int, int], int]:
    comps = find_components(grid, bg=BLACK)
    rectangles: list[dict] = []
    grey_singletons: list[tuple[int, int]] = []
    color_singletons: list[tuple[int, int, int]] = []

    for comp in comps:
        color = int(comp["color"])
        if _is_hollow_rectangle(grid, comp):
            rectangles.append(
                {
                    "color": color,
                    "bbox": tuple(map(int, comp["bbox"])),
                }
            )
            continue
        if int(comp["area"]) != 1:
            raise AssertionError("expected only hollow rectangles plus a seed pair")
        r, c = map(int, comp["coords"][0])
        if color == GREY:
            grey_singletons.append((r, c))
        else:
            color_singletons.append((r, c, color))

    assert len(grey_singletons) == 1, "expected exactly one grey direction marker"
    marker = grey_singletons[0]
    assert len(color_singletons) == 1, "expected exactly one colored singleton seed"
    sr, sc, seed_color = color_singletons[0]
    assert abs(sr - marker[0]) + abs(sc - marker[1]) == 1, "seed must touch the grey marker"
    return rectangles, (sr, sc), marker, int(seed_color)


def _fill_output(grid: np.ndarray, rects: list[dict], infections: dict[int, tuple[int, str | None, str]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for idx, (fill_color, _parent, _incoming) in infections.items():
        r0, c0, r1, c1 = rects[idx]["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(fill_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate invisible infection rays through hollow rectangles so each hit interior records its parent ray color.

    Input structure:
    - A black background contains several non-touching hollow rectangles with distinct border colors and odd side lengths.
    - One seed consists of a colored singleton adjacent to a grey singleton that points along the initial ray direction.
    - Rectangle interiors are black in the input.

    Rule:
    - The seed emits a ray in the direction indicated by the grey pixel.
    - When a ray first hits a rectangle border, fill that rectangle's interior with the ray color.
    - Then emit new rays from the outward midpoints of the other three sides, using the rectangle's own border color.
    - Continue until every ray exits the grid or misses; keep the original borders, seed, and marker unchanged.

    Key insight:
    - A rectangle's interior color is the color of the ray that infected it, so solving requires tracing the full directed cascade rather than recoloring rectangles independently.
    """
    g = np.asarray(grid, dtype=int)
    rects, seed, marker, seed_color = _parse_objects(g)
    seed_ray_side = _seed_ray_side(seed, marker)
    infections = _simulate_rays(rects, seed, seed_ray_side, seed_color)
    return _fill_output(g, rects, infections)


def _seed_ray_side(seed: tuple[int, int], marker: tuple[int, int]) -> str:
    sr, sc = seed
    mr, mc = marker
    if mr == sr - 1 and mc == sc:
        return "up"
    if mr == sr + 1 and mc == sc:
        return "down"
    if mr == sr and mc == sc - 1:
        return "left"
    if mr == sr and mc == sc + 1:
        return "right"
    raise AssertionError("seed and marker must be adjacent")


def _slot_side(src: tuple[int, int], dst: tuple[int, int]) -> str | None:
    sr, sc = src
    tr, tc = dst
    if sr == tr and tc > sc:
        return "right"
    if sr == tr and tc < sc:
        return "left"
    if sc == tc and tr > sr:
        return "down"
    if sc == tc and tr < sr:
        return "up"
    return None


def _slot_distance(src: tuple[int, int], dst: tuple[int, int], side: str) -> int | None:
    rel = _slot_side(src, dst)
    if rel != side:
        return None
    sr, sc = src
    tr, tc = dst
    return abs(tr - sr) + abs(tc - sc)


def _nearest_visible_slot(
    occupied_slots: list[tuple[int, int]],
    src: tuple[int, int],
    side: str,
) -> tuple[tuple[int, int], int] | None:
    best_slot = None
    best_dist = None
    for slot in occupied_slots:
        if slot == src:
            continue
        dist = _slot_distance(src, slot, side)
        if dist is None:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_slot = slot
    if best_slot is None:
        return None
    return best_slot, int(best_dist)


def _free_slots_in_direction(
    slot: tuple[int, int],
    side: str,
    occupied: set[tuple[int, int]],
    n_rows: int,
    n_cols: int,
) -> list[tuple[int, int]]:
    sr, sc = slot
    dr, dc = DIRS[side]
    out = []
    step = 1
    while True:
        rr = sr + dr * step
        cc = sc + dc * step
        if not (0 <= rr < n_rows and 0 <= cc < n_cols):
            break
        nxt = (rr, cc)
        if nxt in occupied:
            break
        out.append(nxt)
        step += 1
    return out


def _candidate_respects_visibility(
    candidate: tuple[int, int],
    *,
    nodes: list[dict],
    occupied: set[tuple[int, int]],
    parent_id: int,
    parent_side: str,
    root_slot: tuple[int, int],
    root_incoming_side: str,
) -> bool:
    root_seed_side = OPPOSITE[root_incoming_side]
    cand_sr, cand_sc = candidate
    root_sr, root_sc = root_slot
    if root_seed_side == "right" and cand_sr == root_sr and cand_sc < root_sc:
        return False
    if root_seed_side == "left" and cand_sr == root_sr and cand_sc > root_sc:
        return False
    if root_seed_side == "down" and cand_sc == root_sc and cand_sr < root_sr:
        return False
    if root_seed_side == "up" and cand_sc == root_sc and cand_sr > root_sr:
        return False

    occupied_slots = list(occupied)
    for node_id, node in enumerate(nodes):
        src = node["slot"]
        for side in SIDES:
            if side == node["incoming_side"]:
                continue
            dist = _slot_distance(src, candidate, side)
            if dist is None:
                continue
            nearest = _nearest_visible_slot(occupied_slots, src, side)
            is_first = nearest is None or dist < nearest[1]
            if not is_first:
                continue
            if node_id == parent_id and side == parent_side:
                continue
            return False
    return True


def _centers_for_limit(limit: int) -> list[int]:
    max_shift = 1 if limit >= 30 else 0
    shift = random.randint(0, max_shift)
    centers = [c + shift for c in BASE_CENTERS if c + shift + 2 < limit]
    assert len(centers) >= 4
    return centers


def _build_tree(mode: int, n_rows: int, n_cols: int) -> tuple[list[dict], tuple[int, int], str] | None:
    root_incoming_side = random.choice(list(SIDES))
    if root_incoming_side == "left":
        root_slots = [(r, c) for r in range(n_rows) for c in range(1, n_cols)]
    elif root_incoming_side == "right":
        root_slots = [(r, c) for r in range(n_rows) for c in range(0, n_cols - 1)]
    elif root_incoming_side == "up":
        root_slots = [(r, c) for r in range(1, n_rows) for c in range(n_cols)]
    else:
        root_slots = [(r, c) for r in range(0, n_rows - 1) for c in range(n_cols)]
    root_slot = random.choice(root_slots)

    profiles = {
        0: {"infected": random.randint(2, 3), "extra": random.randint(1, 2), "needs_branch": False},
        1: {"infected": random.randint(3, 4), "extra": random.randint(0, 1), "needs_branch": True},
        2: {"infected": random.randint(4, 5), "extra": random.randint(0, 1), "needs_branch": True},
        3: {"infected": random.randint(3, 5), "extra": random.randint(1, 2), "needs_branch": True},
    }
    profile = profiles[mode]

    nodes = [
        {
            "slot": root_slot,
            "incoming_side": root_incoming_side,
            "children": {},
            "color": None,
        }
    ]
    occupied = {root_slot}
    while len(nodes) < profile["infected"]:
        options: list[tuple[int, int, str, tuple[int, int]]] = []
        for node_id, node in enumerate(nodes):
            open_sides = [side for side in SIDES if side != node["incoming_side"] and side not in node["children"]]
            for side in open_sides:
                for slot in _free_slots_in_direction(node["slot"], side, occupied, n_rows, n_cols):
                    if not _candidate_respects_visibility(
                        slot,
                        nodes=nodes,
                        occupied=occupied,
                        parent_id=node_id,
                        parent_side=side,
                        root_slot=root_slot,
                        root_incoming_side=root_incoming_side,
                    ):
                        continue
                    options.append((node_id, len(node["children"]), side, slot))
        if not options:
            return None

        def weight(item: tuple[int, int, str, tuple[int, int]]) -> int:
            node_id, child_count, _side, _slot = item
            score = 1
            if mode == 0 and node_id == len(nodes) - 1:
                score += 6
            if profile["needs_branch"] and node_id == 0 and child_count < 2:
                score += 7
            if mode == 2 and node_id < len(nodes) - 1:
                score += 2
            if mode == 3 and child_count == 0:
                score += 3
            return score

        weights = [weight(item) for item in options]
        node_id, _child_count, side, slot = random.choices(options, weights=weights, k=1)[0]
        child_id = len(nodes)
        nodes[node_id]["children"][side] = child_id
        nodes.append(
            {
                "slot": slot,
                "incoming_side": OPPOSITE[side],
                "children": {},
                "color": None,
            }
        )
        occupied.add(slot)

    branch_count = sum(1 for node in nodes if len(node["children"]) >= 2)
    if profile["needs_branch"] and branch_count == 0:
        return None
    nodes[0]["extra_count"] = profile["extra"]
    return nodes, root_slot, root_incoming_side


def _choose_extra_slots(nodes: list[dict], n_rows: int, n_cols: int) -> list[tuple[int, int]] | None:
    occupied = {node["slot"] for node in nodes}
    root_slot = nodes[0]["slot"]
    root_incoming_side = nodes[0]["incoming_side"]
    extra_count = int(nodes[0]["extra_count"])
    choices: list[tuple[int, int]] = []
    all_slots = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    random.shuffle(all_slots)
    for slot in all_slots:
        if slot in occupied:
            continue
        if not _candidate_respects_visibility(
            slot,
            nodes=nodes,
            occupied=occupied,
            parent_id=-1,
            parent_side="up",
            root_slot=root_slot,
            root_incoming_side=root_incoming_side,
        ):
            continue
        choices.append(slot)
        occupied.add(slot)
        if len(choices) == extra_count:
            return choices
    if extra_count == 0:
        return []
    return None


def _assign_sizes_and_colors(
    nodes: list[dict],
    extra_slots: list[tuple[int, int]],
    row_centers: list[int],
    col_centers: list[int],
) -> tuple[list[dict], int] | None:
    total_rects = len(nodes) + len(extra_slots)
    if total_rects + 1 > len(PALETTE):
        return None
    colors = random.sample(PALETTE, total_rects + 1)
    seed_color = int(colors[0])
    border_colors = colors[1:]
    rects: list[dict] = []

    slots = [node["slot"] for node in nodes] + extra_slots
    for idx, slot in enumerate(slots):
        h = random.choice(SIZE_OPTIONS)
        w = random.choice(SIZE_OPTIONS)
        center_r = row_centers[slot[0]]
        center_c = col_centers[slot[1]]
        rects.append(
            {
                "id": idx,
                "slot": slot,
                "center": (center_r, center_c),
                "bbox": (
                    center_r - h // 2,
                    center_c - w // 2,
                    center_r + h // 2,
                    center_c + w // 2,
                ),
                "color": int(border_colors[idx]),
                "is_tree": idx < len(nodes),
            }
        )
    return rects, seed_color


def _place_seed(
    root_rect: dict,
    root_incoming_side: str,
    h: int,
    w: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r0, c0, r1, c1 = root_rect["bbox"]
    rr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    if root_incoming_side == "left":
        options = []
        for gap in (3, 2):
            seed = (rr, c0 - gap)
            marker = (rr, seed[1] + 1)
            if 0 <= seed[1] and marker[1] < c0:
                options.append((seed, marker))
        return random.choice(options) if options else None
    if root_incoming_side == "right":
        options = []
        for gap in (3, 2):
            seed = (rr, c1 + gap)
            marker = (rr, seed[1] - 1)
            if seed[1] < w and marker[1] > c1:
                options.append((seed, marker))
        return random.choice(options) if options else None
    if root_incoming_side == "up":
        options = []
        for gap in (3, 2):
            seed = (r0 - gap, cc)
            marker = (seed[0] + 1, cc)
            if 0 <= seed[0] and marker[0] < r0:
                options.append((seed, marker))
        return random.choice(options) if options else None
    options = []
    for gap in (3, 2):
        seed = (r1 + gap, cc)
        marker = (seed[0] - 1, cc)
        if seed[0] < h and marker[0] > r1:
            options.append((seed, marker))
    return random.choice(options) if options else None


def _draw_rectangles(h: int, w: int, rects: list[dict], seed: tuple[int, int], marker: tuple[int, int], seed_color: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        color = int(rect["color"])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
    grid[seed] = int(seed_color)
    grid[marker] = GREY
    return grid


def _planned_infections(nodes: list[dict], rects: list[dict], seed_color: int) -> dict[int, tuple[int, str | None, str]]:
    slot_to_rect = {rect["slot"]: rect for rect in rects}
    out: dict[int, tuple[int, str | None, str]] = {}
    q = deque([(0, int(seed_color), None, nodes[0]["incoming_side"])])
    while q:
        node_id, color, parent_id, incoming_side = q.popleft()
        rect = slot_to_rect[nodes[node_id]["slot"]]
        out[rect["id"]] = (int(color), parent_id, incoming_side)
        for side, child_id in nodes[node_id]["children"].items():
            child_rect = slot_to_rect[nodes[child_id]["slot"]]
            q.append((child_id, int(rect["color"]), rect["id"], OPPOSITE[side]))
    assert len(out) == len(nodes)
    return out


def _same_plan(a: dict[int, tuple[int, str | None, str]], b: dict[int, tuple[int, str | None, str]]) -> bool:
    if set(a) != set(b):
        return False
    for key in a:
        if a[key] != b[key]:
            return False
    return True


def _gen_attempt(mode: int) -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(23, 30)
    w = random.randint(23, 30)
    row_centers = _centers_for_limit(h)
    col_centers = _centers_for_limit(w)

    built = _build_tree(mode, len(row_centers), len(col_centers))
    if built is None:
        return None
    nodes, _root_slot, root_incoming_side = built
    extra_slots = _choose_extra_slots(nodes, len(row_centers), len(col_centers))
    if extra_slots is None:
        return None
    assigned = _assign_sizes_and_colors(nodes, extra_slots, row_centers, col_centers)
    if assigned is None:
        return None
    rects, seed_color = assigned
    root_rect = rects[0]
    seed_pair = _place_seed(root_rect, root_incoming_side, h, w)
    if seed_pair is None:
        return None
    seed, marker = seed_pair
    grid = _draw_rectangles(h, w, rects, seed, marker, seed_color)

    expected_plan = _planned_infections(nodes, rects, seed_color)
    actual_plan = _simulate_rays(rects, seed, OPPOSITE[root_incoming_side], seed_color)
    if not _same_plan(expected_plan, actual_plan):
        return None

    expected = _fill_output(grid, rects, expected_plan)
    if not np.array_equal(main(grid), expected):
        return None

    untouched_ids = {rect["id"] for rect in rects[len(nodes) :]}
    if any(rect_id in actual_plan for rect_id in untouched_ids):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(GEN_RETRIES):
        mode = _GEN_COUNTER % 4
        _GEN_COUNTER += 1
        generated = _gen_attempt(mode)
        if generated is None:
            continue
        grid, _expected = generated
        return grid
    raise RuntimeError("failed to generate a valid chain-infection ARC sample")
