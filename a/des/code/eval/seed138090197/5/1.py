from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

# --------------- shared helpers ---------------


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_positions(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    out = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c))
        out.add((r1, c))
    for r in range(r0, r1 + 1):
        out.add((r, c0))
        out.add((r, c1))
    return out


def _door_side(frame: dict) -> str:
    r0, c0, r1, c1 = frame["bbox"]
    dr, dc = frame["door"]
    if dr == r0:
        return "top"
    if dr == r1:
        return "bottom"
    if dc == c0:
        return "left"
    if dc == c1:
        return "right"
    raise ValueError("door must lie on the frame perimeter")


def _inward_cell(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    dr, dc = frame["door"]
    side = _door_side(frame)
    if side == "top":
        return r0 + 1, dc
    if side == "bottom":
        return r1 - 1, dc
    if side == "left":
        return dr, c0 + 1
    return dr, c1 - 1


def _frame_masks(shape: tuple[int, int], frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    wall_mask = np.zeros(shape, dtype=bool)
    door_mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        door = frame["door"]
        for cell in _perimeter_positions(r0, c0, r1, c1):
            if cell == door:
                door_mask[cell] = True
            else:
                wall_mask[cell] = True
    return wall_mask, door_mask


def _label_rooms(shape: tuple[int, int], wall_mask: np.ndarray, door_mask: np.ndarray) -> np.ndarray:
    h, w = shape
    passable = ~(wall_mask | door_mask)
    room_id = np.full((h, w), -1, dtype=int)
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or room_id[r, c] >= 0:
                continue
            room_id[r, c] = next_id
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and room_id[nr, nc] < 0:
                        room_id[nr, nc] = next_id
                        stack.append((nr, nc))
            next_id += 1
    return room_id


def _room_graph(room_id: np.ndarray, door_mask: np.ndarray) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {}
    for rid in np.unique(room_id):
        rid = int(rid)
        if rid >= 0:
            graph[rid] = set()
    for r, c in np.argwhere(door_mask):
        touching = {
            int(room_id[nr, nc])
            for nr, nc in _neighbors4(int(r), int(c))
            if 0 <= nr < room_id.shape[0] and 0 <= nc < room_id.shape[1] and int(room_id[nr, nc]) >= 0
        }
        for a in touching:
            graph.setdefault(a, set()).update(touching - {a})
    return graph


def _reachable(graph: dict[int, set[int]], start: int, target: int) -> bool:
    if start == target:
        return True
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in graph.get(cur, ()):
            if nxt == target:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _outside_room_id(room_id: np.ndarray) -> int:
    boundary = set()
    h, w = room_id.shape
    for c in range(w):
        if room_id[0, c] >= 0:
            boundary.add(int(room_id[0, c]))
        if room_id[h - 1, c] >= 0:
            boundary.add(int(room_id[h - 1, c]))
    for r in range(h):
        if room_id[r, 0] >= 0:
            boundary.add(int(room_id[r, 0]))
        if room_id[r, w - 1] >= 0:
            boundary.add(int(room_id[r, w - 1]))
    assert len(boundary) == 1, "expected a unique outside room"
    return next(iter(boundary))


def _orthogonal_clear(grid: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = grid.shape
    own = {(r, c) for r in range(top, top + height) for c in range(left, left + width)}
    for r, c in own:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in own and int(grid[nr, nc]) != BG:
                return False
    return True


def _parse_scene(
    grid: np.ndarray,
) -> tuple[list[dict], list[dict], list[dict], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    locks: list[dict] = []
    wall_mask = np.zeros(g.shape, dtype=bool)
    lock_mask = np.zeros(g.shape, dtype=bool)

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            area = len(comp)

            if h == 2 and w == 2 and area == 4 and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
                locks.append({"color": color, "pos": (r0, c0)})
                lock_mask[r0 : r1 + 1, c0 : c1 + 1] = True
                continue

            if h < 3 or w < 3:
                continue

            comp_set = set(comp)
            perimeter = _perimeter_positions(r0, c0, r1, c1)
            missing = perimeter - comp_set
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            if comp_set <= perimeter and len(comp_set) == len(perimeter) - 1 and len(missing) == 1:
                door = next(iter(missing))
                if door not in corners:
                    frames.append({"color": color, "bbox": (r0, c0, r1, c1), "door": door})
                    for r, c in comp:
                        wall_mask[r, c] = True

    remaining = (g != BG) & ~wall_mask & ~lock_mask
    keys: list[dict] = []
    for comp in _components(remaining):
        if len(comp) != 4:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        outer = int(patch[0, 0])
        inner = int(patch[1, 1])
        if outer == BG or inner == BG or outer == inner:
            continue
        if int(patch[0, 1]) == outer and int(patch[1, 0]) == outer:
            keys.append({"outer": outer, "inner": inner, "pos": (r0, c0)})

    return frames, locks, keys, wall_mask, lock_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each 2x2 key by its outer color to a same-colored 2x2 lock, move reachable keys onto the
    lock, and fill the lock's room with the key's inner color. Room filling treats door cells as
    boundaries, while reachability treats them as graph edges between rooms.
    """
    g = np.asarray(grid, dtype=int)
    frames, locks, keys, wall_mask, _ = _parse_scene(g)
    if not frames or not locks or not keys:
        return g.copy()

    _, door_mask = _frame_masks(g.shape, frames)
    room_id = _label_rooms(g.shape, wall_mask, door_mask)
    graph = _room_graph(room_id, door_mask)
    out = g.copy()

    locks_by_color: dict[int, list[dict]] = {}
    for lock in locks:
        locks_by_color.setdefault(int(lock["color"]), []).append(lock)

    keys_by_color: dict[int, list[dict]] = {}
    for key in keys:
        keys_by_color.setdefault(int(key["outer"]), []).append(key)

    for color, key_list in keys_by_color.items():
        lock_list = locks_by_color.get(color, [])
        if len(key_list) != 1 or len(lock_list) != 1:
            continue

        key = key_list[0]
        lock = lock_list[0]
        kr, kc = key["pos"]
        lr, lc = lock["pos"]
        start_room = int(room_id[kr, kc])
        target_room = int(room_id[lr, lc])
        if start_room < 0 or target_room < 0 or not _reachable(graph, start_room, target_room):
            continue

        out[kr : kr + 2, kc : kc + 2] = BG
        out[lr : lr + 2, lc : lc + 2] = np.array(
            [[color, color], [color, int(key["inner"])]],
            dtype=int,
        )
        fill_mask = (room_id == target_room) & (out == BG)
        out[fill_mask] = int(key["inner"])

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
GRID_H_BOUNDS = (24, 34)
GRID_W_BOUNDS = (24, 34)
CHAIN_DEPTH_BOUNDS = (2, 3)
INNER_SIZE_BOUNDS = (6, 8)
CHAIN_MARGIN_BOUNDS = (3, 4)
EXTRA_FRAME_PROB = 0.55
EXTRA_SIZE_BOUNDS = (8, 11)
EDGE_MARGIN_BOUNDS = (2, 5)
FRAME_SEPARATION = 3
PLACEMENT_TRIES = 80


def _sample_door(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return r0, random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    return random.randint(r0 + 1, r1 - 1), c1


def _build_chain_node(depth: int) -> dict:
    node: dict = {
        "h": random.randint(*INNER_SIZE_BOUNDS),
        "w": random.randint(*INNER_SIZE_BOUNDS),
    }
    for _ in range(depth - 1):
        mt = random.randint(*CHAIN_MARGIN_BOUNDS)
        mb = random.randint(*CHAIN_MARGIN_BOUNDS)
        ml = random.randint(*CHAIN_MARGIN_BOUNDS)
        mr = random.randint(*CHAIN_MARGIN_BOUNDS)
        node = {
            "h": int(node["h"]) + mt + mb + 2,
            "w": int(node["w"]) + ml + mr + 2,
            "child": node,
            "child_offset": (1 + mt, 1 + ml),
        }
    return node


def _flatten_chain(node: dict, top: int, left: int, frames: list[dict], parent: int | None = None) -> int:
    idx = len(frames)
    bbox = (top, left, top + int(node["h"]) - 1, left + int(node["w"]) - 1)
    frames.append({"bbox": bbox, "parent": parent, "children": []})
    if "child" in node:
        off_r, off_c = node["child_offset"]
        child_idx = _flatten_chain(node["child"], top + off_r, left + off_c, frames, idx)
        frames[idx]["children"].append(child_idx)
    return idx


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = int(frame["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    dr, dc = frame["door"]
    grid[dr, dc] = BG


def _lock_candidates(grid: np.ndarray, room_id: np.ndarray, frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    seed_r, seed_c = _inward_cell(frame)
    target_room = int(room_id[seed_r, seed_c])
    out: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if r + 1 >= r1 or c + 1 >= c1:
                continue
            if r < r0 + 2 or r + 1 > r1 - 2 or c < c0 + 2 or c + 1 > c1 - 2:
                continue
            if not np.all(grid[r : r + 2, c : c + 2] == BG):
                continue
            if np.all(room_id[r : r + 2, c : c + 2] == target_room):
                out.append((r, c))
    random.shuffle(out)
    return out


def _key_candidates(grid: np.ndarray, room_id: np.ndarray, outside_room: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if not np.all(grid[r : r + 2, c : c + 2] == BG):
                continue
            if not np.all(room_id[r : r + 2, c : c + 2] == outside_room):
                continue
            if _orthogonal_clear(grid, r, c, 2, 2):
                out.append((r, c))
    random.shuffle(out)
    return out


def _expected_output(grid: np.ndarray, frames: list[dict], lock_specs: list[dict]) -> np.ndarray:
    wall_mask, door_mask = _frame_masks(grid.shape, frames)
    room_id = _label_rooms(grid.shape, wall_mask, door_mask)
    graph = _room_graph(room_id, door_mask)
    out = grid.copy()

    for spec in lock_specs:
        kr, kc = spec["key_pos"]
        lr, lc = spec["lock_pos"]
        start_room = int(room_id[kr, kc])
        target_room = int(room_id[lr, lc])
        if not _reachable(graph, start_room, target_room):
            continue
        outer = int(spec["outer"])
        inner = int(spec["inner"])
        out[kr : kr + 2, kc : kc + 2] = BG
        out[lr : lr + 2, lc : lc + 2] = np.array([[outer, outer], [outer, inner]], dtype=int)
        out[(room_id == target_room) & (out == BG)] = inner

    return out


def _generate_once() -> np.ndarray | None:
    depth = random.randint(*CHAIN_DEPTH_BOUNDS)
    chain = _build_chain_node(depth)
    chain_h = int(chain["h"])
    chain_w = int(chain["w"])

    include_extra = random.random() < EXTRA_FRAME_PROB
    extra_h = random.randint(*EXTRA_SIZE_BOUNDS) if include_extra else 0
    extra_w = random.randint(*EXTRA_SIZE_BOUNDS) if include_extra else 0

    min_h = max(GRID_H_BOUNDS[0], chain_h + 6, extra_h + 6)
    min_w = max(GRID_W_BOUNDS[0], chain_w + 6, extra_w + 6)
    if min_h > GRID_H_BOUNDS[1] or min_w > GRID_W_BOUNDS[1]:
        return None

    h = random.randint(min_h, GRID_H_BOUNDS[1])
    w = random.randint(min_w, GRID_W_BOUNDS[1])

    frames: list[dict] = []
    chain_top = random.randint(EDGE_MARGIN_BOUNDS[0], h - chain_h - EDGE_MARGIN_BOUNDS[0])
    chain_left = random.randint(EDGE_MARGIN_BOUNDS[0], w - chain_w - EDGE_MARGIN_BOUNDS[0])
    _flatten_chain(chain, chain_top, chain_left, frames)

    if include_extra:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(EDGE_MARGIN_BOUNDS[0], h - extra_h - EDGE_MARGIN_BOUNDS[0])
            left = random.randint(EDGE_MARGIN_BOUNDS[0], w - extra_w - EDGE_MARGIN_BOUNDS[0])
            bbox = (top, left, top + extra_h - 1, left + extra_w - 1)
            if all(_boxes_separated(bbox, frame["bbox"], FRAME_SEPARATION) for frame in frames):
                frames.append({"bbox": bbox, "parent": None, "children": []})
                placed = True
                break
        if not placed:
            return None

    frame_count = len(frames)
    frame_colors = random.sample(list(COLORS), frame_count)
    for frame, color in zip(frames, frame_colors, strict=False):
        frame["color"] = int(color)
        frame["door"] = _sample_door(frame["bbox"])

    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)

    wall_mask, door_mask = _frame_masks(grid.shape, frames)
    room_id = _label_rooms(grid.shape, wall_mask, door_mask)
    outside_room = _outside_room_id(room_id)

    lock_options = {idx: _lock_candidates(grid, room_id, frame) for idx, frame in enumerate(frames)}
    lockable = [idx for idx, options in lock_options.items() if options]
    fill_palette = [c for c in COLORS if c not in frame_colors]
    max_locks = min(len(lockable), len(fill_palette), 4)
    if max_locks < 2:
        return None

    forced: list[int] = []
    non_leaf_chain = [idx for idx in range(depth) if frames[idx]["children"] and idx in lockable]
    if non_leaf_chain:
        forced.append(random.choice(non_leaf_chain))

    min_locks = max(2, len(forced))
    if max_locks < min_locks:
        return None
    lock_count = random.randint(min_locks, max_locks)
    remaining = [idx for idx in lockable if idx not in forced]
    chosen = forced + random.sample(remaining, lock_count - len(forced))
    random.shuffle(chosen)

    fill_colors = random.sample(fill_palette, lock_count)
    lock_specs: list[dict] = []
    for frame_idx, fill_color in zip(chosen, fill_colors, strict=False):
        options = [pos for pos in lock_options[frame_idx] if np.all(grid[pos[0] : pos[0] + 2, pos[1] : pos[1] + 2] == BG)]
        if not options:
            return None
        lr, lc = random.choice(options)
        frame = frames[frame_idx]
        grid[lr : lr + 2, lc : lc + 2] = int(frame["color"])
        lock_specs.append(
            {
                "frame_idx": frame_idx,
                "lock_pos": (lr, lc),
                "outer": int(frame["color"]),
                "inner": int(fill_color),
            }
        )

    key_slots = _key_candidates(grid, room_id, outside_room)
    if len(key_slots) < len(lock_specs):
        return None

    for spec in lock_specs:
        kr, kc = key_slots.pop()
        outer = int(spec["outer"])
        inner = int(spec["inner"])
        grid[kr : kr + 2, kc : kc + 2] = np.array([[outer, outer], [outer, inner]], dtype=int)
        spec["key_pos"] = (kr, kc)

    expected = _expected_output(grid, frames, lock_specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    parsed_frames, parsed_locks, parsed_keys, _, _ = _parse_scene(grid)
    if len(parsed_frames) != len(frames) or len(parsed_locks) != len(lock_specs) or len(parsed_keys) != len(lock_specs):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-lock-key instance")
