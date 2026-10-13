import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

LEGEND_H = 2
EMITTER_COLORS = (BLUE, GREEN)
OBSTACLE_COLORS = (RED, YELLOW, ORANGE)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_ORDER = ("U", "R", "D", "L")
DIR_INDEX = {name: i for i, name in enumerate(DIR_ORDER)}

# Base body-only layout found by path search, then randomized by rotation/translation.
# A single emitter plus five obstacles keeps the object count within the puzzle range
# while guaranteeing that one beam re-enters the emitter and therefore needs transparency.
BASE_BODY_H = 15
BASE_BODY_W = 16
BASE_LAYOUT = [
    {"kind": "emitter", "r0": 5, "c0": 6},
    {"kind": "obstacle", "role": "N", "turn": "L", "r0": 0, "c0": 6},
    {"kind": "obstacle", "role": "E", "turn": "L", "r0": 6, "c0": 12},
    {"kind": "obstacle", "role": "S", "turn": "R", "r0": 11, "c0": 5},
    {"kind": "obstacle", "role": "W", "turn": "R", "r0": 4, "c0": 1},
    {"kind": "obstacle", "role": "SW", "turn": "R", "r0": 9, "c0": 0},
]

BODY_SIDE_RANGE = (18, 24)  # output remains <= 26x24 after adding the legend rows.
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_legend(grid: np.ndarray) -> dict[int, str]:
    legend = np.asarray(grid[:LEGEND_H], dtype=int)
    colored_cols = {}
    for color in OBSTACLE_COLORS:
        cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == color)]
        assert len(cols) == 1, f"legend color {color} must appear in exactly one column"
        colored_cols[color] = cols[0]

    candidates = []
    for anchor in range(legend.shape[1] - 7):
        slots = [(anchor, anchor + 1), (anchor + 3, anchor + 4), (anchor + 6, anchor + 7)]
        used_slots = []
        turn_map: dict[int, str] = {}
        ok = True
        for color in OBSTACLE_COLORS:
            col = colored_cols[color]
            slot_index = None
            side = None
            for idx, (left, right) in enumerate(slots):
                if col == left:
                    slot_index = idx
                    side = "L"
                    break
                if col == right:
                    slot_index = idx
                    side = "R"
                    break
            if slot_index is None or slot_index in used_slots:
                ok = False
                break
            used_slots.append(slot_index)
            turn_map[color] = side
        if ok:
            candidates.append(turn_map)

    assert len(candidates) == 1, "legend slot structure must be uniquely decodable"
    return candidates[0]


def _classify_bbox(sub: np.ndarray):
    if sub.shape != (4, 4):
        return None
    border_mask = np.ones((4, 4), dtype=bool)
    border_mask[1:3, 1:3] = False
    border_vals = sub[border_mask]
    if np.all(sub[1:3, 1:3] == BG) and np.all(border_vals == border_vals[0]) and border_vals[0] in EMITTER_COLORS:
        return "emitter", int(border_vals[0])
    if np.all(sub == sub[0, 0]) and int(sub[0, 0]) in OBSTACLE_COLORS:
        return "obstacle", int(sub[0, 0])
    return None


def _find_objects(grid: np.ndarray) -> list[dict]:
    body = np.asarray(grid[LEGEND_H:], dtype=int)
    comps = _components(body != BG)
    objects = []
    for comp in comps:
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        sub = body[r0 : r1 + 1, c0 : c1 + 1]
        info = _classify_bbox(sub)
        if info is None:
            continue
        kind, color = info
        objects.append(
            {
                "kind": kind,
                "color": color,
                "r0": int(r0 + LEGEND_H),
                "c0": int(c0),
                "r1": int(r1 + LEGEND_H),
                "c1": int(c1),
                "bbox": (int(r0 + LEGEND_H), int(c0), int(r1 + LEGEND_H), int(c1)),
                "sub": sub.copy(),
            }
        )
    objects.sort(key=lambda obj: (obj["r0"], obj["c0"], obj["kind"]))
    return objects


def _rotate_dir(direction: int, turn: str) -> int:
    return (direction - 1) % 4 if turn == "L" else (direction + 1) % 4


def _launch_state(emitter: dict, direction_name: str) -> tuple[int, int, int]:
    r0, c0 = emitter["r0"], emitter["c0"]
    if direction_name == "U":
        return r0, c0 + 1, DIR_INDEX["U"]
    if direction_name == "R":
        return r0 + 1, c0 + 3, DIR_INDEX["R"]
    if direction_name == "D":
        return r0 + 3, c0 + 1, DIR_INDEX["D"]
    if direction_name == "L":
        return r0 + 1, c0, DIR_INDEX["L"]
    raise ValueError(direction_name)


def _build_static_masks(grid: np.ndarray, objects: list[dict]):
    h, w = grid.shape
    obstacle_color = np.zeros((h, w), dtype=int)
    emitter_region = np.zeros((h, w), dtype=bool)
    immutable = np.zeros((h, w), dtype=bool)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        if obj["kind"] == "obstacle":
            obstacle_color[r0 : r1 + 1, c0 : c1 + 1] = obj["color"]
            immutable[r0 : r1 + 1, c0 : c1 + 1] = True
        else:
            emitter_region[r0 : r1 + 1, c0 : c1 + 1] = True
            immutable[r0 : r1 + 1, c0 : c1 + 1] |= obj["sub"] != BG
    return obstacle_color, emitter_region, immutable


def _simulate(
    grid: np.ndarray,
    turn_map: dict[int, str],
    objects: list[dict],
    *,
    opaque_emitters: bool = False,
    forced_turn: str | None = None,
):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[:LEGEND_H, :] = BG

    obstacle_color, emitter_region, immutable = _build_static_masks(g, objects)
    emitters = [obj for obj in objects if obj["kind"] == "emitter"]
    emitters.sort(key=lambda obj: (obj["r0"], obj["c0"]))

    info = {
        "ray_hits": [],
        "reentered_emitter": False,
        "max_hits": 0,
        "cycled": False,
    }

    for emitter in emitters:
        for direction_name in DIR_ORDER:
            r, c, direction = _launch_state(emitter, direction_name)
            hit_count = 0
            left_any_emitter = False
            seen: set[tuple[int, int, int]] = set()
            while True:
                state = (r, c, direction)
                if state in seen:
                    info["cycled"] = True
                    return out, info
                seen.add(state)

                dr, dc = DIRS[direction]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                    break

                if obstacle_color[nr, nc] != BG:
                    hit_count += 1
                    turn = forced_turn if forced_turn is not None else turn_map[int(obstacle_color[nr, nc])]
                    direction = _rotate_dir(direction, turn)
                    continue

                if opaque_emitters and emitter_region[nr, nc]:
                    break

                r, c = nr, nc
                if emitter_region[r, c] and left_any_emitter:
                    info["reentered_emitter"] = True
                if not emitter_region[r, c]:
                    left_any_emitter = True

                if not immutable[r, c]:
                    out[r, c] = emitter["color"]

            info["ray_hits"].append(hit_count)
            info["max_hits"] = max(info["max_hits"], hit_count)

    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify the 4x4 rectangles into emitters or obstacles, then trace the legend-guided rays.

    Input structure:
    - The grid has black background, plus a 2-row grey legend strip at the top.
    - The legend contains colored vertical bars for obstacle colors; a bar on the left half of its 2-column slot means counterclockwise, and on the right half means clockwise.
    - Below the legend are separated 4x4 objects: hollow blue/green frames and solid red/yellow/orange blocks.

    Rule:
    - Remove the legend strip in the output by turning the top two rows black.
    - Classify each 4x4 object by its interior: a 2x2 black interior makes it an emitter, otherwise it is an obstacle.
    - For emitters, launch one 1-cell ray in each cardinal direction in emitter order (top, right, bottom, left), using the emitter color.
    - A ray paints black cells, turns 90 degrees at obstacle colors according to the legend, passes through emitter regions without changing those object pixels, and stops only at the grid boundary.

    Key insight:
    - The same 4x4 shape test that separates hollow from solid objects determines the physical role in the ray system, and emitter transparency is necessary to recover the full path.
    """
    g = np.asarray(grid, dtype=int)
    turn_map = _parse_legend(g)
    objects = _find_objects(g)
    out, _info = _simulate(g, turn_map, objects)
    return out


def _rotate_layout_entry(entry: dict, body_h: int, body_w: int, k: int) -> tuple[dict, int, int]:
    r0, c0 = entry["r0"], entry["c0"]
    h, w = body_h, body_w
    for _ in range(k % 4):
        r0, c0 = c0, h - 4 - r0
        h, w = w, h
    new_entry = dict(entry)
    new_entry["r0"] = int(r0)
    new_entry["c0"] = int(c0)
    return new_entry, h, w


def _place_base_objects(body_h: int, body_w: int) -> list[dict]:
    k = random.randrange(4)
    rotated = []
    rot_h, rot_w = BASE_BODY_H, BASE_BODY_W
    for entry in BASE_LAYOUT:
        new_entry, rot_h, rot_w = _rotate_layout_entry(entry, BASE_BODY_H, BASE_BODY_W, k)
        rotated.append(new_entry)

    row_off = random.randint(0, body_h - rot_h)
    col_off = random.randint(0, body_w - rot_w)
    placed = []
    for entry in rotated:
        placed.append(
            {
                "kind": entry["kind"],
                "role": entry.get("role"),
                "turn": entry.get("turn"),
                "r0": int(row_off + entry["r0"]),
                "c0": int(col_off + entry["c0"]),
            }
        )
    return placed


def _choose_turn_map() -> dict[int, str]:
    turns = ["L", "R", "R"] if random.random() < 0.5 else ["L", "L", "R"]
    random.shuffle(turns)
    return {color: turn for color, turn in zip(OBSTACLE_COLORS, turns)}


def _assign_obstacle_colors(base_objects: list[dict], turn_map: dict[int, str]) -> None:
    colors_by_turn = {"L": [], "R": []}
    for color, turn in turn_map.items():
        colors_by_turn[turn].append(color)
    assert colors_by_turn["L"] and colors_by_turn["R"]

    role_groups = {"L": [], "R": []}
    for obj in base_objects:
        if obj["kind"] == "obstacle":
            role_groups[obj["turn"]].append(obj)

    for turn, roles in role_groups.items():
        palette = colors_by_turn[turn][:]
        random.shuffle(palette)
        random.shuffle(roles)
        for role, color in zip(roles, palette):
            role["color"] = color
        for role in roles[len(palette) :]:
            role["color"] = random.choice(palette)


def _draw_legend(grid: np.ndarray, turn_map: dict[int, str]) -> None:
    h, w = grid.shape
    assert h >= LEGEND_H
    grid[:LEGEND_H, :] = GREY
    anchor = random.randint(0, w - 8)
    starts = [anchor, anchor + 3, anchor + 6]
    colors = random.sample(OBSTACLE_COLORS, 3)
    for color, start in zip(colors, starts):
        if turn_map[color] == "L":
            grid[:LEGEND_H, start] = color
        else:
            grid[:LEGEND_H, start + 1] = color


def _draw_objects(grid: np.ndarray, objects: list[dict], emitter_color: int) -> None:
    for obj in objects:
        r0, c0 = obj["r0"] + LEGEND_H, obj["c0"]
        if obj["kind"] == "emitter":
            grid[r0 : r0 + 4, c0 : c0 + 4] = emitter_color
            grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = BG
        else:
            grid[r0 : r0 + 4, c0 : c0 + 4] = obj["color"]


def _bbox_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    return ar + 3 < br - 1 or br + 3 < ar - 1 or ac + 3 < bc - 1 or bc + 3 < ac - 1


def _try_add_extra_obstacle(grid: np.ndarray, base_objects: list[dict], turn_map: dict[int, str]) -> list[dict]:
    objects = [dict(obj) for obj in base_objects]
    scene_grid = grid.copy()
    parsed = _find_objects(scene_grid)
    out, _info = _simulate(scene_grid, turn_map, parsed)
    ray_mask = out != scene_grid.copy().astype(int)
    ray_mask[:LEGEND_H, :] &= out[:LEGEND_H, :] != BG
    occupied = [(obj["r0"], obj["c0"]) for obj in objects]
    body_h = grid.shape[0] - LEGEND_H
    body_w = grid.shape[1]

    candidates = [(r, c) for r in range(body_h - 3) for c in range(body_w - 3)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if not all(_bbox_separated((r0, c0), other) for other in occupied):
            continue
        rr = slice(r0 + LEGEND_H, r0 + LEGEND_H + 4)
        cc = slice(c0, c0 + 4)
        if np.any(ray_mask[rr, cc]):
            continue
        extra = {"kind": "obstacle", "r0": r0, "c0": c0, "color": random.choice(OBSTACLE_COLORS)}
        objects.append(extra)
        return objects
    return objects


def _materialize_grid(body_h: int, body_w: int, objects: list[dict], turn_map: dict[int, str], emitter_color: int) -> np.ndarray:
    grid = np.zeros((body_h + LEGEND_H, body_w), dtype=int)
    _draw_legend(grid, turn_map)
    _draw_objects(grid, objects, emitter_color)
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    objects = _find_objects(grid)
    emitters = [obj for obj in objects if obj["kind"] == "emitter"]
    obstacles = [obj for obj in objects if obj["kind"] == "obstacle"]
    assert len(emitters) == 1
    assert 5 <= len(objects) <= 7

    turn_map = _parse_legend(grid)
    assert set(turn_map) == set(OBSTACLE_COLORS)
    assert len(set(turn_map.values())) == 2
    assert {obj["color"] for obj in obstacles} == set(OBSTACLE_COLORS)

    out, info = _simulate(grid, turn_map, objects)
    assert not info["cycled"]
    assert all(hit >= 1 for hit in info["ray_hits"])
    assert info["max_hits"] >= 2
    assert info["reentered_emitter"]

    opaque_out, _ = _simulate(grid, turn_map, objects, opaque_emitters=True)
    assert not np.array_equal(out, opaque_out)

    all_left_out, _ = _simulate(grid, turn_map, objects, forced_turn="L")
    all_right_out, _ = _simulate(grid, turn_map, objects, forced_turn="R")
    assert not np.array_equal(out, all_left_out)
    assert not np.array_equal(out, all_right_out)
    assert np.any(out[:LEGEND_H, :] != BG)


def _gen_attempt() -> np.ndarray | None:
    body_h = random.randint(*BODY_SIDE_RANGE)
    body_w = random.randint(*BODY_SIDE_RANGE)

    base_objects = _place_base_objects(body_h, body_w)
    turn_map = _choose_turn_map()
    _assign_obstacle_colors(base_objects, turn_map)
    emitter_color = random.choice(EMITTER_COLORS)

    objects = base_objects
    if random.random() < 0.55:
        temp_grid = _materialize_grid(body_h, body_w, objects, turn_map, emitter_color)
        objects = _try_add_extra_obstacle(temp_grid, objects, turn_map)

    grid = _materialize_grid(body_h, body_w, objects, turn_map, emitter_color)

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
