import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

REGION_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, CYAN)
DIRECTIONS = ("up", "down", "left", "right")

WORKSPACE_SIZE = 30
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
GROUP_PADDING = 1
PAIR_GAP_RANGE = (1, 3)
RECT_SIZE_RANGE = (3, 5)


def _cc4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _copy_bbox(r0: int, r1: int, c0: int, c1: int, direction: str) -> tuple[int, int, int, int]:
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if direction == "up":
        return r0 - h, r1 - h, c0, c1
    if direction == "down":
        return r0 + h, r1 + h, c0, c1
    if direction == "left":
        return r0, r1, c0 - w, c1 - w
    if direction == "right":
        return r0, r1, c0 + w, c1 + w
    raise ValueError(f"unknown direction: {direction}")


def _direction_from_pointer(pr: int, pc: int, r0: int, r1: int, c0: int, c1: int) -> str:
    distances = [
        ("up", pr - r0),
        ("down", r1 - pr),
        ("left", pc - c0),
        ("right", c1 - pc),
    ]
    order = {"up": 0, "down": 1, "left": 2, "right": 3}
    return min(distances, key=lambda item: (item[1], order[item[0]]))[0]


def _pointer_candidates(height: int, width: int, direction: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            distances = {
                "up": r,
                "down": height - 1 - r,
                "left": c,
                "right": width - 1 - c,
            }
            d = distances[direction]
            if all(d < distances[other] for other in DIRECTIONS if other != direction):
                candidates.append((r, c))
    return candidates


def _analyze_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rects: list[dict] = []
    for comp in _cc4(g != BLACK):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            raise AssertionError("each object must be a filled rectangle")

        pointers = [(r, c) for r, c in comp if int(g[r, c]) == MAROON]
        if len(pointers) != 1:
            raise AssertionError("each rectangle must contain exactly one maroon pointer")
        pr, pc = pointers[0]

        body_colors = {
            int(g[r, c])
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if int(g[r, c]) != MAROON
        }
        if len(body_colors) != 1:
            raise AssertionError("rectangle body must have exactly one non-pointer color")
        color = next(iter(body_colors))

        direction = _direction_from_pointer(pr, pc, r0, r1, c0, c1)
        copy_r0, copy_r1, copy_c0, copy_c1 = _copy_bbox(r0, r1, c0, c1, direction)
        rects.append(
            {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "color": color,
                "pointer": (pr, pc),
                "direction": direction,
                "copy_bbox": (copy_r0, copy_r1, copy_c0, copy_c1),
            }
        )
    rects.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["r1"], rect["c1"]))
    return rects


def _first_hit_color(rects: list[dict], source: dict) -> int:
    direction = source["direction"]
    candidates: list[tuple[int, int, int, int]] = []
    for other in rects:
        if other is source:
            continue
        if direction in ("up", "down"):
            if not _intervals_overlap(source["c0"], source["c1"], other["c0"], other["c1"]):
                continue
            if direction == "up" and other["r1"] < source["r0"]:
                gap = source["r0"] - other["r1"] - 1
            elif direction == "down" and other["r0"] > source["r1"]:
                gap = other["r0"] - source["r1"] - 1
            else:
                continue
        else:
            if not _intervals_overlap(source["r0"], source["r1"], other["r0"], other["r1"]):
                continue
            if direction == "left" and other["c1"] < source["c0"]:
                gap = source["c0"] - other["c1"] - 1
            elif direction == "right" and other["c0"] > source["c1"]:
                gap = other["c0"] - source["c1"] - 1
            else:
                continue
        candidates.append((gap, other["r0"], other["c0"], other["color"]))
    if not candidates:
        return GREY
    candidates.sort()
    return candidates[0][3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each pointed rectangle outward and color the copy from the first rectangle found along that direction.

    Input structure:
    - The background is black.
    - Several disjoint filled rectangles of distinct colors appear on the grid, and each rectangle contains exactly one maroon pointer pixel replacing one body cell.
    - The pointer sits closer to one edge than to the other three, and enough black space is left around rectangles for the projected copies to stay inside the grid without overlapping existing rectangles.

    Rule:
    - For each rectangle, locate its maroon pointer and compare the pointer's distances to the top, bottom, left, and right edges of that rectangle.
    - The nearest edge gives the projection direction. Draw one same-sized rectangular copy immediately outside that side of the source rectangle.
    - The copy is filled uniformly with the color of the first other rectangle encountered in that direction, considering only rectangles whose row-span or column-span overlaps the source strip. If no rectangle is encountered, fill the copy with grey.
    - Keep every original rectangle unchanged, including its maroon pointer pixel.

    Key insight:
    - The maroon pixel does not name the target color directly; it only encodes a direction via edge proximity, while the visible neighbor lying first along that projection strip supplies the copy color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rects = _analyze_rectangles(g)
    h, w = g.shape

    for rect in rects:
        copy_r0, copy_r1, copy_c0, copy_c1 = rect["copy_bbox"]
        if not (0 <= copy_r0 <= copy_r1 < h and 0 <= copy_c0 <= copy_c1 < w):
            raise AssertionError("projected copy must stay inside the grid")
        out[copy_r0 : copy_r1 + 1, copy_c0 : copy_c1 + 1] = _first_hit_color(rects, rect)

    return out


def _rand_size(rng: random.Random) -> int:
    return rng.randint(*RECT_SIZE_RANGE)


def _make_pair_group(rng: random.Random, direction: str) -> dict:
    gap = rng.randint(*PAIR_GAP_RANGE)
    if direction in ("left", "right"):
        height = _rand_size(rng)
        src_w = _rand_size(rng)
        tgt_w = _rand_size(rng)
        if direction == "right":
            zone_h = height
            zone_w = 2 * src_w + gap + 2 * tgt_w
            objects = [
                {"r0": 0, "c0": 0, "h": height, "w": src_w, "direction": "right", "role": "pair_source", "pair_to": 1},
                {
                    "r0": 0,
                    "c0": 2 * src_w + gap,
                    "h": height,
                    "w": tgt_w,
                    "direction": "right",
                    "role": "pair_target",
                },
            ]
        else:
            zone_h = height
            zone_w = 2 * tgt_w + gap + 2 * src_w
            objects = [
                {"r0": 0, "c0": tgt_w, "h": height, "w": tgt_w, "direction": "left", "role": "pair_target"},
                {
                    "r0": 0,
                    "c0": 2 * tgt_w + gap + src_w,
                    "h": height,
                    "w": src_w,
                    "direction": "left",
                    "role": "pair_source",
                    "pair_to": 0,
                },
            ]
    else:
        width = _rand_size(rng)
        src_h = _rand_size(rng)
        tgt_h = _rand_size(rng)
        if direction == "down":
            zone_h = 2 * src_h + gap + 2 * tgt_h
            zone_w = width
            objects = [
                {"r0": 0, "c0": 0, "h": src_h, "w": width, "direction": "down", "role": "pair_source", "pair_to": 1},
                {
                    "r0": 2 * src_h + gap,
                    "c0": 0,
                    "h": tgt_h,
                    "w": width,
                    "direction": "down",
                    "role": "pair_target",
                },
            ]
        else:
            zone_h = 2 * tgt_h + gap + 2 * src_h
            zone_w = width
            objects = [
                {"r0": tgt_h, "c0": 0, "h": tgt_h, "w": width, "direction": "up", "role": "pair_target"},
                {
                    "r0": 2 * tgt_h + gap + src_h,
                    "c0": 0,
                    "h": src_h,
                    "w": width,
                    "direction": "up",
                    "role": "pair_source",
                    "pair_to": 0,
                },
            ]

    return {"zone_h": zone_h, "zone_w": zone_w, "objects": objects}


def _make_single_group(rng: random.Random, direction: str) -> dict:
    height = _rand_size(rng)
    width = _rand_size(rng)
    if direction == "right":
        zone_h, zone_w = height, 2 * width
        objects = [{"r0": 0, "c0": 0, "h": height, "w": width, "direction": "right", "role": "single"}]
    elif direction == "left":
        zone_h, zone_w = height, 2 * width
        objects = [{"r0": 0, "c0": width, "h": height, "w": width, "direction": "left", "role": "single"}]
    elif direction == "down":
        zone_h, zone_w = 2 * height, width
        objects = [{"r0": 0, "c0": 0, "h": height, "w": width, "direction": "down", "role": "single"}]
    else:
        zone_h, zone_w = 2 * height, width
        objects = [{"r0": height, "c0": 0, "h": height, "w": width, "direction": "up", "role": "single"}]
    return {"zone_h": zone_h, "zone_w": zone_w, "objects": objects}


def _choose_groups(rng: random.Random) -> list[dict]:
    groups = [_make_pair_group(rng, rng.choice(DIRECTIONS))]
    if rng.random() < 0.5:
        groups.append(_make_pair_group(rng, rng.choice(DIRECTIONS)))
    else:
        groups.append(_make_single_group(rng, rng.choice(DIRECTIONS)))
    if rng.random() < 0.45:
        groups.append(_make_single_group(rng, rng.choice(DIRECTIONS)))
    return groups


def _place_groups(groups: list[dict], rng: random.Random) -> list[tuple[int, int]] | None:
    occ = np.zeros((WORKSPACE_SIZE, WORKSPACE_SIZE), dtype=bool)
    positions: dict[int, tuple[int, int]] = {}
    order = sorted(range(len(groups)), key=lambda idx: groups[idx]["zone_h"] * groups[idx]["zone_w"], reverse=True)

    for idx in order:
        group = groups[idx]
        if group["zone_h"] > WORKSPACE_SIZE or group["zone_w"] > WORKSPACE_SIZE:
            return None
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(0, WORKSPACE_SIZE - group["zone_h"])
            c0 = rng.randint(0, WORKSPACE_SIZE - group["zone_w"])
            rr0 = max(0, r0 - GROUP_PADDING)
            rr1 = min(WORKSPACE_SIZE, r0 + group["zone_h"] + GROUP_PADDING)
            cc0 = max(0, c0 - GROUP_PADDING)
            cc1 = min(WORKSPACE_SIZE, c0 + group["zone_w"] + GROUP_PADDING)
            if bool(occ[rr0:rr1, cc0:cc1].any()):
                continue
            occ[r0 : r0 + group["zone_h"], c0 : c0 + group["zone_w"]] = True
            positions[idx] = (r0, c0)
            break
        else:
            return None

    return [positions[idx] for idx in range(len(groups))]


def _build_objects(groups: list[dict], positions: list[tuple[int, int]], rng: random.Random) -> list[dict]:
    objects: list[dict] = []
    for group, (base_r, base_c) in zip(groups, positions):
        start = len(objects)
        for obj in group["objects"]:
            placed = {
                "r0": base_r + obj["r0"],
                "c0": base_c + obj["c0"],
                "h": obj["h"],
                "w": obj["w"],
                "direction": obj["direction"],
                "role": obj["role"],
            }
            if "pair_to" in obj:
                placed["pair_to"] = start + obj["pair_to"]
            objects.append(placed)

    colors = rng.sample(list(REGION_COLORS), len(objects))
    for obj, color in zip(objects, colors):
        obj["color"] = color
        pointer = rng.choice(_pointer_candidates(obj["h"], obj["w"], obj["direction"]))
        obj["pointer_local"] = pointer
        obj["pointer"] = (obj["r0"] + pointer[0], obj["c0"] + pointer[1])
    return objects


def _render_workspace(objects: list[dict]) -> np.ndarray:
    grid = np.full((WORKSPACE_SIZE, WORKSPACE_SIZE), BLACK, dtype=int)
    for obj in objects:
        r0, c0 = obj["r0"], obj["c0"]
        r1 = r0 + obj["h"]
        c1 = c0 + obj["w"]
        grid[r0:r1, c0:c1] = obj["color"]
        pr, pc = obj["pointer"]
        grid[pr, pc] = MAROON
    return grid


def _crop_workspace(grid: np.ndarray, groups: list[dict], positions: list[tuple[int, int]], objects: list[dict], rng: random.Random) -> tuple[np.ndarray, list[dict]]:
    min_r = min(r0 for r0, _ in positions)
    min_c = min(c0 for _, c0 in positions)
    max_r = max(r0 + group["zone_h"] - 1 for group, (r0, _) in zip(groups, positions))
    max_c = max(c0 + group["zone_w"] - 1 for group, (_, c0) in zip(groups, positions))

    top_margin = rng.randint(0, min_r)
    left_margin = rng.randint(0, min_c)
    bottom_margin = rng.randint(0, WORKSPACE_SIZE - 1 - max_r)
    right_margin = rng.randint(0, WORKSPACE_SIZE - 1 - max_c)

    row_start = min_r - top_margin
    col_start = min_c - left_margin
    row_end = max_r + bottom_margin
    col_end = max_c + right_margin

    cropped = grid[row_start : row_end + 1, col_start : col_end + 1].copy()
    shifted: list[dict] = []
    for obj in objects:
        new_obj = obj.copy()
        new_obj["r0"] -= row_start
        new_obj["c0"] -= col_start
        pr, pc = new_obj["pointer"]
        new_obj["pointer"] = (pr - row_start, pc - col_start)
        shifted.append(new_obj)
    return cropped, shifted


def _validate_generated(grid: np.ndarray, objects: list[dict]) -> bool:
    try:
        rects = _analyze_rectangles(grid)
    except AssertionError:
        return False

    if len(rects) != len(objects):
        return False

    by_bbox = {
        (obj["r0"], obj["r0"] + obj["h"] - 1, obj["c0"], obj["c0"] + obj["w"] - 1): obj
        for obj in objects
    }
    if len(by_bbox) != len(objects):
        return False

    h, w = grid.shape
    copy_mask = np.zeros((h, w), dtype=bool)
    saw_hit = False
    saw_miss = False

    rect_by_bbox = {}
    for rect in rects:
        bbox = (rect["r0"], rect["r1"], rect["c0"], rect["c1"])
        spec = by_bbox.get(bbox)
        if spec is None:
            return False
        if rect["direction"] != spec["direction"]:
            return False
        rect_by_bbox[bbox] = rect

    for rect in rects:
        bbox = (rect["r0"], rect["r1"], rect["c0"], rect["c1"])
        spec = by_bbox[bbox]
        copy_r0, copy_r1, copy_c0, copy_c1 = rect["copy_bbox"]
        if not (0 <= copy_r0 <= copy_r1 < h and 0 <= copy_c0 <= copy_c1 < w):
            return False
        if not np.all(grid[copy_r0 : copy_r1 + 1, copy_c0 : copy_c1 + 1] == BLACK):
            return False
        if bool(copy_mask[copy_r0 : copy_r1 + 1, copy_c0 : copy_c1 + 1].any()):
            return False
        copy_mask[copy_r0 : copy_r1 + 1, copy_c0 : copy_c1 + 1] = True

        actual = _first_hit_color(rects, rect)
        if spec["role"] == "pair_source":
            target = objects[spec["pair_to"]]
            if actual != target["color"]:
                return False
            saw_hit = True
        else:
            if actual != GREY:
                return False
            saw_miss = True

    if not (saw_hit and saw_miss):
        return False

    out = main(grid)
    return out.shape == grid.shape and max(out.shape) <= WORKSPACE_SIZE


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    groups = _choose_groups(rng)
    if sum(len(group["objects"]) for group in groups) > len(REGION_COLORS):
        return None

    positions = _place_groups(groups, rng)
    if positions is None:
        return None

    objects = _build_objects(groups, positions, rng)
    workspace = _render_workspace(objects)
    cropped, shifted_objects = _crop_workspace(workspace, groups, positions, objects, rng)

    if not _validate_generated(cropped, shifted_objects):
        return None
    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
