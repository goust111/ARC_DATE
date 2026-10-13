from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste


BACKGROUND = 0
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
BRANCHES = ("relay", "edge", "free")
BRANCH_WEIGHTS = (0.45, 0.40, 0.15)


def _box_overlap(a: dict, b: dict) -> bool:
    return not (a["r1"] < b["r0"] or b["r1"] < a["r0"] or a["c1"] < b["c0"] or b["c1"] < a["c0"])


def _boxes_clear(box: tuple[int, int, int, int], other: tuple[int, int, int, int], pad: int = 0) -> bool:
    r0, c0, r1, c1 = box
    or0, oc0, or1, oc1 = other
    return r1 + pad < or0 or or1 + pad < r0 or c1 + pad < oc0 or oc1 + pad < c0


def _shift_object(obj: dict, steps: int, direction: str) -> None:
    dr, dc = DIRECTIONS[direction]
    delta_r = dr * int(steps)
    delta_c = dc * int(steps)
    obj["r0"] += delta_r
    obj["r1"] += delta_r
    obj["c0"] += delta_c
    obj["c1"] += delta_c


def _copy_state(objects: list[dict]) -> list[dict]:
    state = []
    for obj in objects:
        copied = obj.copy()
        copied["patch"] = obj["patch"].copy()
        state.append(copied)
    return state


def _orthogonally_overlaps(a: dict, b: dict, direction: str) -> bool:
    if direction in {"left", "right"}:
        return not (a["r1"] < b["r0"] or b["r1"] < a["r0"])
    return not (a["c1"] < b["c0"] or b["c1"] < a["c0"])


def _gap_ahead(mover: dict, target: dict, direction: str) -> int | None:
    if not _orthogonally_overlaps(mover, target, direction):
        return None
    if direction == "right" and target["c0"] > mover["c1"]:
        return int(target["c0"] - mover["c1"] - 1)
    if direction == "left" and target["c1"] < mover["c0"]:
        return int(mover["c0"] - target["c1"] - 1)
    if direction == "down" and target["r0"] > mover["r1"]:
        return int(target["r0"] - mover["r1"] - 1)
    if direction == "up" and target["r1"] < mover["r0"]:
        return int(mover["r0"] - target["r1"] - 1)
    return None


def _distance_to_edge(obj: dict, shape: tuple[int, int], direction: str) -> int:
    height, width = map(int, shape)
    if direction == "right":
        return int(width - 1 - obj["c1"])
    if direction == "left":
        return int(obj["c0"])
    if direction == "down":
        return int(height - 1 - obj["r1"])
    return int(obj["r0"])


def _render_state(shape: tuple[int, int], bg: int, objects: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for obj in objects:
        paste(out, obj["patch"], obj["r0"], obj["c0"])
    return out


def _slot_properties(patch: np.ndarray, bg: int) -> tuple[int, str]:
    hole_mask = patch == int(bg)
    hole_box = bbox(hole_mask)
    assert hole_box is not None, "engine must contain a slot"

    slot_area = int(hole_mask.sum())
    assert 1 <= slot_area <= 4, "slot length must be 1..4"

    sr0, sc0, sr1, sc1 = hole_box
    assert slot_area == bbox_h(hole_box) * bbox_w(hole_box), "slot must be a straight solid line"
    assert sr0 > 0 and sc0 > 0 and sr1 < patch.shape[0] - 1 and sc1 < patch.shape[1] - 1, "slot must be strictly internal"
    assert bbox_h(hole_box) == 1 or bbox_w(hole_box) == 1, "slot must be horizontal or vertical"

    center_r2 = patch.shape[0] - 1
    center_c2 = patch.shape[1] - 1
    slot_r2 = sr0 + sr1
    slot_c2 = sc0 + sc1

    if slot_r2 == center_r2 and slot_c2 < center_c2:
        nozzle = "left"
    elif slot_r2 == center_r2 and slot_c2 > center_c2:
        nozzle = "right"
    elif slot_c2 == center_c2 and slot_r2 < center_r2:
        nozzle = "up"
    elif slot_c2 == center_c2 and slot_r2 > center_r2:
        nozzle = "down"
    else:
        raise AssertionError("slot midpoint must be strictly above, below, left, or right of the rectangle center")

    return slot_area, OPPOSITE[nozzle]


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected foreground rectangles"

    objects = []
    for comp in comps:
        box = comp["bbox"]
        r0, c0, r1, c1 = map(int, box)
        patch = crop_bbox(g, box)
        color = int(comp["color"])
        area = int(comp["area"])
        h, w = patch.shape

        vals = set(map(int, patch.flatten()))
        assert vals <= {bg, color}, "each object must be monochrome on the background"

        full_area = int(h * w)
        obj = {
            "bbox": (r0, c0, r1, c1),
            "r0": r0,
            "c0": c0,
            "r1": r1,
            "c1": c1,
            "h": int(h),
            "w": int(w),
            "patch": patch,
            "color": color,
            "is_engine": False,
            "slot_len": None,
            "direction": None,
        }

        if area == full_area:
            assert np.all(patch == color), "cargo must be a solid rectangle"
        else:
            assert np.all((patch == bg) | (patch == color)), "engine bbox may only contain its own color and background"
            slot_len, direction = _slot_properties(patch, bg)
            assert area + slot_len == full_area, "engine must be a full rectangle with exactly one slot removed"
            obj["is_engine"] = True
            obj["slot_len"] = int(slot_len)
            obj["direction"] = direction

        objects.append(obj)

    objects.sort(key=lambda obj: (obj["r0"], obj["c0"], obj["r1"], obj["c1"], obj["color"]))
    engine_indices = [idx for idx, obj in enumerate(objects) if obj["is_engine"]]
    assert len(engine_indices) == 1, "expected exactly one engine rectangle"
    engine_idx = engine_indices[0]
    assert len(objects) >= 2, "expected at least one cargo rectangle"

    return {
        "grid": g,
        "shape": g.shape,
        "bg": bg,
        "objects": objects,
        "engine_idx": engine_idx,
        "direction": objects[engine_idx]["direction"],
        "slot_len": int(objects[engine_idx]["slot_len"]),
    }


def _simulate_events(scene: dict) -> list[dict]:
    state = _copy_state(scene["objects"])
    direction = scene["direction"]
    remaining = int(scene["slot_len"])
    moving_idx = int(scene["engine_idx"])

    while remaining > 0:
        mover = state[moving_idx]
        edge_dist = _distance_to_edge(mover, scene["shape"], direction)

        collisions = []
        for idx, target in enumerate(state):
            if idx == moving_idx:
                continue
            gap = _gap_ahead(mover, target, direction)
            if gap is not None:
                collisions.append((gap, idx))

        hit_dist = None
        hit_idx = None
        if collisions:
            collisions.sort()
            hit_dist = int(collisions[0][0])
            tied = [idx for gap, idx in collisions if gap == hit_dist]
            assert len(tied) == 1, "collision target must be unique"
            hit_idx = tied[0]

        event_dist = edge_dist if hit_dist is None else min(edge_dist, hit_dist)
        if remaining <= event_dist:
            _shift_object(mover, remaining, direction)
            break
        if hit_dist is None or edge_dist < hit_dist:
            _shift_object(mover, edge_dist, direction)
            break

        _shift_object(mover, hit_dist, direction)
        remaining -= hit_dist
        moving_idx = int(hit_idx)

    return state


def _simulate_stepwise(scene: dict) -> tuple[list[dict], dict]:
    state = _copy_state(scene["objects"])
    direction = scene["direction"]
    remaining = int(scene["slot_len"])
    moving_idx = int(scene["engine_idx"])
    dr, dc = DIRECTIONS[direction]

    trace = {
        "collisions": 0,
        "edge_hit": False,
    }

    while remaining > 0:
        mover = state[moving_idx]
        trial = mover.copy()
        trial["patch"] = mover["patch"]
        _shift_object(trial, 1, direction)

        if trial["r0"] < 0 or trial["c0"] < 0 or trial["r1"] >= scene["shape"][0] or trial["c1"] >= scene["shape"][1]:
            trace["edge_hit"] = True
            break

        colliders = []
        for idx, target in enumerate(state):
            if idx == moving_idx:
                continue
            if _box_overlap(trial, target):
                colliders.append(idx)
        assert len(colliders) <= 1, "collision target must be unique"

        if colliders:
            moving_idx = colliders[0]
            trace["collisions"] += 1
            continue

        mover["r0"] += dr
        mover["r1"] += dr
        mover["c0"] += dc
        mover["c1"] += dc
        remaining -= 1

    return state, trace


def _analyze_output(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    scene = _parse_scene(grid)
    state = _simulate_events(scene)
    return _render_state(scene["shape"], scene["bg"], state), scene


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the slotted engine opposite its nozzle and relay the remaining momentum through any collision chain.

    Input structure:
    - A mostly black background contains exactly one monochrome rectangular engine whose
      strictly internal straight black slot is the only missing segment in its bbox.
    - The remaining foreground objects are solid monochrome rectangles without slots.
    - Some cargo rectangles lie on the engine's motion line while others may be off-line distractors.

    Rule:
    - The slot midpoint's position relative to the engine center gives the nozzle direction;
      the engine moves in the opposite direction for exactly as many steps as the slot length.
    - When a moving rectangle would enter another rectangle, it stops adjacent to it and
      passes all still-unused steps to the struck rectangle, which continues in the same direction.
    - If the currently moving rectangle would leave the grid, it instead stops at the edge and
      the remaining momentum is discarded; the engine's slot moves with the engine unchanged.

    Key insight:
    - The missing internal slot simultaneously encodes both direction and total momentum, and
      the correct output requires simulating that momentum as a conserved relay across collisions.
    """
    out, _scene = _analyze_output(np.asarray(grid, dtype=int))
    return out


def _engine_patch(direction: str, slot_len: int, color: int) -> np.ndarray:
    if direction in {"left", "right"}:
        height_options = [h for h in (slot_len + 2, slot_len + 4) if 4 <= h <= 6]
        height = random.choice(height_options)
        width = random.randint(4, 5)
        patch = np.full((height, width), int(color), dtype=int)
        slot_r0 = (height - slot_len) // 2
        if direction == "right":
            col_options = [c for c in range(1, width - 1) if 2 * c < width - 1]
        else:
            col_options = [c for c in range(1, width - 1) if 2 * c > width - 1]
        slot_c = random.choice(col_options)
        patch[slot_r0 : slot_r0 + slot_len, slot_c] = BACKGROUND
        return patch

    width_options = [w for w in (slot_len + 2, slot_len + 4) if 4 <= w <= 6]
    width = random.choice(width_options)
    height = random.randint(4, 5)
    patch = np.full((height, width), int(color), dtype=int)
    slot_c0 = (width - slot_len) // 2
    if direction == "down":
        row_options = [r for r in range(1, height - 1) if 2 * r < height - 1]
    else:
        row_options = [r for r in range(1, height - 1) if 2 * r > height - 1]
    slot_r = random.choice(row_options)
    patch[slot_r, slot_c0 : slot_c0 + slot_len] = BACKGROUND
    return patch


def _solid_line_patch(direction: str, axis_len: int, cross_len: int, color: int) -> np.ndarray:
    if direction in {"left", "right"}:
        return np.full((cross_len, axis_len), int(color), dtype=int)
    return np.full((axis_len, cross_len), int(color), dtype=int)


def _layout_axis(
    sizes: list[int],
    gaps: list[int],
    front_margin: int,
    back_margin: int,
    direction: str,
) -> tuple[list[int], int]:
    starts = [0] * len(sizes)
    if direction in {"right", "down"}:
        starts[0] = int(back_margin)
        for idx in range(1, len(sizes)):
            starts[idx] = starts[idx - 1] + sizes[idx - 1] + gaps[idx - 1]
        axis_len = starts[-1] + sizes[-1] + int(front_margin)
        return starts, int(axis_len)

    coord = int(front_margin)
    for idx in range(len(sizes) - 1, -1, -1):
        starts[idx] = coord
        if idx > 0:
            coord += sizes[idx] + gaps[idx - 1]
    axis_len = starts[0] + sizes[0] + int(back_margin)
    return starts, int(axis_len)


def _choose_chain(slot_len: int, line_cargo_count: int, branch: str) -> tuple[list[int], int, int]:
    if branch == "free":
        gaps = [random.randint(slot_len + 1, slot_len + 4)]
        for _ in range(line_cargo_count - 1):
            gaps.append(random.randint(0, 2))
        return gaps, slot_len, random.randint(1, 4)

    remaining = int(slot_len)
    gaps = []
    for idx in range(line_cargo_count):
        max_gap = remaining - 1
        assert max_gap >= 0
        force_positive = idx == 0 and line_cargo_count > 1 and remaining > 1 and random.random() < 0.45
        low = 1 if force_positive else 0
        low = min(low, max_gap)
        gap = random.randint(low, max_gap)
        gaps.append(gap)
        remaining -= gap

    if branch == "relay":
        front_margin = random.randint(remaining + 1, remaining + 4)
    else:
        front_margin = random.randint(0, remaining - 1)
    return gaps, remaining, front_margin


def _ahead_of_engine(engine: dict, obj: dict, direction: str) -> bool:
    if direction == "right":
        return obj["c0"] > engine["c1"]
    if direction == "left":
        return obj["c1"] < engine["c0"]
    if direction == "down":
        return obj["r0"] > engine["r1"]
    return obj["r1"] < engine["r0"]


def _place_distractors(
    shape: tuple[int, int],
    direction: str,
    lane_box: tuple[int, int, int, int],
    occupied_boxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> list[tuple[np.ndarray, int, int]]:
    height, width = map(int, shape)
    lane_r0, lane_c0, lane_r1, lane_c1 = lane_box
    placed = []
    desired = random.randint(1, min(2, len(colors)))

    if direction in {"left", "right"}:
        bands = []
        if lane_r0 >= 3:
            bands.append((0, lane_r0 - 2, 0, width - 1))
        if height - 1 - lane_r1 >= 3:
            bands.append((lane_r1 + 2, height - 1, 0, width - 1))
    else:
        bands = []
        if lane_c0 >= 3:
            bands.append((0, height - 1, 0, lane_c0 - 2))
        if width - 1 - lane_c1 >= 3:
            bands.append((0, height - 1, lane_c1 + 2, width - 1))

    if not bands:
        return placed

    for color in colors:
        if len(placed) >= desired:
            break
        for _ in range(80):
            br0, br1, bc0, bc1 = random.choice(bands)
            max_h = min(4, br1 - br0 + 1)
            max_w = min(4, bc1 - bc0 + 1)
            if max_h < 2 or max_w < 2:
                continue
            patch_h = random.randint(2, max_h)
            patch_w = random.randint(2, max_w)
            r0 = random.randint(br0, br1 - patch_h + 1)
            c0 = random.randint(bc0, bc1 - patch_w + 1)
            box = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
            if all(_boxes_clear(box, other, pad=1) for other in occupied_boxes):
                patch = np.full((patch_h, patch_w), int(color), dtype=int)
                placed.append((patch, r0, c0))
                occupied_boxes.append(box)
                break
    return placed


def _generate_attempt() -> tuple[np.ndarray, str] | None:
    direction = random.choice(tuple(DIRECTIONS))
    branch = random.choices(BRANCHES, weights=BRANCH_WEIGHTS, k=1)[0]
    slot_len = random.randint(1, 4)

    palette = list(range(1, 10))
    random.shuffle(palette)
    engine_color = palette.pop()
    engine_patch = _engine_patch(direction, slot_len, engine_color)

    if direction in {"left", "right"}:
        cross_len = engine_patch.shape[0]
        engine_axis = engine_patch.shape[1]
    else:
        cross_len = engine_patch.shape[1]
        engine_axis = engine_patch.shape[0]

    line_cargo_count = random.randint(1, 3 if branch != "free" else 2)
    line_colors = [palette.pop() for _ in range(line_cargo_count)]
    line_axis_sizes = [random.randint(2, 4) for _ in range(line_cargo_count)]
    gaps, final_remaining, front_margin = _choose_chain(slot_len, line_cargo_count, branch)
    back_margin = random.randint(1, 4)
    starts, axis_len = _layout_axis([engine_axis] + line_axis_sizes, gaps, front_margin, back_margin, direction)

    if direction in {"left", "right"}:
        top_margin = random.randint(2, 6)
        bottom_margin = random.randint(2, 6)
        height = top_margin + cross_len + bottom_margin
        width = axis_len
        lane_r0 = top_margin
        lane_c0 = 0
    else:
        left_margin = random.randint(2, 6)
        right_margin = random.randint(2, 6)
        height = axis_len
        width = left_margin + cross_len + right_margin
        lane_r0 = 0
        lane_c0 = left_margin

    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    placements = [(engine_patch, starts[0], engine_color)]
    for axis_start, axis_len_i, color in zip(starts[1:], line_axis_sizes, line_colors):
        placements.append((_solid_line_patch(direction, axis_len_i, cross_len, color), axis_start, color))

    occupied_boxes = []
    for patch, axis_start, _color in placements:
        if direction in {"left", "right"}:
            r0, c0 = lane_r0, axis_start
        else:
            r0, c0 = axis_start, lane_c0
        paste(grid, patch, r0, c0)
        occupied_boxes.append((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1))

    lane_box = occupied_boxes[0]
    for box in occupied_boxes[1:]:
        lane_box = (
            min(lane_box[0], box[0]),
            min(lane_box[1], box[1]),
            max(lane_box[2], box[2]),
            max(lane_box[3], box[3]),
        )

    distractors = _place_distractors(grid.shape, direction, lane_box, occupied_boxes[:], palette)
    if not distractors:
        return None
    for patch, r0, c0 in distractors:
        paste(grid, patch, r0, c0)

    if np.bincount(grid.ravel(), minlength=10).argmax() != BACKGROUND:
        return None

    scene = _parse_scene(grid)
    event_state = _simulate_events(scene)
    step_state, trace = _simulate_stepwise(scene)
    event_out = _render_state(scene["shape"], scene["bg"], event_state)
    step_out = _render_state(scene["shape"], scene["bg"], step_state)
    if np.array_equal(grid, event_out):
        return None
    if not np.array_equal(event_out, step_out):
        return None
    if not np.array_equal(event_out, main(grid)):
        return None

    engine = scene["objects"][scene["engine_idx"]]
    line_cargos = 0
    off_line_cargos = 0
    for idx, obj in enumerate(scene["objects"]):
        if idx == scene["engine_idx"]:
            continue
        if _orthogonally_overlaps(engine, obj, scene["direction"]) and _ahead_of_engine(engine, obj, scene["direction"]):
            line_cargos += 1
        elif not _orthogonally_overlaps(engine, obj, scene["direction"]):
            off_line_cargos += 1
    if line_cargos < 1 or off_line_cargos < 1:
        return None

    if branch == "free":
        if trace["collisions"] != 0 or trace["edge_hit"]:
            return None
    elif branch == "relay":
        if trace["collisions"] < 1 or trace["edge_hit"]:
            return None
        if final_remaining <= 0:
            return None
    else:
        if not trace["edge_hit"]:
            return None

    return grid, branch


def generate_input() -> np.ndarray:
    for _ in range(500):
        generated = _generate_attempt()
        if generated is not None:
            return generated[0]
    raise RuntimeError("failed to generate a valid recoil-engine relay input")
