from __future__ import annotations

import itertools
import math
import random
import sys

import numpy as np

UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

from arc_utils import Color, bbox_coords, components_mask, scale_sprite


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREY = Color.GREY

BASE_SIZE = 3
MIN_INSTANCES = 3
MAX_INSTANCES = 4
GRID_MIN = 24
GRID_MAX = 30
INSTANCE_GAP = 2
GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 250

DIRS = {
    "right": (0, 1),
    "down": (1, 0),
}


def _connected4(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS.values():
            for rr, cc in ((r + dr, c + dc), (r - dr, c - dc)):
                if (rr, cc) in cells and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    stack.append((rr, cc))
    return seen == cells


def _enumerate_prototypes() -> list[dict]:
    prototypes: list[dict] = []
    cells_all = [(r, c) for r in range(BASE_SIZE) for c in range(BASE_SIZE)]
    for n_cells in range(4, 7):
        for subset in itertools.combinations(cells_all, n_cells):
            cell_set = set(subset)
            if not _connected4(cell_set):
                continue
            rr = [r for r, _c in cell_set]
            cc = [c for _r, c in cell_set]
            if min(rr) != 0 or max(rr) != 2 or min(cc) != 0 or max(cc) != 2:
                continue
            active_dirs = []
            external_dirs = []
            if (0, 1) not in cell_set:
                active_dirs.append("right")
                if (0, 2) not in cell_set:
                    external_dirs.append("right")
            if (1, 0) not in cell_set:
                active_dirs.append("down")
                if (2, 0) not in cell_set:
                    external_dirs.append("down")
            if not active_dirs:
                continue
            mask = np.zeros((BASE_SIZE, BASE_SIZE), dtype=int)
            for r, c in cell_set:
                mask[r, c] = 1
            prototypes.append(
                {
                    "mask": mask,
                    "red_cells": tuple(sorted(cell_set)),
                    "active_dirs": tuple(active_dirs),
                    "external_dirs": tuple(external_dirs),
                }
            )
    return prototypes


PROTOTYPES = _enumerate_prototypes()
assert PROTOTYPES, "expected valid 3x3 prototypes"

_SCALED_RED_CACHE: dict[tuple[bytes, int], tuple[tuple[int, int], ...]] = {}


def _scaled_red_coords(proto: dict, scale: int) -> tuple[tuple[int, int], ...]:
    key = (proto["mask"].tobytes(), int(scale))
    if key not in _SCALED_RED_CACHE:
        scaled = scale_sprite(proto["mask"], int(scale))
        _SCALED_RED_CACHE[key] = tuple(map(tuple, np.argwhere(scaled > 0)))
    return _SCALED_RED_CACHE[key]


def _instance_record(
    proto: dict, seed_dirs: tuple[str, ...], scale: int, top: int, left: int, idx: int
) -> dict:
    red = {(top + r, left + c) for r, c in _scaled_red_coords(proto, scale)}
    seeds = set()
    if "right" in seed_dirs:
        seeds.add((top, left + 1))
    if "down" in seed_dirs:
        seeds.add((top + 1, left))
    assert red.isdisjoint(seeds)
    span = BASE_SIZE * int(scale)
    return {
        "id": int(idx),
        "scale": int(scale),
        "top": int(top),
        "left": int(left),
        "bbox": (int(top), int(left), int(top + span - 1), int(left + span - 1)),
        "red": red,
        "seeds": seeds,
    }


def _bbox_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _cells_in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _can_place_instance(
    inst: dict, instances: list[dict], occupied: set[tuple[int, int]], shape: tuple[int, int]
) -> bool:
    h, w = shape
    r0, c0, r1, c1 = inst["bbox"]
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    all_cells = inst["red"] | inst["seeds"]
    if not _cells_in_bounds(shape, all_cells):
        return False
    if all_cells & occupied:
        return False
    for other in instances:
        if not _bbox_far_enough(inst["bbox"], other["bbox"], INSTANCE_GAP):
            return False
    return True


def _red_components_with_scale(grid: np.ndarray) -> list[dict]:
    comps = components_mask(grid == RED, connectivity=4)
    instances = []
    min_bbox_area = None
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = bbox_coords(comp)
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        min_bbox_area = bbox_area if min_bbox_area is None else min(min_bbox_area, bbox_area)
        instances.append(
            {
                "id": idx,
                "coords": {tuple(map(int, rc)) for rc in comp},
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "bbox_area": int(bbox_area),
            }
        )
    if not instances:
        return []
    for inst in instances:
        ratio = inst["bbox_area"] / min_bbox_area
        inst["scale"] = int(round(math.sqrt(ratio)))
    instances.sort(key=lambda inst: (inst["scale"], inst["bbox"][0], inst["bbox"][1]))
    return instances


def _active_dirs_from_smallest(grid: np.ndarray, smallest: dict) -> tuple[str, ...]:
    h, w = grid.shape
    r0, c0, _r1, _c1 = smallest["bbox"]
    dirs = []
    if c0 + 1 < w and int(grid[r0, c0 + 1]) == BLUE:
        dirs.append("right")
    if r0 + 1 < h and int(grid[r0 + 1, c0]) == BLUE:
        dirs.append("down")
    return tuple(dirs)


def _block_top_left(inst: dict, direction: str, step: int) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return (
        int(inst["bbox"][0] + step * inst["scale"] * dr),
        int(inst["bbox"][1] + step * inst["scale"] * dc),
    )


def _block_in_bounds(shape: tuple[int, int], top: int, left: int, scale: int) -> bool:
    h, w = shape
    return 0 <= top and 0 <= left and top + scale <= h and left + scale <= w


def _block_cells(top: int, left: int, scale: int) -> set[tuple[int, int]]:
    return {
        (top + dr, left + dc)
        for dr in range(scale)
        for dc in range(scale)
    }


def _paint_block(out: np.ndarray, top: int, left: int, scale: int, color: int) -> None:
    out[top : top + scale, left : left + scale] = int(color)


def _apply_shadow_copy(
    grid: np.ndarray, instances: list[dict], active_dirs: tuple[str, ...]
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    obstacle = (grid == RED) | (grid == GREY)
    for inst in sorted(instances, key=lambda cur: (cur["scale"], cur["bbox"][0], cur["bbox"][1])):
        for direction in active_dirs:
            step = 1
            while True:
                top, left = _block_top_left(inst, direction, step)
                if not _block_in_bounds(grid.shape, top, left, inst["scale"]):
                    break
                if bool(obstacle[top : top + inst["scale"], left : left + inst["scale"]].any()):
                    break
                _paint_block(out, top, left, inst["scale"], BLUE)
                step += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend scale-controlled blue shadows from a shared sprite prototype until red or grey blockers cut them off.

    Input structure:
    - A black grid contains 3 or 4 red 4-connected sprites that are integer-scaled copies of one 3x3 prototype.
    - The smallest sprite defines the unit scale; each sprite bbox is a 3k x 3k scaled copy.
    - Near the smallest bbox anchor (its top-left bbox coordinate), blue seed pixels can appear one cell to the right and/or one cell below; grey single-pixel distractors may also be present.

    Rule:
    - Detect every red component and infer its scale factor k from the square root of its bbox-area ratio against the smallest red bbox.
    - Read the active shadow directions from the smallest instance: a blue seed at (anchor_r, anchor_c+1) activates the rightward shadow, and a blue seed at (anchor_r+1, anchor_c) activates the downward shadow.
    - For each instance, repeat each active direction with step vector multiplied by k; at every step place a blue k x k block at the stepped location.
    - Stop a direction for that instance as soon as the next whole k x k block would overlap any red sprite pixel or any grey distractor; existing blue pixels never block later blocks.

    Key insight:
    - The smallest sprite supplies the prototype-level shadow offsets, while each larger sprite turns the same offsets into longer jumps and larger blue blocks, so the final extent depends jointly on scale and global clearance.
    """
    g = np.asarray(grid, dtype=int)
    instances = _red_components_with_scale(g)
    if not instances:
        return g.copy()
    active_dirs = _active_dirs_from_smallest(g, instances[0])
    if not active_dirs:
        return g.copy()
    return _apply_shadow_copy(g, instances, active_dirs)


def _empty_grid(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, BG, dtype=int)


def _scene_input(scene: dict) -> np.ndarray:
    grid = _empty_grid(scene["shape"])
    for inst in scene["instances"]:
        for r, c in inst["red"]:
            grid[r, c] = RED
    for inst in scene["instances"]:
        for r, c in inst["seeds"]:
            grid[r, c] = BLUE
    for r, c in scene["greys"]:
        grid[r, c] = GREY
    return grid


def _scene_expected_output(scene: dict) -> np.ndarray:
    grid = _scene_input(scene)
    return _apply_shadow_copy(grid, scene["instances"], scene["active_dirs"])


def _path_reports(scene: dict) -> list[dict]:
    shape = scene["shape"]
    red_owner: dict[tuple[int, int], int] = {}
    for inst in scene["instances"]:
        for cell in inst["red"]:
            red_owner[cell] = inst["id"]
    reports: list[dict] = []
    for inst in sorted(scene["instances"], key=lambda cur: (cur["scale"], cur["bbox"][0], cur["bbox"][1])):
        for direction in scene["active_dirs"]:
            drawn = 0
            step = 1
            stop_reason = "oob"
            blocker_owner = None
            free_blocks: list[tuple[int, int, set[tuple[int, int]]]] = []
            while True:
                top, left = _block_top_left(inst, direction, step)
                if not _block_in_bounds(shape, top, left, inst["scale"]):
                    stop_reason = "oob"
                    break
                cells = _block_cells(top, left, inst["scale"])
                gray_overlap = cells & scene["greys"]
                red_overlap = [cell for cell in cells if cell in red_owner]
                if red_overlap or gray_overlap:
                    if red_overlap:
                        owners = {red_owner[cell] for cell in red_overlap}
                        blocker_owner = next(iter(owners))
                        stop_reason = "self_red" if blocker_owner == inst["id"] else "other_red"
                    else:
                        stop_reason = "grey"
                    break
                free_blocks.append((step, top, cells))
                drawn += 1
                step += 1
            reports.append(
                {
                    "instance_id": inst["id"],
                    "scale": inst["scale"],
                    "direction": direction,
                    "drawn": drawn,
                    "stop_reason": stop_reason,
                    "blocker_owner": blocker_owner,
                    "free_blocks": free_blocks,
                }
            )
    return reports


def _choose_prototype_and_dirs() -> tuple[dict, tuple[str, ...]]:
    prototypes = [proto for proto in PROTOTYPES if proto["external_dirs"]]
    proto = random.choice(prototypes)
    if len(proto["external_dirs"]) == 1:
        return proto, (proto["external_dirs"][0],)
    if random.random() < 0.5:
        return proto, tuple(proto["external_dirs"])
    return proto, (random.choice(proto["external_dirs"]),)


def _random_place_instance(
    proto: dict,
    seed_dirs: tuple[str, ...],
    travel_dirs: tuple[str, ...],
    scale: int,
    shape: tuple[int, int],
    instances: list[dict],
    occupied: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    span = BASE_SIZE * scale
    candidates: list[tuple[float, dict]] = []
    for top in range(0, h - span + 1):
        for left in range(0, w - span + 1):
            inst = _instance_record(proto, seed_dirs, scale, top, left, len(instances))
            if not _can_place_instance(inst, instances, occupied, shape):
                continue
            score = 0.0
            if "right" in travel_dirs:
                score += (w - (left + span)) / max(1, scale)
            if "down" in travel_dirs:
                score += (h - (top + span)) / max(1, scale)
            score += random.random()
            candidates.append((score, inst))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    pool = [inst for _score, inst in candidates[: min(35, len(candidates))]]
    return random.choice(pool)


def _targeted_block_pair(
    proto: dict,
    active_dirs: tuple[str, ...],
    shape: tuple[int, int],
    target_scale: int,
) -> tuple[dict, dict, str] | None:
    external_dirs = [direction for direction in active_dirs if direction in proto["external_dirs"]]
    if not external_dirs:
        return None
    h, w = shape
    source_choices = []
    for direction in external_dirs:
        if direction == "right":
            for top in range(1, h - BASE_SIZE):
                for left in range(1, max(2, w - 8)):
                    source_choices.append((direction, top, left))
        else:
            for top in range(1, max(2, h - 8)):
                for left in range(1, w - BASE_SIZE):
                    source_choices.append((direction, top, left))
    random.shuffle(source_choices)
    for direction, top, left in source_choices[:180]:
        source = _instance_record(proto, active_dirs, 1, top, left, 0)
        if not _can_place_instance(source, [], set(), shape):
            continue
        target_rel = list(_scaled_red_coords(proto, target_scale))
        if direction == "right":
            preferred = [rc for rc in target_rel if rc[0] == 0] or target_rel
        else:
            preferred = [rc for rc in target_rel if rc[1] == 0] or target_rel
        random.shuffle(preferred)
        max_steps = 0
        if direction == "right":
            max_steps = w - 1 - left
        else:
            max_steps = h - 1 - top
        for step in range(min(7, max_steps), 3, -1):
            block_top, block_left = _block_top_left(source, direction, step)
            if not _block_in_bounds(shape, block_top, block_left, 1):
                continue
            for rr, cc in preferred:
                target = _instance_record(
                    proto,
                    (),
                    target_scale,
                    block_top - rr,
                    block_left - cc,
                    1,
                )
                occupied = source["red"] | source["seeds"]
                if _can_place_instance(target, [source], occupied, shape):
                    scene = {
                        "shape": shape,
                        "active_dirs": active_dirs,
                        "instances": [
                            source,
                            {**target, "id": 1},
                        ],
                        "greys": set(),
                    }
                    reports = _path_reports(scene)
                    source_report = next(
                        report
                        for report in reports
                        if report["instance_id"] == 0 and report["direction"] == direction
                    )
                    if source_report["stop_reason"] == "other_red" and source_report["drawn"] >= 2:
                        return source, {**target, "id": 1}, direction
    return None


def _add_grey_blocker(scene: dict) -> bool:
    reports = _path_reports(scene)
    candidate_blocks = []
    all_other_cells: list[set[tuple[int, int]]] = []
    for report in reports:
        for step, _top, cells in report["free_blocks"]:
            all_other_cells.append(cells)
    for report in reports:
        if report["drawn"] < 3:
            continue
        for step, top, cells in report["free_blocks"]:
            if step < 3:
                continue
            unique = True
            for other in all_other_cells:
                if other is cells:
                    continue
                if cells & other:
                    unique = False
                    break
            if unique:
                candidate_blocks.append((report, step, top, cells))
    random.shuffle(candidate_blocks)
    occupied = set()
    for inst in scene["instances"]:
        occupied |= inst["red"] | inst["seeds"]
    for _report, _step, _top, cells in candidate_blocks:
        free_cells = sorted(cells - occupied - scene["greys"])
        if not free_cells:
            continue
        scene["greys"].add(random.choice(free_cells))
        return True
    return False


def _add_extra_greys(scene: dict) -> None:
    reports = _path_reports(scene)
    forbidden = set(scene["greys"])
    for inst in scene["instances"]:
        forbidden |= inst["red"] | inst["seeds"]
    for report in reports:
        for _step, _top, cells in report["free_blocks"]:
            forbidden |= cells
    h, w = scene["shape"]
    empty = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in forbidden
    ]
    random.shuffle(empty)
    extra = random.randint(1, 3)
    for cell in empty[:extra]:
        scene["greys"].add(cell)


def _scene_quality(scene: dict) -> bool:
    reports = _path_reports(scene)
    if not reports:
        return False
    drawn_counts = [report["drawn"] for report in reports]
    if max(drawn_counts) < 3:
        return False
    if len(set(drawn_counts)) < 2:
        return False
    reasons = {report["stop_reason"] for report in reports}
    if "grey" not in reasons:
        return False
    if "other_red" not in reasons and "self_red" not in reasons:
        return False
    if sum(drawn_counts) < 5:
        return False
    if not scene["greys"]:
        return False
    inp = _scene_input(scene)
    out = _scene_expected_output(scene)
    if np.array_equal(inp, out):
        return False
    return True


def _assemble_scene() -> dict | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    proto, active_dirs = _choose_prototype_and_dirs()
    target_scale = random.choice([2, 3])
    pair = _targeted_block_pair(proto, active_dirs, shape, target_scale)
    if pair is None:
        return None
    source, target, red_block_dir = pair
    instances = [source, target]
    occupied = source["red"] | source["seeds"] | target["red"] | target["seeds"]

    remaining_scales = [scale for scale in [2, 3] if scale != target_scale]
    if random.random() < 0.55:
        remaining_scales.append(random.choice([2, 3]))
    while len(remaining_scales) + len(instances) > MAX_INSTANCES:
        remaining_scales.pop()

    for scale in remaining_scales:
        inst = _random_place_instance(proto, (), active_dirs, scale, shape, instances, occupied)
        if inst is None:
            return None
        inst["id"] = len(instances)
        instances.append(inst)
        occupied |= inst["red"] | inst["seeds"]

    scene = {
        "shape": shape,
        "proto": proto,
        "active_dirs": active_dirs,
        "instances": instances,
        "greys": set(),
        "red_block_dir": red_block_dir,
    }
    if not _add_grey_blocker(scene):
        return None
    _add_extra_greys(scene)
    if not _scene_quality(scene):
        return None
    inp = _scene_input(scene)
    if not np.array_equal(main(inp), _scene_expected_output(scene)):
        return None
    return scene


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scene = _assemble_scene()
        if scene is not None:
            return _scene_input(scene)
    raise RuntimeError("failed to generate a valid puzzle instance")
