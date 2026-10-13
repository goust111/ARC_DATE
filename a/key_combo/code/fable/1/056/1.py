from __future__ import annotations

import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_shared_arc_utils_fable_1_056_1", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"unable to load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


BG = 0
SOLID = "solid"
HOLLOW = "hollow"
SIDES = ("top", "bottom", "left", "right")
TURN_TYPES = (SOLID, HOLLOW)
PALETTE = tuple(range(1, 10))

GRID_H_RANGE = (15, 23)
GRID_W_RANGE = (15, 23)
TURN_COUNT_RANGE = (2, 4)
DISTRACTOR_COUNT_RANGE = (1, 3)
MIN_SEGMENT_LEN = 2
MIN_EXIT_LEN = 2
MAX_GENERATION_ATTEMPTS = 500

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

CW = {
    (-1, 0): (0, 1),
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
}
CCW = {value: key for key, value in CW.items()}


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    h, w = map(int, shape)
    r, c = map(int, cell)
    return 0 <= r < h and 0 <= c < w


def _rotate(direction: Coord, turn_kind: str) -> Coord:
    direction = (int(direction[0]), int(direction[1]))
    if turn_kind == SOLID:
        return CW[direction]
    if turn_kind == HOLLOW:
        return CCW[direction]
    raise ValueError(f"unknown turn kind: {turn_kind}")


def _flip_kind(turn_kind: str) -> str:
    return HOLLOW if turn_kind == SOLID else SOLID


def _render_object(turn_kind: str, color: int) -> np.ndarray:
    patch = np.full((3, 3), int(color), dtype=int)
    if turn_kind == HOLLOW:
        patch[1, 1] = BG
    return patch


def _is_hollow_frame(patch: np.ndarray, color: int, bg: int) -> bool:
    return (
        patch.shape == (3, 3)
        and np.all(patch[0, :] == color)
        and np.all(patch[2, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, 2] == color)
        and int(patch[1, 1]) == int(bg)
    )


def _full_bbox_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = map(int, bbox)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _bbox_top_left(bbox: BBox) -> Coord:
    return int(bbox[0]), int(bbox[1])


def _boxes_too_close(a: BBox, b: BBox, gap: int = 1) -> bool:
    gap = int(gap)
    return not (
        int(a[2]) + gap < int(b[0])
        or int(b[2]) + gap < int(a[0])
        or int(a[3]) + gap < int(b[1])
        or int(b[3]) + gap < int(a[1])
    )


def _object_bbox_for_turn(turn_point: Coord, direction: Coord) -> BBox:
    r, c = map(int, turn_point)
    dr, dc = map(int, direction)
    if (dr, dc) == (1, 0):
        return r + 1, c - 1, r + 3, c + 1
    if (dr, dc) == (-1, 0):
        return r - 3, c - 1, r - 1, c + 1
    if (dr, dc) == (0, 1):
        return r - 1, c + 1, r + 1, c + 3
    if (dr, dc) == (0, -1):
        return r - 1, c - 3, r + 1, c - 1
    raise ValueError(f"invalid direction: {direction}")


def _bbox_fits_field(bbox: BBox, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    return 1 <= r0 and 1 <= c0 and r1 <= h - 2 and c1 <= w - 2


def _seed_direction(seed: Coord, shape: tuple[int, int]) -> Coord:
    h, w = map(int, shape)
    r, c = map(int, seed)
    edge_count = int(r in (0, h - 1)) + int(c in (0, w - 1))
    assert edge_count == 1, "seed must lie on exactly one border edge"
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    return 0, -1


def _parse_scene(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    assert bg == BG, "expected black background"

    seed_candidates: list[dict] = []
    objects: list[dict] = []
    for comp in find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        color = int(comp["color"])
        area = int(comp["area"])
        bbox = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(g, bbox)
        if (
            bbox_h(bbox) == 3
            and bbox_w(bbox) == 3
            and area == 9
            and bool(np.all(patch == color))
        ):
            kind = SOLID
        elif bbox_h(bbox) == 3 and bbox_w(bbox) == 3 and area == 8 and _is_hollow_frame(patch, color, bg):
            kind = HOLLOW
        else:
            assert area == 1, "unexpected foreground component"
            coord = tuple(map(int, comp["coords"][0]))
            assert coord[0] in (0, h - 1) or coord[1] in (0, w - 1), "seed must be on the border"
            seed_candidates.append({"coord": coord, "color": color})
            continue

        assert _bbox_fits_field(bbox, g.shape), "objects must stay inside the field"
        objects.append(
            {
                "bbox": bbox,
                "kind": kind,
                "color": color,
                "cells": {tuple(map(int, cell)) for cell in comp["coords"]},
            }
        )

    assert len(seed_candidates) == 1, "expected exactly one border seed pixel"
    assert objects, "expected at least one 3x3 turning object"
    return bg, seed_candidates[0], objects


def _trace_beam(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    _bg, seed, objects = _parse_scene(g)

    object_by_cell: dict[Coord, int] = {}
    for idx, obj in enumerate(objects):
        for cell in obj["cells"]:
            object_by_cell[cell] = int(idx)

    direction = _seed_direction(seed["coord"], (h, w))
    out = g.copy()
    path: list[Coord] = [seed["coord"]]
    hits: list[int] = []
    pos = seed["coord"]
    seen_states: set[tuple[Coord, Coord]] = set()

    for _ in range(h * w * 8):
        state = (pos, direction)
        assert state not in seen_states, "beam loop detected"
        seen_states.add(state)

        nr = int(pos[0] + direction[0])
        nc = int(pos[1] + direction[1])
        if not _in_bounds((h, w), (nr, nc)):
            break

        obj_idx = object_by_cell.get((nr, nc))
        if obj_idx is not None:
            hits.append(int(obj_idx))
            direction = _rotate(direction, objects[obj_idx]["kind"])
            continue

        pos = (nr, nc)
        out[nr, nc] = int(seed["color"])
        path.append(pos)
    else:
        raise AssertionError("beam did not leave the grid")

    return out, {"seed": seed, "objects": objects, "hits": hits, "path": path}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the edge seed's beam through the field, turning at 3x3 solids and hollow frames until the beam exits.

    Input structure:
    - The background is black, with exactly one colored seed pixel on a non-corner border cell.
    - Inside the field are separated monochrome 3x3 objects: either solid 3x3 squares or hollow 3x3 square frames.
    - Object colors are arbitrary labels; only each object's filled-vs-hollow topology matters.

    Rule:
    - Launch a beam from the seed perpendicular to its border edge and paint every traversed non-object cell with the seed color.
    - Whenever the next cell straight ahead belongs to a 3x3 object, stop on the current cell and turn there.
    - Solid 3x3 squares turn the beam 90 degrees clockwise, while hollow 3x3 frames turn it 90 degrees counterclockwise; continue chaining turns until the beam leaves the grid.

    Key insight:
    - The 3x3 objects are turning devices rather than blockers, and the beam's full polyline is determined step-by-step by repeatedly reading each encountered object's topology.
    """
    out, _meta = _trace_beam(grid)
    return out


def _sample_seed(shape: tuple[int, int], side: str) -> Coord:
    h, w = map(int, shape)
    if side == "top":
        return 0, random.randint(2, w - 3)
    if side == "bottom":
        return h - 1, random.randint(2, w - 3)
    if side == "left":
        return random.randint(2, h - 3), 0
    if side == "right":
        return random.randint(2, h - 3), w - 1
    raise ValueError(f"invalid side: {side}")


def _sample_turn_types(count: int) -> list[str]:
    kinds = [SOLID, HOLLOW]
    while len(kinds) < int(count):
        kinds.append(random.choice(TURN_TYPES))
    random.shuffle(kinds)
    return kinds


def _exit_cells(
    shape: tuple[int, int],
    pos: Coord,
    direction: Coord,
    path_cells: set[Coord],
    reserved_cells: set[Coord],
) -> list[Coord] | None:
    cells: list[Coord] = []
    r, c = map(int, pos)
    dr, dc = map(int, direction)
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, (r, c)):
            break
        if (r, c) in path_cells or (r, c) in reserved_cells:
            return None
        cells.append((r, c))
    if len(cells) < MIN_EXIT_LEN:
        return None
    return cells


def _build_path_plan(shape: tuple[int, int], seed: Coord, turn_types: list[str]) -> dict | None:
    h, w = map(int, shape)
    start_direction = _seed_direction(seed, (h, w))

    def dfs(
        pos: Coord,
        direction: Coord,
        turn_idx: int,
        path: list[Coord],
        path_cells: set[Coord],
        bboxes: list[BBox],
        reserved_cells: set[Coord],
        turns: list[dict],
    ) -> dict | None:
        if turn_idx == len(turn_types):
            exit_segment = _exit_cells((h, w), pos, direction, path_cells, reserved_cells)
            if exit_segment is None:
                return None
            return {"path": path + exit_segment, "turns": turns}

        kind = turn_types[turn_idx]
        candidates: list[tuple[list[Coord], BBox, Coord, Coord]] = []
        for dist in range(MIN_SEGMENT_LEN, max(h, w)):
            turn_point = (
                int(pos[0] + direction[0] * dist),
                int(pos[1] + direction[1] * dist),
            )
            if not _in_bounds((h, w), turn_point):
                break

            segment_cells = [
                (
                    int(pos[0] + direction[0] * step),
                    int(pos[1] + direction[1] * step),
                )
                for step in range(1, dist + 1)
            ]
            if any(cell in path_cells or cell in reserved_cells for cell in segment_cells):
                continue

            bbox = _object_bbox_for_turn(turn_point, direction)
            if not _bbox_fits_field(bbox, (h, w)):
                continue
            if any(_boxes_too_close(bbox, other, gap=1) for other in bboxes):
                continue

            bbox_cells = _full_bbox_cells(bbox)
            if bbox_cells & path_cells or bbox_cells & reserved_cells:
                continue

            new_direction = _rotate(direction, kind)
            first_step = (int(turn_point[0] + new_direction[0]), int(turn_point[1] + new_direction[1]))
            if not _in_bounds((h, w), first_step):
                continue
            if first_step in path_cells or first_step in reserved_cells or first_step in bbox_cells:
                continue

            candidates.append((segment_cells, bbox, new_direction, turn_point))

        random.shuffle(candidates)
        for segment_cells, bbox, new_direction, turn_point in candidates:
            bbox_cells = _full_bbox_cells(bbox)
            result = dfs(
                turn_point,
                new_direction,
                turn_idx + 1,
                path + segment_cells,
                path_cells | set(segment_cells),
                bboxes + [bbox],
                reserved_cells | bbox_cells,
                turns + [{"bbox": bbox, "kind": kind, "turn_point": turn_point}],
            )
            if result is not None:
                return result
        return None

    return dfs(
        seed,
        start_direction,
        0,
        [seed],
        {seed},
        [],
        set(),
        [],
    )


def _place_distractors(
    shape: tuple[int, int],
    path_cells: set[Coord],
    used_bboxes: list[BBox],
    colors: list[int],
) -> list[dict] | None:
    h, w = map(int, shape)
    distractors: list[dict] = []
    current_bboxes = list(used_bboxes)
    for color in colors:
        candidates: list[tuple[BBox, str]] = []
        for r0 in range(1, h - 3):
            for c0 in range(1, w - 3):
                bbox = (r0, c0, r0 + 2, c0 + 2)
                if any(_boxes_too_close(bbox, other, gap=1) for other in current_bboxes):
                    continue
                if _full_bbox_cells(bbox) & path_cells:
                    continue
                candidates.append((bbox, random.choice(TURN_TYPES)))
        if not candidates:
            return None
        bbox, kind = random.choice(candidates)
        distractors.append({"bbox": bbox, "kind": kind, "color": int(color), "on_path": False})
        current_bboxes.append(bbox)
    return distractors


def _replace_object(grid: np.ndarray, bbox: BBox, kind: str, color: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0 = _bbox_top_left(bbox)
    out[r0 : r0 + 3, c0 : c0 + 3] = _render_object(kind, color)
    return out


def _validate_generated(grid: np.ndarray, planned_turns: list[dict], placed_objects: list[dict]) -> bool:
    try:
        expected = main(grid)
        _repeat = main(grid.copy())
        if not np.array_equal(expected, _repeat):
            return False
        _out, meta = _trace_beam(grid)
    except Exception:
        return False

    actual_hits = [tuple(map(int, meta["objects"][idx]["bbox"])) for idx in meta["hits"]]
    planned_hits = [tuple(map(int, turn["bbox"])) for turn in planned_turns]
    if actual_hits != planned_hits:
        return False

    actual_kinds = [meta["objects"][idx]["kind"] for idx in meta["hits"]]
    planned_kinds = [turn["kind"] for turn in planned_turns]
    if actual_kinds != planned_kinds:
        return False

    if {SOLID, HOLLOW} - set(actual_kinds):
        return False
    if len(meta["path"]) != len(set(meta["path"])) or len(meta["path"]) < 7:
        return False
    if np.array_equal(expected, np.asarray(grid, dtype=int)):
        return False

    on_path_objects = [obj for obj in placed_objects if obj["on_path"]]
    off_path_objects = [obj for obj in placed_objects if not obj["on_path"]]

    for obj in on_path_objects:
        flipped = _replace_object(grid, obj["bbox"], _flip_kind(obj["kind"]), obj["color"])
        try:
            _flipped_out, flipped_meta = _trace_beam(flipped)
        except Exception:
            return False
        flipped_hits = [tuple(map(int, flipped_meta["objects"][idx]["bbox"])) for idx in flipped_meta["hits"]]
        if flipped_meta["path"] == meta["path"] and flipped_hits == actual_hits:
            return False

    for obj in off_path_objects:
        flipped = _replace_object(grid, obj["bbox"], _flip_kind(obj["kind"]), obj["color"])
        try:
            _flipped_out, flipped_meta = _trace_beam(flipped)
        except Exception:
            return False
        flipped_hits = [tuple(map(int, flipped_meta["objects"][idx]["bbox"])) for idx in flipped_meta["hits"]]
        if flipped_meta["path"] != meta["path"] or flipped_hits != actual_hits:
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    turn_count = random.randint(*TURN_COUNT_RANGE)
    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    side = random.choice(SIDES)
    seed = _sample_seed((h, w), side)
    turn_types = _sample_turn_types(turn_count)

    plan = _build_path_plan((h, w), seed, turn_types)
    if plan is None:
        return None

    total_colors = 1 + turn_count + distractor_count
    if total_colors > len(PALETTE):
        return None
    chosen_colors = random.sample(PALETTE, total_colors)
    seed_color = int(chosen_colors[0])
    object_colors = [int(color) for color in chosen_colors[1:]]
    turn_colors = object_colors[:turn_count]
    distractor_colors = object_colors[turn_count:]

    distractors = _place_distractors(
        (h, w),
        set(plan["path"]),
        [tuple(map(int, turn["bbox"])) for turn in plan["turns"]],
        distractor_colors,
    )
    if distractors is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[seed] = seed_color

    placed_objects: list[dict] = []
    for turn, color in zip(plan["turns"], turn_colors):
        bbox = tuple(map(int, turn["bbox"]))
        r0, c0 = _bbox_top_left(bbox)
        paste(grid, _render_object(turn["kind"], color), r0, c0)
        placed_objects.append(
            {
                "bbox": bbox,
                "kind": turn["kind"],
                "color": color,
                "on_path": True,
            }
        )

    for distractor in distractors:
        bbox = tuple(map(int, distractor["bbox"]))
        r0, c0 = _bbox_top_left(bbox)
        paste(grid, _render_object(distractor["kind"], distractor["color"]), r0, c0)
        placed_objects.append(distractor)

    if not _validate_generated(grid, plan["turns"], placed_objects):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
