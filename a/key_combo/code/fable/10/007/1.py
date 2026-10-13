from __future__ import annotations

from collections import Counter, deque
import math
import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
OBJECT_COLOR_POOL = [1, 2, 5, 6, 7, 8, 9]


# Local templates are expressed as (perpendicular_offset, along_track_offset)
# relative to the unique anchor cell at (0, 0). Only the anchor has
# perpendicular_offset == 0, so the occupied slot is always uniquely recoverable.
SHAPE_TEMPLATES = {
    1: [
        ((0, 0),),
    ],
    2: [
        ((0, 0), (-1, 0)),
    ],
    3: [
        ((0, 0), (-1, 0), (-2, 0)),
        ((0, 0), (-1, 0), (-1, 1)),
        ((0, 0), (-1, 0), (1, 0)),
    ],
    4: [
        ((0, 0), (-1, 0), (-2, 0), (-1, 1)),
        ((0, 0), (-1, 0), (1, 0), (-1, 1)),
        ((0, 0), (-1, 0), (-2, 0), (-3, 0)),
    ],
    5: [
        ((0, 0), (-1, 0), (-2, 0), (-1, -1), (-1, 1)),
        ((0, 0), (-1, 0), (-2, 0), (-2, 1), (-3, 1)),
        ((0, 0), (-1, 0), (1, 0), (-1, 1), (1, 1)),
    ],
    6: [
        ((0, 0), (-1, 0), (-2, 0), (-3, 0), (-1, -1), (-1, 1)),
        ((0, 0), (-1, 0), (-2, 0), (-2, -1), (-2, 1), (-3, 0)),
        ((0, 0), (-1, 0), (1, 0), (-1, 1), (1, 1), (2, 1)),
    ],
}


TRACK_CHOICES = [(3, 6), (3, 7), (3, 8), (4, 6)]
SCENARIOS = (
    "mixed",
    "mixed",
    "collision",
    "collision",
    "endpoint",
    "clear",
)
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in {bg, GREEN, YELLOW}:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells = [(r, c)]
            while queue:
                cr, cc = queue.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
                    cells.append((nr, nc))
            coords = np.array(cells, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": int(coords.shape[0]),
                }
            )
    return objects


def _extract_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _get_bg(g)

    yellow = np.argwhere(g == YELLOW)
    green = np.argwhere(g == GREEN)
    assert yellow.shape == (1, 2), "expected exactly one yellow endpoint"
    assert green.shape[0] >= 1, "expected visible green track dots"

    track_pixels = np.vstack([green, yellow])
    if np.all(track_pixels[:, 0] == track_pixels[0, 0]):
        axis = "h"
        fixed = int(track_pixels[0, 0])
        visible_positions = sorted({int(c) for _r, c in track_pixels})
        yellow_pos = int(yellow[0, 1])
    elif np.all(track_pixels[:, 1] == track_pixels[0, 1]):
        axis = "v"
        fixed = int(track_pixels[0, 1])
        visible_positions = sorted({int(r) for r, _c in track_pixels})
        yellow_pos = int(yellow[0, 0])
    else:
        raise AssertionError("track dots must form a straight row or column")

    assert len(visible_positions) >= 2, "need at least two visible track marks"
    step = 0
    for a, b in zip(visible_positions, visible_positions[1:]):
        step = math.gcd(step, b - a)
    assert step > 0, "track spacing must be positive"

    objects = _find_objects(g, bg)
    assert objects, "expected at least one foreground object"
    areas = [obj["area"] for obj in objects]
    assert len(set(areas)) == len(areas), "object areas must be distinct"

    anchor_positions = []
    for obj in objects:
        coords = obj["coords"]
        candidates = []
        for r, c in coords:
            if axis == "h":
                on_track = int(r) == fixed
                pos = int(c)
            else:
                on_track = int(c) == fixed
                pos = int(r)
            if on_track and (pos - yellow_pos) % step == 0:
                candidates.append((int(r), int(c), pos))
        assert len(candidates) == 1, "each object must occupy exactly one track slot"
        ar, ac, anchor_pos = candidates[0]
        obj["anchor"] = (ar, ac)
        obj["slot_pos"] = anchor_pos
        anchor_positions.append(anchor_pos)

    all_positions = sorted(set(visible_positions + anchor_positions))
    assert len(all_positions) == len(visible_positions) + len(objects)
    if yellow_pos == all_positions[-1]:
        start = all_positions[0]
        sign = 1
    elif yellow_pos == all_positions[0]:
        start = all_positions[-1]
        sign = -1
    else:
        raise AssertionError("yellow endpoint must be at a track end")

    span = abs(yellow_pos - start)
    assert span % step == 0
    slot_count = span // step + 1
    slot_positions = [start + sign * step * i for i in range(slot_count)]
    assert set(slot_positions) == set(all_positions), "all slots must be represented by dots or anchored objects"

    index_by_pos = {pos: i for i, pos in enumerate(slot_positions)}
    for obj in objects:
        obj["slot_index"] = index_by_pos[obj["slot_pos"]]

    return {
        "shape": tuple(map(int, g.shape)),
        "bg": bg,
        "axis": axis,
        "fixed": fixed,
        "step": step,
        "slot_positions": slot_positions,
        "objects": objects,
    }


def _simulate_slots(objects: list[dict], slot_count: int, *, descending: bool = True, unit_steps: bool = False) -> dict:
    order = sorted(
        range(len(objects)),
        key=lambda idx: (-objects[idx]["area"], objects[idx]["slot_index"]) if descending else (objects[idx]["area"], objects[idx]["slot_index"]),
    )
    slots = [int(obj["slot_index"]) for obj in objects]
    occupied = {slot: idx for idx, slot in enumerate(slots)}
    events = []

    for idx in order:
        start = slots[idx]
        occupied.pop(start)
        dest = start
        blocked = False
        capped = False
        steps = 1 if unit_steps else int(objects[idx]["area"])
        for _ in range(steps):
            cand = dest + 1
            if cand >= slot_count:
                capped = True
                break
            if cand in occupied:
                blocked = True
                break
            dest = cand
            if dest == slot_count - 1:
                capped = True
                break
        slots[idx] = dest
        occupied[dest] = idx
        events.append(
            {
                "object": idx,
                "start": start,
                "end": dest,
                "blocked": blocked,
                "capped": capped,
            }
        )

    return {"slots": tuple(slots), "events": events}


def _render(layout: dict, final_slots: tuple[int, ...]) -> np.ndarray:
    out = np.full(layout["shape"], layout["bg"], dtype=int)
    axis = layout["axis"]
    fixed = int(layout["fixed"])
    slot_positions = layout["slot_positions"]

    for i, pos in enumerate(slot_positions):
        if axis == "h":
            out[fixed, pos] = YELLOW if i == len(slot_positions) - 1 else GREEN
        else:
            out[pos, fixed] = YELLOW if i == len(slot_positions) - 1 else GREEN

    for obj, new_slot in zip(layout["objects"], final_slots):
        old_pos = slot_positions[int(obj["slot_index"])]
        new_pos = slot_positions[int(new_slot)]
        delta = new_pos - old_pos
        coords = obj["coords"]
        rr = coords[:, 0] + (delta if axis == "v" else 0)
        cc = coords[:, 1] + (delta if axis == "h" else 0)
        out[rr, cc] = int(obj["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide every anchored object toward the yellow endpoint by a number of track slots equal to its area, with larger objects moving first.

    Input structure:
    - A black background contains a straight equally spaced track of green dots, with the terminal track dot colored yellow.
    - Several monochrome 4-connected foreground objects of colors other than black, green, and yellow each cover exactly one track slot.
    - The track is horizontal or vertical, and the objects have pairwise distinct areas.

    Rule:
    - Reconstruct the full ordered list of track slots from the visible green dots, the yellow endpoint, and the unique anchored cell of each object.
    - Process objects in descending area order; each object attempts to advance toward the yellow endpoint by exactly its own pixel count in slots.
    - If the next slot on its route is occupied by another object, it stops at the last free slot before that blocker; it can never move past the yellow endpoint.
    - Redraw the track dots, then redraw every object in its final translated position with unchanged shape and color.

    Key insight:
    - The result depends on two coupled discrete structures: area determines how many slots an object tries to traverse, and the descending-area move order turns earlier landings into later obstacles.
    """
    layout = _extract_layout(grid)
    final_slots = _simulate_slots(layout["objects"], len(layout["slot_positions"]))["slots"]
    return _render(layout, final_slots)


def _transform_offsets(template, orientation: str, flip_perp: bool, flip_along: bool):
    out = []
    for perp, along in template:
        p = -perp if flip_perp else perp
        a = -along if flip_along else along
        if orientation == "h":
            out.append((p, a))
        else:
            out.append((a, p))
    return tuple(out)


def _slot_metrics(slot_count: int, areas: list[int], indices: tuple[int, ...]) -> dict:
    raw_objects = [
        {
            "area": int(area),
            "slot_index": int(slot),
        }
        for area, slot in zip(areas, indices)
    ]
    desc = _simulate_slots(raw_objects, slot_count, descending=True)
    asc = _simulate_slots(raw_objects, slot_count, descending=False)
    one = _simulate_slots(raw_objects, slot_count, descending=True, unit_steps=True)

    blocked = any(event["blocked"] for event in desc["events"])
    capped = any(event["capped"] for event in desc["events"])
    moved = any(event["start"] != event["end"] for event in desc["events"])
    order_sensitive = desc["slots"] != asc["slots"]
    step_sensitive = desc["slots"] != one["slots"]
    visible = sorted(set(range(slot_count)) - set(indices))
    has_adjacent_visible = any(b == a + 1 for a, b in zip(visible, visible[1:]))

    return {
        "desc": desc,
        "blocked": blocked,
        "capped": capped,
        "moved": moved,
        "order_sensitive": order_sensitive,
        "step_sensitive": step_sensitive,
        "has_adjacent_visible": has_adjacent_visible,
    }


def _choose_initial_slots(slot_count: int, areas: list[int], scenario: str) -> tuple[int, ...] | None:
    n_objects = len(areas)
    candidates = list(range(slot_count - 1))
    for _ in range(300):
        indices = tuple(random.sample(candidates, n_objects))
        metrics = _slot_metrics(slot_count, areas, indices)
        if not metrics["has_adjacent_visible"] or not metrics["moved"] or not metrics["step_sensitive"]:
            continue

        if scenario == "mixed":
            if metrics["blocked"] and metrics["capped"] and metrics["order_sensitive"]:
                return indices
        elif scenario == "collision":
            if metrics["blocked"] and not metrics["capped"] and metrics["order_sensitive"]:
                return indices
        elif scenario == "endpoint":
            if metrics["capped"] and not metrics["blocked"]:
                return indices
        elif scenario == "clear":
            if not metrics["blocked"] and not metrics["capped"]:
                return indices
    return None


def _build_case() -> tuple[np.ndarray, dict] | None:
    scenario = random.choice(SCENARIOS)
    orientation = random.choice(["h", "v"])
    step, slot_count = random.choice(TRACK_CHOICES)
    n_objects = random.randint(2, min(4, slot_count - 2))
    max_area = min(6, slot_count)
    areas = random.sample(range(1, max_area + 1), n_objects)
    slot_indices = _choose_initial_slots(slot_count, areas, scenario)
    if slot_indices is None:
        return None

    colors = random.sample(OBJECT_COLOR_POOL, n_objects)
    direction_high = bool(random.getrandbits(1))
    objects = []
    row_offsets = []
    col_offsets = []
    for area, color, slot_index in zip(areas, colors, slot_indices):
        template = random.choice(SHAPE_TEMPLATES[area])
        offsets = _transform_offsets(
            template,
            orientation,
            flip_perp=bool(random.getrandbits(1)),
            flip_along=bool(random.getrandbits(1)),
        )
        rows = [dr for dr, _dc in offsets]
        cols = [dc for _dr, dc in offsets]
        row_offsets.extend(rows)
        col_offsets.extend(cols)
        objects.append(
            {
                "area": area,
                "color": color,
                "slot_index": slot_index,
                "offsets": offsets,
            }
        )

    axis_offsets = col_offsets if orientation == "h" else row_offsets
    cross_offsets = row_offsets if orientation == "h" else col_offsets
    axis_req_low = max(0, -min(axis_offsets))
    axis_req_high = max(0, max(axis_offsets))
    cross_req_low = max(0, -min(cross_offsets))
    cross_req_high = max(0, max(cross_offsets))

    axis_low = axis_req_low + random.randint(1, 3)
    axis_high = axis_req_high + random.randint(1, 3)
    cross_low = cross_req_low + random.randint(1, 3)
    cross_high = cross_req_high + random.randint(1, 3)

    low_slot = axis_low
    high_slot = low_slot + step * (slot_count - 1)
    axis_size = high_slot + axis_high + 1
    fixed = cross_low
    cross_size = fixed + cross_high + 1
    if axis_size > 30 or cross_size > 30:
        return None

    if direction_high:
        slot_positions = [low_slot + step * i for i in range(slot_count)]
    else:
        slot_positions = [high_slot - step * i for i in range(slot_count)]

    shape = (cross_size, axis_size) if orientation == "h" else (axis_size, cross_size)
    grid = np.full(shape, BLACK, dtype=int)
    occupied_slots = {obj["slot_index"] for obj in objects}

    for i, pos in enumerate(slot_positions):
        if i in occupied_slots:
            continue
        if orientation == "h":
            grid[fixed, pos] = YELLOW if i == slot_count - 1 else GREEN
        else:
            grid[pos, fixed] = YELLOW if i == slot_count - 1 else GREEN

    occupied_cells = set()
    for obj in objects:
        anchor_pos = slot_positions[int(obj["slot_index"])]
        anchor = (fixed, anchor_pos) if orientation == "h" else (anchor_pos, fixed)
        cells = []
        for dr, dc in obj["offsets"]:
            rr = anchor[0] + dr
            cc = anchor[1] + dc
            if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
                return None
            if (rr, cc) in occupied_cells:
                return None
            occupied_cells.add((rr, cc))
            cells.append((rr, cc))
        for rr, cc in cells:
            grid[rr, cc] = int(obj["color"])
        obj["cells"] = tuple(cells)

    metadata = {
        "scenario": scenario,
        "orientation": orientation,
        "fixed": fixed,
        "step": step,
        "slot_positions": tuple(slot_positions),
        "shape": shape,
        "objects": [
            {
                "area": int(obj["area"]),
                "color": int(obj["color"]),
                "slot_index": int(obj["slot_index"]),
                "cells": tuple(obj["cells"]),
            }
            for obj in objects
        ],
    }
    return grid, metadata


def _metadata_oracle(metadata: dict) -> np.ndarray:
    slot_positions = list(metadata["slot_positions"])
    slot_count = len(slot_positions)
    raw_objects = [
        {"area": obj["area"], "slot_index": obj["slot_index"]}
        for obj in metadata["objects"]
    ]
    final_slots = _simulate_slots(raw_objects, slot_count)["slots"]

    out = np.full(metadata["shape"], BLACK, dtype=int)
    for i, pos in enumerate(slot_positions):
        if metadata["orientation"] == "h":
            out[metadata["fixed"], pos] = YELLOW if i == slot_count - 1 else GREEN
        else:
            out[pos, metadata["fixed"]] = YELLOW if i == slot_count - 1 else GREEN

    for obj, new_slot in zip(metadata["objects"], final_slots):
        old_pos = slot_positions[obj["slot_index"]]
        new_pos = slot_positions[new_slot]
        delta = new_pos - old_pos
        for rr, cc in obj["cells"]:
            nr = rr + (delta if metadata["orientation"] == "v" else 0)
            nc = cc + (delta if metadata["orientation"] == "h" else 0)
            out[nr, nc] = obj["color"]
    return out


def _validate_case(grid: np.ndarray, metadata: dict) -> bool:
    try:
        layout = _extract_layout(grid)
    except AssertionError:
        return False

    if (
        layout["axis"] != metadata["orientation"]
        or layout["fixed"] != metadata["fixed"]
        or tuple(layout["slot_positions"]) != tuple(metadata["slot_positions"])
        or layout["step"] != metadata["step"]
    ):
        return False

    extracted = {int(obj["color"]): obj for obj in layout["objects"]}
    for obj in metadata["objects"]:
        parsed = extracted.get(int(obj["color"]))
        if parsed is None or int(parsed["area"]) != int(obj["area"]) or int(parsed["slot_index"]) != int(obj["slot_index"]):
            return False

    metrics = _slot_metrics(
        len(metadata["slot_positions"]),
        [obj["area"] for obj in metadata["objects"]],
        tuple(obj["slot_index"] for obj in metadata["objects"]),
    )
    if metadata["scenario"] in {"mixed", "collision"} and not metrics["order_sensitive"]:
        return False
    if metadata["scenario"] == "mixed" and not (metrics["blocked"] and metrics["capped"]):
        return False
    if metadata["scenario"] == "collision" and not (metrics["blocked"] and not metrics["capped"]):
        return False
    if metadata["scenario"] == "endpoint" and not (metrics["capped"] and not metrics["blocked"]):
        return False
    if metadata["scenario"] == "clear" and not (not metrics["blocked"] and not metrics["capped"]):
        return False
    if not (metrics["has_adjacent_visible"] and metrics["step_sensitive"] and metrics["moved"]):
        return False

    oracle = _metadata_oracle(metadata)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, oracle)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        grid, metadata = built
        if _validate_case(grid, metadata):
            return grid
    raise RuntimeError("failed to generate a valid track-sliding input")
