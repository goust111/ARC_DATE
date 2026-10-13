from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
SPRITE_COLORS = (BLUE, RED, GREEN, YELLOW)
CARDINALS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_ROWS = (18, 23)
GRID_COLS = (18, 23)
MAX_GENERATION_ATTEMPTS = 500
MIN_RAY_LEN = 2
MIN_STAMP_PIXELS = 2
PRIMARY_GAP_ROWS = 4
PRIMARY_GAP_COLS = 4


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for nxt in _neighbors4(cur):
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _find_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    comps: list[set[tuple[int, int]]] = []
    while coords:
        start = coords.pop()
        comp = {start}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nxt in _neighbors4(cur):
                if nxt in coords:
                    coords.remove(nxt)
                    comp.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _trail_signature(
    component: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]:
    candidates = []
    cells = list(component)
    for a, b in itertools.combinations(cells, 2):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            continue
        trail = {a, b}
        sprite = component - trail
        if not (3 <= len(sprite) <= 5):
            continue
        if not _connected(sprite):
            continue
        attachments: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for t in trail:
            for s in _neighbors4(t):
                if s in sprite:
                    attachments.append((t, s))
        if len(attachments) != 1:
            continue
        near, attach = attachments[0]
        far = next(iter(trail - {near}))
        d1 = (far[0] - near[0], far[1] - near[1])
        d2 = (near[0] - attach[0], near[1] - attach[1])
        if d1 != d2:
            continue
        candidates.append((sprite, (near, far), d1))
    assert len(candidates) == 1, f"expected one valid 2-pixel trail split, got {len(candidates)}"
    return candidates[0]


def _parse_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) in SPRITE_COLORS):
        comps = _find_components(g == color)
        assert len(comps) == 1, f"expected one component for color {color}"
        component = comps[0]
        sprite, trail_ordered, trail_dir = _trail_signature(component)
        r0, c0, r1, c1 = _bbox(sprite)
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        shape_offsets = tuple(sorted((r - r0, c - c0) for r, c in sprite))
        objects.append(
            {
                "color": color,
                "component": component,
                "sprite": sprite,
                "trail": trail_ordered,
                "trail_dir": trail_dir,
                "trail_len": 2,
                "bbox": (r0, c0, r1, c1),
                "shape_offsets": shape_offsets,
                "center": center,
                "row_key": r0,
                "col_key": c0,
                "ray_dir": (-trail_dir[0], -trail_dir[1]),
            }
        )
    objects.sort(key=lambda obj: (obj["row_key"], obj["col_key"], obj["color"]))
    for idx, obj in enumerate(objects):
        obj["index"] = idx
    return objects


def _simulate_rays(grid: np.ndarray, objects: list[dict]) -> tuple[np.ndarray, list[dict], list[list[tuple[int, int]]]]:
    out = np.asarray(grid, dtype=int).copy()
    owner = np.full(out.shape, -1, dtype=int)
    events: list[dict] = []
    paths: list[list[tuple[int, int]]] = []
    h, w = out.shape
    for obj in objects:
        r, c = obj["center"]
        dr, dc = obj["ray_dir"]
        path: list[tuple[int, int]] = []
        cur_r, cur_c = r, c
        while True:
            nr, nc = cur_r + dr, cur_c + dc
            if not _in_bounds(out.shape, nr, nc) or nr in (0, h - 1) or nc in (0, w - 1):
                events.append({"kind": "boundary", "object": obj["index"], "path": path[:]})
                break
            val = int(out[nr, nc])
            if val == BLACK:
                out[nr, nc] = GREY
                owner[nr, nc] = obj["index"]
                path.append((nr, nc))
                cur_r, cur_c = nr, nc
                continue
            if val == GREY:
                other = int(owner[nr, nc])
                event = {
                    "kind": "ray_hit",
                    "object": obj["index"],
                    "other": other,
                    "blocked_cell": (nr, nc),
                    "path": path[:],
                }
                if path:
                    event["pair_point"] = path[-1]
                events.append(event)
                break
            events.append(
                {
                    "kind": "obstacle",
                    "object": obj["index"],
                    "blocked_cell": (nr, nc),
                    "color": val,
                    "path": path[:],
                }
            )
            break
        paths.append(path)
    return out, events, paths


def _provider_and_donor(objects: list[dict], a_idx: int, b_idx: int) -> tuple[dict, dict]:
    a = objects[a_idx]
    b = objects[b_idx]
    if a["trail_len"] != b["trail_len"]:
        return (a, b) if a["trail_len"] > b["trail_len"] else (b, a)
    return (a, b) if (a["row_key"], a["col_key"], a["color"]) < (b["row_key"], b["col_key"], b["color"]) else (b, a)


def _apply_stamps(ray_grid: np.ndarray, objects: list[dict], events: list[dict]) -> tuple[np.ndarray, list[dict]]:
    out = np.asarray(ray_grid, dtype=int).copy()
    stamp_infos: list[dict] = []
    for event in events:
        if event["kind"] != "ray_hit" or "pair_point" not in event:
            continue
        provider, donor = _provider_and_donor(objects, event["object"], event["other"])
        anchor_r, anchor_c = event["pair_point"]
        painted: list[tuple[int, int]] = []
        for dr, dc in provider["shape_offsets"]:
            rr = anchor_r + dr
            cc = anchor_c + dc
            if _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == BLACK:
                out[rr, cc] = int(donor["color"])
                painted.append((rr, cc))
        stamp_infos.append(
            {
                "event": event,
                "provider": provider["index"],
                "donor": donor["index"],
                "anchor": (anchor_r, anchor_c),
                "painted": painted,
                "bbox_shape": provider["bbox"],
            }
        )
    return out, stamp_infos


def main(grid: np.ndarray) -> np.ndarray:
    """Cast reverse rays from trailed sprites, stop them on collisions, and stamp a cross-derived sprite copy at each ray-intersection event.

    Input structure:
    - The grid has a black background and 2-4 differently colored sprites in blue, red, green, or yellow.
    - Each sprite is a 4-connected 3-5 pixel shape with one same-colored 2-pixel cardinal trail attached to one edge pixel.
    - No grey pixels appear in the input; grey is reserved for generated rays.

    Rule:
    - For each color component, split off the unique 2-pixel trail, recover the sprite bbox, and read the trail direction from the sprite edge toward the trail tip.
    - Start a grey ray from the sprite bbox center (floor) in the opposite direction, drawing sprites in increasing sprite-row order and stopping before a non-black cell, an already drawn grey ray, or the grid boundary.
    - Whenever a later ray stops because its next cell is an earlier grey ray, treat that later ray's last grey pixel as the intersection event anchor; stamp the smaller-row sprite's shape there (trail lengths tie here), but recolor the stamp with the other sprite's color and only paint onto black cells.

    Key insight:
    - The 2-pixel trail is not motion itself but a directional clue: reversing it reveals a ray whose collision point is the only place where shape information from one sprite and color information from another are fused.
    """
    objects = _parse_objects(np.asarray(grid, dtype=int))
    ray_grid, events, _paths = _simulate_rays(np.asarray(grid, dtype=int), objects)
    stamped, _stamp_infos = _apply_stamps(ray_grid, objects, events)
    return stamped


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_catalog() -> dict[str, list[dict]]:
    entries = {name: [] for name in CARDINALS}
    positions = [(r, c) for r in range(3) for c in range(3)]
    seen = set()
    for size in range(3, 6):
        for combo in itertools.combinations(positions, size):
            cells = set(combo)
            if not _connected(cells):
                continue
            norm = _normalize_shape(cells)
            if norm in seen:
                continue
            r0, c0, r1, c1 = _bbox(cells)
            trimmed = {(r - r0, c - c0) for r, c in cells}
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            center = ((h - 1) // 2, (w - 1) // 2)
            if center not in trimmed:
                continue
            info = {
                "shape_offsets": tuple(sorted(trimmed)),
                "h": h,
                "w": w,
                "center_rel": center,
                "trail_attach": {},
            }
            any_dir = False
            for name, (dr, dc) in CARDINALS.items():
                nr, nc = center[0] + dr, center[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in trimmed:
                    continue
                trail_dr, trail_dc = -dr, -dc
                options = []
                for ar, ac in trimmed:
                    t1 = (ar + trail_dr, ac + trail_dc)
                    t2 = (ar + 2 * trail_dr, ac + 2 * trail_dc)
                    if 0 <= t1[0] < h and 0 <= t1[1] < w:
                        continue
                    if t1 in trimmed or t2 in trimmed:
                        continue
                    trial = set(trimmed)
                    trial.add(t1)
                    trial.add(t2)
                    shifted = {(r + 10, c + 10) for r, c in trial}
                    try:
                        sprite, _trail, d = _trail_signature(shifted)
                    except AssertionError:
                        continue
                    if _normalize_shape({(r - 10, c - 10) for r, c in sprite}) != info["shape_offsets"]:
                        continue
                    if d != (trail_dr, trail_dc):
                        continue
                    options.append((ar, ac))
                if options:
                    info["trail_attach"][name] = tuple(sorted(options))
                    entries[name].append(info)
                    any_dir = True
            if any_dir:
                seen.add(norm)
    for name in CARDINALS:
        unique = {}
        for info in entries[name]:
            unique[info["shape_offsets"]] = info
        entries[name] = list(unique.values())
    return entries


SHAPE_CATALOG = _shape_catalog()


def _make_object(
    shape: dict,
    ray_name: str,
    color: int,
    center: tuple[int, int],
    attach: tuple[int, int],
) -> dict:
    trail_dr, trail_dc = (-CARDINALS[ray_name][0], -CARDINALS[ray_name][1])
    cr, cc = shape["center_rel"]
    top = center[0] - cr
    left = center[1] - cc
    sprite = {(top + r, left + c) for r, c in shape["shape_offsets"]}
    attach_abs = (top + attach[0], left + attach[1])
    trail = (
        (attach_abs[0] + trail_dr, attach_abs[1] + trail_dc),
        (attach_abs[0] + 2 * trail_dr, attach_abs[1] + 2 * trail_dc),
    )
    return {
        "color": color,
        "shape": shape,
        "ray_name": ray_name,
        "ray_dir": CARDINALS[ray_name],
        "trail_dir": (trail_dr, trail_dc),
        "center": center,
        "sprite": sprite,
        "trail": trail,
        "cells": sprite | set(trail),
        "bbox_top": top,
        "bbox_left": left,
        "bbox": (top, left, top + shape["h"] - 1, left + shape["w"] - 1),
        "shape_offsets": shape["shape_offsets"],
    }


def _object_fits(obj: dict, shape: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    h, w = shape
    if any(not _in_bounds(shape, r, c) for r, c in obj["cells"]):
        return False
    if any(r in (0, h - 1) or c in (0, w - 1) for r, c in obj["cells"]):
        return False
    if obj["cells"] & occupied:
        return False
    return True


def _chebyshev_safe(a: set[tuple[int, int]], b: set[tuple[int, int]], gap: int = 1) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= gap:
                return False
    return True


def _planned_boundary_path(obj: dict, grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid_shape
    r, c = obj["center"]
    dr, dc = obj["ray_dir"]
    path = []
    cur_r, cur_c = r, c
    while True:
        nr, nc = cur_r + dr, cur_c + dc
        if not _in_bounds(grid_shape, nr, nc) or nr in (0, h - 1) or nc in (0, w - 1):
            return path
        path.append((nr, nc))
        cur_r, cur_c = nr, nc


def _render_input(grid_shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for obj in objects:
        for r, c in obj["sprite"]:
            grid[r, c] = int(obj["color"])
        for r, c in obj["trail"]:
            grid[r, c] = int(obj["color"])
    return grid


def _box_cells(anchor: tuple[int, int], dims: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = anchor
    h, w = dims
    return {(r0 + r, c0 + c) for r in range(h) for c in range(w)}


def _sample_shape(ray_name: str) -> tuple[dict, tuple[int, int]]:
    shape = random.choice(SHAPE_CATALOG[ray_name])
    attach = random.choice(shape["trail_attach"][ray_name])
    return shape, attach


def _valid_center_range(obj_template: dict, grid_shape: tuple[int, int]) -> tuple[range, range]:
    rel_cells = []
    cr, cc = obj_template["shape"]["center_rel"]
    for r, c in obj_template["shape_offsets"]:
        rel_cells.append((r - cr, c - cc))
    ar, ac = obj_template["attach"]
    rel_attach = (ar - cr, ac - cc)
    tr, tc = obj_template["trail_dir"]
    rel_cells.append((rel_attach[0] + tr, rel_attach[1] + tc))
    rel_cells.append((rel_attach[0] + 2 * tr, rel_attach[1] + 2 * tc))
    min_r = min(r for r, _c in rel_cells)
    max_r = max(r for r, _c in rel_cells)
    min_c = min(c for _r, c in rel_cells)
    max_c = max(c for _r, c in rel_cells)
    rows = range(-min_r + 1, grid_shape[0] - max_r - 1)
    cols = range(-min_c + 1, grid_shape[1] - max_c - 1)
    return rows, cols


def _range_nonempty(rng: range) -> bool:
    return len(rng) > 0


def _place_primary_objects(grid_shape: tuple[int, int], n_objects: int) -> list[dict] | None:
    for _ in range(200):
        b_shape, b_attach = _sample_shape("down")
        a_dir = random.choice(("left", "right"))
        a_shape, a_attach = _sample_shape(a_dir)
        b_template = {"shape": b_shape, "shape_offsets": b_shape["shape_offsets"], "attach": b_attach, "trail_dir": (-CARDINALS["down"][0], -CARDINALS["down"][1])}
        a_template = {"shape": a_shape, "shape_offsets": a_shape["shape_offsets"], "attach": a_attach, "trail_dir": (-CARDINALS[a_dir][0], -CARDINALS[a_dir][1])}
        b_rows, b_cols = _valid_center_range(b_template, grid_shape)
        a_rows, a_cols = _valid_center_range(a_template, grid_shape)
        if not (_range_nonempty(b_rows) and _range_nonempty(b_cols) and _range_nonempty(a_rows) and _range_nonempty(a_cols)):
            continue
        a_row = random.choice(list(a_rows))
        b_row_choices = [r for r in b_rows if r <= a_row - PRIMARY_GAP_ROWS]
        if not b_row_choices:
            continue
        b_row = random.choice(b_row_choices)
        provider_dims = (b_shape["h"], b_shape["w"])
        if a_dir == "right":
            possible_cols = [c for c in b_cols if c - 1 >= 0 and c - 1 + provider_dims[1] <= grid_shape[1]]
            if not possible_cols:
                continue
            b_col = random.choice(possible_cols)
            a_col_choices = [c for c in a_cols if c <= b_col - PRIMARY_GAP_COLS]
            if not a_col_choices:
                continue
            a_col = random.choice(a_col_choices)
            if b_col - a_col - 1 < MIN_RAY_LEN:
                continue
            pair_anchor = (a_row, b_col - 1)
        else:
            possible_cols = [c for c in b_cols if c + 1 + provider_dims[1] <= grid_shape[1]]
            if not possible_cols:
                continue
            b_col = random.choice(possible_cols)
            a_col_choices = [c for c in a_cols if c >= b_col + PRIMARY_GAP_COLS]
            if not a_col_choices:
                continue
            a_col = random.choice(a_col_choices)
            if a_col - b_col - 1 < MIN_RAY_LEN:
                continue
            pair_anchor = (a_row, b_col + 1)
        if pair_anchor[0] + provider_dims[0] > grid_shape[0]:
            continue
        reserved_boxes = [_box_cells(pair_anchor, provider_dims)]
        occupied: set[tuple[int, int]] = set()
        colors = random.sample(list(SPRITE_COLORS), n_objects)
        b_obj = _make_object(b_shape, "down", colors[0], (b_row, b_col), b_attach)
        a_obj = _make_object(a_shape, a_dir, colors[1], (a_row, a_col), a_attach)
        if not _object_fits(b_obj, grid_shape, occupied):
            continue
        occupied |= b_obj["cells"]
        if any(cell in reserved_boxes[0] for cell in b_obj["cells"]):
            continue
        if not _object_fits(a_obj, grid_shape, occupied):
            continue
        if not _chebyshev_safe(a_obj["cells"], b_obj["cells"]):
            continue
        if any(cell in reserved_boxes[0] for cell in a_obj["cells"]):
            continue
        occupied |= a_obj["cells"]
        objects = [b_obj, a_obj]
        reserved_rays = set(_planned_boundary_path(b_obj, grid_shape))
        if a_dir == "right":
            a_path = [(a_row, c) for c in range(a_col + 1, b_col)]
        else:
            a_path = [(a_row, c) for c in range(b_col + 1, a_col)]
        if len(a_path) < MIN_RAY_LEN:
            continue
        reserved_rays |= set(a_path)
        if set(a_path) & occupied:
            continue
        second_pair = n_objects >= 3 and random.random() < 0.55
        if second_pair:
            c_dir = "left" if a_dir == "right" else "right"
            c_shape, c_attach = _sample_shape(c_dir)
            c_template = {"shape": c_shape, "shape_offsets": c_shape["shape_offsets"], "attach": c_attach, "trail_dir": (-CARDINALS[c_dir][0], -CARDINALS[c_dir][1])}
            c_rows, c_cols = _valid_center_range(c_template, grid_shape)
            if not (_range_nonempty(c_rows) and _range_nonempty(c_cols)):
                c_row_choices = []
            else:
                c_row_choices = [r for r in c_rows if r >= a_row + PRIMARY_GAP_ROWS and r + provider_dims[0] <= grid_shape[0]]
            if c_row_choices:
                c_row = random.choice(c_row_choices)
                if c_dir == "left":
                    c_col_choices = [c for c in c_cols if c >= b_col + PRIMARY_GAP_COLS]
                    if c_col_choices:
                        c_col = random.choice(c_col_choices)
                        if c_col - b_col - 1 >= MIN_RAY_LEN:
                            anchor2 = (c_row, b_col + 1)
                            if anchor2[0] + provider_dims[0] <= grid_shape[0] and anchor2[1] + provider_dims[1] <= grid_shape[1]:
                                c_obj = _make_object(c_shape, c_dir, colors[2], (c_row, c_col), c_attach)
                                c_path = [(c_row, c) for c in range(b_col + 1, c_col)]
                                if (
                                    _object_fits(c_obj, grid_shape, occupied)
                                    and _chebyshev_safe(c_obj["cells"], occupied)
                                    and not (set(c_path) & occupied)
                                    and not (set(c_path) & reserved_rays)
                                    and not any(cell in reserved_boxes[0] for cell in c_obj["cells"])
                                    and not any(cell in _box_cells(anchor2, provider_dims) for cell in c_obj["cells"])
                                ):
                                    objects.append(c_obj)
                                    occupied |= c_obj["cells"]
                                    reserved_boxes.append(_box_cells(anchor2, provider_dims))
                                    reserved_rays |= set(c_path)
                else:
                    c_col_choices = [c for c in c_cols if c <= b_col - PRIMARY_GAP_COLS]
                    if c_col_choices:
                        c_col = random.choice(c_col_choices)
                        if b_col - c_col - 1 >= MIN_RAY_LEN:
                            anchor2 = (c_row, b_col - 1)
                            if anchor2[0] + provider_dims[0] <= grid_shape[0] and anchor2[1] + provider_dims[1] <= grid_shape[1]:
                                c_obj = _make_object(c_shape, c_dir, colors[2], (c_row, c_col), c_attach)
                                c_path = [(c_row, c) for c in range(c_col + 1, b_col)]
                                if (
                                    _object_fits(c_obj, grid_shape, occupied)
                                    and _chebyshev_safe(c_obj["cells"], occupied)
                                    and not (set(c_path) & occupied)
                                    and not (set(c_path) & reserved_rays)
                                    and not any(cell in reserved_boxes[0] for cell in c_obj["cells"])
                                    and not any(cell in _box_cells(anchor2, provider_dims) for cell in c_obj["cells"])
                                ):
                                    objects.append(c_obj)
                                    occupied |= c_obj["cells"]
                                    reserved_boxes.append(_box_cells(anchor2, provider_dims))
                                    reserved_rays |= set(c_path)
        while len(objects) < n_objects:
            dirs = list(CARDINALS)
            random.shuffle(dirs)
            placed = False
            for dname in dirs:
                x_shape, x_attach = _sample_shape(dname)
                x_template = {"shape": x_shape, "shape_offsets": x_shape["shape_offsets"], "attach": x_attach, "trail_dir": (-CARDINALS[dname][0], -CARDINALS[dname][1])}
                x_rows, x_cols = _valid_center_range(x_template, grid_shape)
                if not (_range_nonempty(x_rows) and _range_nonempty(x_cols)):
                    continue
                row = random.choice(list(x_rows))
                col = random.choice(list(x_cols))
                x_obj = _make_object(x_shape, dname, colors[len(objects)], (row, col), x_attach)
                x_path = _planned_boundary_path(x_obj, grid_shape)
                if len(x_path) < MIN_RAY_LEN:
                    continue
                if not _object_fits(x_obj, grid_shape, occupied):
                    continue
                if any(not _chebyshev_safe(x_obj["cells"], prev["cells"]) for prev in objects):
                    continue
                if set(x_path) & occupied:
                    continue
                if set(x_path) & reserved_rays:
                    continue
                if any(set(x_path) & box for box in reserved_boxes):
                    continue
                if any(x_obj["cells"] & box for box in reserved_boxes):
                    continue
                objects.append(x_obj)
                occupied |= x_obj["cells"]
                reserved_rays |= set(x_path)
                placed = True
                break
            if not placed:
                break
        if len(objects) != n_objects:
            continue
        return objects
    return None


def _validate_instance(inp: np.ndarray) -> bool:
    try:
        objects = _parse_objects(inp)
        assert 2 <= len(objects) <= 4
        assert len({obj["color"] for obj in objects}) == len(objects)
        ray_grid, events, paths = _simulate_rays(inp, objects)
        output, stamp_infos = _apply_stamps(ray_grid, objects, events)
    except AssertionError:
        return False

    pair_events = [event for event in events if event["kind"] == "ray_hit" and "pair_point" in event]
    if not pair_events:
        return False
    if any(event["kind"] == "obstacle" for event in events):
        return False
    if any(event["kind"] == "ray_hit" and "pair_point" not in event for event in events):
        return False
    if sum(len(path) for path in paths) < 4:
        return False
    for info in stamp_infos:
        if len(info["painted"]) < MIN_STAMP_PIXELS:
            return False
        provider = objects[info["provider"]]
        anchor = info["anchor"]
        shape_dims = (provider["bbox"][2] - provider["bbox"][0] + 1, provider["bbox"][3] - provider["bbox"][1] + 1)
        box = _box_cells(anchor, shape_dims)
        if not all(_in_bounds(inp.shape, r, c) for r, c in box):
            return False
        if any(int(inp[r, c]) != BLACK for r, c in box):
            return False
    if np.array_equal(inp, output):
        return False
    changed = int(np.count_nonzero(output != inp))
    if changed < 6:
        return False
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, output):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid_shape = (
            random.randint(GRID_ROWS[0], GRID_ROWS[1]),
            random.randint(GRID_COLS[0], GRID_COLS[1]),
        )
        n_objects = random.randint(2, 4)
        objects = _place_primary_objects(grid_shape, n_objects)
        if objects is None:
            continue
        inp = _render_input(grid_shape, objects)
        if _validate_instance(inp):
            return inp
    raise RuntimeError("failed to generate a valid ray-intersection stamp input")
