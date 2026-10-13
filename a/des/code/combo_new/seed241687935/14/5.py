from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("up", "right", "down", "left")
SHAPE_KINDS = ("U", "L", "C")

GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 400
SINGLE_ATTEMPTS = 400
GRID_H_RANGE = (16, 26)
GRID_W_RANGE = (16, 26)
OBJECT_COUNT_OPTIONS = (4, 5)


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _edge_center(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if side == "up":
        return r0, c0 + (width - 1) // 2
    if side == "down":
        return r1, c0 + (width - 1) // 2
    if side == "left":
        return r0 + (height - 1) // 2, c0
    return r0 + (height - 1) // 2, c1


def _open_side(component: dict) -> str:
    coords = component["coords"]
    r0, c0, r1, c1 = component["bbox"]
    counts = {
        "up": int(np.sum(coords[:, 0] == r0)),
        "down": int(np.sum(coords[:, 0] == r1)),
        "left": int(np.sum(coords[:, 1] == c0)),
        "right": int(np.sum(coords[:, 1] == c1)),
    }
    best = min(counts.values())
    candidates = [side for side, count in counts.items() if count == best]
    assert len(candidates) == 1, "expected a unique opening side"
    side = candidates[0]
    start = _edge_center(component["bbox"], side)
    coord_set = {tuple(map(int, rc)) for rc in coords}
    assert start not in coord_set, "opening center must be background"
    return side


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground object"

    height, width = g.shape
    owner = np.full((height, width), -1, dtype=int)
    objects: list[dict] = []
    for idx, component in enumerate(components):
        side = _open_side(component)
        start = _edge_center(component["bbox"], side)
        coord_set = {tuple(map(int, rc)) for rc in component["coords"]}
        for r, c in coord_set:
            owner[r, c] = idx
        objects.append(
            {
                "id": idx,
                "color": int(component["color"]),
                "coords": component["coords"],
                "coord_set": coord_set,
                "bbox": component["bbox"],
                "area": int(component["area"]),
                "open_side": side,
                "ray_start": start,
            }
        )

    rays: list[dict] = []
    for obj in objects:
        dr, dc = DIRS[obj["open_side"]]
        r, c = obj["ray_start"]
        path: list[tuple[int, int]] = []
        hit = None
        stopped_by_boundary = False

        for step in range(obj["area"]):
            if not (0 <= r < height and 0 <= c < width):
                break
            target = int(owner[r, c])
            if target != -1:
                hit = target
                break

            path.append((int(r), int(c)))
            if step < obj["area"] - 1 and (r == 0 or r == height - 1 or c == 0 or c == width - 1):
                stopped_by_boundary = True
                break
            r += dr
            c += dc

        outcome = "hit" if hit is not None else "boundary" if stopped_by_boundary else "clear"
        rays.append(
            {
                "object_id": int(obj["id"]),
                "color": int(obj["color"]),
                "path": path,
                "hit": hit,
                "outcome": outcome,
            }
        )

    hit_sources: dict[int, list[int]] = defaultdict(list)
    path_sources: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ray in rays:
        if ray["hit"] is not None:
            hit_sources[int(ray["hit"])].append(int(ray["object_id"]))
        for cell in ray["path"]:
            path_sources[cell].append(int(ray["object_id"]))

    return {
        "bg": bg,
        "objects": objects,
        "rays": rays,
        "hit_sources": dict(hit_sources),
        "path_sources": dict(path_sources),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each object's open side, shoot a same-color ray of area length, and recolor any uniquely struck target.

    Input structure:
    - The dominant color is background.
    - The foreground is split into separated monochrome 4-connected objects with distinct visible shapes.
    - Each object is a U-like, L-like, or C-like open contour whose tight bounding box has one uniquely sparsest border side; that side is the opening direction.

    Rule:
    - For each object, determine the opening side from its bounding-box border occupancy and count its pixels as N.
    - Start a one-pixel-wide ray from the center of that opening side, using the left/upper center when the side length is even.
    - Extend the ray in the opening direction for up to N cells; stop early at the boundary or just before the first original foreground pixel hit.
    - Recolor the entire hit object to the ray color, while all rays are computed simultaneously from the original grid state.

    Key insight:
    - The object's shape encodes a direction, its area becomes a ray length, and collisions cause whole-object color transfer without changing how other rays are emitted.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_grid(g)
    out = g.copy()

    for target_id, source_ids in analysis["hit_sources"].items():
        assert len(source_ids) <= 1, "expected unambiguous target recoloring"
        source = source_ids[0]
        new_color = int(analysis["objects"][source]["color"])
        coords = analysis["objects"][target_id]["coords"]
        out[coords[:, 0], coords[:, 1]] = new_color

    for cell, source_ids in analysis["path_sources"].items():
        assert len(source_ids) <= 1, "expected non-overlapping rays"
        source = source_ids[0]
        out[cell] = int(analysis["objects"][source]["color"])

    return out


def _rotate_coords(
    coords: set[tuple[int, int]],
    height: int,
    width: int,
    side: str,
) -> tuple[set[tuple[int, int]], int, int]:
    turns = DIR_ORDER.index(side)
    rotated = set(coords)
    h, w = height, width
    for _ in range(turns):
        rotated = {(c, h - 1 - r) for r, c in rotated}
        h, w = w, h
    return rotated, h, w


def _build_open_up_shape(kind: str, height: int, width: int, rng: random.Random) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()

    if kind == "U":
        for r in range(height):
            cells.add((r, 0))
            cells.add((r, width - 1))
        for c in range(width):
            cells.add((height - 1, c))
    elif kind == "L":
        stub_start = max(1, height // 2)
        for r in range(height):
            cells.add((r, 0))
        for c in range(width):
            cells.add((height - 1, c))
        for r in range(stub_start, height):
            cells.add((r, width - 1))
    else:
        mid = height // 2
        for r in range(height):
            cells.add((r, 0))
            cells.add((r, width - 1))
        for c in range(width):
            cells.add((height - 1, c))
        for c in range(width - 1):
            cells.add((mid, c))

    extra_budget = rng.randint(0, max(1, (height * width - len(cells)) // 3))
    for _ in range(extra_budget):
        candidates = []
        for r in range(1, height):
            for c in range(width):
                if (r, c) in cells:
                    continue
                if any((r + dr, c + dc) in cells for dr, dc in DIRS.values() if 0 <= r + dr < height and 0 <= c + dc < width):
                    candidates.append((r, c))
        if not candidates:
            break
        cells.add(rng.choice(candidates))
    return cells


def _make_shape_spec(rng: random.Random, color: int) -> dict:
    kind = rng.choice(SHAPE_KINDS)
    side = rng.choice(DIR_ORDER)
    height = rng.randint(3, 6)
    width = rng.randint(3, 7)
    base = _build_open_up_shape(kind, height, width, rng)
    cells, height, width = _rotate_coords(base, height, width, side)
    start_local = _edge_center((0, 0, height - 1, width - 1), side)
    return {
        "kind": kind,
        "color": int(color),
        "open_side": side,
        "cells": cells,
        "height": int(height),
        "width": int(width),
        "area": int(len(cells)),
        "start_local": start_local,
    }


def _global_cells(spec: dict, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + r, ac + c) for r, c in spec["cells"]}


def _make_object(spec: dict, anchor: tuple[int, int]) -> dict:
    coords = sorted(_global_cells(spec, anchor))
    arr = np.array(coords, dtype=int)
    start_r = anchor[0] + spec["start_local"][0]
    start_c = anchor[1] + spec["start_local"][1]
    return {
        "color": int(spec["color"]),
        "coords": arr,
        "open_side": spec["open_side"],
        "area": int(spec["area"]),
        "ray_start": (int(start_r), int(start_c)),
    }


def _edge_count_for_spec(spec: dict, side: str) -> int:
    cells = spec["cells"]
    h = spec["height"]
    w = spec["width"]
    if side == "up":
        return sum(r == 0 for r, _c in cells)
    if side == "down":
        return sum(r == h - 1 for r, _c in cells)
    if side == "left":
        return sum(c == 0 for _r, c in cells)
    return sum(c == w - 1 for _r, c in cells)


def _spec_is_valid(spec: dict) -> bool:
    counts = {side: _edge_count_for_spec(spec, side) for side in DIR_ORDER}
    best = min(counts.values())
    if [side for side, count in counts.items() if count == best] != [spec["open_side"]]:
        return False
    return spec["start_local"] not in spec["cells"]


def _frontier_pixels(spec: dict, ray_side: str) -> list[tuple[int, int]]:
    cells = spec["cells"]
    if ray_side == "right":
        by_row: dict[int, list[int]] = defaultdict(list)
        for r, c in cells:
            by_row[r].append(c)
        return [(r, min(cols)) for r, cols in by_row.items()]
    if ray_side == "left":
        by_row = defaultdict(list)
        for r, c in cells:
            by_row[r].append(c)
        return [(r, max(cols)) for r, cols in by_row.items()]
    if ray_side == "down":
        by_col: dict[int, list[int]] = defaultdict(list)
        for r, c in cells:
            by_col[c].append(r)
        return [(min(rows), c) for c, rows in by_col.items()]
    by_col = defaultdict(list)
    for r, c in cells:
        by_col[c].append(r)
    return [(max(rows), c) for c, rows in by_col.items()]


def _analyze_placed(objects: list[dict], height: int, width: int) -> dict | None:
    owner = np.full((height, width), -1, dtype=int)
    for idx, obj in enumerate(objects):
        for r, c in obj["coords"]:
            r = int(r)
            c = int(c)
            if owner[r, c] != -1:
                return None
            owner[r, c] = idx

    rays: list[dict] = []
    for idx, obj in enumerate(objects):
        dr, dc = DIRS[obj["open_side"]]
        r, c = obj["ray_start"]
        path: list[tuple[int, int]] = []
        hit = None
        stopped_by_boundary = False

        for step in range(obj["area"]):
            if not (0 <= r < height and 0 <= c < width):
                break
            target = int(owner[r, c])
            if target != -1:
                hit = target
                break
            path.append((int(r), int(c)))
            if step < obj["area"] - 1 and (r == 0 or r == height - 1 or c == 0 or c == width - 1):
                stopped_by_boundary = True
                break
            r += dr
            c += dc

        outcome = "hit" if hit is not None else "boundary" if stopped_by_boundary else "clear"
        rays.append(
            {
                "object_id": idx,
                "path": path,
                "hit": hit,
                "outcome": outcome,
            }
        )

    path_hits: dict[tuple[int, int], list[int]] = defaultdict(list)
    target_hits: dict[int, list[int]] = defaultdict(list)
    for ray in rays:
        for cell in ray["path"]:
            path_hits[cell].append(int(ray["object_id"]))
        if ray["hit"] is not None:
            target_hits[int(ray["hit"])].append(int(ray["object_id"]))

    if any(len(sources) > 1 for sources in path_hits.values()):
        return None
    if any(len(sources) > 1 for sources in target_hits.values()):
        return None

    return {
        "rays": rays,
        "reserved_paths": set(path_hits.keys()),
        "target_hits": dict(target_hits),
    }


def _occupied_cells(objects: list[dict]) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    for obj in objects:
        for r, c in obj["coords"]:
            occupied.add((int(r), int(c)))
    return occupied


def _can_place_cells(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width):
            return False
        if (r, c) in occupied or (r, c) in reserved_paths:
            return False
        for nr, nc in _neighbors4(r, c, height, width):
            if (nr, nc) in occupied:
                return False
    return True


def _random_anchor(spec: dict, height: int, width: int, rng: random.Random, desired: str | None) -> tuple[int, int]:
    max_r = height - spec["height"]
    max_c = width - spec["width"]
    if desired == "boundary":
        if spec["open_side"] == "up":
            r = rng.randint(0, min(max_r, max(0, spec["area"] - 2)))
            c = rng.randint(0, max_c)
            return r, c
        if spec["open_side"] == "down":
            min_r = max(0, height - spec["height"] - spec["area"] + 2)
            r = rng.randint(min_r, max_r)
            c = rng.randint(0, max_c)
            return r, c
        if spec["open_side"] == "left":
            r = rng.randint(0, max_r)
            c = rng.randint(0, min(max_c, max(0, spec["area"] - 2)))
            return r, c
        r = rng.randint(0, max_r)
        min_c = max(0, width - spec["width"] - spec["area"] + 2)
        c = rng.randint(min_c, max_c)
        return r, c

    if desired == "clear":
        if spec["open_side"] == "up":
            min_r = min(max_r, max(0, spec["area"]))
            r = rng.randint(min_r, max_r)
            c = rng.randint(0, max_c)
            return r, c
        if spec["open_side"] == "down":
            max_r2 = max(0, height - spec["height"] - spec["area"] - 1)
            r = rng.randint(0, min(max_r, max_r2))
            c = rng.randint(0, max_c)
            return r, c
        if spec["open_side"] == "left":
            r = rng.randint(0, max_r)
            min_c = min(max_c, max(0, spec["area"]))
            c = rng.randint(min_c, max_c)
            return r, c
        r = rng.randint(0, max_r)
        max_c2 = max(0, width - spec["width"] - spec["area"] - 1)
        c = rng.randint(0, min(max_c, max_c2))
        return r, c

    return rng.randint(0, max_r), rng.randint(0, max_c)


def _place_hit_pair(
    shooter_spec: dict,
    target_spec: dict,
    objects: list[dict],
    analysis: dict | None,
    height: int,
    width: int,
    rng: random.Random,
) -> tuple[list[dict], dict] | None:
    reserved_paths = set() if analysis is None else set(analysis["reserved_paths"])
    occupied = _occupied_cells(objects)
    dr, dc = DIRS[shooter_spec["open_side"]]
    if shooter_spec["area"] < 2:
        return None

    for _ in range(PAIR_ATTEMPTS):
        target_anchor = _random_anchor(target_spec, height, width, rng, None)
        target_cells = _global_cells(target_spec, target_anchor)
        if not _can_place_cells(target_cells, occupied, reserved_paths, height, width):
            continue

        frontier = _frontier_pixels(target_spec, shooter_spec["open_side"])
        rng.shuffle(frontier)
        for fr, fc in frontier:
            hit_r = target_anchor[0] + fr
            hit_c = target_anchor[1] + fc
            steps = list(range(1, shooter_spec["area"]))
            rng.shuffle(steps)
            for step in steps:
                start_r = hit_r - dr * step
                start_c = hit_c - dc * step
                shooter_anchor = (
                    start_r - shooter_spec["start_local"][0],
                    start_c - shooter_spec["start_local"][1],
                )
                shooter_cells = _global_cells(shooter_spec, shooter_anchor)
                if not _can_place_cells(shooter_cells, occupied | target_cells, reserved_paths, height, width):
                    continue

                trial_objects = objects + [
                    _make_object(shooter_spec, shooter_anchor),
                    _make_object(target_spec, target_anchor),
                ]
                trial = _analyze_placed(trial_objects, height, width)
                if trial is None:
                    continue

                shooter_idx = len(objects)
                target_idx = len(objects) + 1
                if trial["rays"][shooter_idx]["outcome"] != "hit":
                    continue
                if trial["rays"][shooter_idx]["hit"] != target_idx:
                    continue
                if any(cell in reserved_paths for cell in trial["rays"][shooter_idx]["path"]):
                    continue
                if any(cell in reserved_paths for cell in trial["rays"][target_idx]["path"]):
                    continue
                return trial_objects, trial

    return None


def _place_single(
    spec: dict,
    desired: str,
    objects: list[dict],
    analysis: dict | None,
    height: int,
    width: int,
    rng: random.Random,
) -> tuple[list[dict], dict] | None:
    reserved_paths = set() if analysis is None else set(analysis["reserved_paths"])
    occupied = _occupied_cells(objects)

    for _ in range(SINGLE_ATTEMPTS):
        anchor = _random_anchor(spec, height, width, rng, desired)
        cells = _global_cells(spec, anchor)
        if not _can_place_cells(cells, occupied, reserved_paths, height, width):
            continue

        trial_objects = objects + [_make_object(spec, anchor)]
        trial = _analyze_placed(trial_objects, height, width)
        if trial is None:
            continue

        idx = len(objects)
        ray = trial["rays"][idx]
        if ray["outcome"] != desired:
            continue
        if any(cell in reserved_paths for cell in ray["path"]):
            continue
        return trial_objects, trial

    return None


def _build_grid(height: int, width: int, bg: int, objects: list[dict]) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for obj in objects:
        rr = obj["coords"][:, 0]
        cc = obj["coords"][:, 1]
        grid[rr, cc] = int(obj["color"])
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(10)
    object_count = rng.choice(OBJECT_COUNT_OPTIONS)
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    colors = colors[:object_count]

    specs = [_make_shape_spec(rng, color) for color in colors]
    if not all(_spec_is_valid(spec) for spec in specs):
        return None

    shooter_spec, target_spec = specs[0], specs[1]
    other_specs = specs[2:]
    desired_outcomes = ["boundary", "clear"]
    while len(desired_outcomes) < len(other_specs):
        desired_outcomes.append(rng.choice(("boundary", "clear")))
    rng.shuffle(desired_outcomes)

    objects: list[dict] = []
    analysis = None

    pair_result = _place_hit_pair(shooter_spec, target_spec, objects, analysis, height, width, rng)
    if pair_result is None:
        return None
    objects, analysis = pair_result

    for spec, desired in zip(other_specs, desired_outcomes):
        placed = _place_single(spec, desired, objects, analysis, height, width, rng)
        if placed is None:
            return None
        objects, analysis = placed

    if analysis is None:
        return None

    outcomes = Counter(ray["outcome"] for ray in analysis["rays"])
    if outcomes["hit"] < 1 or outcomes["boundary"] < 1 or outcomes["clear"] < 1:
        return None

    grid = _build_grid(height, width, bg, objects)
    parsed = _analyze_grid(grid)
    if len(parsed["objects"]) != len(objects):
        return None
    parsed_outcomes = Counter(ray["outcome"] for ray in parsed["rays"])
    if parsed_outcomes["hit"] < 1 or parsed_outcomes["boundary"] < 1 or parsed_outcomes["clear"] < 1:
        return None
    if any(len(sources) > 1 for sources in parsed["hit_sources"].values()):
        return None
    if any(len(sources) > 1 for sources in parsed["path_sources"].values()):
        return None

    fg = int(np.count_nonzero(grid != bg))
    if fg * 2 >= grid.size:
        return None

    main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid open-ray ARC input")
