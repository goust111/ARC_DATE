from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
MAROON = 9
PALETTE = (BLUE, GREEN, ORANGE, MAROON)
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SLOT_NAMES = ("tl", "tr", "bl", "br")
GEN_ATTEMPTS = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield int(r) + dr, int(c) + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sorted_coords(coords) -> np.ndarray:
    arr = np.array(sorted((int(r), int(c)) for r, c in coords), dtype=int)
    assert arr.ndim == 2 and arr.shape[1] == 2
    return arr


def find_components(
    grid: np.ndarray,
    *,
    ignored: set[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    ignored = {BG} if ignored is None else {int(v) for v in ignored}
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) in ignored:
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds((h, w), nr, nc) or seen[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            coords = _sorted_coords(cells)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _component_distance_to_marker(coords: np.ndarray, marker: tuple[int, int]) -> int:
    mr, mc = map(int, marker)
    return min(max(abs(int(r) - mr), abs(int(c) - mc)) for r, c in coords)


def _centroid(coords: np.ndarray) -> tuple[float, float]:
    center = coords.mean(axis=0)
    return float(center[0]), float(center[1])


def _ray_direction(marker: tuple[int, int], coords: np.ndarray) -> tuple[int, int]:
    mr, mc = map(int, marker)
    cr, cc = _centroid(coords)
    dr = cr - mr
    dc = cc - mc
    if abs(dc) >= abs(dr):
        return 0, 1 if dc > 0 else -1
    return 1 if dr > 0 else -1, 0


def _ray_start(coords: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, direction)
    cr, cc = _centroid(coords)
    cells = [(int(r), int(c)) for r, c in coords]
    if (dr, dc) == (0, 1):
        edge = max(c for _r, c in cells)
        candidates = [(r, c) for r, c in cells if c == edge]
        return min(candidates, key=lambda rc: (abs(rc[0] - cr), rc[0], rc[1]))
    if (dr, dc) == (0, -1):
        edge = min(c for _r, c in cells)
        candidates = [(r, c) for r, c in cells if c == edge]
        return min(candidates, key=lambda rc: (abs(rc[0] - cr), rc[0], rc[1]))
    if (dr, dc) == (1, 0):
        edge = max(r for r, _c in cells)
        candidates = [(r, c) for r, c in cells if r == edge]
        return min(candidates, key=lambda rc: (abs(rc[1] - cc), rc[1], rc[0]))
    edge = min(r for r, _c in cells)
    candidates = [(r, c) for r, c in cells if r == edge]
    return min(candidates, key=lambda rc: (abs(rc[1] - cc), rc[1], rc[0]))


def _reflect_coords(coords: np.ndarray, marker: tuple[int, int]) -> np.ndarray:
    mr, mc = map(int, marker)
    reflected = [(2 * mr - int(r), 2 * mc - int(c)) for r, c in coords]
    return _sorted_coords(reflected)


def _mirror_payloads(out: np.ndarray, scenes: list[dict]) -> None:
    h, w = out.shape
    for scene in scenes:
        payload = scene["payload"]
        mirrored = _reflect_coords(payload["coords"], scene["marker"])
        for r, c in mirrored:
            if 0 <= r < h and 0 <= c < w and int(out[r, c]) == BG:
                out[r, c] = int(payload["color"])


def _trace_ray(
    out: np.ndarray,
    source: dict,
    marker: tuple[int, int],
) -> tuple[list[tuple[int, int]], str]:
    h, w = out.shape
    direction = _ray_direction(marker, source["coords"])
    start_r, start_c = _ray_start(source["coords"], direction)
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    step = 1
    r = start_r + dr
    c = start_c + dc
    while 0 <= r < h and 0 <= c < w:
        if int(out[r, c]) != BG:
            return cells, "blocked"
        cells.append((r, c))
        out[r, c] = int(source["color"]) if step % 2 == 0 else GREY
        step += 1
        r += dr
        c += dc
    return cells, "boundary"


def _analyze_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    markers = [tuple(map(int, pt)) for pt in np.argwhere(g == RED)]
    markers.sort()
    assert 2 <= len(markers) <= 3, "expected 2 or 3 red markers"

    components = find_components(g, ignored={BG, RED})
    assert components, "expected colored components around markers"

    memberships: list[list[int]] = []
    for comp in components:
        owners = [
            idx
            for idx, marker in enumerate(markers)
            if _component_distance_to_marker(comp["coords"], marker) <= 4
        ]
        assert len(owners) == 1, "each component must belong to exactly one marker neighborhood"
        memberships.append(owners)

    scenes: list[dict] = []
    for marker_idx, marker in enumerate(markers):
        scene_components = [
            components[i]
            for i, owners in enumerate(memberships)
            if owners[0] == marker_idx
        ]
        assert 2 <= len(scene_components) <= 3, "each marker needs 2 or 3 nearby components"
        colors = {int(comp["color"]) for comp in scene_components}
        assert len(colors) == len(scene_components), "scene colors must be distinct"
        areas = sorted((int(comp["area"]) for comp in scene_components), reverse=True)
        assert len(areas) == len(set(areas)), "scene component areas must be distinct"
        payload = max(scene_components, key=lambda comp: int(comp["area"]))
        assert sum(int(comp["area"]) == int(payload["area"]) for comp in scene_components) == 1
        sources = [comp for comp in scene_components if comp is not payload]
        sources.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["area"]))
        scenes.append({"marker": marker, "payload": payload, "sources": sources})
    return scenes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Point-reflect the largest nearby component around each red marker, then emit alternating orthogonal rays from the smaller components.

    Input structure:
    - Black background with 2 or 3 isolated red single-pixel markers.
    - Around each marker, within 8-neighborhood distance at most 4, there are 2 or 3 separate 4-connected monochrome components.
    - The nearby component colors are chosen from the non-red foreground palette, are distinct within each marker neighborhood, and have distinct areas.

    Rule:
    - For each red marker, collect the nearby non-black non-red 4-connected components.
    - The unique largest nearby component is the payload: point-reflect all of its pixels through the red marker and paint the reflected pixels in the payload color only onto black cells.
    - Every smaller nearby component becomes a ray source: use the vector from the marker to the component centroid, choose the nearest orthogonal direction, and break exact diagonal ties in favor of the horizontal direction.
    - Starting from the source pixel farthest in that direction, extend one cell at a time beyond the source; odd offsets become grey and even offsets become the source color, stopping before the first non-black cell or the grid boundary.

    Key insight:
    - Component area does not choose which object moves versus stays; it chooses which nearby object creates a mirrored blocker and which ones emit rays, so the mirror must be drawn first because it changes how far the rays can travel.
    """
    g = np.asarray(grid, dtype=int)
    scenes = _analyze_grid(g)
    out = g.copy()
    _mirror_payloads(out, scenes)
    for scene in scenes:
        for source in scene["sources"]:
            _trace_ray(out, source, scene["marker"])
    return out


def _transform_point(point: tuple[int, int], name: str) -> tuple[int, int]:
    r, c = map(int, point)
    if name == "rot0":
        return r, c
    if name == "rot90":
        return -c, r
    if name == "rot180":
        return -r, -c
    if name == "rot270":
        return c, -r
    if name == "flip_ud":
        return -r, c
    if name == "flip_lr":
        return r, -c
    raise ValueError(f"unknown transform {name}")


def _transform_coords(coords, name: str) -> np.ndarray:
    return _sorted_coords(_transform_point(pt, name) for pt in coords)


PATTERNS = {
    "basic_block": {
        "components": [
            {"role": "payload", "coords": ((0, -4), (-1, -4), (-2, -4), (-2, -3))},
            {"role": "source", "coords": ((1, 1), (1, 2)), "tag": "mirror_block"},
        ],
        "transforms": ("rot0", "rot90", "rot180", "rot270"),
        "open_index": None,
    },
    "tie_double_block": {
        "components": [
            {"role": "payload", "coords": ((1, -4), (0, -4), (-1, -4), (-2, -4), (-2, -3))},
            {"role": "source", "coords": ((1, 1), (1, 2)), "tag": "mirror_block"},
            {"role": "source", "coords": ((-2, 1), (-1, 1), (-1, 2)), "tag": "tie_mirror_block"},
        ],
        "transforms": ("rot0", "rot180", "flip_ud", "flip_lr"),
        "open_index": None,
    },
    "open_and_block": {
        "components": [
            {"role": "payload", "coords": ((2, -4), (1, -4), (0, -4), (-1, -4), (-1, -3), (-1, -2))},
            {"role": "source", "coords": ((0, 1), (0, 2)), "tag": "mirror_block"},
            {"role": "source", "coords": ((-3, 0), (-2, 0), (-2, 1)), "tag": "boundary_open"},
        ],
        "transforms": ("rot0", "rot90", "rot180", "rot270"),
        "open_index": 1,
    },
}


def _slot_marker(slot: str, h: int, w: int) -> tuple[int, int]:
    row_lo, row_hi = (4, 5) if slot in {"tl", "tr"} else (h - 6, h - 5)
    col_lo, col_hi = (4, 5) if slot in {"tl", "bl"} else (w - 6, w - 5)
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _outward_dirs(slot: str) -> set[tuple[int, int]]:
    if slot == "tl":
        return {(-1, 0), (0, -1)}
    if slot == "tr":
        return {(-1, 0), (0, 1)}
    if slot == "bl":
        return {(1, 0), (0, -1)}
    return {(1, 0), (0, 1)}


def _build_scene(pattern_name: str, slot: str, marker: tuple[int, int]) -> dict | None:
    pattern = PATTERNS[pattern_name]
    components = pattern["components"]
    transforms = list(pattern["transforms"])
    if pattern["open_index"] is not None:
        valid_transforms = []
        for name in transforms:
            coords = _transform_coords(components[pattern["open_index"] + 1]["coords"], name)
            if _ray_direction((0, 0), coords) in _outward_dirs(slot):
                valid_transforms.append(name)
        transforms = valid_transforms
    if not transforms:
        return None

    transform = random.choice(transforms)
    colors = random.sample(PALETTE, len(components))
    built_components = []
    for comp_def, color in zip(components, colors):
        rel_coords = _transform_coords(comp_def["coords"], transform)
        if any(max(abs(int(r)), abs(int(c))) > 4 for r, c in rel_coords):
            return None
        abs_coords = rel_coords + np.array(marker, dtype=int)
        built_components.append(
            {
                "role": comp_def["role"],
                "coords": abs_coords,
                "color": int(color),
                "tag": comp_def.get("tag"),
            }
        )

    payload = next(comp for comp in built_components if comp["role"] == "payload")
    mirrored = _reflect_coords(payload["coords"], marker)
    if any(max(abs(int(r) - marker[0]), abs(int(c) - marker[1])) > 4 for r, c in mirrored):
        return None

    for i, comp_a in enumerate(built_components):
        for r, c in comp_a["coords"]:
            if max(abs(int(r) - marker[0]), abs(int(c) - marker[1])) > 4:
                return None
        for comp_b in built_components[i + 1 :]:
            cells_a = {tuple(map(int, pt)) for pt in comp_a["coords"]}
            cells_b = {tuple(map(int, pt)) for pt in comp_b["coords"]}
            for r, c in cells_a:
                for rr, cc in cells_b:
                    if max(abs(r - rr), abs(c - cc)) <= 1:
                        return None

    scene = {
        "name": pattern_name,
        "marker": tuple(map(int, marker)),
        "payload": next(comp for comp in built_components if comp["role"] == "payload"),
        "sources": [comp for comp in built_components if comp["role"] == "source"],
    }
    scene["sources"].sort(key=lambda comp: (comp["coords"][:, 0].min(), comp["coords"][:, 1].min()))
    return scene


def _paint_scene_input(grid: np.ndarray, scene: dict) -> bool:
    h, w = grid.shape
    mr, mc = scene["marker"]
    if not in_bounds((h, w), mr, mc) or int(grid[mr, mc]) != BG:
        return False
    grid[mr, mc] = RED
    for comp in [scene["payload"], *scene["sources"]]:
        for r, c in comp["coords"]:
            if not in_bounds((h, w), int(r), int(c)) or int(grid[int(r), int(c)]) != BG:
                return False
        for r, c in comp["coords"]:
            grid[int(r), int(c)] = int(comp["color"])
    return True


def _oracle_from_scenes(inp: np.ndarray, scenes: list[dict]) -> tuple[np.ndarray, list[dict]]:
    out = np.array(inp, dtype=int, copy=True)
    _mirror_payloads(out, scenes)
    ray_infos: list[dict] = []
    for scene in scenes:
        for source in scene["sources"]:
            cells, stop = _trace_ray(out, source, scene["marker"])
            ray_infos.append(
                {
                    "marker": scene["marker"],
                    "source_color": int(source["color"]),
                    "tag": source.get("tag"),
                    "cells": cells,
                    "stop": stop,
                    "direction": _ray_direction(scene["marker"], source["coords"]),
                }
            )
    return out, ray_infos


def _compare_scene_to_analysis(scene: dict, analyzed: dict) -> bool:
    if tuple(map(int, scene["marker"])) != tuple(map(int, analyzed["marker"])):
        return False
    expected = [scene["payload"], *scene["sources"]]
    actual = [analyzed["payload"], *analyzed["sources"]]
    expected.sort(key=lambda comp: (comp["coords"][:, 0].min(), comp["coords"][:, 1].min(), comp["coords"].shape[0]))
    actual.sort(key=lambda comp: (comp["coords"][:, 0].min(), comp["coords"][:, 1].min(), comp["coords"].shape[0]))
    if len(expected) != len(actual):
        return False
    for exp_comp, act_comp in zip(expected, actual):
        if int(exp_comp["color"]) != int(act_comp["color"]):
            return False
        if exp_comp["coords"].shape != act_comp["coords"].shape:
            return False
        if not np.array_equal(_sorted_coords(exp_comp["coords"]), _sorted_coords(act_comp["coords"])):
            return False
    return True


def _validate_case(inp: np.ndarray, scenes: list[dict]) -> bool:
    try:
        analyzed = _analyze_grid(inp)
    except AssertionError:
        return False
    if len(analyzed) != len(scenes):
        return False
    for scene, found in zip(sorted(scenes, key=lambda s: s["marker"]), analyzed):
        if not _compare_scene_to_analysis(scene, found):
            return False

    expected, ray_infos = _oracle_from_scenes(inp, scenes)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    all_ray_cells: set[tuple[int, int]] = set()
    has_mirror_stop = False
    has_boundary_stop = False
    has_tie_source = False
    for info in ray_infos:
        if info["cells"]:
            cell_set = set(info["cells"])
            if all_ray_cells & cell_set:
                return False
            all_ray_cells |= cell_set
        if info["tag"] and "mirror_block" in info["tag"] and info["stop"] != "blocked":
            return False
        if info["tag"] == "boundary_open" and info["stop"] != "boundary":
            return False
        if info["tag"] == "boundary_open":
            has_boundary_stop = True
        if info["stop"] == "blocked":
            has_mirror_stop = True
        if info["tag"] == "tie_mirror_block":
            marker = info["marker"]
            scene = next(scene for scene in scenes if scene["marker"] == marker)
            tie_source = next(source for source in scene["sources"] if source.get("tag") == "tie_mirror_block")
            cr, cc = _centroid(tie_source["coords"])
            dr = abs(cr - marker[0])
            dc = abs(cc - marker[1])
            if abs(dr - dc) > 1e-6:
                return False
            if info["direction"][0] != 0:
                return False
            has_tie_source = True
        if not info["cells"] and info["tag"] != "mirror_block":
            return False

    if not has_mirror_stop or not has_tie_source or not has_boundary_stop:
        return False
    return True


def _pattern_schedule(n_markers: int) -> list[str]:
    base = ["tie_double_block", "open_and_block"]
    if n_markers == 2:
        return base
    extra = random.choice(["basic_block", "tie_double_block", "open_and_block"])
    items = base + [extra]
    random.shuffle(items)
    return items


def _build_candidate() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    n_markers = random.choice([2, 3])
    slots = random.sample(SLOT_NAMES, n_markers)
    pattern_names = _pattern_schedule(n_markers)

    scenes: list[dict] = []
    grid = np.full((h, w), BG, dtype=int)
    for slot, pattern_name in zip(slots, pattern_names):
        marker = _slot_marker(slot, h, w)
        scene = _build_scene(pattern_name, slot, marker)
        if scene is None:
            return None
        scenes.append(scene)

    scenes.sort(key=lambda scene: scene["marker"])
    for scene in scenes:
        if not _paint_scene_input(grid, scene):
            return None

    if not _validate_case(grid, scenes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
