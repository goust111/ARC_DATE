import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
MARKER = 9
PALETTE = (1, 2, 3, 4, 7)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_SLIDE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
SIDE_TO_OUTWARD = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _find_gray_rectangle(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    grey = grid == GREY
    coords = np.argwhere(grey)
    assert len(coords) >= 9, "expected a gray rectangle"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros_like(grey, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert np.array_equal(mask, grey), "gray object must be a solid rectangle"
    assert 3 <= (r1 - r0 + 1) <= 4 and 3 <= (c1 - c0 + 1) <= 4
    return grey, (int(r0), int(c0), int(r1), int(c1))


def _parse_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    colored_positions = list(zip(*np.where((grid != BG) & (grid != GREY) & (grid != MARKER))))
    marker_positions = set((int(r), int(c)) for r, c in zip(*np.where(grid == MARKER)))
    seen_markers = set()
    used_colors = set()
    for r0, c0 in colored_positions:
        r = int(r0)
        c = int(c0)
        color = int(grid[r, c])
        assert color not in used_colors, "colored pixels must have distinct colors"
        used_colors.add(color)
        adjacent_markers = []
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in marker_positions:
                adjacent_markers.append((nr, nc))
        assert len(adjacent_markers) == 1, "each colored pixel needs exactly one marker"
        mr, mc = adjacent_markers[0]
        seen_markers.add((mr, mc))
        objects.append(
            {
                "color": color,
                "start": (r, c),
                "marker": (mr, mc),
                "ray_dir": (mr - r, mc - c),
            }
        )
    assert len(seen_markers) == len(marker_positions), "each marker must belong to one colored pixel"
    assert 3 <= len(objects) <= 5, "expected 3-5 colored pixels"
    return objects


def _distance_and_direction_to_body(pos: tuple[int, int], body: np.ndarray) -> tuple[int, tuple[int, int]]:
    h, w = body.shape
    r, c = pos
    candidates = []
    for dr, dc in DIRS4:
        steps = 0
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w and not body[nr, nc]:
            steps += 1
            nr += dr
            nc += dc
        if 0 <= nr < h and 0 <= nc < w and body[nr, nc]:
            candidates.append((steps, (dr, dc)))
    assert candidates, "each colored pixel must have a cardinal route to the body"
    best_steps = min(steps for steps, _ in candidates)
    best_dirs = [direction for steps, direction in candidates if steps == best_steps]
    assert len(best_dirs) == 1, "shortest docking direction must be unique"
    return best_steps, best_dirs[0]


def _slide_with_order(objects: list[dict], initial_body: np.ndarray, order: list[int]) -> tuple[list[dict], np.ndarray]:
    body = initial_body.copy()
    finals = []
    for idx in order:
        obj = objects[idx]
        steps, slide_dir = _distance_and_direction_to_body(obj["start"], body)
        dr, dc = slide_dir
        fr = obj["start"][0] + dr * steps
        fc = obj["start"][1] + dc * steps
        assert not body[fr, fc], "final docking cell must be empty"
        body[fr, fc] = True
        finals.append(
            {
                **obj,
                "slide_dir": slide_dir,
                "slide_steps": steps,
                "final": (fr, fc),
            }
        )
    return finals, body


def _canonical_order(objects: list[dict], initial_body: np.ndarray) -> list[int]:
    keyed = []
    for idx, obj in enumerate(objects):
        dist, _ = _distance_and_direction_to_body(obj["start"], initial_body)
        r, c = obj["start"]
        keyed.append((dist, r, c, idx))
    keyed.sort()
    return [idx for _, _, _, idx in keyed]


def _render_post_slide(grid: np.ndarray, finals: list[dict]) -> np.ndarray:
    out = np.where(grid == MARKER, BG, grid)
    out[out != GREY] = BG
    for obj in finals:
        r, c = obj["final"]
        out[r, c] = obj["color"]
    return out


def _ray_cells(start: tuple[int, int], ray_dir: tuple[int, int], blocked: np.ndarray) -> tuple[list[tuple[int, int]], str]:
    h, w = blocked.shape
    r, c = start
    dr, dc = ray_dir
    cells = []
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w:
        if blocked[nr, nc]:
            return cells, "obstacle"
        cells.append((nr, nc))
        nr += dr
        nc += dc
    return cells, "boundary"


def _apply_rays(post_slide: np.ndarray, finals: list[dict]) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[str]]:
    blocked = post_slide != BG
    out = post_slide.copy()
    ray_paths = []
    reasons = []
    claimed = set()
    for obj in finals:
        cells, reason = _ray_cells(obj["final"], obj["ray_dir"], blocked)
        for cell in cells:
            assert cell not in claimed, "ray paint cells must be unique"
            claimed.add(cell)
        ray_paths.append(cells)
        reasons.append(reason)
        for r, c in cells:
            out[r, c] = obj["color"]
    return out, ray_paths, reasons


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    gray_body, gray_bbox = _find_gray_rectangle(g)
    objects = _parse_objects(g)
    order = _canonical_order(objects, gray_body)
    finals, accumulated_body = _slide_with_order(objects, gray_body, order)
    post_slide = _render_post_slide(g, finals)
    out, ray_paths, ray_reasons = _apply_rays(post_slide, finals)
    meta = {
        "gray_bbox": gray_bbox,
        "objects": objects,
        "order": order,
        "finals": finals,
        "accumulated_body": accumulated_body,
        "post_slide": post_slide,
        "ray_paths": ray_paths,
        "ray_reasons": ray_reasons,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored pixels onto the growing gray body, then emit rays from their docked positions.

    Input structure:
    - A black background contains one solid gray 3x3 or 4x4 rectangle.
    - Around it are 3-5 single colored pixels, each with one adjacent maroon marker.
    - The marker sits on one cardinal side of its colored pixel and encodes the later ray direction.

    Rule:
    - Sort colored pixels by their original shortest cardinal distance to the gray rectangle; break ties top-to-bottom, then left-to-right.
    - Process them in that order: for each one, find its current shortest cardinal docking direction to the growing gray body, slide it until the next step would collide, and treat the docked colored pixel as part of the body for later dockings.
    - Remove all maroon markers, then from every docked colored pixel emit a same-color ray in its original marker direction.
    - Each ray paints black cells only and stops just before the first non-black cell or the grid boundary; the gray rectangle stays gray.

    Key insight:
    - Docking is cumulative: early attachments change where later pixels stop, and that changed final layout determines the ray lengths.
    """
    out, _ = _solve(grid)
    return out


def _make_position(side: str, lane: int, distance: int, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0 - 1 - distance, c0 + lane
    if side == "bottom":
        return r1 + 1 + distance, c0 + lane
    if side == "left":
        return r0 + lane, c0 - 1 - distance
    if side == "right":
        return r0 + lane, c1 + 1 + distance
    raise ValueError(side)


def _lane_span(side: str, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        return c1 - c0 + 1
    return r1 - r0 + 1


def _occupied_from_specs(specs: list[dict], bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    occ = set()
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            occ.add((r, c))
    for spec in specs:
        occ.add(spec["pos"])
    return occ


def _paths_to_initial_body(specs: list[dict], gray_body: np.ndarray) -> list[set[tuple[int, int]]]:
    paths = []
    for spec in specs:
        steps, slide_dir = _distance_and_direction_to_body(spec["pos"], gray_body)
        dr, dc = slide_dir
        r, c = spec["pos"]
        path = set()
        for k in range(1, steps + 1):
            path.add((r + dr * k, c + dc * k))
        paths.append(path)
    return paths


def _choose_marker_dirs(
    rng: random.Random,
    specs: list[dict],
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    special: dict[int, tuple[int, int]],
) -> bool:
    gray_body = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    gray_body[r0 : r1 + 1, c0 : c1 + 1] = True
    paths = _paths_to_initial_body(specs, gray_body)
    occupied = _occupied_from_specs(specs, bbox)
    all_path_cells = set().union(*paths)
    for idx, spec in enumerate(specs):
        slide_dir = SIDE_TO_SLIDE[spec["side"]]
        candidates = []
        for ray_dir in [SIDE_TO_OUTWARD[spec["side"]], (slide_dir[1], slide_dir[0]), (-slide_dir[1], -slide_dir[0])]:
            mr = spec["pos"][0] + ray_dir[0]
            mc = spec["pos"][1] + ray_dir[1]
            if not _in_bounds(shape, mr, mc):
                continue
            if (mr, mc) in occupied:
                continue
            if (mr, mc) in (all_path_cells - paths[idx]):
                continue
            candidates.append(ray_dir)
        if idx in special:
            if special[idx] not in candidates:
                return False
            spec["ray_dir"] = special[idx]
        else:
            if not candidates:
                return False
            spec["ray_dir"] = rng.choice(candidates)
        occupied.add((spec["pos"][0] + spec["ray_dir"][0], spec["pos"][1] + spec["ray_dir"][1]))
    return True


def _render_input(shape: tuple[int, int], bbox: tuple[int, int, int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for spec in specs:
        r, c = spec["pos"]
        mr, mc = r + spec["ray_dir"][0], c + spec["ray_dir"][1]
        grid[r, c] = spec["color"]
        grid[mr, mc] = MARKER
    return grid


def _analyze_instance(grid: np.ndarray) -> dict:
    out, meta = _solve(grid)
    gray_body, _ = _find_gray_rectangle(grid)
    objects = _parse_objects(grid)
    reverse_only, _ = _slide_with_order(objects, gray_body, list(reversed(meta["order"])))
    reverse_post = _render_post_slide(grid, reverse_only)
    reverse_out, reverse_rays, _ = _apply_rays(reverse_post, reverse_only)
    changed_by_accumulation = False
    actual_positions = {obj["color"]: obj["final"] for obj in meta["finals"]}
    for obj in objects:
        steps, slide_dir = _distance_and_direction_to_body(obj["start"], gray_body)
        dr, dc = slide_dir
        static_pos = (obj["start"][0] + dr * steps, obj["start"][1] + dc * steps)
        if actual_positions[obj["color"]] != static_pos:
            changed_by_accumulation = True
            break
    distances = [_distance_and_direction_to_body(obj["start"], gray_body)[0] for obj in objects]
    tie = len(distances) != len(set(distances))
    ray_lengths = [len(path) for path in meta["ray_paths"]]
    return {
        "out": out,
        "post_slide": meta["post_slide"],
        "ray_lengths": ray_lengths,
        "max_ray_len": max(ray_lengths),
        "total_ray_cells": sum(ray_lengths),
        "obstacle_stops": sum(reason == "obstacle" for reason in meta["ray_reasons"]),
        "boundary_stops": sum(reason == "boundary" for reason in meta["ray_reasons"]),
        "changed_by_accumulation": changed_by_accumulation,
        "has_tie": tie,
        "reverse_diff": not np.array_equal(out, reverse_out),
        "reverse_ray_total": sum(len(path) for path in reverse_rays),
    }


def _build_specs(
    rng: random.Random,
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    count: int,
    force_tie: bool,
) -> list[dict] | None:
    primary_choices = [side for side in ["top", "bottom", "left", "right"] if _lane_span(side, bbox) >= 4]
    if not primary_choices:
        return None
    primary = rng.choice(primary_choices)
    span = _lane_span(primary, bbox)
    separated_pairs = [(a, b) for a in range(span) for b in range(span) if abs(a - b) >= 3]
    if not separated_pairs:
        return None
    lane_chain, lane_mate = rng.choice(separated_pairs)
    d_near = rng.randint(1, 2)
    d_far = rng.randint(d_near + 2, 4)

    specs = [
        {"side": primary, "lane": lane_chain, "distance": d_near, "pos": _make_position(primary, lane_chain, d_near, bbox)},
        {"side": primary, "lane": lane_chain, "distance": d_far, "pos": _make_position(primary, lane_chain, d_far, bbox)},
        {
            "side": primary,
            "lane": lane_mate,
            "distance": d_near if force_tie else rng.randint(1, 4),
            "pos": None,
        },
    ]
    specs[2]["pos"] = _make_position(primary, lane_mate, specs[2]["distance"], bbox)

    if primary in ("top", "bottom"):
        other_sides = ["left", "right"]
    else:
        other_sides = ["top", "bottom"]
    used_positions = {spec["pos"] for spec in specs}
    while len(specs) < count:
        side = rng.choice(other_sides)
        lane = rng.randrange(_lane_span(side, bbox))
        distance = rng.randint(1, 4)
        pos = _make_position(side, lane, distance, bbox)
        if pos in used_positions:
            continue
        if any(abs(pos[0] - rr) + abs(pos[1] - cc) == 1 for rr, cc in used_positions):
            continue
        specs.append({"side": side, "lane": lane, "distance": distance, "pos": pos})
        used_positions.add(pos)

    for spec in specs:
        r, c = spec["pos"]
        if not _in_bounds(shape, r, c):
            return None

    colors = list(PALETTE)
    rng.shuffle(colors)
    for spec, color in zip(specs, colors):
        spec["color"] = color

    toward = None
    away = None
    if primary in ("top", "bottom"):
        if lane_mate > lane_chain:
            toward = (0, -1)
            away = (0, -1)
        else:
            toward = (0, 1)
            away = (0, 1)
    else:
        if lane_mate > lane_chain:
            toward = (-1, 0)
            away = (-1, 0)
        else:
            toward = (1, 0)
            away = (1, 0)

    special = {
        0: away,
        1: SIDE_TO_OUTWARD[primary],
        2: toward,
    }
    if not _choose_marker_dirs(rng, specs, bbox, shape, special):
        return None
    return specs


def _valid_layout_position(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return 0 <= r0 < r1 < h and 0 <= c0 < c1 < w


def _gen_attempt(rng: random.Random, force_tie: bool) -> np.ndarray | None:
    h = rng.randint(15, 19)
    w = rng.randint(15, 19)
    rect_h = rng.choice([3, 4])
    rect_w = rng.choice([3, 4])
    row_min = 5
    row_max = h - rect_h - 5
    col_min = 5
    col_max = w - rect_w - 5
    if row_min > row_max or col_min > col_max:
        return None
    r0 = rng.randint(row_min, row_max)
    c0 = rng.randint(col_min, col_max)
    bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    if not _valid_layout_position((h, w), bbox):
        return None
    center_r = r0 + (rect_h - 1) / 2.0
    center_c = c0 + (rect_w - 1) / 2.0
    if abs(center_r - (h - 1) / 2.0) < 0.6 and abs(center_c - (w - 1) / 2.0) < 0.6:
        return None

    count = rng.randint(3, 5)
    specs = _build_specs(rng, (h, w), bbox, count, force_tie)
    if specs is None:
        return None

    grid = _render_input((h, w), bbox, specs)
    try:
        analysis = _analyze_instance(grid)
    except AssertionError:
        return None

    if analysis["total_ray_cells"] < 4:
        return None
    if analysis["max_ray_len"] < 2:
        return None
    if analysis["obstacle_stops"] < 1 or analysis["boundary_stops"] < 1:
        return None
    if not analysis["changed_by_accumulation"]:
        return None
    if not analysis["reverse_diff"]:
        return None
    if force_tie and not analysis["has_tie"]:
        return None
    if np.array_equal(analysis["out"], analysis["post_slide"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for attempt in range(500):
        force_tie = attempt % 2 == 0
        grid = _gen_attempt(rng, force_tie=force_tie)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
