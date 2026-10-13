from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE_SIDE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
SHAPES_BY_SIDE = {
    "up": [(2, 3), (2, 4), (3, 3)],
    "down": [(2, 3), (2, 4), (3, 3)],
    "left": [(3, 2), (4, 2), (3, 3)],
    "right": [(3, 2), (4, 2), (3, 3)],
}
ALL_SIDES = tuple(SIDE_TO_VEC)
ALL_COLORS = list(range(1, 10))
GEN_MODES = ("line_only", "collision", "crossing", "both")
GEN_RETRY_LIMIT = 500
FILLER_RETRY_LIMIT = 120


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _infer_emitter(component: dict) -> dict:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    missing = np.argwhere(~mask)
    assert missing.shape[0] == 1, "expected one bbox notch"
    mr, mc = map(int, missing[0])

    sides = []
    if mr == 0:
        sides.append("up")
    if mr == h - 1:
        sides.append("down")
    if mc == 0:
        sides.append("left")
    if mc == w - 1:
        sides.append("right")
    assert len(sides) == 1, "expected a unique notch side"

    side = sides[0]
    origin = (r0 + mr, c0 + mc)
    return {
        "component": component,
        "color": int(component["color"]),
        "area": int(component["area"]),
        "side": side,
        "origin": origin,
        "vec": SIDE_TO_VEC[side],
    }


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    components = _find_components(g)
    emitters = [_infer_emitter(component) for component in components]

    owner = np.full((h, w), -1, dtype=int)
    for index, component in enumerate(components):
        coords = component["coords"]
        owner[coords[:, 0], coords[:, 1]] = index

    line_claims: dict[tuple[int, int], list[int]] = {}
    hit_claims: dict[int, list[int]] = {}

    for emitter_index, emitter in enumerate(emitters):
        r, c = emitter["origin"]
        dr, dc = emitter["vec"]
        for _ in range(int(emitter["area"])):
            if not (0 <= r < h and 0 <= c < w):
                break
            target_index = int(owner[r, c])
            if target_index != -1:
                if target_index != emitter_index:
                    hit_claims.setdefault(target_index, []).append(emitter_index)
                break
            line_claims.setdefault((r, c), []).append(emitter_index)
            r += dr
            c += dc

    def _winner(indices: list[int]) -> int:
        return min(indices, key=lambda idx: tuple(emitters[idx]["origin"]))

    line_winners = {cell: _winner(indices) for cell, indices in line_claims.items()}
    hit_winners = {target: _winner(indices) for target, indices in hit_claims.items()}
    crossing_cells = {
        cell for cell, indices in line_claims.items() if len(set(indices)) > 1
    }
    return {
        "components": components,
        "emitters": emitters,
        "line_claims": line_claims,
        "line_winners": line_winners,
        "hit_claims": hit_claims,
        "hit_winners": hit_winners,
        "crossing_cells": crossing_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an area-length ray from each object's unique side notch and let hits recolor whole target objects.

    Input structure:
    - The background is black.
    - Every non-black 4-connected object is monochrome and almost fills its
      bounding box, except for one missing side-edge cell.
    - The missing cell lies on exactly one of the four bbox sides and defines a
      unique outward direction.

    Rule:
    - For each object, find its bbox notch, use that missing cell as the ray
      origin, and shoot outward in the notch direction.
    - The ray color matches the emitting object, and its intended length equals
      the object's pixel count, clipped by the grid boundary.
    - Rays are evaluated simultaneously against the original objects: when a ray
      first reaches another object's pixel, it stops there and recolors that
      entire object to the emitter's color.
    - If multiple rays claim the same empty cell, keep the color from the ray
      whose origin has the smaller row, then smaller column.

    Key insight:
    - The single missing bbox cell encodes both ray origin and direction, while
      object area supplies the ray length and simultaneous ray interactions
      decide recoloring and crossing outcomes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    analysis = _analyze_scene(g)
    out = g.copy()

    for target_index, emitter_index in analysis["hit_winners"].items():
        color = int(analysis["emitters"][emitter_index]["color"])
        coords = analysis["components"][target_index]["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    for (r, c), emitter_index in analysis["line_winners"].items():
        out[r, c] = int(analysis["emitters"][emitter_index]["color"])

    return out


def _notch_choices(side: str, h: int, w: int) -> list[int]:
    if side in ("up", "down"):
        return list(range(1, w - 1))
    return list(range(1, h - 1))


def _build_object(
    side: str,
    h: int,
    w: int,
    origin: tuple[int, int],
    notch_index: int,
    color: int,
) -> dict:
    origin_r, origin_c = map(int, origin)
    if side == "up":
        r0 = origin_r
        c0 = origin_c - notch_index
        missing = (0, notch_index)
    elif side == "down":
        r0 = origin_r - (h - 1)
        c0 = origin_c - notch_index
        missing = (h - 1, notch_index)
    elif side == "left":
        r0 = origin_r - notch_index
        c0 = origin_c
        missing = (notch_index, 0)
    else:
        assert side == "right"
        r0 = origin_r - notch_index
        c0 = origin_c - (w - 1)
        missing = (notch_index, w - 1)

    cells = []
    for rr in range(h):
        for cc in range(w):
            if (rr, cc) != missing:
                cells.append((r0 + rr, c0 + cc))

    coords = np.array(cells, dtype=int)
    return {
        "side": side,
        "h": int(h),
        "w": int(w),
        "color": int(color),
        "origin": (origin_r, origin_c),
        "notch_index": int(notch_index),
        "cells": coords,
        "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
        "area": int(coords.shape[0]),
    }


def _object_in_bounds(obj: dict, height: int, width: int) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    return 0 <= r0 <= r1 < height and 0 <= c0 <= c1 < width


def _object_touches_existing(obj: dict, objects: list[dict]) -> bool:
    occupied = {
        (int(r), int(c))
        for other in objects
        for r, c in other["cells"]
    }
    for r_raw, c_raw in obj["cells"]:
        r = int(r_raw)
        c = int(c_raw)
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return True
    return False


def _render_grid(objects: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for obj in objects:
        coords = obj["cells"]
        grid[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return grid


def _mode_matches(grid: np.ndarray, mode: str) -> bool:
    analysis = _analyze_scene(grid)
    has_hit = bool(analysis["hit_winners"])
    has_cross = bool(analysis["crossing_cells"])
    if mode == "line_only":
        return (not has_hit) and (not has_cross)
    if mode == "collision":
        return has_hit and (not has_cross)
    if mode == "crossing":
        return has_cross and (not has_hit)
    assert mode == "both"
    return has_hit and has_cross


def _sample_object(side: str | None, color: int) -> dict:
    chosen_side = random.choice(ALL_SIDES) if side is None else side
    h, w = random.choice(SHAPES_BY_SIDE[chosen_side])
    notch_index = random.choice(_notch_choices(chosen_side, h, w))
    return {
        "side": chosen_side,
        "h": int(h),
        "w": int(w),
        "notch_index": int(notch_index),
        "color": int(color),
    }


def _face_cells(obj: dict, face: str) -> list[tuple[int, int]]:
    coords = obj["cells"]
    if face == "up":
        edge = int(coords[:, 0].min())
        return [(int(r), int(c)) for r, c in coords if int(r) == edge]
    if face == "down":
        edge = int(coords[:, 0].max())
        return [(int(r), int(c)) for r, c in coords if int(r) == edge]
    if face == "left":
        edge = int(coords[:, 1].min())
        return [(int(r), int(c)) for r, c in coords if int(c) == edge]
    edge = int(coords[:, 1].max())
    return [(int(r), int(c)) for r, c in coords if int(c) == edge]


def _build_crossing_pair(
    height: int,
    width: int,
    colors: list[int],
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
) -> list[dict] | None:
    row_lo, row_hi = row_range if row_range is not None else (2, height - 3)
    col_lo, col_hi = col_range if col_range is not None else (2, width - 3)
    row_lo = max(1, row_lo)
    row_hi = min(height - 2, row_hi)
    col_lo = max(1, col_lo)
    col_hi = min(width - 2, col_hi)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(120):
        vertical_side = random.choice(("up", "down"))
        horizontal_side = random.choice(("left", "right"))
        v_spec = _sample_object(vertical_side, colors[0])
        h_spec = _sample_object(horizontal_side, colors[1])
        cross_r = random.randint(row_lo, row_hi)
        cross_c = random.randint(col_lo, col_hi)
        v_dr, v_dc = SIDE_TO_VEC[vertical_side]
        h_dr, h_dc = SIDE_TO_VEC[horizontal_side]
        v_delta = random.randint(1, v_spec["h"] * v_spec["w"] - 2)
        h_delta = random.randint(1, h_spec["h"] * h_spec["w"] - 2)
        v_origin = (cross_r - v_delta * v_dr, cross_c - v_delta * v_dc)
        h_origin = (cross_r - h_delta * h_dr, cross_c - h_delta * h_dc)
        v_obj = _build_object(
            v_spec["side"],
            v_spec["h"],
            v_spec["w"],
            v_origin,
            v_spec["notch_index"],
            v_spec["color"],
        )
        h_obj = _build_object(
            h_spec["side"],
            h_spec["h"],
            h_spec["w"],
            h_origin,
            h_spec["notch_index"],
            h_spec["color"],
        )
        objects = [v_obj]
        if not _object_in_bounds(v_obj, height, width):
            continue
        if not _object_in_bounds(h_obj, height, width):
            continue
        if _object_touches_existing(h_obj, objects):
            continue
        objects.append(h_obj)
        grid = _render_grid(objects, height, width)
        if _mode_matches(grid, "crossing"):
            return objects
    return None


def _build_collision_pair(
    height: int,
    width: int,
    colors: list[int],
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
) -> list[dict] | None:
    row_lo, row_hi = row_range if row_range is not None else (1, height - 2)
    col_lo, col_hi = col_range if col_range is not None else (1, width - 2)
    row_lo = max(1, row_lo)
    row_hi = min(height - 2, row_hi)
    col_lo = max(1, col_lo)
    col_hi = min(width - 2, col_hi)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(160):
        emitter_side = random.choice(ALL_SIDES)
        target_side_options = [side for side in ALL_SIDES if side != OPPOSITE_SIDE[emitter_side]]
        target_side = random.choice(target_side_options)
        target_spec = _sample_object(target_side, colors[1])
        e_spec = _sample_object(emitter_side, colors[0])

        target_origin = (
            random.randint(row_lo, row_hi),
            random.randint(col_lo, col_hi),
        )
        target_obj = _build_object(
            target_spec["side"],
            target_spec["h"],
            target_spec["w"],
            target_origin,
            target_spec["notch_index"],
            target_spec["color"],
        )
        if not _object_in_bounds(target_obj, height, width):
            continue

        face = OPPOSITE_SIDE[emitter_side]
        candidates = _face_cells(target_obj, face)
        if not candidates:
            continue
        hit_r, hit_c = random.choice(candidates)
        dr, dc = SIDE_TO_VEC[emitter_side]
        max_steps = e_spec["h"] * e_spec["w"] - 2
        delta = random.randint(1, max_steps)
        emitter_origin = (hit_r - delta * dr, hit_c - delta * dc)
        emitter_obj = _build_object(
            e_spec["side"],
            e_spec["h"],
            e_spec["w"],
            emitter_origin,
            e_spec["notch_index"],
            e_spec["color"],
        )
        if not _object_in_bounds(emitter_obj, height, width):
            continue
        if _object_touches_existing(emitter_obj, [target_obj]):
            continue

        objects = [emitter_obj, target_obj]
        grid = _render_grid(objects, height, width)
        if _mode_matches(grid, "collision"):
            return objects
    return None


def _sample_edge_object(height: int, width: int, color: int) -> dict | None:
    for _ in range(40):
        spec = _sample_object(None, color)
        side = spec["side"]
        h = spec["h"]
        w = spec["w"]
        notch = spec["notch_index"]
        if side == "up":
            origin_r = random.randint(0, min(3, height - h))
            origin_c = random.randint(notch, width - (w - notch))
        elif side == "down":
            origin_r = random.randint(max(h - 1, height - 4), height - 1)
            origin_c = random.randint(notch, width - (w - notch))
        elif side == "left":
            origin_r = random.randint(notch, height - (h - notch))
            origin_c = random.randint(0, min(3, width - w))
        else:
            origin_r = random.randint(notch, height - (h - notch))
            origin_c = random.randint(max(w - 1, width - 4), width - 1)

        obj = _build_object(side, h, w, (origin_r, origin_c), notch, color)
        if _object_in_bounds(obj, height, width):
            return obj
    return None


def _try_add_fillers(
    base_objects: list[dict],
    height: int,
    width: int,
    target_count: int,
    mode: str,
    available_colors: list[int],
) -> list[dict] | None:
    objects = list(base_objects)
    remaining_colors = list(available_colors)
    random.shuffle(remaining_colors)

    while len(objects) < target_count:
        added = False
        for _ in range(FILLER_RETRY_LIMIT):
            if not remaining_colors:
                return None
            color = remaining_colors[0]
            candidate = _sample_edge_object(height, width, color)
            if candidate is None:
                continue
            if _object_touches_existing(candidate, objects):
                continue
            trial = objects + [candidate]
            grid = _render_grid(trial, height, width)
            if _mode_matches(grid, mode):
                objects = trial
                remaining_colors.pop(0)
                added = True
                break
        if not added:
            return None
    return objects


def _make_line_only(height: int, width: int, colors: list[int], target_count: int) -> list[dict] | None:
    return _try_add_fillers([], height, width, target_count, "line_only", colors)


def _make_crossing(height: int, width: int, colors: list[int], target_count: int) -> list[dict] | None:
    core = _build_crossing_pair(height, width, colors[:2])
    if core is None:
        return None
    return _try_add_fillers(core, height, width, target_count, "crossing", colors[2:])


def _make_collision(height: int, width: int, colors: list[int], target_count: int) -> list[dict] | None:
    core = _build_collision_pair(height, width, colors[:2])
    if core is None:
        return None
    return _try_add_fillers(core, height, width, target_count, "collision", colors[2:])


def _make_both(height: int, width: int, colors: list[int], target_count: int) -> list[dict] | None:
    cross_rows = (2, max(2, height // 2 - 2))
    cross_cols = (2, max(2, width // 2 - 2))
    collision_rows = (max(1, height // 2 + 1), height - 2)
    collision_cols = (max(1, width // 2 + 1), width - 2)

    core_cross = _build_crossing_pair(height, width, colors[:2], cross_rows, cross_cols)
    if core_cross is None:
        return None
    core_collision = _build_collision_pair(
        height,
        width,
        colors[2:4],
        collision_rows,
        collision_cols,
    )
    if core_collision is None:
        return None

    objects = list(core_cross)
    for obj in core_collision:
        if _object_touches_existing(obj, objects):
            return None
        objects.append(obj)

    grid = _render_grid(objects, height, width)
    if not _mode_matches(grid, "both"):
        return None
    return _try_add_fillers(objects, height, width, target_count, "both", colors[4:])


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        mode = random.choice(GEN_MODES)
        if mode == "both":
            height = random.randint(20, 28)
            width = random.randint(20, 28)
            target_count = random.randint(4, 5)
        else:
            height = random.randint(14, 24)
            width = random.randint(14, 24)
            target_count = random.randint(2, 5)

        colors = random.sample(ALL_COLORS, target_count)
        if mode == "line_only":
            objects = _make_line_only(height, width, colors, target_count)
        elif mode == "collision":
            objects = _make_collision(height, width, colors, target_count)
        elif mode == "crossing":
            objects = _make_crossing(height, width, colors, target_count)
        else:
            objects = _make_both(height, width, colors, target_count)

        if objects is None:
            continue

        grid = _render_grid(objects, height, width)
        if not _mode_matches(grid, mode):
            continue
        _ = main(grid)
        return grid

    raise RuntimeError("failed to generate a valid ARC input")
