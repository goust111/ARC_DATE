from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MARKER_COLORS = (1, 2, 3, 4)
TRIGGER_COLOR = 6
TARGET_COLOR = 8

GRID_SIZE_RANGE = (13, 18)
OBJECT_BOX_RANGE = (3, 5)
OBJECT_AREA_RANGE = (5, 10)
MARKER_COUNT_CHOICES = (3, 4)
GENERATION_RETRIES = 500
OBJECT_SHAPE_RETRIES = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _target_mask(grid: np.ndarray) -> np.ndarray:
    mask = np.asarray(grid == TARGET_COLOR, dtype=bool)
    comps = _components(mask)
    assert len(comps) == 1, "expected exactly one teal target object"
    return mask


def _clockwise_perimeter(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    coords = [(0, c) for c in range(w)]
    coords.extend((r, w - 1) for r in range(1, h))
    coords.extend((h - 1, c) for c in range(w - 2, -1, -1))
    coords.extend((r, 0) for r in range(h - 2, 0, -1))
    return coords


def _projection_direction(r: int, c: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if c == 0:
        return (0, 1)
    if c == w - 1:
        return (0, -1)
    if r == 0:
        return (1, 0)
    if r == h - 1:
        return (-1, 0)
    raise AssertionError("marker must lie on the border")


def _perpendicular_dirs(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (-dc, dr), (dc, -dr)


def _path_from_marker(marker: tuple[int, int], target: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    mr, mc = marker
    tr, tc = target
    dr, dc = _projection_direction(mr, mc, shape)
    path: list[tuple[int, int]] = []
    r, c = mr + dr, mc + dc
    while _in_bounds(shape, r, c):
        path.append((r, c))
        if (r, c) == (tr, tc):
            return path
        r += dr
        c += dc
    raise AssertionError("target must be reachable from marker")


def _solve(grid: np.ndarray, *, collect: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    target = _target_mask(g)
    triggers = {tuple(map(int, rc)) for rc in np.argwhere(g == TRIGGER_COLOR)}

    markers: list[tuple[int, int, int]] = []
    for r, c in _clockwise_perimeter((h, w)):
        color = int(g[r, c])
        if color not in (BACKGROUND, TRIGGER_COLOR, TARGET_COLOR):
            markers.append((r, c, color))
    assert markers, "expected at least one border marker"

    out = g.copy()
    details = []
    for mr, mc, color in markers:
        dr, dc = _projection_direction(mr, mc, (h, w))
        projection_cells: list[tuple[int, int]] = []
        projection_overwrites: list[tuple[int, int]] = []
        trigger_hit: tuple[int, int] | None = None
        target_hit: tuple[int, int] | None = None

        r, c = mr + dr, mc + dc
        while _in_bounds((h, w), r, c):
            if (r, c) in triggers:
                trigger_hit = (r, c)
                break
            if target[r, c]:
                out[r, c] = color
                projection_cells.append((r, c))
                target_hit = (r, c)
                break
            if out[r, c] != BACKGROUND:
                projection_overwrites.append((r, c))
            out[r, c] = color
            projection_cells.append((r, c))
            r += dr
            c += dc

        assert trigger_hit is not None or target_hit is not None, "projection must terminate on trigger or target"

        ray_details = []
        if trigger_hit is not None:
            for br, bc in _perpendicular_dirs(dr, dc):
                ray_cells: list[tuple[int, int]] = []
                stop_reason = "out"
                stop_cell: tuple[int, int] | None = None
                nr, nc = trigger_hit[0] + br, trigger_hit[1] + bc
                while _in_bounds((h, w), nr, nc):
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        stop_reason = "boundary"
                        stop_cell = (nr, nc)
                        break
                    if out[nr, nc] != BACKGROUND:
                        stop_reason = "occupied"
                        stop_cell = (nr, nc)
                        break
                    out[nr, nc] = color
                    ray_cells.append((nr, nc))
                    nr += br
                    nc += bc
                ray_details.append(
                    {
                        "dir": (br, bc),
                        "cells": ray_cells,
                        "stop_reason": stop_reason,
                        "stop_cell": stop_cell,
                    }
                )

        if collect:
            details.append(
                {
                    "marker": (mr, mc),
                    "color": color,
                    "direction": (dr, dc),
                    "projection_cells": projection_cells,
                    "projection_overwrites": projection_overwrites,
                    "trigger": trigger_hit,
                    "target": target_hit,
                    "rays": ray_details,
                }
            )

    return (out, details) if collect else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each border marker toward the teal object, splitting into perpendicular rays when a magenta trigger is encountered.

    Input structure:
    - The grid has a black background, one compact teal target object, several single-pixel colored markers on the border, and one or more interior magenta trigger pixels.
    - Each border marker projects inward along the unique row or column determined by its border side.
    - Corner markers, if present, use the horizontal inward direction.

    Rule:
    - Process border markers in clockwise order starting from the top-left perimeter cell.
    - For each marker, paint its inward projection until it either reaches the teal object or encounters a magenta trigger.
    - If no trigger is met, continue through the first teal surface pixel and recolor that target pixel with the marker color.
    - If a trigger is met, stop the original projection just before the trigger, keep the trigger unchanged, and emit two perpendicular rays from the trigger position.
    - Each split ray paints black cells only and stops before the first non-black cell or before entering a border cell.

    Key insight:
    - A trigger converts a straight border-to-target projection into a T-shaped structure: the main beam is cut short, and only the rotated side rays continue to propagate.
    """
    return _solve(grid, collect=False)


def _make_object_patch() -> np.ndarray:
    for _ in range(OBJECT_SHAPE_RETRIES):
        ph = random.randint(*OBJECT_BOX_RANGE)
        pw = random.randint(*OBJECT_BOX_RANGE)
        max_area = min(OBJECT_AREA_RANGE[1], ph * pw - 1)
        min_area = max(OBJECT_AREA_RANGE[0], ph + pw - 1)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = np.zeros((ph, pw), dtype=bool)
        start = (random.randrange(ph), random.randrange(pw))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier = []
            for r, c in zip(*np.where(mask)):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < ph and 0 <= nc < pw and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            frontier.sort(key=lambda rc: (abs(rc[0] - ph / 2), abs(rc[1] - pw / 2), random.random()))
            pick = frontier[random.randrange(min(len(frontier), max(3, len(frontier) // 2)))]
            mask[pick] = True

        coords = np.argwhere(mask)
        if int(mask.sum()) != area:
            continue
        if coords[:, 0].min() != 0 or coords[:, 0].max() != ph - 1:
            continue
        if coords[:, 1].min() != 0 or coords[:, 1].max() != pw - 1:
            continue
        if np.all(mask):
            continue
        if coords.shape[0] in (ph * pw,):
            continue
        patch = np.zeros((ph, pw), dtype=int)
        patch[mask] = TARGET_COLOR
        return patch
    raise RuntimeError("failed to generate a non-rectangular target patch")


def _central_positions(limit: int, size: int) -> list[int]:
    positions = list(range(4, limit - size - 3))
    center = (limit - size) / 2.0
    good = [pos for pos in positions if abs(pos - center) <= 2.5]
    return good or positions


def _surface_maps(object_cells: set[tuple[int, int]]):
    cols: dict[int, list[int]] = {}
    rows: dict[int, list[int]] = {}
    for r, c in object_cells:
        cols.setdefault(c, []).append(r)
        rows.setdefault(r, []).append(c)

    top = sorted((min(rs), c) for c, rs in cols.items())
    bottom = sorted((max(rs), c) for c, rs in cols.items())
    left = sorted((r, min(cs)) for r, cs in rows.items())
    right = sorted((r, max(cs)) for r, cs in rows.items())
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _marker_for_side(side: str, target: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    tr, tc = target
    if side == "top":
        return (0, tc)
    if side == "bottom":
        return (h - 1, tc)
    if side == "left":
        return (tr, 0)
    if side == "right":
        return (tr, w - 1)
    raise AssertionError("unknown side")


def _pick_target(candidates: list[tuple[int, int]], *, prefer_high_col: bool = False, prefer_high_row: bool = False):
    ordered = list(candidates)
    if prefer_high_col:
        ordered.sort(key=lambda rc: (rc[1], rc[0]), reverse=True)
    elif prefer_high_row:
        ordered.sort(key=lambda rc: (rc[0], rc[1]), reverse=True)
    else:
        random.shuffle(ordered)
    return ordered[0]


def _trigger_band(side: str, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> list[int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    if side == "top":
        return list(range(2, r0 - 1))
    if side == "bottom":
        return list(range(r1 + 2, h - 2))
    if side == "left":
        return list(range(2, c0 - 1))
    if side == "right":
        return list(range(c1 + 2, w - 2))
    raise AssertionError("unknown side")


def _place_general_trigger(spec: dict, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int] | None:
    side = spec["side"]
    target = spec["target"]
    band = _trigger_band(side, bbox, shape)
    random.shuffle(band)
    for value in band:
        if side in ("top", "bottom"):
            return (value, target[1])
        return (target[0], value)
    return None


def _cell_adjacent_to_object(cell: tuple[int, int], object_cells: set[tuple[int, int]]) -> bool:
    for nr, nc in _neighbors4(*cell):
        if (nr, nc) in object_cells:
            return True
    return False


def _projected_target(marker: tuple[int, int], target_mask: np.ndarray) -> tuple[int, int] | None:
    h, w = target_mask.shape
    dr, dc = _projection_direction(marker[0], marker[1], (h, w))
    r, c = marker[0] + dr, marker[1] + dc
    while _in_bounds((h, w), r, c):
        if target_mask[r, c]:
            return (r, c)
        r += dr
        c += dc
    return None


def _validate_input_structure(grid: np.ndarray, specs: list[dict]) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    target = _target_mask(g)
    object_cells = {tuple(map(int, rc)) for rc in np.argwhere(target)}

    marker_cells = []
    for r, c in _clockwise_perimeter((h, w)):
        color = int(g[r, c])
        if color not in (BACKGROUND, TRIGGER_COLOR, TARGET_COLOR):
            marker_cells.append((r, c, color))
    if len(marker_cells) != len(specs):
        return False
    if len({color for _, _, color in marker_cells}) != len(marker_cells):
        return False

    triggers = {tuple(map(int, rc)) for rc in np.argwhere(g == TRIGGER_COLOR)}
    if not (1 <= len(triggers) <= 3):
        return False
    for cell in triggers:
        if cell[0] in (0, h - 1) or cell[1] in (0, w - 1):
            return False
        if cell in object_cells or _cell_adjacent_to_object(cell, object_cells):
            return False

    target_hits = {spec["side"]: spec["target"] for spec in specs}
    for spec in specs:
        marker = spec["marker"]
        target_cell = _projected_target(marker, target)
        if target_cell != target_hits[spec["side"]]:
            return False
        path = _path_from_marker(marker, target_hits[spec["side"]], (h, w))
        path_triggers = [cell for cell in path if cell in triggers]
        if len(path_triggers) > 1:
            return False
        if spec["trigger"] is None and path_triggers:
            return False
        if spec["trigger"] is not None and path_triggers != [spec["trigger"]]:
            return False
    return True


def _validate_semantics(grid: np.ndarray, specs: list[dict], *, want_blocking: bool) -> bool:
    if not _validate_input_structure(grid, specs):
        return False

    solved, details = _solve(grid, collect=True)
    if np.array_equal(grid, solved):
        return False

    saw_trigger = False
    saw_plain = False
    saw_order_block = False

    original = np.asarray(grid, dtype=int)
    for info in details:
        if info["trigger"] is None:
            saw_plain = True
            if len(info["projection_cells"]) < 2:
                return False
        else:
            saw_trigger = True
            if info["projection_overwrites"]:
                return False
            if any(len(ray["cells"]) == 0 for ray in info["rays"]):
                return False
            for ray in info["rays"]:
                if ray["stop_reason"] == "occupied" and ray["stop_cell"] is not None:
                    sr, sc = ray["stop_cell"]
                    if original[sr, sc] == BACKGROUND:
                        saw_order_block = True
        if info["trigger"] is None and info["projection_overwrites"]:
            return False

    if not (saw_trigger and saw_plain):
        return False
    if want_blocking and not saw_order_block:
        return False
    if int(np.sum(solved != grid)) < 8:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)

    object_patch = _make_object_patch()
    ph, pw = object_patch.shape
    possible_r0 = _central_positions(h, ph)
    possible_c0 = _central_positions(w, pw)
    if not possible_r0 or not possible_c0:
        return None
    obj_r0 = random.choice(possible_r0)
    obj_c0 = random.choice(possible_c0)
    obj_r1 = obj_r0 + ph - 1
    obj_c1 = obj_c0 + pw - 1
    bbox = (obj_r0, obj_c0, obj_r1, obj_c1)

    grid = np.zeros((h, w), dtype=int)
    patch_cells = np.argwhere(object_patch == TARGET_COLOR)
    object_cells = {(obj_r0 + int(r), obj_c0 + int(c)) for r, c in patch_cells}
    for r, c in object_cells:
        grid[r, c] = TARGET_COLOR

    surfaces = _surface_maps(object_cells)
    want_blocking = random.random() < 0.55
    marker_count = random.choice(MARKER_COUNT_CHOICES)

    if want_blocking:
        side_pool = ["top", "left", "right", "bottom"]
        sides = ["top", "left"]
        extras = random.sample([side for side in side_pool if side not in sides], marker_count - 2)
        sides.extend(extras)
    else:
        sides = random.sample(["top", "right", "bottom", "left"], marker_count)

    specs: list[dict] = []
    used_marker_cells: set[tuple[int, int]] = set()
    used_targets: set[tuple[int, int]] = set()
    colors = random.sample(list(MARKER_COLORS), marker_count)

    for idx, side in enumerate(sides):
        candidates = surfaces[side]
        if side == "top" and want_blocking:
            target = _pick_target(candidates, prefer_high_col=True)
        elif side == "left" and want_blocking:
            target = _pick_target(candidates, prefer_high_row=True)
        else:
            random.shuffle(candidates)
            target = None
            for cand in candidates:
                marker = _marker_for_side(side, cand, (h, w))
                if marker not in used_marker_cells and cand not in used_targets:
                    target = cand
                    break
            if target is None:
                target = random.choice(candidates)

        marker = _marker_for_side(side, target, (h, w))
        if marker in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
            return None
        used_marker_cells.add(marker)
        used_targets.add(target)
        specs.append(
            {
                "side": side,
                "marker": marker,
                "target": target,
                "color": colors[idx],
                "trigger": None,
            }
        )

    by_side = {spec["side"]: spec for spec in specs}

    triggered_sides: set[str]
    if want_blocking:
        triggered_sides = {"top", "left"}
        remaining = [side for side in sides if side not in triggered_sides]
        if remaining and random.random() < 0.35 and len(triggered_sides) < 3:
            triggered_sides.add(random.choice(remaining))
    else:
        max_triggered = min(3, marker_count - 1)
        count = random.randint(1, max_triggered)
        triggered_sides = set(random.sample(sides, count))

    for side in triggered_sides:
        spec = by_side[side]
        spec["trigger"] = _place_general_trigger(spec, bbox, (h, w))
        if spec["trigger"] is None:
            return None

    if want_blocking:
        top_spec = by_side["top"]
        left_spec = by_side["left"]
        top_rows = [r for r in _trigger_band("top", bbox, (h, w)) if r < left_spec["target"][0]]
        left_cols = [c for c in _trigger_band("left", bbox, (h, w)) if c < top_spec["target"][1]]
        if not top_rows or not left_cols:
            return None
        top_trigger = (random.choice(top_rows), top_spec["target"][1])
        left_trigger = (left_spec["target"][0], random.choice(left_cols))
        if top_trigger[0] >= left_trigger[0]:
            return None
        top_spec["trigger"] = top_trigger
        left_spec["trigger"] = left_trigger

    for spec in specs:
        mr, mc = spec["marker"]
        grid[mr, mc] = spec["color"]
        if spec["trigger"] is not None:
            tr, tc = spec["trigger"]
            if grid[tr, tc] != BACKGROUND:
                return None
            grid[tr, tc] = TRIGGER_COLOR

    if not _validate_semantics(grid, specs, want_blocking=want_blocking):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
