from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 500
BG = 0


@dataclass(frozen=True)
class Wall:
    row: int
    color: int
    door_c0: int
    door_c1: int


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _rect_coords(r0: int, c0: int, h: int, w: int) -> np.ndarray:
    return np.argwhere(np.ones((int(h), int(w)), dtype=bool)) + np.array([int(r0), int(c0)], dtype=int)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _find_walls(grid: np.ndarray, bg: int) -> list[Wall]:
    h, w = grid.shape
    walls: list[Wall] = []
    for r in range(h):
        row = grid[r]
        bg_cols = np.flatnonzero(row == bg)
        if bg_cols.size == 0 or bg_cols.size == w:
            continue
        if int(bg_cols[0]) == 0 or int(bg_cols[-1]) == w - 1:
            continue
        if not np.array_equal(bg_cols, np.arange(int(bg_cols[0]), int(bg_cols[-1]) + 1)):
            continue
        non_bg_vals = sorted(set(int(v) for v in row if int(v) != bg))
        if len(non_bg_vals) != 1:
            continue
        walls.append(Wall(r, non_bg_vals[0], int(bg_cols[0]), int(bg_cols[-1])))
    assert 1 <= len(walls) <= 2, "expected one or two horizontal walls"
    return sorted(walls, key=lambda wall: wall.row)


def _extract_objects(grid: np.ndarray, walls: list[Wall], bg: int) -> tuple[np.ndarray, list[dict]]:
    work = np.array(grid, dtype=int, copy=True)
    for wall in walls:
        work[wall.row, :] = bg
    comps = _components_by_color(work, bg)
    base = np.array(grid, dtype=int, copy=True)
    objects: list[dict] = []
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = comp["bbox"]
        base[comp["coords"][:, 0], comp["coords"][:, 1]] = bg
        objects.append(
            {
                "name": f"obj_{idx}",
                "color": int(comp["color"]),
                "coords": comp["coords"].copy(),
                "bbox": (r0, c0, r1, c1),
                "bottom": int(r1),
                "left": int(c0),
            }
        )
    return base, objects


def _simulate(base_grid: np.ndarray, objects: list[dict], bg: int) -> tuple[np.ndarray, list[dict]]:
    h, w = base_grid.shape
    out = np.array(base_grid, dtype=int, copy=True)
    occupied = out != bg
    order = sorted(objects, key=lambda obj: (-int(obj["bottom"]), int(obj["left"]), int(obj["color"])))
    traces: list[dict] = []

    for obj in order:
        coords = np.array(obj["coords"], dtype=int, copy=True)
        while True:
            candidate = coords + np.array([1, 0], dtype=int)
            if np.any(candidate[:, 0] >= h):
                break
            if np.any(occupied[candidate[:, 0], candidate[:, 1]]):
                break
            coords = candidate

        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
        occupied[coords[:, 0], coords[:, 1]] = True
        traces.append(
            {
                "name": obj["name"],
                "color": int(obj["color"]),
                "initial_bbox": tuple(map(int, obj["bbox"])),
                "final_bbox": _bbox(coords),
                "final_coords": coords,
            }
        )

    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the floating objects through the wall doors until earlier landings progressively clog later passages.

    Input structure:
    - Black background with one or two full-width horizontal wall rows.
    - Each wall has exactly one contiguous door segment where the wall row is background.
    - Several detached monochrome objects float above the top wall and do not touch each other.

    Rule:
    - Remove the floating objects from their starting cells, then process them from lower to higher starting bottom row.
    - Each object falls straight down as a rigid shape until moving one more row would overlap a wall segment, a previously settled object, or the grid bottom.
    - Door cells count as empty wall cells, so only objects whose full column span lies inside a door can pass through that wall.
    - Earlier objects can stack inside the cavity under a door and eventually occupy the door itself, after which later objects stop above that clog instead of passing through.

    Key insight:
    - The doors are conditional channels: whether a later object can traverse them depends on both its own horizontal span and the exact landing history of lower objects settled earlier.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    walls = _find_walls(g, bg)
    base, objects = _extract_objects(g, walls, bg)
    out, _traces = _simulate(base, objects, bg)
    return out


def _paint_wall(grid: np.ndarray, wall: Wall, bg: int) -> None:
    grid[wall.row, :] = int(wall.color)
    grid[wall.row, wall.door_c0 : wall.door_c1 + 1] = int(bg)


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = int(color)


def _span_fits(span_c0: int, span_c1: int, wall: Wall) -> bool:
    return int(wall.door_c0) <= int(span_c0) and int(span_c1) <= int(wall.door_c1)


def _object_trace_map(traces: list[dict]) -> dict[str, dict]:
    return {trace["name"]: trace for trace in traces}


def _late_fit_changes_due_to_clog(
    base_grid: np.ndarray,
    objects: list[dict],
    late_name: str,
    bg: int,
    top_wall: Wall,
) -> bool:
    late_obj = next(obj for obj in objects if obj["name"] == late_name)
    actual_out, actual_traces = _simulate(base_grid, objects, bg)
    del actual_out
    actual = _object_trace_map(actual_traces)[late_name]

    without_fillers = [obj for obj in objects if not obj["name"].startswith("fill_")]
    counter_out, counter_traces = _simulate(base_grid, without_fillers, bg)
    del counter_out
    counter = _object_trace_map(counter_traces)[late_name]

    actual_r0, _ac0, actual_r1, _ac1 = actual["final_bbox"]
    counter_r0, _cc0, counter_r1, _cc1 = counter["final_bbox"]
    late_r0, late_c0, late_r1, late_c1 = late_obj["bbox"]
    del late_r0, late_r1

    assert _span_fits(late_c0, late_c1, top_wall)
    return actual_r1 < top_wall.row and counter_r0 > top_wall.row


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    base_grid: np.ndarray,
    walls: list[Wall],
    objects: list[dict],
    traces: list[dict],
    wall_count: int,
) -> bool:
    if np.array_equal(inp, out):
        return False
    if not np.array_equal(main(inp), out):
        return False

    trace_map = _object_trace_map(traces)
    top_wall = walls[0]

    if len({int(obj["bottom"]) for obj in objects}) != len(objects):
        return False

    late = trace_map["late_fit"]
    if late["final_bbox"][2] >= top_wall.row:
        return False
    if not _late_fit_changes_due_to_clog(base_grid, objects, "late_fit", BG, top_wall):
        return False

    wide = trace_map["wide_blocker"]
    if wide["final_bbox"][2] != top_wall.row - 1:
        return False

    passed_top = 0
    for obj in objects:
        tr = trace_map[obj["name"]]
        if tr["final_bbox"][0] > top_wall.row:
            passed_top += 1
    if passed_top == 0:
        return False

    if wall_count == 2:
        lower_wall = walls[1]
        deep = trace_map["deep_pass"]
        if deep["final_bbox"][0] <= lower_wall.row:
            return False
        fill_traces = [trace_map[obj["name"]] for obj in objects if obj["name"].startswith("fill_")]
        if not fill_traces:
            return False
        if max(trace["final_bbox"][2] for trace in fill_traces) < top_wall.row:
            return False
    else:
        if max(trace_map[obj["name"]]["final_bbox"][2] for obj in objects if obj["name"].startswith("fill_")) < top_wall.row:
            return False

    return True


def _sample_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Wall], list[dict], list[dict]]:
    wall_count = random.choice([1, 2])
    support_gap = random.randint(2, 3)
    fill_count = support_gap + 1

    has_deep = wall_count == 2
    roles = []
    if has_deep:
        roles.append("deep_pass")
    roles.extend(f"fill_{i}" for i in range(fill_count))
    roles.append("wide_blocker")
    roles.append("late_fit")

    n_objects = len(roles)
    top_wall_row = 2 * n_objects + random.randint(0, 2)
    max_input_row = 2 * (n_objects - 1)
    row_shift = random.randint(0, max(0, top_wall_row - 2 - max_input_row))
    input_rows = {role: row_shift + 2 * (n_objects - 1 - idx) for idx, role in enumerate(roles)}

    if wall_count == 1:
        h = top_wall_row + support_gap + 1
    else:
        lower_wall_row = top_wall_row + support_gap + 1
        h = lower_wall_row + random.randint(3, 5) + 1
    assert h <= 30

    clog_w = random.randint(2, 3) if wall_count == 2 else random.randint(1, 3)
    top_door_w = clog_w + random.randint(1, 2)

    w = random.randint(max(12, top_door_w + 7), 20)
    top_door_c0 = random.randint(2, w - top_door_w - 3)
    top_wall_color, lower_wall_color = random.sample(range(1, 10), 2)
    walls = [Wall(top_wall_row, top_wall_color, top_door_c0, top_door_c0 + top_door_w - 1)]

    if wall_count == 2:
        lower_door_w = random.randint(1, clog_w - 1)
        lower_door_c0 = random.randint(walls[0].door_c0, walls[0].door_c1 - lower_door_w + 1)
        walls.append(Wall(lower_wall_row, lower_wall_color, lower_door_c0, lower_door_c0 + lower_door_w - 1))

    clog_c0 = random.randint(walls[0].door_c0, walls[0].door_c1 - clog_w + 1)
    blocker_w = random.randint(2, min(4, w - 4))
    blocker_candidates = [
        c0
        for c0 in range(1, w - blocker_w - 1)
        if not _span_fits(c0, c0 + blocker_w - 1, walls[0])
    ]
    assert blocker_candidates
    blocker_c0 = random.choice(blocker_candidates)

    deep_w = 1
    if wall_count == 2:
        deep_wall = walls[1]
        deep_c0 = random.randint(deep_wall.door_c0, deep_wall.door_c1 - deep_w + 1)
    else:
        deep_c0 = -1

    color_pool = [c for c in range(1, 10) if c not in {wall.color for wall in walls}]
    random.shuffle(color_pool)
    assert len(color_pool) >= n_objects

    objects: list[dict] = []
    for idx, role in enumerate(roles):
        if role.startswith("fill_") or role == "late_fit":
            r0 = input_rows[role]
            c0 = clog_c0
            h0 = 1
            w0 = clog_w
        elif role == "wide_blocker":
            r0 = input_rows[role]
            c0 = blocker_c0
            h0 = 1
            w0 = blocker_w
        else:
            r0 = input_rows[role]
            c0 = deep_c0
            h0 = 1
            w0 = deep_w

        coords = _rect_coords(r0, c0, h0, w0)
        objects.append(
            {
                "name": role,
                "color": color_pool[idx],
                "coords": coords,
                "bbox": (r0, c0, r0 + h0 - 1, c0 + w0 - 1),
                "bottom": r0 + h0 - 1,
                "left": c0,
            }
        )

    inp = np.full((h, w), BG, dtype=int)
    for wall in walls:
        _paint_wall(inp, wall, BG)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        if np.any(inp[r0 : r1 + 1, c0 : c1 + 1] != BG):
            raise AssertionError("generated object overlaps an existing structure")
        _paint_rect(inp, r0, c0, r1 - r0 + 1, c1 - c0 + 1, int(obj["color"]))

    base = np.full((h, w), BG, dtype=int)
    for wall in walls:
        _paint_wall(base, wall, BG)

    out, traces = _simulate(base, objects, BG)
    return inp, out, base, walls, objects, traces


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            inp, out, base, walls, objects, traces = _sample_case()
        except AssertionError:
            continue
        if _semantic_checks(inp, out, base, walls, objects, traces, len(walls)):
            return inp
    raise RuntimeError("failed to generate a valid screening-chain sample")
