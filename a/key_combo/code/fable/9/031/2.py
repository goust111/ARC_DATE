from __future__ import annotations

from collections import Counter
import random
from typing import Iterable

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNER_NAMES = ("TL", "TR", "BL", "BR")
CORNER_DIRS = {
    "TL": (-1, -1),
    "TR": (-1, 1),
    "BL": (1, -1),
    "BR": (1, 1),
}

GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
MAX_OBJECTS = 7
MIN_OBJECTS = 5
MAX_GENERATION_ATTEMPTS = 500
MAX_PLACEMENT_TRIES = 250
MAX_REVERSE_STEPS = 4
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _shift_cells(cells: Iterable[tuple[int, int]], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple((int(r + dr), int(c + dc)) for r, c in cells)


def _paint(grid: np.ndarray, cells: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _normalize_cells(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in cells]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = [(int(r), int(c)) for r, c in cells]
    return min(r for r, _ in pts), min(c for _, c in pts), max(r for r, _ in pts), max(c for _, c in pts)


def _component_cells(grid: np.ndarray, bg: int, blocked: set[tuple[int, int]]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg or (r, c) in blocked:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or (nr, nc) in blocked or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            rel = _normalize_cells(pts)
            objects.append(
                {
                    "color": color,
                    "cells": tuple(sorted(pts)),
                    "rel_cells": rel,
                    "bbox": _bbox(pts),
                }
            )
    return objects


def _legend_info(grid: np.ndarray, bg: int) -> dict:
    h, w = grid.shape
    corners = {
        "TL": ((0, 0), slice(0, 2), slice(0, 2)),
        "TR": ((0, w - 2), slice(0, 2), slice(w - 2, w)),
        "BL": ((h - 2, 0), slice(h - 2, h), slice(0, 2)),
        "BR": ((h - 2, w - 2), slice(h - 2, h), slice(w - 2, w)),
    }
    candidates: list[dict] = []
    for name, (_origin, rs, cs) in corners.items():
        if rs.start < 0 or cs.start < 0:
            continue
        block = grid[rs, cs]
        colors = [int(v) for v in block.ravel()]
        if bg in colors or len(set(colors)) != 4:
            continue
        cells = {(r, c) for r in range(rs.start, rs.stop) for c in range(cs.start, cs.stop)}
        ok = True
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not _in_bounds(grid.shape, nr, nc) or (nr, nc) in cells:
                    continue
                if int(grid[nr, nc]) != bg:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        color_to_corner = {
            int(block[0, 0]): "TL",
            int(block[0, 1]): "TR",
            int(block[1, 0]): "BL",
            int(block[1, 1]): "BR",
        }
        candidates.append(
            {
                "legend_corner": name,
                "legend_cells": cells,
                "block": block.copy(),
                "color_to_corner": color_to_corner,
            }
        )
    assert len(candidates) == 1, "expected exactly one isolated 2x2 corner legend"
    return candidates[0]


def _target_corner(shape: tuple[int, int], corner_name: str) -> tuple[int, int]:
    h, w = shape
    if corner_name == "TL":
        return 0, 0
    if corner_name == "TR":
        return 0, w - 1
    if corner_name == "BL":
        return h - 1, 0
    return h - 1, w - 1


def _chebyshev_distance(cells: Iterable[tuple[int, int]], target: tuple[int, int]) -> int:
    tr, tc = target
    return min(max(abs(int(r) - tr), abs(int(c) - tc)) for r, c in cells)


def _classify_stop(
    cells: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    legend_cells: set[tuple[int, int]],
) -> tuple[bool, str | None]:
    moved = _shift_cells(cells, dr, dc)
    for r, c in moved:
        if not _in_bounds(shape, r, c):
            return True, "boundary"
    overlap = set(moved) & occupied
    if overlap:
        return True, "legend" if overlap <= legend_cells else "object"
    return False, None


def _simulate(grid: np.ndarray, return_details: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    legend = _legend_info(g, bg)
    objects = _component_cells(g, bg, legend["legend_cells"])
    assert objects, "expected at least one movable object"

    h, w = g.shape
    ordering = []
    for obj in objects:
        target_name = legend["color_to_corner"].get(int(obj["color"]))
        assert target_name is not None, "object color must appear in the legend"
        target = _target_corner((h, w), target_name)
        dist = _chebyshev_distance(obj["cells"], target)
        r0, c0, r1, c1 = obj["bbox"]
        ordering.append(
            (
                dist,
                CORNER_NAMES.index(target_name),
                r0,
                c0,
                r1,
                c1,
                obj["color"],
                obj,
            )
        )
    ordering.sort(key=lambda item: item[:-1])

    out = np.full_like(g, bg)
    legend_rows, legend_cols = zip(*legend["legend_cells"])
    out[legend_rows, legend_cols] = g[legend_rows, legend_cols]
    occupied = set(legend["legend_cells"])
    details: list[dict] = []

    for dist, _corner_idx, _r0, _c0, _r1, _c1, _color, obj in ordering:
        target_name = legend["color_to_corner"][int(obj["color"])]
        dr, dc = CORNER_DIRS[target_name]
        pos = obj["cells"]
        steps = 0
        while True:
            blocked, reason = _classify_stop(pos, dr, dc, g.shape, occupied, legend["legend_cells"])
            if blocked:
                stop_reason = reason
                break
            pos = _shift_cells(pos, dr, dc)
            steps += 1

        _paint(out, pos, obj["color"])
        occupied |= set(pos)
        details.append(
            {
                "color": int(obj["color"]),
                "target": target_name,
                "direction": (dr, dc),
                "start_cells": obj["cells"],
                "end_cells": pos,
                "distance": int(dist),
                "steps": int(steps),
                "stop_reason": stop_reason,
            }
        )

    if return_details:
        return out, {
            "bg": bg,
            "legend_cells": set(legend["legend_cells"]),
            "legend_corner": legend["legend_corner"],
            "color_to_corner": dict(legend["color_to_corner"]),
            "objects": details,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide every colored object diagonally toward the corner encoded by its color's position in the 2x2 legend.

    Input structure:
    - One grid corner contains an isolated 2x2 legend whose four cells have four different colors.
    - The rest of the grid contains several monochromatic 4-connected objects, and every object color appears in the legend.
    - The legend is separated from the movable objects by background cells.

    Rule:
    - The legend's top-left, top-right, bottom-left, and bottom-right colors map to the grid's top-left, top-right, bottom-left, and bottom-right corners.
    - Each object keeps its shape and color and repeatedly shifts by one diagonal cell toward its mapped corner.
    - An object stops when the next diagonal step would leave the grid or overlap the fixed legend or an earlier-settled object.
    - Objects closer to their own target corner by Chebyshev distance move first, so later objects stack outside earlier ones.

    Key insight:
    - The legend is a positional codebook: a color's place inside the 2x2 block determines the motion vector, and collision order determines the final stack.
    """
    return _simulate(grid, return_details=False)


def _random_polyomino() -> tuple[tuple[int, int], ...]:
    for _ in range(80):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        target_size = random.randint(2, min(5, h * w - 1))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = [start]
        while len(cells) < target_size and frontier:
            cr, cc = random.choice(frontier)
            nbrs = []
            for dr, dc in DIRS4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.remove((cr, cc))
                continue
            pick = random.choice(nbrs)
            cells.add(pick)
            frontier.append(pick)
        rel = _normalize_cells(cells)
        rr0, cc0, rr1, cc1 = _bbox(rel)
        if len(rel) >= 2 and (rr1 - rr0 + 1 >= 2 or cc1 - cc0 + 1 >= 2):
            return rel
    raise RuntimeError("failed to build a polyomino")


def _cells_from_rel(rel_cells: tuple[tuple[int, int], ...], top: int, left: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((top + r, left + c) for r, c in rel_cells))


def _overlaps(cells: Iterable[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    return any((r, c) in occupied for r, c in cells)


def _sample_anchor(h: int, w: int, rel_cells: tuple[tuple[int, int], ...], target: str) -> tuple[int, int]:
    rr0, cc0, rr1, cc1 = _bbox(rel_cells)
    obj_h = rr1 - rr0 + 1
    obj_w = cc1 - cc0 + 1
    mid_r = max(0, h // 2)
    mid_c = max(0, w // 2)

    if target == "TL":
        row_lo, row_hi = 0, max(0, min(h - obj_h, mid_r))
        col_lo, col_hi = 0, max(0, min(w - obj_w, mid_c))
    elif target == "TR":
        row_lo, row_hi = 0, max(0, min(h - obj_h, mid_r))
        col_lo, col_hi = max(0, w - obj_w - mid_c), w - obj_w
    elif target == "BL":
        row_lo, row_hi = max(0, h - obj_h - mid_r), h - obj_h
        col_lo, col_hi = 0, max(0, min(w - obj_w, mid_c))
    else:
        row_lo, row_hi = max(0, h - obj_h - mid_r), h - obj_h
        col_lo, col_hi = max(0, w - obj_w - mid_c), w - obj_w

    if row_lo > row_hi:
        row_lo, row_hi = 0, h - obj_h
    if col_lo > col_hi:
        col_lo, col_hi = 0, w - obj_w
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _one_step_reverse_clear(
    cells: tuple[tuple[int, int], ...],
    direction: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    dr, dc = direction
    shifted = _shift_cells(cells, -dr, -dc)
    for r, c in shifted:
        if not _in_bounds(shape, r, c):
            return False
    return not _overlaps(shifted, occupied)


def _place_final_layout() -> tuple[np.ndarray, list[dict], dict] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = 0
    grid = np.full((h, w), bg, dtype=int)

    legend_corner = random.choice(CORNER_NAMES)
    legend_colors = random.sample(PALETTE, 4)
    legend_block = np.array(legend_colors, dtype=int).reshape(2, 2)

    if legend_corner == "TL":
        rs, cs = slice(0, 2), slice(0, 2)
    elif legend_corner == "TR":
        rs, cs = slice(0, 2), slice(w - 2, w)
    elif legend_corner == "BL":
        rs, cs = slice(h - 2, h), slice(0, 2)
    else:
        rs, cs = slice(h - 2, h), slice(w - 2, w)
    grid[rs, cs] = legend_block
    legend_cells = {(r, c) for r in range(rs.start, rs.stop) for c in range(cs.start, cs.stop)}

    color_to_target = {
        int(legend_block[0, 0]): "TL",
        int(legend_block[0, 1]): "TR",
        int(legend_block[1, 0]): "BL",
        int(legend_block[1, 1]): "BR",
    }
    target_to_color = {v: k for k, v in color_to_target.items()}

    chain_target = random.choice(CORNER_NAMES)
    target_counts = {name: 1 for name in CORNER_NAMES}
    target_counts[chain_target] += 1
    extras = random.randint(0, 2)
    for _ in range(extras):
        target_counts[random.choice(CORNER_NAMES)] += 1
    total_objects = sum(target_counts.values())
    if not (MIN_OBJECTS <= total_objects <= MAX_OBJECTS):
        return None

    occupied = set(legend_cells)
    used_distances: set[int] = set()
    objects: list[dict] = []
    placement_plan: list[tuple[str, bool]] = []
    for name in CORNER_NAMES:
        placement_plan.append((name, False))
    placement_plan.append((chain_target, True))
    for name in CORNER_NAMES:
        surplus = target_counts[name] - 1 - (1 if name == chain_target else 0)
        for _ in range(max(0, surplus)):
            placement_plan.append((name, False))
    random.shuffle(placement_plan[5:])

    for target_name, need_object_stop in placement_plan:
        color = target_to_color[target_name]
        placed = None
        for _ in range(MAX_PLACEMENT_TRIES):
            rel = _random_polyomino()
            top, left = _sample_anchor(h, w, rel, target_name)
            cells = _cells_from_rel(rel, top, left)
            if _overlaps(cells, occupied):
                continue
            dist = _chebyshev_distance(cells, _target_corner((h, w), target_name))
            if dist in used_distances:
                continue
            blocked, stop_reason = _classify_stop(cells, *CORNER_DIRS[target_name], (h, w), occupied, legend_cells)
            if not blocked:
                continue
            if need_object_stop and stop_reason != "object":
                continue
            if not _one_step_reverse_clear(cells, CORNER_DIRS[target_name], (h, w), occupied):
                continue
            placed = {
                "color": color,
                "target": target_name,
                "rel_cells": rel,
                "final_cells": cells,
                "distance": dist,
                "stop_reason": stop_reason,
            }
            break
        if placed is None:
            return None
        objects.append(placed)
        occupied |= set(placed["final_cells"])
        used_distances.add(int(placed["distance"]))

    for obj in objects:
        _paint(grid, obj["final_cells"], obj["color"])
    meta = {
        "legend_cells": legend_cells,
        "legend_corner": legend_corner,
        "color_to_target": color_to_target,
        "bg": bg,
    }
    return grid, objects, meta


def _path_clear_for_reverse(
    final_cells: tuple[tuple[int, int], ...],
    direction: tuple[int, int],
    steps: int,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    dr, dc = direction
    start = _shift_cells(final_cells, -dr * steps, -dc * steps)
    for r, c in start:
        if not _in_bounds(shape, r, c):
            return None
    if _overlaps(start, occupied):
        return None

    pos = start
    for _ in range(steps):
        nxt = _shift_cells(pos, dr, dc)
        for r, c in nxt:
            if not _in_bounds(shape, r, c):
                return None
        if _overlaps(nxt, occupied):
            return None
        pos = nxt
    if tuple(sorted(pos)) != tuple(sorted(final_cells)):
        return None
    return start


def _build_input_from_output(output_grid: np.ndarray, objects: list[dict], meta: dict) -> np.ndarray | None:
    h, w = output_grid.shape
    bg = meta["bg"]
    inp = np.full((h, w), bg, dtype=int)
    legend_block = output_grid.copy()
    for r, c in meta["legend_cells"]:
        inp[r, c] = int(legend_block[r, c])

    occupied = set(meta["legend_cells"])
    reverse_steps: dict[int, int] = {}
    start_cells_by_index: dict[int, tuple[tuple[int, int], ...]] = {}
    ordered = sorted(enumerate(objects), key=lambda item: item[1]["distance"], reverse=True)
    for idx, obj in ordered:
        occupied |= set()
        candidates: list[tuple[int, tuple[tuple[int, int], ...]]] = []
        direction = CORNER_DIRS[obj["target"]]
        final_cells = obj["final_cells"]
        current_occupied = set(meta["legend_cells"])
        for j, other in enumerate(objects):
            if j == idx:
                continue
            if j in start_cells_by_index:
                current_occupied |= set(start_cells_by_index[j])
            else:
                current_occupied |= set(other["final_cells"])
        current_occupied -= set(final_cells)
        for steps in range(MAX_REVERSE_STEPS, 0, -1):
            start = _path_clear_for_reverse(final_cells, direction, steps, (h, w), current_occupied)
            if start is None:
                continue
            if any(any((nr, nc) in meta["legend_cells"] for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))) for r, c in start):
                continue
            candidates.append((steps, start))
        if not candidates:
            return None
        steps, start = random.choice(candidates)
        reverse_steps[idx] = steps
        start_cells_by_index[idx] = start

    for idx, obj in enumerate(objects):
        _paint(inp, start_cells_by_index[idx], obj["color"])
    return inp


def _legend_is_isolated(grid: np.ndarray, legend_cells: set[tuple[int, int]], bg: int) -> bool:
    for r, c in legend_cells:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc) or (nr, nc) in legend_cells:
                continue
            if int(grid[nr, nc]) != bg:
                return False
    return True


def _validate_instance(inp: np.ndarray, expected_out: np.ndarray) -> bool:
    try:
        out, info = _simulate(inp, return_details=True)
    except AssertionError:
        return False
    if not np.array_equal(out, expected_out):
        return False

    bg = info["bg"]
    if not _legend_is_isolated(inp, info["legend_cells"], bg):
        return False

    objects = info["objects"]
    if not (MIN_OBJECTS <= len(objects) <= MAX_OBJECTS):
        return False
    if len({obj["target"] for obj in objects}) != 4:
        return False
    if len(set(obj["distance"] for obj in objects)) != len(objects):
        return False
    if any(obj["steps"] < 1 for obj in objects):
        return False
    if max(obj["steps"] for obj in objects) < 2:
        return False
    if sum(obj["stop_reason"] == "object" for obj in objects) < 1:
        return False
    if sum(obj["stop_reason"] in {"boundary", "legend"} for obj in objects) < 2:
        return False
    color_counts = Counter(obj["color"] for obj in objects)
    if max(color_counts.values()) < 2:
        return False
    if np.array_equal(inp, expected_out):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    built = _place_final_layout()
    if built is None:
        return None
    out_grid, objects, meta = built
    inp = _build_input_from_output(out_grid, objects, meta)
    if inp is None:
        return None
    if not _validate_instance(inp, out_grid):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-slide legend puzzle")
