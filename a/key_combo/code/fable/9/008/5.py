from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


BLACK = 0
GRAY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

SIDE_TO_DIR = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
TARGET_SIDE_TO_VIEW_DIR = {
    "left": (0, 1),
    "right": (0, -1),
    "top": (1, 0),
    "bottom": (-1, 0),
}
TARGET_SIDE_TO_VIEWER_OPEN_SIDE = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
}

GEN_ATTEMPTS = 400
CUP_PLACEMENT_ATTEMPTS = 350
GRID_MIN = 20
GRID_MAX = 28
CUP_MIN = 4
CUP_MAX = 7
MIN_RAY_DISTANCE = 3
MAX_RAY_DISTANCE = 6


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != color or (r, c) in seen:
                continue
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            seen.add((r, c))
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if int(grid[nr, nc]) != color or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    queue.append((nr, nc))
            components.append(sorted(comp))
    components.sort(key=lambda comp: _bbox_from_coords(comp))
    return components


def _boundary_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_corner(coord: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = coord
    r0, c0, r1, c1 = bbox
    return (r, c) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _opening_side(opening: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = opening
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise ValueError("opening must lie on the cup boundary")


def _opening_abs(top: int, left: int, h: int, w: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return top, left + offset
    if side == "bottom":
        return top + h - 1, left + offset
    if side == "left":
        return top + offset, left
    if side == "right":
        return top + offset, left + w - 1
    raise ValueError(f"unknown side: {side}")


def _cup_patch(h: int, w: int, side: str, offset: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = GRAY
    patch[-1, :] = GRAY
    patch[:, 0] = GRAY
    patch[:, -1] = GRAY
    rr, cc = _opening_abs(0, 0, h, w, side, offset)
    patch[rr, cc] = BLACK
    return patch


def _cup_bbox_cells(top: int, left: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + h) for c in range(left, left + w)}


def _expanded_bbox_cells(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return {
        (r, c)
        for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1)
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1)
    }


def _wall_cells_for_side(cup: dict[str, Any], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = cup["bbox"]
    if side == "left":
        cells = [(r, c0) for r in range(r0 + 1, r1)]
    elif side == "right":
        cells = [(r, c1) for r in range(r0 + 1, r1)]
    elif side == "top":
        cells = [(r0, c) for c in range(c0 + 1, c1)]
    elif side == "bottom":
        cells = [(r1, c) for c in range(c0 + 1, c1)]
    else:
        raise ValueError(f"unknown side: {side}")
    return [cell for cell in cells if cell != cup["opening"]]


def _analyze_cups(grid: np.ndarray) -> tuple[list[dict[str, Any]], dict[tuple[int, int], int]]:
    cups: list[dict[str, Any]] = []
    wall_to_cup: dict[tuple[int, int], int] = {}
    for comp in _components_of_color(grid, GRAY):
        bbox = _bbox_from_coords(comp)
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            raise ValueError("cup bbox is too small")
        wall = set(comp)
        expected_wall = _boundary_cells(bbox)
        if not wall.issubset(expected_wall):
            raise ValueError("gray component is not a rectangular cup wall")
        missing = expected_wall - wall
        if len(missing) != 1:
            raise ValueError("each cup must have exactly one opening")
        opening = next(iter(missing))
        if _is_corner(opening, bbox):
            raise ValueError("opening cannot be at a corner")
        side = _opening_side(opening, bbox)
        direction = SIDE_TO_DIR[side]
        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        if not interior:
            raise ValueError("cup interior must be non-empty")
        if any(int(grid[r, c]) != BLACK for r, c in interior):
            raise ValueError("cup interior must be empty in the input")
        cup = {
            "bbox": bbox,
            "opening": opening,
            "opening_side": side,
            "direction": direction,
            "interior": interior,
            "wall": wall,
        }
        idx = len(cups)
        cups.append(cup)
        for cell in wall:
            wall_to_cup[cell] = idx
    if not cups:
        raise ValueError("input must contain at least one cup")
    return cups, wall_to_cup


def _trace_first_nonblack(
    grid: np.ndarray, opening: tuple[int, int], direction: tuple[int, int]
) -> tuple[tuple[int, int] | None, int | None]:
    dr, dc = direction
    r, c = opening[0] + dr, opening[1] + dc
    while _in_bounds(grid.shape, r, c):
        value = int(grid[r, c])
        if value != BLACK:
            return (r, c), value
        r += dr
        c += dc
    return None, None


def _solve_details(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    cups, wall_to_cup = _analyze_cups(g)
    fill_by_idx: dict[int, int | None] = {}
    hit_by_idx: dict[int, dict[str, Any]] = {}
    visiting: set[int] = set()
    consumed_markers: set[tuple[int, int]] = set()

    def resolve(idx: int) -> int | None:
        if idx in fill_by_idx:
            return fill_by_idx[idx]
        if idx in visiting:
            raise ValueError("cup dependency graph must be acyclic")
        visiting.add(idx)
        cup = cups[idx]
        coord, value = _trace_first_nonblack(g, cup["opening"], cup["direction"])
        if coord is None:
            color = None
            hit = {"kind": "void"}
        elif value == GRAY:
            target_idx = wall_to_cup[coord]
            color = resolve(target_idx)
            hit = {
                "kind": "cup",
                "coord": coord,
                "target_bbox": cups[target_idx]["bbox"],
            }
        else:
            color = value
            consumed_markers.add(coord)
            hit = {"kind": "marker", "coord": coord, "color": value}
        visiting.remove(idx)
        fill_by_idx[idx] = color
        hit_by_idx[idx] = hit
        return color

    for idx in range(len(cups)):
        resolve(idx)

    out = g.copy()
    for r, c in consumed_markers:
        out[r, c] = BLACK
    for idx, cup in enumerate(cups):
        color = fill_by_idx[idx]
        if color is None:
            continue
        for r, c in cup["interior"]:
            out[r, c] = color

    colors_by_bbox = {cup["bbox"]: fill_by_idx[idx] for idx, cup in enumerate(cups)}
    hits_by_bbox = {cup["bbox"]: hit_by_idx[idx] for idx, cup in enumerate(cups)}
    return {
        "output": out,
        "cups": cups,
        "colors_by_bbox": colors_by_bbox,
        "hits_by_bbox": hits_by_bbox,
        "consumed_markers": consumed_markers,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each cup from the first object seen through its opening, following inheritance chains.

    Input structure:
    - A black background contains several gray hollow rectangular cups, each with exactly one boundary opening.
    - Colored single-pixel markers are scattered outside the cups.
    - A cup's opening points along a cardinal sightline that may first meet a marker, another cup wall, or nothing before leaving the grid.

    Rule:
    - For each cup, start from its opening cell and look outward along the opening's normal direction until the first non-black cell.
    - If that first seen cell is a marker, fill the cup interior with the marker's color and erase that marker.
    - If that first seen cell is another cup wall, fill the current cup with that other cup's final fill color, recursively resolving upstream cups first.
    - If nothing non-black is seen before the ray exits the grid, leave the cup interior empty; markers not seen by any cup stay unchanged.

    Key insight:
    - The opening is a line of sight rather than a pouring outlet, so colors propagate through an acyclic "who sees whom" dependency chain before any downstream cup can be determined.
    """
    return _solve_details(grid)["output"]


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _free_prefix_from(
    opening: tuple[int, int],
    direction: tuple[int, int],
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    r, c = opening[0] + dr, opening[1] + dc
    while _in_bounds(shape, r, c):
        if (r, c) in blocked:
            return cells, False
        cells.append((r, c))
        r += dr
        c += dc
    return cells, True


def _cup_bbox_is_clear(
    top: int,
    left: int,
    h: int,
    w: int,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    occupied_nonblack: set[tuple[int, int]],
) -> bool:
    grid_h, grid_w = shape
    if top < 0 or left < 0 or top + h > grid_h or left + w > grid_w:
        return False
    bbox = (top, left, top + h - 1, left + w - 1)
    bbox_cells = _cup_bbox_cells(top, left, h, w)
    if any(cell in blocked for cell in bbox_cells):
        return False
    expanded = _expanded_bbox_cells(bbox, shape, pad=1)
    if any(cell in occupied_nonblack for cell in expanded):
        return False
    return True


def _stamp_cup(grid: np.ndarray, cup: dict[str, Any]) -> None:
    top, left = cup["top"], cup["left"]
    patch = _cup_patch(cup["h"], cup["w"], cup["opening_side"], cup["opening_offset"])
    grid[top : top + cup["h"], left : left + cup["w"]] = patch


def _make_cup_plan(
    top: int,
    left: int,
    h: int,
    w: int,
    opening_side: str,
    opening_offset: int,
    source_kind: str,
) -> dict[str, Any]:
    opening = _opening_abs(top, left, h, w, opening_side, opening_offset)
    bbox = (top, left, top + h - 1, left + w - 1)
    wall_patch = _cup_patch(h, w, opening_side, opening_offset)
    wall_cells = {
        (top + r, left + c)
        for r, c in zip(*np.where(wall_patch == GRAY))
    }
    return {
        "top": top,
        "left": left,
        "h": h,
        "w": w,
        "bbox": bbox,
        "opening": opening,
        "opening_side": opening_side,
        "opening_offset": opening_offset,
        "direction": SIDE_TO_DIR[opening_side],
        "source_kind": source_kind,
        "wall_cells": wall_cells,
        "used_target_cells": set(),
    }


def _place_root_cup(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    occupied_nonblack: set[tuple[int, int]],
    source_kind: str,
    marker_color: int | None = None,
) -> dict[str, Any] | None:
    h_grid, w_grid = grid.shape
    for _ in range(CUP_PLACEMENT_ATTEMPTS):
        h = _randint(rng, CUP_MIN, CUP_MAX)
        w = _randint(rng, CUP_MIN, CUP_MAX)
        opening_side = str(rng.choice(["top", "right", "bottom", "left"]))
        opening_offset = _randint(rng, 1, (w - 2) if opening_side in {"top", "bottom"} else (h - 2))
        top = _randint(rng, 0, h_grid - h)
        left = _randint(rng, 0, w_grid - w)
        if not _cup_bbox_is_clear(top, left, h, w, grid.shape, blocked, occupied_nonblack):
            continue
        cup = _make_cup_plan(top, left, h, w, opening_side, opening_offset, source_kind)
        prefix, clear_to_border = _free_prefix_from(cup["opening"], cup["direction"], blocked, grid.shape)
        if source_kind == "marker":
            if marker_color is None:
                raise ValueError("marker root requires a marker color")
            if len(prefix) < MIN_RAY_DISTANCE:
                continue
            distance = _randint(rng, MIN_RAY_DISTANCE, min(MAX_RAY_DISTANCE, len(prefix)))
            cup["marker_coord"] = prefix[distance - 1]
            cup["marker_color"] = marker_color
            cup["reserved_path"] = prefix[: distance - 1]
        elif source_kind == "void":
            if not clear_to_border or len(prefix) < MIN_RAY_DISTANCE:
                continue
            cup["reserved_path"] = prefix
        else:
            raise ValueError(f"unknown root source kind: {source_kind}")
        _stamp_cup(grid, cup)
        blocked.update(_cup_bbox_cells(top, left, h, w))
        blocked.update(cup["reserved_path"])
        occupied_nonblack.update(cup["wall_cells"])
        if source_kind == "marker":
            mr, mc = cup["marker_coord"]
            grid[mr, mc] = marker_color
            blocked.add((mr, mc))
            occupied_nonblack.add((mr, mc))
        return cup
    return None


def _inherit_candidate_origin(
    target_cell: tuple[int, int],
    target_side: str,
    h: int,
    w: int,
    opening_offset: int,
    distance: int,
) -> tuple[int, int]:
    tr, tc = target_cell
    if target_side == "left":
        return tr - opening_offset, tc - distance - w + 1
    if target_side == "right":
        return tr - opening_offset, tc + distance
    if target_side == "top":
        return tr - distance - h + 1, tc - opening_offset
    if target_side == "bottom":
        return tr + distance, tc - opening_offset
    raise ValueError(f"unknown target side: {target_side}")


def _clear_path_to_target(
    opening: tuple[int, int],
    direction: tuple[int, int],
    target: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    dr, dc = direction
    path: list[tuple[int, int]] = []
    r, c = opening[0] + dr, opening[1] + dc
    while (r, c) != target:
        if (r, c) in blocked:
            return None
        path.append((r, c))
        r += dr
        c += dc
    return path


def _place_inherit_cup(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    occupied_nonblack: set[tuple[int, int]],
    source_cup: dict[str, Any],
) -> dict[str, Any] | None:
    side_order = list(rng.permutation(["left", "right", "top", "bottom"]))
    for _ in range(CUP_PLACEMENT_ATTEMPTS):
        target_side = side_order[_ % 4]
        targets = [
            cell
            for cell in _wall_cells_for_side(source_cup, target_side)
            if cell not in source_cup["used_target_cells"]
        ]
        if not targets:
            continue
        target_cell = targets[_randint(rng, 0, len(targets) - 1)]
        opening_side = TARGET_SIDE_TO_VIEWER_OPEN_SIDE[target_side]
        direction = TARGET_SIDE_TO_VIEW_DIR[target_side]
        h = _randint(rng, CUP_MIN, CUP_MAX)
        w = _randint(rng, CUP_MIN, CUP_MAX)
        opening_offset = _randint(
            rng,
            1,
            (w - 2) if opening_side in {"top", "bottom"} else (h - 2),
        )
        distance = _randint(rng, MIN_RAY_DISTANCE, MAX_RAY_DISTANCE)
        top, left = _inherit_candidate_origin(target_cell, target_side, h, w, opening_offset, distance)
        if not _cup_bbox_is_clear(top, left, h, w, grid.shape, blocked, occupied_nonblack):
            continue
        cup = _make_cup_plan(top, left, h, w, opening_side, opening_offset, "cup")
        path = _clear_path_to_target(cup["opening"], direction, target_cell, blocked)
        if path is None:
            continue
        cup["source_bbox"] = source_cup["bbox"]
        cup["source_target_cell"] = target_cell
        cup["reserved_path"] = path
        _stamp_cup(grid, cup)
        blocked.update(_cup_bbox_cells(top, left, h, w))
        blocked.update(path)
        occupied_nonblack.update(cup["wall_cells"])
        source_cup["used_target_cells"].add(target_cell)
        return cup
    return None


def _place_leftover_markers(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    occupied_nonblack: set[tuple[int, int]],
    colors: list[int],
) -> list[tuple[tuple[int, int], int]]:
    leftovers: list[tuple[tuple[int, int], int]] = []
    h, w = grid.shape
    for color in colors:
        candidates = []
        for r in range(h):
            for c in range(w):
                if (r, c) in blocked:
                    continue
                around = {
                    (rr, cc)
                    for rr in range(max(0, r - 1), min(h - 1, r + 1) + 1)
                    for cc in range(max(0, c - 1), min(w - 1, c + 1) + 1)
                }
                if any(cell in occupied_nonblack for cell in around):
                    continue
                candidates.append((r, c))
        if not candidates:
            raise RuntimeError("failed to place leftover marker")
        coord = candidates[_randint(rng, 0, len(candidates) - 1)]
        grid[coord] = color
        blocked.add(coord)
        occupied_nonblack.add(coord)
        leftovers.append((coord, color))
    return leftovers


def _resolve_plan_colors(plans: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], int | None]:
    by_bbox = {plan["bbox"]: plan for plan in plans}
    memo: dict[tuple[int, int, int, int], int | None] = {}

    def resolve(bbox: tuple[int, int, int, int]) -> int | None:
        if bbox in memo:
            return memo[bbox]
        plan = by_bbox[bbox]
        if plan["source_kind"] == "marker":
            color = plan["marker_color"]
        elif plan["source_kind"] == "cup":
            color = resolve(plan["source_bbox"])
        elif plan["source_kind"] == "void":
            color = None
        else:
            raise ValueError("unknown source kind")
        memo[bbox] = color
        return color

    for bbox in by_bbox:
        resolve(bbox)
    return memo


def _validate_scene(
    grid: np.ndarray,
    cup_plans: list[dict[str, Any]],
    leftover_markers: list[tuple[tuple[int, int], int]],
) -> bool:
    details = _solve_details(grid)
    planned_bboxes = {plan["bbox"] for plan in cup_plans}
    if planned_bboxes != set(details["colors_by_bbox"]):
        return False

    expected_colors = _resolve_plan_colors(cup_plans)
    planned_consumed = {
        plan["marker_coord"] for plan in cup_plans if plan["source_kind"] == "marker"
    }
    if planned_consumed != details["consumed_markers"]:
        return False

    inherit_count = 0
    for plan in cup_plans:
        bbox = plan["bbox"]
        hit = details["hits_by_bbox"][bbox]
        if details["colors_by_bbox"][bbox] != expected_colors[bbox]:
            return False
        if plan["source_kind"] == "marker":
            if hit["kind"] != "marker":
                return False
            if hit["coord"] != plan["marker_coord"] or hit["color"] != plan["marker_color"]:
                return False
        elif plan["source_kind"] == "cup":
            inherit_count += 1
            if hit["kind"] != "cup":
                return False
            if hit["target_bbox"] != plan["source_bbox"] or hit["coord"] != plan["source_target_cell"]:
                return False
        elif plan["source_kind"] == "void":
            if hit["kind"] != "void":
                return False
        else:
            return False

    if inherit_count == 0:
        return False

    cup_outcomes = {details["colors_by_bbox"][plan["bbox"]] for plan in cup_plans}
    if len(cup_outcomes) < 2:
        return False

    direct_colors = [plan["marker_color"] for plan in cup_plans if plan["source_kind"] == "marker"]
    if len(set(direct_colors)) < 2:
        return False

    if not leftover_markers:
        return False
    out = details["output"]
    for coord, color in leftover_markers:
        if int(out[coord]) != color:
            return False

    if np.array_equal(out, grid):
        return False
    return True


def _build_scene(rng: np.random.Generator) -> np.ndarray | None:
    height = _randint(rng, GRID_MIN, GRID_MAX)
    width = _randint(rng, GRID_MIN, GRID_MAX)
    grid = np.zeros((height, width), dtype=int)
    blocked: set[tuple[int, int]] = set()
    occupied_nonblack: set[tuple[int, int]] = set()

    n_cups = int(rng.choice([3, 4, 4, 5]))
    chain_len = _randint(rng, 2, min(3, n_cups - 1))
    remaining = n_cups - chain_len

    extra_modes = ["marker"]
    if remaining >= 2:
        extra_modes.append("void")
    if remaining >= 3:
        extra_modes.append(str(rng.choice(["marker", "void"])))
    if len(extra_modes) != remaining:
        return None

    marker_roots = 1 + extra_modes.count("marker")
    root_colors = list(rng.choice(MARKER_COLORS, size=marker_roots, replace=False))
    color_idx = 0

    cup_plans: list[dict[str, Any]] = []

    chain_root = _place_root_cup(
        rng,
        grid,
        blocked,
        occupied_nonblack,
        "marker",
        marker_color=int(root_colors[color_idx]),
    )
    color_idx += 1
    if chain_root is None:
        return None
    cup_plans.append(chain_root)
    prev = chain_root
    for _ in range(chain_len - 1):
        child = _place_inherit_cup(rng, grid, blocked, occupied_nonblack, prev)
        if child is None:
            return None
        cup_plans.append(child)
        prev = child

    order = list(rng.permutation(extra_modes)) if extra_modes else []
    for mode in order:
        if mode == "marker":
            cup = _place_root_cup(
                rng,
                grid,
                blocked,
                occupied_nonblack,
                "marker",
                marker_color=int(root_colors[color_idx]),
            )
            color_idx += 1
        else:
            cup = _place_root_cup(rng, grid, blocked, occupied_nonblack, "void")
        if cup is None:
            return None
        cup_plans.append(cup)

    leftover_count = _randint(rng, 1, 3)
    leftover_colors = [int(rng.choice(root_colors))]
    while len(leftover_colors) < leftover_count:
        leftover_colors.append(int(rng.choice(MARKER_COLORS)))
    leftover_markers = _place_leftover_markers(
        rng, grid, blocked, occupied_nonblack, leftover_colors
    )

    if not _validate_scene(grid, cup_plans, leftover_markers):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_scene(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
