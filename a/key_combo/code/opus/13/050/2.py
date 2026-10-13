from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BG = 0
FRAME = 3
CONTROL = 5
MARKER = 9
OBJECT_COLORS = [1, 2, 4, 6, 7, 8]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400
SELECT_RETRIES = 300

# Base shapes are defined for a rightward push with contact cell at (0, 0).
BASE_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _cheb_parity(cell: tuple[int, int], control: tuple[int, int]) -> int:
    return max(abs(cell[0] - control[0]), abs(cell[1] - control[1])) % 2


def _frontier_mask(coords: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    dr, dc = direction
    if dr == 1:
        return coords[:, 0] == coords[:, 0].max()
    if dr == -1:
        return coords[:, 0] == coords[:, 0].min()
    if dc == 1:
        return coords[:, 1] == coords[:, 1].max()
    return coords[:, 1] == coords[:, 1].min()


def _frontier(coords: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    return coords[_frontier_mask(coords, direction)]


def _orient_shape(base_shape: list[tuple[int, int]], direction: tuple[int, int]) -> np.ndarray:
    if direction == (0, 1):
        pts = base_shape
    elif direction == (1, 0):
        pts = [(c, r) for r, c in base_shape]
    elif direction == (0, -1):
        pts = [(r, -c) for r, c in base_shape]
    else:
        pts = [(-c, r) for r, c in base_shape]
    return np.array(pts, dtype=int)


def _shift_coords(coords: np.ndarray, direction: tuple[int, int], steps: int) -> np.ndarray:
    return coords + np.array(direction, dtype=int) * int(steps)


def _steps_to_interior_boundary(
    coords: np.ndarray, direction: tuple[int, int], inner_bbox: tuple[int, int, int, int]
) -> int:
    r0, c0, r1, c1 = inner_bbox
    dr, dc = direction
    if dr == 1:
        return int(np.min(r1 - coords[:, 0]))
    if dr == -1:
        return int(np.min(coords[:, 0] - r0))
    if dc == 1:
        return int(np.min(c1 - coords[:, 1]))
    return int(np.min(coords[:, 1] - c0))


def _steps_to_parity_stop(
    coords: np.ndarray,
    direction: tuple[int, int],
    start_parity: int,
    control: tuple[int, int],
    inner_bbox: tuple[int, int, int, int],
) -> int:
    max_steps = _steps_to_interior_boundary(coords, direction, inner_bbox)
    for step in range(1, max_steps + 1):
        shifted = _shift_coords(coords, direction, step)
        front = _frontier(shifted, direction)
        if all(_cheb_parity((int(r), int(c)), control) != start_parity for r, c in front):
            return step
    raise ValueError("object never reaches the opposite parity frontier before leaving the frame")


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame_coords = np.argwhere(grid == FRAME)
    if frame_coords.size == 0:
        raise ValueError("missing frame")
    r0, c0, r1, c1 = _bbox(frame_coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise ValueError("frame must have an interior")
    if not np.all(grid[r0, c0 : c1 + 1] == FRAME):
        raise ValueError("frame top edge is broken")
    if not np.all(grid[r1, c0 : c1 + 1] == FRAME):
        raise ValueError("frame bottom edge is broken")
    if not np.all(grid[r0 : r1 + 1, c0] == FRAME):
        raise ValueError("frame left edge is broken")
    if not np.all(grid[r0 : r1 + 1, c1] == FRAME):
        raise ValueError("frame right edge is broken")
    border = np.zeros_like(grid, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    if not np.array_equal(grid == FRAME, border):
        raise ValueError("green pixels must form exactly one rectangular border")
    return r0, c0, r1, c1


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    frame_bbox = _find_frame_bbox(g)
    r0, c0, r1, c1 = frame_bbox
    inner_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)

    controls = np.argwhere(g == CONTROL)
    if controls.shape[0] != 1:
        raise ValueError("expected exactly one control pixel")
    control = (int(controls[0, 0]), int(controls[0, 1]))
    if not (inner_bbox[0] <= control[0] <= inner_bbox[2] and inner_bbox[1] <= control[1] <= inner_bbox[3]):
        raise ValueError("control must be inside the frame")

    marker_positions = {tuple(map(int, coord)) for coord in np.argwhere(g == MARKER)}
    objects = []
    used_markers = set()

    for color in sorted(set(map(int, np.unique(g))) - {BG, FRAME, CONTROL, MARKER}):
        for coords in _components_of_color(g, color):
            if coords.shape[0] == 0:
                continue
            if np.any(coords[:, 0] < inner_bbox[0]) or np.any(coords[:, 0] > inner_bbox[2]):
                raise ValueError("object leaves the frame interior")
            if np.any(coords[:, 1] < inner_bbox[1]) or np.any(coords[:, 1] > inner_bbox[3]):
                raise ValueError("object leaves the frame interior")

            adjacencies = []
            for rr, cc in coords.tolist():
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(g.shape, nr, nc) and g[nr, nc] == MARKER:
                        adjacencies.append(((rr, cc), (nr, nc), (dr, dc)))
            marker_cells = {adj[1] for adj in adjacencies}
            dir_to_marker = {adj[2] for adj in adjacencies}
            contact_cells = {adj[0] for adj in adjacencies}
            if len(marker_cells) != 1 or len(dir_to_marker) != 1 or len(contact_cells) == 0:
                raise ValueError("each object must touch exactly one marker on one side")

            marker = next(iter(marker_cells))
            used_markers.add(marker)
            dir_marker = next(iter(dir_to_marker))
            push_dir = (-dir_marker[0], -dir_marker[1])
            parities = {_cheb_parity(cell, control) for cell in contact_cells}
            if len(parities) != 1:
                raise ValueError("contact-side parity must be unambiguous")
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": {tuple(map(int, cell)) for cell in coords.tolist()},
                    "marker": marker,
                    "push_dir": push_dir,
                    "start_parity": next(iter(parities)),
                    "contact_cells": contact_cells,
                }
            )

    if used_markers != marker_positions:
        raise ValueError("every marker must belong to exactly one object")
    if not objects:
        raise ValueError("missing movable objects")
    return {"frame_bbox": frame_bbox, "inner_bbox": inner_bbox, "control": control, "objects": objects}


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    scene = _parse_scene(grid)
    out = np.asarray(grid, dtype=int).copy()
    objects = scene["objects"]
    inner_bbox = scene["inner_bbox"]
    control = scene["control"]

    placements = []
    blockers = {FRAME, CONTROL, MARKER}
    for obj in objects:
        coords = obj["coords"]
        direction = obj["push_dir"]
        stop_step = _steps_to_parity_stop(coords, direction, obj["start_parity"], control, inner_bbox)

        own = obj["coord_set"]
        for step in range(1, stop_step + 1):
            shifted = _shift_coords(coords, direction, step)
            for rr, cc in shifted.tolist():
                cell = (int(rr), int(cc))
                if cell in own:
                    continue
                if out[cell] in blockers or (out[cell] not in {BG, int(obj["color"])}):
                    raise ValueError("movement path is obstructed")

        out[coords[:, 0], coords[:, 1]] = BG
        marker_r, marker_c = obj["marker"]
        out[marker_r, marker_c] = BG
        placements.append((int(obj["color"]), _shift_coords(coords, direction, stop_step)))

    for color, shifted in placements:
        if not np.all(out[shifted[:, 0], shifted[:, 1]] == BG):
            raise ValueError("moved objects would overlap")
        out[shifted[:, 0], shifted[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each marker-launched object until its leading edge first crosses the control pixel's odd/even Chebyshev boundary.

    Input structure:
    - A black background contains a one-pixel-thick green rectangular frame.
    - Inside the frame there is exactly one grey control pixel, several small monochrome objects, and one maroon marker pixel touching each object on exactly one cardinal side.
    - Objects are 4-connected, remain separated from each other, and sit entirely inside the frame interior.

    Rule:
    - Use the grey control pixel as the center of an implicit Chebyshev-distance parity map over the frame interior.
    - For each object, read the maroon contact side and push the object one cell at a time away from that marker.
    - Stop as soon as every cell on the object's current leading edge lies in the parity opposite to the object's marker-contact parity, then erase all maroon markers.

    Key insight:
    - The green frame only bounds the arena; the true stopping line is the invisible odd/even ring boundary induced by the grey control pixel, while the maroon marker fixes the launch direction.
    """
    return _apply_rule(np.asarray(grid, dtype=int))


def _make_candidate(
    anchor: tuple[int, int],
    shape: np.ndarray,
    direction: tuple[int, int],
    control: tuple[int, int],
    inner_bbox: tuple[int, int, int, int],
) -> dict | None:
    coords = shape + np.array(anchor, dtype=int)
    marker = (anchor[0] - direction[0], anchor[1] - direction[1])
    r0, c0, r1, c1 = inner_bbox
    cells = {tuple(map(int, cell)) for cell in coords.tolist()}
    all_cells = set(cells)
    all_cells.add(marker)
    if len(all_cells) != len(cells) + 1:
        return None
    if control in all_cells:
        return None
    if any(abs(rr - control[0]) + abs(cc - control[1]) == 1 for rr, cc in all_cells):
        return None
    if not (r0 <= marker[0] <= r1 and c0 <= marker[1] <= c1):
        return None
    if np.any(coords[:, 0] < r0) or np.any(coords[:, 0] > r1) or np.any(coords[:, 1] < c0) or np.any(coords[:, 1] > c1):
        return None

    contact = (anchor[0], anchor[1])
    adjacent = {
        cell for cell in cells if abs(cell[0] - marker[0]) + abs(cell[1] - marker[1]) == 1
    }
    if adjacent != {contact}:
        return None

    start_parity = _cheb_parity(contact, control)
    max_steps = _steps_to_interior_boundary(coords, direction, inner_bbox)
    if max_steps <= 1:
        return None

    stop_step = None
    for step in range(1, max_steps + 1):
        shifted = _shift_coords(coords, direction, step)
        front = _frontier(shifted, direction)
        all_opposite = all(
            _cheb_parity((int(rr), int(cc)), control) != start_parity for rr, cc in front
        )
        if all_opposite:
            stop_step = step
            if step == 1:
                previous_ok = True
            else:
                prev_front = _frontier(_shift_coords(coords, direction, step - 1), direction)
                previous_ok = not all(
                    _cheb_parity((int(rr), int(cc)), control) != start_parity for rr, cc in prev_front
                )
            if not previous_ok:
                return None
            break
    if stop_step is None or stop_step >= max_steps:
        return None

    later_same = False
    for step in range(stop_step + 1, max_steps + 1):
        front = _frontier(_shift_coords(coords, direction, step), direction)
        if all(_cheb_parity((int(rr), int(cc)), control) == start_parity for rr, cc in front):
            later_same = True
            break
    if not later_same:
        return None

    final_coords = _shift_coords(coords, direction, stop_step)
    swept = set()
    for step in range(stop_step + 1):
        shifted = _shift_coords(coords, direction, step)
        swept.update({tuple(map(int, cell)) for cell in shifted.tolist()})

    input_cells = set(cells)
    input_occ = set(input_cells)
    input_occ.add(marker)
    final_cells = {tuple(map(int, cell)) for cell in final_coords.tolist()}
    input_buffer = set(input_occ)
    final_buffer = set(final_cells)
    for rr, cc in input_occ:
        input_buffer.update(_neighbors4(rr, cc))
    for rr, cc in final_cells:
        final_buffer.update(_neighbors4(rr, cc))

    return {
        "coords": coords,
        "shape": shape,
        "marker": marker,
        "push_dir": direction,
        "start_parity": start_parity,
        "stop_step": stop_step,
        "max_steps": max_steps,
        "contact": contact,
        "input_occ": input_occ,
        "input_buffer": input_buffer,
        "final_occ": final_cells,
        "final_buffer": final_buffer,
        "swept": swept,
    }


def _candidates_compatible(a: dict, b: dict) -> bool:
    if a["input_buffer"] & b["input_occ"]:
        return False
    if b["input_buffer"] & a["input_occ"]:
        return False
    if a["final_buffer"] & b["final_occ"]:
        return False
    if b["final_buffer"] & a["final_occ"]:
        return False
    if a["swept"] & b["swept"]:
        return False
    if a["swept"] & b["input_occ"]:
        return False
    if b["swept"] & a["input_occ"]:
        return False
    if a["swept"] & b["final_occ"]:
        return False
    if b["swept"] & a["final_occ"]:
        return False
    return True


def _enumerate_candidates(
    control: tuple[int, int], inner_bbox: tuple[int, int, int, int]
) -> list[dict]:
    r0, c0, r1, c1 = inner_bbox
    candidates = []
    for direction in DIRS:
        for base_shape in BASE_SHAPES:
            shape = _orient_shape(base_shape, direction)
            for ar in range(r0, r1 + 1):
                for ac in range(c0, c1 + 1):
                    cand = _make_candidate((ar, ac), shape, direction, control, inner_bbox)
                    if cand is not None:
                        candidates.append(cand)
    return candidates


def _pick_with_quotas(
    by_dir: dict[tuple[int, int], list[dict]],
    quotas: dict[tuple[int, int], int],
    rng: np.random.Generator,
) -> list[dict] | None:
    dirs = list(quotas)
    for _ in range(SELECT_RETRIES):
        picked: list[dict] = []
        good = True
        rng.shuffle(dirs)
        for direction in dirs:
            quota = quotas[direction]
            options = by_dir[direction][:]
            rng.shuffle(options)
            used = 0
            for cand in options:
                if all(_candidates_compatible(cand, prev) for prev in picked):
                    picked.append(cand)
                    used += 1
                    if used == quota:
                        break
            if used != quota:
                good = False
                break
        if good:
            return picked
    return None


def _choose_candidates(candidates: list[dict], count: int, rng: np.random.Generator) -> list[dict] | None:
    by_dir: dict[tuple[int, int], list[dict]] = {direction: [] for direction in DIRS}
    for cand in candidates:
        by_dir[cand["push_dir"]].append(cand)
    usable_dirs = [direction for direction, items in by_dir.items() if items]
    if len(usable_dirs) < 2:
        return None

    for _ in range(SELECT_RETRIES):
        if count == 2:
            chosen_dirs = list(rng.choice(len(usable_dirs), size=2, replace=False))
            quotas = {usable_dirs[idx]: 1 for idx in chosen_dirs}
        else:
            if len(usable_dirs) >= 3 and rng.random() < 0.55:
                chosen_dirs = list(rng.choice(len(usable_dirs), size=3, replace=False))
                quotas = {usable_dirs[idx]: 1 for idx in chosen_dirs}
            else:
                pair = list(rng.choice(len(usable_dirs), size=2, replace=False))
                quotas = {usable_dirs[pair[0]]: 2, usable_dirs[pair[1]]: 1}
        picked = _pick_with_quotas(by_dir, quotas, rng)
        if picked is not None:
            return picked
    return None


def _render_grid(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    control: tuple[int, int],
    candidates: list[dict],
    colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[control] = CONTROL
    for cand, color in zip(candidates, colors):
        rr, cc = cand["marker"]
        grid[rr, cc] = MARKER
        coords = cand["coords"]
        grid[coords[:, 0], coords[:, 1]] = color
    return grid


def _boundary_push_output(grid: np.ndarray) -> np.ndarray:
    scene = _parse_scene(grid)
    out = np.asarray(grid, dtype=int).copy()
    placements = []
    for obj in scene["objects"]:
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = BG
        mr, mc = obj["marker"]
        out[mr, mc] = BG
        steps = _steps_to_interior_boundary(obj["coords"], obj["push_dir"], scene["inner_bbox"])
        placements.append((obj["color"], _shift_coords(obj["coords"], obj["push_dir"], steps)))
    for color, shifted in placements:
        out[shifted[:, 0], shifted[:, 1]] = color
    return out


def _shift_control(grid: np.ndarray, new_control: tuple[int, int]) -> np.ndarray:
    shifted = np.asarray(grid, dtype=int).copy()
    control = tuple(map(int, np.argwhere(shifted == CONTROL)[0]))
    shifted[control] = BG
    shifted[new_control] = CONTROL
    return shifted


def _validate_generated(grid: np.ndarray) -> bool:
    scene = _parse_scene(grid)
    objects = scene["objects"]
    control = scene["control"]
    inner_bbox = scene["inner_bbox"]

    if not (2 <= len(objects) <= 3):
        return False
    colors = [obj["color"] for obj in objects]
    if len(set(colors)) != len(colors):
        return False

    occupied = set()
    for obj in objects:
        if not (2 <= obj["coords"].shape[0] <= 4):
            return False
        marker = obj["marker"]
        if abs(marker[0] - control[0]) + abs(marker[1] - control[1]) == 1:
            return False
        if any(abs(rr - control[0]) + abs(cc - control[1]) == 1 for rr, cc in obj["coord_set"]):
            return False
        boundary_steps = _steps_to_interior_boundary(obj["coords"], obj["push_dir"], inner_bbox)
        stop_step = _steps_to_parity_stop(
            obj["coords"], obj["push_dir"], obj["start_parity"], control, inner_bbox
        )
        if not (1 <= stop_step < boundary_steps):
            return False
        front = _frontier(_shift_coords(obj["coords"], obj["push_dir"], stop_step), obj["push_dir"])
        if not all(_cheb_parity((int(rr), int(cc)), control) != obj["start_parity"] for rr, cc in front):
            return False
        prev_front = _frontier(_shift_coords(obj["coords"], obj["push_dir"], stop_step - 1), obj["push_dir"])
        if all(_cheb_parity((int(rr), int(cc)), control) != obj["start_parity"] for rr, cc in prev_front):
            return False
        later_same = False
        for step in range(stop_step + 1, boundary_steps + 1):
            later_front = _frontier(_shift_coords(obj["coords"], obj["push_dir"], step), obj["push_dir"])
            if all(_cheb_parity((int(rr), int(cc)), control) == obj["start_parity"] for rr, cc in later_front):
                later_same = True
                break
        if not later_same:
            return False
        own = obj["coord_set"]
        for rr, cc in own | {marker}:
            if (rr, cc) in occupied:
                return False
            occupied.add((rr, cc))
        for step in range(1, stop_step + 1):
            shifted = _shift_coords(obj["coords"], obj["push_dir"], step)
            for rr, cc in shifted.tolist():
                cell = (int(rr), int(cc))
                if cell in own:
                    continue
                if grid[cell] != BG:
                    return False

    out = main(grid)
    if np.any(out == MARKER):
        return False
    if np.array_equal(out, _boundary_push_output(grid)):
        return False

    control_changes_output = False
    for dr, dc in DIRS:
        candidate = (control[0] + dr, control[1] + dc)
        if not (inner_bbox[0] <= candidate[0] <= inner_bbox[2] and inner_bbox[1] <= candidate[1] <= inner_bbox[3]):
            continue
        if grid[candidate] != BG:
            continue
        shifted = _shift_control(grid, candidate)
        try:
            if not np.array_equal(main(shifted), out):
                control_changes_output = True
                break
        except Exception:
            continue
    if not control_changes_output:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        height = int(rng.integers(13, 20))
        width = int(rng.integers(13, 20))

        frame_heights = [size for size in (9, 11, 13) if size <= height - 2]
        frame_widths = [size for size in (9, 11, 13) if size <= width - 2]
        if not frame_heights or not frame_widths:
            continue
        frame_h = int(rng.choice(frame_heights))
        frame_w = int(rng.choice(frame_widths))
        top = int(rng.integers(1, height - frame_h))
        left = int(rng.integers(1, width - frame_w))
        frame_bbox = (top, left, top + frame_h - 1, left + frame_w - 1)
        inner_bbox = (top + 1, left + 1, top + frame_h - 2, left + frame_w - 2)

        control_row = int(rng.integers(inner_bbox[0] + 1, inner_bbox[2]))
        control_col = int(rng.integers(inner_bbox[1] + 1, inner_bbox[3]))
        control = (control_row, control_col)

        candidates = _enumerate_candidates(control, inner_bbox)
        if len(candidates) < 20:
            continue

        object_count = 2 if rng.random() < 0.45 else 3
        chosen = _choose_candidates(candidates, object_count, rng)
        if chosen is None:
            continue

        colors = rng.choice(np.array(OBJECT_COLORS, dtype=int), size=object_count, replace=False).tolist()
        grid = _render_grid((height, width), frame_bbox, control, chosen, colors)
        if _validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid input grid")
