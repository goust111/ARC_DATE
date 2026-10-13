from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
TEAL = 8

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Every template is 4-connected, has area 3/4/5, and is centered around the
# median cell so its implicit 3x3 gate box is uniquely recoverable.
AREA_SHAPES: dict[int, list[np.ndarray]] = {
    3: [
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=int),
    ],
    4: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=int),
    ],
    5: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[0, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
    ],
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _median_coord(vals: list[int]) -> int:
    vals = sorted(int(v) for v in vals)
    n = len(vals)
    if n % 2 == 1:
        return int(vals[n // 2])
    return int((vals[n // 2 - 1] + vals[n // 2]) // 2)


def _rotate_direction(direction: tuple[int, int], area: int) -> tuple[int, int]:
    dr, dc = map(int, direction)
    if area == 3:
        return dr, dc
    if area == 4:
        return dc, -dr
    if area == 5:
        return -dc, dr
    raise ValueError(f"unsupported gate area: {area}")


def _find_gates(grid: np.ndarray) -> list[dict]:
    gates: list[dict] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        area = len(comp)
        if area not in (3, 4, 5):
            continue
        center = (_median_coord(rows), _median_coord(cols))
        r0, c0 = center[0] - 1, center[1] - 1
        r1, c1 = center[0] + 1, center[1] + 1
        if r0 < 0 or c0 < 0 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
            continue
        cells = {(r, c) for r, c in comp}
        if not all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells):
            continue
        gates.append(
            {
                "area": int(area),
                "center": (int(center[0]), int(center[1])),
                "box": (int(r0), int(c0), int(r1), int(c1)),
                "cells": cells,
            }
        )
    return gates


def _step_with_reflection(
    r: int, c: int, dr: int, dc: int, W: int
) -> tuple[int, int, int, int, int]:
    nr = r + dr
    nc = c + dc
    reflected = 0
    if nc < 0 or nc >= W:
        dc = -dc
        reflected = 1
        nc = c + dc
    return int(nr), int(nc), int(dr), int(dc), int(reflected)


def _solve_with_meta(
    grid: np.ndarray, override_areas: dict[int, int] | None = None
) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    starts = np.argwhere(g == BLUE)
    assert starts.shape[0] == 1, "expected exactly one blue start pixel"

    gates = _find_gates(g)
    lookup: dict[tuple[int, int], int] = {}
    for gid, gate in enumerate(gates):
        r0, c0, r1, c1 = gate["box"]
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                lookup[(int(rr), int(cc))] = int(gid)

    r, c = map(int, starts[0])
    dr, dc = -1, 1
    reflections = 0
    path_mask = np.zeros_like(g, dtype=bool)
    path_mask[r, c] = True
    gate_hits: list[int] = []

    for _ in range(H * W * 20):
        nr, nc, dr, dc, bounced = _step_with_reflection(r, c, dr, dc, W)
        reflections += bounced
        if nr < 0:
            exit_side = "top"
            break
        if nr >= H:
            exit_side = "bottom"
            break

        gid = lookup.get((nr, nc))
        if gid is None:
            r, c = nr, nc
            path_mask[r, c] = True
            continue

        gate = gates[int(gid)]
        area = int(override_areas.get(gid, gate["area"])) if override_areas else int(gate["area"])
        new_dr, new_dc = _rotate_direction((dr, dc), area)
        cr, cc = gate["center"]
        exit_cell = (int(cr + new_dr), int(cc + new_dc))
        for rr, cc2 in ((nr, nc), (cr, cc), exit_cell):
            if 0 <= rr < H and 0 <= cc2 < W:
                path_mask[int(rr), int(cc2)] = True
        r, c = exit_cell
        dr, dc = new_dr, new_dc
        gate_hits.append(int(gid))
    else:
        raise RuntimeError("ray simulation did not terminate")

    out = g.copy()
    out[path_mask & (g != GREY)] = BLUE

    regions = _components(~path_mask)
    region_sizes = [len(region) for region in regions]
    min_size = min(region_sizes) if region_sizes else 0
    recolored_black = 0
    for region in regions:
        if len(region) != min_size:
            continue
        for rr, cc in region:
            if out[rr, cc] == BLACK:
                out[rr, cc] = TEAL
                recolored_black += 1

    meta = {
        "gates": gates,
        "gate_hits": gate_hits,
        "reflections": int(reflections),
        "path_mask": path_mask,
        "regions": regions,
        "region_sizes": region_sizes,
        "exit_side": exit_side,
        "recolored_black": int(recolored_black),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a bouncing diagonal ray through glyph gates, then recolor the smallest path-split region.

    Input structure:
    - Black background with exactly one blue start pixel on the bottom edge.
    - Several gray glyph gates, each occupying an implicit 3x3 local box and having area 3, 4, or 5.
    - The ray starts by moving one row up and one column right; left and right edges are reflecting walls.

    Rule:
    - Extend the blue ray diagonally, reflecting only on the left/right borders.
    - When the next step would enter a gate's 3x3 box, step into the box, read that gate's gray-pixel area, rotate the ray (area 3 = straight, 4 = clockwise 90 degrees, 5 = counterclockwise 90 degrees), pass through the gate center, and leave through the corresponding opposite-side cell.
    - Path cells become blue unless they are already gray gate pixels, which stay gray while still counting as part of the path boundary.
    - After the ray exits the grid, treat the whole traced path as a barrier, compute the 4-connected regions of all non-path cells, and recolor every black cell inside the strictly smallest region(s) teal.

    Key insight:
    - The gate shape itself is not the turn; its gray-pixel count changes the ray's motion, and only the full bounced trajectory reveals which side-region is minimal.
    """
    out, _meta = _solve_with_meta(np.asarray(grid, dtype=int))
    return out


def _boxes_separated(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 < or0 - 1 or or1 < r0 - 1 or c1 < oc0 - 1 or oc1 < c0 - 1):
            return False
    return True


def _stamp_gate(grid: np.ndarray, area: int, box: tuple[int, int, int, int]) -> None:
    shape = random.choice(AREA_SHAPES[int(area)])
    r0, c0, _r1, _c1 = box
    for rr in range(3):
        for cc in range(3):
            if shape[rr, cc]:
                grid[r0 + rr, c0 + cc] = GREY


def _build_candidate() -> tuple[np.ndarray, bool] | None:
    H = random.randint(15, 21)
    W = random.randint(15, 21)
    start_c = random.randint(2, W - 3)
    want_reflection = random.random() < 0.65

    grid = np.zeros((H, W), dtype=int)
    grid[H - 1, start_c] = BLUE

    state = (H - 1, start_c, -1, 1)
    occupied = {(H - 1, start_c)}
    gate_boxes: list[tuple[int, int, int, int]] = []

    for area in random.sample([3, 4, 5], 3):
        found = False
        target_steps = random.randint(1, 7)
        for step in range(100):
            nr, nc, dr, dc, _bounced = _step_with_reflection(*state, W)
            if nr < 1 or nr >= H - 1:
                return None

            center = (int(nr + dr), int(nc + dc))
            box = (center[0] - 1, center[1] - 1, center[0] + 1, center[1] + 1)
            can_place = (
                step >= target_steps
                and 1 <= center[0] < H - 1
                and 1 <= center[1] < W - 1
                and _boxes_separated(box, gate_boxes)
                and all((rr, cc) not in occupied for rr in range(box[0], box[2] + 1) for cc in range(box[1], box[3] + 1))
            )
            if can_place:
                _stamp_gate(grid, area, box)
                new_dr, new_dc = _rotate_direction((dr, dc), area)
                exit_cell = (center[0] + new_dr, center[1] + new_dc)
                occupied.update([(nr, nc), center, exit_cell])
                gate_boxes.append(box)
                state = (int(exit_cell[0]), int(exit_cell[1]), int(new_dr), int(new_dc))
                found = True
                break

            state = (int(nr), int(nc), int(dr), int(dc))
            occupied.add((int(nr), int(nc)))

        if not found:
            return None

    return grid, want_reflection


def _valid_template(shape: np.ndarray, area: int) -> bool:
    if int(shape.sum()) != int(area):
        return False
    cells = {(int(r), int(c)) for r, c in np.argwhere(shape)}
    if len(cells) != area:
        return False
    comp = _components(shape.astype(bool))
    if len(comp) != 1 or len(comp[0]) != area:
        return False
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return _median_coord(rows) == 1 and _median_coord(cols) == 1


def _validate_candidate(grid: np.ndarray, want_reflection: bool) -> bool:
    out, meta = _solve_with_meta(grid)
    gates = meta["gates"]
    if len(gates) != 3:
        return False
    if sorted(int(g["area"]) for g in gates) != [3, 4, 5]:
        return False
    if meta["exit_side"] != "top":
        return False
    if len(meta["gate_hits"]) != 3 or len(set(meta["gate_hits"])) != 3:
        return False
    if want_reflection and meta["reflections"] < 1:
        return False
    if len(meta["regions"]) < 2:
        return False
    if meta["recolored_black"] < 3:
        return False
    if meta["recolored_black"] > grid.size // 2:
        return False
    if not np.any(out == TEAL):
        return False

    # Every gate must matter: perturbing any instruction must change the answer.
    for gid, gate in enumerate(gates):
        alt_area = 4 if int(gate["area"]) == 3 else 3
        alt_out, _alt_meta = _solve_with_meta(grid, {gid: alt_area})
        if np.array_equal(out, alt_out):
            return False

    # The turning logic as a whole must matter too.
    all_straight_out, _all_straight_meta = _solve_with_meta(grid, {gid: 3 for gid in range(len(gates))})
    if np.array_equal(out, all_straight_out):
        return False

    return True


def generate_input() -> np.ndarray:
    for area, shapes in AREA_SHAPES.items():
        for shape in shapes:
            assert _valid_template(shape, area), f"invalid template for area {area}"

    for _ in range(1200):
        built = _build_candidate()
        if built is None:
            continue
        grid, want_reflection = built
        try:
            if _validate_candidate(grid, want_reflection):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid glyph-gated bounce puzzle")
