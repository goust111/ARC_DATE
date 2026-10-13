from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "N": (1, 0),
    "S": (-1, 0),
    "W": (0, 1),
    "E": (0, -1),
}
SIDE_CYCLE = ("N", "E", "S", "W")
COLOR_POOL = (1, 2, 3, 4)
MAX_GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([[int(dr), int(dc)]], dtype=int)


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _object_cells(grid: np.ndarray, coords: np.ndarray) -> dict[tuple[int, int], int]:
    return {tuple(map(int, rc)): int(grid[tuple(rc)]) for rc in coords}


def _contact_layer(coords: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    dr, dc = direction
    if dr == 1:
        bound = int(coords[:, 0].max())
        return coords[coords[:, 0] == bound]
    if dr == -1:
        bound = int(coords[:, 0].min())
        return coords[coords[:, 0] == bound]
    if dc == 1:
        bound = int(coords[:, 1].max())
        return coords[coords[:, 1] == bound]
    bound = int(coords[:, 1].min())
    return coords[coords[:, 1] == bound]


def _touches(mask: np.ndarray, coords: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _overlaps(mask: np.ndarray, coords: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return True
        if mask[rr, cc]:
            return True
    return False


def _shortest_direction_and_distance(
    coords: np.ndarray,
    absorber_mask: np.ndarray,
) -> tuple[tuple[int, int], int]:
    h, w = absorber_mask.shape
    best: list[tuple[int, tuple[int, int]]] = []
    limit = h + w + 2
    for direction in DIRS:
        dr, dc = direction
        for dist in range(1, limit):
            moved = _translate(coords, dr * dist, dc * dist)
            if _overlaps(absorber_mask, moved):
                break
            if _touches(absorber_mask, moved):
                best.append((dist, direction))
                break
    assert best, "each foreground object must have a valid shortest cardinal path"
    best.sort(key=lambda item: item[0])
    if len(best) > 1 and best[0][0] == best[1][0]:
        raise AssertionError("foreground object has ambiguous shortest direction")
    return best[0][1], int(best[0][0])


def _slide_until_contact(
    coords: np.ndarray,
    direction: tuple[int, int],
    blocker_mask: np.ndarray,
) -> np.ndarray:
    dr, dc = direction
    current = coords.copy()
    limit = blocker_mask.shape[0] + blocker_mask.shape[1] + 2
    for _ in range(limit):
        nxt = _translate(current, dr, dc)
        if _overlaps(blocker_mask, nxt):
            assert _touches(blocker_mask, current), "stopped without reaching the absorber"
            return current
        current = nxt
    raise AssertionError("sliding exceeded limit")


def _extract_objects(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != GREY)
    objects = []
    for coords in _components(mask):
        objects.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "cells": _object_cells(grid, coords),
            }
        )
    return objects


def _surface_cells(layer: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(int(r + dr), int(c + dc)) for r, c in layer]


def _simulate(
    grid: np.ndarray,
    *,
    swap_mode: str = "swap",
    dynamic_absorber: bool = True,
    order_override: list[int] | None = None,
    return_trace: bool = False,
):
    g = np.asarray(grid, dtype=int)
    grey_parts = _components(g == GREY)
    assert grey_parts, "expected at least one grey component"
    base_coords = max(grey_parts, key=lambda pts: int(pts.shape[0]))
    base_mask = _mask_from_coords(g.shape, base_coords)

    objects = _extract_objects(g)
    enriched = []
    for idx, obj in enumerate(objects):
        direction, distance = _shortest_direction_and_distance(obj["coords"], base_mask)
        enriched.append(
            {
                **obj,
                "index": idx,
                "direction": direction,
                "distance": distance,
            }
        )

    if order_override is None:
        enriched.sort(key=lambda obj: (obj["distance"], obj["bbox"][0], obj["bbox"][1]))
    else:
        by_index = {obj["index"]: obj for obj in enriched}
        enriched = [by_index[idx] for idx in order_override]

    out = np.full_like(g, BG)
    out[base_mask] = GREY
    absorber_mask = base_mask.copy()
    trace: list[dict] = []

    for obj in enriched:
        blocker = absorber_mask if dynamic_absorber else base_mask
        final_coords = _slide_until_contact(obj["coords"], obj["direction"], blocker)
        layer = _contact_layer(final_coords, obj["direction"])
        surfaces = _surface_cells(layer, obj["direction"])
        moved_cells = {
            tuple(map(int, final_rc)): obj["cells"][tuple(map(int, src_rc))]
            for final_rc, src_rc in zip(final_coords.tolist(), obj["coords"].tolist(), strict=True)
        }
        surface_before = [int(out[r, c]) for r, c in surfaces]
        assert all(color != BG for color in surface_before), "surface must belong to the absorber"

        if swap_mode == "swap":
            for r, c in layer:
                rr = int(r)
                cc = int(c)
                surf = (rr + obj["direction"][0], cc + obj["direction"][1])
                moved_color = moved_cells[(rr, cc)]
                surface_color = int(out[surf])
                moved_cells[(rr, cc)] = surface_color
                out[surf] = moved_color
        elif swap_mode == "grey":
            for pos in moved_cells:
                moved_cells[pos] = GREY
        else:
            assert swap_mode == "keep"

        for (r, c), color in moved_cells.items():
            out[r, c] = int(color)
            absorber_mask[r, c] = True

        trace.append(
            {
                "index": obj["index"],
                "distance": obj["distance"],
                "direction": obj["direction"],
                "surface_before": tuple(surface_before),
                "final_bbox": _bbox(final_coords),
                "surface_nonbase": any(not base_mask[r, c] for r, c in surfaces),
            }
        )

    if return_trace:
        return out, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored rectangles toward the central gray absorber, swapping each contact layer with the current surface it hits.

    Input structure:
    - The background is black, with one largest gray 4-connected rectangle acting as the absorber.
    - Three or four other foreground objects are detached single-color rectangles placed on cardinal sides of the absorber.
    - Each object is aligned so it can move straight up, down, left, or right toward the absorber.

    Rule:
    - Identify the largest gray component and every non-black, non-gray foreground object.
    - For each object, compute the unique shortest cardinal direction and distance to the gray absorber, then process objects in increasing distance order, breaking ties by top-left bbox position.
    - When an object is processed, slide it along that precomputed direction until it is flush with the current absorber mass, swap its contact layer colors with the opposite absorber surface cell-by-cell, and then merge the full recolored object into the absorber for subsequent steps.

    Key insight:
    - The absorber is not just a blocker: every collision rewrites the surface that the next object will see, so both arrival order and accumulated geometry matter.
    """
    return _simulate(grid, swap_mode="swap", dynamic_absorber=True)


def _surface_run_candidates(
    occupied: np.ndarray,
    side: str,
    height: int,
    width: int,
) -> list[dict]:
    h, w = occupied.shape
    candidates: list[dict] = []

    def add_candidate(coords: np.ndarray, direction: tuple[int, int]) -> None:
        coords_set = {tuple(map(int, rc)) for rc in coords}
        if any(occupied[r, c] for r, c in coords_set):
            return
        layer = _contact_layer(coords, direction)
        surfaces = _surface_cells(layer, direction)
        if not all(0 <= r < h and 0 <= c < w and occupied[r, c] for r, c in surfaces):
            return
        for r, c in coords_set:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in coords_set:
                    if (nr, nc) != (r + direction[0], c + direction[1]):
                        return
        candidates.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "surfaces": surfaces,
            }
        )

    if side == "N":
        for r in range(1, h):
            exposed = [c for c in range(w) if occupied[r, c] and not occupied[r - 1, c]]
            if not exposed or r - height < 0:
                continue
            runs: list[tuple[int, int]] = []
            start = prev = exposed[0]
            for c in exposed[1:]:
                if c == prev + 1:
                    prev = c
                else:
                    runs.append((start, prev))
                    start = prev = c
            runs.append((start, prev))
            for c0, c1 in runs:
                if c1 - c0 + 1 < width:
                    continue
                for left in range(c0, c1 - width + 2):
                    rows = np.arange(r - height, r)[:, None]
                    cols = np.arange(left, left + width)[None, :]
                    coords = np.stack(np.broadcast_arrays(rows, cols), axis=-1).reshape(-1, 2)
                    add_candidate(coords, SIDE_TO_DIR[side])
    elif side == "S":
        for r in range(h - 1):
            exposed = [c for c in range(w) if occupied[r, c] and not occupied[r + 1, c]]
            if not exposed or r + height >= h:
                continue
            runs = []
            start = prev = exposed[0]
            for c in exposed[1:]:
                if c == prev + 1:
                    prev = c
                else:
                    runs.append((start, prev))
                    start = prev = c
            runs.append((start, prev))
            for c0, c1 in runs:
                if c1 - c0 + 1 < width:
                    continue
                for left in range(c0, c1 - width + 2):
                    rows = np.arange(r + 1, r + 1 + height)[:, None]
                    cols = np.arange(left, left + width)[None, :]
                    coords = np.stack(np.broadcast_arrays(rows, cols), axis=-1).reshape(-1, 2)
                    add_candidate(coords, SIDE_TO_DIR[side])
    elif side == "W":
        for c in range(1, w):
            exposed = [r for r in range(h) if occupied[r, c] and not occupied[r, c - 1]]
            if not exposed or c - width < 0:
                continue
            runs = []
            start = prev = exposed[0]
            for r in exposed[1:]:
                if r == prev + 1:
                    prev = r
                else:
                    runs.append((start, prev))
                    start = prev = r
            runs.append((start, prev))
            for r0, r1 in runs:
                if r1 - r0 + 1 < height:
                    continue
                for top in range(r0, r1 - height + 2):
                    rows = np.arange(top, top + height)[:, None]
                    cols = np.arange(c - width, c)[None, :]
                    coords = np.stack(np.broadcast_arrays(rows, cols), axis=-1).reshape(-1, 2)
                    add_candidate(coords, SIDE_TO_DIR[side])
    else:
        for c in range(w - 1):
            exposed = [r for r in range(h) if occupied[r, c] and not occupied[r, c + 1]]
            if not exposed or c + width >= w:
                continue
            runs = []
            start = prev = exposed[0]
            for r in exposed[1:]:
                if r == prev + 1:
                    prev = r
                else:
                    runs.append((start, prev))
                    start = prev = r
            runs.append((start, prev))
            for r0, r1 in runs:
                if r1 - r0 + 1 < height:
                    continue
                for top in range(r0, r1 - height + 2):
                    rows = np.arange(top, top + height)[:, None]
                    cols = np.arange(c + 1, c + 1 + width)[None, :]
                    coords = np.stack(np.broadcast_arrays(rows, cols), axis=-1).reshape(-1, 2)
                    add_candidate(coords, SIDE_TO_DIR[side])

    return candidates


def _overhang_axis(side: str, next_side: str) -> tuple[str, int] | None:
    if side == "N" and next_side == "E":
        return ("c1", 1)
    if side == "N" and next_side == "W":
        return ("c0", -1)
    if side == "S" and next_side == "E":
        return ("c1", 1)
    if side == "S" and next_side == "W":
        return ("c0", -1)
    if side == "W" and next_side == "N":
        return ("r0", -1)
    if side == "W" and next_side == "S":
        return ("r1", 1)
    if side == "E" and next_side == "N":
        return ("r0", -1)
    if side == "E" and next_side == "S":
        return ("r1", 1)
    return None


def _max_gap(coords: np.ndarray, direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = direction
    if dr == 1:
        return int(coords[:, 0].min())
    if dr == -1:
        return int(h - 1 - coords[:, 0].max())
    if dc == 1:
        return int(coords[:, 1].min())
    return int(w - 1 - coords[:, 1].max())


def _move_outward(coords: np.ndarray, direction: tuple[int, int], gap: int) -> np.ndarray:
    return _translate(coords, -direction[0] * gap, -direction[1] * gap)


def _paths_clear(base_mask: np.ndarray, placements: list[dict], shape: tuple[int, int]) -> bool:
    occ = base_mask.copy()
    h, w = shape
    for place in placements:
        coords = place["input_coords"]
        if _overlaps(occ, coords):
            return False
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if not (0 <= rr < h and 0 <= cc < w):
                return False
        occ[coords[:, 0], coords[:, 1]] = True
    return True


def _pick_gap_assignment(
    base_mask: np.ndarray,
    placements: list[dict],
    shape: tuple[int, int],
) -> bool:
    usable = [list(range(1, min(7, place["max_gap"] + 1))) for place in placements]
    if any(not choices for choices in usable):
        return False
    for _ in range(150):
        chosen = []
        used = set()
        for place, choices in zip(placements, usable, strict=True):
            pool = [gap for gap in choices if gap not in used]
            if not pool:
                break
            gap = random.choice(pool)
            chosen.append(gap)
            used.add(gap)
        else:
            for place, gap in zip(placements, chosen, strict=True):
                place["gap"] = gap
                place["input_coords"] = _move_outward(place["final_coords"], place["direction"], gap)
            if not _paths_clear(base_mask, placements, shape):
                continue
            grid = np.zeros(shape, dtype=int)
            grid[base_mask] = GREY
            for place in placements:
                grid[place["input_coords"][:, 0], place["input_coords"][:, 1]] = place["color"]
            objs = _extract_objects(grid)
            base_component = max(_components(grid == GREY), key=lambda pts: int(pts.shape[0]))
            base_dir_mask = _mask_from_coords(shape, base_component)
            distances = []
            for obj in objs:
                _direction, dist = _shortest_direction_and_distance(obj["coords"], base_dir_mask)
                distances.append(int(dist))
            if len(set(distances)) == len(distances):
                return True
    return False


def _alternative_outputs(grid: np.ndarray) -> list[np.ndarray]:
    _, trace = _simulate(grid, return_trace=True)
    reverse_order = [step["index"] for step in reversed(trace)]
    outputs: list[np.ndarray] = []
    variants = [
        dict(swap_mode="grey", dynamic_absorber=True),
        dict(swap_mode="swap", dynamic_absorber=True, order_override=reverse_order),
    ]
    for kwargs in variants:
        try:
            outputs.append(_simulate(grid, **kwargs))
        except Exception:
            continue
    return outputs


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        out, trace = _simulate(grid, return_trace=True)
    except Exception:
        return False
    if len(trace) < 3:
        return False
    if not any(step["surface_nonbase"] for step in trace[1:]):
        return False
    for alt in _alternative_outputs(grid):
        if np.array_equal(out, alt):
            return False
    return True


def _build_attempt(reason_mode: bool = False):
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    grid = np.zeros((h, w), dtype=int)

    base_h = random.randint(3, 4)
    base_w = random.randint(3, 4)
    base_r0 = random.randint(h // 2 - 2, h // 2)
    base_c0 = random.randint(w // 2 - 2, w // 2)
    base_r0 = max(2, min(base_r0, h - base_h - 3))
    base_c0 = max(2, min(base_c0, w - base_w - 3))
    base_mask = np.zeros_like(grid, dtype=bool)
    base_mask[base_r0 : base_r0 + base_h, base_c0 : base_c0 + base_w] = True

    num_objects = random.choice((3, 4))
    start_idx = random.randrange(4)
    step = random.choice((1, -1))
    side_order = [SIDE_CYCLE[(start_idx + step * i) % 4] for i in range(num_objects)]
    colors = list(random.sample(COLOR_POOL, num_objects))

    occupied = base_mask.copy()
    placements: list[dict] = []
    for idx, side in enumerate(side_order):
        direction = SIDE_TO_DIR[side]
        dims = (
            (random.randint(1, 2), random.randint(3, 5))
            if side in ("N", "S")
            else (random.randint(3, 5), random.randint(1, 2))
        )
        candidates = _surface_run_candidates(occupied, side, dims[0], dims[1])
        if not candidates:
            return "no_candidates" if reason_mode else None

        current_bbox = _bbox(np.argwhere(occupied))
        next_side = side_order[idx + 1] if idx + 1 < len(side_order) else None
        overhang_pref: list[dict] = []
        if next_side is not None:
            axis = _overhang_axis(side, next_side)
            if axis is not None:
                key, sign = axis
                if key == "c1":
                    target = current_bbox[3]
                    overhang_pref = [cand for cand in candidates if cand["bbox"][3] > target]
                elif key == "c0":
                    target = current_bbox[1]
                    overhang_pref = [cand for cand in candidates if cand["bbox"][1] < target]
                elif key == "r1":
                    target = current_bbox[2]
                    overhang_pref = [cand for cand in candidates if cand["bbox"][2] > target]
                else:
                    target = current_bbox[0]
                    overhang_pref = [cand for cand in candidates if cand["bbox"][0] < target]
        if overhang_pref:
            candidates = overhang_pref
        if idx >= 1:
            history_pref = [cand for cand in candidates if any(not base_mask[r, c] for r, c in cand["surfaces"])]
            if history_pref:
                candidates = history_pref
            elif idx == 1:
                return "no_history_candidate" if reason_mode else None

        chosen = random.choice(candidates)
        max_gap = _max_gap(chosen["coords"], direction, grid.shape)
        if max_gap < 1:
            return "no_gap_margin" if reason_mode else None
        placements.append(
            {
                "index": idx,
                "side": side,
                "direction": direction,
                "final_coords": chosen["coords"],
                "surfaces": chosen["surfaces"],
                "color": colors[idx],
                "max_gap": max_gap,
            }
        )
        occupied[chosen["coords"][:, 0], chosen["coords"][:, 1]] = True

    if not _pick_gap_assignment(base_mask, placements, grid.shape):
        return "gap_assignment_failed" if reason_mode else None

    grid[base_mask] = GREY
    for place in placements:
        grid[place["input_coords"][:, 0], place["input_coords"][:, 1]] = place["color"]

    if not _semantic_ok(grid):
        return "semantic_failed" if reason_mode else None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
