from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GENERATE_RETRIES = 400
PLACEMENT_RETRIES = 200
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
ARROW_COUNT_RANGE = (3, 7)
PAIR_RAY_LEN_RANGE = (2, 7)
EDGE_RAY_MIN = 2

DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_OFFSETS = {
    "up": [(0, 0), (1, -1), (1, 1)],
    "down": [(0, 0), (-1, -1), (-1, 1)],
    "left": [(0, 0), (-1, 1), (1, 1)],
    "right": [(0, 0), (-1, -1), (1, -1)],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _arrow_cells(head: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    hr, hc = head
    return [(hr + dr, hc + dc) for dr, dc in ARROW_OFFSETS[direction]]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if (
                        _in_bounds((h, w), nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append({"color": color, "coords": coords})
    return components


def _detect_arrow(component: dict) -> dict:
    coords = set(component["coords"])
    assert len(coords) == 3, "every foreground component must contain exactly three pixels"
    for head in coords:
        for direction in DIR_STEPS:
            if set(_arrow_cells(head, direction)) == coords:
                arrow = {
                    "color": int(component["color"]),
                    "coords": sorted(coords),
                    "coord_set": coords,
                    "head": tuple(map(int, head)),
                    "direction": direction,
                    "step": DIR_STEPS[direction],
                }
                return arrow
    raise AssertionError("foreground component is not a valid three-pixel arrow")


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    arrows = [_detect_arrow(comp) for comp in _find_components(g)]
    assert arrows, "expected at least one arrow"

    cell_to_arrow: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow["coords"]:
            if cell in cell_to_arrow:
                raise AssertionError("arrows must not overlap")
            cell_to_arrow[cell] = idx

    h, w = g.shape
    rays: list[dict] = []
    for idx, arrow in enumerate(arrows):
        dr, dc = arrow["step"]
        r, c = arrow["head"][0] + dr, arrow["head"][1] + dc
        path: list[tuple[int, int]] = []
        while _in_bounds((h, w), r, c) and (r, c) not in cell_to_arrow:
            path.append((r, c))
            r += dr
            c += dc
        hit_idx = cell_to_arrow.get((r, c)) if _in_bounds((h, w), r, c) else None
        rays.append(
            {
                "source": idx,
                "path": path,
                "hit": hit_idx,
                "branch": "capture" if hit_idx is not None else "edge",
            }
        )
    return arrows, rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot a colored ray from each tiny arrow and recolor the captured ray tip by the arrow it hits.

    Input structure:
    - The background is black.
    - Every foreground object is a monochrome three-pixel arrow: one head pixel and two tail pixels.
    - The tail pixels lie on a line perpendicular to the arrow direction, so the shape itself encodes whether the arrow points up, down, left, or right.

    Rule:
    - Keep every arrow body unchanged.
    - From each arrow head, extend a straight ray in the encoded direction through background cells only.
    - The ray stops when the next step would leave the grid or would enter any pixel of another arrow.
    - If the ray stops because it reaches another arrow, recolor the ray's last drawn pixel to the hit arrow's color to mark that capture.

    Key insight:
    - The three-pixel shape is not just decoration: its asymmetric head-versus-tail layout determines the ray direction, and the final ray pixel changes color only when another arrow terminates the line.
    """
    g = np.asarray(grid, dtype=int)
    arrows, rays = _analyze_scene(g)

    paint: dict[tuple[int, int], int] = {}
    for ray in rays:
        src = arrows[ray["source"]]
        for cell in ray["path"]:
            prev = paint.get(cell)
            if prev is not None and prev != src["color"]:
                raise AssertionError("overlapping rays produce an ambiguous output")
            paint[cell] = src["color"]
        if ray["hit"] is not None and ray["path"]:
            captured_color = arrows[ray["hit"]]["color"]
            tip = ray["path"][-1]
            prev = paint.get(tip)
            if prev is not None and prev not in (src["color"], captured_color):
                raise AssertionError("captured ray tip conflicts with another ray")
            paint[tip] = captured_color

    out = g.copy()
    for (r, c), color in paint.items():
        out[r, c] = color
    return out


def _ray_cells_to_edge(head: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = DIR_STEPS[direction]
    r, c = head[0] + dr, head[1] + dc
    path: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        path.append((r, c))
        r += dr
        c += dc
    return path


def _padded(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if _in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def _can_place_arrow(
    cells: list[tuple[int, int]],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    clearance: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
) -> bool:
    cell_set = set(cells)
    return (
        all(_in_bounds(shape, r, c) for r, c in cells)
        and not (cell_set & occupied)
        and not (cell_set & clearance)
        and not (cell_set & reserved_rays)
    )


def _can_reserve_ray(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
    extra_forbidden: set[tuple[int, int]],
) -> bool:
    path_set = set(path)
    return not (path_set & occupied) and not (path_set & reserved_rays) and not (path_set & extra_forbidden)


def _sample_single(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    clearance: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    directions = list(DIR_STEPS)
    random.shuffle(directions)
    for direction in directions:
        for _ in range(PLACEMENT_RETRIES):
            if direction == "right":
                row = random.randint(1, h - 2)
                col_hi = w - 1 - EDGE_RAY_MIN
                if col_hi < 1:
                    break
                head = (row, random.randint(1, col_hi))
            elif direction == "left":
                row = random.randint(1, h - 2)
                col_lo = EDGE_RAY_MIN
                col_hi = w - 2
                if col_lo > col_hi:
                    break
                head = (row, random.randint(col_lo, col_hi))
            elif direction == "down":
                row_hi = h - 1 - EDGE_RAY_MIN
                if row_hi < 1:
                    break
                head = (random.randint(1, row_hi), random.randint(1, w - 2))
            else:
                row_lo = EDGE_RAY_MIN
                row_hi = h - 2
                if row_lo > row_hi:
                    break
                head = (random.randint(row_lo, row_hi), random.randint(1, w - 2))

            cells = _arrow_cells(head, direction)
            if not _can_place_arrow(cells, shape, occupied, clearance, reserved_rays):
                continue

            ray = _ray_cells_to_edge(head, direction, shape)
            if len(ray) < EDGE_RAY_MIN:
                continue
            if not _can_reserve_ray(ray, occupied, reserved_rays, set(cells)):
                continue
            return {"head": head, "direction": direction, "cells": cells, "ray": ray}
    return None


def _sample_pair(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    clearance: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
) -> tuple[dict, dict] | None:
    h, w = shape
    directions = list(DIR_STEPS)
    random.shuffle(directions)
    for direction in directions:
        for _ in range(PLACEMENT_RETRIES):
            path_len = random.randint(*PAIR_RAY_LEN_RANGE)
            if direction == "right":
                row = random.randint(1, h - 2)
                src_max = w - EDGE_RAY_MIN - path_len - 2
                if src_max < 1:
                    break
                src_head = (row, random.randint(1, src_max))
                tgt_head = (row, src_head[1] + path_len + 1)
            elif direction == "left":
                row = random.randint(1, h - 2)
                tgt_lo = EDGE_RAY_MIN
                tgt_hi = w - path_len - 3
                if tgt_lo > tgt_hi:
                    break
                tgt_head = (row, random.randint(tgt_lo, tgt_hi))
                src_head = (row, tgt_head[1] + path_len + 1)
            elif direction == "down":
                col = random.randint(1, w - 2)
                src_max = h - EDGE_RAY_MIN - path_len - 2
                if src_max < 1:
                    break
                src_head = (random.randint(1, src_max), col)
                tgt_head = (src_head[0] + path_len + 1, col)
            else:
                col = random.randint(1, w - 2)
                tgt_lo = EDGE_RAY_MIN
                tgt_hi = h - path_len - 3
                if tgt_lo > tgt_hi:
                    break
                tgt_head = (random.randint(tgt_lo, tgt_hi), col)
                src_head = (tgt_head[0] + path_len + 1, col)

            src_cells = _arrow_cells(src_head, direction)
            tgt_cells = _arrow_cells(tgt_head, direction)
            combined = set(src_cells) | set(tgt_cells)
            if len(combined) != 6:
                continue
            if not _can_place_arrow(src_cells, shape, occupied, clearance, reserved_rays):
                continue
            if not _can_place_arrow(tgt_cells, shape, occupied, clearance, reserved_rays):
                continue
            if set(src_cells) & _padded(tgt_cells, shape):
                continue
            if set(tgt_cells) & _padded(src_cells, shape):
                continue

            dr, dc = DIR_STEPS[direction]
            capture_ray = [(src_head[0] + dr * step, src_head[1] + dc * step) for step in range(1, path_len + 1)]
            target_ray = _ray_cells_to_edge(tgt_head, direction, shape)
            if len(target_ray) < EDGE_RAY_MIN:
                continue
            if not _can_reserve_ray(capture_ray, occupied, reserved_rays, combined):
                continue
            if not _can_reserve_ray(target_ray, occupied, reserved_rays, combined):
                continue
            if set(capture_ray) & set(target_ray):
                continue

            return (
                {"head": src_head, "direction": direction, "cells": src_cells, "ray": capture_ray},
                {"head": tgt_head, "direction": direction, "cells": tgt_cells, "ray": target_ray},
            )
    return None


def _register_arrow(
    arrow: dict,
    occupied: set[tuple[int, int]],
    clearance: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
    shape: tuple[int, int],
) -> None:
    occupied.update(arrow["cells"])
    clearance.update(_padded(arrow["cells"], shape))
    reserved_rays.update(arrow["ray"])


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)
    arrow_count = random.randint(*ARROW_COUNT_RANGE)
    pair_count = random.randint(1, arrow_count // 2)
    single_count = arrow_count - 2 * pair_count

    occupied: set[tuple[int, int]] = set()
    clearance: set[tuple[int, int]] = set()
    reserved_rays: set[tuple[int, int]] = set()
    arrows: list[dict] = []

    for _ in range(pair_count):
        pair = _sample_pair(shape, occupied, clearance, reserved_rays)
        if pair is None:
            return None
        for arrow in pair:
            arrows.append(arrow)
            _register_arrow(arrow, occupied, clearance, reserved_rays, shape)

    for _ in range(single_count):
        arrow = _sample_single(shape, occupied, clearance, reserved_rays)
        if arrow is None:
            return None
        arrows.append(arrow)
        _register_arrow(arrow, occupied, clearance, reserved_rays, shape)

    random.shuffle(arrows)
    colors = random.sample(PALETTE, len(arrows))
    grid = np.zeros(shape, dtype=int)
    for arrow, color in zip(arrows, colors):
        for r, c in arrow["cells"]:
            grid[r, c] = int(color)

    detected_arrows, rays = _analyze_scene(grid)
    if len(detected_arrows) != len(arrows):
        return None
    branch_counts = {"capture": 0, "edge": 0}
    painted: set[tuple[int, int]] = set()
    for ray in rays:
        if not ray["path"]:
            return None
        branch_counts[ray["branch"]] += 1
        for cell in ray["path"]:
            if cell in painted:
                return None
            painted.add(cell)
    if branch_counts["capture"] != pair_count or branch_counts["edge"] != len(arrows) - pair_count:
        return None
    if branch_counts["capture"] == 0 or branch_counts["edge"] == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-ray input")
