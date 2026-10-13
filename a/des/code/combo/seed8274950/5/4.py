from __future__ import annotations

import random

import numpy as np


GREY = 5
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
BAR_COLORS = {
    "up": BLUE,
    "down": RED,
    "left": GREEN,
    "right": YELLOW,
}
ORIENTATIONS = ("up", "down", "left", "right")

GEN_RETRIES = 400
GRID_H_RANGE = (10, 22)
GRID_W_RANGE = (10, 22)
OBJECT_COUNT_RANGE = (1, 4)
STEM_LEN_RANGE = (2, 5)
BG_CHOICES = (0, 6, 7, 8, 9)
PLACEMENT_PAD = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _neighbor_count(coords: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in coords for dr, dc in DIRS.values())


def _perpendicular_dirs(direction: str) -> tuple[str, str]:
    if direction in ("up", "down"):
        return ("left", "right")
    return ("up", "down")


def _parse_t_shape(coords: list[tuple[int, int]]) -> dict:
    pts = {tuple(map(int, p)) for p in coords}
    assert len(pts) == len(coords), "component must not repeat coordinates"

    degrees = {cell: _neighbor_count(pts, cell) for cell in pts}
    centers = [cell for cell, degree in degrees.items() if degree == 3]
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(centers) == 1, "expected one T junction"
    assert len(endpoints) == 3, "expected three T endpoints"
    assert all(degree in (1, 2, 3) for degree in degrees.values()), "T pixels cannot have isolated holes or extra branches"

    center = centers[0]
    present_dirs = [
        name
        for name, (dr, dc) in DIRS.items()
        if (center[0] + dr, center[1] + dc) in pts
    ]
    assert len(present_dirs) == 3, "junction must connect in exactly three directions"

    bar_side = next(name for name in ORIENTATIONS if name not in present_dirs)
    stem_dir = OPPOSITE[bar_side]
    cap_dirs = tuple(name for name in present_dirs if name != stem_dir)
    assert set(cap_dirs) == set(_perpendicular_dirs(stem_dir)), "cap must be perpendicular to stem"

    cap_cells = [center]
    for name in cap_dirs:
        dr, dc = DIRS[name]
        arm = (center[0] + dr, center[1] + dc)
        assert degrees[arm] == 1, "cap arms must end immediately"
        cap_cells.append(arm)

    dr, dc = DIRS[stem_dir]
    stem_cells = [center]
    cur = center
    while (cur[0] + dr, cur[1] + dc) in pts:
        nxt = (cur[0] + dr, cur[1] + dc)
        stem_cells.append(nxt)
        cur = nxt

    assert len(stem_cells) >= 2, "stem must extend beyond the cap center"
    assert degrees[stem_cells[-1]] == 1, "stem tail must terminate"
    for cell in stem_cells[1:-1]:
        assert degrees[cell] == 2, "interior stem pixels must form a straight run"

    expected = set(cap_cells)
    for cell in stem_cells:
        expected.add(cell)
    assert expected == pts, "component must be exactly a 3-pixel cap plus one stem"

    return {
        "center": center,
        "bar_side": bar_side,
        "stem_dir": stem_dir,
        "cap_cells": cap_cells,
        "tail": stem_cells[-1],
        "stem_len": len(stem_cells),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey T cap by orientation and extend its stem by one equal-length grey ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    out = g.copy()
    comps = _components(g != bg)
    assert comps, "expected at least one foreground T"

    for comp in comps:
        info = _parse_t_shape(comp)
        color = BAR_COLORS[info["bar_side"]]
        for r, c in info["cap_cells"]:
            out[r, c] = int(color)

        dr, dc = DIRS[info["stem_dir"]]
        tail_r, tail_c = info["tail"]
        for step in range(1, int(info["stem_len"]) + 1):
            nr = tail_r + dr * step
            nc = tail_c + dc * step
            assert 0 <= nr < h and 0 <= nc < w, "extended ray must stay inside the grid"
            assert int(out[nr, nc]) == int(bg), "extended rays must not overlap other objects"
            out[nr, nc] = GREY

    return out


def _output_bbox_dims(bar_side: str, stem_len: int) -> tuple[int, int]:
    if bar_side in ("up", "down"):
        return 2 * int(stem_len), 3
    return 3, 2 * int(stem_len)


def _input_coords_in_output_box(bar_side: str, stem_len: int) -> list[tuple[int, int]]:
    stem_len = int(stem_len)
    coords: set[tuple[int, int]] = set()
    if bar_side == "up":
        coords.update({(0, 0), (0, 1), (0, 2)})
        coords.update((r, 1) for r in range(stem_len))
    elif bar_side == "down":
        base = 2 * stem_len - 1
        coords.update({(base, 0), (base, 1), (base, 2)})
        coords.update((base - r, 1) for r in range(stem_len))
    elif bar_side == "left":
        coords.update({(0, 0), (1, 0), (2, 0)})
        coords.update((1, c) for c in range(stem_len))
    else:
        base = 2 * stem_len - 1
        coords.update({(0, base), (1, base), (2, base)})
        coords.update((1, base - c) for c in range(stem_len))
    return sorted(coords)


def _placeable(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - PLACEMENT_PAD)
    c0 = max(0, left - PLACEMENT_PAD)
    r1 = min(h, top + height + PLACEMENT_PAD)
    c1 = min(w, left + width + PLACEMENT_PAD)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - PLACEMENT_PAD)
    c0 = max(0, left - PLACEMENT_PAD)
    r1 = min(h, top + height + PLACEMENT_PAD)
    c1 = min(w, left + width + PLACEMENT_PAD)
    occupied[r0:r1, c0:c1] = True


def _sample_grid_shape(specs: list[tuple[str, int]]) -> tuple[int, int]:
    min_h = max(_output_bbox_dims(bar_side, stem_len)[0] for bar_side, stem_len in specs) + 2
    min_w = max(_output_bbox_dims(bar_side, stem_len)[1] for bar_side, stem_len in specs) + 2
    min_area = sum(
        (_output_bbox_dims(bar_side, stem_len)[0] + 2 * PLACEMENT_PAD)
        * (_output_bbox_dims(bar_side, stem_len)[1] + 2 * PLACEMENT_PAD)
        for bar_side, stem_len in specs
    )
    h_lo = max(GRID_H_RANGE[0], min_h)
    w_lo = max(GRID_W_RANGE[0], min_w)
    h_hi = GRID_H_RANGE[1]
    w_hi = GRID_W_RANGE[1]
    assert h_lo <= h_hi and w_lo <= w_hi, "requested T cannot fit the configured grid range"

    for _ in range(120):
        h = random.randint(h_lo, h_hi)
        w = random.randint(w_lo, w_hi)
        if h * w >= min_area:
            return h, w
    return h_hi, w_hi


def _generate_once() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    orientations = random.sample(ORIENTATIONS, count)
    specs = [(bar_side, random.randint(*STEM_LEN_RANGE)) for bar_side in orientations]
    h, w = _sample_grid_shape(specs)
    bg = random.choice(BG_CHOICES)

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    placements: list[tuple[int, int, str, int]] = []
    for bar_side, stem_len in sorted(
        specs,
        key=lambda item: _output_bbox_dims(item[0], item[1])[0] * _output_bbox_dims(item[0], item[1])[1],
        reverse=True,
    ):
        box_h, box_w = _output_bbox_dims(bar_side, stem_len)
        placed = False
        for _ in range(200):
            top = random.randint(0, h - box_h)
            left = random.randint(0, w - box_w)
            if not _placeable(occupied, top, left, box_h, box_w):
                continue
            _mark_occupied(occupied, top, left, box_h, box_w)
            placements.append((top, left, bar_side, stem_len))
            placed = True
            break
        if not placed:
            return None

    for top, left, bar_side, stem_len in placements:
        for dr, dc in _input_coords_in_output_box(bar_side, stem_len):
            grid[top + dr, left + dc] = GREY

    solved = main(grid)
    assert isinstance(solved, np.ndarray)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid T-ray puzzle")
