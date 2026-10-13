from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS))

from arc_utils import Color, find_components, get_bg, in_bounds, label_mask


BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
GREY = Color.GREY
BG = Color.BLACK

DIR_TO_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

BASE_NOTCH_PATCH = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
    ],
    dtype=int,
)

BASE_MAIN_RAY = [(2, 1)]

NOTCH_PATCHES = {
    "down": BASE_NOTCH_PATCH,
    "right": np.rot90(BASE_NOTCH_PATCH, 1),
    "up": np.rot90(BASE_NOTCH_PATCH, 2),
    "left": np.rot90(BASE_NOTCH_PATCH, 3),
}

NOTCH_START_OFFSETS = {
    "up": (-1, 1),
    "down": (2, 1),
    "left": (1, -1),
    "right": (1, 2),
}

PLAIN_PATCHES = [
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1], [1], [1], [1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [0, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
]


def _notch_gap_index(line: np.ndarray) -> int | None:
    idx = np.flatnonzero(line)
    if idx.size < 2:
        return None
    lo = int(idx[0])
    hi = int(idx[-1])
    missing = np.flatnonzero(~line[lo : hi + 1]) + lo
    if missing.size != 1:
        return None
    return int(missing[0])


def _component_notch(grid: np.ndarray, coords: np.ndarray) -> dict[str, object] | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    patch = grid[r0 : r1 + 1, c0 : c1 + 1] == BLUE

    candidates: list[dict[str, object]] = []

    gap = _notch_gap_index(patch[0, :])
    if gap is not None:
        candidates.append(
            {"direction": "up", "start": (r0 - 1, c0 + gap), "step": DIR_TO_STEP["up"]}
        )

    gap = _notch_gap_index(patch[-1, :])
    if gap is not None:
        candidates.append(
            {"direction": "down", "start": (r1 + 1, c0 + gap), "step": DIR_TO_STEP["down"]}
        )

    gap = _notch_gap_index(patch[:, 0])
    if gap is not None:
        candidates.append(
            {"direction": "left", "start": (r0 + gap, c0 - 1), "step": DIR_TO_STEP["left"]}
        )

    gap = _notch_gap_index(patch[:, -1])
    if gap is not None:
        candidates.append(
            {
                "direction": "right",
                "start": (r0 + gap, c1 + 1),
                "step": DIR_TO_STEP["right"],
            }
        )

    if len(candidates) != 1:
        return None
    return candidates[0]


def _blue_components(grid: np.ndarray) -> list[dict]:
    return find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
        return_format="dict",
    )


def _trace_rays(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = np.asarray(grid, dtype=int).copy()
    rays: list[dict[str, object]] = []
    for comp in _blue_components(out):
        notch = _component_notch(out, comp["coords"])
        if notch is None:
            continue

        start_r, start_c = notch["start"]
        dr, dc = notch["step"]
        cells: list[tuple[int, int]] = []
        r = int(start_r)
        c = int(start_c)
        while in_bounds(out.shape, r, c) and int(out[r, c]) == BG:
            out[r, c] = RED
            cells.append((r, c))
            r += int(dr)
            c += int(dc)

        rays.append(
            {
                "direction": str(notch["direction"]),
                "start": tuple(map(int, notch["start"])),
                "step": tuple(map(int, notch["step"])),
                "cells": cells,
            }
        )
    return out, rays


def _enclosed_bg_components(grid: np.ndarray) -> list[np.ndarray]:
    bg_mask = np.asarray(grid, dtype=int) == BG
    labels, n = label_mask(bg_mask, connectivity=4)
    h, w = grid.shape
    enclosed: list[np.ndarray] = []
    for comp_id in range(1, n + 1):
        coords = np.argwhere(labels == comp_id)
        if coords.size == 0:
            continue
        touches_edge = np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == w - 1)
        )
        if not touches_edge:
            enclosed.append(coords)
    return enclosed


def _fill_enclosed_regions(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for coords in _enclosed_bg_components(out):
        out[coords[:, 0], coords[:, 1]] = GREEN
    return out


def _solve_with_details(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    ray_grid, rays = _trace_rays(grid)
    filled = _fill_enclosed_regions(ray_grid)
    return ray_grid, filled, rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire a ray from each blue notch and fill the background pockets that become enclosed by the new barriers.

    Input structure:
    - The background is black.
    - Blue 4-connected compact objects are scattered on the grid; some have exactly one bbox-edge notch and some have no notch.
    - Grey one-cell-thick horizontal or vertical segments act as fixed barriers.

    Rule:
    - For each blue object whose bbox edge contains a single bracketed background gap, use that edge as the notch direction.
    - Start from the background cell immediately outside the gap and extend a red ray cell by cell along that direction until the next non-background cell or the grid edge.
    - Blue objects without a notch do not emit a ray.
    - After all rays are drawn, treat blue cells, grey cells, red rays, and the grid edge as barriers, find every 4-connected background region that cannot reach any boundary cell, and fill those enclosed regions green.
    - Keep every blue object, grey barrier, and red ray unchanged.

    Key insight:
    The notch is a directional cue: it chooses which ray to add, and that added ray changes the topology of the background so enclosed compartments can be detected only afterward.
    """
    _ray_grid, filled, _rays = _solve_with_details(np.asarray(grid, dtype=int))
    return filled


def _rotated_main_patch(inner_h: int, inner_w: int, direction: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    right_col = inner_w + 2
    patch = np.zeros((inner_h + 3, inner_w + 3), dtype=int)

    patch[:2, :3] = BASE_NOTCH_PATCH
    patch[1, 3 : right_col + 1] = GREY
    patch[1 : inner_h + 3, right_col] = GREY
    patch[inner_h + 2, 1 : right_col + 1] = GREY
    patch[patch == 1] = BLUE

    ray_cells = [(r, 1) for r in range(2, inner_h + 2)]
    rotations = {"down": 0, "right": 1, "up": 2, "left": 3}[direction]
    rotated = np.rot90(patch, rotations)

    h0, w0 = patch.shape

    def rotate_coord(coord: tuple[int, int], k: int) -> tuple[int, int]:
        r, c = coord
        hh, ww = h0, w0
        for _ in range(k):
            r, c, hh, ww = ww - 1 - c, r, ww, hh
        return int(r), int(c)

    rotated_ray_cells = [rotate_coord(cell, rotations) for cell in ray_cells]
    return rotated, rotated_ray_cells


def _mark_blue_buffer(buffer: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = buffer.shape
    for r, c in coords:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    buffer[rr, cc] = True


def _place_cells(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    color: int,
    blue_buffer: np.ndarray | None = None,
) -> None:
    for r, c in coords:
        grid[r, c] = color
    if blue_buffer is not None:
        _mark_blue_buffer(blue_buffer, coords)


def _patch_cells(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    local = np.argwhere(patch != 0)
    return [(top + int(r), left + int(c)) for r, c in local]


def _patch_color_cells(
    patch: np.ndarray, top: int, left: int, color: int
) -> list[tuple[int, int]]:
    local = np.argwhere(patch == color)
    return [(top + int(r), left + int(c)) for r, c in local]


def _can_place_blue_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    blue_buffer: np.ndarray,
    ray_reserved: np.ndarray,
) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    for r, c in _patch_color_cells(patch, top, left, BLUE):
        if grid[r, c] != BG or blue_buffer[r, c] or ray_reserved[r, c]:
            return False
    return True


def _place_main_patch(
    grid: np.ndarray, blue_buffer: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    inner_h = random.randint(2, 4)
    inner_w = random.randint(3, 5)
    direction = random.choice(["up", "down", "left", "right"])
    patch, ray_cells = _rotated_main_patch(inner_h, inner_w, direction)
    ph, pw = patch.shape

    row_choices = list(range(1, max(2, grid.shape[0] - ph)))
    col_choices = list(range(1, max(2, grid.shape[1] - pw)))
    random.shuffle(row_choices)
    random.shuffle(col_choices)

    for top in row_choices:
        for left in col_choices:
            all_cells = _patch_cells(patch, top, left)
            if any(grid[r, c] != BG for r, c in all_cells):
                continue
            blue_cells = _patch_color_cells(patch, top, left, BLUE)
            if any(blue_buffer[r, c] for r, c in blue_cells):
                continue
            for r, c in all_cells:
                grid[r, c] = int(patch[r - top, c - left])
            _mark_blue_buffer(blue_buffer, blue_cells)

            ray_reserved = np.zeros_like(grid, dtype=bool)
            for rr, cc in ray_cells:
                ray_reserved[top + rr, left + cc] = True
            return grid, ray_reserved
    return None


def _random_plain_patch() -> np.ndarray:
    return random.choice(PLAIN_PATCHES)


def _place_plain_object(
    grid: np.ndarray,
    blue_buffer: np.ndarray,
    ray_reserved: np.ndarray,
) -> bool:
    patch = _random_plain_patch()
    ph, pw = patch.shape
    candidates = [
        (r, c)
        for r in range(0, grid.shape[0] - ph + 1)
        for c in range(0, grid.shape[1] - pw + 1)
    ]
    random.shuffle(candidates)
    for top, left in candidates:
        if not _can_place_blue_patch(grid, patch, top, left, blue_buffer, ray_reserved):
            continue
        blue_cells = _patch_color_cells(patch, top, left, 1)
        _place_cells(grid, blue_cells, BLUE, blue_buffer)
        return True
    return False


def _ray_path_to_edge(
    grid: np.ndarray,
    start: tuple[int, int],
    step: tuple[int, int],
    ray_reserved: np.ndarray,
) -> list[tuple[int, int]] | None:
    r, c = start
    dr, dc = step
    path: list[tuple[int, int]] = []
    while in_bounds(grid.shape, r, c):
        if grid[r, c] != BG or ray_reserved[r, c]:
            return None
        path.append((r, c))
        r += int(dr)
        c += int(dc)
    if len(path) < 2:
        return None
    return path


def _place_extra_notch(
    grid: np.ndarray,
    blue_buffer: np.ndarray,
    ray_reserved: np.ndarray,
) -> bool:
    directions = ["up", "down", "left", "right"]
    random.shuffle(directions)
    for direction in directions:
        patch = NOTCH_PATCHES[direction]
        ph, pw = patch.shape
        candidates = [
            (r, c)
            for r in range(0, grid.shape[0] - ph + 1)
            for c in range(0, grid.shape[1] - pw + 1)
        ]
        random.shuffle(candidates)
        for top, left in candidates:
            if not _can_place_blue_patch(grid, patch, top, left, blue_buffer, ray_reserved):
                continue
            off_r, off_c = NOTCH_START_OFFSETS[direction]
            start = (top + off_r, left + off_c)
            if not in_bounds(grid.shape, start[0], start[1]):
                continue
            path = _ray_path_to_edge(grid, start, DIR_TO_STEP[direction], ray_reserved)
            if path is None:
                continue
            blue_cells = _patch_color_cells(patch, top, left, 1)
            _place_cells(grid, blue_cells, BLUE, blue_buffer)
            for r, c in path:
                ray_reserved[r, c] = True
            return True
    return False


def _can_place_grey_segment(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    ray_reserved: np.ndarray,
    blue_buffer: np.ndarray,
) -> bool:
    for r, c in cells:
        if grid[r, c] != BG or ray_reserved[r, c] or blue_buffer[r, c]:
            return False
    return True


def _place_grey_segments(
    grid: np.ndarray,
    ray_reserved: np.ndarray,
    blue_buffer: np.ndarray,
) -> None:
    segment_count = random.randint(1, 3)
    for _ in range(segment_count):
        placed = False
        for _attempt in range(120):
            horizontal = random.choice([True, False])
            length = random.randint(3, 6)
            if horizontal:
                r = random.randrange(grid.shape[0])
                c = random.randrange(grid.shape[1] - length + 1)
                cells = [(r, c + k) for k in range(length)]
            else:
                r = random.randrange(grid.shape[0] - length + 1)
                c = random.randrange(grid.shape[1])
                cells = [(r + k, c) for k in range(length)]
            if not _can_place_grey_segment(grid, cells, ray_reserved, blue_buffer):
                continue
            _place_cells(grid, cells, GREY)
            placed = True
            break
        if not placed:
            return


def _blue_spacing_ok(grid: np.ndarray) -> bool:
    components = [comp["coords"] for comp in _blue_components(grid)]
    for comp in components:
        area = int(comp.shape[0])
        if not (3 <= area <= 6):
            return False
    for i, coords_a in enumerate(components):
        for coords_b in components[i + 1 :]:
            for r0, c0 in coords_a:
                for r1, c1 in coords_b:
                    if max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1))) <= 1:
                        return False
    return True


def _validate_generated(grid: np.ndarray) -> bool:
    comps = _blue_components(grid)
    if not (3 <= len(comps) <= 5):
        return False
    notch_count = 0
    plain_count = 0
    for comp in comps:
        if _component_notch(grid, comp["coords"]) is None:
            plain_count += 1
        else:
            notch_count += 1
    if notch_count < 1 or plain_count < 1:
        return False
    if not _blue_spacing_ok(grid):
        return False

    enclosed_before = {
        (int(r), int(c))
        for coords in _enclosed_bg_components(grid)
        for r, c in coords
    }
    ray_grid, out, rays = _solve_with_details(grid)
    if not rays or not any(ray["cells"] for ray in rays):
        return False
    if not np.any(ray_grid == RED):
        return False
    if not np.any(out == GREEN):
        return False
    enclosed_after = {
        (int(r), int(c))
        for coords in _enclosed_bg_components(ray_grid)
        for r, c in coords
    }
    if not (enclosed_after - enclosed_before):
        return False
    if np.array_equal(grid, out):
        return False
    return True


def _generate_once() -> np.ndarray | None:
    h = random.randint(11, 15)
    w = random.randint(13, 18)
    grid = np.full((h, w), BG, dtype=int)
    blue_buffer = np.zeros((h, w), dtype=bool)

    placed = _place_main_patch(grid, blue_buffer)
    if placed is None:
        return None
    grid, ray_reserved = placed

    total_blue = random.randint(3, 5)
    max_extra_notches = min(2, total_blue - 2)
    extra_notches = random.choices(
        population=list(range(max_extra_notches + 1)),
        weights=[3, 3, 1][: max_extra_notches + 1],
        k=1,
    )[0]
    plain_needed = total_blue - 1 - extra_notches
    if plain_needed < 1:
        return None

    for _ in range(extra_notches):
        if not _place_extra_notch(grid, blue_buffer, ray_reserved):
            return None
    for _ in range(plain_needed):
        if not _place_plain_object(grid, blue_buffer, ray_reserved):
            return None

    _place_grey_segments(grid, ray_reserved, blue_buffer)

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _attempt in range(400):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid notch-ray enclosure puzzle")
