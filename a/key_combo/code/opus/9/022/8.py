from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

SEED_COLOR = PINK
OBSTACLE_COLOR = GREY

SPIRAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SAFE_MASKS = [
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
]
BODY_COLORS = [BLUE, YELLOW, ORANGE, TEAL, MAROON]

GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors(r: int, c: int, connectivity: int) -> Iterable[tuple[int, int]]:
    if connectivity == 4:
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1
        return
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    out[rr, cc] = grid[coords[:, 0], coords[:, 1]]
    return out


def _extract_sprite(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid != BG, connectivity=8)
    assert comps, "expected foreground"
    areas = [int(comp.shape[0]) for comp in comps]
    largest = max(areas)
    largest_comps = [comp for comp, area in zip(comps, areas) if area == largest]
    assert len(largest_comps) == 1, "expected unique largest foreground component"
    return _crop_component(grid, largest_comps[0])


def _marker_color(sprite: np.ndarray) -> int:
    if np.count_nonzero(sprite == RED) == 1:
        return RED
    assert np.count_nonzero(sprite == GREEN) == 1, "expected exactly one green marker when red is absent"
    return GREEN


def _find_seed(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == SEED_COLOR)
    assert coords.shape == (1, 2), "expected exactly one seed"
    return int(coords[0, 0]), int(coords[0, 1])


def _spiral_corners(seed: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = seed
    corners: list[tuple[int, int]] = []
    leg_length = 2
    direction_index = 0
    consecutive_oob = 0
    while True:
        dr, dc = SPIRAL_DIRS[direction_index]
        r += dr * leg_length
        c += dc * leg_length
        corners.append((r, c))
        if _in_bounds(shape, r, c):
            consecutive_oob = 0
        else:
            consecutive_oob += 1
            if consecutive_oob >= 2:
                break
        direction_index = (direction_index + 1) % 4
        if direction_index in (0, 2):
            leg_length += 2
    return corners


def _stamp_cells(sprite: np.ndarray, center: tuple[int, int]) -> list[tuple[int, int, int]]:
    h, w = sprite.shape
    anchor_r = h // 2
    anchor_c = w // 2
    cr, cc = center
    cells: list[tuple[int, int, int]] = []
    for sr, sc in np.argwhere(sprite != BG):
        rr = cr + int(sr) - anchor_r
        cc2 = cc + int(sc) - anchor_c
        cells.append((rr, cc2, int(sprite[sr, sc])))
    return cells


def _stamp_coord_set(sprite: np.ndarray, center: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r, c, color in _stamp_cells(sprite, center)
        if color != BG and _in_bounds(shape, r, c)
    }


def _full_fit(sprite: np.ndarray, center: tuple[int, int], shape: tuple[int, int]) -> bool:
    return all(_in_bounds(shape, r, c) for r, c, _ in _stamp_cells(sprite, center))


def _hits_obstacle(sprite: np.ndarray, center: tuple[int, int], grid: np.ndarray) -> bool:
    for r, c, color in _stamp_cells(sprite, center):
        if color != BG and _in_bounds(grid.shape, r, c) and int(grid[r, c]) == OBSTACLE_COLOR:
            return True
    return False


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, center: tuple[int, int]) -> None:
    for r, c, color in _stamp_cells(sprite, center):
        if color != BG and _in_bounds(grid.shape, r, c):
            grid[r, c] = color


def _analyze_instance(grid: np.ndarray) -> dict:
    sprite = _extract_sprite(grid)
    marker = _marker_color(sprite)
    seed = _find_seed(grid)
    corners = _spiral_corners(seed, grid.shape)
    events = []
    for idx, center in enumerate(corners, start=1):
        mirrored = marker == RED and idx % 2 == 0
        stamp = np.fliplr(sprite) if mirrored else sprite
        in_bounds_corner = _in_bounds(grid.shape, center[0], center[1])
        full_fit = _full_fit(stamp, center, grid.shape)
        if not in_bounds_corner:
            status = "oob"
        elif _hits_obstacle(stamp, center, grid):
            status = "obstacle"
        else:
            status = "placed"
        events.append(
            {
                "index": idx,
                "center": center,
                "mirrored": mirrored,
                "status": status,
                "full_fit": full_fit,
                "cells": _stamp_coord_set(stamp, center, grid.shape),
            }
        )
    return {"sprite": sprite, "marker": marker, "seed": seed, "corners": corners, "events": events}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the corner positions of a square spiral, mirroring the sprite only at even corners when its marker is red.

    Input structure:
    - A black background contains one compact multicolor sprite in a corner area, one magenta seed pixel, and 0-2 grey obstacle pixels.
    - The sprite is the unique largest foreground component and contains exactly one red or green marker pixel.
    - The seed is isolated from the sprite, and obstacles are isolated background interruptions.

    Rule:
    - Extract the largest foreground component as the sprite and note whether its unique marker is red or green.
    - From the magenta seed, follow the unbounded spiral corner sequence produced by legs up, right, down, left with lengths 2, 2, 4, 4, 6, 6, ...
    - For each numbered corner, skip it if the corner itself is out of bounds; otherwise stamp the sprite with its center aligned to that corner, but skip the whole stamp if any sprite cell would land on a grey obstacle.
    - If the marker is red, use a left-right mirrored sprite at even-numbered corners only; if the marker is green, never mirror. Stop after two consecutive corners are both out of bounds. Keep the original sprite and the magenta seed.

    Key insight:
    - The spiral does not draw a path: it only supplies ordered corner coordinates, and the sprite's marker color globally decides whether even corner numbers flip the stamp.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sprite = _extract_sprite(g)
    marker = _marker_color(sprite)
    seed = _find_seed(g)

    for idx, center in enumerate(_spiral_corners(seed, g.shape), start=1):
        if not _in_bounds(g.shape, center[0], center[1]):
            continue
        stamp = np.fliplr(sprite) if marker == RED and idx % 2 == 0 else sprite
        if _hits_obstacle(stamp, center, g):
            continue
        _paste_sprite(out, stamp, center)

    out[g == OBSTACLE_COLOR] = OBSTACLE_COLOR
    out[g == SEED_COLOR] = SEED_COLOR
    return out


def _make_sprite(marker_color: int) -> np.ndarray:
    mask = random.choice(SAFE_MASKS)
    sprite = np.zeros_like(mask, dtype=int)
    occupied = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    left_column = [(r, c) for r, c in occupied if c == 0]
    marker_cell = random.choice(left_column)
    other_cells = [cell for cell in occupied if cell != marker_cell]

    if marker_color == RED:
        extra_count = random.choice([0, 1])
        palette = [GREEN] + random.sample(BODY_COLORS, extra_count)
    else:
        palette = random.sample(BODY_COLORS, random.choice([1, 2]))

    random.shuffle(other_cells)
    assignments: list[int] = []
    for color in palette:
        assignments.append(color)
    while len(assignments) < len(other_cells):
        assignments.append(random.choice(palette))
    random.shuffle(assignments)

    sprite[marker_cell] = marker_color
    for (r, c), color in zip(other_cells, assignments):
        sprite[r, c] = color

    assert not np.array_equal(sprite != BG, np.fliplr(sprite != BG))
    assert not np.array_equal(sprite, np.fliplr(sprite))
    return sprite


def _choose_seed(shape: tuple[int, int], sprite: np.ndarray) -> tuple[int, int] | None:
    h, w = shape
    candidates = []
    for r in range(6, h - 6):
        for c in range(6, w - 6):
            corners = _spiral_corners((r, c), shape)
            if len(corners) < 6:
                continue
            in_bounds_corners = [(rr, cc) for rr, cc in corners if _in_bounds(shape, rr, cc)]
            if len(in_bounds_corners) < 5:
                continue
            if not all(_full_fit(sprite, center, shape) for center in in_bounds_corners):
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _corner_positions(shape: tuple[int, int], corner_name: str) -> list[tuple[int, int]]:
    h, w = shape
    if corner_name == "tl":
        return [(r, c) for r in range(0, 3) for c in range(0, 3)]
    if corner_name == "tr":
        return [(r, c) for r in range(0, 3) for c in range(w - 3, w)]
    if corner_name == "bl":
        return [(r, c) for r in range(h - 3, h) for c in range(0, 3)]
    return [(r, c) for r in range(h - 3, h) for c in range(w - 3, w)]


def _place_original_sprite(grid: np.ndarray, sprite: np.ndarray, forbidden: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    choices = ["tl", "tr", "bl", "br"]
    random.shuffle(choices)
    for name in choices:
        positions = _corner_positions(grid.shape, name)
        random.shuffle(positions)
        for r0, c0 in positions:
            if r0 + sprite.shape[0] > h or c0 + sprite.shape[1] > w:
                continue
            cells = {
                (r0 + int(sr), c0 + int(sc))
                for sr, sc in np.argwhere(sprite != BG)
            }
            if cells & forbidden:
                continue
            for sr, sc in np.argwhere(sprite != BG):
                grid[r0 + int(sr), c0 + int(sc)] = int(sprite[sr, sc])
            return True
    return False


def _first_nonzero_cell(sprite: np.ndarray, center: tuple[int, int]) -> tuple[int, int]:
    for r, c, color in _stamp_cells(sprite, center):
        if color != BG:
            return r, c
    raise AssertionError("sprite must contain foreground")


def _safe_scatter_cell(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in occupied or (r, c) in protected:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in protected):
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _build_case() -> np.ndarray | None:
    n = random.randint(18, 24)
    grid = np.zeros((n, n), dtype=int)
    marker_color = random.choice([RED, GREEN])
    sprite = _make_sprite(marker_color)
    seed = _choose_seed(grid.shape, sprite)
    if seed is None:
        return None

    corners = _spiral_corners(seed, grid.shape)
    actual_stamps = []
    reserved = {seed}
    for idx, center in enumerate(corners, start=1):
        stamp = np.fliplr(sprite) if marker_color == RED and idx % 2 == 0 else sprite
        if _in_bounds(grid.shape, center[0], center[1]) and _full_fit(stamp, center, grid.shape):
            cells = _stamp_coord_set(stamp, center, grid.shape)
            actual_stamps.append((idx, center, stamp, cells))
            reserved |= cells

    if len(actual_stamps) < 5:
        return None

    first_pair = actual_stamps[0][3] & actual_stamps[1][3]
    if marker_color == GREEN and first_pair:
        return None

    if not _place_original_sprite(grid, sprite, reserved):
        return None

    grid[seed] = SEED_COLOR
    protected = {tuple(map(int, rc)) for rc in np.argwhere(grid != BG)}

    need_blocking = marker_color == RED or random.random() < 0.45
    obstacle_targets: list[tuple[int, int]] = []

    if need_blocking:
        preferred_indices = [2] if marker_color == RED else [2, 3, 4, 5]
        candidate_events = [(idx, center, stamp) for idx, center, stamp, _cells in actual_stamps if idx in preferred_indices]
        if not candidate_events:
            if marker_color == RED:
                return None
            candidate_events = [(idx, center, stamp) for idx, center, stamp, _cells in actual_stamps[:4]]
        if not candidate_events:
            return None
        target_idx, target_center, target_stamp = random.choice(candidate_events)
        obstacle_targets.append(_first_nonzero_cell(target_stamp, target_center))

        if random.random() < 0.35:
            more = [
                (idx, center, stamp)
                for idx, center, stamp, _cells in actual_stamps
                if idx != target_idx and idx <= 6
            ]
            if more:
                idx2, center2, stamp2 = random.choice(more)
                obstacle_targets.append(_first_nonzero_cell(stamp2, center2))

    scatter_target = 2 if random.random() < 0.35 else 1 if obstacle_targets else 0
    while len(obstacle_targets) < scatter_target:
        cell = _safe_scatter_cell(grid.shape, reserved | set(obstacle_targets), protected)
        if cell is None:
            break
        obstacle_targets.append(cell)

    if len(obstacle_targets) > 2:
        obstacle_targets = obstacle_targets[:2]
    for r, c in obstacle_targets:
        if grid[r, c] != BG:
            return None
        grid[r, c] = OBSTACLE_COLOR

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_instance(grid)
        out = main(grid)
    except Exception:
        return False

    sprite = analysis["sprite"]
    marker = analysis["marker"]
    events = analysis["events"]

    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.count_nonzero(grid == SEED_COLOR) != 1:
        return False
    if np.count_nonzero(grid == OBSTACLE_COLOR) > 2:
        return False
    if sprite.shape != (3, 3):
        return False
    if len(events) < 4:
        return False

    placed = [e for e in events if e["status"] == "placed"]
    blocked = [e for e in events if e["status"] == "obstacle"]
    if len(placed) < 4:
        return False
    if not any(e["index"] >= 4 for e in placed):
        return False
    if not all(e["full_fit"] for e in placed):
        return False

    cell_sets = [e["cells"] for e in placed]
    for i in range(len(cell_sets)):
        for j in range(i + 1, len(cell_sets)):
            if cell_sets[i] & cell_sets[j]:
                return False

    oob_tail = [e["status"] for e in events[-2:]]
    if oob_tail != ["oob", "oob"]:
        return False

    if marker == RED:
        if not any(e["mirrored"] and e["status"] == "placed" for e in events):
            return False
        if blocked:
            if not any(e["index"] % 2 == 0 for e in blocked):
                return False
            after = [e for e in events if e["index"] > blocked[0]["index"] and e["status"] == "placed"]
            if not any(e["index"] % 2 == 1 for e in after):
                return False
            if not any(e["index"] % 2 == 0 for e in after):
                return False
    else:
        if any(e["mirrored"] and e["status"] == "placed" for e in events):
            return False

    if np.array_equal(grid, out):
        return False
    if not np.array_equal(main(grid.copy()), out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid spiral corner stamping puzzle")
