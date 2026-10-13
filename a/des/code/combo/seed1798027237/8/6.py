from __future__ import annotations

import random

import numpy as np


SHADOW_COLOR = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRECTIONS = ("up", "down", "left", "right")
INPUT_COLORS = tuple(color for color in range(10) if color != SHADOW_COLOR)

# --------------- solver ---------------


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(grid == color, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp.sort()
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 0].max()),
        int(arr[:, 1].max()),
    )


def _direction_from_marker(
    grid: np.ndarray,
    bg: int,
    obj_coords: list[tuple[int, int]],
    marker: tuple[int, int],
) -> str:
    r0, c0, r1, c1 = _bbox(obj_coords)
    mr, mc = marker
    candidates: list[str] = []

    if r0 <= mr <= r1 and mc <= c0 - 2 and np.all(grid[mr, mc + 1 : c0] == bg):
        candidates.append("left")
    if r0 <= mr <= r1 and mc >= c1 + 2 and np.all(grid[mr, c1 + 1 : mc] == bg):
        candidates.append("right")
    if c0 <= mc <= c1 and mr <= r0 - 2 and np.all(grid[mr + 1 : r0, mc] == bg):
        candidates.append("up")
    if c0 <= mc <= c1 and mr >= r1 + 2 and np.all(grid[r1 + 1 : mr, mc] == bg):
        candidates.append("down")

    assert len(candidates) == 1, "marker must indicate exactly one cardinal direction"
    return candidates[0]


def _shadow_cells(
    obj_coords: list[tuple[int, int]], direction: str
) -> list[tuple[int, int]]:
    rows = sorted({int(r) for r, _ in obj_coords})
    cols = sorted({int(c) for _, c in obj_coords})
    r0, c0, r1, c1 = _bbox(obj_coords)

    if direction == "left":
        return [(r, c0 - 1) for r in rows]
    if direction == "right":
        return [(r, c1 + 1) for r in rows]
    if direction == "up":
        return [(r0 - 1, c) for c in cols]
    assert direction == "down"
    return [(r1 + 1, c) for c in cols]


def _object_and_marker(
    grid: np.ndarray, color: int, bg: int
) -> tuple[list[tuple[int, int]], tuple[int, int], str]:
    comps = _components_for_color(grid, color)
    objects = [comp for comp in comps if len(comp) > 1]
    markers = [comp[0] for comp in comps if len(comp) == 1]

    assert len(objects) == 1, "each non-background color must have one multi-pixel object"
    assert len(markers) == 1, "each non-background color must have one singleton marker"

    obj_coords = objects[0]
    marker = markers[0]
    direction = _direction_from_marker(grid, bg, obj_coords, marker)
    return obj_coords, marker, direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a grey one-dimensional shadow for each monochrome object using its same-color direction marker.

    Each non-background color appears as exactly one 4-connected multi-pixel object plus
    one detached singleton marker of the same color. The marker sits strictly above,
    below, left, or right of the object's bounding box and encodes the shadow direction.
    The shadow is a one-cell-offset projection: vertical for left/right markers and
    horizontal for up/down markers. Original objects and markers stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground object"
    assert SHADOW_COLOR not in colors, "grey is reserved for output shadows"

    out = g.copy()
    height, width = g.shape
    for color in colors:
        obj_coords, _marker, direction = _object_and_marker(g, color, bg)
        for r, c in _shadow_cells(obj_coords, direction):
            if 0 <= r < height and 0 <= c < width:
                assert g[r, c] == bg or out[r, c] == SHADOW_COLOR
                out[r, c] = SHADOW_COLOR

    return out


# --------------- generator ---------------

GRID_MIN_SIDE = 11
GRID_MAX_SIDE = 24
OBJECT_COUNT_MIN = 1
OBJECT_COUNT_MAX = 4
SHAPE_SIDE_MIN = 2
SHAPE_SIDE_MAX = 5
SHAPE_AREA_MIN = 3
MARKER_DIST_MIN = 2
MARKER_DIST_MAX = 4
GEN_MAX_ATTEMPTS = 400
SHAPE_MAX_ATTEMPTS = 200


def _local_neighbors(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            out.append((nr, nc))
    return out


def _sample_shape() -> list[tuple[int, int]]:
    for _ in range(SHAPE_MAX_ATTEMPTS):
        height = random.randint(SHAPE_SIDE_MIN, SHAPE_SIDE_MAX)
        width = random.randint(SHAPE_SIDE_MIN, SHAPE_SIDE_MAX)
        max_area = height * width - 1
        if max_area < SHAPE_AREA_MIN:
            continue

        target_area = random.randint(SHAPE_AREA_MIN, max_area)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        frontier = set(_local_neighbors(start[0], start[1], height, width))

        while len(cells) < target_area:
            choice = random.choice(tuple(frontier))
            frontier.remove(choice)
            cells.add(choice)
            for nbr in _local_neighbors(choice[0], choice[1], height, width):
                if nbr not in cells:
                    frontier.add(nbr)

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        cropped = sorted((r - r0, c - c0) for r, c in cells)
        cropped_h = max(r for r, _ in cropped) + 1
        cropped_w = max(c for _, c in cropped) + 1

        if cropped_h < 2 or cropped_w < 2:
            continue
        if len(cropped) == cropped_h * cropped_w:
            continue
        return cropped

    raise RuntimeError("failed to sample an irregular connected shape")


def _dilate_once(
    cells: set[tuple[int, int]], height: int, width: int
) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                out.add((nr, nc))
    return out


def _build_spec(color: int) -> dict[str, object]:
    cells = _sample_shape()
    height = max(r for r, _ in cells) + 1
    width = max(c for _, c in cells) + 1
    direction = random.choice(DIRECTIONS)
    distance = random.randint(MARKER_DIST_MIN, MARKER_DIST_MAX)

    if direction == "left":
        marker = (random.randrange(height), -distance)
        corridor = {(r, c) for r in range(height) for c in range(-distance, 0)}
    elif direction == "right":
        marker = (random.randrange(height), width - 1 + distance)
        corridor = {
            (r, c) for r in range(height) for c in range(width, width - 1 + distance + 1)
        }
    elif direction == "up":
        marker = (-distance, random.randrange(width))
        corridor = {(r, c) for r in range(-distance, 0) for c in range(width)}
    else:
        marker = (height - 1 + distance, random.randrange(width))
        corridor = {
            (r, c) for r in range(height, height - 1 + distance + 1) for c in range(width)
        }

    shadow = set(_shadow_cells(cells, direction))
    colored = set(cells) | {marker}
    required = set(cells) | corridor
    protected = colored | shadow

    return {
        "color": color,
        "cells": set(cells),
        "marker": marker,
        "direction": direction,
        "required": required,
        "protected": protected,
        "shadow": shadow,
    }


def _translated(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _valid_positions(
    height: int,
    width: int,
    blocked: np.ndarray,
    spec: dict[str, object],
) -> list[tuple[int, int]]:
    required = spec["required"]
    assert isinstance(required, set)
    min_r = min(r for r, _ in required)
    max_r = max(r for r, _ in required)
    min_c = min(c for _, c in required)
    max_c = max(c for _, c in required)

    positions: list[tuple[int, int]] = []
    for dr in range(-min_r, height - max_r):
        for dc in range(-min_c, width - max_c):
            translated = _translated(required, dr, dc)
            if any(blocked[r, c] for r, c in translated):
                continue
            positions.append((dr, dc))
    random.shuffle(positions)
    return positions


def _commit_blocked(
    blocked: np.ndarray,
    spec: dict[str, object],
    dr: int,
    dc: int,
) -> None:
    height, width = blocked.shape
    required = spec["required"]
    protected = spec["protected"]
    assert isinstance(required, set) and isinstance(protected, set)

    for r, c in _translated(required, dr, dc):
        blocked[r, c] = True
    for r, c in _dilate_once(_translated(protected, dr, dc), height, width):
        blocked[r, c] = True


def _render_grid(
    shape: tuple[int, int],
    bg: int,
    placements: list[tuple[dict[str, object], int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    inp = np.full((height, width), bg, dtype=int)
    expected = np.full((height, width), bg, dtype=int)

    for spec, dr, dc in placements:
        color = int(spec["color"])
        cells = spec["cells"]
        marker = spec["marker"]
        assert isinstance(cells, set) and isinstance(marker, tuple)

        for r, c in _translated(cells, dr, dc):
            inp[r, c] = color
            expected[r, c] = color

        mr, mc = marker
        inp[mr + dr, mc + dc] = color
        expected[mr + dr, mc + dc] = color

    for spec, dr, dc in placements:
        shadow = spec["shadow"]
        assert isinstance(shadow, set)
        for r, c in _translated(shadow, dr, dc):
            if 0 <= r < height and 0 <= c < width:
                assert expected[r, c] == bg
                expected[r, c] = SHADOW_COLOR

    return inp, expected


def _background_is_unique(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    bg_count = count_map[bg]
    return bg_count == max(count_map.values()) and sum(
        count == bg_count for count in count_map.values()
    ) == 1


def _place_specs(
    height: int,
    width: int,
    specs: list[dict[str, object]],
) -> list[tuple[dict[str, object], int, int]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    placements: list[tuple[dict[str, object], int, int]] = []

    for spec in sorted(specs, key=lambda item: len(item["required"]), reverse=True):
        positions = _valid_positions(height, width, blocked, spec)
        if not positions:
            return None
        dr, dc = random.choice(positions)
        _commit_blocked(blocked, spec, dr, dc)
        placements.append((spec, dr, dc))

    return placements


def _gen_attempt() -> np.ndarray | None:
    obj_count = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    height = random.randint(
        max(GRID_MIN_SIDE, 8 + 2 * obj_count),
        min(GRID_MAX_SIDE, 14 + 3 * obj_count),
    )
    width = random.randint(
        max(GRID_MIN_SIDE, 8 + 2 * obj_count),
        min(GRID_MAX_SIDE, 14 + 3 * obj_count),
    )

    palette = random.sample(INPUT_COLORS, obj_count + 1)
    bg = palette[0]
    colors = palette[1:]
    specs = [_build_spec(color) for color in colors]
    placements = _place_specs(height, width, specs)
    if placements is None:
        return None

    inp, expected = _render_grid((height, width), bg, placements)
    if not _background_is_unique(inp, bg):
        return None
    if np.count_nonzero(inp == bg) <= inp.size // 2:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-projection input")
