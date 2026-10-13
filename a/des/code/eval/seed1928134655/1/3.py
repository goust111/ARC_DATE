from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 9
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
FRAME_H_RANGE = (4, 8)
FRAME_W_RANGE = (4, 8)
FRAME_COUNT_RANGE = (2, 5)
MARKER_CAP = 10
GEN_ATTEMPTS = 400
OBJECT_ATTEMPTS = 250
RESERVE_RADIUS = 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(height: int, width: int, r: int, c: int) -> bool:
    return 0 <= r < height and 0 <= c < width


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _gap_direction(
    gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int
) -> tuple[int, int] | None:
    r, c = gap
    if r == r0 and c0 < c < c1:
        return (-1, 0)
    if r == r1 and c0 < c < c1:
        return (1, 0)
    if c == c0 and r0 < r < r1:
        return (0, -1)
    if c == c1 and r0 < r < r1:
        return (0, 1)
    return None


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, MARKER):
            continue

        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue

        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = _border_coords(r0, c0, r1, c1)
        pixels = {(int(r), int(c)) for r, c in coords}
        if pixels - border:
            continue
        if len(pixels) != len(border) - 1:
            continue

        missing = border - pixels
        if len(missing) != 1:
            continue

        gap = next(iter(missing))
        direction = _gap_direction(gap, r0, c0, r1, c1)
        if direction is None:
            continue

        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "direction": direction,
            }
        )

    assert frames, "expected at least one gapped rectangular frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each colored rectangular frame with a one-pixel gap, count the maroon
    markers inside the frame and emit a same-colored ray of that length from
    the gap in the outward direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _most_common_color(g)

    out = g.copy()
    for frame in _detect_frames(g, bg):
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        gap_r, gap_c = frame["gap"]
        dr, dc = frame["direction"]

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int(np.sum(interior == MARKER))

        for step in range(marker_count):
            rr = int(gap_r + dr * step)
            cc = int(gap_c + dc * step)
            if not _in_bounds(height, width, rr, cc):
                break
            out[rr, cc] = color

    return out


def _balanced_directions(n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    while len(out) < n:
        chunk = DIRS[:]
        random.shuffle(chunk)
        out.extend(chunk)
    return out[:n]


def _balanced_flags(n: int) -> list[bool]:
    out = [False, True] * ((n + 1) // 2)
    random.shuffle(out)
    return out[:n]


def _expanded_rect(
    height: int, width: int, r0: int, c0: int, r1: int, c1: int
) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - RESERVE_RADIUS), min(height, r1 + RESERVE_RADIUS + 1)):
        for c in range(max(0, c0 - RESERVE_RADIUS), min(width, c1 + RESERVE_RADIUS + 1)):
            pts.add((r, c))
    return pts


def _expanded_points(
    height: int, width: int, points: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for r, c in points:
        for rr in range(max(0, r - RESERVE_RADIUS), min(height, r + RESERVE_RADIUS + 1)):
            for cc in range(max(0, c - RESERVE_RADIUS), min(width, c + RESERVE_RADIUS + 1)):
                pts.add((rr, cc))
    return pts


def _sample_bbox(
    height: int,
    width: int,
    box_h: int,
    box_w: int,
    direction: tuple[int, int],
    truncated: bool,
    area: int,
) -> tuple[int, int] | None:
    dr, dc = direction

    if dr != 0:
        if width - box_w - 2 < 0:
            return None
        c0 = random.randint(1, width - box_w - 1)
        max_space = height - box_h - 1
        if max_space < 0:
            return None
        if truncated:
            max_trunc = min(max_space, area - 2)
            if max_trunc < 1:
                return None
            side_space = random.randint(1, max_trunc)
        else:
            side_space = random.randint(0, max_space)
        if dr < 0:
            r0 = side_space
        else:
            r0 = height - box_h - side_space
        return (r0, c0)

    if height - box_h - 2 < 0:
        return None
    r0 = random.randint(1, height - box_h - 1)
    max_space = width - box_w - 1
    if max_space < 0:
        return None
    if truncated:
        max_trunc = min(max_space, area - 2)
        if max_trunc < 1:
            return None
        side_space = random.randint(1, max_trunc)
    else:
        side_space = random.randint(0, max_space)
    if dc < 0:
        c0 = side_space
    else:
        c0 = width - box_w - side_space
    return (r0, c0)


def _gap_from_bbox(
    r0: int, c0: int, box_h: int, box_w: int, direction: tuple[int, int]
) -> tuple[int, int]:
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    dr, dc = direction
    if dr < 0:
        return (r0, random.randint(c0 + 1, c1 - 1))
    if dr > 0:
        return (r1, random.randint(c0 + 1, c1 - 1))
    if dc < 0:
        return (random.randint(r0 + 1, r1 - 1), c0)
    return (random.randint(r0 + 1, r1 - 1), c1)


def _available_ray_length(
    height: int, width: int, gap: tuple[int, int], direction: tuple[int, int]
) -> int:
    gap_r, gap_c = gap
    dr, dc = direction
    if dr < 0:
        return gap_r + 1
    if dr > 0:
        return height - gap_r
    if dc < 0:
        return gap_c + 1
    return width - gap_c


def _ray_cells(
    height: int,
    width: int,
    gap: tuple[int, int],
    direction: tuple[int, int],
    count: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    gap_r, gap_c = gap
    dr, dc = direction
    for step in range(count):
        rr = gap_r + dr * step
        cc = gap_c + dc * step
        if not _in_bounds(height, width, rr, cc):
            break
        cells.append((rr, cc))
    return cells


def _choose_marker_count(area: int, available: int, truncated: bool) -> int | None:
    max_count = min(area, MARKER_CAP)
    if truncated:
        if available < 2 or max_count <= available:
            return None
        return random.randint(available + 1, max_count)
    if max_count < 1:
        return None
    return random.randint(1, min(max_count, available))


def _place_object(
    height: int,
    width: int,
    reserved: set[tuple[int, int]],
    color: int,
    direction: tuple[int, int],
    truncated: bool,
) -> dict[str, object] | None:
    for _ in range(OBJECT_ATTEMPTS):
        box_h = random.randint(*FRAME_H_RANGE)
        box_w = random.randint(*FRAME_W_RANGE)
        area = (box_h - 2) * (box_w - 2)
        anchor = _sample_bbox(height, width, box_h, box_w, direction, truncated, area)
        if anchor is None:
            continue
        r0, c0 = anchor
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1

        gap = _gap_from_bbox(r0, c0, box_h, box_w, direction)
        available = _available_ray_length(height, width, gap, direction)
        marker_count = _choose_marker_count(area, available, truncated)
        if marker_count is None:
            continue

        frame_cells = list(_border_coords(r0, c0, r1, c1) - {gap})
        ray_cells = _ray_cells(height, width, gap, direction, marker_count)
        if not ray_cells:
            continue

        interior_cells = [
            (r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)
        ]
        markers = random.sample(interior_cells, marker_count)

        reserve = _expanded_rect(height, width, r0, c0, r1, c1)
        reserve |= _expanded_points(height, width, ray_cells)
        if reserve & reserved:
            continue

        return {
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "direction": direction,
            "marker_count": marker_count,
            "frame_cells": frame_cells,
            "markers": markers,
            "ray_cells": ray_cells,
            "reserve": reserve,
            "truncated": len(ray_cells) < marker_count,
        }

    return None


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    grid = np.full((height, width), BG, dtype=int)

    area = height * width
    max_frames = min(
        FRAME_COUNT_RANGE[1],
        len(FRAME_COLORS),
        max(FRAME_COUNT_RANGE[0], area // 170),
    )
    if max_frames < FRAME_COUNT_RANGE[0]:
        return None
    frame_count = random.randint(FRAME_COUNT_RANGE[0], max_frames)

    colors = random.sample(list(FRAME_COLORS), frame_count)
    directions = _balanced_directions(frame_count)
    truncation_flags = _balanced_flags(frame_count)

    reserved: set[tuple[int, int]] = set()
    objects: list[dict[str, object]] = []
    for color, direction, truncated in zip(colors, directions, truncation_flags):
        obj = _place_object(height, width, reserved, color, direction, truncated)
        if obj is None:
            return None
        objects.append(obj)
        reserved |= obj["reserve"]

    if len({tuple(obj["direction"]) for obj in objects}) < 2:
        return None
    if not any(bool(obj["truncated"]) for obj in objects):
        return None
    if not any(not bool(obj["truncated"]) for obj in objects):
        return None

    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["frame_cells"]:
            grid[r, c] = color
        for r, c in obj["markers"]:
            grid[r, c] = MARKER

    expected = grid.copy()
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["ray_cells"]:
            expected[r, c] = color

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
