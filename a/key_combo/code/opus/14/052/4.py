from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
MARKER = 5
GREEN = 3
RECT_COLORS = (1, 7, 9)
DIR_SPECS = (
    ("up", -1, 0, 2),
    ("down", 1, 0, 8),
    ("left", 0, -1, 7),
    ("right", 0, 1, 6),
)
ALLOWED_RECT_SIZES = [
    (h, w)
    for h in range(2, 5)
    for w in range(2, 5)
    if not (h == 2 and w == 2)
]
ZONE_SIZE = 12
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rectangles(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    rect_id_grid = np.full((h, w), -1, dtype=int)
    rectangles: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, MARKER) or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        _in_bounds(g.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            r0, c0, r1, c1 = _bbox(cells)
            expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(cells) == expected_area, "foreground components must be solid rectangles"
            assert color in RECT_COLORS, "unexpected rectangle color"

            rect_id = len(rectangles)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    assert int(g[rr, cc]) == color
                    rect_id_grid[rr, cc] = rect_id

            rectangles.append(
                {
                    "id": rect_id,
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "cells": cells,
                }
            )

    return rectangles, rect_id_grid


def _trace_ray(
    grid: np.ndarray,
    rect_id_grid: np.ndarray,
    start: tuple[int, int],
    dr: int,
    dc: int,
) -> dict:
    g = np.asarray(grid, dtype=int)
    r, c = start
    path: list[tuple[int, int]] = []

    while True:
        r += dr
        c += dc
        if not _in_bounds(g.shape, r, c):
            return {"valid": False, "stop": "boundary", "path": path}

        value = int(g[r, c])
        if value == BG:
            path.append((r, c))
            continue
        if value == MARKER:
            return {"valid": False, "stop": "marker", "path": path}

        rect_id = int(rect_id_grid[r, c])
        assert rect_id >= 0, "ray must stop on a known rectangle"
        return {"valid": True, "stop": "rectangle", "rect_id": rect_id, "path": path}


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rectangles, rect_id_grid = _find_rectangles(g)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    markers.sort()

    marker_infos: list[dict] = []
    for marker in markers:
        rays = []
        hit_rect_ids: set[int] = set()
        for name, dr, dc, color in DIR_SPECS:
            ray = _trace_ray(g, rect_id_grid, marker, dr, dc)
            ray["direction"] = name
            ray["color"] = color
            if ray["valid"]:
                hit_rect_ids.add(int(ray["rect_id"]))
            rays.append(ray)
        marker_infos.append(
            {
                "marker": marker,
                "rays": rays,
                "hit_rect_ids": hit_rect_ids,
            }
        )

    return {
        "rectangles": rectangles,
        "rect_id_grid": rect_id_grid,
        "markers": marker_infos,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast cardinal sightline rays from grey markers to rectangles, color the successful bridge gaps by direction, and turn exactly-two-rectangle bridge markers green.

    Input structure:
    - The grid has a black background with 3-5 filled monochrome rectangles in blue, orange, or maroon.
    - It also contains 2-4 isolated grey singleton marker pixels outside every rectangle.
    - Each marker has at least one clear cardinal sightline through background cells toward some rectangle edge.

    Rule:
    - For each grey marker, scan upward, downward, leftward, and rightward through background cells until the ray hits a non-background cell or leaves the grid.
    - A ray is valid only when the first non-background cell is a rectangle; then recolor every traversed background cell by direction: up red, down teal, left orange, right pink.
    - Rays that stop at the border or another marker draw nothing, and the marker itself turns green only if its valid rays hit exactly two different rectangles; otherwise it stays grey.

    Key insight:
    - The colored ray segments identify which sightlines truly reach rectangles, while the marker's own final color is a separate connectivity summary: green means that one anchor bridges exactly two distinct rectangles.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_grid(g)

    assert 3 <= len(analysis["rectangles"]) <= 5, "expected 3-5 rectangles"
    assert 2 <= len(analysis["markers"]) <= 4, "expected 2-4 markers"

    out = g.copy()
    for marker_info in analysis["markers"]:
        hit_rect_ids = set()
        for ray in marker_info["rays"]:
            if not ray["valid"]:
                continue
            hit_rect_ids.add(int(ray["rect_id"]))
            for rr, cc in ray["path"]:
                out[rr, cc] = int(ray["color"])

        mr, mc = marker_info["marker"]
        out[mr, mc] = GREEN if len(hit_rect_ids) == 2 else MARKER

    return out


def _choose_profiles(marker_count: int) -> list[int]:
    options = {
        2: [(2, 1), (2, 3)],
        3: [(2, 1, 1), (2, 1, 2)],
        4: [(2, 1, 1, 1)],
    }
    profiles = list(random.choice(options[marker_count]))
    random.shuffle(profiles)
    return profiles


def _choose_directions(hit_count: int) -> tuple[str, ...]:
    names = [spec[0] for spec in DIR_SPECS]
    if hit_count == 1:
        return (random.choice(names),)
    if hit_count == 2:
        weighted = [
            ("up", "down"),
            ("left", "right"),
            ("up", "right"),
            ("up", "left"),
            ("down", "right"),
            ("down", "left"),
        ]
        return tuple(random.choice(weighted))
    if hit_count == 3:
        return tuple(random.choice(list(combinations(names, 3))))
    raise ValueError(f"unsupported hit count: {hit_count}")


def _layout_zones(marker_count: int) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    if marker_count == 2:
        zone_rows, zone_cols = random.choice([(1, 2), (2, 1)])
    else:
        zone_rows, zone_cols = 2, 2

    margin = random.randint(1, 2)
    gap = random.randint(1, 2)
    height = 2 * margin + zone_rows * ZONE_SIZE + (zone_rows - 1) * gap
    width = 2 * margin + zone_cols * ZONE_SIZE + (zone_cols - 1) * gap

    all_zones = []
    for zr in range(zone_rows):
        for zc in range(zone_cols):
            r0 = margin + zr * (ZONE_SIZE + gap)
            c0 = margin + zc * (ZONE_SIZE + gap)
            all_zones.append((r0, c0, r0 + ZONE_SIZE - 1, c0 + ZONE_SIZE - 1))

    chosen = random.sample(all_zones, marker_count)
    random.shuffle(chosen)
    return chosen, (height, width)


def _paint_rectangle(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _ray_cells_between_marker_and_rect(
    marker: tuple[int, int],
    rect: tuple[int, int, int, int],
    direction: str,
) -> list[tuple[int, int]]:
    mr, mc = marker
    r0, c0, r1, c1 = rect

    if direction == "up":
        return [(r, mc) for r in range(mr - 1, r1, -1)]
    if direction == "down":
        return [(r, mc) for r in range(mr + 1, r0)]
    if direction == "left":
        return [(mr, c) for c in range(mc - 1, c1, -1)]
    if direction == "right":
        return [(mr, c) for c in range(mc + 1, c0)]
    raise ValueError(direction)


def _candidate_rectangles_for_direction(
    zone: tuple[int, int, int, int],
    marker: tuple[int, int],
    direction: str,
) -> list[tuple[int, int, int, int]]:
    zr0, zc0, zr1, zc1 = zone
    mr, mc = marker
    candidates: list[tuple[int, int, int, int]] = []

    sizes = ALLOWED_RECT_SIZES[:]
    gaps = [1, 2, 3]
    random.shuffle(sizes)
    random.shuffle(gaps)

    for height, width in sizes:
        for gap in gaps:
            if direction == "up":
                rect_r1 = mr - gap - 1
                rect_r0 = rect_r1 - height + 1
                if rect_r0 < zr0:
                    continue
                c_lo = max(zc0, mc - width + 1)
                c_hi = min(zc1 - width + 1, mc)
                for c0 in range(c_lo, c_hi + 1):
                    candidates.append((rect_r0, c0, rect_r1, c0 + width - 1))
            elif direction == "down":
                rect_r0 = mr + gap + 1
                rect_r1 = rect_r0 + height - 1
                if rect_r1 > zr1:
                    continue
                c_lo = max(zc0, mc - width + 1)
                c_hi = min(zc1 - width + 1, mc)
                for c0 in range(c_lo, c_hi + 1):
                    candidates.append((rect_r0, c0, rect_r1, c0 + width - 1))
            elif direction == "left":
                rect_c1 = mc - gap - 1
                rect_c0 = rect_c1 - width + 1
                if rect_c0 < zc0:
                    continue
                r_lo = max(zr0, mr - height + 1)
                r_hi = min(zr1 - height + 1, mr)
                for r0 in range(r_lo, r_hi + 1):
                    candidates.append((r0, rect_c0, r0 + height - 1, rect_c1))
            elif direction == "right":
                rect_c0 = mc + gap + 1
                rect_c1 = rect_c0 + width - 1
                if rect_c1 > zc1:
                    continue
                r_lo = max(zr0, mr - height + 1)
                r_hi = min(zr1 - height + 1, mr)
                for r0 in range(r_lo, r_hi + 1):
                    candidates.append((r0, rect_c0, r0 + height - 1, rect_c1))

    random.shuffle(candidates)
    return candidates


def _rect_is_clear(grid: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG))


def _path_is_clear(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(int(grid[r, c]) == BG for r, c in cells)


def _marker_cell_for_zone(zone: tuple[int, int, int, int]) -> tuple[int, int]:
    zr0, zc0, _zr1, _zc1 = zone
    return zr0 + random.choice([5, 6]), zc0 + random.choice([5, 6])


def _crop_to_content(grid: np.ndarray) -> np.ndarray:
    nonzero = np.argwhere(grid != BG)
    r0, c0 = np.min(nonzero, axis=0)
    r1, c1 = np.max(nonzero, axis=0)
    top = random.randint(1, min(3, int(r0)))
    left = random.randint(1, min(3, int(c0)))
    bottom = random.randint(1, min(3, int(grid.shape[0] - 1 - r1)))
    right = random.randint(1, min(3, int(grid.shape[1] - 1 - c1)))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1].copy()


def _build_attempt() -> np.ndarray | None:
    marker_count = random.randint(2, 4)
    profiles = _choose_profiles(marker_count)
    zones, shape = _layout_zones(marker_count)
    grid = np.full(shape, BG, dtype=int)

    for zone, profile in zip(zones, profiles):
        marker = _marker_cell_for_zone(zone)
        if int(grid[marker]) != BG:
            return None
        grid[marker] = MARKER

        directions = _choose_directions(profile)
        for direction in directions:
            placed = False
            for rect in _candidate_rectangles_for_direction(zone, marker, direction):
                path_cells = _ray_cells_between_marker_and_rect(marker, rect, direction)
                if len(path_cells) == 0:
                    continue
                if not _rect_is_clear(grid, rect):
                    continue
                if not _path_is_clear(grid, path_cells):
                    continue
                _paint_rectangle(grid, rect, random.choice(RECT_COLORS))
                placed = True
                break
            if not placed:
                return None

    return _crop_to_content(grid)


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    try:
        analysis = _analyze_grid(g)
        out = main(g)
    except Exception:
        return False

    rectangles = analysis["rectangles"]
    marker_infos = analysis["markers"]
    if not (3 <= len(rectangles) <= 5 and 2 <= len(marker_infos) <= 4):
        return False

    hit_counts = [len(info["hit_rect_ids"]) for info in marker_infos]
    if any(count < 1 for count in hit_counts):
        return False
    if 2 not in hit_counts:
        return False
    if all(count == 2 for count in hit_counts):
        return False

    used_rect_ids: list[int] = []
    path_cells_seen: set[tuple[int, int]] = set()
    path_colors: set[int] = set()
    for info in marker_infos:
        for ray in info["rays"]:
            if not ray["valid"]:
                continue
            used_rect_ids.append(int(ray["rect_id"]))
            path_colors.add(int(ray["color"]))
            if len(ray["path"]) == 0:
                return False
            for cell in ray["path"]:
                if cell in path_cells_seen:
                    return False
                path_cells_seen.add(cell)

    if len(set(used_rect_ids)) != len(rectangles):
        return False
    if len(used_rect_ids) != len(rectangles):
        return False
    if len(path_colors) < 2:
        return False

    green_markers = sum(count == 2 for count in hit_counts)
    if green_markers < 1 or green_markers == len(marker_infos):
        return False

    if np.array_equal(out, g):
        return False

    for info in marker_infos:
        mr, mc = info["marker"]
        expected = GREEN if len(info["hit_rect_ids"]) == 2 else MARKER
        if int(out[mr, mc]) != expected:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
