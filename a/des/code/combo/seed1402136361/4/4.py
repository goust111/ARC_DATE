from __future__ import annotations

import math
import random
from collections import defaultdict
from functools import lru_cache

import numpy as np


CORNERS = ("TL", "TR", "BL", "BR")
GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def find_components(grid: np.ndarray, bg: int):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def nearest_corner_from_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> str:
    r0, c0, r1, c1 = bbox
    center_r2 = int(r0 + r1)
    center_c2 = int(c0 + c1)
    bottom_r2 = int(2 * (h - 1))
    right_c2 = int(2 * (w - 1))

    dists = {
        "TL": center_r2 * center_r2 + center_c2 * center_c2,
        "TR": center_r2 * center_r2 + (center_c2 - right_c2) ** 2,
        "BL": (center_r2 - bottom_r2) ** 2 + center_c2 * center_c2,
        "BR": (center_r2 - bottom_r2) ** 2 + (center_c2 - right_c2) ** 2,
    }
    ordered = sorted(dists.items(), key=lambda item: item[1])
    assert ordered[0][1] < ordered[1][1], "bbox center must have a unique nearest corner"
    return ordered[0][0]


def _local_to_global(corner: str, lr: int, lc: int, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return lr, lc
    if corner == "TR":
        return lr, w - 1 - lc
    if corner == "BL":
        return h - 1 - lr, lc
    if corner == "BR":
        return h - 1 - lr, w - 1 - lc
    raise ValueError(f"unknown corner: {corner}")


def _paint_corner_triangle(out: np.ndarray, corner: str, area: int, color: int) -> None:
    h, w = out.shape
    capacity = min(h, w) * (min(h, w) + 1) // 2
    assert 0 <= area <= capacity, "triangle does not fit in the grid"

    filled = 0
    diag = 0
    while filled < area:
        wrote = False
        for offset in range(diag + 1):
            lr = offset
            lc = diag - offset
            if lr >= h or lc >= w:
                continue
            r, c = _local_to_global(corner, lr, lc, h, w)
            out[r, c] = int(color)
            filled += 1
            wrote = True
            if filled == area:
                break
        assert wrote, "triangle area exceeds the available corner capacity"
        diag += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group foreground objects by their nearest grid corner and replace each group
    with a filled corner triangle whose area matches the group's total area.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    comps = find_components(g, bg)

    groups: dict[str, list[dict]] = defaultdict(list)
    for comp in comps:
        corner = nearest_corner_from_bbox(comp["bbox"], h, w)
        groups[corner].append(comp)

    out = np.full_like(g, bg)
    for corner, group in groups.items():
        total_area = sum(int(comp["area"]) for comp in group)
        leader = max(group, key=lambda comp: int(comp["area"]))
        _paint_corner_triangle(out, corner, total_area, int(leader["color"]))

    return out


def _triangle_number(side: int) -> int:
    return side * (side + 1) // 2


@lru_cache(maxsize=None)
def _positive_compositions(total: int, count: int):
    if count == 1:
        return ((total,),)
    out = []
    max_first = total - count + 1
    for first in range(1, max_first + 1):
        for rest in _positive_compositions(total - first, count - 1):
            out.append((first,) + rest)
    return tuple(out)


@lru_cache(maxsize=None)
def _unique_largest_partitions(total: int, count: int):
    out = []
    for parts in _positive_compositions(total, count):
        best = max(parts)
        if sum(1 for value in parts if value == best) == 1:
            out.append(parts)
    return tuple(out)


def _zone_bounds(corner: str, h: int, w: int) -> tuple[int, int, int, int]:
    top_hi = (h - 2) // 2
    left_hi = (w - 2) // 2
    bottom_lo = h - 1 - top_hi
    right_lo = w - 1 - left_hi

    if corner == "TL":
        return 1, top_hi, 1, left_hi
    if corner == "TR":
        return 1, top_hi, right_lo, w - 2
    if corner == "BL":
        return bottom_lo, h - 2, 1, left_hi
    if corner == "BR":
        return bottom_lo, h - 2, right_lo, w - 2
    raise ValueError(f"unknown corner: {corner}")


def _random_polyomino(area: int, max_h: int, max_w: int, rng: random.Random):
    if area == 1:
        return ((0, 0),), 1, 1

    for _ in range(150):
        min_h = max(1, math.ceil(area / max_w))
        box_h = rng.randint(min_h, min(max_h, area))
        min_w = max(1, math.ceil(area / box_h))
        max_box_w = min(max_w, area)
        if min_w > max_box_w:
            continue
        box_w = rng.randint(min_w, max_box_w)
        if box_h * box_w < area:
            continue

        start = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        if len(cells) != area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        norm = tuple(sorted((r - r0, c - c0) for r, c in cells))
        return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)

    return None


def _can_place(
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
    occupied: np.ndarray,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
) -> bool:
    h, w = occupied.shape
    placed = []
    shape_set = set()
    for rr, cc in shape:
        r = int(top + rr)
        c = int(left + cc)
        if not (row_lo <= r <= row_hi and col_lo <= c <= col_hi):
            return False
        if occupied[r, c]:
            return False
        placed.append((r, c))
        shape_set.add((r, c))

    for r, c in placed:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in shape_set:
                return False
    return True


def _place_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    corner: str,
    area: int,
    color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    row_lo, row_hi, col_lo, col_hi = _zone_bounds(corner, h, w)
    zone_h = row_hi - row_lo + 1
    zone_w = col_hi - col_lo + 1

    for _ in range(200):
        built = _random_polyomino(area, zone_h, zone_w, rng)
        if built is None:
            continue
        shape, shape_h, shape_w = built
        max_top = row_hi - shape_h + 1
        max_left = col_hi - shape_w + 1
        if max_top < row_lo or max_left < col_lo:
            continue

        for _ in range(120):
            top = rng.randint(row_lo, max_top)
            left = rng.randint(col_lo, max_left)
            if not _can_place(shape, top, left, occupied, row_lo, row_hi, col_lo, col_hi):
                continue

            rows = [top + rr for rr, _ in shape]
            cols = [left + cc for _, cc in shape]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            if nearest_corner_from_bbox(bbox, h, w) != corner:
                continue

            for rr, cc in shape:
                r = int(top + rr)
                c = int(left + cc)
                grid[r, c] = int(color)
                occupied[r, c] = True
            return True

    return False


def _choose_group_areas(active_corners: list[str], h: int, w: int, rng: random.Random):
    max_side = 5 if min(h, w) >= 16 else 4
    remaining_colors = 9
    remaining_groups = len(active_corners)
    groups: dict[str, list[int]] = {}

    for corner in active_corners:
        side = rng.randint(2, max_side)
        total = _triangle_number(side)

        row_lo, row_hi, col_lo, col_hi = _zone_bounds(corner, h, w)
        zone_span = min(row_hi - row_lo + 1, col_hi - col_lo + 1)
        local_cap = 2 if total >= 15 and zone_span <= 6 else 3
        max_objects = min(local_cap, total, remaining_colors - (remaining_groups - 1))

        valid_counts = []
        for count in range(1, max_objects + 1):
            if _unique_largest_partitions(total, count):
                valid_counts.append(count)
        if not valid_counts:
            return None

        if any(count > 1 for count in valid_counts) and rng.random() < 0.65:
            choices = [count for count in valid_counts if count > 1]
        else:
            choices = valid_counts
        count = rng.choice(choices)
        parts = list(rng.choice(_unique_largest_partitions(total, count)))
        groups[corner] = sorted(parts, reverse=True)
        remaining_colors -= count
        remaining_groups -= 1

    return groups


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        h, w = g.shape
        bg = get_bg(g)
        comps = find_components(g, bg)
        if not comps:
            return False
        if len(comps) > 9:
            return False
        if len({int(comp["color"]) for comp in comps}) != len(comps):
            return False
        groups: dict[str, list[dict]] = defaultdict(list)
        for comp in comps:
            groups[nearest_corner_from_bbox(comp["bbox"], h, w)].append(comp)
        if not groups:
            return False

        out = main(g)
        total_fg = sum(int(comp["area"]) for comp in comps)
        if int(np.count_nonzero(out != bg)) != total_fg:
            return False

        leaders = {int(max(group, key=lambda comp: int(comp["area"]))["color"]) for group in groups.values()}
        out_colors = set(map(int, np.unique(out))) - {bg}
        return bool(out_colors) and out_colors <= leaders
    except AssertionError:
        return False


def _gen_attempt(rng: random.Random):
    h = rng.randint(14, 18)
    w = rng.randint(14, 18)
    bg = rng.randrange(10)

    active_count = rng.randint(1, 4)
    active_corners = rng.sample(list(CORNERS), active_count)
    group_areas = _choose_group_areas(active_corners, h, w, rng)
    if group_areas is None:
        return None

    total_objects = sum(len(areas) for areas in group_areas.values())
    if total_objects > 9:
        return None

    colors = rng.sample([c for c in range(10) if c != bg], total_objects)
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    color_idx = 0
    for corner in active_corners:
        for area in group_areas[corner]:
            color = colors[color_idx]
            color_idx += 1
            if not _place_object(grid, occupied, corner, area, color, rng):
                return None

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
