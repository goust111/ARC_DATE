from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(x) for x in coords[0])
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], int]:
    h, w = grid.shape
    assert h >= 3 and w >= 3, "grid too small for a 2-row bottom legend"

    top = grid[h - 2]
    bottom = grid[h - 1]
    mapping: dict[int, np.ndarray] = {}

    c = 0
    while c <= w - 3:
        marker_color = int(top[c])
        if marker_color != bg and bool(np.all(top[c : c + 3] == marker_color)):
            left_ok = c == 0 or int(top[c - 1]) != marker_color
            right_ok = c + 3 == w or int(top[c + 3]) != marker_color
            pattern = np.array(bottom[c : c + 3], dtype=int)
            if left_ok and right_ok and all(int(x) != bg for x in pattern) and len(set(map(int, pattern))) >= 2:
                mapping[marker_color] = pattern
                c += 3
                continue
        c += 1

    assert mapping, "failed to detect the bottom legend"
    return mapping, h - 2


def _find_frame_color(work: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in work.flat if int(v) != bg]
    assert colors, "expected at least one non-background color in the work area"
    return int(Counter(colors).most_common(1)[0][0])


def _non_frame_components(work: np.ndarray, frame_color: int) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = work.shape
    passable = work != frame_color
    seen = np.zeros_like(passable, dtype=bool)
    comps: list[tuple[list[tuple[int, int]], bool]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(passable[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_boundary = False

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_boundary = True
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(passable[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append((coords, touches_boundary))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed region with the legend pattern indexed by its interior marker color.

    Input structure:
    - The background is the dominant color.
    - Several one-pixel-wide closed outlines of a shared color enclose separate regions.
    - Each enclosed region contains exactly one single-pixel color marker.
    - A 2-row legend lies on the bottom edge: each 3-pixel entry has a uniform marker-color row above a 3-pixel pattern row.

    Rule:
    - Read the bottom legend as a mapping from marker color to a 3-pixel horizontal fill pattern.
    - In the area above the legend, find every connected non-outline region that does not touch the grid boundary.
    - For each such region, read its unique marker color, look up the matching legend pattern, and tile that pattern across the whole region.
    - The pattern phase starts at the region bbox's top-left cell, repeats only horizontally, covers the marker pixel, and never overwrites outline pixels.

    Key insight:
    - The bottom strip is a lookup table from interior marker colors to fill patterns, and the enclosed region's bbox fixes the phase of the horizontal tiling.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    mapping, legend_row = _find_legend(g, bg)

    work = g[:legend_row]
    frame_color = _find_frame_color(work, bg)
    out = g.copy()

    for coords, touches_boundary in _non_frame_components(work, frame_color):
        if touches_boundary:
            continue

        marker_cells = [(r, c) for r, c in coords if int(work[r, c]) != bg]
        marker_colors = {int(work[r, c]) for r, c in marker_cells}
        assert len(marker_cells) == 1 and len(marker_colors) == 1, "each enclosed region must have one marker"
        marker_color = marker_colors.pop()
        assert marker_color in mapping, "missing legend entry for a region marker"

        cols = [c for _r, c in coords]
        c0 = min(cols)
        pattern = mapping[marker_color]
        for r, c in coords:
            out[r, c] = int(pattern[(c - c0) % 3])

    return out


MAX_RETRIES = 400
MIN_REGIONS = 2
MAX_REGIONS = 4


def _rectangle_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 5)
    w = rng.randint(3, 7)
    return np.ones((h, w), dtype=bool)


def _corner_notch_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 6)
    w = rng.randint(3, 7)
    mask = np.ones((h, w), dtype=bool)
    notch_h = rng.randint(1, h - 2)
    notch_w = rng.randint(1, w - 2)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        mask[:notch_h, :notch_w] = False
    elif corner == "tr":
        mask[:notch_h, w - notch_w :] = False
    elif corner == "bl":
        mask[h - notch_h :, :notch_w] = False
    else:
        mask[h - notch_h :, w - notch_w :] = False
    return mask


def _side_notch_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(4, 6)
    w = rng.randint(4, 7)
    mask = np.ones((h, w), dtype=bool)

    if rng.random() < 0.5:
        depth = rng.randint(1, h - 2)
        start = rng.randint(1, w - 2)
        end = rng.randint(start + 1, w - 1)
        if rng.random() < 0.5:
            mask[:depth, start:end] = False
        else:
            mask[h - depth :, start:end] = False
    else:
        depth = rng.randint(1, w - 2)
        start = rng.randint(1, h - 2)
        end = rng.randint(start + 1, h - 1)
        if rng.random() < 0.5:
            mask[start:end, :depth] = False
        else:
            mask[start:end, w - depth :] = False

    return mask


def _make_shape_mask(rng: random.Random) -> np.ndarray:
    family = rng.choice(("rect", "corner", "side"))
    if family == "rect":
        mask = _rectangle_mask(rng)
    elif family == "corner":
        mask = _corner_notch_mask(rng)
    else:
        mask = _side_notch_mask(rng)

    mask = _trim_mask(mask)
    assert mask.sum() >= 4 and _is_connected(mask)
    return mask


def _make_region_spec(mask: np.ndarray) -> dict[str, object]:
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1 : h + 1, 1 : w + 1] = mask
    boundary = np.zeros_like(padded)

    for r, c in np.argwhere(padded):
        rr, cc = int(r), int(c)
        for dr, dc in DIR4:
            nr, nc = int(rr + dr), int(cc + dc)
            if not bool(padded[nr, nc]):
                boundary[nr, nc] = True

    region_coords = [(int(r), int(c)) for r, c in np.argwhere(padded)]
    boundary_coords = [(int(r), int(c)) for r, c in np.argwhere(boundary)]
    occupied_coords = region_coords + boundary_coords
    return {
        "patch_h": int(padded.shape[0]),
        "patch_w": int(padded.shape[1]),
        "region_coords": region_coords,
        "boundary_coords": boundary_coords,
        "occupied_coords": occupied_coords,
    }


def _reserve_cells(blocked: np.ndarray, coords: list[tuple[int, int]], top: int, left: int) -> None:
    h, w = blocked.shape
    for rr, cc in coords:
        r = int(top + rr)
        c = int(left + cc)
        for dr, dc in DIR8:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_regions(
    specs: list[dict[str, object]],
    work_h: int,
    width: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((work_h, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["patch_h"]) * int(specs[i]["patch_w"]), reverse=True)

    for idx in order:
        spec = specs[idx]
        patch_h = int(spec["patch_h"])
        patch_w = int(spec["patch_w"])
        row_min = 1
        col_min = 1
        row_max = work_h - patch_h - 1
        col_max = width - patch_w - 1
        if row_max < row_min or col_max < col_min:
            return None

        placed = False
        for _ in range(180):
            top = rng.randint(row_min, row_max)
            left = rng.randint(col_min, col_max)
            occupied_coords = spec["occupied_coords"]
            if any(bool(blocked[top + rr, left + cc]) for rr, cc in occupied_coords):
                continue
            placements[idx] = (top, left)
            _reserve_cells(blocked, occupied_coords, top, left)
            placed = True
            break
        if not placed:
            return None

    return [pos for pos in placements if pos is not None]


def _all_patterns(color_pool: list[int]) -> list[tuple[int, int, int]]:
    patterns: list[tuple[int, int, int]] = []
    for i, a in enumerate(color_pool):
        for b in color_pool[i + 1 :]:
            patterns.extend(
                [
                    (a, a, b),
                    (a, b, a),
                    (b, a, a),
                    (a, b, b),
                    (b, a, b),
                    (b, b, a),
                ]
            )
    return patterns


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.randint(MIN_REGIONS, MAX_REGIONS)
    specs = [_make_region_spec(_make_shape_mask(rng)) for _ in range(region_count)]

    legend_gaps = [rng.randint(0, 1) for _ in range(region_count - 1)]
    legend_width = 3 * region_count + sum(legend_gaps)
    max_patch_h = max(int(spec["patch_h"]) for spec in specs)
    max_patch_w = max(int(spec["patch_w"]) for spec in specs)

    min_h = max(14, max_patch_h + 5, 8 + 3 * region_count)
    min_w = max(16, legend_width + 2, max_patch_w + 5, 10 + 3 * region_count)
    if min_h > 30 or min_w > 30:
        return None

    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)
    work_h = height - 2
    placements = _place_regions(specs, work_h, width, rng)
    if placements is None:
        return None

    color_order = list(range(10))
    rng.shuffle(color_order)
    bg = color_order.pop()
    frame_color = color_order.pop()
    marker_colors = [color_order.pop() for _ in range(region_count)]

    color_pool_size = max(2, min(4, len(color_order)))
    color_pool = color_order[:color_pool_size]
    patterns = _all_patterns(color_pool)
    if len(patterns) < region_count:
        return None
    rng.shuffle(patterns)
    region_patterns = patterns[:region_count]

    grid = np.full((height, width), bg, dtype=int)

    for idx, spec in enumerate(specs):
        top, left = placements[idx]
        for rr, cc in spec["boundary_coords"]:
            grid[top + rr, left + cc] = frame_color
        region_abs = [(top + rr, left + cc) for rr, cc in spec["region_coords"]]
        mr, mc = rng.choice(region_abs)
        grid[mr, mc] = marker_colors[idx]

    left_pad = rng.randint(0, width - legend_width)
    legend_order = list(range(region_count))
    rng.shuffle(legend_order)
    c = left_pad
    for j, idx in enumerate(legend_order):
        grid[height - 2, c : c + 3] = marker_colors[idx]
        grid[height - 1, c : c + 3] = np.array(region_patterns[idx], dtype=int)
        c += 3
        if j < region_count - 1:
            c += legend_gaps[j]

    if _get_bg(grid) != bg:
        return None

    marker_counter = Counter(marker_colors)
    if max(marker_counter.values()) != 1:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
