from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

SPRITE_COLORS = (BLUE, GREEN, YELLOW, PINK)

LINE_OVERLAP_TEMPLATES = (
    (2, [(-2, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)]),
    (3, [(-1, 0), (-1, 1), (-1, 2), (1, 0), (1, 1), (2, 1)]),
    (3, [(-2, 1), (-1, 0), (1, 0), (1, 1), (1, 2)]),
)
LINE_CLEAR_TEMPLATES = (
    (2, [(-2, 0), (-1, 0), (-1, 1), (-2, 1), (-2, 2)]),
    (3, [(-2, 1), (-1, 0), (-1, 1), (-1, 2), (-2, 2)]),
    (3, [(1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]),
)

POINT_OVERLAP_TEMPLATES = (
    [(-1, -1), (-1, -2), (0, -2), (1, -1), (2, 0), (1, 1)],
    [(-2, 0), (-1, -1), (0, -2), (1, -1), (2, 0)],
)
POINT_CLEAR_TEMPLATES = (
    [(-1, -1), (-2, -1), (-2, 0), (-1, 1), (0, 2)],
    [(-1, -1), (-1, -2), (0, -2), (1, -2), (2, -1), (2, 0)],
)


def _neighbors(coords: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    r, c = coords
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            out.append((r + dr, c + dc))
    return out


def _is_connected(coords: set[tuple[int, int]], connectivity: int) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in _neighbors(cur, connectivity):
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(coords)


def _divider_spans(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for idx in sorted(dividers):
        if start <= idx - 1:
            spans.append((start, idx - 1))
        start = idx + 1
    if start <= length - 1:
        spans.append((start, length - 1))
    return spans


def _classify_marker(red_coords: list[tuple[int, int]]) -> tuple[str, int | tuple[int, int]]:
    if len(red_coords) == 1:
        return "point", red_coords[0]

    rows = sorted({r for r, _ in red_coords})
    cols = sorted({c for _, c in red_coords})
    if len(rows) == 1:
        ordered = sorted(c for _, c in red_coords)
        if ordered == list(range(ordered[0], ordered[-1] + 1)):
            return "row", rows[0]
    if len(cols) == 1:
        ordered = sorted(r for r, _ in red_coords)
        if ordered == list(range(ordered[0], ordered[-1] + 1)):
            return "col", cols[0]
    raise AssertionError("ambiguous marker geometry")


def _mirror_coords(
    sprite_coords: set[tuple[int, int]],
    marker_mode: str,
    marker_data: int | tuple[int, int],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    mirrored = set()
    for r, c in sprite_coords:
        if marker_mode == "row":
            rr, cc = 2 * int(marker_data) - r, c
        elif marker_mode == "col":
            rr, cc = r, 2 * int(marker_data) - c
        else:
            mr, mc = marker_data
            rr, cc = 2 * mr - r, 2 * mc - c
        if 0 <= rr < h and 0 <= cc < w:
            mirrored.add((rr, cc))
    return mirrored


def _analyze_region(region: np.ndarray) -> tuple[int, int]:
    colors = sorted(set(map(int, region.flatten())) - {BLACK, RED})
    if len(colors) != 1:
        raise AssertionError("each region must contain exactly one sprite color")
    sprite_color = int(colors[0])
    sprite_coords = {tuple(map(int, p)) for p in np.argwhere(region == sprite_color)}
    red_coords = [tuple(map(int, p)) for p in np.argwhere(region == RED)]
    marker_mode, marker_data = _classify_marker(red_coords)
    mirrored = _mirror_coords(sprite_coords, marker_mode, marker_data, region.shape)
    overlap = len(sprite_coords & mirrored)
    return sprite_color, overlap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each region's marked sprite and keep only regions whose reflection overlaps itself.

    Input structure:
    - Full grey divider rows and columns split the grid into rectangular black-background regions.
    - Each region contains one monochrome sprite and one red marker: either a short straight red line embedded in the sprite area or a single isolated red pixel near it.
    - Divider lines stay grey and region interiors contain only black, the sprite color, and the red marker.

    Rule:
    - Split the grid by the full grey rows and columns.
    - If the red marker is a horizontal or vertical line, mirror the sprite across that axis; if it is a single red pixel, point-reflect the sprite 180 degrees around that pixel.
    - Count how many non-red sprite pixels are covered by both the original sprite and its reflected copy.
    - Regions with positive overlap are flood-filled entirely with the sprite color; regions with zero overlap are cleared to black.

    Key insight:
    - The red marker's geometry chooses the reflection type, but the actual gate is whether that reflected sprite overlaps the original sprite at all.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    divider_rows = [r for r in range(h) if np.all(g[r, :] == GREY)]
    divider_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    row_spans = _divider_spans(h, divider_rows)
    col_spans = _divider_spans(w, divider_cols)

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            sprite_color, overlap = _analyze_region(region)
            out[r0 : r1 + 1, c0 : c1 + 1] = sprite_color if overlap > 0 else BLACK

    return out


def _transform_point_offsets(
    offsets: list[tuple[int, int]], rot_k: int, flip_lr: bool
) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in offsets]
    if flip_lr:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rot_k % 4):
        pts = [(-c, r) for r, c in pts]
    return pts


def _transform_line_offsets(
    offsets: list[tuple[int, int]], vertical: bool, flip_r: bool, flip_c: bool
) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in offsets]
    if flip_r:
        pts = [(-r, c) for r, c in pts]
    if flip_c:
        pts = [(r, -c) for r, c in pts]
    if vertical:
        pts = [(c, r) for r, c in pts]
    return pts


def _build_point_region(
    region_h: int, region_w: int, color: int, want_overlap: bool, rng: random.Random
) -> np.ndarray:
    templates = POINT_OVERLAP_TEMPLATES if want_overlap else POINT_CLEAR_TEMPLATES
    for _ in range(200):
        offsets = list(rng.choice(templates))
        offsets = _transform_point_offsets(offsets, rng.randrange(4), rng.random() < 0.5)
        mirrored_offsets = [(-r, -c) for r, c in offsets]
        all_offsets = offsets + mirrored_offsets + [(0, 0)]
        min_r = min(r for r, _ in all_offsets)
        max_r = max(r for r, _ in all_offsets)
        min_c = min(c for _, c in all_offsets)
        max_c = max(c for _, c in all_offsets)
        if max_r - min_r + 1 > region_h or max_c - min_c + 1 > region_w:
            continue

        marker_r = rng.randint(-min_r, region_h - 1 - max_r)
        marker_c = rng.randint(-min_c, region_w - 1 - max_c)
        sprite = {(marker_r + dr, marker_c + dc) for dr, dc in offsets}

        if len(sprite) != len(offsets):
            continue
        if not _is_connected(sprite, connectivity=8):
            continue

        if any(abs(r - marker_r) + abs(c - marker_c) == 1 for r, c in sprite):
            continue
        if not any(max(abs(r - marker_r), abs(c - marker_c)) == 1 for r, c in sprite):
            continue

        region = np.full((region_h, region_w), BLACK, dtype=int)
        for r, c in sprite:
            region[r, c] = color
        region[marker_r, marker_c] = RED

        sprite_color, overlap = _analyze_region(region)
        if sprite_color != color:
            continue
        if (overlap > 0) != want_overlap:
            continue
        return region

    raise RuntimeError("failed to build point-reflection region")


def _build_line_region(
    region_h: int, region_w: int, color: int, want_overlap: bool, rng: random.Random
) -> np.ndarray:
    templates = LINE_OVERLAP_TEMPLATES if want_overlap else LINE_CLEAR_TEMPLATES
    for _ in range(200):
        line_len, color_offsets = rng.choice(templates)
        vertical = rng.random() < 0.5
        flip_r = rng.random() < 0.5
        flip_c = rng.random() < 0.5

        line_offsets = [(0, c) for c in range(line_len)]
        line_offsets = _transform_line_offsets(line_offsets, vertical, flip_r, flip_c)
        color_offsets = _transform_line_offsets(color_offsets, vertical, flip_r, flip_c)

        red_coords = set(line_offsets)
        color_coords = set(color_offsets)
        if red_coords & color_coords:
            continue

        marker_mode, marker_data = _classify_marker(sorted(red_coords))
        mirrored_colors = _mirror_coords(color_coords, marker_mode, marker_data, (100, 100))
        all_offsets = list(red_coords | color_coords | mirrored_colors)
        min_r = min(r for r, _ in all_offsets)
        max_r = max(r for r, _ in all_offsets)
        min_c = min(c for _, c in all_offsets)
        max_c = max(c for _, c in all_offsets)
        if max_r - min_r + 1 > region_h or max_c - min_c + 1 > region_w:
            continue

        base_r = rng.randint(-min_r, region_h - 1 - max_r)
        base_c = rng.randint(-min_c, region_w - 1 - max_c)

        placed_reds = {(base_r + r, base_c + c) for r, c in red_coords}
        placed_colors = {(base_r + r, base_c + c) for r, c in color_coords}

        if len(placed_colors) < 4 or len(placed_colors) > 7:
            continue
        if not _is_connected(placed_reds | placed_colors, connectivity=8):
            continue

        region = np.full((region_h, region_w), BLACK, dtype=int)
        for r, c in placed_colors:
            region[r, c] = color
        for r, c in placed_reds:
            region[r, c] = RED

        sprite_color, overlap = _analyze_region(region)
        if sprite_color != color:
            continue
        if (overlap > 0) != want_overlap:
            continue
        return region

    raise RuntimeError("failed to build line-reflection region")


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(300):
        region_heights = [rng.randint(6, 8), rng.randint(6, 8)]
        region_widths = [rng.randint(6, 8), rng.randint(6, 8)]
        grid_h = sum(region_heights) + 1
        grid_w = sum(region_widths) + 1
        if grid_h > 30 or grid_w > 30:
            continue

        specs = [("point", True), ("point", False), ("line", True), ("line", False)]
        rng.shuffle(specs)
        colors = list(SPRITE_COLORS)
        rng.shuffle(colors)

        regions = []
        ok = True
        for idx, (kind, want_overlap) in enumerate(specs):
            region_h = region_heights[idx // 2]
            region_w = region_widths[idx % 2]
            color = colors[idx]
            try:
                if kind == "point":
                    region = _build_point_region(region_h, region_w, color, want_overlap, rng)
                else:
                    region = _build_line_region(region_h, region_w, color, want_overlap, rng)
            except RuntimeError:
                ok = False
                break
            regions.append((region, color, want_overlap))
        if not ok:
            continue

        grid = np.full((grid_h, grid_w), BLACK, dtype=int)
        expected = np.full((grid_h, grid_w), BLACK, dtype=int)
        split_r = region_heights[0]
        split_c = region_widths[0]
        grid[split_r, :] = GREY
        grid[:, split_c] = GREY
        expected[split_r, :] = GREY
        expected[:, split_c] = GREY

        anchors = [
            (0, 0),
            (0, split_c + 1),
            (split_r + 1, 0),
            (split_r + 1, split_c + 1),
        ]
        for (region, color, want_overlap), (r0, c0) in zip(regions, anchors):
            rh, rw = region.shape
            grid[r0 : r0 + rh, c0 : c0 + rw] = region
            expected[r0 : r0 + rh, c0 : c0 + rw] = color if want_overlap else BLACK

        produced = main(grid)
        if not np.array_equal(produced, expected):
            continue

        counts = Counter(spec[1] for spec in specs)
        if counts[True] < 1 or counts[False] < 1:
            continue
        return grid

    raise RuntimeError("generate_input failed to produce a valid sample")
