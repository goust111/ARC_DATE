from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SPRITE_COLORS = {1, 8}
MARKER_COLOR = 1
ANCHOR_COLOR = 1
CONTROL_COLORS = (2, 3, 4, 7)
MAX_GEN_ATTEMPTS = 500


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_to_bbox(grid: np.ndarray, coords: np.ndarray, fill: int = BG) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(fill), dtype=int)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    out[rr, cc] = grid[coords[:, 0], coords[:, 1]]
    return out


def components_of_colors(
    grid: np.ndarray, allowed: set[int], connectivity: int = 4
) -> list[np.ndarray]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) not in allowed:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) in allowed:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2-D grid"

    control_points = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere(np.isin(g, CONTROL_COLORS))
    ]
    assert len(control_points) == 4, "expected four control pixels"
    assert len({color for _, _, color in control_points}) == 4, "control colors must be distinct"

    top = min(control_points, key=lambda item: item[0])
    bottom = max(control_points, key=lambda item: item[0])
    left = min(control_points, key=lambda item: item[1])
    right = max(control_points, key=lambda item: item[1])
    assert len({top[:2], bottom[:2], left[:2], right[:2]}) == 4, "control roles must be unique"

    top_r, bottom_r = int(top[0]), int(bottom[0])
    left_c, right_c = int(left[1]), int(right[1])
    assert top_r < bottom_r and left_c < right_c

    inner = g[top_r : bottom_r + 1, left_c : right_c + 1]
    comps = components_of_colors(inner, SPRITE_COLORS, connectivity=4)
    sprite_cells: np.ndarray | None = None
    anchor: tuple[int, int] | None = None

    for comp in comps:
        abs_comp = comp + np.array([top_r, left_c])
        colors = {int(g[r, c]) for r, c in abs_comp}
        if int(abs_comp.shape[0]) == 1 and colors == {ANCHOR_COLOR}:
            ar, ac = map(int, abs_comp[0])
            if all(
                not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]) or int(g[nr, nc]) == BG
                for nr, nc in neighbors4(ar, ac)
            ):
                anchor = (ar, ac)
                continue
        if colors <= SPRITE_COLORS and 1 in colors and 8 in colors:
            if sprite_cells is not None:
                raise AssertionError("expected a single multi-color sprite")
            sprite_cells = abs_comp

    assert sprite_cells is not None, "sprite not found"
    assert anchor is not None, "anchor not found"

    marker_cells = [(int(r), int(c)) for r, c in sprite_cells if int(g[r, c]) == MARKER_COLOR]
    assert len(marker_cells) == 1, "sprite must have exactly one blue marker"
    marker = marker_cells[0]

    same_row = marker[0] == anchor[0] and marker[1] != anchor[1]
    same_col = marker[1] == anchor[1] and marker[0] != anchor[0]
    assert same_row ^ same_col, "marker and anchor must share exactly one axis"

    sprite_bbox = bbox_coords(sprite_cells)
    tile = crop_to_bbox(g, sprite_cells, fill=BG)

    return {
        "grid": g,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "rect": (top_r, left_c, bottom_r, right_c),
        "sprite_cells": sprite_cells,
        "anchor": anchor,
        "marker": marker,
        "axis": "h" if same_row else "v",
        "bbox": sprite_bbox,
        "tile": tile,
    }


def periodic_fill(
    rect: tuple[int, int, int, int], tile: np.ndarray, phase_origin: tuple[int, int]
) -> np.ndarray:
    top_r, left_c, bottom_r, right_c = rect
    h, w = tile.shape
    out = np.zeros((bottom_r - top_r + 1, right_c - left_c + 1), dtype=int)
    origin_r, origin_c = phase_origin
    for rr, r in enumerate(range(top_r, bottom_r + 1)):
        for cc, c in enumerate(range(left_c, right_c + 1)):
            out[rr, cc] = int(tile[(r - origin_r) % h, (c - origin_c) % w])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the marked blue-cyan sprite to an internal blue anchor, then tile its aligned pattern across the control rectangle.

    Input structure:
    - Black background with four distinct control pixels in colors {2,3,4,7} at the unique top, bottom, left, and right extremes.
    - The control extrema define one rectangle containing a single 4-connected blue/cyan sprite and one isolated blue anchor pixel.
    - The sprite has exactly one blue marker pixel; the anchor shares either its row or its column.

    Rule:
    - Use the four control extrema to recover the rectangle bounds and their side colors.
    - Find the multi-color sprite, the isolated blue anchor, and the sprite's unique blue marker.
    - Slide the whole sprite horizontally when marker and anchor are in the same row, otherwise vertically when they are in the same column, until the marker lands on the anchor.
    - Treat the aligned sprite's tight bounding box as one repeating tile and extend that periodic pattern over the whole rectangle, then overwrite the rectangle border with the four control colors.

    Key insight:
    - The controls define the fill region, but the anchor fixes the tiling phase by choosing where one aligned copy of the sprite's bbox sits before repetition.
    """
    info = parse_input(np.asarray(grid, dtype=int))
    g = info["grid"]
    top_r, left_c, bottom_r, right_c = info["rect"]
    sprite_cells = info["sprite_cells"]
    marker_r, marker_c = info["marker"]
    anchor_r, anchor_c = info["anchor"]
    bbox_r0, bbox_c0, bbox_r1, bbox_c1 = info["bbox"]
    tile = info["tile"]

    if info["axis"] == "h":
        dr, dc = 0, anchor_c - marker_c
    else:
        dr, dc = anchor_r - marker_r, 0

    aligned_origin = (bbox_r0 + dr, bbox_c0 + dc)
    tiled = periodic_fill(info["rect"], tile, aligned_origin)

    out = g.copy()
    out[top_r : bottom_r + 1, left_c : right_c + 1] = tiled

    out[top_r, left_c : right_c + 1] = int(info["top"][2])
    out[bottom_r, left_c : right_c + 1] = int(info["bottom"][2])
    out[top_r + 1 : bottom_r, left_c] = int(info["left"][2])
    out[top_r + 1 : bottom_r, right_c] = int(info["right"][2])
    return out


def random_connected_mask(h: int, w: int) -> np.ndarray | None:
    area_min = max(4, (h * w) // 2)
    area_max = h * w - 1
    target = random.randint(area_min, area_max)

    for _ in range(120):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(cells) != target or len(rows) != h or len(cols) != w:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if mask.all():
            continue
        return mask
    return None


def random_sprite_tile(h: int, w: int) -> np.ndarray | None:
    mask = random_connected_mask(h, w)
    if mask is None:
        return None

    tile = np.zeros((h, w), dtype=int)
    teal_cells = list(map(tuple, np.argwhere(mask)))
    marker = random.choice(teal_cells)
    for r, c in teal_cells:
        tile[r, c] = 8
    tile[marker] = 1
    return tile


def tile_coords_nonzero(tile: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {
        (top + int(r), left + int(c))
        for r, c in np.argwhere(tile != BG)
    }


def choose_shift_and_anchor(
    rect: tuple[int, int, int, int],
    tile: np.ndarray,
    bbox_origin: tuple[int, int],
    axis: str,
    marker_rel: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    top_r, left_c, bottom_r, right_c = rect
    h, w = tile.shape
    aligned_r0, aligned_c0 = bbox_origin
    marker_abs = (aligned_r0 + marker_rel[0], aligned_c0 + marker_rel[1])

    candidates = []
    if axis == "h":
        for delta in range(-(w * 2), w * 2 + 1):
            if delta == 0 or delta % w == 0 or abs(delta) < 2:
                continue
            inp_c0 = aligned_c0 - delta
            if inp_c0 < left_c + 1 or inp_c0 + w - 1 > right_c - 1:
                continue
            inp_cells = tile_coords_nonzero(tile, aligned_r0, inp_c0)
            if marker_abs in inp_cells:
                continue
            if any(abs(r - marker_abs[0]) + abs(c - marker_abs[1]) == 1 for r, c in inp_cells):
                continue
            candidates.append(((aligned_r0, inp_c0), marker_abs))
    else:
        for delta in range(-(h * 2), h * 2 + 1):
            if delta == 0 or delta % h == 0 or abs(delta) < 2:
                continue
            inp_r0 = aligned_r0 - delta
            if inp_r0 < top_r + 1 or inp_r0 + h - 1 > bottom_r - 1:
                continue
            inp_cells = tile_coords_nonzero(tile, inp_r0, aligned_c0)
            if marker_abs in inp_cells:
                continue
            if any(abs(r - marker_abs[0]) + abs(c - marker_abs[1]) == 1 for r, c in inp_cells):
                continue
            candidates.append(((inp_r0, aligned_c0), marker_abs))

    if not candidates:
        return None
    return random.choice(candidates)


def build_expected_input() -> np.ndarray | None:
    tile_h = random.randint(3, 4)
    tile_w = random.randint(3, 4)
    repeat_h = random.randint(2, 4)
    repeat_w = random.randint(2, 4)
    rect_h = tile_h * repeat_h
    rect_w = tile_w * repeat_w

    grid_h = rect_h + random.randint(2, 6)
    grid_w = rect_w + random.randint(2, 6)
    if grid_h > 30 or grid_w > 30:
        return None

    top_r = random.randint(1, grid_h - rect_h - 1)
    left_c = random.randint(1, grid_w - rect_w - 1)
    bottom_r = top_r + rect_h - 1
    right_c = left_c + rect_w - 1
    rect = (top_r, left_c, bottom_r, right_c)

    tile = random_sprite_tile(tile_h, tile_w)
    if tile is None:
        return None
    marker_rel = tuple(map(int, np.argwhere(tile == MARKER_COLOR)[0]))

    phase_r = random.randrange(tile_h)
    phase_c = random.randrange(tile_w)
    base_r_choices = [
        r for r in range(top_r + phase_r, bottom_r - tile_h + 2, tile_h)
    ]
    base_c_choices = [
        c for c in range(left_c + phase_c, right_c - tile_w + 2, tile_w)
    ]
    if not base_r_choices or not base_c_choices:
        return None

    aligned_r0 = random.choice(base_r_choices)
    aligned_c0 = random.choice(base_c_choices)
    axis = random.choice(("h", "v"))
    choice = choose_shift_and_anchor(rect, tile, (aligned_r0, aligned_c0), axis, marker_rel)
    if choice is None:
        return None
    (inp_r0, inp_c0), anchor = choice

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for r, c in np.argwhere(tile != BG):
        grid[inp_r0 + int(r), inp_c0 + int(c)] = int(tile[int(r), int(c)])
    grid[anchor] = ANCHOR_COLOR

    top_color, bottom_color, left_color, right_color = random.sample(CONTROL_COLORS, 4)
    top_candidates = [c for c in range(left_c + 1, right_c) if c != anchor[1]]
    bottom_candidates = [c for c in range(left_c + 1, right_c) if c != anchor[1]]
    left_candidates = [r for r in range(top_r + 1, bottom_r) if r != anchor[0]]
    right_candidates = [r for r in range(top_r + 1, bottom_r) if r != anchor[0]]
    if not top_candidates or not bottom_candidates or not left_candidates or not right_candidates:
        return None

    grid[top_r, random.choice(top_candidates)] = top_color
    grid[bottom_r, random.choice(bottom_candidates)] = bottom_color
    grid[random.choice(left_candidates), left_c] = left_color
    grid[random.choice(right_candidates), right_c] = right_color

    if not validate_generated_case(grid):
        return None
    return grid


def fill_with_phase_for_validation(
    info: dict, phase_origin: tuple[int, int]
) -> np.ndarray:
    out = info["grid"].copy()
    top_r, left_c, bottom_r, right_c = info["rect"]
    tiled = periodic_fill(info["rect"], info["tile"], phase_origin)
    out[top_r : bottom_r + 1, left_c : right_c + 1] = tiled
    out[top_r, left_c : right_c + 1] = int(info["top"][2])
    out[bottom_r, left_c : right_c + 1] = int(info["bottom"][2])
    out[top_r + 1 : bottom_r, left_c] = int(info["left"][2])
    out[top_r + 1 : bottom_r, right_c] = int(info["right"][2])
    return out


def validate_generated_case(grid: np.ndarray) -> bool:
    try:
        info = parse_input(grid)
    except AssertionError:
        return False

    top_r, left_c, bottom_r, right_c = info["rect"]
    tile = info["tile"]
    tile_h, tile_w = tile.shape
    rect_h = bottom_r - top_r + 1
    rect_w = right_c - left_c + 1
    if rect_h % tile_h != 0 or rect_w % tile_w != 0:
        return False

    sprite_bbox = info["bbox"]
    if info["axis"] == "h":
        delta = info["anchor"][1] - info["marker"][1]
        aligned_origin = (sprite_bbox[0], sprite_bbox[1] + delta)
    else:
        delta = info["anchor"][0] - info["marker"][0]
        aligned_origin = (sprite_bbox[0] + delta, sprite_bbox[1])
    if delta == 0:
        return False

    expected = fill_with_phase_for_validation(info, aligned_origin)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    naive_top_left = fill_with_phase_for_validation(info, (top_r, left_c))
    if np.array_equal(actual, naive_top_left):
        return False

    naive_input_bbox = fill_with_phase_for_validation(info, (sprite_bbox[0], sprite_bbox[1]))
    if np.array_equal(actual, naive_input_bbox):
        return False

    inner = actual[top_r + 1 : bottom_r, left_c + 1 : right_c]
    if inner.size == 0 or len(set(map(int, inner.flatten()))) < 2:
        return False
    if np.count_nonzero(inner == MARKER_COLOR) == 0 or np.count_nonzero(inner == 8) == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = build_expected_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-tiling ARC input")
