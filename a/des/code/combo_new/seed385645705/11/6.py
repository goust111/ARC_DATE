from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
ORANGE = 7
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3

BORDER_COLORS = (1, 2, 3, 5, 6, 8, 9)

GEN_RETRIES = 500
STAMP_ATTEMPTS = 100
PLACEMENT_ATTEMPTS = 200
GRID_MAX = 30

FALLBACK_STAMPS = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -int(turns) % 4)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _marker_side_and_offset(
    marker: tuple[int, int], r0: int, c0: int, r1: int, c1: int
) -> tuple[int, int]:
    mr, mc = marker
    if mr == r0 and c0 < mc < c1:
        return TOP, mc - (c0 + 1)
    if mc == c1 and r0 < mr < r1:
        return RIGHT, mr - (r0 + 1)
    if mr == r1 and c0 < mc < c1:
        return BOTTOM, (c1 - 1) - mc
    if mc == c0 and r0 < mr < r1:
        return LEFT, (r1 - 1) - mr
    raise AssertionError("yellow marker must lie on a non-corner border edge")


def _extract_stamp_mask(interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    comps = _components(interior == ORANGE)
    assert len(comps) == 1, "each frame interior must contain exactly one orange stamp"
    coords = comps[0]
    assert 2 <= len(coords) <= 4, "stamp area must be between 2 and 4 cells"
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, coords


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    infos: list[dict[str, object]] = []
    for comp in _components((g != BG) & (g != ORANGE)):
        values = {int(g[r, c]) for r, c in comp}
        if YELLOW not in values or len(values) != 2:
            continue
        border_color = next(v for v in values if v != YELLOW)
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _border_cells(r0, c0, r1, c1)
        pts = {(int(r), int(c)) for r, c in comp}
        if pts != border:
            continue

        markers = [(int(r), int(c)) for r, c in comp if int(g[r, c]) == YELLOW]
        if len(markers) != 1:
            continue
        side, offset = _marker_side_and_offset(markers[0], r0, c0, r1, c1)

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        stamp_mask, stamp_coords = _extract_stamp_mask(interior)
        abs_stamp = stamp_coords + np.array([r0 + 1, c0 + 1])

        infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border_color": border_color,
                "marker": markers[0],
                "start_side": side,
                "start_offset": offset,
                "stamp_mask": stamp_mask,
                "stamp_cells": abs_stamp,
            }
        )

    infos.sort(key=lambda info: tuple(int(v) for v in info["bbox"]))
    return infos


def _placement_origin(
    side: int, offset: int, bbox: tuple[int, int, int, int], mask: np.ndarray
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    mh, mw = mask.shape
    if side == TOP:
        return r0 + 1, c0 + 1 + offset
    if side == RIGHT:
        return r0 + 1 + offset, c1 - mw
    if side == BOTTOM:
        return r1 - mh, c1 - (offset + mw)
    if side == LEFT:
        return r1 - (offset + mh), c0 + 1
    raise AssertionError("invalid side")


def _tile_stamp_along_frame(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    start_side: int,
    start_offset: int,
    stamp_mask: np.ndarray,
) -> int:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1

    current_side = int(start_side)
    current_offset = int(start_offset)
    looped = False
    placed = 0

    while True:
        rot = (current_side - start_side) % 4
        oriented = _rotate_cw(stamp_mask, rot)
        span = oriented.shape[1] if current_side in (TOP, BOTTOM) else oriented.shape[0]
        seg_len = inner_w if current_side in (TOP, BOTTOM) else inner_h
        limit = start_offset if looped and current_side == start_side else seg_len

        if current_offset + span <= limit:
            rr, cc = _placement_origin(current_side, current_offset, bbox, oriented)
            patch = out[rr : rr + oriented.shape[0], cc : cc + oriented.shape[1]]
            patch[oriented] = ORANGE
            current_offset += span
            placed += 1
            continue

        if looped and current_side == start_side:
            break

        current_side = (current_side + 1) % 4
        current_offset = 0
        if current_side == start_side:
            if looped:
                break
            looped = True

    return placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each central orange stamp by repeated copies tiled clockwise along the inner wall of its marked rectangular frame.

    Input structure:
    - The background is black.
    - One or more one-pixel-thick rectangular frames appear in non-black colors, and each frame border contains exactly one yellow marker pixel.
    - Inside each frame there is a single compact 4-connected orange stamp of area 2-4, with the rest of the interior black.

    Rule:
    - Detect every rectangular frame, recover its border color, yellow marker position, and interior orange stamp shape.
    - Remove the original central stamp and repaint the yellow marker back to the frame color.
    - Starting from the marker's side, place the observed stamp against the frame's inner wall and continue clockwise with no gaps between consecutive placements on the same wall.
    - Whenever the path turns a corner, rotate the stamp 90 degrees clockwise to follow the next wall; stop when the remaining space before the start point can no longer fit a full stamp.

    Key insight:
    - The orange object is a reusable template, the yellow border pixel is the unique starting anchor, and the clockwise wall-following tiling determines both placement order and 90-degree corner rotations.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    assert frames, "expected at least one marked rectangular frame"

    for info in frames:
        r0, c0, r1, c1 = info["bbox"]
        border_color = int(info["border_color"])
        mr, mc = info["marker"]
        stamp_mask = np.asarray(info["stamp_mask"], dtype=bool)
        stamp_cells = np.asarray(info["stamp_cells"], dtype=int)

        out[mr, mc] = border_color
        out[stamp_cells[:, 0], stamp_cells[:, 1]] = BG

        placed = _tile_stamp_along_frame(
            out,
            (int(r0), int(c0), int(r1), int(c1)),
            int(info["start_side"]),
            int(info["start_offset"]),
            stamp_mask,
        )
        assert placed > 0, "each frame should receive at least one tiled stamp"

    return out


def _random_stamp(rng: random.Random) -> np.ndarray:
    for _ in range(STAMP_ATTEMPTS):
        h = rng.randint(2, 3)
        w = rng.randint(2, 3)
        area = rng.randint(2, min(4, h * w))
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if 2 <= int(mask.sum()) <= 4:
            return _rotate_cw(mask, rng.randrange(4))
    return np.array(rng.choice(FALLBACK_STAMPS), copy=True)


def _central_stamp_position(
    rng: random.Random, inner_h: int, inner_w: int, stamp_h: int, stamp_w: int
) -> tuple[int, int]:
    row_span = inner_h - stamp_h
    col_span = inner_w - stamp_w
    base_r = row_span // 2
    base_c = col_span // 2

    min_r = 1 if row_span >= 2 else 0
    max_r = row_span - 1 if row_span >= 2 else row_span
    min_c = 1 if col_span >= 2 else 0
    max_c = col_span - 1 if col_span >= 2 else col_span

    row_lo = max(min_r, base_r - 1)
    row_hi = min(max_r, base_r + 1)
    col_lo = max(min_c, base_c - 1)
    col_hi = min(max_c, base_c + 1)

    rr = rng.randint(row_lo, row_hi) if row_lo <= row_hi else base_r
    cc = rng.randint(col_lo, col_hi) if col_lo <= col_hi else base_c
    return int(rr), int(cc)


def _make_frame_spec(
    rng: random.Random, border_color: int, start_side: int
) -> dict[str, object]:
    stamp_mask = _random_stamp(rng)
    stamp_h, stamp_w = stamp_mask.shape

    inner_h = rng.randint(max(6, stamp_h + 3), 10)
    inner_w = rng.randint(max(6, stamp_w + 3), 10)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    span = stamp_w if start_side in (TOP, BOTTOM) else stamp_h
    seg_len = inner_w if start_side in (TOP, BOTTOM) else inner_h
    start_offset = rng.randint(0, seg_len - span)
    center_r, center_c = _central_stamp_position(rng, inner_h, inner_w, stamp_h, stamp_w)

    return {
        "border_color": border_color,
        "start_side": start_side,
        "start_offset": start_offset,
        "stamp_mask": stamp_mask,
        "stamp_pos": (center_r, center_c),
        "outer_h": outer_h,
        "outer_w": outer_w,
    }


def _marker_from_spec(
    r0: int, c0: int, outer_h: int, outer_w: int, side: int, offset: int
) -> tuple[int, int]:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    if side == TOP:
        return r0, c0 + 1 + offset
    if side == RIGHT:
        return r0 + 1 + offset, c1
    if side == BOTTOM:
        return r1, c1 - 1 - offset
    if side == LEFT:
        return r1 - 1 - offset, c0
    raise AssertionError("invalid side")


def _draw_frame(grid: np.ndarray, spec: dict[str, object], r0: int, c0: int) -> None:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    color = int(spec["border_color"])

    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    mr, mc = _marker_from_spec(
        r0,
        c0,
        outer_h,
        outer_w,
        int(spec["start_side"]),
        int(spec["start_offset"]),
    )
    grid[mr, mc] = YELLOW

    stamp_mask = np.asarray(spec["stamp_mask"], dtype=bool)
    sr, sc = spec["stamp_pos"]
    rr = r0 + 1 + int(sr)
    cc = c0 + 1 + int(sc)
    patch = grid[rr : rr + stamp_mask.shape[0], cc : cc + stamp_mask.shape[1]]
    patch[stamp_mask] = ORANGE


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 2) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 2) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _grid_shape(rng: random.Random, count: int) -> tuple[int, int]:
    base = min(GRID_MAX, 15 + 4 * count)
    h = rng.randint(base, GRID_MAX)
    w = rng.randint(base, GRID_MAX)
    return h, w


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(1, 3)
    grid_h, grid_w = _grid_shape(rng, count)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    colors = list(rng.sample(BORDER_COLORS, count))
    side_order = [TOP, RIGHT, BOTTOM, LEFT]
    rng.shuffle(side_order)
    start_sides = side_order[:count]

    specs = [_make_frame_spec(rng, colors[i], start_sides[i]) for i in range(count)]
    specs.sort(key=lambda spec: int(spec["outer_h"]) * int(spec["outer_w"]), reverse=True)

    for spec in specs:
        placed = False
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        if outer_h + 2 > grid_h or outer_w + 2 > grid_w:
            return None

        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(1, grid_h - outer_h - 1)
            c0 = rng.randint(1, grid_w - outer_w - 1)
            if not _can_place(occupied, r0, c0, outer_h, outer_w):
                continue
            _draw_frame(grid, spec, r0, c0)
            _mark_occupied(occupied, r0, c0, outer_h, outer_w)
            placed = True
            break
        if not placed:
            return None

    infos = _find_frames(grid)
    if len(infos) != count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
