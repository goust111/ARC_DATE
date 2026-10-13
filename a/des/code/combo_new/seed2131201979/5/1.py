from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


PALETTE_ORDER = ("up", "right", "down", "left")
DIR_TO_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
ALL_COLORS = tuple(range(10))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def frame_info_from_component(
    coords: list[tuple[int, int]]
) -> dict[str, object] | None:
    r0, c0, r1, c1 = bbox_of(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    pts = set(coords)
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    if not pts <= border:
        return None

    missing = list(border - pts)
    if len(missing) != 1:
        return None
    if len(pts) != len(border) - 1:
        return None

    gap = missing[0]
    gr, gc = gap
    is_corner = gr in (r0, r1) and gc in (c0, c1)
    if is_corner:
        return None

    directions = []
    if gr == r0:
        directions.append("up")
    if gc == c1:
        directions.append("right")
    if gr == r1:
        directions.append("down")
    if gc == c0:
        directions.append("left")
    if len(directions) != 1:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gr), int(gc)),
        "direction": directions[0],
        "area": int(len(pts)),
    }


def find_frames(
    grid: np.ndarray, bg: int
) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    best_color = None
    best_key = None
    best_frames: list[dict[str, object]] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        frames: list[dict[str, object]] = []
        total_area = 0
        for comp in components_mask(g == color):
            info = frame_info_from_component(comp)
            if info is None:
                continue
            frames.append(info)
            total_area += int(info["area"])
        if not frames:
            continue
        key = (len(frames), total_area)
        if best_key is None or key > best_key:
            best_key = key
            best_color = color
            best_frames = frames

    assert best_color is not None and best_frames, "expected at least one gapped frame"
    return int(best_color), best_frames


def find_palette(
    grid: np.ndarray, bg: int, frames: list[dict[str, object]]
) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    outside = g != bg
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    candidates: list[list[int]] = []
    for comp in components_mask(outside):
        if len(comp) != 4:
            continue
        r0, c0, r1, c1 = bbox_of(comp)
        if r0 == r1 and c1 - c0 == 3:
            ordered = [int(g[r0, c]) for c in range(c0, c1 + 1)]
        elif c0 == c1 and r1 - r0 == 3:
            ordered = [int(g[r, c0]) for r in range(r0, r1 + 1)]
        else:
            continue
        if len(set(ordered)) != 4:
            continue
        candidates.append(ordered)

    assert len(candidates) == 1, "expected a unique 4-color palette line"
    return {direction: color for direction, color in zip(PALETTE_ORDER, candidates[0])}


def draw_ray(
    out: np.ndarray,
    start: tuple[int, int],
    direction: str,
    length: int,
    color: int,
) -> None:
    dr, dc = DIR_TO_DELTA[direction]
    r, c = start
    h, w = out.shape
    for step in range(length):
        rr = r + dr * step
        cc = c + dc * step
        assert 0 <= rr < h and 0 <= cc < w, "ray must stay within the grid"
        out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the palette to color a ray that exits each gapped frame in its opening direction.

    Input structure:
    - The most common color is the background.
    - Several one-pixel-thick rectangular hollow frames share one color; each frame
      has exactly one missing border pixel on one side.
    - Each frame interior contains a few non-background, non-frame pixels, and one
      separate palette component is a row or column of four distinct single pixels.

    Rule:
    - Identify the palette order as up, right, down, left when read left-to-right
      or top-to-bottom.
    - For every frame, locate the missing border pixel and use its side as the
      direction key into the palette.
    - Count the non-background pixels inside that frame and draw an outward ray of
      that length starting at the gap cell, using the palette color for that side.

    Key insight:
    - The output combines three independent cues: palette order gives the
      direction-to-color mapping, the gap determines which direction to query, and
      the number of interior pixels determines the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    frame_color, frames = find_frames(g, bg)
    palette = find_palette(g, bg, frames)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap = frame["gap"]
        direction = frame["direction"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        length = int(np.count_nonzero((interior != bg) & (interior != frame_color)))
        draw_ray(out, gap, direction, length, palette[direction])

    return out


GRID_SIDE_RANGE = (14, 30)
FRAME_COUNT_RANGE = (2, 4)
FRAME_H_RANGE = (3, 8)
FRAME_W_RANGE = (3, 8)
MAX_INTERIOR_PIXELS = 8
PALETTE_PLACEMENT_TRIES = 120
FRAME_PLACEMENT_TRIES = 200
GEN_RETRIES = 500


def halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((int(nr), int(nc)))
    return out


def choose_palette_colors(bg: int, frame_color: int) -> list[int]:
    pool = [color for color in ALL_COLORS if color not in (bg, frame_color)]
    return random.sample(pool, 4)


def choose_dot_colors(
    bg: int, frame_color: int, count: int
) -> list[int]:
    pool = [color for color in ALL_COLORS if color not in (bg, frame_color)]
    base_count = random.randint(1, min(4, count, len(pool)))
    base = random.sample(pool, base_count)
    return [random.choice(base) for _ in range(count)]


def place_palette(
    reserved: np.ndarray, h: int, w: int, colors: list[int], horizontal: bool
) -> dict[str, object] | None:
    for _ in range(PALETTE_PLACEMENT_TRIES):
        if horizontal:
            r = random.randint(1, h - 2)
            c = random.randint(1, w - 5)
            cells = [(r, c + i) for i in range(4)]
        else:
            r = random.randint(1, h - 5)
            c = random.randint(1, w - 2)
            cells = [(r + i, c) for i in range(4)]

        halo_cells = halo(set(cells), h, w)
        if any(reserved[r0, c0] for r0, c0 in halo_cells):
            continue

        rows, cols = zip(*halo_cells)
        reserved[np.array(rows), np.array(cols)] = True
        return {
            "cells": cells,
            "colors": list(colors),
        }
    return None


def sample_frame_shape() -> tuple[int, int, str, int]:
    frame_h = random.randint(*FRAME_H_RANGE)
    frame_w = random.randint(*FRAME_W_RANGE)
    direction = random.choice(PALETTE_ORDER)
    max_count = min(MAX_INTERIOR_PIXELS, (frame_h - 2) * (frame_w - 2))
    count = random.randint(1, max_count)
    return frame_h, frame_w, direction, count


def choose_gap(
    top: int, left: int, frame_h: int, frame_w: int, direction: str
) -> tuple[int, int]:
    if direction == "up":
        return top, left + random.randint(1, frame_w - 2)
    if direction == "right":
        return top + random.randint(1, frame_h - 2), left + frame_w - 1
    if direction == "down":
        return top + frame_h - 1, left + random.randint(1, frame_w - 2)
    return top + random.randint(1, frame_h - 2), left


def frame_footprint(
    top: int,
    left: int,
    frame_h: int,
    frame_w: int,
    gap: tuple[int, int],
    direction: str,
    length: int,
) -> set[tuple[int, int]]:
    cells = {
        (r, c)
        for r in range(top, top + frame_h)
        for c in range(left, left + frame_w)
    }
    dr, dc = DIR_TO_DELTA[direction]
    for step in range(length):
        cells.add((gap[0] + dr * step, gap[1] + dc * step))
    return cells


def place_frame(
    reserved: np.ndarray, h: int, w: int
) -> dict[str, object] | None:
    for _ in range(FRAME_PLACEMENT_TRIES):
        frame_h, frame_w, direction, count = sample_frame_shape()
        top = random.randint(1, h - frame_h - 2)
        left = random.randint(1, w - frame_w - 2)
        gap = choose_gap(top, left, frame_h, frame_w, direction)
        footprint = frame_footprint(top, left, frame_h, frame_w, gap, direction, count)

        if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint):
            continue

        halo_cells = halo(footprint, h, w)
        if any(reserved[r0, c0] for r0, c0 in halo_cells):
            continue

        inner_cells = [
            (r, c)
            for r in range(top + 1, top + frame_h - 1)
            for c in range(left + 1, left + frame_w - 1)
        ]
        dots = random.sample(inner_cells, count)

        rows, cols = zip(*halo_cells)
        reserved[np.array(rows), np.array(cols)] = True
        return {
            "bbox": (top, left, top + frame_h - 1, left + frame_w - 1),
            "gap": gap,
            "direction": direction,
            "dot_cells": dots,
        }
    return None


def render_input(
    h: int,
    w: int,
    bg: int,
    frame_color: int,
    palette: dict[str, object],
    frames: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)

    for (r, c), color in zip(palette["cells"], palette["colors"]):
        grid[r, c] = int(color)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        grid[frame["gap"]] = bg
        for (r, c), color in zip(frame["dot_cells"], frame["dot_colors"]):
            grid[r, c] = int(color)

    return grid


def render_expected(
    inp: np.ndarray,
    palette_colors: list[int],
    frames: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    palette_map = {
        direction: color for direction, color in zip(PALETTE_ORDER, palette_colors)
    }
    for frame in frames:
        draw_ray(
            out,
            frame["gap"],
            frame["direction"],
            len(frame["dot_cells"]),
            palette_map[frame["direction"]],
        )
    return out


def gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    bg = random.choice(ALL_COLORS)
    frame_color = random.choice([color for color in ALL_COLORS if color != bg])
    palette_colors = choose_palette_colors(bg, frame_color)
    reserved = np.zeros((h, w), dtype=bool)

    palette_horizontal = random.choice([False, True])
    palette = place_palette(reserved, h, w, palette_colors, palette_horizontal)
    if palette is None:
        return None

    frame_count = random.randint(*FRAME_COUNT_RANGE)
    frames: list[dict[str, object]] = []
    for _ in range(frame_count):
        frame = place_frame(reserved, h, w)
        if frame is None:
            return None
        frame["dot_colors"] = choose_dot_colors(bg, frame_color, len(frame["dot_cells"]))
        frames.append(frame)

    inp = render_input(h, w, bg, frame_color, palette, frames)
    expected = render_expected(inp, palette_colors, frames)

    if get_bg(inp) != bg:
        return None
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
