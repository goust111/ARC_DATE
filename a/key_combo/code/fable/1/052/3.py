import random
from collections import Counter, deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PIXEL_COLORS = [
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
FRAME_COLOR = Color.BLUE
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def monochromatic_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    seen = np.zeros(grid.shape, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q = deque([(rr, cc)])
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cd = q.popleft()
            comp.append((cr, cd))
            for nr, nc in neighbors4(cr, cd):
                if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and mask[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def _grey_bar(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = monochromatic_components(grid, Color.GREY)
    assert len(comps) == 1, "expected one grey magnet"
    comp = comps[0]
    box = bbox_coords(comp)
    r0, c0, r1, c1 = box
    expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert len(comp) == expected_area, "grey magnet must be a solid rectangle"
    return box


def _frame_from_component(
    grid: np.ndarray, bg: int, comp: list[tuple[int, int]]
) -> dict | None:
    if len(comp) < 8:
        return None
    r0, c0, r1, c1 = bbox_coords(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 3:
        return None

    color = int(grid[comp[0]])
    expected: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    comp_set = set(comp)
    if comp_set != expected:
        return None

    interior_pixels: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            value = int(grid[r, c])
            if value == color:
                return None
            if value != bg:
                interior_pixels.append((r, c))
    if len(interior_pixels) != 1:
        return None

    captive = interior_pixels[0]
    return {
        "key": (r0, c0, r1, c1, captive[0], captive[1]),
        "bbox": (r0, c0, r1, c1),
        "color": color,
        "cells": comp_set,
        "captive": captive,
    }


def _detect_frames(grid: np.ndarray, bg: int) -> dict:
    frames: dict[tuple[int, ...], dict] = {}
    for color in np.unique(grid):
        col = int(color)
        if col in (bg, Color.GREY):
            continue
        for comp in monochromatic_components(grid, col):
            frame = _frame_from_component(grid, bg, comp)
            if frame is not None:
                frames[frame["key"]] = frame
    return frames


def _free_pixels(
    grid: np.ndarray, bg: int, frames: dict
) -> list[tuple[int, int]]:
    frame_cells = set()
    captive_cells = set()
    for frame in frames.values():
        frame_cells |= frame["cells"]
        captive_cells.add(frame["captive"])

    free: list[tuple[int, int]] = []
    for r, c in np.argwhere((grid != bg) & (grid != Color.GREY)):
        cell = (int(r), int(c))
        if cell in frame_cells or cell in captive_cells:
            continue
        free.append(cell)
    return free


def _steps_to_grey(
    grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int]
) -> int | None:
    r, c = start
    dr, dc = direction
    rr = r + dr
    cc = c + dc
    step = 1
    while in_bounds(grid.shape, rr, cc):
        if int(grid[rr, cc]) == Color.GREY:
            return step - 1
        rr += dr
        cc += dc
        step += 1
    return None


def _choose_move(
    grid: np.ndarray, free_pixels: list[tuple[int, int]]
) -> dict:
    pixel_records = []
    for pixel in sorted(free_pixels):
        options = []
        for direction in DIRS:
            steps = _steps_to_grey(grid, pixel, direction)
            if steps is not None:
                options.append((steps, direction))
        assert options, "every free pixel must have a way to approach the grey mass"
        best_steps = min(steps for steps, _ in options)
        best_dirs = [direction for steps, direction in options if steps == best_steps]
        assert len(best_dirs) == 1, "direction choice must be unique"
        pixel_records.append((best_steps, pixel, best_dirs[0]))

    assert pixel_records, "need at least one free pixel to move"
    best_distance = min(distance for distance, _, _ in pixel_records)
    best_pixels = [(pixel, direction) for distance, pixel, direction in pixel_records if distance == best_distance]
    assert len(best_pixels) == 1, "closest free pixel must be unique"
    pixel, direction = best_pixels[0]
    return {"start": pixel, "direction": direction, "priority": best_distance}


def _slide_one(grid: np.ndarray, bg: int, frames: dict, move: dict) -> tuple[np.ndarray, dict]:
    out = grid.copy()
    start = move["start"]
    direction = move["direction"]
    dr, dc = direction
    color = int(out[start])
    frame_cells = {}
    for key, frame in frames.items():
        for cell in frame["cells"]:
            frame_cells[cell] = key

    stop = start
    result = "grey"
    if move["priority"] > 0:
        while True:
            nr = stop[0] + dr
            nc = stop[1] + dc
            assert in_bounds(out.shape, nr, nc), "sliding path must stay in bounds"
            cell = int(out[nr, nc])
            if cell == bg:
                stop = (nr, nc)
                continue
            if cell == Color.GREY:
                result = "grey"
                break
            if (nr, nc) in frame_cells:
                result = "frame"
                break
            raise AssertionError("free pixels should not block each other")

    if stop != start:
        out[start] = bg
    out[stop] = Color.GREY if result == "grey" else color

    unlocked = []
    for nr, nc in neighbors4(*stop):
        frame_key = frame_cells.get((nr, nc))
        if frame_key is not None and frame_key not in unlocked:
            unlocked.append(frame_key)

    for frame_key in unlocked:
        for rr, cc in frames[frame_key]["cells"]:
            out[rr, cc] = bg

    return out, {
        "start": start,
        "stop": stop,
        "direction": direction,
        "priority": move["priority"],
        "result": result,
        "unlocked": tuple(unlocked),
    }


def _simulate(grid: np.ndarray, *, with_trace: bool = False):
    g = np.asarray(grid, dtype=int).copy()
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    _grey_bar(g)

    initial_frames = _detect_frames(g, bg)
    assert initial_frames, "expected at least one frame"
    initial_captives = {frame["captive"] for frame in initial_frames.values()}

    trace = []
    for _ in range(g.size * 4):
        frames = _detect_frames(g, bg)
        free_pixels = _free_pixels(g, bg, frames)
        if not free_pixels:
            break
        move = _choose_move(g, free_pixels)
        g, event = _slide_one(g, bg, frames, move)
        trace.append(event)
    else:
        raise AssertionError("simulation did not terminate")

    final_frames = _detect_frames(g, bg)
    free_pixels = _free_pixels(g, bg, final_frames)
    assert not free_pixels, "all free pixels should eventually be resolved"

    info = {
        "initial_frame_keys": tuple(initial_frames.keys()),
        "final_frame_keys": tuple(final_frames.keys()),
        "initial_captives": tuple(sorted(initial_captives)),
        "unlocked_frame_keys": tuple(
            key
            for event in trace
            for key in event["unlocked"]
        ),
    }
    if with_trace:
        return g, trace, info
    return g


def _lane_cell(
    orientation: str,
    side: str,
    boundary: int,
    orth: int,
    distance: int,
) -> tuple[int, int]:
    if orientation == "vertical":
        if side == "left":
            return orth, boundary - distance
        return orth, boundary + distance
    if side == "top":
        return boundary - distance, orth
    return boundary + distance, orth


def _paint_frame(
    grid: np.ndarray,
    orientation: str,
    side: str,
    boundary: int,
    orth: int,
    center_distance: int,
    captive_color: int,
) -> tuple[int, int, int, int, int, int]:
    cr, cc = _lane_cell(orientation, side, boundary, orth, center_distance)
    coords = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = cr + dr
            nc = cc + dc
            assert in_bounds(grid.shape, rr, nc)
            if abs(dr) == 1 or abs(dc) == 1:
                assert int(grid[rr, nc]) == Color.BLACK
                grid[rr, nc] = FRAME_COLOR
                coords.append((rr, nc))
            else:
                assert int(grid[rr, nc]) == Color.BLACK
                grid[rr, nc] = int(captive_color)
    return cr - 1, cc - 1, cr + 1, cc + 1, cr, cc


def _paint_free_pixel(
    grid: np.ndarray,
    orientation: str,
    side: str,
    boundary: int,
    orth: int,
    distance: int,
    color: int,
) -> tuple[int, int]:
    rr, cc = _lane_cell(orientation, side, boundary, orth, distance)
    assert in_bounds(grid.shape, rr, cc)
    assert int(grid[rr, cc]) == Color.BLACK
    grid[rr, cc] = int(color)
    return rr, cc


def _spaced_choices(values: list[int], count: int) -> list[int] | None:
    pool = values[:]
    random.shuffle(pool)
    chosen: list[int] = []
    for value in pool:
        if all(abs(value - prev) >= 3 for prev in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _build_candidate() -> np.ndarray | None:
    orientation = random.choice(["vertical", "horizontal"])
    if orientation == "vertical":
        bar_h = random.randint(10, 15)
        bar_w = random.randint(2, 3)
        primary_side = random.choice(["left", "right"])
        primary_margin = random.randint(14, 16)
        secondary_margin = random.randint(8, 11)
        top_margin = random.randint(2, 4)
        bottom_margin = random.randint(2, 4)
        left_margin = primary_margin if primary_side == "left" else secondary_margin
        right_margin = primary_margin if primary_side == "right" else secondary_margin
        h = top_margin + bar_h + bottom_margin
        w = left_margin + bar_w + right_margin
        if h > 30 or w > 30:
            return None
        grid = np.full((h, w), Color.BLACK, dtype=int)
        r0 = top_margin
        c0 = left_margin
        grid[r0 : r0 + bar_h, c0 : c0 + bar_w] = Color.GREY
        bar_box = (r0, c0, r0 + bar_h - 1, c0 + bar_w - 1)
        available = list(range(r0 + 1, r0 + bar_h - 1))
    else:
        bar_h = random.randint(2, 3)
        bar_w = random.randint(10, 15)
        primary_side = random.choice(["top", "bottom"])
        primary_margin = random.randint(14, 16)
        secondary_margin = random.randint(8, 11)
        left_margin = random.randint(2, 4)
        right_margin = random.randint(2, 4)
        top_margin = primary_margin if primary_side == "top" else secondary_margin
        bottom_margin = primary_margin if primary_side == "bottom" else secondary_margin
        h = top_margin + bar_h + bottom_margin
        w = left_margin + bar_w + right_margin
        if h > 30 or w > 30:
            return None
        grid = np.full((h, w), Color.BLACK, dtype=int)
        r0 = top_margin
        c0 = left_margin
        grid[r0 : r0 + bar_h, c0 : c0 + bar_w] = Color.GREY
        bar_box = (r0, c0, r0 + bar_h - 1, c0 + bar_w - 1)
        available = list(range(c0 + 1, c0 + bar_w - 1))

    lane_count = random.choice([3, 4])
    orths = _spaced_choices(available, lane_count)
    if orths is None:
        return None

    if orientation == "vertical":
        primary_boundary = bar_box[1]
        secondary_boundary = bar_box[3]
        if primary_side == "right":
            primary_boundary, secondary_boundary = secondary_boundary, primary_boundary
        secondary_side = "right" if primary_side == "left" else "left"
    else:
        primary_boundary = bar_box[0]
        secondary_boundary = bar_box[2]
        if primary_side == "bottom":
            primary_boundary, secondary_boundary = secondary_boundary, primary_boundary
        secondary_side = "bottom" if primary_side == "top" else "top"

    lane_specs = [
        (primary_side, "active2"),
        (secondary_side, random.choice(["active1", "direct"])),
        (random.choice([primary_side, secondary_side]), "passive"),
    ]
    if lane_count == 4:
        lane_specs.append((random.choice([primary_side, secondary_side]), random.choice(["active1", "direct", "passive"])))

    random.shuffle(lane_specs)
    used_pixels = []
    for orth, (side, lane_type) in zip(orths, lane_specs):
        if side == primary_side:
            boundary = primary_boundary
            max_margin = primary_margin
        else:
            boundary = secondary_boundary
            max_margin = secondary_margin

        if lane_type == "active2":
            base = random.randint(4, 6)
            if base + 9 > max_margin:
                return None
            _paint_frame(
                grid,
                orientation,
                side,
                boundary,
                orth,
                base,
                random.choice(PIXEL_COLORS),
            )
            _paint_free_pixel(
                grid,
                orientation,
                side,
                boundary,
                orth,
                base + 3,
                random.choice(PIXEL_COLORS),
            )
            _paint_frame(
                grid,
                orientation,
                side,
                boundary,
                orth,
                base + 6,
                random.choice(PIXEL_COLORS),
            )
            used_pixels.append(
                _paint_free_pixel(
                    grid,
                    orientation,
                    side,
                    boundary,
                    orth,
                    base + 9,
                    random.choice(PIXEL_COLORS),
                )
            )
        elif lane_type == "active1":
            base = random.randint(4, min(6, max_margin - 3))
            _paint_frame(
                grid,
                orientation,
                side,
                boundary,
                orth,
                base,
                random.choice(PIXEL_COLORS),
            )
            used_pixels.append(
                _paint_free_pixel(
                    grid,
                    orientation,
                    side,
                    boundary,
                    orth,
                    base + 3,
                    random.choice(PIXEL_COLORS),
                )
            )
        elif lane_type == "direct":
            distance = random.randint(3, min(7, max_margin))
            used_pixels.append(
                _paint_free_pixel(
                    grid,
                    orientation,
                    side,
                    boundary,
                    orth,
                    distance,
                    random.choice(PIXEL_COLORS),
                )
            )
        else:
            center = random.randint(4, max(4, min(9, max_margin - 1)))
            _paint_frame(
                grid,
                orientation,
                side,
                boundary,
                orth,
                center,
                random.choice(PIXEL_COLORS),
            )

    if not used_pixels:
        return None
    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        out, trace, info = _simulate(grid, with_trace=True)
    except AssertionError:
        return False

    initial_frames = set(info["initial_frame_keys"])
    final_frames = set(info["final_frame_keys"])
    unlocked = [key for key in info["unlocked_frame_keys"]]
    unlocked_set = set(unlocked)
    initial_captives = set(info["initial_captives"])

    if len(initial_frames) < 2:
        return False
    if len(unlocked_set) < 2:
        return False
    if not final_frames:
        return False
    if not final_frames < initial_frames:
        return False
    if not any(event["result"] == "frame" for event in trace):
        return False
    if not any(event["result"] == "grey" for event in trace):
        return False
    if not any(event["start"] in initial_captives for event in trace):
        return False
    if np.count_nonzero(out == Color.GREY) <= np.count_nonzero(grid == Color.GREY):
        return False

    untouched = final_frames
    for frame_key in untouched:
        r0, c0, r1, c1, cr, cc = frame_key
        if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == grid[r0 : r1 + 1, c0 : c1 + 1]):
            return False
        if (cr, cc) not in initial_captives:
            return False

    moved_released = False
    for event in trace:
        if event["start"] in initial_captives:
            moved_released = True
            break
    if not moved_released:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid chain-adsorption input")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide free pixels toward the grey magnet, breaking adjacent cages and releasing new pixels into the same chain.

    Input structure:
    - The background is black and there is exactly one solid grey rectangle or bar.
    - Other visible colored cells are either free single pixels, or one-cell-thick hollow square frames with exactly one colored captive pixel inside each frame.
    - The frames are separate from the grey bar and from one another.

    Rule:
    - Repeatedly choose the unique free pixel with the smallest axial distance to the current grey mass, and move it in the unique cardinal direction that would reach the grey mass fastest.
    - While sliding, if the next occupied cell is a frame wall, the pixel stops just before that wall and keeps its original color; if the next occupied cell is grey, it stops there and turns grey.
    - After every stop, any frame that is 4-neighbor adjacent to the stopped pixel disappears, and its captive pixel becomes a new free pixel; continue until no free pixels remain. Unopened frames and their captive pixels stay unchanged.

    Key insight:
    - The process is sequential rather than parallel: earlier absorptions reshape the grey pile, and frame breaks inject new movable pixels that change the rest of the chain.
    """
    return _simulate(grid)
