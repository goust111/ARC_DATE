from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ALL_DIRECTIONS = ("up", "down", "left", "right")
BG_COLOR = 0
BRACKET_COLOR = 5
NON_BRACKET_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    assert connectivity == 4
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            seen[r0, c0] = True
            color = int(grid[r0, c0])
            q = deque([(r0, c0)])
            cells = [(r0, c0)]
            while q:
                r, c = q.popleft()
                for nr, nc in neighbors4(r, c):
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) == bg:
                        continue
                    if monochromatic and int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
                    cells.append((nr, nc))

            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "color": color if monochromatic else None,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _u_cells(box: tuple[int, int, int, int], direction: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if direction == "up":
        return (
            {(r, c0) for r in range(r0, r1 + 1)}
            | {(r, c1) for r in range(r0, r1 + 1)}
            | {(r1, c) for c in range(c0, c1 + 1)}
        )
    if direction == "down":
        return (
            {(r, c0) for r in range(r0, r1 + 1)}
            | {(r, c1) for r in range(r0, r1 + 1)}
            | {(r0, c) for c in range(c0, c1 + 1)}
        )
    if direction == "left":
        return (
            {(r0, c) for c in range(c0, c1 + 1)}
            | {(r1, c) for c in range(c0, c1 + 1)}
            | {(r, c1) for r in range(r0, r1 + 1)}
        )
    if direction == "right":
        return (
            {(r0, c) for c in range(c0, c1 + 1)}
            | {(r1, c) for c in range(c0, c1 + 1)}
            | {(r, c0) for r in range(r0, r1 + 1)}
        )
    raise ValueError(direction)


def _u_direction(component: dict) -> str | None:
    coords = {tuple(map(int, p)) for p in component["coords"]}
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    for direction in ALL_DIRECTIONS:
        if coords == _u_cells((r0, c0, r1, c1), direction):
            return direction
    return None


def _extract_instructions(grid: np.ndarray) -> tuple[int, list[dict], set[tuple[int, int]]]:
    bg = get_bg(grid)
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    instructions = []
    bracket_cells: set[tuple[int, int]] = set()

    for comp in comps:
        direction = _u_direction(comp)
        if direction is None:
            continue

        r0, c0, r1, c1 = comp["bbox"]
        coords = {tuple(map(int, p)) for p in comp["coords"]}
        marker_cells = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if (r, c) not in coords and int(grid[r, c]) not in (bg, int(comp["color"]))
        ]
        if len(marker_cells) != 1:
            continue

        marker = marker_cells[0]
        instructions.append(
            {
                "bracket_color": int(comp["color"]),
                "direction": direction,
                "marker": marker,
                "marker_color": int(grid[marker]),
                "bbox": tuple(map(int, comp["bbox"])),
                "coords": coords,
            }
        )
        bracket_cells |= coords

    assert instructions, "expected at least one U-bracket instruction"
    bracket_colors = {inst["bracket_color"] for inst in instructions}
    assert len(bracket_colors) == 1, "expected all brackets to share one color"
    marker_colors = [inst["marker_color"] for inst in instructions]
    assert len(marker_colors) == len(set(marker_colors)), "expected distinct marker colors"
    return bg, instructions, bracket_cells


def _slide_pixels(
    base: np.ndarray,
    movers: list[tuple[int, int]],
    direction: str,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = base.shape
    mover_set = set(movers)
    static = base != get_bg(base)
    placed: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if direction == "up":
        for c in range(w):
            write = 0
            for r in range(h):
                if static[r, c]:
                    write = r + 1
                elif (r, c) in mover_set:
                    placed.append(((r, c), (write, c)))
                    write += 1
    elif direction == "down":
        for c in range(w):
            write = h - 1
            for r in range(h - 1, -1, -1):
                if static[r, c]:
                    write = r - 1
                elif (r, c) in mover_set:
                    placed.append(((r, c), (write, c)))
                    write -= 1
    elif direction == "left":
        for r in range(h):
            write = 0
            for c in range(w):
                if static[r, c]:
                    write = c + 1
                elif (r, c) in mover_set:
                    placed.append(((r, c), (r, write)))
                    write += 1
    elif direction == "right":
        for r in range(h):
            write = w - 1
            for c in range(w - 1, -1, -1):
                if static[r, c]:
                    write = c - 1
                elif (r, c) in mover_set:
                    placed.append(((r, c), (r, write)))
                    write -= 1
    else:
        raise ValueError(direction)

    return placed


def _path_cells(start: tuple[int, int], end: tuple[int, int]) -> set[tuple[int, int]]:
    (r0, c0), (r1, c1) = start, end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return {(r0, c) for c in range(lo, hi + 1)}
    lo, hi = sorted((r0, r1))
    return {(r, c0) for r in range(lo, hi + 1)}


def _apply_instructions(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg, instructions, _bracket_cells = _extract_instructions(g)

    selected_colors = {inst["marker_color"] for inst in instructions}
    markers = {inst["marker"] for inst in instructions}
    base = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if (r, c) in markers:
                continue
            if int(g[r, c]) in selected_colors:
                base[r, c] = bg

    move_info: list[dict] = []
    destinations: dict[tuple[int, int], int] = {}
    for inst in instructions:
        color = inst["marker_color"]
        movers = [
            (r, c)
            for r in range(g.shape[0])
            for c in range(g.shape[1])
            if int(g[r, c]) == color and (r, c) != inst["marker"]
        ]
        assert movers, "each instruction must name at least one movable pixel"
        moves = _slide_pixels(base, movers, inst["direction"])
        move_info.append(
            {
                "color": color,
                "direction": inst["direction"],
                "marker": inst["marker"],
                "moves": moves,
                "paths": [_path_cells(start, end) for start, end in moves],
            }
        )
        for _start, end in moves:
            assert 0 <= end[0] < g.shape[0] and 0 <= end[1] < g.shape[1]
            assert int(base[end]) == bg, "movable pixels must settle in empty cells"
            assert end not in destinations, "different selected colors must not overlap after sliding"
            destinations[end] = color

    out = base.copy()
    for (r, c), color in destinations.items():
        out[r, c] = color
    return out, move_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each selected color in the direction encoded by its enclosing U-bracket.

    Input structure:
    - The background is the dominant color.
    - Several gray monochrome U-brackets appear, each missing exactly one side and containing one non-gray marker pixel.
    - Outside the brackets are loose pixels in the marker colors, plus unrelated distractor colors that are named by no bracket.

    Rule:
    - For each bracket, read its inner marker color and its opening direction.
    - Remove every other pixel of that marker color from its current position and slide those pixels individually in the opening direction.
    - Each moving pixel stops just before the grid edge, a bracket, a static pixel of any color, or an earlier-settled pixel of the same color; the marker pixel, the brackets, and all unselected colors stay fixed.

    Key insight:
    - A bracket is a complete instruction: the marker selects which color to act on, and the missing side selects how that color moves.
    """
    out, _move_info = _apply_instructions(np.asarray(grid, dtype=int))
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _paint_u(grid: np.ndarray, box: tuple[int, int, int, int], direction: str, color: int) -> set[tuple[int, int]]:
    cells = _u_cells(box, direction)
    for r, c in cells:
        grid[r, c] = color
    return cells


def _strict_inner_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _rand_box(h: int, w: int, direction: str) -> tuple[int, int, int, int]:
    if direction == "up":
        r0 = random.randint(1, max(1, h // 2 - 5))
        c0 = random.randint(1, w - 7)
    elif direction == "down":
        r0 = random.randint(max(1, h // 2), h - 7)
        c0 = random.randint(1, w - 7)
    elif direction == "left":
        r0 = random.randint(1, h - 7)
        c0 = random.randint(1, max(1, w // 2 - 5))
    else:
        r0 = random.randint(1, h - 7)
        c0 = random.randint(max(1, w // 2), w - 7)

    bh = random.randint(4, 6)
    bw = random.randint(4, 6)
    r0 = min(r0, h - bh - 2)
    c0 = min(c0, w - bw - 2)
    return r0, c0, r0 + bh - 1, c0 + bw - 1


def _place_brackets(
    grid: np.ndarray,
    directions: list[str],
    marker_colors: list[int],
) -> list[dict] | None:
    h, w = grid.shape
    boxes: list[tuple[int, int, int, int]] = []
    out: list[dict] = []
    for direction, marker_color in zip(directions, marker_colors):
        candidates = []
        for _ in range(80):
            box = _rand_box(h, w, direction)
            if any(_boxes_overlap(box, other, gap=2) for other in boxes):
                continue
            candidates.append(box)
        if not candidates:
            return None
        box = random.choice(candidates)
        cells = _paint_u(grid, box, direction, BRACKET_COLOR)
        inner = _strict_inner_cells(box)
        if not inner:
            return None
        marker = random.choice(inner)
        grid[marker] = marker_color
        boxes.append(box)
        out.append(
            {
                "direction": direction,
                "color": marker_color,
                "bbox": box,
                "cells": cells,
                "marker": marker,
            }
        )
    return out


def _interval_clear(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    cells: list[tuple[int, int]],
) -> bool:
    return all(int(grid[r, c]) == BG_COLOR and (r, c) not in reserved for r, c in cells)


def _choose_initial_positions(candidates: list[int], count: int) -> list[int] | None:
    if len(candidates) < count:
        return None
    picks = sorted(random.sample(candidates, count))
    return picks


def _lane_candidate_cells(
    shape: tuple[int, int],
    direction: str,
    axis_value: int,
    stop: int,
    count: int,
    starts: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = shape
    finals: list[tuple[int, int]] = []
    initials: list[tuple[int, int]] = []
    swept: set[tuple[int, int]] = set()

    if direction == "up":
        finals = [(stop + 1 + i, axis_value) for i in range(count)]
        initials = list(zip(starts, [axis_value] * count))
    elif direction == "down":
        finals = [(stop - count + i, axis_value) for i in range(count)]
        initials = list(zip(starts, [axis_value] * count))
    elif direction == "left":
        finals = [(axis_value, stop + 1 + i) for i in range(count)]
        initials = list(zip([axis_value] * count, starts))
    else:
        finals = [(axis_value, stop - count + i) for i in range(count)]
        initials = list(zip([axis_value] * count, starts))

    for start, end in zip(initials, finals):
        swept |= _path_cells(start, end)
    return finals, initials, sorted(swept)


def _place_lane(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    color: int,
    noise_color: int,
    direction: str,
    *,
    force_noise_stop: bool,
) -> dict | None:
    h, w = grid.shape
    obstacle_modes = ["noise", "edge"] if not force_noise_stop else ["noise"]
    random.shuffle(obstacle_modes)

    for mode in obstacle_modes:
        counts = [3, 2, 1]
        random.shuffle(counts)
        for count in counts:
            axes = list(range(w)) if direction in ("up", "down") else list(range(h))
            random.shuffle(axes)
            for axis in axes:
                if direction == "up":
                    stops = list(range(1, h - count - 2)) if mode == "noise" else [-1]
                elif direction == "down":
                    stops = list(range(count + 1, h - 1)) if mode == "noise" else [h]
                elif direction == "left":
                    stops = list(range(1, w - count - 2)) if mode == "noise" else [-1]
                else:
                    stops = list(range(count + 1, w - 1)) if mode == "noise" else [w]
                random.shuffle(stops)

                for stop in stops:
                    if direction == "up":
                        tail = list(range(stop + 1 + count, h))
                    elif direction == "down":
                        tail = list(range(0, stop - count))
                    elif direction == "left":
                        tail = list(range(stop + 1 + count, w))
                    else:
                        tail = list(range(0, stop - count))
                    starts = _choose_initial_positions(tail, count)
                    if starts is None:
                        continue
                    if direction in ("down", "right"):
                        starts = sorted(starts)
                    finals, initials, swept = _lane_candidate_cells(grid.shape, direction, axis, stop, count, starts)
                    if any((cell in reserved) for cell in swept):
                        continue
                    if not _interval_clear(grid, reserved, initials + finals):
                        continue

                    if direction in ("up", "down"):
                        lo = min(r for r, _c in swept)
                        hi = max(r for r, _c in swept)
                        corridor = [(r, axis) for r in range(lo, hi + 1)]
                    else:
                        lo = min(c for _r, c in swept)
                        hi = max(c for _r, c in swept)
                        corridor = [(axis, c) for c in range(lo, hi + 1)]
                    if not _interval_clear(grid, reserved, corridor):
                        continue

                    stopper = None
                    if mode == "noise":
                        stopper = (stop, axis) if direction in ("up", "down") else (axis, stop)
                        if int(grid[stopper]) != BG_COLOR or stopper in reserved:
                            continue
                        grid[stopper] = noise_color

                    for r, c in initials:
                        grid[r, c] = color
                    reserved.update(swept)
                    return {
                        "direction": direction,
                        "color": color,
                        "count": count,
                        "starts": initials,
                        "finals": finals,
                        "swept": set(swept),
                        "stopper": stopper,
                        "stop_mode": mode,
                    }
    return None


def _place_movables(
    grid: np.ndarray,
    instructions: list[dict],
    noise_colors: list[int],
) -> tuple[list[dict], bool] | None:
    reserved: set[tuple[int, int]] = set()
    used_noise_stop = False
    lanes: list[dict] = []

    for idx, inst in enumerate(instructions):
        color = inst["color"]
        lane_count = random.randint(1, 2)
        placed_for_color = 0
        for lane_idx in range(lane_count):
            force_noise = idx == 0 and lane_idx == 0
            noise_color = random.choice(noise_colors)
            lane = _place_lane(
                grid,
                reserved,
                color,
                noise_color,
                inst["direction"],
                force_noise_stop=force_noise,
            )
            if lane is None:
                return None
            lanes.append(lane)
            placed_for_color += lane["count"]
            used_noise_stop = used_noise_stop or lane["stop_mode"] == "noise"
        if placed_for_color < 2:
            noise_color = random.choice(noise_colors)
            lane = _place_lane(
                grid,
                reserved,
                color,
                noise_color,
                inst["direction"],
                force_noise_stop=False,
            )
            if lane is None:
                return None
            lanes.append(lane)
            placed_for_color += lane["count"]
            used_noise_stop = used_noise_stop or lane["stop_mode"] == "noise"

    return lanes, used_noise_stop


def _scatter_noise(
    grid: np.ndarray,
    selected_colors: set[int],
    reserved: set[tuple[int, int]],
    noise_colors: list[int],
) -> None:
    h, w = grid.shape
    count = random.randint(3, 7)
    cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) == BG_COLOR and (r, c) not in reserved
    ]
    random.shuffle(cells)
    for r, c in cells[:count]:
        grid[r, c] = random.choice(noise_colors)
    assert any(int(v) not in selected_colors and int(v) not in (BG_COLOR, BRACKET_COLOR) for v in grid.ravel())


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        bg, instructions, bracket_cells = _extract_instructions(grid)
        out, move_info = _apply_instructions(grid)
    except Exception:
        return False

    if not np.any(out != grid):
        return False

    selected = {inst["marker_color"] for inst in instructions}
    nonselected_colors = {
        int(v) for v in np.unique(grid) if int(v) not in selected and int(v) != bg
    }
    for color in nonselected_colors:
        if not np.array_equal(grid == color, out == color):
            return False

    all_paths: list[set[tuple[int, int]]] = []
    moved_any = False
    same_color_stack = False
    obstacle_stop = False
    for info in move_info:
        if not info["moves"]:
            return False
        path_union: set[tuple[int, int]] = set()
        for start, end in info["moves"]:
            moved_any = moved_any or start != end
            path_union |= _path_cells(start, end)
            dr, dc = DIR_VECTORS[info["direction"]]
            front = (end[0] + dr, end[1] + dc)
            if not (0 <= front[0] < grid.shape[0] and 0 <= front[1] < grid.shape[1]):
                continue
            if int(out[front]) != bg:
                obstacle_stop = True
        all_paths.append(path_union)

        lines = Counter(start[1] if info["direction"] in ("up", "down") else start[0] for start, _end in info["moves"])
        same_color_stack = same_color_stack or any(v >= 2 for v in lines.values())

    for i in range(len(all_paths)):
        for j in range(i + 1, len(all_paths)):
            if all_paths[i] & all_paths[j]:
                return False

    if not moved_any or not same_color_stack or not obstacle_stop:
        return False

    for cell in bracket_cells:
        if int(out[cell]) != BRACKET_COLOR:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 29)
    w = random.randint(22, 29)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    instruction_count = random.choice([2, 3, 4])
    directions = random.sample(list(ALL_DIRECTIONS), instruction_count)
    palette = random.sample(NON_BRACKET_COLORS, instruction_count + 2)
    marker_colors = palette[:instruction_count]
    noise_colors = palette[instruction_count:]

    instructions = _place_brackets(grid, directions, marker_colors)
    if instructions is None:
        return None

    placed = _place_movables(grid, instructions, noise_colors)
    if placed is None:
        return None
    lanes, used_noise_stop = placed
    if not used_noise_stop:
        return None

    reserved = set()
    for lane in lanes:
        reserved |= lane["swept"]
    _scatter_noise(grid, set(marker_colors), reserved, noise_colors)

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for bracket command console")
