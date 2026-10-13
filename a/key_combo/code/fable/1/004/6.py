from __future__ import annotations

import importlib.util
import math
import random
from collections import Counter, deque
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_shared_arc_utils_fable_1_004_6", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


if _ARC_UTILS is not None:
    bbox_coords = _ARC_UTILS.bbox_coords
    components_mask = _ARC_UTILS.components_mask
    get_bg = _ARC_UTILS.get_bg
    in_bounds = _ARC_UTILS.in_bounds
else:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flat)).most_common(1)[0][0])

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = map(int, shape)
        return 0 <= int(r) < h and 0 <= int(c) < w

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if int(connectivity) == 8:
            offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        comps: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                q: deque[tuple[int, int]] = deque([(r, c)])
                seen[r, c] = True
                comp: list[tuple[int, int]] = []
                while q:
                    rr, cc = q.popleft()
                    comp.append((int(rr), int(cc)))
                    for dr, dc in offsets:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((int(nr), int(nc)))
                comps.append(np.array(comp, dtype=int))
        return comps


BG = 0
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {direction: idx for idx, direction in enumerate(DIRS)}
SIDE_TO_DIR = {"top": (-1, 0), "right": (0, 1), "bottom": (1, 0), "left": (0, -1)}
DIR_TO_SIDE = {value: key for key, value in SIDE_TO_DIR.items()}
PALETTE = list(range(1, 10))
MIN_SEGMENTS = 4
MAX_SEGMENTS = 6
MIN_STEP = 2
MAX_STEP = 8
MAX_GENERATION_ATTEMPTS = 500


def _triangle_sprite(size: int, direction: tuple[int, int], color: int) -> np.ndarray:
    h, w = size, 2 * size - 1
    sprite = np.zeros((h, w), dtype=int)
    center = size - 1
    for r in range(size):
        sprite[r, center - r : center + r + 1] = int(color)
    if direction == (-1, 0):
        return sprite
    if direction == (0, 1):
        return np.rot90(sprite, k=3)
    if direction == (1, 0):
        return np.rot90(sprite, k=2)
    if direction == (0, -1):
        return np.rot90(sprite, k=1)
    raise ValueError("unsupported direction")


def _triangle_info(main_grid: np.ndarray, bg: int) -> dict:
    comps = components_mask(main_grid != int(bg), connectivity=4)
    if len(comps) != 1:
        raise AssertionError("main area must contain exactly one foreground component")
    coords = comps[0]
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    row_sums = mask.sum(axis=1)
    col_sums = mask.sum(axis=0)

    if np.all(np.diff(row_sums) > 0):
        direction = (-1, 0)
        tip = (int(r0), int(c0 + np.flatnonzero(mask[0, :])[0]))
    elif np.all(np.diff(row_sums) < 0):
        direction = (1, 0)
        tip = (int(r1), int(c0 + np.flatnonzero(mask[-1, :])[0]))
    elif np.all(np.diff(col_sums) > 0):
        direction = (0, -1)
        tip = (int(r0 + np.flatnonzero(mask[:, 0])[0]), int(c0))
    elif np.all(np.diff(col_sums) < 0):
        direction = (0, 1)
        tip = (int(r0 + np.flatnonzero(mask[:, -1])[0]), int(c1))
    else:
        raise AssertionError("triangle orientation must be uniquely identifiable")
    start = (int(tip[0] + direction[0]), int(tip[1] + direction[1]))
    if not in_bounds(main_grid.shape, *start):
        raise AssertionError("triangle start cell must stay inside the main area")
    if int(main_grid[start]) != int(bg):
        raise AssertionError("start cell must be empty")
    return {
        "coords": coords,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "tip": tip,
        "direction": direction,
        "start": start,
    }


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    candidates = []
    for divider_col in range(w):
        divider_color = int(g[0, divider_col])
        if divider_color == bg or not np.all(g[:, divider_col] == divider_color):
            continue
        left_w = divider_col
        right_w = w - divider_col - 1
        if left_w < 1 or right_w < 1 or left_w == right_w:
            continue
        sidebar_side = "left" if left_w < right_w else "right"
        sidebar = g[:, :divider_col] if sidebar_side == "left" else g[:, divider_col + 1 :]
        main = g[:, divider_col + 1 :] if sidebar_side == "left" else g[:, :divider_col]
        if main.shape[1] <= sidebar.shape[1]:
            continue
        sep_rows = [r for r in range(h) if np.all(sidebar[r, :] == divider_color)]
        cells: list[tuple[int, int]] = []
        start = 0
        for r in sep_rows + [h]:
            if r > start:
                cells.append((int(start), int(r)))
            start = r + 1
        if not cells:
            continue
        colors: list[int] = []
        counts: list[int] = []
        ok = True
        for r0, r1 in cells:
            cell = sidebar[r0:r1, :]
            present = sorted(set(map(int, np.unique(cell))) - {bg, divider_color})
            if len(present) != 1:
                ok = False
                break
            color = present[0]
            if color in colors:
                ok = False
                break
            count = int(np.count_nonzero(cell == color))
            if count < 1:
                ok = False
                break
            colors.append(color)
            counts.append(count)
        if not ok:
            continue
        try:
            triangle = _triangle_info(main, bg)
        except AssertionError:
            continue
        candidates.append(
            {
                "bg": bg,
                "grid": g,
                "divider_col": int(divider_col),
                "divider_color": divider_color,
                "sidebar_side": sidebar_side,
                "sidebar": sidebar,
                "main": main,
                "cells": cells,
                "counts": counts,
                "colors": colors,
                "triangle": triangle,
            }
        )
    if len(candidates) != 1:
        raise AssertionError("divider/sidebar layout must be uniquely identifiable")
    return candidates[0]


def _simulate_on_main(
    main_grid: np.ndarray,
    *,
    counts: list[int],
    colors: list[int],
    start: tuple[int, int],
    initial_direction: tuple[int, int],
    turn_step: int = 1,
    strict: bool = True,
) -> np.ndarray | None:
    out = main_grid.copy()
    bg = int(get_bg(main_grid))
    direction_index = DIR_TO_INDEX[tuple(initial_direction)]
    pos = (int(start[0]), int(start[1]))
    for idx, (count, color) in enumerate(zip(counts, colors)):
        dr, dc = DIRS[(direction_index + turn_step * idx) % 4]
        cells = []
        r, c = pos
        for step in range(int(count)):
            rr = int(r + dr * step)
            cc = int(c + dc * step)
            if not in_bounds(main_grid.shape, rr, cc):
                if strict:
                    raise AssertionError("program left the main area")
                return None
            if strict and int(main_grid[rr, cc]) != bg:
                raise AssertionError("program must only paint background cells")
            if not strict and int(main_grid[rr, cc]) != bg:
                return None
            cells.append((rr, cc))
        for rr, cc in cells:
            out[rr, cc] = int(color)
        if idx + 1 < len(counts):
            ndr, ndc = DIRS[(direction_index + turn_step * (idx + 1)) % 4]
            pos = (int(cells[-1][0] + ndr), int(cells[-1][1] + ndc))
    return out


def _render_variant(
    parsed: dict,
    *,
    counts: list[int] | None = None,
    colors: list[int] | None = None,
    initial_direction: tuple[int, int] | None = None,
    turn_step: int = 1,
    strict: bool = True,
) -> np.ndarray | None:
    counts = list(parsed["counts"] if counts is None else counts)
    colors = list(parsed["colors"] if colors is None else colors)
    direction = parsed["triangle"]["direction"] if initial_direction is None else initial_direction
    painted_main = _simulate_on_main(
        parsed["main"],
        counts=counts,
        colors=colors,
        start=parsed["triangle"]["start"],
        initial_direction=direction,
        turn_step=turn_step,
        strict=strict,
    )
    if painted_main is None:
        return None
    out = parsed["grid"].copy()
    divider = parsed["divider_col"]
    if parsed["sidebar_side"] == "left":
        out[:, divider + 1 :] = painted_main
    else:
        out[:, :divider] = painted_main
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the sidebar cell counts as a colored step program and draw its clockwise polyline from the triangle tip.

    Input structure:
    - A full-height divider column separates a wide main area from a narrow sidebar.
    - The sidebar is split top-to-bottom by full-width separator rows; each sidebar cell contains one unique foreground color scattered on the background.
    - The main area contains a single filled triangle whose one-pixel tip indicates the initial facing direction.

    Rule:
    - Parse the sidebar cells from top to bottom.
    - In each cell, the foreground pixel count is the step length and the cell color is the segment color.
    - Start one cell outside the triangle tip, paint the first segment in the tip direction, then rotate 90 degrees clockwise after every segment; keep the divider, sidebar, and triangle unchanged.

    Key insight:
    - The sidebar is not a legend: its row order is execution order, its pixel counts are distances, and the triangle gives the only starting direction for a fixed clockwise-turn drawing program.
    """
    parsed = _parse_input(np.asarray(grid, dtype=int))
    out = _render_variant(parsed, strict=True)
    assert out is not None
    return out


def _neighbors8(coords: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    result: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    result.add((nr, nc))
    return result


def _free_run(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    forbidden: set[tuple[int, int]],
    visited: set[tuple[int, int]],
    *,
    allow_current: bool,
) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = current
    run: list[tuple[int, int]] = []
    while in_bounds(shape, r, c):
        cell = (int(r), int(c))
        if cell in visited:
            break
        if cell in forbidden and not (allow_current and not run):
            break
        run.append(cell)
        r += dr
        c += dc
    return run


def _build_lengths(
    shape: tuple[int, int],
    start: tuple[int, int],
    initial_direction: tuple[int, int],
    forbidden: set[tuple[int, int]],
    n_segments: int,
) -> tuple[list[int], list[tuple[int, int]]] | None:
    initial_index = DIR_TO_INDEX[tuple(initial_direction)]

    def search(depth: int, current: tuple[int, int], visited: set[tuple[int, int]], path: list[tuple[int, int]], lengths: list[int]):
        direction = DIRS[(initial_index + depth) % 4]
        run = _free_run(shape, current, direction, forbidden, visited, allow_current=(depth == 0))
        max_len = min(MAX_STEP, len(run))
        if max_len < MIN_STEP:
            return None
        candidates = list(range(MIN_STEP, max_len + 1))
        random.shuffle(candidates)
        for length in candidates:
            segment = run[:length]
            new_visited = visited | set(segment)
            new_path = path + segment
            new_lengths = lengths + [int(length)]
            if depth + 1 == n_segments:
                if len(set(new_lengths)) < 2:
                    continue
                if max(new_lengths) - min(new_lengths) < 2:
                    continue
                return new_lengths, new_path
            next_direction = DIRS[(initial_index + depth + 1) % 4]
            next_pos = (int(segment[-1][0] + next_direction[0]), int(segment[-1][1] + next_direction[1]))
            if not in_bounds(shape, *next_pos) or next_pos in forbidden or next_pos in new_visited:
                continue
            preview = _free_run(shape, next_pos, next_direction, forbidden, new_visited, allow_current=False)
            if len(preview) < MIN_STEP:
                continue
            found = search(depth + 1, next_pos, new_visited, new_path, new_lengths)
            if found is not None:
                return found
        return None

    return search(0, start, set(), [], [])


def _place_sidebar_program(
    grid: np.ndarray,
    *,
    sidebar_side: str,
    sidebar_width: int,
    divider_color: int,
    counts: list[int],
    colors: list[int],
) -> bool:
    h, w = grid.shape
    min_heights = [max(3, math.ceil((count + 2) / sidebar_width)) for count in counts]
    needed = sum(min_heights) + len(counts) - 1
    if needed > h:
        return False
    heights = min_heights[:]
    leftover = h - needed
    while leftover > 0:
        idx = random.randrange(len(heights))
        heights[idx] += 1
        leftover -= 1

    if sidebar_side == "left":
        sidebar_slice = (slice(None), slice(0, sidebar_width))
        divider_col = sidebar_width
    else:
        sidebar_slice = (slice(None), slice(w - sidebar_width, w))
        divider_col = w - sidebar_width - 1

    grid[:, divider_col] = int(divider_color)
    sidebar = grid[sidebar_slice]
    row = 0
    for idx, (height, count, color) in enumerate(zip(heights, counts, colors)):
        cell = sidebar[row : row + height, :]
        cells = [(r, c) for r in range(height) for c in range(sidebar_width)]
        if count >= len(cells):
            return False
        picks = random.sample(cells, count)
        for rr, cc in picks:
            cell[rr, cc] = int(color)
        row += height
        if idx + 1 < len(heights):
            sidebar[row, :] = int(divider_color)
            row += 1
    return True


def _validate_instance(grid: np.ndarray) -> None:
    parsed = _parse_input(grid)
    assert MIN_SEGMENTS <= len(parsed["counts"]) <= MAX_SEGMENTS
    assert len(set(parsed["colors"])) == len(parsed["colors"])
    assert min(parsed["counts"]) >= MIN_STEP

    out = main(grid)
    assert out.shape == grid.shape
    diff = out != grid
    assert int(diff.sum()) == sum(parsed["counts"])
    assert np.array_equal(out[:, parsed["divider_col"]], grid[:, parsed["divider_col"]])

    clockwise = _render_variant(parsed, turn_step=1, strict=True)
    assert clockwise is not None and np.array_equal(clockwise, out)

    counterclockwise = _render_variant(parsed, turn_step=-1, strict=False)
    assert counterclockwise is None or not np.array_equal(counterclockwise, out)

    reversed_order = _render_variant(
        parsed,
        counts=list(reversed(parsed["counts"])),
        colors=list(reversed(parsed["colors"])),
        strict=False,
    )
    assert reversed_order is None or not np.array_equal(reversed_order, out)

    opposite_direction = _render_variant(
        parsed,
        initial_direction=DIRS[(DIR_TO_INDEX[parsed["triangle"]["direction"]] + 2) % 4],
        strict=False,
    )
    assert opposite_direction is None or not np.array_equal(opposite_direction, out)

    for idx, count in enumerate(parsed["counts"]):
        tweaked = list(parsed["counts"])
        tweaked[idx] = count - 1
        variant = _render_variant(parsed, counts=tweaked, strict=False)
        assert variant is None or not np.array_equal(variant, out)


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(18, 28)
    sidebar_width = random.randint(4, 6)
    max_main_width = 29 - sidebar_width
    if max_main_width < 12:
        return None
    main_width = random.randint(12, max_main_width)
    total_width = main_width + sidebar_width + 1
    if total_width > 30:
        return None

    sidebar_side = random.choice(["left", "right"])
    initial_direction = random.choice(DIRS)
    n_segments = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)

    triangle_size = random.randint(3, 4)
    all_colors = random.sample(PALETTE, n_segments + 2)
    divider_color = all_colors[0]
    triangle_color = all_colors[1]
    program_colors = all_colors[2:]

    triangle = _triangle_sprite(triangle_size, initial_direction, triangle_color)
    th, tw = triangle.shape
    local_main = np.zeros((height, main_width), dtype=int)

    placements = []
    for r0 in range(height - th + 1):
        for c0 in range(main_width - tw + 1):
            placements.append((r0, c0))
    random.shuffle(placements)

    for r0, c0 in placements:
        local_main.fill(BG)
        local_main[r0 : r0 + th, c0 : c0 + tw] = np.maximum(local_main[r0 : r0 + th, c0 : c0 + tw], triangle)
        try:
            tri = _triangle_info(local_main, BG)
        except AssertionError:
            continue
        halo = _neighbors8(tri["coords"], local_main.shape)
        forbidden = set(halo)
        forbidden.discard(tri["start"])
        built = _build_lengths(local_main.shape, tri["start"], initial_direction, forbidden, n_segments)
        if built is None:
            continue
        counts, path_cells = built
        if len(set(path_cells)) != len(path_cells):
            continue
        grid = np.zeros((height, total_width), dtype=int)
        if sidebar_side == "left":
            grid[:, sidebar_width + 1 :] = local_main
        else:
            grid[:, :main_width] = local_main
        if not _place_sidebar_program(
            grid,
            sidebar_side=sidebar_side,
            sidebar_width=sidebar_width,
            divider_color=divider_color,
            counts=counts,
            colors=program_colors,
        ):
            continue
        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sidebar-program puzzle")
