from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


BG = Color.BLACK
ARROW = Color.TEAL
OBSTACLE = Color.GREY

DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
CW_DIR = {
    "U": "R",
    "R": "D",
    "D": "L",
    "L": "U",
}
ALL_DIRS = tuple(DIRS)

GEN_RETRIES = 500
GRID_H_RANGE = (18, 26)
GRID_W_MAX = 26
ARROW_COUNT_CHOICES = (3, 4)
SQUARE_SIDE_CHOICES = (3, 4, 5)
SQUARE_SIDE_WEIGHTS = (2, 4, 4)
PLACEMENT_TRIALS = 220
LEVEL_SAMPLE_TRIALS = 24


def _on_boundary(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _square_side(total: int) -> int:
    side = math.isqrt(int(total))
    if side * side != int(total):
        raise ValueError(f"path size is not a perfect square: {total}")
    return side


def _arrow_patch(direction: str, variant: int) -> np.ndarray:
    patch = None
    if direction in ("U", "D"):
        patch = np.full((2, 3), ARROW, dtype=int)
        miss_col = 0 if int(variant) == 0 else 2
        miss_row = 0 if direction == "U" else 1
        patch[miss_row, miss_col] = BG
    else:
        patch = np.full((3, 2), ARROW, dtype=int)
        miss_row = 0 if int(variant) == 0 else 2
        miss_col = 0 if direction == "L" else 1
        patch[miss_row, miss_col] = BG
    return patch


def _patch_start(top: int, left: int, direction: str) -> tuple[int, int]:
    if direction == "U":
        return int(top), int(left + 1)
    if direction == "D":
        return int(top + 1), int(left + 1)
    if direction == "L":
        return int(top + 1), int(left)
    return int(top + 1), int(left + 1)


def _expand8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _direct_steps(shape: tuple[int, int], start: tuple[int, int], direction: str) -> int:
    h, w = shape
    r, c = start
    if direction == "U":
        return int(r)
    if direction == "D":
        return int(h - 1 - r)
    if direction == "L":
        return int(c)
    return int(w - 1 - c)


def _side_clearance(shape: tuple[int, int], start: tuple[int, int], direction: str) -> int:
    h, w = shape
    r, c = start
    if direction == "U":
        return int(w - 2 - c)
    if direction == "D":
        return int(c - 1)
    if direction == "L":
        return int(r - 1)
    return int(h - 2 - r)


def _available_levels(shape: tuple[int, int], start: tuple[int, int], direction: str) -> list[int]:
    h, w = shape
    r, c = start
    if direction == "U":
        return list(range(r - 2, 0, -1))
    if direction == "D":
        return list(range(r + 2, h - 1))
    if direction == "L":
        return list(range(c - 2, 0, -1))
    return list(range(c + 2, w - 1))


def _sort_levels(direction: str, levels: list[int]) -> list[int]:
    reverse = direction in ("U", "L")
    return sorted((int(v) for v in levels), reverse=reverse)


def _obstacles_from_levels(
    start: tuple[int, int],
    direction: str,
    levels: list[int],
) -> set[tuple[int, int]]:
    r, c = start
    cells: set[tuple[int, int]] = set()
    ordered = _sort_levels(direction, levels)
    for i, level in enumerate(ordered):
        if direction == "U":
            cells.add((int(level), int(c + i)))
        elif direction == "D":
            cells.add((int(level), int(c - i)))
        elif direction == "L":
            cells.add((int(r - i), int(level)))
        else:
            cells.add((int(r + i), int(level)))
    return cells


def _simulate_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: str,
    obstacles: set[tuple[int, int]],
    blocked: set[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | tuple[None, None]:
    h, w = shape
    blocked = blocked or set()
    dr, dc = DIRS[direction]
    sdr, sdc = DIRS[CW_DIR[direction]]

    pos = (int(start[0]), int(start[1]))
    path: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    hits: set[tuple[int, int]] = set()

    while not _on_boundary(pos, shape):
        fr, fc = pos[0] + dr, pos[1] + dc
        if not (0 <= fr < h and 0 <= fc < w):
            return None, None

        if (fr, fc) in obstacles:
            hits.add((fr, fc))
            nr, nc = pos[0] + sdr, pos[1] + sdc
        else:
            nr, nc = fr, fc

        nxt = (int(nr), int(nc))
        if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
            return None, None
        if nxt in obstacles or nxt in blocked or nxt in seen:
            return None, None

        path.append(nxt)
        seen.add(nxt)
        pos = nxt

        if len(path) > h * w:
            return None, None

    return path, hits


def _infer_arrow(component: dict, grid: np.ndarray) -> dict:
    bbox = tuple(map(int, component["bbox"]))
    r0, c0, r1, c1 = bbox
    sprite = crop_bbox(grid, bbox)
    mask = sprite == ARROW
    h, w = mask.shape

    if int(mask.sum()) != h * w - 1:
        raise ValueError("arrow sprite must be a full rectangle with one missing corner")

    missing = np.argwhere(~mask)
    if missing.shape != (1, 2):
        raise ValueError("arrow sprite must have exactly one missing cell")
    mr, mc = map(int, missing[0])

    if h > w:
        direction = "L" if mc == 0 else "R"
        start = (r0 + h // 2, c0 if direction == "L" else c1)
    elif w > h:
        direction = "U" if mr == 0 else "D"
        start = (r0 if direction == "U" else r1, c0 + w // 2)
    else:
        raise ValueError("ambiguous square arrow sprite")

    return {
        "bbox": bbox,
        "coords": component["coords"],
        "area": int(component["area"]),
        "direction": direction,
        "start": (int(start[0]), int(start[1])),
    }


def _extract_groups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    obstacle_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE)}
    arrow_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == ARROW)}
    comps = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[ARROW],
        return_format="dict",
    )
    groups: list[dict] = []
    for comp in sorted(comps, key=lambda item: tuple(map(int, item["bbox"]))):
        info = _infer_arrow(comp, g)
        path, _hits = _simulate_path(
            g.shape,
            info["start"],
            info["direction"],
            obstacle_cells,
            blocked=arrow_cells,
        )
        if path is None:
            raise ValueError("invalid detour path")
        total = int(info["area"] + len(path))
        side = _square_side(total)
        groups.append(
            {
                **info,
                "path": path,
                "total": total,
                "side": side,
            }
        )
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each missing-corner arrow toward the border, count its pixels, then repack the counts as sorted squares.

    Input structure:
    - A black background contains several teal arrow sprites and some grey obstacle pixels.
    - Each arrow sprite is a small filled rectangle with exactly one missing corner, and the rectangle is taller for left/right arrows or wider for up/down arrows.
    - Grey obstacles are isolated from the arrow sprites and only affect the later growth paths.

    Rule:
    - Infer each arrow's orthogonal direction from which side of its bounding box contains the missing corner.
    - Start from the center pixel on that side, grow one teal path cell at a time in the facing direction, and if the next forward cell is grey, move one cell in the clockwise direction instead before resuming forward growth.
    - Count the total teal pixels for each arrow-plus-path group, clear the grid to black, then redraw one solid teal square per group with side length equal to the square root of that pixel count.
    - Sort the squares from small to large and place them left-to-right along the bottom edge with one black column between neighbors.

    Key insight:
    - The visible arrow only tells you where growth begins; the final square size depends on actually simulating the obstacle-driven detours because every sidestep changes the group's total pixel count.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    if get_bg(g) != BG:
        raise ValueError("expected black background")

    groups = _extract_groups(g)
    groups.sort(key=lambda item: (int(item["side"]), int(item["bbox"][1]), int(item["bbox"][0])))

    out = np.full_like(g, BG)
    h, w = out.shape
    col = 0
    for group in groups:
        side = int(group["side"])
        if col + side > w or side > h:
            raise ValueError("repacked squares do not fit in the canvas")
        out[h - side : h, col : col + side] = ARROW
        col += side + 1
    return out


def _sample_square_sides(count: int) -> list[int]:
    for _ in range(200):
        sides = random.choices(SQUARE_SIDE_CHOICES, weights=SQUARE_SIDE_WEIGHTS, k=count)
        if len(set(sides)) == 1 and random.random() < 0.85:
            continue
        if sum(sides) + count - 1 <= GRID_W_MAX:
            return [int(v) for v in sides]
    raise RuntimeError("failed to sample square sides")


def _sample_directions(count: int) -> list[str]:
    distinct = random.randint(2, min(4, count))
    picks = random.sample(ALL_DIRS, distinct)
    while len(picks) < count:
        picks.append(random.choice(ALL_DIRS))
    random.shuffle(picks)
    return picks


def _cells_from_patch(patch: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    coords = np.argwhere(patch == ARROW)
    return {(int(top + r), int(left + c)) for r, c in coords}


def _try_place_arrow(
    shape: tuple[int, int],
    direction: str,
    side: int,
    reserved_steps: set[tuple[int, int]],
    visible_halo: set[tuple[int, int]],
    occupied_visible: set[tuple[int, int]],
    arrow_halo: set[tuple[int, int]],
    global_obstacles: set[tuple[int, int]],
) -> dict | None:
    extra = int(side * side - 5)
    variants = [0, 1]
    random.shuffle(variants)

    for variant in variants:
        patch = _arrow_patch(direction, variant)
        ph, pw = patch.shape
        positions = [
            (r, c)
            for r in range(1, shape[0] - ph)
            for c in range(1, shape[1] - pw)
        ]
        random.shuffle(positions)

        for top, left in positions[:PLACEMENT_TRIALS]:
            arrow_cells = _cells_from_patch(patch, top, left)
            if arrow_cells & reserved_steps:
                continue
            if arrow_cells & visible_halo:
                continue

            start = _patch_start(top, left, direction)
            detours = extra - _direct_steps(shape, start, direction)
            if detours < 0:
                continue

            levels = _available_levels(shape, start, direction)
            if detours > len(levels) or detours > _side_clearance(shape, start, direction):
                continue

            local_arrow_halo = _expand8(arrow_cells)
            for _ in range(LEVEL_SAMPLE_TRIALS):
                chosen_levels = _sort_levels(
                    direction,
                    random.sample(levels, detours) if detours else [],
                )
                obstacle_cells = _obstacles_from_levels(start, direction, chosen_levels)
                if obstacle_cells & reserved_steps:
                    continue
                if obstacle_cells & occupied_visible:
                    continue
                if obstacle_cells & (arrow_halo | local_arrow_halo):
                    continue
                if any(_on_boundary(cell, shape) for cell in obstacle_cells):
                    continue

                path, hits = _simulate_path(
                    shape,
                    start,
                    direction,
                    global_obstacles | obstacle_cells,
                    blocked=reserved_steps | arrow_cells,
                )
                if path is None or hits != obstacle_cells or len(path) != extra:
                    continue

                return {
                    "direction": direction,
                    "side": int(side),
                    "detours": int(detours),
                    "top": int(top),
                    "left": int(left),
                    "patch": patch,
                    "arrow_cells": arrow_cells,
                    "obstacles": obstacle_cells,
                    "path": path,
                }
    return None


def _render_scene(shape: tuple[int, int], items: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for item in items:
        paste(grid, item["patch"], int(item["top"]), int(item["left"]), transparent=BG)
    for item in items:
        for r, c in item["obstacles"]:
            grid[r, c] = OBSTACLE
    return grid


def _valid_output_layout(out: np.ndarray, sides: list[int]) -> bool:
    h, _w = out.shape
    comps = find_components(
        out,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[ARROW],
        return_format="dict",
    )
    if len(comps) != len(sides):
        return False
    found = sorted(math.isqrt(int(comp["area"])) for comp in comps)
    if found != sorted(int(s) for s in sides):
        return False
    for comp in comps:
        area = int(comp["area"])
        side = math.isqrt(area)
        if side * side != area:
            return False
        r0, _c0, r1, _c1 = map(int, comp["bbox"])
        if r1 != h - 1 or r0 != h - side:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        count = random.choice(ARROW_COUNT_CHOICES)
        square_sides = _sample_square_sides(count)
        min_width = max(18, sum(square_sides) + count - 1)
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(min_width, GRID_W_MAX)
        directions = _sample_directions(count)
        plan = list(zip(square_sides, directions))
        random.shuffle(plan)
        plan.sort(key=lambda item: int(item[0]), reverse=True)

        reserved_steps: set[tuple[int, int]] = set()
        visible_halo: set[tuple[int, int]] = set()
        occupied_visible: set[tuple[int, int]] = set()
        arrow_halo: set[tuple[int, int]] = set()
        global_obstacles: set[tuple[int, int]] = set()
        items: list[dict] = []

        ok = True
        for side, direction in plan:
            item = _try_place_arrow(
                (h, w),
                direction,
                int(side),
                reserved_steps,
                visible_halo,
                occupied_visible,
                arrow_halo,
                global_obstacles,
            )
            if item is None:
                ok = False
                break

            items.append(item)
            reserved_steps.update(item["arrow_cells"])
            reserved_steps.update(item["path"])
            occupied_visible.update(item["arrow_cells"])
            occupied_visible.update(item["obstacles"])
            visible_halo.update(_expand8(item["arrow_cells"] | item["obstacles"]))
            arrow_halo.update(_expand8(item["arrow_cells"]))
            global_obstacles.update(item["obstacles"])

        if not ok:
            continue
        if not any(int(item["detours"]) > 0 for item in items):
            continue

        grid = _render_scene((h, w), items)
        try:
            groups = _extract_groups(grid)
            out = main(grid)
        except Exception:
            continue

        found_sides = sorted(int(group["side"]) for group in groups)
        if found_sides != sorted(square_sides):
            continue
        if not _valid_output_layout(out, square_sides):
            continue
        return grid

    raise RuntimeError("failed to generate a valid arrow-detour input")
