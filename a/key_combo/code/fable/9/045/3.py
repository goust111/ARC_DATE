from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np

try:
    from arc_utils import Color, find_components, get_bg, in_bounds
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils_fallback", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    Color = _ARC_UTILS.Color
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg
    in_bounds = _ARC_UTILS.in_bounds


CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
PALETTE = list(Color.NOT_BLACK)
SIZE_POOL = [(h, w) for h in range(2, 6) for w in range(2, 6)]
TARGET_SIZE_POOL = [size for size in SIZE_POOL if 6 <= size[0] * size[1] <= 20]

GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (18, 26)
RECT_COUNT_RANGE = (3, 5)
PATCH_DISTANCE_RANGE = (1, 4)
SAME_COLOR_NOISE_RANGE = (1, 2)
EXTRA_NOISE_RANGE = (3, 8)
GENERATION_ATTEMPTS = 400


def _corner_direction(corner: tuple[int, int]) -> tuple[int, int]:
    return (-1 if corner[0] == 0 else 1, -1 if corner[1] == 0 else 1)


def _missing_cell_from_bbox(
    r0: int, c0: int, h: int, w: int, corner: tuple[int, int]
) -> tuple[int, int]:
    return (
        r0 if corner[0] == 0 else r0 + h - 1,
        c0 if corner[1] == 0 else c0 + w - 1,
    )


def _shape_cells(
    r0: int, c0: int, h: int, w: int, corner: tuple[int, int]
) -> list[tuple[int, int]]:
    missing = _missing_cell_from_bbox(r0, c0, h, w, corner)
    return [
        (r, c)
        for r in range(r0, r0 + h)
        for c in range(c0, c0 + w)
        if (r, c) != missing
    ]


def _make_rect(
    color: int, r0: int, c0: int, h: int, w: int, corner: tuple[int, int]
) -> dict:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    missing = _missing_cell_from_bbox(r0, c0, h, w, corner)
    direction = _corner_direction(corner)
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "h": int(h),
        "w": int(w),
        "corner": corner,
        "missing": missing,
        "direction": direction,
        "cells": _shape_cells(r0, c0, h, w, corner),
    }


def _point_on_ray(point: tuple[int, int], rect: dict) -> bool:
    pr, pc = map(int, point)
    mr, mc = rect["missing"]
    dr, dc = rect["direction"]
    drow = pr - mr
    dcol = pc - mc
    if drow == 0 or dcol == 0:
        return False
    if drow * dr <= 0 or dcol * dc <= 0:
        return False
    return drow == dcol if dr == dc else drow == -dcol


def _ray_distance(point: tuple[int, int], rect: dict) -> int:
    pr, _pc = map(int, point)
    mr, _mc = rect["missing"]
    dr, _dc = rect["direction"]
    return abs(pr - mr) if dr != 0 else 0


def _ray_cells(shape: tuple[int, int], rect: dict) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    mr, mc = rect["missing"]
    dr, dc = rect["direction"]
    cells = []
    step = 1
    while True:
        rr = mr + step * dr
        cc = mc + step * dc
        if not in_bounds((h, w), rr, cc):
            break
        cells.append((int(rr), int(cc)))
        step += 1
    return cells


def _classify_missing_corner_rectangle(comp: dict) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    area = int(comp["area"])
    if h < 2 or w < 2 or area != h * w - 1:
        return None
    present = {tuple(map(int, rc)) for rc in comp["coords"]}
    full = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    missing = list(full - present)
    if len(missing) != 1:
        return None
    missing_cell = missing[0]
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if missing_cell not in corners:
        return None
    corner = (
        0 if missing_cell[0] == r0 else 1,
        0 if missing_cell[1] == c0 else 1,
    )
    rect = _make_rect(int(comp["color"]), r0, c0, h, w, corner)
    rect["coords"] = comp["coords"]
    rect["area"] = area
    return rect


def _scene_parts(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int, int]], list[dict]]:
    bg = int(get_bg(grid))
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    rects: list[dict] = []
    singletons: list[tuple[int, int, int]] = []
    others: list[dict] = []
    for comp in comps:
        if int(comp["area"]) == 1:
            r, c = map(int, comp["coords"][0])
            singletons.append((r, c, int(comp["color"])))
            continue
        rect = _classify_missing_corner_rectangle(comp)
        if rect is None:
            others.append(comp)
        else:
            rects.append(rect)
    return rects, singletons, others


def _winner_and_patch(
    grid: np.ndarray, rects: list[dict], singletons: list[tuple[int, int, int]]
) -> tuple[dict, tuple[int, int]]:
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, color in singletons:
        by_color.setdefault(int(color), []).append((int(r), int(c)))

    winners: list[tuple[dict, list[tuple[int, int]]]] = []
    for rect in rects:
        hits = [pt for pt in by_color.get(int(rect["color"]), []) if _point_on_ray(pt, rect)]
        if hits:
            hits.sort(key=lambda pt: _ray_distance(pt, rect))
            winners.append((rect, hits))

    assert len(winners) == 1, "exactly one missing-corner rectangle must have a same-color ray hit"
    winner, hits = winners[0]
    assert len(hits) == 1, "the selected rectangle must have a unique same-color patch on its ray"
    return winner, hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate the only corner-missing rectangle whose missing-corner ray hits a same-color patch, then complete it.

    Input structure:
    - The grid has a black background, several solid monochrome rectangles each missing exactly one corner cell, and scattered isolated noise pixels.
    - Every rectangle's missing corner defines a diagonal ray pointing outward through that absent corner.
    - Exactly one rectangle has an isolated pixel of the same color somewhere on its own missing-corner ray.

    Rule:
    - Detect every corner-missing rectangle and test the diagonal ray starting at its missing corner.
    - Select the unique rectangle whose ray hits a same-color isolated pixel.
    - Clear everything, translate that rectangle along the ray until its missing corner lands on the patch pixel, and draw the resulting complete filled rectangle.

    Key insight:
    - All rectangles are defective, so the selector is not the defect itself; it is the conjunction of missing-corner direction, diagonal alignment, and same-color ray hit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    rects, singletons, others = _scene_parts(g)
    assert len(rects) >= 2, "expected multiple corner-missing rectangles"
    assert not others, "all non-noise multi-cell components must be corner-missing rectangles"
    assert singletons, "expected isolated noise pixels"

    winner, patch = _winner_and_patch(g, rects, singletons)
    mr, mc = winner["missing"]
    dr, dc = winner["direction"]
    shift_r = int(patch[0] - mr)
    shift_c = int(patch[1] - mc)
    assert shift_r != 0 and abs(shift_r) == abs(shift_c)
    assert shift_r * dr > 0 and shift_c * dc > 0

    out = np.full_like(g, bg)
    r0, c0, r1, c1 = winner["bbox"]
    out[r0 + shift_r : r1 + shift_r + 1, c0 + shift_c : c1 + shift_c + 1] = int(winner["color"])
    return out


def _mark_rect_buffer(blocked: np.ndarray, rect: dict) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = rect["bbox"]
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _mark_radius(blocked: np.ndarray, r: int, c: int, radius: int = 1) -> None:
    h, w = blocked.shape
    rr0 = max(0, r - radius)
    cc0 = max(0, c - radius)
    rr1 = min(h - 1, r + radius)
    cc1 = min(w - 1, c + radius)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _rect_candidates(
    shape: tuple[int, int],
    blocked: np.ndarray,
    size: tuple[int, int],
    corner: tuple[int, int],
    patch_distance: int | None = None,
) -> list[dict]:
    h, w = map(int, shape)
    rh, rw = map(int, size)
    candidates: list[dict] = []
    for r0 in range(h - rh + 1):
        for c0 in range(w - rw + 1):
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h - 1, r1 + 1)
            cc1 = min(w - 1, c1 + 1)
            if bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                continue
            rect = _make_rect(1, r0, c0, rh, rw, corner)
            if patch_distance is not None:
                mr, mc = rect["missing"]
                dr, dc = rect["direction"]
                shift_r = patch_distance * dr
                shift_c = patch_distance * dc
                if not (
                    0 <= r0 + shift_r
                    and r1 + shift_r < h
                    and 0 <= c0 + shift_c
                    and c1 + shift_c < w
                ):
                    continue
                if not in_bounds((h, w), mr + shift_r, mc + shift_c):
                    continue
            candidates.append(rect)
    return candidates


def _paint_rectangles(shape: tuple[int, int], rects: list[dict], bg: int = 0) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for rect in rects:
        for r, c in rect["cells"]:
            grid[r, c] = int(rect["color"])
    return grid


def _is_isolated_empty(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != Color.BLACK:
        return False
    h, w = grid.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if (rr, cc) != (r, c) and int(grid[rr, cc]) != Color.BLACK:
                return False
    return True


def _safe_colors_for_cell(
    cell: tuple[int, int], rects: list[dict], exclude: set[int] | None = None
) -> list[int]:
    banned = set() if exclude is None else set(map(int, exclude))
    allowed = []
    for color in PALETTE:
        if int(color) in banned:
            continue
        if any(int(rect["color"]) == int(color) and _point_on_ray(cell, rect) for rect in rects):
            continue
        allowed.append(int(color))
    return allowed


def _noise_candidates(
    grid: np.ndarray,
    rects: list[dict],
    *,
    on_ray_for: dict | None = None,
    off_ray_for: dict | None = None,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if not _is_isolated_empty(grid, r, c):
                continue
            pt = (int(r), int(c))
            if on_ray_for is not None and not _point_on_ray(pt, on_ray_for):
                continue
            if off_ray_for is not None and _point_on_ray(pt, off_ray_for):
                continue
            if on_ray_for is None and off_ray_for is None and not _safe_colors_for_cell(pt, rects):
                continue
            candidates.append(pt)
    return candidates


def _expected_output(shape: tuple[int, int], target: dict, patch: tuple[int, int], bg: int = 0) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    mr, mc = target["missing"]
    shift_r = int(patch[0] - mr)
    shift_c = int(patch[1] - mc)
    r0, c0, _r1, _c1 = target["bbox"]
    out[
        r0 + shift_r : r0 + shift_r + int(target["h"]),
        c0 + shift_c : c0 + shift_c + int(target["w"]),
    ] = int(target["color"])
    return out


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    target: dict,
    target_patch: tuple[int, int],
    rect_count: int,
) -> bool:
    rects, singletons, others = _scene_parts(grid)
    if len(rects) != rect_count or others:
        return False
    try:
        winner, patch = _winner_and_patch(grid, rects, singletons)
    except AssertionError:
        return False
    if winner["bbox"] != target["bbox"] or int(winner["color"]) != int(target["color"]):
        return False
    if patch != target_patch:
        return False
    areas = [int(rect["h"]) * int(rect["w"]) for rect in rects]
    target_area = int(target["h"]) * int(target["w"])
    if not (min(areas) < target_area < max(areas)):
        return False
    if not any(rect["corner"] == target["corner"] and rect["bbox"] != target["bbox"] for rect in rects):
        return False

    off_ray_counts: dict[tuple[int, int, int, int], int] = {tuple(rect["bbox"]): 0 for rect in rects}
    wrong_color_on_distractor_ray = 0
    for r, c, color in singletons:
        pt = (r, c)
        for rect in rects:
            if int(color) == int(rect["color"]):
                if _point_on_ray(pt, rect):
                    if tuple(rect["bbox"]) != tuple(target["bbox"]) or pt != target_patch:
                        return False
                else:
                    off_ray_counts[tuple(rect["bbox"])] += 1
            elif _point_on_ray(pt, rect) and tuple(rect["bbox"]) != tuple(target["bbox"]):
                wrong_color_on_distractor_ray += 1

    if any(count < 1 for count in off_ray_counts.values()):
        return False
    if wrong_color_on_distractor_ray < 1:
        return False
    try:
        solved = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, expected))


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    rect_count = random.randint(*RECT_COUNT_RANGE)
    colors = random.sample(PALETTE, rect_count)

    target_corner = random.choice(CORNERS)
    target_size = random.choice(TARGET_SIZE_POOL)
    target_area = target_size[0] * target_size[1]
    smaller = [size for size in SIZE_POOL if size[0] * size[1] < target_area]
    larger = [size for size in SIZE_POOL if size[0] * size[1] > target_area]
    if not smaller or not larger:
        return None

    patch_distance = random.randint(*PATCH_DISTANCE_RANGE)
    blocked = np.zeros((h, w), dtype=bool)
    target_candidates = _rect_candidates((h, w), blocked, target_size, target_corner, patch_distance)
    if not target_candidates:
        return None
    target = random.choice(target_candidates)
    target["color"] = colors[0]
    target["cells"] = _shape_cells(target["bbox"][0], target["bbox"][1], target["h"], target["w"], target["corner"])
    target_patch = (
        int(target["missing"][0] + patch_distance * target["direction"][0]),
        int(target["missing"][1] + patch_distance * target["direction"][1]),
    )
    _mark_rect_buffer(blocked, target)
    _mark_radius(blocked, target_patch[0], target_patch[1], radius=1)

    distractor_specs: list[tuple[int, tuple[int, int], tuple[int, int]]] = [
        (colors[1], random.choice(smaller), target_corner)
    ]
    distractor_specs.append((colors[2], random.choice(larger), random.choice(CORNERS)))
    for color in colors[3:]:
        distractor_specs.append((color, random.choice(SIZE_POOL), random.choice(CORNERS)))
    random.shuffle(distractor_specs)
    distractor_specs.sort(key=lambda item: item[1][0] * item[1][1], reverse=True)

    rects = [target]
    for color, size, corner in distractor_specs:
        candidates = _rect_candidates((h, w), blocked, size, corner)
        if not candidates:
            return None
        rect = random.choice(candidates)
        rect["color"] = int(color)
        rect["cells"] = _shape_cells(rect["bbox"][0], rect["bbox"][1], rect["h"], rect["w"], rect["corner"])
        rects.append(rect)
        _mark_rect_buffer(blocked, rect)

    grid = _paint_rectangles((h, w), rects, bg=Color.BLACK)
    grid[target_patch] = int(target["color"])

    distractors = rects[1:].copy()
    random.shuffle(distractors)
    ray_noise_count = 0
    for rect in distractors:
        must_place = ray_noise_count == 0
        if not must_place and random.random() < 0.45:
            continue
        candidates = _noise_candidates(grid, rects, on_ray_for=rect)
        random.shuffle(candidates)
        placed = False
        for cell in candidates:
            colors_here = _safe_colors_for_cell(cell, rects, exclude={int(rect["color"])})
            if not colors_here:
                continue
            grid[cell] = random.choice(colors_here)
            ray_noise_count += 1
            placed = True
            break
        if must_place and not placed:
            return None
    if ray_noise_count == 0:
        return None

    off_ray_needs = {tuple(rect["bbox"]): random.randint(*SAME_COLOR_NOISE_RANGE) for rect in rects}
    if all(off_ray_needs[tuple(rect["bbox"])] == 1 for rect in rects[1:]):
        off_ray_needs[tuple(random.choice(rects[1:])["bbox"])] = 2

    order = rects.copy()
    random.shuffle(order)
    for rect in order:
        for _ in range(off_ray_needs[tuple(rect["bbox"])]):
            candidates = _noise_candidates(grid, rects, off_ray_for=rect)
            random.shuffle(candidates)
            if not candidates:
                return None
            grid[random.choice(candidates)] = int(rect["color"])

    extra_noise = random.randint(*EXTRA_NOISE_RANGE)
    for _ in range(extra_noise):
        candidates = _noise_candidates(grid, rects)
        random.shuffle(candidates)
        placed = False
        for cell in candidates:
            colors_here = _safe_colors_for_cell(cell, rects)
            if not colors_here:
                continue
            grid[cell] = random.choice(colors_here)
            placed = True
            break
        if not placed:
            break

    expected = _expected_output((h, w), target, target_patch, bg=Color.BLACK)
    if not _validate_generated(grid, expected, target, target_patch, rect_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-missing-rectangle scene")
