from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste, random_free_location


GREY = Color.GREY
BG_COLOR = Color.BLACK
GEN_MAX_ATTEMPTS = 400
GRID_H_RANGE = (19, 29)
GRID_W_RANGE = (20, 30)
POLE_COUNTS = (2, 3)
POLE_HEIGHT_RANGE = (3, 9)
POLE_COLUMN_GAP = 2
OBJECT_TO_POLE_GAP = 2
OBJECT_AREA_RANGE = (4, 14)
OBJECT_AREA_MIN_GAP = 2
OBJECT_MAX_SIDE = 6
OBJECT_PADDING = 1
OBJECT_BORDER = 1
MIN_OBJECT_REGION_HEIGHT = 8
_GEN_CALL_COUNT = 0


def _extract_poles(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    pole_mask = np.zeros_like(grid, dtype=bool)
    poles: list[dict] = []

    for c in range(w):
        if int(grid[h - 1, c]) == bg:
            continue

        r = h - 1
        while r >= 0 and int(grid[r, c]) != bg:
            r -= 1
        top = r + 1
        segment = grid[top:, c]
        if segment.size < 2:
            continue

        flag_color = int(segment[0])
        if flag_color in (bg, GREY):
            continue
        if not np.all(segment[1:] == GREY):
            continue

        coords = tuple((rr, int(c)) for rr in range(int(top), h))
        pole_mask[top:, c] = True
        poles.append(
            {
                "col": int(c),
                "top": int(top),
                "height": int(h - top),
                "flag_color": flag_color,
                "coords": coords,
            }
        )

    return poles, pole_mask


def _extract_objects(grid: np.ndarray, bg: int, pole_mask: np.ndarray) -> list[dict]:
    stripped = np.asarray(grid, dtype=int).copy()
    stripped[pole_mask] = int(bg)

    objects: list[dict] = []
    for comp in find_components(
        stripped,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        coords = tuple((int(r), int(c)) for r, c in comp["coords"])
        objects.append(
            {
                "color": int(comp["color"]),
                "area": int(comp["area"]),
                "bbox": tuple(map(int, comp["bbox"])),
                "coords": coords,
            }
        )
    return objects


def _analyze_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = int(get_bg(g))
    poles, pole_mask = _extract_poles(g, bg)
    objects = _extract_objects(g, bg, pole_mask)

    assert poles, "expected at least one pole"
    assert len(objects) == len(poles) + 1, "expected exactly one more object than poles"

    pole_heights = [int(pole["height"]) for pole in poles]
    assert len(set(pole_heights)) == len(poles), "pole heights must be unique"

    flag_colors = [int(pole["flag_color"]) for pole in poles]
    assert len(set(flag_colors)) == len(poles), "flag colors must be unique"

    object_areas = [int(obj["area"]) for obj in objects]
    assert len(set(object_areas)) == len(objects), "object areas must be unique"

    return bg, poles, objects, pole_mask


def _apply_rank_mapping(grid: np.ndarray, bg: int, poles: list[dict], objects: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    ordered_poles = sorted(poles, key=lambda pole: int(pole["height"]), reverse=True)
    ordered_objects = sorted(objects, key=lambda obj: int(obj["area"]), reverse=True)

    for idx, obj in enumerate(ordered_objects):
        new_color = bg if idx >= len(ordered_poles) else int(ordered_poles[idx]["flag_color"])
        for r, c in obj["coords"]:
            out[int(r), int(c)] = int(new_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the larger objects by matching area rank to flagpole height rank, and erase the smallest object.

    Input structure:
    - The bottom edge contains m one-cell-wide flagpoles: each pole is a vertical grey body with a single colored flag cell at its top.
    - Elsewhere on the canvas there are m+1 separated single-color objects with distinct shapes, colors, and areas.
    - Pole left-right order and object positions are arbitrary; only the bottom anchoring distinguishes poles.

    Rule:
    - Rank the m+1 objects by area from largest to smallest.
    - Rank the m flagpoles by total height from tallest to shortest, using the color of each top flag cell.
    - Recolor the i-th largest object to the i-th tallest pole's flag color for i = 1..m, keep every shape and position fixed, and erase the smallest object to background.
    - Preserve every flagpole exactly as drawn.

    Key insight:
    - The useful keys are two hidden orderings on different visual properties: object area and pole height. Spatial proximity and original object colors are distractors.
    """
    bg, poles, objects, _pole_mask = _analyze_scene(np.asarray(grid, dtype=int))
    return _apply_rank_mapping(np.asarray(grid, dtype=int), bg, poles, objects)


def _grow_polyomino(area: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    mask[seed] = True

    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        assert frontier, "polyomino frontier unexpectedly empty"
        cell = random.choice(frontier)
        cells.add(cell)
        mask[cell] = True

    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _make_object_patch(area: int, color: int, bg: int) -> np.ndarray:
    dims = [
        (h, w)
        for h in range(2, OBJECT_MAX_SIDE + 1)
        for w in range(2, OBJECT_MAX_SIDE + 1)
        if area <= h * w <= area + 6
    ]
    random.shuffle(dims)

    for h, w in dims:
        for _ in range(50):
            mask = _grow_polyomino(area, h, w)
            if mask.shape[0] < 2 or mask.shape[1] < 2:
                continue
            patch = np.full(mask.shape, int(bg), dtype=int)
            patch[mask] = int(color)
            return patch
    raise RuntimeError(f"failed to build object patch with area {area}")


def _sample_object_areas(count: int) -> list[int]:
    values = list(range(OBJECT_AREA_RANGE[0], OBJECT_AREA_RANGE[1] + 1))
    for _ in range(200):
        sample = sorted(random.sample(values, count))
        if all(b - a >= OBJECT_AREA_MIN_GAP for a, b in zip(sample, sample[1:])):
            return sample
    raise RuntimeError("failed to sample well-separated object areas")


def _sample_pole_columns(width: int, count: int) -> list[int] | None:
    cols = list(range(1, width - 1))
    for _ in range(200):
        sample = sorted(random.sample(cols, count))
        if all(b - a >= POLE_COLUMN_GAP for a, b in zip(sample, sample[1:])):
            return sample
    return None


def _is_monotonic(seq: list[int]) -> bool:
    if len(seq) < 2:
        return True
    increasing = all(a < b for a, b in zip(seq, seq[1:]))
    decreasing = all(a > b for a, b in zip(seq, seq[1:]))
    return increasing or decreasing


def _sample_poles(width: int, height: int, count: int, flag_colors: list[int]) -> list[dict] | None:
    max_height = min(POLE_HEIGHT_RANGE[1], height - MIN_OBJECT_REGION_HEIGHT - OBJECT_TO_POLE_GAP)
    if max_height < POLE_HEIGHT_RANGE[0] or max_height - POLE_HEIGHT_RANGE[0] + 1 < count:
        return None

    cols = _sample_pole_columns(width, count)
    if cols is None:
        return None

    heights_desc = sorted(random.sample(range(POLE_HEIGHT_RANGE[0], max_height + 1), count), reverse=True)
    if count >= 3:
        perms = [
            perm
            for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
            if not _is_monotonic([heights_desc[i] for i in perm])
        ]
        perm = random.choice(perms)
        heights_by_x = [heights_desc[i] for i in perm]
    else:
        heights_by_x = random.sample(heights_desc, len(heights_desc))

    random.shuffle(flag_colors)
    poles: list[dict] = []
    for col, pole_height, flag_color in zip(cols, heights_by_x, flag_colors):
        top = height - pole_height
        coords = tuple((rr, int(col)) for rr in range(int(top), height))
        poles.append(
            {
                "col": int(col),
                "top": int(top),
                "height": int(pole_height),
                "flag_color": int(flag_color),
                "coords": coords,
            }
        )
    return poles


def _paint_poles(grid: np.ndarray, poles: list[dict]) -> None:
    for pole in poles:
        top = int(pole["top"])
        col = int(pole["col"])
        grid[top, col] = int(pole["flag_color"])
        if top + 1 < grid.shape[0]:
            grid[top + 1 :, col] = GREY


def _spec_coords_from_patch(patch: np.ndarray, r0: int, c0: int, color: int) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(patch == int(color))
    return tuple((int(r0 + r), int(c0 + c)) for r, c in coords)


def _place_objects(
    grid: np.ndarray,
    object_colors: list[int],
    areas: list[int],
    max_pole_height: int,
    bg: int,
) -> list[dict] | None:
    upper_h = grid.shape[0] - max_pole_height - OBJECT_TO_POLE_GAP
    if upper_h < MIN_OBJECT_REGION_HEIGHT:
        return None

    canvas = np.full((upper_h, grid.shape[1]), int(bg), dtype=int)
    placed: list[dict] = []

    order = sorted(zip(areas, object_colors), reverse=True)
    for area, color in order:
        patch = _make_object_patch(int(area), int(color), int(bg))
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=int(bg),
                border=OBJECT_BORDER,
                padding=OBJECT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(canvas, patch, int(r0), int(c0), transparent=int(bg))
        coords = _spec_coords_from_patch(patch, int(r0), int(c0), int(color))
        bbox = (
            min(r for r, _c in coords),
            min(c for _r, c in coords),
            max(r for r, _c in coords),
            max(c for _r, c in coords),
        )
        placed.append(
            {
                "color": int(color),
                "area": int(area),
                "bbox": tuple(map(int, bbox)),
                "coords": coords,
            }
        )

    paste(grid, canvas, 0, 0, transparent=int(bg))
    return placed


def _objects_need_more_scrambling(objects: list[dict]) -> bool:
    areas_by_x = [int(obj["area"]) for obj in sorted(objects, key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))]
    areas_by_y = [int(obj["area"]) for obj in sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))]
    return _is_monotonic(areas_by_x) or _is_monotonic(areas_by_y)


def _metadata_matches(parsed_poles: list[dict], parsed_objects: list[dict], poles: list[dict], objects: list[dict]) -> bool:
    parsed_pole_map = {
        int(pole["flag_color"]): (int(pole["col"]), int(pole["height"]), tuple(pole["coords"]))
        for pole in parsed_poles
    }
    built_pole_map = {
        int(pole["flag_color"]): (int(pole["col"]), int(pole["height"]), tuple(pole["coords"]))
        for pole in poles
    }
    if parsed_pole_map != built_pole_map:
        return False

    parsed_obj_map = {
        int(obj["color"]): (int(obj["area"]), tuple(sorted(obj["coords"])))
        for obj in parsed_objects
    }
    built_obj_map = {
        int(obj["color"]): (int(obj["area"]), tuple(sorted(obj["coords"])))
        for obj in objects
    }
    return parsed_obj_map == built_obj_map


def _build_expected(grid: np.ndarray, poles: list[dict], objects: list[dict], bg: int) -> np.ndarray:
    return _apply_rank_mapping(grid, int(bg), poles, objects)


def _gen_attempt(preferred_poles: int) -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = BG_COLOR

    usable_colors = [c for c in Color.ALL_COLORS if c not in (bg, GREY)]
    needed = 2 * preferred_poles + 1
    palette = random.sample(usable_colors, needed)
    flag_colors = palette[:preferred_poles]
    object_colors = palette[preferred_poles:]

    poles = _sample_poles(w, h, preferred_poles, flag_colors)
    if poles is None:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    _paint_poles(grid, poles)

    areas = _sample_object_areas(preferred_poles + 1)
    objects = _place_objects(grid, object_colors, areas, max(int(p["height"]) for p in poles), bg)
    if objects is None:
        return None

    if preferred_poles >= 3:
        heights_by_x = [int(pole["height"]) for pole in sorted(poles, key=lambda pole: int(pole["col"]))]
        if _is_monotonic(heights_by_x):
            return None
    if _objects_need_more_scrambling(objects):
        return None

    parsed_bg, parsed_poles, parsed_objects, _pole_mask = _analyze_scene(grid)
    if parsed_bg != bg:
        return None
    if not _metadata_matches(parsed_poles, parsed_objects, poles, objects):
        return None

    expected = _build_expected(grid, poles, objects, bg)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    preferred_poles = POLE_COUNTS[_GEN_CALL_COUNT % len(POLE_COUNTS)]
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(preferred_poles)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid flag-rank input")
