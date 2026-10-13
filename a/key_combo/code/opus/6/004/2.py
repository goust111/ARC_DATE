from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import find_components, get_bg


BG = 0
FRAME = 5
CRACK = 6
BODY_COLORS = (1, 2, 3)


def _row_span_interior(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[tuple[int, int], ...]:
    occupied = {(int(r), int(c)) for r, c in coords}
    rows = sorted({r for r, _ in occupied})
    interior: list[tuple[int, int]] = []
    for r in rows:
        cols = sorted(c for rr, c in occupied if rr == r)
        if len(cols) < 2:
            continue
        for c in range(cols[0] + 1, cols[-1]):
            if (r, c) in occupied:
                continue
            assert int(grid[r, c]) == int(bg), "frame cavity must be background in the input"
            interior.append((r, c))
    return tuple(interior)


def _split_fractured_object(
    grid: np.ndarray, component: dict, bg: int
) -> dict[str, object]:
    coords = component["coords"]
    r0, c0, r1, c1 = map(int, component["bbox"])
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()

    colors = {int(patch[r, c]) for r, c in np.argwhere(patch != bg)}
    assert CRACK in colors, "fractured object must contain the crack color"
    body_colors = colors - {CRACK}
    assert len(body_colors) == 1, "each fractured object must have exactly one body color"
    body = int(next(iter(body_colors)))

    crack_coords = np.argwhere(patch == CRACK)
    crack_rows = {int(r) for r, _ in crack_coords}
    crack_cols = {int(c) for _, c in crack_coords}
    if len(crack_cols) == 1:
        orientation = "vertical"
        crack_index = int(next(iter(crack_cols)))
        assert len(crack_rows) == patch.shape[0], "vertical crack must span the full short axis"
    else:
        assert len(crack_rows) == 1, "crack must be a single row or a single column"
        orientation = "horizontal"
        crack_index = int(next(iter(crack_rows)))
        assert len(crack_cols) == patch.shape[1], "horizontal crack must span the full short axis"

    patch[patch == CRACK] = bg
    halves = find_components(
        patch,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[body],
    )
    assert len(halves) == 2, "erasing the crack must split the object into exactly two halves"

    if orientation == "vertical":
        halves.sort(key=lambda comp_: int(comp_["bbox"][1]))
        left, right = halves
        assert int(left["bbox"][3]) < crack_index < int(right["bbox"][1])
        shifted_left = {(int(r), int(c) - 1) for r, c in left["coords"]}
        shifted_right = {(int(r), int(c) + 1) for r, c in right["coords"]}
        assert shifted_left.isdisjoint(shifted_right), "semantic slide must separate the two halves"
    else:
        halves.sort(key=lambda comp_: int(comp_["bbox"][0]))
        top, bottom = halves
        assert int(top["bbox"][2]) < crack_index < int(bottom["bbox"][0])
        shifted_top = {(int(r) - 1, int(c)) for r, c in top["coords"]}
        shifted_bottom = {(int(r) + 1, int(c)) for r, c in bottom["coords"]}
        assert shifted_top.isdisjoint(shifted_bottom), "semantic slide must separate the two halves"

    return {
        "color": body,
        "orientation": orientation,
        "coords": tuple((int(r), int(c)) for r, c in coords),
        "halves": [
            {
                "color": body,
                "area": int(half["area"]),
            }
            for half in halves
        ],
    }


def _analyze_scene(grid: np.ndarray) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    objects: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    for component in components:
        coords = component["coords"]
        colors = {int(g[r, c]) for r, c in coords}
        if CRACK in colors:
            objects.append(_split_fractured_object(g, component, bg))
            continue
        assert colors == {FRAME}, "non-object foreground components must be gray frames"
        cavity = _row_span_interior(g, coords, bg)
        frames.append(
            {
                "area": len(cavity),
                "cells": cavity,
                "border": tuple((int(r), int(c)) for r, c in coords),
            }
        )

    assert 2 <= len(objects) <= 3, "expected two or three fractured objects"
    assert len(frames) == 2 * len(objects), "frame count must equal the number of object halves"
    return bg, objects, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each cracked object into two halves and fill the uniquely area-matched gray frame cavities.

    Input structure:
    - The grid has black background, two or three multicolor foreground objects, and
      several gray hollow rectangular frames.
    - Each foreground object uses one body color plus a one-pixel magenta crack line
      that spans the object's short axis.
    - Each gray frame has a unique row-span cavity area, and the number of frames equals
      the total number of cracked halves.

    Rule:
    - Erase the magenta crack from every object; the remaining body pixels split into
      exactly two 4-connected halves.
    - Interpret those halves as sliding one cell away from the crack, then count the
      pixels in each half.
    - Compute every frame's row-span cavity area, pair each half with the unique frame
      whose cavity area equals the half's pixel count, erase all source objects, and
      fill the matched cavity with the half's body color while keeping the gray border.

    Key insight:
    - The crack is not decoration: only after mentally removing it and splitting the
      object do the hidden half-areas appear, and those areas are the keys that decide
      which frame each color must fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, objects, frames = _analyze_scene(g)
    frame_by_area: dict[int, tuple[tuple[int, int], ...]] = {}
    for frame in frames:
        area = int(frame["area"])
        assert area not in frame_by_area, "frame cavity areas must be unique"
        frame_by_area[area] = frame["cells"]

    halves: list[dict[str, int]] = []
    for obj in objects:
        halves.extend(obj["halves"])
    half_areas = [int(half["area"]) for half in halves]
    assert len(set(half_areas)) == len(half_areas), "half areas must be unique"
    assert sorted(frame_by_area) == sorted(half_areas), "frame areas must match half areas exactly"

    out = g.copy()
    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = int(bg)
    for half in sorted(halves, key=lambda item: int(item["area"])):
        for r, c in frame_by_area[int(half["area"])]:
            out[r, c] = int(half["color"])
    return out


AREA_POOL = (3, 4, 5, 6, 7, 8, 9, 10)
HALF_COUNT_CHOICES = (2, 3)
CANVAS_RANGE = (20, 28)
PLACEMENT_ATTEMPTS = 250
GEN_ATTEMPTS = 500


def _random_composition(total: int, parts: int, max_value: int = 4) -> list[int] | None:
    if total < parts or total > parts * max_value:
        return None
    remaining = int(total)
    values: list[int] = []
    for i in range(parts):
        slots_left = parts - i - 1
        lo = max(1, remaining - max_value * slots_left)
        hi = min(max_value, remaining - slots_left)
        if lo > hi:
            return None
        values.append(random.randint(lo, hi))
        remaining -= values[-1]
    random.shuffle(values)
    return values


def _maybe_flip(patch: np.ndarray) -> np.ndarray:
    out = patch
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int).copy()


def _build_vertical_object(left_area: int, right_area: int, color: int) -> np.ndarray | None:
    max_height = min(5, left_area, right_area)
    if max_height < 3:
        return None
    for _ in range(120):
        height = random.randint(3, max_height)
        left = _random_composition(left_area, height)
        right = _random_composition(right_area, height)
        if left is None or right is None:
            continue
        width = max(left) + 1 + max(right)
        if width <= height:
            continue
        if len(set(left)) == 1 and len(set(right)) == 1 and max(left) == max(right):
            continue
        patch = np.zeros((height, width), dtype=int)
        crack_col = max(left)
        for r in range(height):
            patch[r, crack_col - left[r] : crack_col] = int(color)
            patch[r, crack_col] = CRACK
            patch[r, crack_col + 1 : crack_col + 1 + right[r]] = int(color)
        return _maybe_flip(patch)
    return None


def _build_horizontal_object(top_area: int, bottom_area: int, color: int) -> np.ndarray | None:
    max_width = min(5, top_area, bottom_area)
    if max_width < 3:
        return None
    for _ in range(120):
        width = random.randint(3, max_width)
        top = _random_composition(top_area, width)
        bottom = _random_composition(bottom_area, width)
        if top is None or bottom is None:
            continue
        height = max(top) + 1 + max(bottom)
        if height <= width:
            continue
        if len(set(top)) == 1 and len(set(bottom)) == 1 and max(top) == max(bottom):
            continue
        patch = np.zeros((height, width), dtype=int)
        crack_row = max(top)
        for c in range(width):
            patch[crack_row - top[c] : crack_row, c] = int(color)
            patch[crack_row, c] = CRACK
            patch[crack_row + 1 : crack_row + 1 + bottom[c], c] = int(color)
        return _maybe_flip(patch)
    return None


def _build_object_patch(area_a: int, area_b: int, color: int) -> np.ndarray | None:
    area1, area2 = (area_a, area_b) if random.random() < 0.5 else (area_b, area_a)
    builders = [_build_vertical_object, _build_horizontal_object]
    random.shuffle(builders)
    for builder in builders:
        patch = builder(area1, area2, color)
        if patch is not None:
            return patch
    return None


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, area + 1):
        if area % h != 0:
            continue
        w = area // h
        if h <= 7 and w <= 7:
            pairs.append((h, w))
    return pairs


def _build_frame_patch(area: int) -> np.ndarray:
    pairs = _factor_pairs(area)
    assert pairs, "every chosen area must fit into a compact rectangular frame"
    inner_h, inner_w = random.choice(pairs)
    if random.random() < 0.5:
        inner_h, inner_w = inner_w, inner_h
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    return patch


def _pair_half_areas(areas: tuple[int, ...]) -> list[tuple[int, int]] | None:
    if not areas:
        return []
    first = areas[0]
    for idx in range(1, len(areas)):
        second = areas[idx]
        total = first + second
        if not (8 <= total <= 20):
            continue
        if total in areas:
            continue
        remainder = areas[1:idx] + areas[idx + 1 :]
        tail = _pair_half_areas(remainder)
        if tail is not None:
            return [(first, second)] + tail
    return None


def _choose_object_specs() -> list[tuple[int, int, int]] | None:
    n_objects = random.choice(HALF_COUNT_CHOICES)
    for _ in range(200):
        areas = tuple(sorted(random.sample(AREA_POOL, 2 * n_objects)))
        pairs = _pair_half_areas(areas)
        if pairs is None:
            continue
        colors = random.sample(list(BODY_COLORS), n_objects)
        specs = [(a, b, color) for (a, b), color in zip(pairs, colors)]
        random.shuffle(specs)
        return specs
    return None


def _boxes_overlap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        b1[2] + gap < b2[0]
        or b2[2] + gap < b1[0]
        or b1[3] + gap < b2[1]
        or b2[3] + gap < b1[1]
    )


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = map(int, patch.shape)
    sub = grid[top : top + h, left : left + w]
    mask = patch != BG
    assert not np.any(sub[mask] != BG)
    sub[mask] = patch[mask]


def _place_patches(
    object_patches: list[np.ndarray], frame_patches: list[np.ndarray]
) -> np.ndarray | None:
    patches = [("object", patch) for patch in object_patches] + [("frame", patch) for patch in frame_patches]
    largest_h = max(int(patch.shape[0]) for _, patch in patches)
    largest_w = max(int(patch.shape[1]) for _, patch in patches)
    footprint = sum((int(patch.shape[0]) + 2) * (int(patch.shape[1]) + 2) for _, patch in patches)
    side_floor = max(CANVAS_RANGE[0], largest_h + 2, largest_w + 2, int(np.ceil(np.sqrt(footprint))) + 2)

    for _ in range(120):
        high = CANVAS_RANGE[1]
        if side_floor > high:
            return None
        height = random.randint(side_floor, high)
        width = random.randint(side_floor, high)
        grid = np.zeros((height, width), dtype=int)

        placements: list[tuple[str, tuple[int, int, int, int]]] = []
        order = sorted(
            range(len(patches)),
            key=lambda i: int(patches[i][1].shape[0] * patches[i][1].shape[1]),
            reverse=True,
        )
        failed = False
        for idx in order:
            kind, patch = patches[idx]
            ph, pw = map(int, patch.shape)
            placed = False
            for _ in range(PLACEMENT_ATTEMPTS):
                top = random.randint(0, height - ph)
                left = random.randint(0, width - pw)
                box = (top, left, top + ph - 1, left + pw - 1)
                ok = True
                for other_kind, other_box in placements:
                    gap = 2 if kind != other_kind else 1
                    if _boxes_overlap(box, other_box, gap):
                        ok = False
                        break
                if not ok:
                    continue
                _paint_patch(grid, patch, top, left)
                placements.append((kind, box))
                placed = True
                break
            if not placed:
                failed = True
                break
        if not failed:
            return grid
    return None


def _validate_generated(grid: np.ndarray) -> None:
    bg, objects, frames = _analyze_scene(grid)
    assert bg == BG
    assert len({int(obj["color"]) for obj in objects}) == len(objects)

    half_areas: list[int] = []
    frame_areas = {int(frame["area"]) for frame in frames}
    orientations = {str(obj["orientation"]) for obj in objects}
    for obj in objects:
        areas = [int(half["area"]) for half in obj["halves"]]
        assert len(areas) == 2
        assert areas[0] != areas[1]
        assert 8 <= sum(areas) <= 20
        assert sum(areas) not in frame_areas, "whole-object area must not masquerade as a frame key"
        half_areas.extend(areas)
    assert len(set(half_areas)) == len(half_areas)
    assert frame_areas == set(half_areas)
    assert orientations <= {"vertical", "horizontal"}

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.any(out == CRACK)

    changed = np.zeros_like(grid, dtype=bool)
    area_to_color = {int(half["area"]): int(half["color"]) for obj in objects for half in obj["halves"]}
    for obj in objects:
        for r, c in obj["coords"]:
            changed[r, c] = True
            assert int(out[r, c]) == BG
    for frame in frames:
        for r, c in frame["cells"]:
            changed[r, c] = True
            assert int(out[r, c]) == area_to_color[int(frame["area"])]
        for r, c in frame["border"]:
            assert int(out[r, c]) == FRAME
    assert np.array_equal(out[~changed], grid[~changed])


def _gen_attempt() -> np.ndarray | None:
    specs = _choose_object_specs()
    if specs is None:
        return None

    object_patches: list[np.ndarray] = []
    frame_patches: list[np.ndarray] = []
    half_areas: list[int] = []
    for area_a, area_b, color in specs:
        patch = _build_object_patch(area_a, area_b, color)
        if patch is None:
            return None
        object_patches.append(patch)
        half_areas.extend([int(area_a), int(area_b)])
    for area in half_areas:
        frame_patches.append(_build_frame_patch(area))

    grid = _place_patches(object_patches, frame_patches)
    if grid is None:
        return None
    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fractured-area matching puzzle")
