from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location(
    "_arc_utils_new_arc_2_040_2",
    _ARC_UTILS_PATH,
)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BG_COLOR = Color.GREY
RECT_COLOR = Color.RED
DIVIDER_COLOR = Color.ORANGE
FALL_COLOR = Color.MAROON
SOURCE_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW)

GRID_H_RANGE = (18, 22)
GRID_W_RANGE = (18, 22)
RECT_SIZE_RANGE = (4, 6)
GEN_RETRIES = 400

CORNER_PATTERNS: list[tuple[tuple[str, ...], int]] = [
    (("tl",), 2),
    (("tr",), 2),
    (("bl",), 2),
    (("br",), 2),
    (("tl", "bl"), 4),
    (("tr", "br"), 4),
    (("tl", "tr"), 2),
    (("bl", "br"), 2),
    (("tl", "br"), 1),
    (("tr", "bl"), 1),
    (("tl", "tr", "bl"), 1),
    (("tl", "tr", "br"), 1),
    (("tl", "bl", "br"), 1),
    (("tr", "bl", "br"), 1),
    (("tl", "tr", "bl", "br"), 1),
]


def _outer_frame(rect_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, rect_bbox)
    return r0 - 1, c0 - 1, r1 + 1, c1 + 1


def _divider_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        if bool(np.all(grid[r] == grid[r, 0])) and int(grid[r, 0]) != int(bg):
            rows.append((int(r), int(grid[r, 0])))
    assert len(rows) == 1, "expected exactly one solid divider row"
    return rows[0]


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return (r1 > r0) and (c1 > c0) and bool((patch == color).all())


def _find_rectangle(grid: np.ndarray, bg: int, divider_color: int) -> dict:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    rects = [comp for comp in comps if int(comp["color"]) != int(divider_color) and _is_solid_rectangle(grid, comp)]
    assert len(rects) == 1, "expected exactly one solid rectangle"
    return rects[0]


def _collect_sources(grid: np.ndarray, bg: int, divider_row: int, divider_color: int, rect: dict) -> list[tuple[int, int, int]]:
    rect_bbox = tuple(map(int, rect["bbox"]))
    rect_color = int(rect["color"])
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    sources: list[tuple[int, int, int]] = []
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        color = int(comp["color"])
        area = int(comp["area"])
        if color == divider_color and bbox[0] == divider_row and bbox[2] == divider_row:
            continue
        if color == rect_color and bbox == rect_bbox:
            continue
        assert area == 1, "all remaining objects must be isolated source pixels"
        r, c = map(int, comp["coords"][0])
        sources.append((r, c, color))
    return sources


def _ray_end(rect_bbox: tuple[int, int, int, int], shape: tuple[int, int], source: tuple[int, int]) -> tuple[int, int] | None:
    h, w = map(int, shape)
    r, c = map(int, source)
    top, left, bottom, right = _outer_frame(rect_bbox)
    if c < left and top <= r <= bottom and 0 <= left < w:
        return r, left
    if c > right and top <= r <= bottom and 0 <= right < w:
        return r, right
    if r < top and left <= c <= right and 0 <= top < h:
        return top, c
    if r > bottom and left <= c <= right and 0 <= bottom < h:
        return bottom, c
    return None


def _segment_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("ray segments must be axis-aligned")


def _trace_rays(
    shape: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    sources: list[tuple[int, int, int]],
) -> list[dict[str, object]]:
    rays: list[dict[str, object]] = []
    for r, c, color in sources:
        end = _ray_end(rect_bbox, shape, (r, c))
        if end is None:
            continue
        rays.append(
            {
                "source": (int(r), int(c)),
                "color": int(color),
                "cells": _segment_cells((r, c), end),
            }
        )
    return rays


def _intersection_cells(rays: list[dict[str, object]]) -> set[tuple[int, int]]:
    touched: dict[tuple[int, int], set[int]] = {}
    for ray in rays:
        color = int(ray["color"])
        for cell in ray["cells"]:
            touched.setdefault(cell, set()).add(color)
    return {cell for cell, colors in touched.items() if len(colors) >= 2}


def main(grid: np.ndarray) -> np.ndarray:
    """Project source rays onto the rectangle's adjacent frame, then drop multi-color overlap cells above the divider.

    Input structure:
    - The dominant background is grey.
    - One solid red rectangle sits in the upper half, and one full-width orange divider row sits lower down.
    - Several isolated blue, green, or yellow pixels surround the rectangle; usable ray sources lie on a straight row or column corridor that reaches the rectangle's one-cell outer frame.

    Rule:
    - Find the rectangle and its one-cell outer frame. Any source directly left or right of that frame draws a horizontal same-color ray to it; any source directly above or below it draws a vertical same-color ray to it.
    - Wherever rays of two or more different colors visit the same cell, create an emergent maroon intersection, clear that cell from the ray drawing, and keep the non-intersection ray cells in place.
    - Drop the maroon intersection cells straight down in column order, stopping on the row above the divider or on top of earlier dropped maroon cells. Keep the rectangle, divider, background, and non-ray source pixels unchanged.

    Key insight:
    - The output's maroon pixels are not present in the input: they emerge only after the ray-extension step, and their final positions are determined by a second gravity step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = map(int, g.shape)

    bg = int(get_bg(g))
    divider_r, divider_color = _divider_row(g, bg)
    rect = _find_rectangle(g, bg, divider_color)
    rect_bbox = tuple(map(int, rect["bbox"]))
    sources = _collect_sources(g, bg, divider_r, divider_color, rect)
    rays = _trace_rays((h, w), rect_bbox, sources)
    intersections = _intersection_cells(rays)

    assert rays, "expected at least one aligned ray source"
    assert intersections, "expected at least one multi-color ray overlap"

    out = g.copy()
    for ray in rays:
        color = int(ray["color"])
        for r, c in ray["cells"]:
            out[int(r), int(c)] = color

    for r, c in intersections:
        out[int(r), int(c)] = bg

    settled: set[tuple[int, int]] = set()
    for r, c in sorted(intersections, key=lambda cell: (int(cell[1]), -int(cell[0]))):
        rr = int(r)
        cc = int(c)
        assert rr < divider_r
        while rr + 1 < divider_r and (rr + 1, cc) not in settled:
            rr += 1
        settled.add((rr, cc))
        out[rr, cc] = FALL_COLOR

    return out


def _is_isolated_pixel(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    h, w = map(int, grid.shape)
    if not (0 <= r < h and 0 <= c < w):
        return False
    if int(grid[r, c]) != int(bg):
        return False
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if int(grid[rr, cc]) != int(bg):
                return False
    return True


def _segment_clear(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], bg: int) -> bool:
    cells = _segment_cells(start, end)
    return all(int(grid[r, c]) == int(bg) for r, c in cells)


def _place_aligned_source(
    grid: np.ndarray,
    rect_bbox: tuple[int, int, int, int],
    divider_r: int,
    bg: int,
    side: str,
    fixed: int,
    color: int,
) -> tuple[int, int, int] | None:
    h, w = map(int, grid.shape)
    top, left, bottom, right = _outer_frame(rect_bbox)

    candidates: list[tuple[int, int]] = []
    if side == "left":
        candidates = [(int(fixed), c) for c in range(1, left - 1)]
    elif side == "right":
        candidates = [(int(fixed), c) for c in range(right + 2, w - 1)]
    elif side == "up":
        candidates = [(r, int(fixed)) for r in range(1, top - 1)]
    elif side == "down":
        candidates = [(r, int(fixed)) for r in range(bottom + 2, divider_r - 1)]
    else:
        raise ValueError(f"unknown side {side}")

    random.shuffle(candidates)
    for r, c in candidates:
        end = _ray_end(rect_bbox, (h, w), (r, c))
        if end is None:
            continue
        if not _is_isolated_pixel(grid, r, c, bg):
            continue
        if not _segment_clear(grid, (r, c), end, bg):
            continue
        grid[r, c] = int(color)
        return int(r), int(c), int(color)
    return None


def _place_unaligned_source(
    grid: np.ndarray,
    rect_bbox: tuple[int, int, int, int],
    divider_r: int,
    bg: int,
    color: int,
) -> tuple[int, int, int] | None:
    h, w = map(int, grid.shape)
    for _ in range(200):
        r = random.randint(1, divider_r - 2)
        c = random.randint(1, w - 2)
        if _ray_end(rect_bbox, (h, w), (r, c)) is not None:
            continue
        if not _is_isolated_pixel(grid, r, c, bg):
            continue
        grid[r, c] = int(color)
        return int(r), int(c), int(color)
    return None


def _choose_pattern() -> tuple[str, ...]:
    patterns = [pattern for pattern, _weight in CORNER_PATTERNS]
    weights = [weight for _pattern, weight in CORNER_PATTERNS]
    return random.choices(patterns, weights=weights, k=1)[0]


def _corner_specs(rect_bbox: tuple[int, int, int, int]) -> dict[str, tuple[tuple[str, int], tuple[str, int]]]:
    top, left, bottom, right = _outer_frame(rect_bbox)
    return {
        "tl": (("left", top), ("up", left)),
        "tr": (("right", top), ("up", right)),
        "bl": (("left", bottom), ("down", left)),
        "br": (("right", bottom), ("down", right)),
    }


def _extra_specs(rect_bbox: tuple[int, int, int, int]) -> list[tuple[str, int]]:
    r0, c0, r1, c1 = map(int, rect_bbox)
    specs: list[tuple[str, int]] = []
    specs.extend(("left", r) for r in range(r0, r1 + 1))
    specs.extend(("right", r) for r in range(r0, r1 + 1))
    specs.extend(("up", c) for c in range(c0, c1 + 1))
    specs.extend(("down", c) for c in range(c0, c1 + 1))
    return specs


def _target_source_count(base_count: int) -> int:
    low = max(4, base_count)
    if base_count <= 2:
        high = min(8, base_count + 4)
    elif base_count <= 4:
        high = min(8, base_count + 3)
    else:
        high = 8
    return random.randint(low, high)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    divider_r = h - random.choice([3, 4])

    rect_h = random.randint(*RECT_SIZE_RANGE)
    rect_w = random.randint(*RECT_SIZE_RANGE)
    rect_r0_min = 4
    rect_r0_max = divider_r - rect_h - 4
    rect_c0_min = 4
    rect_c0_max = w - rect_w - 5
    if rect_r0_min > rect_r0_max or rect_c0_min > rect_c0_max:
        return None

    rect_r0 = random.randint(rect_r0_min, rect_r0_max)
    rect_c0 = random.randint(rect_c0_min, rect_c0_max)
    rect_r1 = rect_r0 + rect_h - 1
    rect_c1 = rect_c0 + rect_w - 1
    rect_bbox = (rect_r0, rect_c0, rect_r1, rect_c1)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[divider_r, :] = DIVIDER_COLOR
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = RECT_COLOR

    specs_by_corner = _corner_specs(rect_bbox)
    corners = _choose_pattern()
    used_specs: set[tuple[str, int]] = set()
    sources: list[tuple[int, int, int]] = []

    for corner in corners:
        first_spec, second_spec = specs_by_corner[corner]
        colors = random.sample(SOURCE_COLORS, 2)
        for spec, color in zip((first_spec, second_spec), colors):
            if spec in used_specs:
                return None
            placed = _place_aligned_source(grid, rect_bbox, divider_r, BG_COLOR, spec[0], spec[1], color)
            if placed is None:
                return None
            used_specs.add(spec)
            sources.append(placed)

    target_count = _target_source_count(len(sources))
    extras_needed = target_count - len(sources)

    if extras_needed > 0:
        extra_specs = [spec for spec in _extra_specs(rect_bbox) if spec not in used_specs]
        random.shuffle(extra_specs)
        extra_aligned = random.randint(0, extras_needed)
        if extras_needed >= 2 and extra_aligned == 0:
            extra_aligned = 1
        extra_aligned = min(extra_aligned, len(extra_specs))

        for spec in extra_specs[:extra_aligned]:
            color = random.choice(SOURCE_COLORS)
            placed = _place_aligned_source(grid, rect_bbox, divider_r, BG_COLOR, spec[0], spec[1], color)
            if placed is None:
                continue
            used_specs.add(spec)
            sources.append(placed)

    while len(sources) < target_count:
        color = random.choice(SOURCE_COLORS)
        placed = _place_unaligned_source(grid, rect_bbox, divider_r, BG_COLOR, color)
        if placed is None:
            break
        sources.append(placed)

    if not (4 <= len(sources) <= 8):
        return None

    rays = _trace_rays(grid.shape, rect_bbox, sources)
    intersections = _intersection_cells(rays)
    if not intersections:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for new puzzle 2")
