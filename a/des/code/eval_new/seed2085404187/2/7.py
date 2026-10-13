from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 8
MAROON = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7)
DIM_OPTIONS = tuple((h, w) for h in range(3, 7) for w in range(3, 7))
GEN_RETRIES = 500
ACTIVE_RETRIES = 250
PASSIVE_RETRIES = 120
EXTRA_RECT_OPTIONS = (0, 1, 2)
MODE_OPTIONS = ("horizontal_wins", "vertical_wins", "equal")


def _spec_from_box(
    r0: int,
    c0: int,
    h: int,
    w: int,
    color: int,
    gap_side: str,
    offset: int,
) -> dict:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if gap_side == "top":
        assert 1 <= offset <= w - 2
        gap = (r0, c0 + offset)
        direction = (-1, 0)
    elif gap_side == "bottom":
        assert 1 <= offset <= w - 2
        gap = (r1, c0 + offset)
        direction = (1, 0)
    elif gap_side == "left":
        assert 1 <= offset <= h - 2
        gap = (r0 + offset, c0)
        direction = (0, -1)
    elif gap_side == "right":
        assert 1 <= offset <= h - 2
        gap = (r0 + offset, c1)
        direction = (0, 1)
    else:
        raise ValueError(gap_side)
    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "h": h,
        "w": w,
        "color": color,
        "gap": gap,
        "dir": direction,
        "area": h * w,
        "gap_side": gap_side,
    }


def _components_by_color(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for comp in _components_by_color(grid):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        color = int(grid[comp[0]])

        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        bg_inside = np.argwhere(window == BG)
        assert len(comp) == h * w - 1, "component must be a full rectangle with one missing cell"
        assert bg_inside.shape == (1, 2), "rectangle bbox must contain exactly one gap"

        gr = int(bg_inside[0, 0]) + r0
        gc = int(bg_inside[0, 1]) + c0
        if gr == r0 and c0 < gc < c1:
            direction = (-1, 0)
            gap_side = "top"
        elif gr == r1 and c0 < gc < c1:
            direction = (1, 0)
            gap_side = "bottom"
        elif gc == c0 and r0 < gr < r1:
            direction = (0, -1)
            gap_side = "left"
        elif gc == c1 and r0 < gr < r1:
            direction = (0, 1)
            gap_side = "right"
        else:
            raise AssertionError("gap must be a non-corner bbox boundary cell")

        rects.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": h,
                "w": w,
                "color": color,
                "gap": (gr, gc),
                "dir": direction,
                "area": h * w,
                "gap_side": gap_side,
            }
        )
    return rects


def _ray_cells(rect: dict, grid: np.ndarray, gap_positions: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    dr, dc = rect["dir"]
    r, c = rect["gap"][0] + dr, rect["gap"][1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != BG:
            break
        if (r, c) not in gap_positions:
            cells.append((r, c))
        r += dr
        c += dc
    return cells


def _paint_rays(grid: np.ndarray, rects: list[dict]) -> np.ndarray:
    gap_positions = {tuple(rect["gap"]) for rect in rects}
    ray_infos = []
    hits: dict[tuple[int, int], list[int]] = defaultdict(list)
    for rect in rects:
        cells = _ray_cells(rect, grid, gap_positions)
        ray_infos.append(
            {
                "color": rect["color"],
                "area": rect["area"],
                "orientation": "H" if rect["dir"][0] == 0 else "V",
                "cells": cells,
            }
        )
    for idx, ray in enumerate(ray_infos):
        for cell in ray["cells"]:
            hits[cell].append(idx)

    out = grid.copy()
    for (r, c), ids in hits.items():
        horiz = [idx for idx in ids if ray_infos[idx]["orientation"] == "H"]
        vert = [idx for idx in ids if ray_infos[idx]["orientation"] == "V"]
        if horiz and vert:
            h_idx = max(horiz, key=lambda idx: ray_infos[idx]["area"])
            v_idx = max(vert, key=lambda idx: ray_infos[idx]["area"])
            h_area = ray_infos[h_idx]["area"]
            v_area = ray_infos[v_idx]["area"]
            if h_area == v_area:
                out[r, c] = MAROON
            elif h_area > v_area:
                out[r, c] = ray_infos[h_idx]["color"]
            else:
                out[r, c] = ray_infos[v_idx]["color"]
        else:
            winner = max(ids, key=lambda idx: ray_infos[idx]["area"])
            out[r, c] = ray_infos[winner]["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast rays out of single-pixel rectangle boundary gaps and color each crossing by the larger source rectangle.

    Input structure:
    - The background is azure.
    - Several separated monochrome rectangles appear, and each rectangle is missing exactly one non-corner boundary pixel.
    - Every colored component therefore occupies a full rectangular bbox except for one background gap on one edge.

    Rule:
    - For each colored component, recover its full bbox, locate the unique boundary gap, and infer the outward direction from the gap side.
    - Fire a one-cell-wide ray from just beyond the gap through background cells until the next step would leave the grid or hit a colored rectangle cell.
    - Keep every rectangle and every gap unchanged; when a horizontal ray and a vertical ray share a pixel, use the color of the larger source bbox area, or maroon if the two source areas are equal.

    Key insight:
    - A component whose area is exactly one less than its bbox area reveals both the hidden rectangle and the missing edge cell, so the recovered bbox area can drive crossing priority without relying on any color-id semantics.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rects = _extract_rectangles(g)
    return _paint_rays(g, rects)


def _render_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        grid[spec["r0"] : spec["r1"] + 1, spec["c0"] : spec["c1"] + 1] = spec["color"]
    for spec in specs:
        gr, gc = spec["gap"]
        grid[gr, gc] = BG
    return grid


def _scene_analysis(specs: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[dict], list[str], bool]:
    grid = _render_input(specs, shape)
    gap_positions = {tuple(spec["gap"]) for spec in specs}
    ray_infos: list[dict] = []
    hits: dict[tuple[int, int], list[int]] = defaultdict(list)
    for spec in specs:
        cells = _ray_cells(spec, grid, gap_positions)
        ray_infos.append(
            {
                "color": spec["color"],
                "area": spec["area"],
                "orientation": "H" if spec["dir"][0] == 0 else "V",
                "cells": cells,
            }
        )
    for idx, ray in enumerate(ray_infos):
        for cell in ray["cells"]:
            hits[cell].append(idx)

    out = grid.copy()
    crossing_kinds: list[str] = []
    same_orientation_overlap = False
    for (r, c), ids in hits.items():
        horiz = [idx for idx in ids if ray_infos[idx]["orientation"] == "H"]
        vert = [idx for idx in ids if ray_infos[idx]["orientation"] == "V"]
        if len(horiz) > 1 or len(vert) > 1:
            same_orientation_overlap = True
        if horiz and vert:
            h_idx = max(horiz, key=lambda idx: ray_infos[idx]["area"])
            v_idx = max(vert, key=lambda idx: ray_infos[idx]["area"])
            h_area = ray_infos[h_idx]["area"]
            v_area = ray_infos[v_idx]["area"]
            if h_area == v_area:
                out[r, c] = MAROON
                crossing_kinds.append("equal")
            elif h_area > v_area:
                out[r, c] = ray_infos[h_idx]["color"]
                crossing_kinds.append("horizontal_wins")
            else:
                out[r, c] = ray_infos[v_idx]["color"]
                crossing_kinds.append("vertical_wins")
        else:
            winner = max(ids, key=lambda idx: ray_infos[idx]["area"])
            out[r, c] = ray_infos[winner]["color"]
    return grid, out, ray_infos, crossing_kinds, same_orientation_overlap


def _too_close(a: dict, b: dict) -> bool:
    separated = (
        a["r1"] + 1 < b["r0"]
        or b["r1"] + 1 < a["r0"]
        or a["c1"] + 1 < b["c0"]
        or b["c1"] + 1 < a["c0"]
    )
    return not separated


def _specs_are_spaced(specs: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    for spec in specs:
        if spec["r0"] < 1 or spec["c0"] < 1 or spec["r1"] > h - 2 or spec["c1"] > w - 2:
            return False
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if _too_close(specs[i], specs[j]):
                return False
    return True


def _sample_dims(mode: str) -> tuple[tuple[int, int], tuple[int, int]]:
    for _ in range(200):
        horiz = random.choice(DIM_OPTIONS)
        vert = random.choice(DIM_OPTIONS)
        ha = horiz[0] * horiz[1]
        va = vert[0] * vert[1]
        if mode == "equal" and ha == va:
            return horiz, vert
        if mode == "horizontal_wins" and ha > va:
            return horiz, vert
        if mode == "vertical_wins" and va > ha:
            return horiz, vert
    raise RuntimeError("failed to sample compatible dimensions")


def _make_horizontal_cross_spec(
    cross_r: int,
    cross_c: int,
    location: str,
    dims: tuple[int, int],
    color: int,
    offset: int,
    ray_len: int,
) -> dict:
    h, w = dims
    r0 = cross_r - offset
    if location == "left":
        c0 = cross_c - ray_len - w + 1
        return _spec_from_box(r0, c0, h, w, color, "right", offset)
    c0 = cross_c + ray_len
    return _spec_from_box(r0, c0, h, w, color, "left", offset)


def _make_vertical_cross_spec(
    cross_r: int,
    cross_c: int,
    location: str,
    dims: tuple[int, int],
    color: int,
    offset: int,
    ray_len: int,
) -> dict:
    h, w = dims
    c0 = cross_c - offset
    if location == "up":
        r0 = cross_r - ray_len - h + 1
        return _spec_from_box(r0, c0, h, w, color, "bottom", offset)
    r0 = cross_r + ray_len
    return _spec_from_box(r0, c0, h, w, color, "top", offset)


def _try_active_pair(shape: tuple[int, int], mode: str, colors: tuple[int, int]) -> tuple[list[dict], int, int] | None:
    h, w = shape
    horiz_dims, vert_dims = _sample_dims(mode)
    for _ in range(ACTIVE_RETRIES):
        h_loc = random.choice(("left", "right"))
        v_loc = random.choice(("up", "down"))
        h_offset = random.randint(1, horiz_dims[0] - 2)
        v_offset = random.randint(1, vert_dims[1] - 2)
        ray_h = random.randint(2, min(6, w // 3))
        ray_v = random.randint(2, min(6, h // 3))

        row_lo = max(3, 1 + h_offset)
        row_hi = min(h - 4, h - horiz_dims[0] - 1 + h_offset)
        if v_loc == "up":
            row_lo = max(row_lo, vert_dims[0] + ray_v)
        else:
            row_hi = min(row_hi, h - vert_dims[0] - ray_v - 1)

        col_lo = max(3, 1 + v_offset)
        col_hi = min(w - 4, w - vert_dims[1] - 1 + v_offset)
        if h_loc == "left":
            col_lo = max(col_lo, horiz_dims[1] + ray_h)
        else:
            col_hi = min(col_hi, w - horiz_dims[1] - ray_h - 1)

        if row_lo > row_hi or col_lo > col_hi:
            continue

        cross_r = random.randint(row_lo, row_hi)
        cross_c = random.randint(col_lo, col_hi)
        specs = [
            _make_horizontal_cross_spec(cross_r, cross_c, h_loc, horiz_dims, colors[0], h_offset, ray_h),
            _make_vertical_cross_spec(cross_r, cross_c, v_loc, vert_dims, colors[1], v_offset, ray_v),
        ]
        if not _specs_are_spaced(specs, shape):
            continue
        grid, out, ray_infos, crossing_kinds, same_overlap = _scene_analysis(specs, shape)
        if same_overlap or crossing_kinds != [mode]:
            continue
        if any(len(ray["cells"]) < 2 for ray in ray_infos):
            continue
        if np.array_equal(grid, out):
            continue
        return specs, cross_r, cross_c
    return None


def _try_passive_spec(
    specs: list[dict],
    shape: tuple[int, int],
    cross_r: int,
    cross_c: int,
    color: int,
    mode: str,
) -> dict | None:
    h, w = shape
    for _ in range(PASSIVE_RETRIES):
        rect_h, rect_w = random.choice(DIM_OPTIONS)
        side = random.choice(("left", "right", "up", "down"))
        if side == "left":
            if cross_c - rect_w - 2 < 2 or h - rect_h - 2 < 1:
                continue
            r0 = random.randint(1, h - rect_h - 2)
            c0 = random.randint(2, cross_c - rect_w - 2)
            offset = random.randint(1, rect_h - 2)
            spec = _spec_from_box(r0, c0, rect_h, rect_w, color, "left", offset)
        elif side == "right":
            lo = max(cross_c + 2, 1)
            hi = w - rect_w - 3
            if lo > hi or h - rect_h - 2 < 1:
                continue
            r0 = random.randint(1, h - rect_h - 2)
            c0 = random.randint(lo, hi)
            offset = random.randint(1, rect_h - 2)
            spec = _spec_from_box(r0, c0, rect_h, rect_w, color, "right", offset)
        elif side == "up":
            if cross_r - rect_h - 2 < 2 or w - rect_w - 2 < 1:
                continue
            r0 = random.randint(2, cross_r - rect_h - 2)
            c0 = random.randint(1, w - rect_w - 2)
            offset = random.randint(1, rect_w - 2)
            spec = _spec_from_box(r0, c0, rect_h, rect_w, color, "top", offset)
        else:
            lo = max(cross_r + 2, 1)
            hi = h - rect_h - 3
            if lo > hi or w - rect_w - 2 < 1:
                continue
            r0 = random.randint(lo, hi)
            c0 = random.randint(1, w - rect_w - 2)
            offset = random.randint(1, rect_w - 2)
            spec = _spec_from_box(r0, c0, rect_h, rect_w, color, "bottom", offset)

        trial = specs + [spec]
        if not _specs_are_spaced(trial, shape):
            continue
        grid, out, ray_infos, crossing_kinds, same_overlap = _scene_analysis(trial, shape)
        if same_overlap or crossing_kinds != [mode]:
            continue
        if any(len(ray["cells"]) < 2 for ray in ray_infos):
            continue
        if np.array_equal(grid, out):
            continue
        return spec
    return None


def _background_is_dominant(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[np.where(values == BG)[0][0]]) if BG in values else 0
    return bg_count == int(counts.max())


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(16, 29), random.randint(16, 29))
    mode = random.choice(MODE_OPTIONS)
    extra_target = random.choice(EXTRA_RECT_OPTIONS)
    colors = random.sample(PALETTE, 2 + extra_target)

    active = _try_active_pair(shape, mode, (colors[0], colors[1]))
    if active is None:
        return None
    specs, cross_r, cross_c = active

    for color in colors[2:]:
        extra = _try_passive_spec(specs, shape, cross_r, cross_c, color, mode)
        if extra is not None:
            specs.append(extra)

    if not _specs_are_spaced(specs, shape):
        return None

    grid, expected, _ray_infos, crossing_kinds, same_overlap = _scene_analysis(specs, shape)
    if same_overlap or crossing_kinds != [mode]:
        return None
    if not _background_is_dominant(grid):
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-ray puzzle input")
