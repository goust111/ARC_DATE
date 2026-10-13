import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_VEC = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ALL_COLORS = tuple(range(10))

# Generator configuration.
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 32)
FENCE_COUNT_RANGE = (2, 4)
SEED_COUNT_RANGE = (1, 8)
INTERIOR_H_RANGE = (2, 5)
INTERIOR_W_RANGE = (2, 6)
PALETTE_MIN_COL_GAP = 2
PALETTE_COL_MARGIN = 1
PLACEMENT_TRIES = 400
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=10)
    return int(np.argmax(counts))


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _monochrome_components(
    grid: np.ndarray, bg: int, blocked: np.ndarray | None = None
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    if blocked is None:
        blocked = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or blocked[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or blocked[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            rows = [r0 for r0, _ in pts]
            cols = [c0 for _, c0 in pts]
            comps.append(
                {
                    "color": color,
                    "cells": pts,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "touches_border": touches_border,
                }
            )
    return comps


def _extract_palette(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping: dict[int, int] = {}
    palette_mask = np.zeros((h, w), dtype=bool)
    for comp in _mask_components(g != bg):
        if len(comp) != 2:
            continue
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        pts = sorted(comp)
        (r0, c0), (r1, c1) = pts
        if c0 != c1 or r1 != r0 + 1:
            continue
        src = int(g[r0, c0])
        dst = int(g[r1, c1])
        if src in mapping:
            assert mapping[src] == dst, "palette mapping must be unique"
        mapping[src] = dst
        palette_mask[r0, c0] = True
        palette_mask[r1, c1] = True
    assert mapping, "expected at least one border palette entry"
    return mapping, palette_mask


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _extract_fences(grid: np.ndarray, bg: int, palette_mask: np.ndarray, palette: dict[int, int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    fences: list[dict] = []
    for comp in _monochrome_components(g, bg, blocked=palette_mask):
        if comp["touches_border"]:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        border = _rect_border_cells(r0, c0, r1, c1)
        pts = set(comp["cells"])
        if not pts.issubset(border):
            continue
        if len(pts) != len(border) - 1:
            continue
        missing = list(border - pts)
        if len(missing) != 1:
            continue
        gap_r, gap_c = missing[0]
        if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
            continue
        if gap_r == r0:
            dr, dc = SIDE_TO_VEC["top"]
        elif gap_r == r1:
            dr, dc = SIDE_TO_VEC["bottom"]
        elif gap_c == c0:
            dr, dc = SIDE_TO_VEC["left"]
        else:
            dr, dc = SIDE_TO_VEC["right"]

        seed_cells: list[tuple[int, int]] = []
        seed_colors: set[int] = set()
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                value = int(g[rr, cc])
                if value == bg:
                    continue
                seed_cells.append((rr, cc))
                seed_colors.add(value)
        if not seed_cells or len(seed_colors) != 1:
            continue
        seed_color = next(iter(seed_colors))
        if seed_color == comp["color"] or seed_color not in palette:
            continue
        fences.append(
            {
                "wall_color": int(comp["color"]),
                "seed_color": seed_color,
                "branch_color": int(palette[seed_color]),
                "seed_count": len(seed_cells),
                "gap": (gap_r, gap_c),
                "direction": (dr, dc),
            }
        )
    assert fences, "expected at least one valid gapped fence"
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a branch from each rectangular fence gap.

    The seed color inside a fence picks the branch color via the border palette,
    and the number of seed pixels gives the branch length.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    palette, palette_mask = _extract_palette(g, bg)
    fences = _extract_fences(g, bg, palette_mask, palette)

    out = g.copy()
    h, w = g.shape
    for fence in fences:
        gap_r, gap_c = fence["gap"]
        dr, dc = fence["direction"]
        for step in range(fence["seed_count"]):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            assert 0 <= rr < h and 0 <= cc < w
            out[rr, cc] = fence["branch_color"]
    return out


def _sample_spaced_positions(
    rng: random.Random, start: int, stop: int, count: int, min_gap: int
) -> list[int] | None:
    choices = list(range(start, stop + 1))
    if len(choices) < count:
        return None
    for _ in range(200):
        picked = sorted(rng.sample(choices, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _choose_dims(rng: random.Random, seed_count: int) -> tuple[int, int]:
    options: list[tuple[int, int]] = []
    for ih in range(INTERIOR_H_RANGE[0], INTERIOR_H_RANGE[1] + 1):
        for iw in range(INTERIOR_W_RANGE[0], INTERIOR_W_RANGE[1] + 1):
            area = ih * iw
            if area >= seed_count + 1 and area <= seed_count + 8:
                options.append((ih, iw))
    if not options:
        for ih in range(INTERIOR_H_RANGE[0], INTERIOR_H_RANGE[1] + 1):
            for iw in range(INTERIOR_W_RANGE[0], INTERIOR_W_RANGE[1] + 1):
                if ih * iw >= seed_count:
                    options.append((ih, iw))
    return rng.choice(options)


def _build_fence_geometry(
    r0: int, c0: int, outer_h: int, outer_w: int, side: str, offset: int
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    wall = _rect_border_cells(r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    if side == "top":
        gap = (r0, c0 + offset)
    elif side == "bottom":
        gap = (r0 + outer_h - 1, c0 + offset)
    elif side == "left":
        gap = (r0 + offset, c0)
    else:
        gap = (r0 + offset, c0 + outer_w - 1)
    wall.remove(gap)
    return wall, gap, SIDE_TO_VEC[side]


def _branch_cells(gap: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(gap[0] + dr * step, gap[1] + dc * step) for step in range(length)]


def _dilate(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _render_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in specs:
        for rr, cc in _branch_cells(spec["gap"], spec["direction"], spec["seed_count"]):
            out[rr, cc] = spec["branch_color"]
    return out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(ALL_COLORS)
    remaining = [c for c in ALL_COLORS if c != bg]
    rng.shuffle(remaining)

    num_fences = rng.randint(*FENCE_COUNT_RANGE)
    if len(remaining) < 1 + 2 * num_fences:
        return None

    wall_color = remaining[0]
    seed_colors = remaining[1 : 1 + num_fences]
    branch_colors = remaining[1 + num_fences : 1 + 2 * num_fences]
    entries = []
    for seed_color, branch_color in zip(seed_colors, branch_colors):
        entries.append(
            {
                "seed_color": seed_color,
                "branch_color": branch_color,
                "seed_count": rng.randint(*SEED_COUNT_RANGE),
                "side": rng.choice(list(SIDE_TO_VEC)),
            }
        )

    grid = np.full((h, w), bg, dtype=int)
    palette_side = rng.choice(["top", "bottom"])
    palette_rows = (0, 1) if palette_side == "top" else (h - 2, h - 1)
    palette_cols = _sample_spaced_positions(
        rng,
        PALETTE_COL_MARGIN,
        w - 1 - PALETTE_COL_MARGIN,
        num_fences,
        PALETTE_MIN_COL_GAP,
    )
    if palette_cols is None:
        return None

    palette_entries = entries[:]
    rng.shuffle(palette_entries)
    palette_cells: set[tuple[int, int]] = set()
    upper_row, lower_row = palette_rows
    for col, entry in zip(palette_cols, palette_entries):
        grid[upper_row, col] = entry["seed_color"]
        grid[lower_row, col] = entry["branch_color"]
        palette_cells.add((upper_row, col))
        palette_cells.add((lower_row, col))

    reserved = _dilate(palette_cells, h, w, pad=1)
    specs: list[dict] = []
    place_order = sorted(entries, key=lambda item: item["seed_count"], reverse=True)

    for entry in place_order:
        seed_count = entry["seed_count"]
        side = entry["side"]
        placed = False
        for _ in range(PLACEMENT_TRIES):
            inner_h, inner_w = _choose_dims(rng, seed_count)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            offset = rng.randint(1, outer_w - 2) if side in ("top", "bottom") else rng.randint(1, outer_h - 2)

            r_min = 1
            r_max = h - outer_h - 1
            c_min = 1
            c_max = w - outer_w - 1
            if side == "top":
                r_min = max(r_min, seed_count - 1)
            elif side == "bottom":
                r_max = min(r_max, h - outer_h - seed_count + 1)
            elif side == "left":
                c_min = max(c_min, seed_count - 1)
            else:
                c_max = min(c_max, w - outer_w - seed_count + 1)
            if r_min > r_max or c_min > c_max:
                continue

            r0 = rng.randint(r_min, r_max)
            c0 = rng.randint(c_min, c_max)
            wall_cells, gap, direction = _build_fence_geometry(r0, c0, outer_h, outer_w, side, offset)
            branch_cells = _branch_cells(gap, direction, seed_count)
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in branch_cells):
                continue

            interior_cells = [
                (rr, cc)
                for rr in range(r0 + 1, r0 + outer_h - 1)
                for cc in range(c0 + 1, c0 + outer_w - 1)
            ]
            seed_cells = set(rng.sample(interior_cells, seed_count))
            footprint = set(wall_cells) | seed_cells | set(branch_cells)
            reserve = _dilate(footprint, h, w, pad=1)
            if reserve & reserved:
                continue

            for rr, cc in wall_cells:
                grid[rr, cc] = wall_color
            for rr, cc in seed_cells:
                grid[rr, cc] = entry["seed_color"]

            specs.append(
                {
                    "seed_color": entry["seed_color"],
                    "branch_color": entry["branch_color"],
                    "seed_count": seed_count,
                    "gap": gap,
                    "direction": direction,
                }
            )
            reserved |= reserve
            placed = True
            break
        if not placed:
            return None

    expected = _render_expected(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        attempt = _gen_attempt(rng)
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("generate_input: failed to build a valid sample")
