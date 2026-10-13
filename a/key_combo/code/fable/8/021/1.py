from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, bbox_h, bbox_w, crop_bbox, find_components, get_bg


PALETTE = list(range(10))
GEN_RETRIES = 400


def _mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if coords.size == 0:
        return mask
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _non_bg_colors(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(c) for c in np.unique(grid) if int(c) != int(bg))


def _is_solid_horizontal_rect(grid: np.ndarray, comp: dict) -> bool:
    b = comp["bbox"]
    h = bbox_h(b)
    w = bbox_w(b)
    if h < 2 or w < 5 or w <= h:
        return False
    patch = crop_bbox(grid, b)
    return bool((patch == int(comp["color"])).all())


def _extract_objects(
    input_grid: np.ndarray,
) -> tuple[np.ndarray, int, int, list[dict], list[tuple[int, int]], dict[int, np.ndarray]]:
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    top_colors = sorted({int(v) for v in g[0] if int(v) != int(bg)})
    assert len(top_colors) == 1, "expected exactly one water color on the top row"
    water = int(top_colors[0])

    shelves: list[dict] = []
    motif_abs_masks: dict[int, np.ndarray] = {}

    for color in _non_bg_colors(g, bg):
        if int(color) == int(water):
            continue
        comps = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(color)],
            return_format="dict",
        )
        shelf_candidates = [comp for comp in comps if _is_solid_horizontal_rect(g, comp)]
        assert shelf_candidates, f"missing shelf for color {color}"
        shelf_candidates.sort(
            key=lambda comp: (
                int(comp["area"]),
                bbox_w(comp["bbox"]),
                -bbox_h(comp["bbox"]),
                -int(comp["bbox"][0]),
                -int(comp["bbox"][1]),
            ),
            reverse=True,
        )
        shelf = shelf_candidates[0]

        color_mask = g == int(color)
        shelf_mask = np.zeros_like(color_mask, dtype=bool)
        shelf_mask[shelf["coords"][:, 0], shelf["coords"][:, 1]] = True
        motif_abs = color_mask & ~shelf_mask
        assert motif_abs.any(), f"missing motif for shelf color {color}"

        mb = bbox(motif_abs)
        assert mb is not None
        motif_coords = np.argwhere(motif_abs)
        rel = motif_coords - np.array([[mb[0], mb[1]]], dtype=int)
        motif_mask = _mask_from_coords(rel, (bbox_h(mb), bbox_w(mb)))

        r0, c0, r1, c1 = map(int, shelf["bbox"])
        shelves.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "motif_mask": motif_mask,
            }
        )
        motif_abs_masks[int(color)] = motif_abs

    water_comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[int(water)],
        return_format="dict",
    )
    source_starts: list[tuple[int, int]] = []
    for comp in water_comps:
        coords = comp["coords"]
        if not bool((coords[:, 0] == 0).any()):
            continue
        bottom = int(coords[:, 0].max())
        cols = sorted({int(c) for r, c in coords if int(r) == int(bottom)})
        for c in cols:
            source_starts.append((int(bottom), int(c)))
    assert source_starts, "expected at least one top-touching water source"

    shelves.sort(key=lambda item: (int(item["r0"]), int(item["c0"])))
    return g, bg, water, shelves, source_starts, motif_abs_masks


def _carve_shelves(
    g: np.ndarray,
    bg: int,
    shelves: list[dict],
    motif_abs_masks: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[dict]]:
    base = np.array(g, copy=True)
    carved_shelves: list[dict] = []

    for color, motif_abs in motif_abs_masks.items():
        base[motif_abs] = int(bg)

    for shelf in shelves:
        motif_mask = shelf["motif_mask"]
        mh, mw = map(int, motif_mask.shape)
        assert mh <= int(shelf["h"]) and mw <= int(shelf["w"])
        off_r = (int(shelf["h"]) - mh) // 2
        off_c = (int(shelf["w"]) - mw) // 2

        hole_rows = shelf["r0"] + off_r + np.argwhere(motif_mask)[:, 0]
        hole_cols = shelf["c0"] + off_c + np.argwhere(motif_mask)[:, 1]
        base[hole_rows, hole_cols] = int(bg)

        through_cols = [
            int(c)
            for c in range(int(shelf["c0"]), int(shelf["c1"]) + 1)
            if bool((base[int(shelf["r0"]) : int(shelf["r1"]) + 1, int(c)] == int(bg)).all())
        ]
        carved = dict(shelf)
        carved["through_cols"] = through_cols
        carved_shelves.append(carved)

    return base, carved_shelves


def _find_shelf_at(shelves: list[dict], r: int, c: int) -> dict | None:
    for shelf in shelves:
        if (
            int(shelf["r0"]) <= int(r) <= int(shelf["r1"])
            and int(shelf["c0"]) <= int(c) <= int(shelf["c1"])
        ):
            return shelf
    return None


def _is_solid(base: np.ndarray, r: int, c: int, bg: int, water: int) -> bool:
    v = int(base[int(r), int(c)])
    return int(v) != int(bg) and int(v) != int(water)


def _is_through_hole(base: np.ndarray, shelf: dict, c: int, bg: int) -> bool:
    col = base[int(shelf["r0"]) : int(shelf["r1"]) + 1, int(c)]
    return bool((col == int(bg)).all())


def _simulate_water(
    base: np.ndarray,
    bg: int,
    water: int,
    shelves: list[dict],
    source_starts: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.array(base, copy=True)
    h, w = out.shape
    q = deque((int(r), int(c)) for r, c in source_starts)
    seen: set[tuple[int, int]] = set()
    stats = {
        "drops": 0,
        "impacts": 0,
        "hole_branches": 0,
        "overflow_branches": 0,
        "spread_cells": 0,
    }

    while q:
        r, c = q.popleft()
        if not (0 <= int(r) < h and 0 <= int(c) < w):
            continue
        if (int(r), int(c)) in seen:
            continue
        seen.add((int(r), int(c)))
        stats["drops"] += 1

        rr, cc = int(r), int(c)
        out[rr, cc] = int(water)

        while True:
            nr = int(rr + 1)
            if nr >= h:
                break

            if not _is_solid(base, nr, cc, bg, water):
                rr = int(nr)
                out[rr, cc] = int(water)
                continue

            shelf = _find_shelf_at(shelves, nr, cc)
            if shelf is None:
                break

            stats["impacts"] += 1
            for direction in (-1, 1):
                x = int(cc)
                while True:
                    nx = int(x + direction)
                    if not (0 <= nx < w):
                        break
                    if _is_solid(base, rr, nx, bg, water):
                        break

                    if int(out[rr, nx]) != int(water):
                        out[rr, nx] = int(water)
                        stats["spread_cells"] += 1

                    if nx < int(shelf["c0"]) or nx > int(shelf["c1"]):
                        q.append((int(rr), int(nx)))
                        stats["overflow_branches"] += 1
                        break

                    if _is_through_hole(base, shelf, nx, bg):
                        q.append((int(rr), int(nx)))
                        stats["hole_branches"] += 1
                        break

                    x = int(nx)
            break

    return out, stats


def _build_output_and_stats(input_grid: np.ndarray) -> tuple[np.ndarray, dict[str, int], dict]:
    g, bg, water, shelves, source_starts, motif_abs_masks = _extract_objects(input_grid)
    base, carved_shelves = _carve_shelves(g, bg, shelves, motif_abs_masks)
    out, stats = _simulate_water(base, bg, water, carved_shelves, source_starts)
    meta = {
        "bg": int(bg),
        "water": int(water),
        "shelves": carved_shelves,
        "source_starts": source_starts,
        "motif_abs_masks": motif_abs_masks,
        "base": base,
    }
    return out, stats, meta


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Carve same-color hole stencils into solid shelves, then simulate leaking waterfall branches.

    Input structure:
    - The background is the most common color.
    - Several solid horizontal shelf rectangles appear at different heights, each with its own color.
    - Elsewhere in empty space, each shelf color has a matching small stencil motif.
    - One water color forms one or more source blobs touching the top row.

    Rule:
    - Pair each motif with the shelf of the same color and center-carve the motif shape into that shelf, turning carved cells into background; then erase the motif itself.
    - Water starts from each top source and falls vertically through background cells.
    - When falling water hits a solid shelf cell, it spreads along the shelf's top surface to the left and right.
    - On each side, the spread drops as soon as it reaches a carved through-hole column; if no hole comes first, it goes one cell past that shelf end and then drops.
    - Apply the same fall/spread/leak process to every new branch until all water exits the grid bottom.

    Key insight:
    - The leak points are not drawn on the shelves directly; they are remotely specified by the same-color motifs, so carving must happen before the water simulation.
    """
    out, _stats, _meta = _build_output_and_stats(input_grid)
    return out


def _random_vertical_slit_mask(height: int, width: int, *, min_cols: int = 1, max_cols: int | None = None) -> np.ndarray:
    max_cols = width if max_cols is None else min(int(max_cols), int(width))
    min_cols = max(1, min(int(min_cols), int(width)))
    count = random.randint(min_cols, max_cols)
    cols = sorted(random.sample(range(width), count))
    mask = np.zeros((height, width), dtype=bool)
    mask[:, cols] = True
    return mask


def _reserve(mask: np.ndarray, r0: int, c0: int, h: int, w: int, value: bool = True) -> None:
    rr0 = max(0, int(r0) - 1)
    cc0 = max(0, int(c0) - 1)
    rr1 = min(mask.shape[0], int(r0) + int(h) + 1)
    cc1 = min(mask.shape[1], int(c0) + int(w) + 1)
    mask[rr0:rr1, cc0:cc1] = bool(value)


def _place_empty_bbox(
    occupied: np.ndarray,
    box_h: int,
    box_w: int,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    tries: int = 120,
) -> tuple[int, int] | None:
    if row_max - row_min + 1 < box_h or col_max - col_min + 1 < box_w:
        return None
    for _ in range(tries):
        r0 = random.randint(row_min, row_max - box_h + 1)
        c0 = random.randint(col_min, col_max - box_w + 1)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(occupied.shape[0], r0 + box_h + 1)
        cc1 = min(occupied.shape[1], c0 + box_w + 1)
        if not occupied[rr0:rr1, cc0:cc1].any():
            return (int(r0), int(c0))
    return None


def _draw_source_blob(grid: np.ndarray, water: int, col: int, max_depth: int) -> None:
    depth = random.randint(1, max(1, max_depth))
    grid[:depth, int(col)] = int(water)
    if depth >= 2 and random.random() < 0.35:
        side = random.choice([-1, 1])
        if 0 <= int(col + side) < grid.shape[1]:
            grid[0, int(col + side)] = int(water)


def _sample_shelves(h: int, w: int, shelf_colors: list[int]) -> list[dict] | None:
    n = len(shelf_colors)
    heights = [random.randint(2, 4) for _ in range(n)]
    top = random.randint(4, 6)
    shelves: list[dict] = []
    current_top = int(top)

    for idx, color in enumerate(shelf_colors):
        shelf_h = int(heights[idx])
        shelf_w = random.randint(8, min(14, w - 4))
        if current_top + shelf_h >= h - 2:
            return None
        c0 = random.randint(2, w - shelf_w - 2)
        shelf = {
            "color": int(color),
            "r0": int(current_top),
            "c0": int(c0),
            "r1": int(current_top + shelf_h - 1),
            "c1": int(c0 + shelf_w - 1),
            "h": int(shelf_h),
            "w": int(shelf_w),
        }
        shelves.append(shelf)
        gap = random.randint(3, 5)
        current_top = int(shelf["r1"] + gap + 1)

    if shelves[-1]["r1"] >= h - 2:
        return None
    return shelves


def _attach_motifs(shelves: list[dict]) -> bool:
    for idx, shelf in enumerate(shelves):
        shelf_h = int(shelf["h"])
        if idx == 0:
            motif_w = random.randint(3, min(5, int(shelf["w"]) - 2))
            motif = _random_vertical_slit_mask(shelf_h, motif_w, min_cols=2, max_cols=max(2, motif_w - 1))
            cols = np.flatnonzero(motif[0]).tolist()
            if len(cols) < 2:
                return False
        else:
            motif_w = random.randint(2, min(5, int(shelf["w"]) - 2))
            motif = _random_vertical_slit_mask(shelf_h, motif_w, min_cols=1, max_cols=max(1, motif_w - 1))
        shelf["motif_mask"] = motif

        off_c = (int(shelf["w"]) - motif_w) // 2
        hole_cols = [int(shelf["c0"] + off_c + c) for c in np.flatnonzero(motif[0])]
        shelf["planned_hole_cols"] = hole_cols
    return True


def _pick_source_columns(top_shelf: dict) -> list[int] | None:
    left = int(top_shelf["c0"])
    right = int(top_shelf["c1"])
    holes = sorted(int(c) for c in top_shelf["planned_hole_cols"])
    solid_cols = [c for c in range(left, right + 1) if c not in holes]
    if not solid_cols:
        return None

    between = [c for c in solid_cols if any(h < c for h in holes) and any(h > c for h in holes)]
    central = between if between else solid_cols
    branch_col = random.choice(central)

    left_over = [c for c in solid_cols if c < holes[0]]
    right_over = [c for c in solid_cols if c > holes[-1]]
    overflow_pool = left_over + right_over
    if not overflow_pool:
        return None
    overflow_col = random.choice(overflow_pool)

    cols = [int(branch_col), int(overflow_col)]
    remaining = [c for c in solid_cols if c not in cols]
    if remaining and random.random() < 0.45:
        cols.append(int(random.choice(remaining)))
    cols = sorted(set(cols))
    return cols


def _paint_scene(
    h: int,
    w: int,
    bg: int,
    water: int,
    shelves: list[dict],
    source_cols: list[int],
) -> np.ndarray | None:
    grid = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for shelf in shelves:
        grid[int(shelf["r0"]) : int(shelf["r1"]) + 1, int(shelf["c0"]) : int(shelf["c1"]) + 1] = int(shelf["color"])
        _reserve(occupied, int(shelf["r0"]), int(shelf["c0"]), int(shelf["h"]), int(shelf["w"]))

    top_limit = max(1, min(int(shelves[0]["r0"]) - 1, 4))
    for col in source_cols:
        _draw_source_blob(grid, water, int(col), top_limit)
        _reserve(occupied, 0, int(col), top_limit, 1, value=True)

    for shelf in shelves:
        motif = shelf["motif_mask"]
        mh, mw = motif.shape
        placed = _place_empty_bbox(
            occupied,
            mh,
            mw,
            1,
            h - mh - 2,
            1,
            w - mw - 2,
        )
        if placed is None:
            return None
        mr, mc = placed
        grid[mr : mr + mh, mc : mc + mw][motif] = int(shelf["color"])
        _reserve(occupied, mr, mc, mh, mw)

    return grid


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(18, 30)
    bg = random.choice(PALETTE)
    palette = [c for c in PALETTE if c != int(bg)]
    random.shuffle(palette)

    n_shelves = random.randint(2, 4)
    if len(palette) < n_shelves + 1:
        return None

    water = int(palette[0])
    shelf_colors = [int(c) for c in palette[1 : n_shelves + 1]]
    shelves = _sample_shelves(h, w, shelf_colors)
    if shelves is None:
        return None
    if not _attach_motifs(shelves):
        return None

    source_cols = _pick_source_columns(shelves[0])
    if source_cols is None:
        return None

    grid = _paint_scene(h, w, bg, water, shelves, source_cols)
    if grid is None:
        return None

    try:
        out, stats, meta = _build_output_and_stats(grid)
    except Exception:
        return None

    if not np.array_equal(main(grid), out):
        return None
    if np.array_equal(grid, out):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    if len(meta["shelves"]) != n_shelves:
        return None
    if any(len(shelf["through_cols"]) == 0 for shelf in meta["shelves"]):
        return None
    if stats["hole_branches"] < 1 or stats["overflow_branches"] < 1:
        return None
    if stats["impacts"] < 2:
        return None
    if int((out == meta["water"]).sum()) <= int((grid == meta["water"]).sum()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid leaking-shelf ARC input")
