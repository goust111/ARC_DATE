from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np
from scipy.ndimage import label


GREY = 5
BLACK = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8)
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
_GEN_BRANCH_PATTERN = (5, 7, 7, 5, 7, 5)
_GEN_CALL_COUNT = 0


def _components(mask: np.ndarray) -> list[np.ndarray]:
    lab, n = label(mask.astype(bool), structure=STRUCT4)
    return [np.argwhere(lab == idx) for idx in range(1, int(n) + 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _factor_pair(n: int) -> tuple[int, int]:
    rows = 1
    for cand in range(1, int(math.isqrt(n)) + 1):
        if n % cand == 0:
            rows = cand
    return rows, n // rows


def _region_records(mask: np.ndarray) -> list[dict]:
    records: list[dict] = []
    for coords in _components(mask):
        b = _bbox(coords)
        records.append(
            {
                "coords": coords,
                "area": int(len(coords)),
                "bbox": b,
                "top_left": (b[0], b[1]),
                "coord_set": {tuple(map(int, p)) for p in coords},
            }
        )
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the smallest wall-bounded region as a miniature blueprint that assigns fill colors to all other regions.

    Input structure:
    - The grid has black background pixels and one connected network of one-cell-thick grey wall pixels.
    - Treating grey as blocked, all non-grey pixels split into 5 to 7 four-connected regions.
    - The smallest region is a blueprint region whose upper-left corner contains a compact color matrix; other regions are larger black areas with scattered colored distractor pixels.

    Rule:
    - Label every non-grey four-connected region.
    - Select the blueprint region as the one with minimum area; ties break by the region bbox's upper-left coordinate.
    - Sort the remaining regions by their bbox upper-left coordinate, then read the blueprint's compact matrix in row-major order with dimensions determined by the number of non-blueprint regions using the most square factor pair.
    - Recolor each sorted non-blueprint region uniformly to its assigned blueprint color, clear the blueprint region to black, and keep grey wall pixels unchanged.

    Key insight:
    - The smallest partition is not a target to fill but a miniature command map: its color cells encode, in sorted-region order, which solid color each larger region should become.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions = _region_records(g != GREY)
    assert 5 <= len(regions) <= 7, "expected 5 to 7 non-grey regions"

    blueprint = min(regions, key=lambda rec: (rec["area"], rec["top_left"]))
    others = [rec for rec in regions if rec is not blueprint]
    others.sort(key=lambda rec: rec["top_left"])

    rows, cols = _factor_pair(len(others))
    r0, c0, r1, c1 = blueprint["bbox"]
    assert rows <= r1 - r0 + 1 and cols <= c1 - c0 + 1, "blueprint bbox too small"

    colors: list[int] = []
    for dr in range(rows):
        for dc in range(cols):
            rr, cc = r0 + dr, c0 + dc
            assert (rr, cc) in blueprint["coord_set"], "blueprint matrix must lie inside the blueprint region"
            color = int(g[rr, cc])
            assert color in PALETTE, "blueprint cells must use visible non-grey colors"
            colors.append(color)
    assert len(colors) == len(others)

    out = g.copy()
    bp = blueprint["coords"]
    out[bp[:, 0], bp[:, 1]] = BLACK
    for color, rec in zip(colors, others):
        coords = rec["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)
    out[g == GREY] = GREY
    return out


def _sample_locked_composition(total: int, mins: list[int], locked: set[int]) -> list[int] | None:
    base = sum(mins)
    if base > total:
        return None
    vals = mins[:]
    free = [idx for idx in range(len(vals)) if idx not in locked]
    extra = total - base
    for _ in range(extra):
        vals[random.choice(free)] += 1
    return vals


def _pick_targets(n: int) -> list[int]:
    while True:
        vals = [random.choice(PALETTE) for _ in range(n)]
        if len(set(vals)) >= 2:
            return vals


def _region_cells(side: str, band_idx: int, left_w: int, right_w: int, left_heights: list[int], right_heights: list[int]) -> np.ndarray:
    if side == "left":
        heights = left_heights
        c0, c1 = 0, left_w - 1
    else:
        heights = right_heights
        c0, c1 = left_w + 1, left_w + right_w
    r0 = sum(heights[:band_idx])
    r1 = r0 + heights[band_idx] - 1
    rows = np.arange(r0, r1 + 1)
    cols = np.arange(c0, c1 + 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.column_stack([rr.ravel(), cc.ravel()])


def _layout_records(left_w: int, right_w: int, left_heights: list[int], right_heights: list[int]) -> list[dict]:
    recs: list[dict] = []
    for idx in range(len(left_heights)):
        coords = _region_cells("left", idx, left_w, right_w, left_heights, right_heights)
        b = _bbox(coords)
        recs.append({"side": "left", "band": idx, "coords": coords, "bbox": b, "area": len(coords), "top_left": (b[0], b[1])})
    for idx in range(len(right_heights)):
        coords = _region_cells("right", idx, left_w, right_w, left_heights, right_heights)
        b = _bbox(coords)
        recs.append({"side": "right", "band": idx, "coords": coords, "bbox": b, "area": len(coords), "top_left": (b[0], b[1])})
    return recs


def _place_noise(inp: np.ndarray, coords: np.ndarray, target_color: int) -> None:
    cells = [tuple(map(int, p)) for p in coords]
    count = random.randint(1, min(3, len(cells)))
    chosen = random.sample(cells, count)
    different = random.choice([c for c in PALETTE if c != target_color])
    inp[chosen[0]] = different
    for cell in chosen[1:]:
        inp[cell] = random.choice(PALETTE)


def _draw_walls(h: int, left_w: int, right_w: int, left_heights: list[int], right_heights: list[int]) -> np.ndarray:
    w = left_w + 1 + right_w
    grid = np.zeros((h, w), dtype=int)
    grid[:, left_w] = GREY
    r = 0
    for seg in left_heights[:-1]:
        r += seg
        grid[r, : left_w + 1] = GREY
    r = 0
    for seg in right_heights[:-1]:
        r += seg
        grid[r, left_w:] = GREY
    return grid


def _check_semantics(inp: np.ndarray, blueprint_top_left: tuple[int, int], blueprint_area: int, matrix_shape: tuple[int, int], target_colors: list[int]) -> bool:
    regions = _region_records(inp != GREY)
    if len(regions) not in (5, 7):
        return False
    wall_components = _components(inp == GREY)
    if len(wall_components) != 1:
        return False

    blueprint = min(regions, key=lambda rec: (rec["area"], rec["top_left"]))
    if blueprint["top_left"] != blueprint_top_left:
        return False
    if blueprint["area"] != blueprint_area:
        return False
    if not all(rec["area"] > blueprint_area for rec in regions if rec is not blueprint):
        return False

    r0, c0, r1, c1 = blueprint["bbox"]
    rows, cols = matrix_shape
    if (r1 - r0 + 1) > 4 or (c1 - c0 + 1) > 4:
        return False
    if rows > (r1 - r0 + 1) or cols > (c1 - c0 + 1):
        return False

    others = [rec for rec in regions if rec is not blueprint]
    others.sort(key=lambda rec: rec["top_left"])
    read = [int(inp[r0 + dr, c0 + dc]) for dr in range(rows) for dc in range(cols)]
    if read != target_colors:
        return False
    if any(rec["area"] < 6 for rec in others):
        return False
    return True


def _gen_attempt(total_regions: int | None = None) -> np.ndarray | None:
    if total_regions is None:
        total_regions = random.choice((5, 7))
    non_blueprint = total_regions - 1
    matrix_shape = _factor_pair(non_blueprint)
    mr, mc = matrix_shape

    split_options = [(2, 3), (3, 2)] if total_regions == 5 else [(3, 4), (4, 3)]
    left_count, right_count = random.choice(split_options)
    blueprint_side = random.choice(("left", "right"))
    side_count = left_count if blueprint_side == "left" else right_count
    bp_band = random.randrange(side_count)

    dim_choices = [(bh, bw) for bh in range(mr, 5) for bw in range(mc, 5)]
    random.shuffle(dim_choices)
    for bp_h, bp_w in dim_choices:
        bp_area = bp_h * bp_w
        side_w = bp_w
        other_w = random.randint(max(5, mc + 2), 10)
        min_same_h = max(2, bp_area // side_w + 1)
        min_other_h = max(2, bp_area // other_w + 1)

        same_count = side_count
        other_count = right_count if blueprint_side == "left" else left_count
        same_mins = [min_same_h] * same_count
        same_mins[bp_band] = bp_h
        other_mins = [min_other_h] * other_count

        min_h = max(sum(same_mins), sum(other_mins))
        max_h = 18
        if min_h > max_h:
            continue
        h = random.randint(min_h, max_h)
        same_locked = {bp_band}
        same_heights = _sample_locked_composition(h, same_mins, same_locked)
        other_heights = _sample_locked_composition(h, other_mins, set())
        if same_heights is None or other_heights is None:
            continue

        if blueprint_side == "left":
            left_w, right_w = side_w, other_w
            left_heights, right_heights = same_heights, other_heights
        else:
            left_w, right_w = other_w, side_w
            left_heights, right_heights = other_heights, same_heights

        recs = _layout_records(left_w, right_w, left_heights, right_heights)
        wanted = [rec for rec in recs if rec["side"] == blueprint_side and rec["band"] == bp_band]
        assert len(wanted) == 1
        blueprint_rec = wanted[0]
        if blueprint_rec["area"] != bp_area:
            continue
        if any(rec["area"] <= bp_area for rec in recs if rec is not blueprint_rec):
            continue

        inp = _draw_walls(h, left_w, right_w, left_heights, right_heights)
        others = [rec for rec in recs if rec is not blueprint_rec]
        others.sort(key=lambda rec: rec["top_left"])
        target_colors = _pick_targets(len(others))

        r0, c0, _r1, _c1 = blueprint_rec["bbox"]
        idx = 0
        for dr in range(mr):
            for dc in range(mc):
                inp[r0 + dr, c0 + dc] = target_colors[idx]
                idx += 1

        for color, rec in zip(target_colors, others):
            _place_noise(inp, rec["coords"], color)

        if not _check_semantics(inp, blueprint_rec["top_left"], bp_area, matrix_shape, target_colors):
            continue

        try:
            out = main(inp)
        except Exception:
            continue

        if out.shape != inp.shape:
            continue
        if len(set(map(int, out.flatten())) - {BLACK, GREY}) < 2:
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    preferred = _GEN_BRANCH_PATTERN[_GEN_CALL_COUNT % len(_GEN_BRANCH_PATTERN)]
    for _ in range(250):
        inp = _gen_attempt(preferred)
        if inp is not None:
            _GEN_CALL_COUNT += 1
            return inp
    for _ in range(150):
        inp = _gen_attempt()
        if inp is not None:
            _GEN_CALL_COUNT += 1
            return inp
    raise RuntimeError("generate_input: failed to construct a valid blueprint-flood example")
