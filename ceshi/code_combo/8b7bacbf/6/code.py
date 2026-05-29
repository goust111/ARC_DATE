# Combo variant of 8b7bacbf (index 6)
# combo_spec: D3/3+D9/1
# Name: 填充为框架色+边界色匹配线路色

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT_DIR = Path("/home/lixinyu/snap/new")
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


"""
Design analysis
---------------
1. Base task recap:
   - Unique border seeds identify 8-connected line networks.
   - Monochrome closed frame rings enclose background cavities.
   - Original output fills reachable cavities with the seed color.

2. Fused behavior for D3/3 + D9/1:
   - D3/3: the cavity is no longer recolored by the seed; it is recolored by its own frame color.
   - D9/1: only cavities whose frame color matches the seed's line color are valid.
   - Combined effect: only line-color frames fill, and they fill with that same line/frame color.

3. Generator visibility constraints:
   - Every accepted sample contains at least one matching attached frame and at least one mismatching
     attached frame on seeded networks, so both sub-rules are visible at once.
   - Seed colors are always distinct from line colors, preventing regression to the original task.
   - Matching and mismatching frames share the same geometric role, so the human-visible difference is
     color compatibility, not hidden metadata.
"""


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    connectivity = 8 if conn8 else 4
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=connectivity)]


def _touches_border(pts: set[tuple[int, int]] | list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        value = int(v)
        if value == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == value)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) != bg:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _line_color_for_seed(g: np.ndarray, seed: tuple[int, int], bg: int) -> int:
    h, w = g.shape
    sr, sc = seed
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w:
            value = int(g[nr, nc])
            if value != bg:
                neigh_cols.append(value)
    assert neigh_cols, "seed pixel must have at least one non-background 4-neighbor"
    return int(Counter(neigh_cols).most_common(1)[0][0])


def _cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    out = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        if _touches_border(region, h, w):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        if not boundary:
            continue
        out.append((region, boundary, {int(g[p]) for p in boundary}))
    return out


def _solve(
    grid: np.ndarray,
    *,
    require_match: bool,
    fill_mode: str,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    cavities = _cavities(g, bg)

    for seed in _seed_pixels(g, bg):
        seed_col = int(g[seed])
        line_col = _line_color_for_seed(g, seed, bg)
        traced = _nonbg_component8_from(g, seed, bg)
        for region, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            frame_col = next(iter(bcols))
            if require_match and frame_col != line_col:
                continue
            if not (boundary & traced):
                continue
            fill_col = seed_col if fill_mode == "seed" else frame_col
            for r, c in region:
                if int(out[r, c]) == bg:
                    out[r, c] = fill_col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only line-color-matched cavities, using the cavity's frame color.

    Input structure:
    - The dominant color is background.
    - One or two unique border seeds touch their own thin 8-connected line network.
    - The network reaches single-color closed frame loops enclosing background cavities.

    Rule:
    - For each seed, trace the entire non-background 8-connected component reachable from it.
    - Consider only enclosed background cavities whose boundary is monochrome.
    - A cavity is valid only when that boundary color equals the seed's line color.
    - Fill the valid cavity with the boundary/frame color itself, not the seed color.

    Key insight:
    - Connectivity still comes from the seed's network, but color selection is inverted twice:
      the fill color comes from the frame, and the frame must color-match the line to activate.
    """
    return _solve(grid, require_match=True, fill_mode="frame")


def _d3_only(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, require_match=False, fill_mode="frame")


def _d9_only(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, require_match=True, fill_mode="seed")


def _profile(grid: np.ndarray) -> dict[str, int | list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    cavities = _cavities(g, bg)
    by_seed: list[dict[str, int]] = []
    matching_total = 0
    mismatching_total = 0

    for seed in _seed_pixels(g, bg):
        line_col = _line_color_for_seed(g, seed, bg)
        traced = _nonbg_component8_from(g, seed, bg)
        matching = 0
        mismatching = 0
        for _region, boundary, bcols in cavities:
            if len(bcols) != 1 or not (boundary & traced):
                continue
            frame_col = next(iter(bcols))
            if frame_col == line_col:
                matching += 1
            else:
                mismatching += 1
        by_seed.append({"matching": matching, "mismatching": mismatching})
        matching_total += matching
        mismatching_total += mismatching

    new_out = main(g)
    d3_out = _d3_only(g)
    d9_out = _d9_only(g)
    return {
        "matching_total": matching_total,
        "mismatching_total": mismatching_total,
        "filled_cells": int(np.count_nonzero(new_out != g)),
        "d3_delta_cells": int(np.count_nonzero(new_out != d3_out)),
        "d9_delta_cells": int(np.count_nonzero(new_out != d9_out)),
        "by_seed": by_seed,
    }


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = ch + 2, cw + 2
    ring = []
    for dr in range(height):
        for dc in range(width):
            on_border = dr == 0 or dc == 0 or dr == height - 1 or dc == width - 1
            corner = dr in (0, height - 1) and dc in (0, width - 1)
            if on_border and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [
        (dr, dc)
        for dr in range(-1, fh + 1)
        for dc in range(-1, fw + 1)
        if not (0 <= dr < fh and 0 <= dc < fw)
    ]


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
) -> bool:
    h, w = canvas.shape
    for rr, cc in ring:
        if int(canvas[rr, cc]) != -1:
            return False
    for rr, cc in cavity:
        if int(canvas[rr, cc]) != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == int(line_col):
                return False

    ring_set = set(ring)
    line_touch = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            value = int(canvas[nr, nc])
            if value == -1:
                continue
            if value == int(line_col):
                line_touch = True
                continue
            return False
    return line_touch


DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
BG_COLORS = [0, 5]
SEED_COUNT_BAG = [1, 1, 1, 2, 2]
MIN_LINE_LEN = 6
LINE_LEN_RANGE = (10, 26)
FRAME_COLORS_MAX = 4
TOTAL_EXTRA_FRAMES = (0, 2)
CAVITY_SIZE_CHOICES = [2, 2, 2, 3, 3, 4]
FRAME_ATTACH_TRIES = 60
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3
DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
GEN_RETRIES = 500


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    path_set = {start}
    cur = start
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1

    while len(path) < target_len:
        weighted_dirs = []
        for ddr, ddc in DIAG_DIRS:
            aligned = ddr == dr_bias or ddc == dc_bias
            weight = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
            weighted_dirs.extend([(ddr, ddc)] * weight)
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            weight = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
            weighted_dirs.extend([(ddr, ddc)] * weight)
        rng.shuffle(weighted_dirs)

        placed = False
        for ddr, ddc in weighted_dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or int(grid[nr, nc]) != -1:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            cur = (nr, nc)
            path.append(cur)
            path_set.add(cur)
            grid[nr, nc] = int(line_col)
            placed = True
            break
        if not placed:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            break
    return path


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    for _ in range(max_tries):
        ch = rng.choice(CAVITY_SIZE_CHOICES)
        cw = rng.choice(CAVITY_SIZE_CHOICES)
        lp = rng.choice(line_path)
        fh, fw = ch + 2, cw + 2
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < 1 or c0 < 1 or r0 + fh > h - 1 or c0 + fw > w - 1:
                continue
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, ring, cavity, line_col):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = int(frame_col)
            return ring, cavity
    return None


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        return (0, 0), rng.choice([(0, 1), (1, 0)])
    if corner_kind == "TR":
        return (0, w - 1), rng.choice([(0, w - 2), (1, w - 1)])
    if corner_kind == "BL":
        return (h - 1, 0), rng.choice([(h - 1, 1), (h - 2, 0)])
    if corner_kind == "BR":
        return (h - 1, w - 1), rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    raise ValueError(corner_kind)


def _seed_components_are_disjoint(grid: np.ndarray, seeds: list[tuple[int, int]], bg: int) -> bool:
    traced_components = [_nonbg_component8_from(grid, seed, bg) for seed in seeds]
    for i, comp_i in enumerate(traced_components):
        for j, seed_j in enumerate(seeds):
            if i != j and seed_j in comp_i:
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_BAG)
    if len(palette) < num_seeds * 2 + 1:
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]
    mismatch_pool = [c for c in palette if c not in line_colors]
    if not mismatch_pool:
        return None
    extra_frame_cols = mismatch_pool[: min(FRAME_COLORS_MAX, len(mismatch_pool))]

    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    seeds: list[tuple[int, int]] = []
    line_infos: list[dict[str, object]] = []

    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners[i], h, w, rng)
        if int(canvas[seed_rc]) != -1 or int(canvas[first_rc]) != -1:
            return None
        seed_col = seed_colors[i]
        line_col = line_colors[i]
        canvas[seed_rc] = int(seed_col)
        canvas[first_rc] = int(line_col)
        max_len = min(LINE_LEN_RANGE[1], max(LINE_LEN_RANGE[0], (h * w) // 5))
        target_len = rng.randint(LINE_LEN_RANGE[0], max_len)
        path = _grow_line(canvas, first_rc, line_col, target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        seeds.append(seed_rc)
        line_infos.append(
            {
                "seed_rc": seed_rc,
                "seed_col": seed_col,
                "line_col": line_col,
                "path": path,
            }
        )

    matching_frames = 0
    mismatching_frames = 0
    for info in line_infos:
        path = info["path"]
        line_col = int(info["line_col"])
        if _try_attach_frame(canvas, path, line_col, line_col, rng) is None:
            return None
        matching_frames += 1

        mismatch_choices = [c for c in extra_frame_cols if c != line_col]
        if not mismatch_choices:
            return None
        mismatch_col = rng.choice(mismatch_choices)
        if _try_attach_frame(canvas, path, line_col, mismatch_col, rng) is None:
            return None
        mismatching_frames += 1

    for _ in range(rng.randint(*TOTAL_EXTRA_FRAMES)):
        info = rng.choice(line_infos)
        line_col = int(info["line_col"])
        frame_col = rng.choice([line_col] + [c for c in extra_frame_cols if c != line_col])
        placed = _try_attach_frame(canvas, info["path"], line_col, frame_col, rng)
        if placed is None:
            continue
        if frame_col == line_col:
            matching_frames += 1
        else:
            mismatching_frames += 1

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != int(bg):
        return None

    for seed_col in seed_colors:
        if int((grid == int(seed_col)).sum()) != 1:
            return None
    for seed_col, line_col in zip(seed_colors, line_colors):
        if int(seed_col) == int(line_col):
            return None

    if not _seed_components_are_disjoint(grid, seeds, bg):
        return None

    try:
        prof = _profile(grid)
    except Exception:
        return None

    if matching_frames == 0 or mismatching_frames == 0:
        return None
    if int(prof["matching_total"]) == 0 or int(prof["mismatching_total"]) == 0:
        return None
    if int(prof["filled_cells"]) == 0:
        return None
    if int(prof["d3_delta_cells"]) == 0:
        return None
    if int(prof["d9_delta_cells"]) == 0:
        return None
    by_seed = prof["by_seed"]
    if any(int(seed_info["matching"]) == 0 for seed_info in by_seed):
        return None
    if int(prof["mismatching_total"]) < num_seeds:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 8b7bacbf combo D3/3+D9/1")
