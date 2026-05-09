# Combo variant of 8b7bacbf (index 9)
# combo_spec: D5/1+D6/0
# Name: 多色边界腔体+框架环重涂色

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
Design analysis for combo D5/1 + D6/0

1. Base task recap:
   - Each unique border seed belongs to one thin 8-connected line network.
   - The network reaches one or more closed frame loops around background cavities.
   - The original output flood-fills the cavity interior with the seed color.

2. Fused behavior:
   - D5/1: relevant loops are no longer monochrome. Each usable frame ring must visibly use
     at least two colors on its boundary.
   - D6/0: the transformation target moves from the cavity interior to the ring itself.
     The interior stays background; every boundary pixel of a valid loop is recolored to the seed color.
   - Combined rule: detect multi-color enclosed cavities that are directly connected to the seed's
     traced path, then repaint the whole ring to the seed color and leave the cavity empty.

3. Generator visibility obligations:
   - Every accepted sample contains at least one attached multi-color ring per seed.
   - Valid rings use two side colors, so "multi-color boundary" is visually obvious in 2-4 examples.
   - Seed colors are never reused on ring pixels, so the recolor effect is always visible.
   - Optional disconnected distractor rings can also be multi-color, preventing "multi-color alone"
     from being mistaken for the activation rule; connectivity to the seeded path remains necessary.
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


def _same_color_component8_from(
    g: np.ndarray, starts: list[tuple[int, int]], color: int
) -> set[tuple[int, int]]:
    h, w = g.shape
    seen = {
        (int(r), int(c))
        for r, c in starts
        if 0 <= r < h and 0 <= c < w and int(g[r, c]) == int(color)
    }
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) == int(color):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) != int(bg):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _line_info(
    g: np.ndarray, seed: tuple[int, int], bg: int
) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = g.shape
    sr, sc = seed
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != int(bg):
            neigh_cols.append(int(g[nr, nc]))
    assert neigh_cols, "seed pixel must have at least one non-background 4-neighbor"
    line_col = Counter(neigh_cols).most_common(1)[0][0]
    starts = [
        (nr, nc)
        for nr, nc in neighbors4(sr, sc)
        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == int(line_col)
    ]
    line_pixels = _same_color_component8_from(g, starts, int(line_col))
    traced = _nonbg_component8_from(g, seed, bg)
    return int(line_col), line_pixels, traced


def _extract_cavities(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = g.shape
    cavities: list[dict[str, object]] = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        if _touches_border(region, h, w):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != int(bg):
                    boundary.add((nr, nc))
        if not boundary:
            continue
        cavities.append(
            {
                "region": region,
                "boundary": boundary,
                "bcols": {int(g[r, c]) for r, c in boundary},
            }
        )
    return cavities


def _boundary_has_line_contact(
    boundary: set[tuple[int, int]], line_pixels: set[tuple[int, int]]
) -> bool:
    if boundary & line_pixels:
        return True
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if (nr, nc) in line_pixels:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor directly path-connected multi-color frame rings to the seed color.

    Input structure:
    - The dominant color is background.
    - One or two unique border seed pixels touch their own thin 8-connected line network.
    - The network reaches closed frame loops that enclose background cavities.
    - Relevant frame rings are multi-colored rather than monochrome.

    Rule:
    - For each seed, recover the connected line network and locate enclosed background cavities.
    - Keep only cavities whose boundary uses at least two colors, belongs to the seed's traced
      non-background component, and directly touches the seed's line.
    - Recolor every boundary pixel of such a ring to the seed color.
    - Do not fill the cavity interior.

    Key insight:
    - The visible action moves from "paint the empty hole" to "overwrite the surrounding loop",
      while the loop is no longer identified by a single border color.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    cavities = _extract_cavities(g, bg)

    for seed in _seed_pixels(g, bg):
        seed_col = int(g[seed])
        _line_col, line_pixels, traced = _line_info(g, seed, bg)
        for cav in cavities:
            boundary = cav["boundary"]
            bcols = cav["bcols"]
            assert isinstance(boundary, set)
            assert isinstance(bcols, set)
            if len(bcols) < 2:
                continue
            if not (boundary & traced):
                continue
            if not _boundary_has_line_contact(boundary, line_pixels):
                continue
            for r, c in boundary:
                out[r, c] = seed_col

    return out


GRID_H_RANGE = (10, 24)
GRID_W_RANGE = (10, 24)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 1, 2]
MIN_LINE_LEN = 5
LINE_LEN_RANGE = (9, 28)
FRAME_COLORS_MAX = 5
TOTAL_FRAMES_MAX = 7
CAVITY_SIZE_CHOICES = [2, 2, 2, 3]
DISTRACTOR_CAVITY_CHOICES = [2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 70
DISTRACTOR_FRAME_TRIES = 60
DISTRACTOR_FRAME_MAX = 2

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.8
BIAS_FLIP_PROB = 0.3

GEN_RETRIES = 500

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_parts(
    r0: int, c0: int, ch: int, cw: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = ch + 2, cw + 2
    horiz = []
    vert = []
    for dc in range(1, width - 1):
        horiz.append((r0, c0 + dc))
        horiz.append((r0 + height - 1, c0 + dc))
    for dr in range(1, height - 1):
        vert.append((r0 + dr, c0))
        vert.append((r0 + dr, c0 + width - 1))
    ring = horiz + vert
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity, horiz, vert


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
    touches_line = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            value = int(canvas[nr, nc])
            if value == -1:
                continue
            if value == int(line_col):
                touches_line = True
                continue
            return False
    return touches_line


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start
    path_set = {start}
    while len(path) < target_len:
        dirs: list[tuple[int, int]] = []
        for ddr, ddc in DIAG_DIRS:
            aligned = ddr == dr_bias or ddc == dc_bias
            weight = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * weight)
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            weight = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * weight)
        rng.shuffle(dirs)
        placed = False
        for ddr, ddc in dirs:
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


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [
        (dr, dc)
        for dr in range(-1, fh + 1)
        for dc in range(-1, fw + 1)
        if not (0 <= dr < fh and 0 <= dc < fw)
    ]


def _paint_ring(
    canvas: np.ndarray,
    horiz: list[tuple[int, int]],
    vert: list[tuple[int, int]],
    color_h: int,
    color_v: int,
) -> None:
    for rr, cc in horiz:
        canvas[rr, cc] = int(color_h)
    for rr, cc in vert:
        canvas[rr, cc] = int(color_v)


def _try_attach_multicolor_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    color_pair: tuple[int, int],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    for _ in range(FRAME_ATTACH_TRIES):
        ch = rng.choice(CAVITY_SIZE_CHOICES)
        cw = rng.choice(CAVITY_SIZE_CHOICES)
        lp = rng.choice(line_path)
        fh, fw = ch + 2, cw + 2
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                continue
            ring, cavity, horiz, vert = _frame_parts(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, ring, cavity, line_col):
                continue
            _paint_ring(canvas, horiz, vert, color_pair[0], color_pair[1])
            return ring, cavity
    return None


def _try_place_distractor_frame(
    canvas: np.ndarray,
    color_pair: tuple[int, int],
    rng: random.Random,
) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_FRAME_TRIES):
        ch = rng.choice(DISTRACTOR_CAVITY_CHOICES)
        cw = rng.choice(DISTRACTOR_CAVITY_CHOICES)
        fh, fw = ch + 2, cw + 2
        if h - fh - 2 < 1 or w - fw - 2 < 1:
            return False
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        ring, cavity, horiz, vert = _frame_parts(r0, c0, ch, cw)
        halo_ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and int(canvas[rr, cc]) != -1:
                    halo_ok = False
                    break
            if not halo_ok:
                break
        if not halo_ok:
            continue
        _paint_ring(canvas, horiz, vert, color_pair[0], color_pair[1])
        return True
    return False


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


def _distribute_budget(num_seeds: int, total_frames: int, rng: random.Random) -> list[int]:
    order = list(range(num_seeds))
    rng.shuffle(order)
    per_seed = [0] * num_seeds
    remain = total_frames
    for idx in order:
        if remain <= 0:
            break
        per_seed[idx] += 1
        remain -= 1
    while remain > 0:
        per_seed[rng.choice(order)] += 1
        remain -= 1
    return per_seed


def _profile(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    out = main(g)
    bg = int(get_bg(g))
    cavities = _extract_cavities(g, bg)
    relevant = 0
    recolored_boundary_cells = 0
    changed_interior_cells = 0
    for seed in _seed_pixels(g, bg):
        seed_col = int(g[seed])
        _line_col, line_pixels, traced = _line_info(g, seed, bg)
        for cav in cavities:
            boundary = cav["boundary"]
            bcols = cav["bcols"]
            region = cav["region"]
            assert isinstance(boundary, set)
            assert isinstance(bcols, set)
            assert isinstance(region, set)
            if len(bcols) < 2:
                continue
            if not (boundary & traced):
                continue
            if not _boundary_has_line_contact(boundary, line_pixels):
                continue
            relevant += 1
            recolored_boundary_cells += sum(int(out[r, c] == seed_col) for r, c in boundary)
            changed_interior_cells += sum(int(out[r, c] != g[r, c]) for r, c in region)
    return {
        "relevant": int(relevant),
        "changed_cells": int(np.count_nonzero(out != g)),
        "recolored_boundary_cells": int(recolored_boundary_cells),
        "changed_interior_cells": int(changed_interior_cells),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if len(palette) < 2 * num_seeds + 2:
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    frame_color_pool = [c for c in palette if c not in seed_colors]
    if len(frame_color_pool) < 2:
        return None
    if len(frame_color_pool) > FRAME_COLORS_MAX:
        rng.shuffle(frame_color_pool)
        frame_color_pool = frame_color_pool[:FRAME_COLORS_MAX]

    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]] = []
    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w, rng)
        if int(canvas[seed_rc]) != -1 or int(canvas[first_rc]) != -1:
            return None
        canvas[seed_rc] = int(seed_colors[i])
        canvas[first_rc] = int(line_colors[i])
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, first_rc, int(line_colors[i]), target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((int(seed_colors[i]), seed_rc, int(line_colors[i]), path))

    total_frames = rng.randint(num_seeds, TOTAL_FRAMES_MAX)
    per_seed = _distribute_budget(num_seeds, total_frames, rng)

    for i, (_seed_col, _seed_rc, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            pair = tuple(rng.sample(frame_color_pool, 2))
            if _try_attach_multicolor_frame(canvas, path, int(line_col), pair, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    for _ in range(rng.randint(0, DISTRACTOR_FRAME_MAX)):
        pair = tuple(rng.sample(frame_color_pool, 2))
        _try_place_distractor_frame(canvas, pair, rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != int(bg):
        return None

    for seed_col, _seed_rc, _line_col, _path in lines_info:
        if int((grid == int(seed_col)).sum()) != 1:
            return None

    profile = _profile(grid)
    if profile["relevant"] < num_seeds:
        return None
    if profile["changed_cells"] <= 0:
        return None
    if profile["recolored_boundary_cells"] <= 0:
        return None
    if profile["changed_interior_cells"] != 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
