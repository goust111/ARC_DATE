# Combo variant of 8b7bacbf (index 8)
# combo_spec: D4/1+D13/0
# Name: 边缘色块种子+线路与种子同色
#
# Design analysis
# ----------------
# 1. Fused behavior:
#    - D4/1: the seed is no longer a unique border singleton; it is a 2-3 pixel straight block
#      lying on one border side.
#    - D13/0: the line uses exactly the same color as that seed block, so color counts can no
#      longer isolate the seed.
#    - Combined rule: identify a non-background component whose only border contact is a short
#      straight border block, trace the whole non-background drawing reachable from that block,
#      then fill every enclosed cavity touched by that drawing with the shared seed/line color.
#
# 2. Why the combo is observable:
#    - Seeds are generated on border sides away from the four corners, so a "corner seed" solver
#      for D13/0 fails.
#    - Seed colors continue along the interior line, so a "2-3 cells total" solver for D4/1 also
#      fails.
#    - The generator rejects samples where either single-variant reference solver matches the fused
#      output, preventing degeneration.
#
# 3. Visual cleanliness:
#    - Border blocks are short and isolated from corners.
#    - Lines remain thin self-avoiding 8-connected paths.
#    - Frames stay at least one cell away from the outer border and from unrelated colors.

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


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    connectivity = 8 if conn8 else 4
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=connectivity)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _is_border_cell(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _touches_border(pts: set[tuple[int, int]] | list[tuple[int, int]], h: int, w: int) -> bool:
    return any(_is_border_cell(r, c, h, w) for r, c in pts)


def _is_straight_border_block(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    if len(pts) not in (2, 3):
        return False
    if not all(_is_border_cell(r, c, h, w) for r, c in pts):
        return False
    rows = {r for r, _ in pts}
    cols = {c for _, c in pts}
    if len(rows) == 1:
        row = next(iter(rows))
        if row not in (0, h - 1):
            return False
        seq = sorted(c for _, c in pts)
        return seq[-1] - seq[0] + 1 == len(seq)
    if len(cols) == 1:
        col = next(iter(cols))
        if col not in (0, w - 1):
            return False
        seq = sorted(r for r, _ in pts)
        return seq[-1] - seq[0] + 1 == len(seq)
    return False


def _combo_seed_blocks(g: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seeds: list[list[tuple[int, int]]] = []
    for color in sorted(set(map(int, g.flatten())) - {int(bg)}):
        for comp in _components_mask(g == int(color), conn8=True):
            comp_set = set(comp)
            border_pts = sorted((r, c) for r, c in comp if _is_border_cell(r, c, h, w))
            if not _is_straight_border_block(border_pts, h, w):
                continue
            if any((r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)} for r, c in border_pts):
                continue
            if len(comp_set) == len(border_pts):
                continue
            has_interior_attachment = False
            for r, c in border_pts:
                for nr, nc in neighbors4(r, c):
                    if _in_bounds(h, w, nr, nc) and (nr, nc) in comp_set and not _is_border_cell(nr, nc, h, w):
                        has_interior_attachment = True
                        break
                if has_interior_attachment:
                    break
            if not has_interior_attachment:
                continue
            seeds.append(border_pts)
    seeds.sort()
    return seeds


def _seed_components_d4_only(g: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seeds: list[list[tuple[int, int]]] = []
    vals, cnts = np.unique(g, return_counts=True)
    for v, cnt in zip(vals, cnts):
        color = int(v)
        count = int(cnt)
        if color == bg or count not in (2, 3):
            continue
        comps = _components_mask(g == color, conn8=False)
        if len(comps) != 1:
            continue
        pts = sorted(comps[0])
        if len(pts) == count and _is_straight_border_block(pts, h, w):
            seeds.append(pts)
    seeds.sort()
    return seeds


def _corner_seed_pixels_d13_only(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    return [(r, c) for r, c in corners if int(g[r, c]) != int(bg)]


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if _in_bounds(h, w, nr, nc) and (nr, nc) not in seen and int(g[nr, nc]) != int(bg):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    out = []
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        if any(p in region for p in corners if int(g[p]) == int(bg)):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if _in_bounds(h, w, nr, nc) and (nr, nc) not in region and int(g[nr, nc]) != int(bg):
                    boundary.add((nr, nc))
        if not boundary:
            continue
        out.append((region, boundary, {int(g[p]) for p in boundary}))
    return out


def _fill_from_seed_blocks(
    g: np.ndarray,
    seed_blocks: list[list[tuple[int, int]]],
    *,
    line_color_mode: str,
    neighbor_mode: str,
) -> np.ndarray:
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()
    cavities = _cavities(g, bg)

    for seed_pts in seed_blocks:
        seed_col = int(g[seed_pts[0]])
        if line_color_mode == "same_as_seed":
            line_col = seed_col
        else:
            neigh_cols = []
            for sr, sc in seed_pts:
                for nr, nc in neighbors4(sr, sc):
                    if not _in_bounds(h, w, nr, nc):
                        continue
                    value = int(g[nr, nc])
                    if value != bg and value != seed_col:
                        neigh_cols.append(value)
            if not neigh_cols:
                continue
            line_col = int(Counter(neigh_cols).most_common(1)[0][0])

        traced = _nonbg_component8_from(g, seed_pts[0], bg)
        for region, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue
            ok = False
            for r, c in boundary_in_traced:
                neighs = neighbors8(r, c) if neighbor_mode == "8" else neighbors4(r, c)
                for nr, nc in neighs:
                    if _in_bounds(h, w, nr, nc) and int(g[nr, nc]) == line_col:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                continue
            for r, c in region:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill seed-owned cavities when the seed is a border block and the line shares its color.

    Input structure:
    - Background is the most common color.
    - Each real seed is a 2-3 cell straight block on one border side, away from corners.
    - The interior line connected to that block has the same color as the block.
    - Monochrome frame loops enclose background cavities.

    Rule:
    - Identify each border block by shape and position, not by global color counts.
    - Trace the full 8-connected non-background drawing reachable from that block.
    - Fill each enclosed background cavity whose boundary belongs to that drawing and is adjacent
      to the block/line color.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    seed_blocks = _combo_seed_blocks(g, bg)
    return _fill_from_seed_blocks(g, seed_blocks, line_color_mode="same_as_seed", neighbor_mode="8")


def _solve_d4_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    seed_blocks = _seed_components_d4_only(g, bg)
    return _fill_from_seed_blocks(g, seed_blocks, line_color_mode="neighbor_vote", neighbor_mode="8")


def _solve_d13_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    seed_blocks = [[seed] for seed in _corner_seed_pixels_d13_only(g, bg)]
    return _fill_from_seed_blocks(g, seed_blocks, line_color_mode="same_as_seed", neighbor_mode="8")


# --------------- generator ---------------

GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
SEED_BLOCK_LENGTHS = [2, 2, 3, 3]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 40
DISTRACTOR_FRAME_TRIES = 40

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_PROB = 0.6
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_SOLO_MAX = 2
DISTRACTOR_NETWORK_FRAMES = (1, 3)

GEN_RETRIES = 400

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = ch + 2, cw + 2
    ring = []
    for dr in range(height):
        for dc in range(width):
            on_border = dr == 0 or dr == height - 1 or dc == 0 or dc == width - 1
            corner = dr in (0, height - 1) and dc in (0, width - 1)
            if on_border and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
    h: int,
    w: int,
) -> bool:
    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return False

    ring_set = set(ring)
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            value = int(canvas[nr, nc])
            if value != -1 and value != int(line_col):
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == int(line_col):
                return True
    return False


def _grow_line(grid: np.ndarray, start: tuple[int, int], line_col: int, target_len: int, rng: random.Random) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start
    path_set = {start}
    while len(path) < target_len:
        dirs = []
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
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            if _is_border_cell(nr, nc, h, w) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            cur = (nr, nc)
            path.append(cur)
            path_set.add(cur)
            grid[nr, nc] = line_col
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
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    for _ in range(max_tries):
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        lp = rng.choice(line_path)
        fh, fw = ch + 2, cw + 2
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                continue
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_distractor_frame(
    canvas: np.ndarray,
    frame_col: int,
    rng: random.Random,
    max_tries: int = DISTRACTOR_FRAME_TRIES,
) -> bool:
    h, w = canvas.shape
    for _ in range(max_tries):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and canvas[rr, cc] != -1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        ring, _ = _frame_ring_coords(r0, c0, ch, cw)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True
    return False


def _try_place_distractor_network(
    canvas: np.ndarray,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(canvas[r + dr, c + dc] != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue
        canvas[r, c] = line_col
        path = _grow_line(canvas, (r, c), line_col, rng.randint(*DISTRACTOR_LINE_LEN), rng)
        if len(path) < 3:
            for pr, pc in path:
                canvas[pr, pc] = -1
            continue
        attached = 0
        for _ in range(rng.randint(*DISTRACTOR_NETWORK_FRAMES)):
            frame_col = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, frame_col, rng) is not None:
                attached += 1
        if attached > 0:
            return True
        for pr, pc in path:
            canvas[pr, pc] = -1
    return False


def _edge_seed_block(side: str, h: int, w: int, rng: random.Random) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    length = rng.choice(SEED_BLOCK_LENGTHS)
    if side == "T":
        c0 = rng.randint(2, w - length - 3)
        seed = [(0, c0 + dc) for dc in range(length)]
        first = (1, c0 + rng.choice([0, length - 1]))
    elif side == "B":
        c0 = rng.randint(2, w - length - 3)
        seed = [(h - 1, c0 + dc) for dc in range(length)]
        first = (h - 2, c0 + rng.choice([0, length - 1]))
    elif side == "L":
        r0 = rng.randint(2, h - length - 3)
        seed = [(r0 + dr, 0) for dr in range(length)]
        first = (r0 + rng.choice([0, length - 1]), 1)
    elif side == "R":
        r0 = rng.randint(2, h - length - 3)
        seed = [(r0 + dr, w - 1) for dr in range(length)]
        first = (r0 + rng.choice([0, length - 1]), w - 2)
    else:
        raise ValueError(side)
    return seed, first


def _distribute_loop_budget(num_seeds: int, num_attached_target: int, rng: random.Random) -> list[int]:
    order = list(range(num_seeds))
    rng.shuffle(order)
    per_seed = [0] * num_seeds
    remaining = num_attached_target
    while remaining > 0 and any(per_seed[i] == 0 for i in range(num_seeds)):
        for idx in order:
            if per_seed[idx] == 0 and remaining > 0:
                per_seed[idx] += 1
                remaining -= 1
    while remaining > 0:
        idx = rng.choice(order)
        per_seed[idx] += 1
        remaining -= 1
    return per_seed


def _grid_has_clean_seed_borders(grid: np.ndarray, expected_seed_colors: set[int], expected_count: int) -> bool:
    bg = int(get_bg(grid))
    seed_blocks = _combo_seed_blocks(grid, bg)
    if len(seed_blocks) != expected_count:
        return False
    found_colors = {int(grid[pts[0]]) for pts in seed_blocks}
    if found_colors != expected_seed_colors:
        return False
    for pts in seed_blocks:
        color = int(grid[pts[0]])
        if int((grid == color).sum()) <= len(pts):
            return False
    h, w = grid.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    if any(int(grid[r, c]) != bg for r, c in corners):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    if h < 8 or w < 8:
        return None
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if num_seeds > 4 or len(palette) < num_seeds + 1:
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = list(seed_colors)

    sides = ["T", "B", "L", "R"]
    rng.shuffle(sides)
    sides_used = sides[:num_seeds]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette) - 1 if len(palette) > 1 else 1))
    if num_frame_cols <= 0:
        return None
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]
    distractor_line_choices = palette[:]

    lines_info: list[tuple[int, list[tuple[int, int]], int, list[tuple[int, int]]]] = []
    for idx in range(num_seeds):
        seed_pts, first_rc = _edge_seed_block(sides_used[idx], h, w, rng)
        if any(canvas[r, c] != -1 for r, c in seed_pts):
            return None
        if canvas[first_rc] != -1:
            return None
        for r, c in seed_pts:
            canvas[r, c] = seed_colors[idx]
        canvas[first_rc] = line_colors[idx]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, first_rc, line_colors[idx], target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[idx], seed_pts, line_colors[idx], path))

    total_loops_target = rng.randint(num_seeds, TOTAL_LOOPS_MAX)
    num_distractors_target = rng.randint(0, max(0, total_loops_target - num_seeds))
    num_attached_target = total_loops_target - num_distractors_target
    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)

    for idx, (_, _seed_pts, line_col, path) in enumerate(lines_info):
        attached = 0
        attach_path = path[1:] if len(path) > 1 else path
        for _ in range(per_seed[idx]):
            frame_col = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, attach_path, line_col, frame_col, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    if distractor_line_choices and rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(distractor_line_choices), frame_colors_pool, rng)
    else:
        for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
            _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)
    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    expected_seed_colors = {seed_col for seed_col, _, _, _ in lines_info}
    if not _grid_has_clean_seed_borders(grid, expected_seed_colors, num_seeds):
        return None

    try:
        fused_out = main(grid)
        d4_out = _solve_d4_only(grid)
        d13_out = _solve_d13_only(grid)
    except Exception:
        return None

    if np.array_equal(fused_out, grid):
        return None
    if np.array_equal(fused_out, d4_out):
        return None
    if np.array_equal(fused_out, d13_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
