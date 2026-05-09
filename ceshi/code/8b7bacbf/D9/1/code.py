# Variant of 8b7bacbf
# Dimension: D9 — 腔体边界颜色约束 (obstacle)
# Change: 单色边界约束——只有边界完全由一种颜色组成的封闭区域才被视为有效腔体
#         -> 边界颜色必须与线路颜色匹配

import random
import sys
from collections import Counter

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the D9 variant
# ---------------------------------------------------------------------------
# variant_spec:
#   unchanged_backbone:
#     - Border seeds are still unique single pixels on the outer border.
#     - Each seed still has exactly one 8-connected meandering line component.
#     - Background cavities are still detected as 4-connected bg regions whose
#       boundary is made of non-bg cells.
#   concrete_visual_change:
#     - Original task: any attached cavity with a monochrome boundary could be
#       filled, even if the loop color differed from the touching line color.
#     - D9 variant: an attached cavity is fillable only when its boundary color
#       is exactly the same as the seed's line color.
#     - Therefore generator samples must contain both:
#       1. attached loops whose ring color == line color (these fill), and
#       2. attached loops whose ring color != line color (these stay bg).
#     - This makes the new rule visible from input/output alone: adjacency to a
#       seed line is no longer sufficient; color agreement is also required.
#   behavior_trace_example:
#     - Suppose a red seed touches a blue line. Along that line sit two loops:
#       one blue loop and one yellow loop. In the original solver, both loops
#       are single-color attached cavities, so both fill red. In D9, only the
#       blue-boundary cavity fills red; the yellow one remains bg.
#   complexity_calibration:
#     - The new degree of freedom is intentionally small: each attached frame is
#       either matching or mismatching the line color. This yields repeated local
#       evidence inside one grid without introducing large color vocabularies or
#       hard-to-track multi-step mappings.
#   implicit_assumption_audit_for_main:
#     - `_seed_pixels`: still valid because seeds remain unique border pixels.
#     - `_line_color_from_seed`: still valid because each seed still has a
#       non-bg 4-neighbor that starts its line.
#     - cavity extraction: still valid because cavities remain bg regions with
#       non-bg boundaries.
#     - original `len(bcols) == 1` check is no longer sufficient: under D9 it
#       would wrongly accept attached monochrome loops of the wrong color.
#     - original "boundary adjacent to at least one line-color pixel" check
#       becomes too weak when ring color == line color, because neighboring ring
#       pixels could satisfy it trivially. The new helper requires contact with a
#       traced line-color pixel outside the boundary set.
#   placement_legality:
#     - Attached frames are placed so their ring is 8-adjacent only to pixels of
#       the intended line path, never to unrelated non-bg pixels. This prevents
#       accidental multi-line merges and keeps cavity ownership unambiguous.
#     - Cavity cells are kept free from pre-existing 8-neighbors outside the ring
#       so the solver sees a clean enclosed bg region.
#   visual_observability:
#     - Every generated grid contains at least one matching attached cavity and
#       at least one mismatching attached cavity. Hence the new D9 rule changes
#       the output relative to the original task on the same input.
#   cleanliness:
#     - The generator avoids overlapping or near-overlapping structures by
#       enforcing empty placement regions and line isolation, so loops remain
#       legible and the line/frame relation is easy to parse by eye.


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _line_color_from_seed(g: np.ndarray, seed: tuple[int, int], bg: int) -> int:
    h, w = g.shape
    sr, sc = seed
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                neigh_cols.append(v)
    assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
    return Counter(neigh_cols).most_common(1)[0][0]


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _collect_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    cavities = []
    for pts in bg_comps:
        region = set(pts)
        if any(p in region and int(g[p]) == bg for p in corners):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        if boundary:
            cavities.append((region, boundary, {int(g[p]) for p in boundary}))
    return cavities


def _has_external_line_contact(
    g: np.ndarray,
    boundary_in_traced: set[tuple[int, int]],
    full_boundary: set[tuple[int, int]],
    traced: set[tuple[int, int]],
    line_col: int,
) -> bool:
    h, w = g.shape
    for r, c in boundary_in_traced:
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in full_boundary or (nr, nc) not in traced:
                continue
            if int(g[nr, nc]) == line_col:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    cavities = _collect_cavities(g, bg)

    for seed in _seed_pixels(g, bg):
        seed_col = int(g[seed])
        line_col = _line_color_from_seed(g, seed, bg)
        traced = _nonbg_component8_from(g, seed, bg)

        for cav, boundary, bcols in cavities:
            if bcols != {line_col}:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue
            if not _has_external_line_contact(g, boundary_in_traced, boundary, traced, line_col):
                continue
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col
    return out


# --------------- generator ---------------

# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 3
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 45
DISTRACTOR_FRAME_TRIES = 45

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

MATCHING_FRAME_PROB = 0.6

DISTRACTOR_NETWORK_PROB = 0.45
DISTRACTOR_NETWORK_TRIES = 40
DISTRACTOR_LINE_LEN = (5, 14)
DISTRACTOR_SOLO_MAX = 1
DISTRACTOR_NETWORK_FRAMES = (1, 2)

GEN_RETRIES = 300
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = ch + 2, cw + 2
    ring = []
    for dr in range(h):
        for dc in range(w):
            on_bdr = dr == 0 or dc == 0 or dr == h - 1 or dc == w - 1
            corner = dr in (0, h - 1) and dc in (0, w - 1)
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    allowed_touch: set[tuple[int, int]],
) -> bool:
    h, w = canvas.shape
    ring_set = set(ring)

    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] != -1:
                return False

    touches_line = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            if canvas[nr, nc] == -1:
                continue
            if (nr, nc) not in allowed_touch:
                return False
            touches_line = True
    return touches_line


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
    allowed_contacts: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start

    while len(path) < target_len:
        dirs = []
        for ddr, ddc in DIAG_DIRS:
            aligned = ddr == dr_bias or ddc == dc_bias
            dirs.extend([(ddr, ddc)] * (DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER))
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            dirs.extend([(ddr, ddc)] * (ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER))
        rng.shuffle(dirs)

        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue

            bad = False
            for nnr, nnc in neighbors8(nr, nc):
                if not _in_bounds(h, w, nnr, nnc):
                    continue
                if (nnr, nnc) == cur or (nnr, nnc) in allowed_contacts:
                    continue
                if grid[nnr, nnc] != -1:
                    bad = True
                    break
            if bad:
                continue

            cur = (nr, nc)
            path.append(cur)
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
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    allowed_touch = set(line_path)

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
            if not _validate_frame_placement(canvas, ring, cavity, allowed_touch):
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
        if h - fh - 2 < 1 or w - fw - 2 < 1:
            return False
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
    frame_colors: list[int],
    rng: random.Random,
) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(canvas[r + dr, c + dc] != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue
        canvas[r, c] = line_col
        path = _grow_line(canvas, (r, c), line_col, rng.randint(*DISTRACTOR_LINE_LEN), rng, set())
        if len(path) < 3:
            for pr, pc in path:
                canvas[pr, pc] = -1
            continue
        attached = 0
        for _ in range(rng.randint(*DISTRACTOR_NETWORK_FRAMES)):
            fc = rng.choice(frame_colors)
            if _try_attach_frame(canvas, path, fc, rng) is not None:
                attached += 1
        if attached > 0:
            return True
        for pr, pc in path:
            canvas[pr, pc] = -1
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


def _choose_mismatch_color(line_idx: int, line_colors: list[int], extra_frame_colors: list[int], rng: random.Random) -> int:
    options = [c for c in extra_frame_colors + line_colors if c != line_colors[line_idx]]
    return rng.choice(options)


def _build_frame_plan(
    num_seeds: int,
    line_colors: list[int],
    extra_frame_colors: list[int],
    total_attached_target: int,
    rng: random.Random,
) -> list[list[int]]:
    plans = [[line_colors[i]] for i in range(num_seeds)]

    mismatch_seed = rng.randrange(num_seeds)
    plans[mismatch_seed].append(_choose_mismatch_color(mismatch_seed, line_colors, extra_frame_colors, rng))

    remaining = total_attached_target - (num_seeds + 1)
    while remaining > 0:
        i = rng.randrange(num_seeds)
        if rng.random() < MATCHING_FRAME_PROB:
            plans[i].append(line_colors[i])
        else:
            plans[i].append(_choose_mismatch_color(i, line_colors, extra_frame_colors, rng))
        remaining -= 1
    return plans


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cavities = _collect_cavities(g, bg)
    matching = 0
    mismatched = 0

    for seed in _seed_pixels(g, bg):
        line_col = _line_color_from_seed(g, seed, bg)
        traced = _nonbg_component8_from(g, seed, bg)
        for _, boundary, bcols in cavities:
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue
            if not _has_external_line_contact(g, boundary_in_traced, boundary, traced, line_col):
                continue
            if len(bcols) != 1:
                continue
            if bcols == {line_col}:
                matching += 1
            else:
                mismatched += 1
    return {"matching_attached": matching, "mismatched_attached": mismatched}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if len(palette) < num_seeds * 2 + 1:
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]

    num_extra_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    extra_frame_colors = [palette.pop() for _ in range(num_extra_frame_cols)]

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    lines_info = []
    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w, rng)
        if canvas[seed_rc] != -1 or canvas[first_rc] != -1:
            return None
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng, {seed_rc})
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    min_attached = max(2, num_seeds + 1)
    max_attached = min(TOTAL_LOOPS_MAX, num_seeds + 4)
    total_attached_target = rng.randint(min_attached, max_attached)
    frame_plans = _build_frame_plan(num_seeds, line_colors, extra_frame_colors, total_attached_target, rng)

    for i, (_, _, _, path) in enumerate(lines_info):
        for frame_col in frame_plans[i]:
            if _try_attach_frame(canvas, path, frame_col, rng) is None:
                return None

    distractor_frame_colors = extra_frame_colors + line_colors
    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(line_colors), distractor_frame_colors, rng)
    elif rng.random() < 0.5:
        _try_attach_distractor_frame(canvas, rng.choice(distractor_frame_colors), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    metrics = _variant_metrics(grid)
    if metrics["matching_attached"] < num_seeds or metrics["mismatched_attached"] < 1:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
