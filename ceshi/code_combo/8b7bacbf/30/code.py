# Combo variant of 8b7bacbf (index 30)
# combo_spec: D3/3+D9/1+D19/0
# Name: 颜色图例+框架色填充+边界色匹配线路

# Design analysis
# ===============
# fusion_summary:
#   - Preserve the original backbone: border seeds launch thin 8-connected lines,
#     lines touch the OUTSIDE of closed frame rings, and enclosed bg cavities are
#     the only cells that may be flood-filled.
#   - Fuse the three dimensions into one chain:
#       1. A cavity is eligible only if its ring color matches the traced line
#          color (D9).
#       2. The ring/line color is then used as the legend key in the corner
#          lookup box (D19, adapted to the combined rule).
#       3. The cavity is filled with the mapped legend output color, not with the
#          seed color and not with the frame color directly (D3 effect mediated
#          through legend lookup).
# concrete_trace:
#   - Example: a blue seed touches an orange line. Along that line there are two
#     attached loops: one orange loop and one green loop. The legend contains the
#     row [orange, bg, yellow].
#   - Original task: both attached monochrome loops would fill blue.
#   - D3 alone: the orange loop would fill orange, and the green loop would fill
#     green.
#   - This combo: only the orange loop qualifies because boundary color must
#     equal the line color; it then fills yellow by legend lookup. The green loop
#     stays background even though it is attached to the same traced component.
# complexity_calibration:
#   - The legend has exactly two rows, matching the task's 1-2 seed regime, and
#     the row structure is fixed as [key, bg, fill] inside a small boxed panel.
#   - Fill colors are sampled distinct from both seed and line colors so the
#     legend-driven recoloring is visible and cannot collapse into either the
#     original seed-fill rule or the direct frame-fill rule.
#   - Each generated sample contains both at least one matching attached cavity
#     and at least one mismatching attached cavity, so the D9 gate is observable
#     from a single grid.
# implicit_assumption_audit:
#   - Original `_seed_pixels()` assumed the decisive color appears once in the
#     whole grid. That breaks once a legend repeats task-relevant colors, so seed
#     detection is restricted to cells outside the legend box.
#   - Original cavity acceptance only required a monochrome boundary. That is now
#     too weak; the boundary color must equal the traced line color.
#   - Original fill assignment wrote `seed_color`. That is replaced by
#     `legend_map[line_color]`, so generator guarantees every active line color
#     has exactly one legend row.
#   - When ring color equals line color, a naive "adjacent to some line-colored
#     pixel" test can self-satisfy via neighboring ring pixels. The solver uses
#     an external-contact check that requires adjacency to a traced line-colored
#     pixel outside the boundary set.
# placement_legality:
#   - The legend is fixed in the top-left corner and surrounded by a reserved
#     one-cell moat so lines and frames cannot accidentally connect to it.
#   - Seeds are restricted to TR / BL / BR, leaving at least one other border
#     background route for the outside bg region.
#   - Attached rings must occupy empty cells, cavities must stay free of
#     pre-existing 8-neighbors, and ring cells may touch only the intended line
#     path. This prevents ambiguous multi-line ownership.
# observability_and_cleanliness:
#   - The legend is spatially separated from the drawing, so it reads as an
#     instruction panel rather than clutter.
#   - The left legend token repeats a visible line/ring color from the main
#     drawing, making the mapping human-readable without relying on color ids.
#   - Mismatching attached loops remain visibly unfilled, while matching ones
#     change to a third color from the legend. That gives repeated support for
#     the full combined rule across small sample counts.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


LEGEND_H = 4
LEGEND_W = 5
LEGEND_ROWS = 2
LEGEND_PAD = 1
RESERVED = -2


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _legend_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:LEGEND_H, :LEGEND_W] = True
    return mask


def _legend_border_coords() -> list[tuple[int, int]]:
    coords = []
    for r in range(LEGEND_H):
        for c in range(LEGEND_W):
            if r in (0, LEGEND_H - 1) or c in (0, LEGEND_W - 1):
                coords.append((r, c))
    return coords


def _extract_legend(g: np.ndarray, bg: int) -> dict[int, int]:
    box = g[:LEGEND_H, :LEGEND_W]
    legend_col = int(box[0, 0])
    assert legend_col != bg
    for r, c in _legend_border_coords():
        assert int(box[r, c]) == legend_col

    mapping: dict[int, int] = {}
    for row in range(1, 1 + LEGEND_ROWS):
        key = int(box[row, 1])
        mid = int(box[row, 2])
        fill = int(box[row, 3])
        assert key != bg and fill != bg
        assert key != legend_col and fill != legend_col
        assert mid == bg
        mapping[key] = fill
    return mapping


def _seed_pixels(g: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = g.shape
    outside = ~legend_mask
    vals, cnts = np.unique(g[outside], return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        rc = np.argwhere((g == v) & outside)
        if rc.shape[0] != 1:
            continue
        r, c = map(int, rc[0])
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
    cavities = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in region):
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


def _eligible_cavities_for_seed(
    g: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    cavities: list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]],
):
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
        yield cav, boundary, line_col


def _solve_with_fill_strategy(
    grid: np.ndarray,
    fill_for_seed,
    *,
    legend_mask: np.ndarray | None = None,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    if legend_mask is None:
        legend_mask = _legend_mask(g.shape)
    out = g.copy()
    cavities = _collect_cavities(g, bg)
    for seed in _seed_pixels(g, bg, legend_mask):
        fill_col = int(fill_for_seed(seed))
        for cav, _, _ in _eligible_cavities_for_seed(g, bg, seed, cavities):
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = fill_col
    return out


def _solve_seed_fill(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_mask = _legend_mask(g.shape)
    return _solve_with_fill_strategy(g, lambda seed: int(g[seed]), legend_mask=legend_mask)


def _solve_direct_frame_fill(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_mask = _legend_mask(g.shape)
    return _solve_with_fill_strategy(g, lambda seed: _line_color_from_seed(g, seed, bg), legend_mask=legend_mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a corner legend, trace border-seeded line networks, and recolor only the
    cavities whose frame color matches their line color.

    Input structure:
    - The top-left corner contains a small boxed legend. Each legend row is
      `[line/frame key, bg, fill color]`, so the left token names a structural
      color visible in the main drawing and the right token gives its output fill.
    - Outside the legend, there are 1-2 unique single-pixel seeds on the border.
      Each seed touches one 8-connected meandering line, and line-adjacent closed
      rings may enclose bg cavities.

    Rule:
    - For each border seed, infer its line color from the adjacent non-bg pixel
      and trace the entire 8-connected non-bg component.
    - A cavity is fillable only when its boundary is monochrome and that boundary
      color equals the traced seed's line color; attached loops of other colors
      remain unchanged.
    - Fill every eligible cavity with the legend-mapped color for that matching
      line/frame color.

    Key insight:
    - Three signals must be combined in sequence: the seed only identifies which
      drawing to trace, the matching frame/line color decides whether a cavity is
      valid, and the legend decides the final paint color. The output therefore
      depends on neither the seed color nor the frame color directly.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_mask = _legend_mask(g.shape)
    legend_map = _extract_legend(g, bg)
    cavities = _collect_cavities(g, bg)

    out = g.copy()
    for seed in _seed_pixels(g, bg, legend_mask):
        line_col = _line_color_from_seed(g, seed, bg)
        assert line_col in legend_map
        fill_col = int(legend_map[line_col])
        for cav, _, _ in _eligible_cavities_for_seed(g, bg, seed, cavities):
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = fill_col
    return out


# --------------- generator ---------------

GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

TOTAL_LOOPS_MAX = 8
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 45
DISTRACTOR_FRAME_TRIES = 30

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

MATCHING_FRAME_PROB = 0.6
DISTRACTOR_SOLO_PROB = 0.5
GEN_RETRIES = 400

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TR":
        return (0, w - 1), rng.choice([(0, w - 2), (1, w - 1)])
    if corner_kind == "BL":
        return (h - 1, 0), rng.choice([(h - 1, 1), (h - 2, 0)])
    if corner_kind == "BR":
        return (h - 1, w - 1), rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    raise ValueError(corner_kind)


def _place_legend(canvas: np.ndarray, legend_col: int, legend_rows: list[tuple[int, int]]) -> None:
    for r in range(LEGEND_H):
        for c in range(LEGEND_W):
            if r in (0, LEGEND_H - 1) or c in (0, LEGEND_W - 1):
                canvas[r, c] = legend_col
    for i, (key_col, fill_col) in enumerate(legend_rows, start=1):
        canvas[i, 1] = key_col
        canvas[i, 3] = fill_col


def _reserve_legend_moat(canvas: np.ndarray) -> None:
    h, w = canvas.shape
    max_r = min(h, LEGEND_H + LEGEND_PAD)
    max_c = min(w, LEGEND_W + LEGEND_PAD)
    for r in range(max_r):
        for c in range(max_c):
            if r < LEGEND_H and c < LEGEND_W:
                continue
            if canvas[r, c] == -1:
                canvas[r, c] = RESERVED


def _flatten_pairs(pairs: list[tuple[int, int]]) -> list[int]:
    return [v for a, b in pairs for v in (a, b)]


def _choose_mismatch_color(
    line_idx: int,
    line_colors: list[int],
    fill_colors: list[int],
    dummy_pairs: list[tuple[int, int]],
    rng: random.Random,
) -> int:
    options = fill_colors + line_colors + _flatten_pairs(dummy_pairs)
    options = [c for c in options if c != line_colors[line_idx]]
    return rng.choice(options)


def _build_frame_plan(
    num_seeds: int,
    line_colors: list[int],
    fill_colors: list[int],
    dummy_pairs: list[tuple[int, int]],
    total_attached_target: int,
    rng: random.Random,
) -> list[list[int]]:
    plans = [[line_colors[i]] for i in range(num_seeds)]

    mismatch_seed = rng.randrange(num_seeds)
    plans[mismatch_seed].append(_choose_mismatch_color(mismatch_seed, line_colors, fill_colors, dummy_pairs, rng))

    remaining = total_attached_target - (num_seeds + 1)
    while remaining > 0:
        i = rng.randrange(num_seeds)
        if rng.random() < MATCHING_FRAME_PROB:
            plans[i].append(line_colors[i])
        else:
            plans[i].append(_choose_mismatch_color(i, line_colors, fill_colors, dummy_pairs, rng))
        remaining -= 1
    return plans


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_mask = _legend_mask(g.shape)
    legend_map = _extract_legend(g, bg)
    cavities = _collect_cavities(g, bg)
    matching = 0
    mismatched = 0

    for seed in _seed_pixels(g, bg, legend_mask):
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
            if bcols == {line_col} and line_col in legend_map:
                matching += 1
            else:
                mismatched += 1

    out = main(g)
    changed = (g == bg) & (out != g) & ~legend_mask
    changed_colors = {int(v) for v in out[changed]}
    legend_fill_values = set(legend_map.values())
    return {
        "matching_attached": matching,
        "mismatched_attached": mismatched,
        "filled_cells": int(changed.sum()),
        "legend_fill_colors_used": len(changed_colors & legend_fill_values),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    needed = 1 + num_seeds + num_seeds + num_seeds + 2 * (LEGEND_ROWS - num_seeds)
    if needed > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)

    legend_col = palette.pop()
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]
    fill_colors = [palette.pop() for _ in range(num_seeds)]
    dummy_pairs = [(palette.pop(), palette.pop()) for _ in range(LEGEND_ROWS - num_seeds)]

    legend_rows = list(zip(line_colors, fill_colors)) + dummy_pairs
    rng.shuffle(legend_rows)
    _place_legend(canvas, legend_col, legend_rows)
    _reserve_legend_moat(canvas)

    all_corners = ["TR", "BL", "BR"]
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
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], fill_colors[i], path))

    min_attached = max(2, num_seeds + 1)
    max_attached = min(TOTAL_LOOPS_MAX, num_seeds + 4)
    total_attached_target = rng.randint(min_attached, max_attached)
    frame_plans = _build_frame_plan(num_seeds, line_colors, fill_colors, dummy_pairs, total_attached_target, rng)

    for i, (_, _, _, _, path) in enumerate(lines_info):
        for frame_col in frame_plans[i]:
            if _try_attach_frame(canvas, path, frame_col, rng) is None:
                return None

    if rng.random() < DISTRACTOR_SOLO_PROB:
        distractor_pool = fill_colors + line_colors + _flatten_pairs(dummy_pairs)
        _try_attach_distractor_frame(canvas, rng.choice(distractor_pool), rng)

    grid = np.where(canvas < 0, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    legend_mask = _legend_mask(grid.shape)
    for seed_col, _, _, _, _ in lines_info:
        if int(((grid == seed_col) & ~legend_mask).sum()) != 1:
            return None

    try:
        out = main(grid)
        seed_fill = _solve_seed_fill(grid)
        frame_fill = _solve_direct_frame_fill(grid)
    except Exception:
        return None

    metrics = _variant_metrics(grid)
    if metrics["matching_attached"] < num_seeds or metrics["mismatched_attached"] < 1:
        return None
    if metrics["filled_cells"] <= 0 or metrics["legend_fill_colors_used"] <= 0:
        return None
    if np.array_equal(out, grid):
        return None
    if np.array_equal(out, seed_fill):
        return None
    if np.array_equal(out, frame_fill):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
