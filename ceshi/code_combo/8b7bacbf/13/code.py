# Combo variant of 8b7bacbf (index 13)
# combo_spec: D4/0+D11/1
# Name: 假种子干扰+非边缘真种子

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
------------------------------
- Original task:
  - A true seed is a globally unique singleton color on the border.
  - The seed touches a line, the line reaches one or more frame rings, and the
    cavities inside those rings are filled with the seed color.
- Combo task:
  - True seeds move into the interior.
  - Several other singleton colors are also present and look seed-like.
  - The most misleading fake seeds are placed on the border and touch short
    lines, so the old border-scan heuristic points at the wrong pixels.
  - The correct discriminator is structural: a true seed is a singleton whose
    own 8-connected non-background component reaches at least one valid cavity
    boundary through its inferred line color.
- Small trace:
  - Suppose an interior singleton S touches line L and L reaches a ring R.
  - A border singleton F touches a short line stub T, but T reaches no ring.
  - Original logic would inspect border pixels first and focus on F.
  - Combo logic scans all singleton colors, tests each candidate against the
    cavity structure, rejects F because it reaches no fillable ring, and keeps S
    because its traced component does reach R.

b) Complexity calibration
-------------------------
- The variant must look different from both the original and the plain D4
  variant, so every generated grid includes:
  - at least one interior true seed,
  - at least one border fake seed,
  - at least one filled cavity.
- Fake seeds are attached to short non-bg stubs instead of being isolated dots,
  so they remain plausible candidates without introducing extra cavity-bearing
  components that would confuse the rule.
- Counts stay small: 1-2 true seeds and 1-2 fake seeds. This preserves visual
  readability and avoids exhausting the ARC palette.

c) Implicit assumption audit for main()
--------------------------------------
- `get_bg()` as most-common color still holds and is revalidated by the
  generator.
- The old `_seed_pixels()` assumption "true seed must be on the border" is no
  longer valid. Candidate discovery must therefore scan all singleton colors.
- The old local line-color inference still works only if a candidate has at
  least one non-bg 4-neighbor. The generator now guarantees that for both true
  and fake seeds, so the solver can reject any singleton lacking such support.
- Tracing the entire non-bg 8-component from a candidate is still sound, but it
  is no longer sufficient by itself because fake seeds also touch non-bg lines.
  The extra requirement "this component must touch a valid cavity boundary via
  the inferred line color" is what separates true and fake seeds.
- The cavity extraction logic itself is unchanged: it depends only on background
  regions and single-color ring boundaries, not on seed placement.

d) Placement legality
---------------------
- Interior true seeds are placed on empty cells with an empty 3x3 halo, then a
  4-neighbor first line cell is reserved before line growth starts.
- Border fake seeds are placed only when the border pixel, its inward adjacent
  first line cell, and their local neighborhoods are empty.
- Line growth rejects any step that would become 8-adjacent to existing foreign
  occupied cells, so true networks and fake stubs stay disconnected.
- Frame placement retains the original legality checks: ring and cavity cells
  must be empty, the cavity cannot touch the line directly, and the ring must be
  8-adjacent to the owning line.

e) Visual observability
-----------------------
- Across a few examples, a human can see:
  - border singleton pixels exist but do not control any fill,
  - interior singleton pixels do control fills,
  - the working singleton is the one whose attached structure actually reaches a
    closed frame loop.
- This is observable without hidden numeric color ordering: the cue is purely
  geometric and repeats across examples.

f) Visual cleanliness
---------------------
- Fake seeds are limited to short disconnected stubs, so they do not clutter the
  same region as the frame-bearing true networks.
- True seeds remain away from the border, and frames stay attached only to true
  lines, which keeps the filled-vs-unfilled distinction legible.

g) Inferability of the change
-----------------------------
- No arbitrary color remapping is introduced. Seed color still directly becomes
  the cavity fill color.
- The only extra reasoning burden is identifying which singleton candidates are
  real. That rule is supported structurally because only the real seed's own
  component reaches frame rings with enclosed background.

Post-change assumption review
-----------------------------
- `main()` now assumes that every real seed has at least one non-bg 4-neighbor
  and at least one reachable cavity satisfying the original frame condition.
- `generate_input()` explicitly guarantees both properties and also guarantees
  that fake seeds fail the cavity-reachability test.
"""

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _on_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _singleton_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    singletons: list[tuple[int, int, int]] = []
    vals, cnts = np.unique(g, return_counts=True)
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        singletons.append((r, c, v))
    return singletons


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
    regions: list[tuple[set[tuple[int, int]], set[tuple[int, int]], bool, int]] = []
    for pts in bg_comps:
        region = set(pts)
        touches = _touches_border(pts, h, w)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary, touches, len(region)))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (reg, _, _, _) in enumerate(regions):
        if any(p in reg for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, bnd, _, _) in enumerate(regions):
        if i in outside_idxs or not bnd:
            continue
        bcols = {int(g[p]) for p in bnd}
        cavities.append((reg, bnd, bcols))
    return cavities


def _candidate_fillable_regions(
    g: np.ndarray,
    bg: int,
    cavities: list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]],
    start: tuple[int, int],
) -> tuple[int | None, list[set[tuple[int, int]]]]:
    h, w = g.shape
    sr, sc = start
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                neigh_cols.append(v)
    if not neigh_cols:
        return None, []

    line_col = Counter(neigh_cols).most_common(1)[0][0]
    traced = _nonbg_component8_from(g, start, bg)
    fillable: list[set[tuple[int, int]]] = []
    for cav, boundary, bcols in cavities:
        if len(bcols) != 1:
            continue
        boundary_in_traced = boundary & traced
        if not boundary_in_traced:
            continue

        ok = False
        for r, c in boundary_in_traced:
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                    ok = True
                    break
            if ok:
                break
        if ok:
            fillable.append(cav)
    return line_col, fillable


def _solve(grid: np.ndarray, *, border_only: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    cavities = _collect_cavities(g, bg)

    for sr, sc, seed_col in _singleton_pixels(g, bg):
        if border_only and not _on_border(sr, sc, h, w):
            continue
        _line_col, fillable = _candidate_fillable_regions(g, bg, cavities, (sr, sc))
        if not fillable:
            continue
        for cav in fillable:
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed cavities using the singleton candidate whose own drawing truly reaches a frame loop.

    Input structure:
    - The background is the most common color.
    - There are several globally unique single-pixel colors that all look seed-like.
    - Real seeds lie away from the border and touch longer line+frame networks, while fake seeds can sit on the border and touch short disconnected stubs.

    Rule:
    - Consider every globally unique non-background pixel as a candidate seed.
    - Trace the 8-connected non-background component from that candidate and infer its line color from adjacent non-background 4-neighbors.
    - Fill only those enclosed background cavities whose single-color boundary is reached by that traced component through the candidate's own line; candidates that do not reach any valid cavity are fake and do nothing.

    Key insight:
    Border placement is no longer a reliable cue. The true seed is the singleton whose attached structure uniquely connects to a closed frame ring, so cavity reachability filters out the visual decoys.
    """
    return _solve(grid, border_only=False)


# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
FAKE_SEED_COUNT_CHOICES = [1, 1, 2, 2]
FAKE_SEED_MAX = 2
SEED_MARGIN = 2
SEED_PLACEMENT_TRIES = 90
BORDER_SEED_PLACEMENT_TRIES = 120
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)
MIN_FAKE_LINE_LEN = 3
FAKE_LINE_LEN_RANGE = (4, 10)

FRAME_COLORS_MAX = 3
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 40

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
SEEDED_LINE_BORDER_AVOID_PROB = 0.35
BIAS_FLIP_PROB = 0.3

GEN_RETRIES = 300
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = ch + 2, cw + 2
    ring = []
    for dr in range(H):
        for dc in range(W):
            on_bdr = dr == 0 or dr == H - 1 or dc == 0 or dc == W - 1
            corner = dr in (0, H - 1) and dc in (0, W - 1)
            if on_bdr and not corner:
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
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return True
    return False


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
    *,
    bias_outward: bool,
    border_avoid_prob: float,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    path_set = {start}
    mid_r = (h - 1) / 2
    mid_c = (w - 1) / 2
    dr_side = -1 if start[0] < mid_r else 1
    dc_side = -1 if start[1] < mid_c else 1
    if bias_outward:
        dr_bias = dr_side
        dc_bias = dc_side
    else:
        dr_bias = -dr_side
        dc_bias = -dc_side
    cur = start
    while len(path) < target_len:
        dirs = []
        for ddr, ddc in DIAG_DIRS:
            aligned = ddr == dr_bias or ddc == dc_bias
            w_d = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * w_d)
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            w_d = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * w_d)
        rng.shuffle(dirs)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if any(
                _in_bounds(h, w, nnr, nnc)
                and grid[nnr, nnc] != -1
                and (nnr, nnc) != cur
                for nnr, nnc in neighbors8(nr, nc)
            ):
                continue
            if _on_border(nr, nc, h, w) and rng.random() < border_avoid_prob:
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


def _pick_true_seed_and_first(canvas: np.ndarray, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = canvas.shape
    mid_r = (h - 1) / 2
    mid_c = (w - 1) / 2
    candidates = []
    for r in range(SEED_MARGIN, h - SEED_MARGIN):
        for c in range(SEED_MARGIN, w - SEED_MARGIN):
            if canvas[r, c] != -1:
                continue
            clear = True
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if canvas[rr, cc] != -1:
                        clear = False
                        break
                if not clear:
                    break
            if not clear:
                continue
            centrality = abs(r - mid_r) + abs(c - mid_c)
            border_dist = min(r, c, h - 1 - r, w - 1 - c)
            candidates.append((centrality, -border_dist, (r, c)))
    if not candidates:
        return None
    candidates.sort()
    top_k = min(12, len(candidates))
    seed_rc = candidates[rng.randrange(top_k)][2]

    seed_dist = abs(seed_rc[0] - mid_r) + abs(seed_rc[1] - mid_c)
    first_choices = []
    for nr, nc in neighbors4(*seed_rc):
        if not _in_bounds(h, w, nr, nc) or canvas[nr, nc] != -1:
            continue
        next_dist = abs(nr - mid_r) + abs(nc - mid_c)
        outward_rank = 0 if next_dist > seed_dist else 1
        first_choices.append((outward_rank, rng.random(), (nr, nc)))
    if not first_choices:
        return None
    first_choices.sort()
    return seed_rc, first_choices[0][2]


def _border_first_choices(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    choices = []
    if r == 0:
        choices.append((1, c))
    if r == h - 1:
        choices.append((h - 2, c))
    if c == 0:
        choices.append((r, 1))
    if c == w - 1:
        choices.append((r, w - 2))
    return choices


def _local_clear_for_border_seed(canvas: np.ndarray, seed_rc: tuple[int, int], first_rc: tuple[int, int]) -> bool:
    h, w = canvas.shape
    cells = {seed_rc, first_rc}
    for base_r, base_c in cells:
        for nr, nc in neighbors8(base_r, base_c):
            if _in_bounds(h, w, nr, nc) and (nr, nc) not in cells and canvas[nr, nc] != -1:
                return False
    return True


def _pick_border_seed_and_first(canvas: np.ndarray, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = canvas.shape
    candidates = []
    border_pts = []
    for c in range(w):
        border_pts.append((0, c))
        border_pts.append((h - 1, c))
    for r in range(1, h - 1):
        border_pts.append((r, 0))
        border_pts.append((r, w - 1))

    for r, c in border_pts:
        if canvas[r, c] != -1:
            continue
        for first_rc in _border_first_choices(r, c, h, w):
            fr, fc = first_rc
            if canvas[fr, fc] != -1:
                continue
            if not _local_clear_for_border_seed(canvas, (r, c), first_rc):
                continue
            corner_dist = min(
                abs(r - 0) + abs(c - 0),
                abs(r - 0) + abs(c - (w - 1)),
                abs(r - (h - 1)) + abs(c - 0),
                abs(r - (h - 1)) + abs(c - (w - 1)),
            )
            candidates.append((corner_dist, rng.random(), (r, c), first_rc))
    if not candidates:
        return None
    candidates.sort()
    top_k = min(12, len(candidates))
    _score, _noise, seed_rc, first_rc = candidates[rng.randrange(top_k)]
    return seed_rc, first_rc


def _distribute_loop_budget(num_seeds: int, num_attached_target: int, rng: random.Random) -> list[int]:
    seed_order = list(range(num_seeds))
    rng.shuffle(seed_order)
    per_seed = [0] * num_seeds
    remaining = num_attached_target
    while remaining > 0 and any(per_seed[i] == 0 for i in range(num_seeds)):
        for i in seed_order:
            if per_seed[i] == 0 and remaining > 0:
                per_seed[i] += 1
                remaining -= 1
    while remaining > 0:
        per_seed[rng.choice(seed_order)] += 1
        remaining -= 1
    return per_seed


def _erase_points(canvas: np.ndarray, pts: list[tuple[int, int]]) -> None:
    for r, c in pts:
        canvas[r, c] = -1


def _place_fake_seed_stub(
    canvas: np.ndarray,
    seed_col: int,
    line_col: int,
    rng: random.Random,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    for _ in range(BORDER_SEED_PLACEMENT_TRIES):
        picked = _pick_border_seed_and_first(canvas, rng)
        if picked is None:
            return None
        seed_rc, first_rc = picked
        canvas[seed_rc] = seed_col
        canvas[first_rc] = line_col
        target_len = rng.randint(*FAKE_LINE_LEN_RANGE)
        path = _grow_line(
            canvas,
            first_rc,
            line_col,
            target_len,
            rng,
            bias_outward=False,
            border_avoid_prob=BORDER_AVOIDANCE_PROB,
        )
        if len(path) >= MIN_FAKE_LINE_LEN:
            return seed_rc, path
        _erase_points(canvas, [seed_rc] + path)
    return None


def _valid_seed_records(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    bg = get_bg(g)
    cavities = _collect_cavities(g, bg)
    records = []
    for r, c, seed_col in _singleton_pixels(g, bg):
        _line_col, fillable = _candidate_fillable_regions(g, bg, cavities, (r, c))
        if fillable:
            records.append((r, c, seed_col, len(fillable)))
    return records


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_true = rng.choice(SEED_COUNT_WEIGHTS)
    max_fake = min(FAKE_SEED_MAX, len(palette) - (num_true * 2 + 2))
    if max_fake < 1:
        return None
    fake_choices = [n for n in FAKE_SEED_COUNT_CHOICES if n <= max_fake]
    if not fake_choices:
        return None
    num_fake = rng.choice(fake_choices)

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_true)]
    line_colors = [palette.pop() for _ in range(num_true)]
    fake_seed_colors = [palette.pop() for _ in range(num_fake)]
    fake_line_col = palette.pop()
    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]] = []
    for i in range(num_true):
        picked = _pick_true_seed_and_first(canvas, rng)
        if picked is None:
            return None
        seed_rc, first_rc = picked
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(
            canvas,
            first_rc,
            line_colors[i],
            target_len,
            rng,
            bias_outward=True,
            border_avoid_prob=SEEDED_LINE_BORDER_AVOID_PROB,
        )
        if len(path) < MIN_LINE_LEN:
            _erase_points(canvas, [seed_rc] + path)
            return None
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    total_loops_target = rng.randint(num_true, TOTAL_LOOPS_MAX)
    per_seed = _distribute_loop_budget(num_true, total_loops_target, rng)
    for i, (_seed_col, _seed_rc, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            frame_col = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, frame_col, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    fake_infos = []
    for fake_seed_col in fake_seed_colors:
        placed = _place_fake_seed_stub(canvas, fake_seed_col, fake_line_col, rng)
        if placed is None:
            return None
        fake_infos.append((fake_seed_col, placed[0], placed[1]))

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for col in seed_colors + fake_seed_colors:
        if int((grid == col).sum()) != 1:
            return None

    singletons = _singleton_pixels(grid, bg)
    if len(singletons) != num_true + num_fake:
        return None

    border_singletons = [(r, c, v) for r, c, v in singletons if _on_border(r, c, h, w)]
    if len(border_singletons) < 1:
        return None

    valid_records = _valid_seed_records(grid)
    if len(valid_records) != num_true:
        return None
    if any(_on_border(r, c, h, w) for r, c, _seed_col, _cnt in valid_records):
        return None

    try:
        true_out = main(grid)
    except Exception:
        return None
    if np.array_equal(true_out, grid):
        return None

    border_out = _solve(grid, border_only=True)
    if np.array_equal(border_out, true_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
