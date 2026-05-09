# Combo variant of 8b7bacbf (index 20)
# combo_spec: D4/0+D11/1+D14/0
# Name: 非边缘种子+假种子+4连通判定

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and behavior trace
---------------------------------------
- Base task:
  - A globally unique border pixel is the seed.
  - The solver traces the whole non-background drawing from that seed with
    8-connectivity.
  - Any enclosed bg cavity whose ring is touched by that traced drawing is
    filled, even if the line only meets the ring at a corner.
- Combo task:
  - True seeds are globally unique singleton pixels strictly inside the grid.
  - Border singleton pixels are now explicit fake seeds and must be ignored.
  - Interior singleton pixels are only true if their own 4-connected non-bg
    component reaches a frame ring that also has an orthogonal line contact.
  - Therefore two extra negative cases matter:
    1. a border singleton that sits on its own perfectly valid-looking
       seed-line-ring network
    2. an interior singleton attached to a 1-pixel stub that only touches a
       ring-bearing line network diagonally
  - Small trace:
    - Original logic on singleton F:
      singleton -> trace 8-connected drawing -> corner-touching frame counts.
    - Combo logic on border singleton B:
      reject immediately because B is on the border.
    - Combo logic on interior fake I:
      trace only its 4-connected non-bg component; the diagonal bridge is not
      traversable, so I never reaches any frame ring.
    - Combo logic on true interior seed S:
      S has a 4-neighbor line pixel; tracing 4-connected non-bg reaches the
      orthogonally attached frame ring; only that cavity fills.

b) Complexity calibration
-------------------------
- The generator keeps the original vocabulary: singleton seeds, thin lines, and
- single-color frame rings with bg cavities.
- To make all three dimensions visible without clutter:
  - every grid has at least one real interior seed,
  - at least one border fake seed on a border-starting decoy network,
  - at least one interior fake seed that is only diagonally bridged,
  - at least one orthogonally attached ring that fills,
  - at least one interior fake whose relaxed 8-connected search would reach a
    ring but whose 4-connected search would not.
- Counts stay small so the learner sees repeated evidence for each rule instead
  of many unrelated special cases.

c) Implicit-assumption audit of main()
--------------------------------------
- Original hidden assumption 1: scanning only the border is sufficient.
  This is now false; singleton discovery must be global and border pixels must
  be filtered out explicitly instead of being treated as the only candidates.
- Original hidden assumption 2: tracing with 8-connectivity from any candidate
  seed is harmless. This is now false because interior fake seeds are created so
  that 8-connectivity would jump across a diagonal bridge and reach a real ring.
- Original hidden assumption 3: a corner touch between line and ring is enough.
  This is now false; a ring only counts when the traced overlap with its
  boundary has a 4-neighbor line-color contact.
- New assumptions introduced by the combo solver:
  - a true seed has at least one non-bg 4-neighbor from which line color can be
    inferred,
  - every fillable cavity boundary is monochrome,
  - every true seed reaches at least one cavity through 4-connected non-bg.
  The generator enforces all three.

d) Placement legality
---------------------
- True seeds are placed with a clear halo and a reserved 4-neighbor first line
  pixel.
- True lines are grown as thin 4-connected orthogonal walks and may not step
  8-adjacent to foreign occupied cells, keeping different networks separated.
- Rings are only placed if ring cells and cavity cells are empty, cavities are
  not 8-adjacent to the line, and the requested contact mode (`ortho` or
  `diag_only`) is satisfied exactly.
- A border fake seed is placed on its own short border-entering line and still
  has exactly one non-bg 4-neighbor, so it visually looks like a real seed.
- An interior fake seed uses a 1-pixel line stub whose only 8-neighbor in the
  existing canvas is one chosen line pixel at a diagonal; this guarantees that
  the fake looks attached under 8-connectivity but is isolated under
  4-connectivity.

e) Visual observability
-----------------------
- Across a few examples, a human can see:
  - some singleton pixels lie inside and cause fills,
  - some singleton pixels lie on the border and do nothing,
  - some interior singleton pixels also do nothing because they only corner-link
    into a line-and-ring network.
- All cues are geometric and directly visible; no color id ordering is used.

f) Visual cleanliness
---------------------
- Fake seeds are only 1-pixel leaves or 1-pixel-stub motifs, so they remain
  easy to identify.
- Old disconnected distractor networks are removed; this avoids overlapping
  sources of ambiguity and keeps attention on the combo rule itself.
- The line growth and ring placement both enforce local spacing so regions do
  not collapse into noisy blobs.

g) Inferability of the changes
------------------------------
- Seed color still directly determines fill color, so there is no new hidden
  color mapping to infer.
- The only changed inference burden is structural:
  - seed must be interior,
  - reachability must be 4-connected,
  - ring contact must be orthogonal.
- Because each sample includes both positive and negative evidence for these
  conditions, the fusion does not degenerate back to the original task.
"""

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.insert(0, SOLVER_UTILS)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _on_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _singleton_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    singles: list[tuple[int, int, int]] = []
    vals, cnts = np.unique(g, return_counts=True)
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        singles.append((r, c, v))
    return singles


def _candidate_seeds(g: np.ndarray, bg: int, allow_border: bool) -> list[tuple[int, int]]:
    h, w = g.shape
    seeds = []
    for r, c, _v in _singleton_pixels(g, bg):
        if not allow_border and _on_border(r, c, h, w):
            continue
        seeds.append((r, c))
    seeds.sort()
    return seeds


def _nonbg_component_from(
    g: np.ndarray,
    start: tuple[int, int],
    bg: int,
    conn8: bool,
) -> set[tuple[int, int]]:
    h, w = g.shape
    neigh_fn = neighbors8 if conn8 else neighbors4
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neigh_fn(r, c):
            if _in_bounds(h, w, nr, nc) and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _extract_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    regions: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []
    for pts in bg_comps:
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if _in_bounds(h, w, nr, nc) and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (reg, _) in enumerate(regions):
        if any(p in reg for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, bnd) in enumerate(regions):
        if i in outside_idxs or not bnd:
            continue
        bcols = {int(g[p]) for p in bnd}
        cavities.append((reg, bnd, bcols))
    return cavities


def _candidate_line_color(g: np.ndarray, seed_rc: tuple[int, int], bg: int) -> int | None:
    h, w = g.shape
    neigh_cols = []
    for nr, nc in neighbors4(*seed_rc):
        if _in_bounds(h, w, nr, nc):
            v = int(g[nr, nc])
            if v != bg:
                neigh_cols.append(v)
    if not neigh_cols:
        return None
    return Counter(neigh_cols).most_common(1)[0][0]


def _contact_mode(
    g: np.ndarray,
    boundary_in_traced: set[tuple[int, int]],
    traced: set[tuple[int, int]],
    line_col: int,
) -> str | None:
    h, w = g.shape
    has8 = False
    has4 = False
    for r, c in boundary_in_traced:
        for nr, nc in neighbors8(r, c):
            if _in_bounds(h, w, nr, nc) and (nr, nc) in traced and int(g[nr, nc]) == line_col:
                has8 = True
                break
        for nr, nc in neighbors4(r, c):
            if _in_bounds(h, w, nr, nc) and (nr, nc) in traced and int(g[nr, nc]) == line_col:
                has4 = True
                break
    if has4:
        return "ortho"
    if has8:
        return "diag_only"
    return None


def _fillable_cavity_indices(
    g: np.ndarray,
    cavities: list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]],
    seed_rc: tuple[int, int],
    bg: int,
    *,
    reach_conn8: bool,
    allow_diag_contact: bool,
) -> tuple[int | None, set[tuple[int, int]], list[int]]:
    line_col = _candidate_line_color(g, seed_rc, bg)
    if line_col is None:
        return None, set(), []

    traced = _nonbg_component_from(g, seed_rc, bg, conn8=reach_conn8)
    fillable: list[int] = []
    for i, (_cav, boundary, bcols) in enumerate(cavities):
        if len(bcols) != 1:
            continue
        boundary_in_traced = boundary & traced
        if not boundary_in_traced:
            continue
        mode = _contact_mode(g, boundary_in_traced, traced, line_col)
        if mode is None:
            continue
        if mode == "diag_only" and not allow_diag_contact:
            continue
        fillable.append(i)
    return line_col, traced, fillable


def _solve(
    grid: np.ndarray,
    *,
    allow_border_candidates: bool,
    reach_conn8: bool,
    allow_diag_contact: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    cavities = _extract_cavities(g, bg)

    for sr, sc in _candidate_seeds(g, bg, allow_border=allow_border_candidates):
        seed_col = int(g[sr, sc])
        _line_col, _traced, fillable = _fillable_cavity_indices(
            g,
            cavities,
            (sr, sc),
            bg,
            reach_conn8=reach_conn8,
            allow_diag_contact=allow_diag_contact,
        )
        for cav_idx in fillable:
            cav, _boundary, _bcols = cavities[cav_idx]
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cavities = _extract_cavities(g, bg)
    h, w = g.shape

    metrics = {
        "singleton_total": 0,
        "interior_singletons": 0,
        "border_singletons": 0,
        "combo_true_candidates": 0,
        "border_reachable_if_allowed": 0,
        "interior_8only_candidates": 0,
        "diag_only_cavities_from_true": 0,
    }

    combo_true: list[tuple[int, int]] = []
    for r, c, _v in _singleton_pixels(g, bg):
        metrics["singleton_total"] += 1
        if _on_border(r, c, h, w):
            metrics["border_singletons"] += 1
        else:
            metrics["interior_singletons"] += 1

        _line_col4, _traced4, fillable4 = _fillable_cavity_indices(
            g,
            cavities,
            (r, c),
            bg,
            reach_conn8=False,
            allow_diag_contact=False,
        )
        if _on_border(r, c, h, w):
            if fillable4:
                metrics["border_reachable_if_allowed"] += 1
        else:
            _line_col8, _traced8, fillable8 = _fillable_cavity_indices(
                g,
                cavities,
                (r, c),
                bg,
                reach_conn8=True,
                allow_diag_contact=True,
            )
            if fillable4:
                metrics["combo_true_candidates"] += 1
                combo_true.append((r, c))
            elif fillable8:
                metrics["interior_8only_candidates"] += 1

    eight_only_cav_idxs: set[int] = set()
    for seed_rc in combo_true:
        _line_col4, _traced4, fillable4 = _fillable_cavity_indices(
            g,
            cavities,
            seed_rc,
            bg,
            reach_conn8=False,
            allow_diag_contact=False,
        )
        _line_col8, _traced8, fillable8 = _fillable_cavity_indices(
            g,
            cavities,
            seed_rc,
            bg,
            reach_conn8=True,
            allow_diag_contact=True,
        )
        for i in fillable8:
            if i not in fillable4:
                eight_only_cav_idxs.add(i)
    metrics["diag_only_cavities_from_true"] = len(eight_only_cav_idxs)
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed cavities using unique interior pixels as labels for valid line-frame networks.

    Input structure:
    - The most common color is the background.
    - Each true network has a unique singleton color placed strictly inside the grid.
    - Additional unique singleton pixels may appear as fake seeds, including on the border.
    - Thin non-background lines connect toward monochrome frame rings that enclose bg cavities.

    Rule:
    - Ignore every singleton on the border.
    - For each interior singleton, inspect its 4-connected non-background component.
    - It is a real seed only if that 4-connected component reaches a frame ring whose boundary
      has an orthogonal contact with the traced line color.
    - Flood-fill only those reachable cavities with the singleton's color.

    Key insight:
    - Three lookalikes must be separated: border singleton fakes, interior singleton fakes that
      connect only by diagonal bridges, and real interior seeds whose own 4-connected route truly
      reaches an orthogonally touched ring. Only that last case may color a cavity.
    """
    return _solve(
        grid,
        allow_border_candidates=False,
        reach_conn8=False,
        allow_diag_contact=False,
    )


# ========================== CFG ==========================
GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (14, 28)
BG_COLORS = [0, 5]

SEED_COUNT_CHOICES = [1]
SEED_MARGIN = 3
SEED_PLACEMENT_TRIES = 100
MIN_LINE_LEN = 6
LINE_LEN_RANGE = (11, 28)

FRAME_COLORS_MAX = 3
ORTHO_FRAMES_PER_SEED = (1, 1)
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 70
DIAG_ATTACH_TRIES = 140

ORTHO_WEIGHT_ALIGNED = 5
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.05
BIAS_FLIP_PROB = 0.35

GEN_RETRIES = 500
# =========================================================

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


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


def _ring_contact_mode(canvas: np.ndarray, ring: list[tuple[int, int]], line_col: int, h: int, w: int) -> str | None:
    has8 = False
    has4 = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
                has8 = True
                break
        for nr, nc in neighbors4(rr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
                has4 = True
                break
    if has4:
        return "ortho"
    if has8:
        return "diag_only"
    return None


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
    h: int,
    w: int,
    require_contact: str,
) -> bool:
    for rr, cc in ring:
        if int(canvas[rr, cc]) != -1:
            return False
    for rr, cc in cavity:
        if int(canvas[rr, cc]) != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
                return False

    ring_set = set(ring)
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return False

    mode = _ring_contact_mode(canvas, ring, line_col, h, w)
    if require_contact == "ortho":
        return mode == "ortho"
    if require_contact == "diag_only":
        return mode == "diag_only"
    raise ValueError(require_contact)


def _grow_line4(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    mid_r = (h - 1) / 2
    mid_c = (w - 1) / 2
    dr_bias = -1 if start[0] < mid_r else 1
    dc_bias = -1 if start[1] < mid_c else 1
    cur = start
    while len(path) < target_len:
        dirs = []
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
            if any(
                _in_bounds(h, w, nnr, nnc)
                and int(grid[nnr, nnc]) != -1
                and (nnr, nnc) != cur
                for nnr, nnc in neighbors8(nr, nc)
            ):
                continue
            if _on_border(nr, nc, h, w) and rng.random() < BORDER_AVOIDANCE_PROB:
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
            else:
                break
    return path


def _build_straight_path(
    canvas: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    h, w = canvas.shape
    path = [start]
    tr, tc = start
    for _ in range(length - 1):
        nr, nc = tr + direction[0], tc + direction[1]
        if not _in_bounds(h, w, nr, nc) or int(canvas[nr, nc]) != -1:
            break
        if any(
            _in_bounds(h, w, ar, ac)
            and int(canvas[ar, ac]) != -1
            and (ar, ac) != (tr, tc)
            for ar, ac in neighbors8(nr, nc)
        ):
            break
        canvas[nr, nc] = line_col
        path.append((nr, nc))
        tr, tc = nr, nc
    return path


def _try_extend_path_to_border(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    line_col: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    if _border_fake_candidates(grid, path):
        return path

    cur = path[-1]
    directions = sorted(
        ORTHO_DIRS,
        key=lambda d: min(
            abs((cur[0] + d[0]) - 1),
            abs((cur[0] + d[0]) - (h - 2)),
            abs((cur[1] + d[1]) - 1),
            abs((cur[1] + d[1]) - (w - 2)),
        ),
    )
    for ddr, ddc in directions:
        trial: list[tuple[int, int]] = []
        tr, tc = cur
        while True:
            nr, nc = tr + ddr, tc + ddc
            if not _in_bounds(h, w, nr, nc) or int(grid[nr, nc]) != -1:
                trial = []
                break
            if any(
                _in_bounds(h, w, nnr, nnc)
                and int(grid[nnr, nnc]) != -1
                and (nnr, nnc) != (tr, tc)
                for nnr, nnc in neighbors8(nr, nc)
            ):
                trial = []
                break
            trial.append((nr, nc))
            tr, tc = nr, nc
            if tr in (1, h - 2) or tc in (1, w - 2):
                break
        if not trial:
            continue
        for rr, cc in trial:
            grid[rr, cc] = line_col
            path.append((rr, cc))
        return path
    return path


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    *,
    require_contact: str,
    max_tries: int,
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
            if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w, require_contact):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_diag_frame_explicit(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = DIAG_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    corner_modes = ("nw", "ne", "sw", "se")
    for _ in range(max_tries):
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        pts = line_path[:]
        rng.shuffle(pts)
        for pr, pc in pts:
            modes = list(corner_modes)
            rng.shuffle(modes)
            for mode in modes:
                if mode == "nw":
                    r0, c0 = pr + 1, pc
                elif mode == "ne":
                    r0, c0 = pr + 1, pc - (cw + 1)
                elif mode == "sw":
                    r0, c0 = pr - (ch + 2), pc
                else:
                    r0, c0 = pr - (ch + 2), pc - (cw + 1)
                fh, fw = ch + 2, cw + 2
                if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                    continue
                ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
                if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w, "diag_only"):
                    continue
                for rr, cc in ring:
                    canvas[rr, cc] = frame_col
                return ring, cavity
    return None


def _pick_seed_and_first(canvas: np.ndarray, rng: random.Random, max_tries: int = SEED_PLACEMENT_TRIES) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = canvas.shape
    mid_r = (h - 1) / 2
    mid_c = (w - 1) / 2
    for _ in range(max_tries):
        candidates = []
        for r in range(SEED_MARGIN, h - SEED_MARGIN):
            for c in range(SEED_MARGIN, w - SEED_MARGIN):
                if int(canvas[r, c]) != -1:
                    continue
                clear = True
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        if int(canvas[rr, cc]) != -1:
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
        top_k = min(16, len(candidates))
        seed_rc = candidates[rng.randrange(top_k)][2]

        seed_dist = abs(seed_rc[0] - mid_r) + abs(seed_rc[1] - mid_c)
        first_choices = []
        for nr, nc in neighbors4(*seed_rc):
            if not _in_bounds(h, w, nr, nc) or int(canvas[nr, nc]) != -1:
                continue
            next_dist = abs(nr - mid_r) + abs(nc - mid_c)
            outward_rank = 0 if next_dist > seed_dist else 1
            first_choices.append((outward_rank, rng.random(), (nr, nc)))
        if first_choices:
            first_choices.sort()
            return seed_rc, first_choices[0][2]
    return None


def _pick_line_start(canvas: np.ndarray, rng: random.Random, margin: int = SEED_MARGIN, tries: int = SEED_PLACEMENT_TRIES) -> tuple[int, int] | None:
    h, w = canvas.shape
    for _ in range(tries):
        r = rng.randint(margin, h - margin - 1)
        c = rng.randint(margin, w - margin - 1)
        if int(canvas[r, c]) != -1:
            continue
        clear = True
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if int(canvas[rr, cc]) != -1:
                    clear = False
                    break
            if not clear:
                break
        if clear:
            return (r, c)
    return None


def _border_fake_candidates(canvas: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = canvas.shape
    cands = []
    for pr, pc in path:
        for sr, sc in neighbors4(pr, pc):
            if not _in_bounds(h, w, sr, sc) or not _on_border(sr, sc, h, w) or int(canvas[sr, sc]) != -1:
                continue
            occ4 = {(nr, nc) for nr, nc in neighbors4(sr, sc) if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) != -1}
            occ8 = {(nr, nc) for nr, nc in neighbors8(sr, sc) if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) != -1}
            if occ4 == {(pr, pc)} and occ8 == {(pr, pc)}:
                cands.append((sr, sc))
    return cands


def _border_fake_extensions(
    canvas: np.ndarray,
    path: list[tuple[int, int]],
    seed_rc: tuple[int, int],
) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
    h, w = canvas.shape
    cands: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
    anchors: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if len(path) >= 2:
        end = path[-1]
        prev = path[-2]
        anchors.append((end, (end[0] - prev[0], end[1] - prev[1])))
        start = path[0]
        anchors.append((start, (start[0] - seed_rc[0], start[1] - seed_rc[1])))
    else:
        for ddr, ddc in ORTHO_DIRS:
            anchors.append((path[-1], (ddr, ddc)))

    for (pr, pc), (ddr, ddc) in anchors:
        tr, tc = pr, pc
        prev = (pr, pc)
        chain: list[tuple[int, int]] = []
        while True:
            nr, nc = tr + ddr, tc + ddc
            if not _in_bounds(h, w, nr, nc) or int(canvas[nr, nc]) != -1:
                break
            if _on_border(nr, nc, h, w):
                occ4 = {(ar, ac) for ar, ac in neighbors4(nr, nc) if _in_bounds(h, w, ar, ac) and int(canvas[ar, ac]) != -1}
                occ8 = {(ar, ac) for ar, ac in neighbors8(nr, nc) if _in_bounds(h, w, ar, ac) and int(canvas[ar, ac]) != -1}
                expected = {prev} if not chain else set()
                if occ4 == expected and occ8 == expected:
                    cands.append((chain[:], (nr, nc)))
                break
            occ8 = {(ar, ac) for ar, ac in neighbors8(nr, nc) if _in_bounds(h, w, ar, ac) and int(canvas[ar, ac]) != -1}
            if any(p != prev for p in occ8):
                break
            chain.append((nr, nc))
            tr, tc = nr, nc
            prev = (nr, nc)
    return cands


def _place_border_fake_network(
    canvas: np.ndarray,
    seed_col: int,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
    tries: int = 80,
) -> bool:
    h, w = canvas.shape
    configs = [
        ((1, 0), lambda t: (0, t), lambda t: (1, t), range(3, w - 3)),
        ((-1, 0), lambda t: (h - 1, t), lambda t: (h - 2, t), range(3, w - 3)),
        ((0, 1), lambda t: (t, 0), lambda t: (t, 1), range(3, h - 3)),
        ((0, -1), lambda t: (t, w - 1), lambda t: (t, w - 2), range(3, h - 3)),
    ]
    for _ in range(tries):
        (ddr, ddc), seed_fn, first_fn, coord_range = rng.choice(configs)
        t = rng.choice(list(coord_range))
        seed_rc = seed_fn(t)
        first_rc = first_fn(t)
        if int(canvas[seed_rc]) != -1 or int(canvas[first_rc]) != -1:
            continue

        target_len = rng.randint(4, 8)
        path = [first_rc]
        tr, tc = first_rc
        ok = True
        while len(path) < target_len:
            nr, nc = tr + ddr, tc + ddc
            if not _in_bounds(h, w, nr, nc) or int(canvas[nr, nc]) != -1:
                ok = False
                break
            if any(
                _in_bounds(h, w, ar, ac)
                and int(canvas[ar, ac]) != -1
                and (ar, ac) != (tr, tc)
                for ar, ac in neighbors8(nr, nc)
            ):
                ok = False
                break
            path.append((nr, nc))
            tr, tc = nr, nc
        if not ok:
            continue

        for rr, cc in path:
            canvas[rr, cc] = line_col
        placed = _try_attach_frame(
            canvas,
            path,
            line_col,
            rng.choice(frame_colors_pool),
            rng,
            require_contact="ortho",
            max_tries=FRAME_ATTACH_TRIES,
        )
        if placed is not None:
            canvas[seed_rc] = seed_col
            return True
        for rr, cc in path:
            canvas[rr, cc] = -1
    return False


def _place_interior_fake_network(
    canvas: np.ndarray,
    seed_col: int,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
    tries: int = 100,
) -> bool:
    for _ in range(tries):
        start = _pick_line_start(canvas, rng)
        if start is None:
            return False
        if int(canvas[start]) != -1:
            continue

        canvas[start] = line_col
        path = _grow_line4(canvas, start, line_col, rng.randint(5, 9), rng)
        if len(path) < 4:
            for rr, cc in path:
                canvas[rr, cc] = -1
            continue

        placed = _try_attach_frame(
            canvas,
            path,
            line_col,
            rng.choice(frame_colors_pool),
            rng,
            require_contact="ortho",
            max_tries=FRAME_ATTACH_TRIES,
        )
        if placed is None:
            for rr, cc in path:
                canvas[rr, cc] = -1
            continue

        cands = _interior_fake_candidates(canvas, path)
        if cands:
            (br, bc), (sr, sc) = rng.choice(cands)
            canvas[br, bc] = line_col
            canvas[sr, sc] = seed_col
            return True

        ring, _cavity = placed
        for rr, cc in ring:
            canvas[rr, cc] = -1
        for rr, cc in path:
            canvas[rr, cc] = -1
    return False


def _interior_fake_candidates(canvas: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = canvas.shape
    cands: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for pr, pc in path:
        for dr, dc in DIAG_DIRS:
            br, bc = pr + dr, pc + dc
            if not _in_bounds(h, w, br, bc) or int(canvas[br, bc]) != -1 or _on_border(br, bc, h, w):
                continue
            occ4_bridge = {(nr, nc) for nr, nc in neighbors4(br, bc) if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) != -1}
            occ8_bridge = {(nr, nc) for nr, nc in neighbors8(br, bc) if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) != -1}
            if occ4_bridge:
                continue
            if occ8_bridge != {(pr, pc)}:
                continue
            for sr, sc in neighbors4(br, bc):
                if not _in_bounds(h, w, sr, sc) or int(canvas[sr, sc]) != -1 or _on_border(sr, sc, h, w):
                    continue
                occ4_seed = {(nr, nc) for nr, nc in neighbors4(sr, sc) if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) != -1}
                occ8_seed = {(nr, nc) for nr, nc in neighbors8(sr, sc) if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) != -1}
                if occ4_seed or occ8_seed:
                    continue
                cands.append(((br, bc), (sr, sc)))
    return cands


def _diagonal_bridge_candidates(canvas: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = canvas.shape
    cands: list[tuple[int, int]] = []
    for pr, pc in path:
        for dr, dc in DIAG_DIRS:
            br, bc = pr + dr, pc + dc
            if not _in_bounds(h, w, br, bc) or int(canvas[br, bc]) != -1 or _on_border(br, bc, h, w):
                continue
            occ4_bridge = {(nr, nc) for nr, nc in neighbors4(br, bc) if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) != -1}
            occ8_bridge = {(nr, nc) for nr, nc in neighbors8(br, bc) if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) != -1}
            if not occ4_bridge and occ8_bridge == {(pr, pc)}:
                cands.append((br, bc))
    return cands


def _place_true_eight_only_branch(
    canvas: np.ndarray,
    seed_rc: tuple[int, int],
    path: list[tuple[int, int]],
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
    tries: int = 80,
) -> bool:
    h, w = canvas.shape
    if len(path) < 2:
        return False

    anchors = [
        (path[-1], (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])),
        (path[0], (path[0][0] - seed_rc[0], path[0][1] - seed_rc[1])),
    ]
    rng.shuffle(anchors)
    for end, (ddr, ddc) in anchors:
        side_order = [-1, 1]
        rng.shuffle(side_order)
        for side in side_order:
            if (ddr, ddc) == (0, 1):
                bridge = (end[0] + side, end[1] + 1)
                branch = [bridge, (bridge[0], bridge[1] + 1), (bridge[0], bridge[1] + 2)]
                r0, c0 = (bridge[0] - 4, bridge[1]) if side == -1 else (bridge[0] + 1, bridge[1])
            elif (ddr, ddc) == (0, -1):
                bridge = (end[0] + side, end[1] - 1)
                branch = [bridge, (bridge[0], bridge[1] - 1), (bridge[0], bridge[1] - 2)]
                r0, c0 = (bridge[0] - 4, bridge[1] - 2) if side == -1 else (bridge[0] + 1, bridge[1] - 2)
            elif (ddr, ddc) == (1, 0):
                bridge = (end[0] + 1, end[1] + side)
                branch = [bridge, (bridge[0] + 1, bridge[1]), (bridge[0] + 2, bridge[1])]
                r0, c0 = (bridge[0], bridge[1] - 4) if side == -1 else (bridge[0], bridge[1] + 1)
            else:
                bridge = (end[0] - 1, end[1] + side)
                branch = [bridge, (bridge[0] - 1, bridge[1]), (bridge[0] - 2, bridge[1])]
                r0, c0 = (bridge[0] - 2, bridge[1] - 4) if side == -1 else (bridge[0] - 2, bridge[1] + 1)

            if any(not _in_bounds(h, w, rr, cc) or int(canvas[rr, cc]) != -1 or _on_border(rr, cc, h, w) for rr, cc in branch):
                continue
            occ4_bridge = {(nr, nc) for nr, nc in neighbors4(*bridge) if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) != -1}
            occ8_bridge = {(nr, nc) for nr, nc in neighbors8(*bridge) if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) != -1}
            if occ4_bridge or occ8_bridge != {end}:
                continue

            ok = True
            prev = end
            for rr, cc in branch:
                if any(
                    _in_bounds(h, w, ar, ac)
                    and int(canvas[ar, ac]) != -1
                    and (ar, ac) != prev
                    for ar, ac in neighbors8(rr, cc)
                ):
                    ok = False
                    break
                canvas[rr, cc] = line_col
                prev = (rr, cc)
            if not ok:
                for rr, cc in branch:
                    if int(canvas[rr, cc]) == line_col:
                        canvas[rr, cc] = -1
                continue

            ring, cavity = _frame_ring_coords(r0, c0, 2, 2)
            if _validate_frame_placement(canvas, ring, cavity, line_col, h, w, "ortho"):
                frame_col = rng.choice(frame_colors_pool)
                for rr, cc in ring:
                    canvas[rr, cc] = frame_col
                return True
            for rr, cc in branch:
                canvas[rr, cc] = -1
    return False


def _distribute_counts(total: int, bins: int, rng: random.Random) -> list[int]:
    out = [0] * bins
    for i in range(total):
        out[rng.randrange(bins)] += 1
    return out


def _gen_attempt(rng: random.Random, debug: bool = False) -> np.ndarray | tuple[None, str] | None:
    def fail(reason: str):
        if debug:
            return None, reason
        return None

    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_true = rng.choice(SEED_COUNT_CHOICES)
    min_colors_needed = num_true * 2 + 1 + 2
    if len(palette) < min_colors_needed:
        return fail("palette")

    canvas = -np.ones((h, w), dtype=int)
    seed_colors = [palette.pop() for _ in range(num_true)]
    line_colors = [palette.pop() for _ in range(num_true)]
    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette) - 2))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]
    fake_colors = palette[:]
    if len(fake_colors) < 2:
        return fail("fake_colors")

    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]] = []
    for i in range(num_true):
        picked = _pick_seed_and_first(canvas, rng)
        if picked is None:
            return fail("pick_seed")
        seed_rc, first_rc = picked
        if int(canvas[seed_rc]) != -1 or int(canvas[first_rc]) != -1:
            return fail("seed_occupied")
        dr, dc = first_rc[0] - seed_rc[0], first_rc[1] - seed_rc[1]
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(8, 12)
        path = _build_straight_path(canvas, first_rc, line_colors[i], (dr, dc), target_len)
        if len(path) < MIN_LINE_LEN:
            return fail("short_line")
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    for _seed_col, _seed_rc, line_col, path in lines_info:
        target_ortho = rng.randint(*ORTHO_FRAMES_PER_SEED)
        attached = 0
        for _ in range(target_ortho):
            fc = rng.choice(frame_colors_pool)
            placed = _try_attach_frame(
                canvas,
                path,
                line_col,
                fc,
                rng,
                require_contact="ortho",
                max_tries=FRAME_ATTACH_TRIES,
            )
            if placed is not None:
                attached += 1
        if attached == 0:
            return fail("no_ortho_frame")

    true_eight_only_count = 0
    shuffled_lines = lines_info[:]
    rng.shuffle(shuffled_lines)
    for _seed_col, seed_rc, line_col, path in shuffled_lines:
        if _place_true_eight_only_branch(
            canvas,
            seed_rc,
            path,
            line_col,
            frame_colors_pool,
            rng,
        ):
            true_eight_only_count += 1
            break
    if true_eight_only_count == 0:
        return fail("no_true_eight_only_branch")

    fake_colors = fake_colors[:]
    rng.shuffle(fake_colors)

    border_fake_count = 0
    interior_fake_count = 0

    if _place_border_fake_network(
        canvas,
        fake_colors.pop(),
        rng.choice(line_colors),
        frame_colors_pool,
        rng,
    ):
        border_fake_count = 1
    if border_fake_count == 0:
        return fail("no_border_fake")

    if _place_interior_fake_network(
        canvas,
        fake_colors.pop(),
        rng.choice(line_colors),
        frame_colors_pool,
        rng,
    ):
        interior_fake_count = 1
    if interior_fake_count == 0:
        return fail("no_interior_fake")

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return fail("bg_not_majority")

    for seed_col, _seed_rc, _line_col, _path in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return fail("seed_not_unique")

    metrics = _behavior_metrics(grid)
    if metrics["combo_true_candidates"] != num_true:
        return fail("combo_true_count")
    if metrics["border_reachable_if_allowed"] < 1:
        return fail("border_metric")
    if metrics["interior_8only_candidates"] < 1:
        return fail("interior_8only_metric")
    if metrics["diag_only_cavities_from_true"] < 1:
        return fail("true_8only_metric")
    try:
        out = main(grid)
    except Exception:
        return fail("main_exception")
    if np.array_equal(out, grid):
        return fail("main_no_change")
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
