# Combo variant of 8b7bacbf (index 2)
# combo_spec: D1/0+D9/3
# Name: 4连通追踪+部分开口框架
#
# Design analysis
# a) Concrete behavior and trace:
#    - Preserve the original object vocabulary: a unique border seed pixel identifies one
#      non-background network, that network reaches rectangular frame rings, and the inner
#      rectangle is flood-filled with the seed color.
#    - D1/0 changes the transport rule: tracing from the seed is 4-connected, so a frame that
#      only corner-touches the line is not owned by that seed even if it is visually nearby.
#    - D9/3 changes cavity recognition: attached frames may be missing 1-2 contiguous pixels on
#      a single side, yet their inner rectangle is still treated as enclosed.
#    - Small example:
#      1. Original solver: seed S reaches line L through any non-bg 8-path; only already-closed
#         background cavities are fillable.
#      2. Fused solver: S reaches only 4-neighbor-connected non-bg pixels; candidate frames are
#         inferred from same-color rectangular perimeters with at most one short side gap.
#      3. Difference: an orthogonally attached broken frame fills, while a diagonally touching
#         broken frame stays empty. Both look similar locally, so the sample must show both.
# b) Complexity calibration:
#    - Keep 1-2 seeds, repeated frame colors, multiple frames, and optional distractors so the
#      task is still about selecting the correct seed-owned structures rather than "fill any
#      broken rectangle".
#    - Restrict openings to 1-2 contiguous side cells and use only full rectangular rings. That
#      keeps the visual degree of freedom narrow and repeatable across 2-4 ARC examples.
#    - Each generated input contains at least one fillable open frame and one diagonal-only open
#      probe, so both control variables are active and observable.
# c) Implicit-assumption audit against the original main():
#    - Original `_nonbg_component8_from(...)` assumed diagonal transport is always legal. That is
#      invalid here; tracing must be configurable and `main()` uses 4-connectivity.
#    - Original cavity extraction from 4-connected background components assumed valid cavities are
#      literally closed in the input. That fails for open frames; the fused solver must infer the
#      intended frame from the colored perimeter instead.
#    - Original `neighbors8(...) == line_col` witness accepted diagonal line/frame contact. That
#      conflicts with D1/0, so the fused solver requires a 4-neighbor line witness for fillable
#      frames and only the legacy comparison mode allows diagonal witnessing.
#    - Rounded-corner rings from the original task are not 4-connected, so the frame geometry is
#      changed to full rectangles. This is a compatibility edit required by D1/0, not a separate
#      puzzle idea.
# d) Placement legality:
#    - Lines are 1px-wide orthogonal self-avoiding walks with 8-neighbor clearance, so different
#      branches do not accidentally merge.
#    - Attached frames are placed only when ring cells and cavity cells are empty, the cavity is
#      not 8-adjacent to the owning line, foreign-color 8-contacts are absent, and at least one
#      ring cell is 4-adjacent to the owning line.
#    - Diagonal probe frames use the same open-rectangle geometry but must have at least one
#      diagonal contact to the line and zero orthogonal contacts, ensuring the D1 difference.
# e) Visual observability:
#    - Human-solvable evidence comes from repeated filled broken rectangles plus at least one
#      visibly broken rectangle that only touches a line at a corner and therefore stays empty.
#    - The rule does not depend on numeric color ids. Colors only mark roles: seed color fills,
#      line color carries connectivity, frame color marks the inferred boundary.
# f) Visual cleanliness:
#    - The grid keeps a 1-cell buffer around frames and rejects foreign-color 8-adjacency near a
#      ring, preventing clutter that would make ownership ambiguous.
#    - Gaps are short and live on exactly one side, so the frame still reads as one object.
# g) New solver assumptions and generator guarantees:
#    - Assumption: each seed has a 4-neighbor non-background continuation. Guaranteed by corner
#      seeding with the first line pixel placed orthogonally.
#    - Assumption: valid frames are axis-aligned rectangles whose visible perimeter equals the
#      ideal ring minus 0-2 contiguous side cells. Guaranteed by generator construction.
#    - Assumption: any frame `main()` should fill is 4-connected to the seed network and has a
#      4-neighbor witness to that seed's line color. Guaranteed by attached-frame placement.
#    - Assumption: at least one open frame would be filled by an 8-connected open-frame solver but
#      not by the fused solver. Guaranteed by explicit diagonal-only open probe placement.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    connectivity = 8 if conn8 else 4
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=connectivity)]


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


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _is_orth_neighbor(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


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
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (ring, cavity) for a full rectangular frame."""
    fh, fw = ch + 2, cw + 2
    ring = []
    for dr in range(fh):
        for dc in range(fw):
            if dr == 0 or dr == fh - 1 or dc == 0 or dc == fw - 1:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _frame_sides(r0: int, c0: int, ch: int, cw: int) -> list[list[tuple[int, int]]]:
    return [
        [(r0, c0 + dc) for dc in range(1, cw + 1)],
        [(r0 + ch + 1, c0 + dc) for dc in range(1, cw + 1)],
        [(r0 + dr, c0) for dr in range(1, ch + 1)],
        [(r0 + dr, c0 + cw + 1) for dr in range(1, ch + 1)],
    ]


def _is_single_short_opening(missing: set[tuple[int, int]], r0: int, c0: int, ch: int, cw: int) -> bool:
    if not missing:
        return True
    if len(missing) > 2:
        return False
    for side in _frame_sides(r0, c0, ch, cw):
        idxs = [i for i, p in enumerate(side) if p in missing]
        if len(idxs) != len(missing):
            continue
        if idxs and idxs[-1] - idxs[0] + 1 == len(idxs):
            return True
    return False


def _component_has_line_contact(g: np.ndarray, pts: set[tuple[int, int]], line_col: int, diag_ok: bool) -> bool:
    h, w = g.shape
    neigh_fn = neighbors8 if diag_ok else neighbors4
    for r, c in pts:
        for nr, nc in neigh_fn(r, c):
            if _in_bounds(h, w, nr, nc) and int(g[nr, nc]) == int(line_col):
                return True
    return False


def _frame_candidate_from_component(
    g: np.ndarray,
    pts: list[tuple[int, int]],
    bg: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int] | None:
    pts_set = set(pts)
    if len(pts_set) < 8:
        return None

    rows = [r for r, _ in pts_set]
    cols = [c for _, c in pts_set]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    ch = r1 - r0 - 1
    cw = c1 - c0 - 1
    if ch < 2 or cw < 2:
        return None

    ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
    ring_set = set(ring)
    if not pts_set <= ring_set:
        return None

    missing = ring_set - pts_set
    if not _is_single_short_opening(missing, r0, c0, ch, cw):
        return None

    for r, c in missing:
        if int(g[r, c]) != int(bg):
            return None
    for r, c in cavity:
        if int(g[r, c]) != int(bg):
            return None

    return set(cavity), pts_set, len(missing)


def _all_frame_candidates(g: np.ndarray, bg: int) -> list[tuple[int, set[tuple[int, int]], set[tuple[int, int]], int]]:
    candidates = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        for comp in _components_mask(g == int(col), conn8=False):
            found = _frame_candidate_from_component(g, comp, bg)
            if found is None:
                continue
            cavity, frame_pts, missing_count = found
            candidates.append((int(col), cavity, frame_pts, int(missing_count)))
    return candidates


def _closed_cavity_candidates(
    g: np.ndarray,
    bg: int,
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    cavities = []
    for pts in bg_comps:
        region = set(pts)
        if any(p in region for p in corners if int(g[p]) == int(bg)):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != int(bg):
                    boundary.add((nr, nc))
        if not boundary:
            continue
        bcols = {int(g[p]) for p in boundary}
        cavities.append((region, boundary, bcols))
    return cavities


def _solve_with_modes(
    grid: np.ndarray,
    *,
    trace_conn8: bool,
    allow_open_frames: bool,
    line_adj8: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    frame_candidates = _all_frame_candidates(g, bg) if allow_open_frames else None
    closed_cavities = None if allow_open_frames else _closed_cavity_candidates(g, bg)

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])
        neigh_cols = []
        for nr, nc in neighbors4(sr, sc):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        assert neigh_cols, "seed pixel must have at least one non-bg 4-neighbor"
        line_col = Counter(neigh_cols).most_common(1)[0][0]
        traced = _nonbg_component_from(g, (sr, sc), bg, conn8=trace_conn8)

        if allow_open_frames:
            assert frame_candidates is not None
            for _frame_col, cavity, frame_pts, _missing_count in frame_candidates:
                if not frame_pts <= traced:
                    continue
                if not _component_has_line_contact(g, frame_pts, line_col, diag_ok=line_adj8):
                    continue
                for r, c in cavity:
                    if int(out[r, c]) == bg:
                        out[r, c] = seed_col
        else:
            assert closed_cavities is not None
            for cavity, boundary, bcols in closed_cavities:
                if len(bcols) != 1:
                    continue
                boundary_in_traced = boundary & traced
                if not boundary_in_traced:
                    continue

                ok = False
                neigh_fn = neighbors8 if line_adj8 else neighbors4
                for r, c in boundary_in_traced:
                    for nr, nc in neigh_fn(r, c):
                        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                            ok = True
                            break
                    if ok:
                        break
                if not ok:
                    continue

                for r, c in cavity:
                    if int(out[r, c]) == bg:
                        out[r, c] = seed_col

    return out


def _activation_signature(grid: np.ndarray) -> dict[str, int | list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frame_candidates = _all_frame_candidates(g, bg)

    fillable_open_frames = 0
    diag_only_open_frames = 0
    per_seed_fillable_frames: list[int] = []

    for sr, sc in _seed_pixels(g, bg):
        neigh_cols = []
        for nr, nc in neighbors4(sr, sc):
            if _in_bounds(*g.shape, nr, nc):
                v = int(g[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        if not neigh_cols:
            per_seed_fillable_frames.append(0)
            continue
        line_col = Counter(neigh_cols).most_common(1)[0][0]
        traced4 = _nonbg_component_from(g, (sr, sc), bg, conn8=False)
        traced8 = _nonbg_component_from(g, (sr, sc), bg, conn8=True)

        seed_fillable = 0
        for _frame_col, _cavity, frame_pts, missing_count in frame_candidates:
            owned4 = frame_pts <= traced4 and _component_has_line_contact(g, frame_pts, line_col, diag_ok=False)
            owned8 = frame_pts <= traced8 and _component_has_line_contact(g, frame_pts, line_col, diag_ok=True)
            if owned4:
                seed_fillable += 1
                if missing_count > 0:
                    fillable_open_frames += 1
            if owned8 and not owned4 and missing_count > 0:
                diag_only_open_frames += 1
        per_seed_fillable_frames.append(seed_fillable)

    new_out = main(g)
    closed4_out = _solve_with_modes(g, trace_conn8=False, allow_open_frames=False, line_adj8=False)
    open8_out = _solve_with_modes(g, trace_conn8=True, allow_open_frames=True, line_adj8=True)
    return {
        "fillable_open_frames": int(fillable_open_frames),
        "diag_only_open_frames": int(diag_only_open_frames),
        "d9_delta_cells": int(np.count_nonzero(new_out != closed4_out)),
        "d1_delta_cells": int(np.count_nonzero(open8_out != new_out)),
        "filled_cells": int(np.count_nonzero(new_out != g)),
        "per_seed_fillable_frames": per_seed_fillable_frames,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the inner rectangles of seed-owned near-closed frames.

    Input structure:
    - The background is the most frequent color.
    - One or two unique seed pixels sit on the border, each touching an orthogonal line.
    - The lines are 1-pixel-wide 4-connected paths, and nearby rectangular frame rings may have
      a 1-2 pixel opening on one side.

    Rule:
    - Starting from each border seed, trace only the 4-connected non-background component.
    - Treat a monochrome rectangular perimeter as a valid frame if its full rectangle is present
      except for at most one short contiguous side gap and its inner rectangle is still background.
    - Fill the interior of every such frame that belongs to the traced component and has a
      4-neighbor contact with that seed's line color.

    Key insight:
    - The task is no longer "find already sealed background holes". You must infer intended
      closure from a nearly complete rectangular frame, while still respecting strict 4-connected
      ownership from the seed so corner-touching broken frames do not count.
    """
    return _solve_with_modes(grid, trace_conn8=False, allow_open_frames=True, line_adj8=False)


# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 28)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 3, 3, 4]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3, 4]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 50
DISTRACTOR_FRAME_TRIES = 40
DIAG_PROBE_TRIES = 60
ATTACHED_GAP_WEIGHTS = [(1, 4), (2, 3)]
PROBE_GAP_WEIGHTS = [(1, 4), (2, 4)]
DISTRACTOR_GAP_WEIGHTS = [(0, 2), (1, 3), (2, 2)]

STRAIGHT_WEIGHT = 4
TURN_WEIGHT = 3
INWARD_WEIGHT = 2
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_PROB = 0.5
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (5, 12)
DISTRACTOR_SOLO_MAX = 1
DISTRACTOR_NETWORK_FRAMES = (1, 2)

GEN_RETRIES = 400
# =========================================================

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _weighted_choice(weighted_values: list[tuple[int, int]], rng: random.Random) -> int:
    bag = []
    for value, weight in weighted_values:
        bag.extend([int(value)] * int(weight))
    return int(rng.choice(bag))


def _adjacent_to_color(canvas: np.ndarray, cell: tuple[int, int], color: int, diag_ok: bool) -> bool:
    h, w = canvas.shape
    neigh_fn = neighbors8 if diag_ok else neighbors4
    r, c = cell
    for nr, nc in neigh_fn(r, c):
        if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == int(color):
            return True
    return False


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Grow a meandering 4-connected orthogonal path."""
    h, w = grid.shape
    path = [start]
    path_set = {start}
    cur = start
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    last_dir: tuple[int, int] | None = None

    while len(path) < target_len:
        weighted_dirs = []
        for ddr, ddc in ORTHO_DIRS:
            if last_dir is not None and (ddr, ddc) == (-last_dir[0], -last_dir[1]):
                continue
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue

            weight = TURN_WEIGHT
            if last_dir == (ddr, ddc):
                weight += STRAIGHT_WEIGHT
            if ddr == dr_bias or ddc == dc_bias:
                weight += INWARD_WEIGHT
            weighted_dirs.extend([(ddr, ddc)] * weight)

        if not weighted_dirs:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            break

        ddr, ddc = rng.choice(weighted_dirs)
        cur = (cur[0] + ddr, cur[1] + ddc)
        path.append(cur)
        path_set.add(cur)
        grid[cur] = line_col
        last_dir = (ddr, ddc)

    return path


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
    h: int,
    w: int,
    *,
    contact_mode: str,
) -> bool:
    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == int(line_col):
                return False

    ring_set = set(ring)
    has_orth_touch = False
    has_diag_touch = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = int(canvas[nr, nc])
            if v == -1:
                continue
            if v == int(line_col):
                if _is_orth_neighbor((rr, cc), (nr, nc)):
                    has_orth_touch = True
                else:
                    has_diag_touch = True
                continue
            return False

    if contact_mode == "orth":
        return has_orth_touch
    if contact_mode == "diag_only":
        return has_diag_touch and not has_orth_touch
    raise ValueError(contact_mode)


def _pick_gap_cells(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    r0: int,
    c0: int,
    ch: int,
    cw: int,
    line_col: int,
    weighted_lengths: list[tuple[int, int]],
    rng: random.Random,
    *,
    contact_diag_ok: bool,
) -> list[tuple[int, int]] | None:
    contact_cells = {cell for cell in ring if _adjacent_to_color(canvas, cell, line_col, diag_ok=contact_diag_ok)}
    if not contact_cells:
        return None

    lengths = []
    for gap_len, weight in weighted_lengths:
        lengths.extend([int(gap_len)] * int(weight))
    rng.shuffle(lengths)

    for gap_len in lengths:
        if gap_len == 0:
            return []
        segments = []
        for side in _frame_sides(r0, c0, ch, cw):
            if len(side) < gap_len:
                continue
            for i in range(len(side) - gap_len + 1):
                seg = side[i : i + gap_len]
                if any(cell in contact_cells for cell in seg):
                    continue
                if not any(cell not in seg for cell in contact_cells):
                    continue
                segments.append(seg)
        if segments:
            return list(rng.choice(segments))
    return None


def _try_place_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    *,
    contact_mode: str,
    gap_weights: list[tuple[int, int]],
    max_tries: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    contact_diag_ok = contact_mode == "diag_only"
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
            if not _validate_frame_placement(
                canvas,
                ring,
                cavity,
                line_col,
                h,
                w,
                contact_mode=contact_mode,
            ):
                continue
            gap_cells = _pick_gap_cells(
                canvas,
                ring,
                r0,
                c0,
                ch,
                cw,
                line_col,
                gap_weights,
                rng,
                contact_diag_ok=contact_diag_ok,
            )
            if gap_cells is None:
                continue
            gap_set = set(gap_cells)
            kept_ring = [cell for cell in ring if cell not in gap_set]
            for rr, cc in kept_ring:
                canvas[rr, cc] = frame_col
            return kept_ring, cavity, gap_cells
    return None


def _try_attach_distractor_frame(
    canvas: np.ndarray,
    frame_col: int,
    rng: random.Random,
    *,
    gap_weights: list[tuple[int, int]],
    max_tries: int = DISTRACTOR_FRAME_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    for _ in range(max_tries):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        if h - fh - 1 <= 1 or w - fw - 1 <= 1:
            return None
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

        ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
        gap_len = _weighted_choice(gap_weights, rng)
        gap_cells: list[tuple[int, int]] = []
        if gap_len > 0:
            segments = []
            for side in _frame_sides(r0, c0, ch, cw):
                if len(side) < gap_len:
                    continue
                for i in range(len(side) - gap_len + 1):
                    segments.append(side[i : i + gap_len])
            if not segments:
                continue
            gap_cells = list(rng.choice(segments))

        gap_set = set(gap_cells)
        kept_ring = [cell for cell in ring if cell not in gap_set]
        for rr, cc in kept_ring:
            canvas[rr, cc] = frame_col
        return kept_ring, cavity, gap_cells
    return None


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
        target_len = rng.randint(*DISTRACTOR_LINE_LEN)
        path = _grow_line(canvas, (r, c), line_col, target_len, rng)
        if len(path) < 3:
            for pr, pc in path:
                canvas[pr, pc] = -1
            continue
        attached = 0
        for _ in range(rng.randint(*DISTRACTOR_NETWORK_FRAMES)):
            fc = rng.choice(frame_colors_pool)
            if _try_place_frame(
                canvas,
                path,
                line_col,
                fc,
                rng,
                contact_mode="orth",
                gap_weights=DISTRACTOR_GAP_WEIGHTS,
                max_tries=FRAME_ATTACH_TRIES,
            ) is not None:
                attached += 1
        if attached > 0:
            return True
        for pr, pc in path:
            canvas[pr, pc] = -1
    return False


def _corner_config(
    corner_kind: str,
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        seed = (0, 0)
        first = rng.choice([(0, 1), (1, 0)])
    elif corner_kind == "TR":
        seed = (0, w - 1)
        first = rng.choice([(0, w - 2), (1, w - 1)])
    elif corner_kind == "BL":
        seed = (h - 1, 0)
        first = rng.choice([(h - 1, 1), (h - 2, 0)])
    elif corner_kind == "BR":
        seed = (h - 1, w - 1)
        first = rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    else:
        raise ValueError(corner_kind)
    return seed, first


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


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if num_seeds * 2 > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    lines_info = []
    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w, rng)
        if canvas[seed_rc] != -1 or canvas[first_rc] != -1:
            return None
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        max_len = min(LINE_LEN_RANGE[1], max(LINE_LEN_RANGE[0], (h * w) // 5))
        target_len = rng.randint(LINE_LEN_RANGE[0], max_len)
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    total_loops_target = rng.randint(max(num_seeds + 1, 3), TOTAL_LOOPS_MAX)
    num_attached_target = rng.randint(num_seeds, total_loops_target - 1)
    num_distractors_target = total_loops_target - num_attached_target
    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)

    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            fc = rng.choice(frame_colors_pool)
            placed = _try_place_frame(
                canvas,
                path,
                line_col,
                fc,
                rng,
                contact_mode="orth",
                gap_weights=ATTACHED_GAP_WEIGHTS,
                max_tries=FRAME_ATTACH_TRIES,
            )
            if placed is not None:
                attached += 1
        if attached == 0:
            return None

    probe_owner = rng.randrange(num_seeds)
    probe_ok = _try_place_frame(
        canvas,
        lines_info[probe_owner][3],
        lines_info[probe_owner][2],
        rng.choice(frame_colors_pool),
        rng,
        contact_mode="diag_only",
        gap_weights=PROBE_GAP_WEIGHTS,
        max_tries=DIAG_PROBE_TRIES,
    )
    if probe_ok is None:
        return None

    extra_distractors = max(0, num_distractors_target - 1)
    if extra_distractors > 0:
        if rng.random() < DISTRACTOR_NETWORK_PROB:
            _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
            extra_distractors -= 1
        for _ in range(min(extra_distractors, DISTRACTOR_SOLO_MAX)):
            _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng, gap_weights=DISTRACTOR_GAP_WEIGHTS)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != int(bg):
        return None

    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    try:
        sig = _activation_signature(grid)
    except Exception:
        return None

    if int(sig["filled_cells"]) == 0:
        return None
    if int(sig["fillable_open_frames"]) == 0:
        return None
    if int(sig["diag_only_open_frames"]) == 0:
        return None
    if int(sig["d9_delta_cells"]) == 0:
        return None
    if int(sig["d1_delta_cells"]) == 0:
        return None
    if any(int(v) == 0 for v in sig["per_seed_fillable_frames"]):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
