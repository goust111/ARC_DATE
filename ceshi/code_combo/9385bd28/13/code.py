# Combo variant of 9385bd28 (index 13)
# combo_spec: D6/4+D8/0+D13/2
# Name: 迭代处理+擦除标记+保护当前条目标记

import sys
import random
from collections import Counter

import numpy as np

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   source_task: 9385bd28
#   fusion_core:
#     - D6/4 changes marker reading from "always inspect the original input" to "inspect the grid state after each
#       previous legend entry has already painted". This is a sequential chain effect, not a simple one-shot overlay.
#     - D8/0 keeps the legend but erases every original non-legend clue pixel at the very end. The erase pass must not
#       delete newly painted fill regions, even when those regions temporarily act as later markers.
#     - D13/2 changes protection from "protect all future markers" to "protect only the current entry's markers",
#       while also freezing every already-painted fill cell. So earlier fills win permanently, later fills cannot
#       repaint them, and future markers may be damaged unless the generator places them safely.
#   concrete_trace:
#     - Small example: legend rows bottom->top are A->B and B->C. Entry A uses an L-pair whose fill color is B.
#       Its painted B-rectangle sits in the top-left corner of a larger future B-bbox. The input only contains the
#       opposite BR L-corner of B. In the original task, B would be skipped or ambiguous because the initial input has
#       only one B corner. In this combo, after A paints, the current grid now contains a B-colored top-left corner
#       plus the old BR corner, so B becomes processable and paints a larger rectangle in color C.
#     - Because processed fills are protected, A's small B rectangle remains visible inside B's later C rectangle
#       instead of being overwritten. Because the final erase only removes original clue pixels, that small B rectangle
#       survives in the output even though it also served as B's temporary marker.
#   complexity_calibration:
#     - If arbitrary fills were allowed to match arbitrary marker colors, later marker detection would often become
#       ambiguous because a full rectangle can accidentally satisfy multiple L-corner tests. The generator therefore
#       uses one explicit chain pair and keeps every other fill color away from legend marker colors.
#     - If later bboxes overlapped heavily, "protect processed fills" would create cluttered holes that are hard to
#       parse. The generator reserves full bbox regions so only the intentional nested chain overlap remains.
#     - If all entries were noop/block/background-fill cases, the chain and erase effects would be invisible. The
#       generator guarantees one visible chain activation and at least one genuine rectangle paint.
#   hidden_assumptions_in_original_main:
#     - `marker_pts` previously came from the immutable input grid. That assumption is invalid now; the active source
#       must be the evolving `state`, otherwise chain-created markers never appear.
#     - Protecting future markers is no longer correct. Only `current marker pixels + legend + already painted fills`
#       can be protected, otherwise the fusion collapses back toward the original overlap rule.
#     - The old final state could simply be `out` because markers stayed visible. Here the terminal erase must target
#       only original non-legend clue pixels from the input `g`; erasing all current marker-colored pixels would also
#       destroy legitimate fill results created during chaining.
#     - Recoloring 2x2 blocks in place is incompatible with D8/0. Blocks remain temporary clues and vanish in the
#       final erase pass.
#   placement_legality:
#     - Every placed input pixel must stay in-bounds and avoid the legend scan corridor.
#     - Actual clue cells use an `occupied` mask; visible fill territories use a broader `region_reserved` mask so
#       future entries do not accidentally overlap them before their turn.
#     - The chain pair is the only intentional exception to independent bbox separation: a smaller painted rectangle is
#       nested inside the later entry's bbox so it can become one of that entry's two marker corners.
#   visual_observability:
#     - Human-solvers can see the lower legend row paints a small rectangle in exactly the same color as the next row's
#       marker chip. In the output, that small rectangle survives as a frozen island inside a larger later rectangle,
#       making the iterative dependency visible from only a few examples.
#     - The final erase is also obvious because all scattered L-corners and distractors disappear, leaving only the
#       legend plus painted regions.
#   visual_cleanliness:
#     - The generator keeps one deliberate nested chain and otherwise avoids bbox collisions, so the output reads as a
#       small rectangle seeding a larger one rather than a pile of mutually occluding fills.
#   color_inference:
#     - The only marker-color reuse is the explicit chain where one legend fill chip equals the next legend marker chip.
#       All other fill colors are chosen outside the marker set, so solvers can attribute the chain to visible legend
#       evidence instead of hidden numeric color identities.
#   new_solver_assumptions_to_match_in_generator:
#     - At least one later legend entry must become newly processable from the evolving state, not from the raw input.
#     - Any entry whose fill color equals a marker color must be the intentional chain entry; otherwise marker grouping
#       risks becoming ambiguous.
#     - Original input clue pixels are the only cells erased at the end.


_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs at bottom-left.
    Each legend row: (marker_color at col c, fill_color at col c+1).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, marker, fill).
    """
    h, w = g.shape
    start_r = None
    start_c = None
    for r in range(h - 1, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size:
            start_r = r
            start_c = int(nz[0])
            break
    assert start_r is not None and start_c is not None and start_c < w - 1, (
        "legend must exist at bottom-left with at least 2 columns"
    )

    entries: list[tuple[int, int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1]) if c + 1 < w else bg
        entries.append((r, marker, fill))
    return c, entries


def _l_anchor_present(
    pts: set[tuple[int, int]],
    r: int,
    c: int,
    arm1: tuple[int, int],
    arm2: tuple[int, int],
) -> bool:
    """
    Marker corners are L-shapes:
      TL: (r,c), (r,c+1), (r+1,c)
      TR: (r,c), (r,c-1), (r+1,c)
      BL: (r,c), (r,c+1), (r-1,c)
      BR: (r,c), (r,c-1), (r-1,c)
    """
    (dr1, dc1) = arm1
    (dr2, dc2) = arm2
    return (r, c) in pts and (r + dr1, c + dc1) in pts and (r + dr2, c + dc2) in pts


def _diamond_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """
    Mask for all cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neigh path between corners a and b:
      dist(a,p) + dist(p,b) == dist(a,b)
    """
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


def _legend_mask(h: int, w: int, legend_c: int, entries: list[tuple[int, int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, _, _ in entries:
        mask[r, legend_c] = True
        mask[r, legend_c + 1] = True
    return mask


def _marker_plan(scan: np.ndarray, marker: int, legend_mask: np.ndarray) -> dict | None:
    marker_mask = (scan == int(marker)) & (~legend_mask)
    pts_arr = np.argwhere(marker_mask)
    if len(pts_arr) < 2:
        return None

    rs = pts_arr[:, 0]
    cs = pts_arr[:, 1]
    rmin, rmax = int(rs.min()), int(rs.max())
    cmin, cmax = int(cs.min()), int(cs.max())

    if rmax - rmin == 1 and cmax - cmin == 1 and len(pts_arr) == 4:
        return {"kind": "block", "marker_mask": marker_mask}

    if rmin == rmax or cmin == cmax:
        return None

    pts = set(map(tuple, pts_arr.tolist()))
    tl = (rmin, cmin)
    tr = (rmin, cmax)
    bl = (rmax, cmin)
    br = (rmax, cmax)

    tl_anchor = _l_anchor_present(pts, tl[0], tl[1], (0, 1), (1, 0))
    tr_anchor = _l_anchor_present(pts, tr[0], tr[1], (0, -1), (1, 0))
    bl_anchor = _l_anchor_present(pts, bl[0], bl[1], (0, 1), (-1, 0))
    br_anchor = _l_anchor_present(pts, br[0], br[1], (0, -1), (-1, 0))

    score_tlbr = int(tl_anchor) + int(br_anchor)
    score_trbl = int(tr_anchor) + int(bl_anchor)
    if score_tlbr == 0 and score_trbl == 0:
        return None
    if score_tlbr == score_trbl:
        return None
    if score_tlbr > score_trbl:
        a, b = tr, bl
    else:
        a, b = tl, br

    fill_mask = _diamond_mask(scan.shape[0], scan.shape[1], rmin, rmax, cmin, cmax, a, b)
    return {
        "kind": "lpair",
        "marker_mask": marker_mask,
        "fill_mask": fill_mask,
    }


def _solve_variant(grid: np.ndarray, iterative: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"
    legend_mask = _legend_mask(h, w, legend_c, entries)

    state = g.copy()
    filled_mask = np.zeros_like(g, dtype=bool)

    for _, marker, fill in entries:
        scan = state if iterative else g
        plan = _marker_plan(scan, marker, legend_mask)
        if plan is None:
            continue
        if plan["kind"] == "block":
            continue

        protect = legend_mask | filled_mask | plan["marker_mask"]
        painted = plan["fill_mask"] & (~protect)
        state[painted] = int(fill)
        filled_mask |= painted

    temporary_mask = (g != bg) & (~legend_mask)
    state[temporary_mask] = bg
    return state


def _iterative_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    legend_mask = _legend_mask(h, w, legend_c, entries)

    state = g.copy()
    filled_mask = np.zeros_like(g, dtype=bool)
    activated_entries = 0
    gained_marker_pixels = 0

    for _, marker, fill in entries:
        base_mask = (g == int(marker)) & (~legend_mask)
        cur_mask = (state == int(marker)) & (~legend_mask)
        base_plan = _marker_plan(g, marker, legend_mask)
        cur_plan = _marker_plan(state, marker, legend_mask)

        if cur_plan is not None and cur_plan["kind"] == "lpair":
            cur_count = int(np.count_nonzero(cur_mask))
            base_count = int(np.count_nonzero(base_mask))
            if base_plan is None or cur_count > base_count:
                activated_entries += 1
                gained_marker_pixels += max(0, cur_count - base_count)

        if cur_plan is None or cur_plan["kind"] == "block":
            continue

        protect = legend_mask | filled_mask | cur_plan["marker_mask"]
        painted = cur_plan["fill_mask"] & (~protect)
        state[painted] = int(fill)
        filled_mask |= painted

    out = state.copy()
    out[(g != bg) & (~legend_mask)] = bg
    baseline = _solve_variant(g, iterative=False)
    return {
        "activated_entries": activated_entries,
        "gained_marker_pixels": gained_marker_pixels,
        "output_delta": int(np.count_nonzero(out != baseline)),
        "erased_input_pixels": int(np.count_nonzero((g != bg) & (~legend_mask) & (out == bg))),
        "visible_fill_pixels": int(np.count_nonzero((g == bg) & (~legend_mask) & (out != bg))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Process the legend on an evolving grid, then remove the original clue pixels.

    Input structure:
    - The input still has one dominant background color and a 2-column legend stacked near the bottom-left.
    - Legend marker colors appear outside the legend as L-corner groups, optional 2x2 blocks, or partial groups that
      only become complete after an earlier legend entry paints a same-colored region.
    - Distractor marker-shaped groups may appear in colors that are not referenced by the legend.

    Rule:
    - Read legend rows from bottom to top, but before each row re-scan that marker color on the current grid state
      rather than the original input.
    - If the current state yields a valid L-pair, fill its opposite-diagonal rectangle with the row's fill color while
      protecting the legend, all cells painted by earlier rows, and the current row's own marker pixels. Future marker
      pixels are not protected.
    - A 2x2 marker block stays a temporary clue and contributes no lasting paint in this combo.
    - After every legend row has had its turn, erase all original non-legend clue pixels back to the background color.

    Key insight: A fill from one row can become part of the next row's marker geometry. That chained recognition only
    works if you read markers from the evolving canvas, and the end-of-task erase only removes the original clue pixels,
    not the genuine fill regions they helped create.
    """
    return _solve_variant(grid, iterative=True)


# --------------- generator (combo) ---------------

GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (3, 4)

NOOP_PROB = 0.30
BLOCK_PROB = 0.30

BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.80, 0.10, 0.10]
TLBR_PROB = 0.75

DISTRACTOR_PROB = 0.15

BLOCK_RETRIES = 120
LPAIR_RETRIES = 180
CHAIN_RETRIES = 220
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 400


def _l_pair_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    if diag == "TLBR":
        anchors = [("TL", rmin, cmin), ("BR", rmax, cmax)]
    else:
        anchors = [("TR", rmin, cmax), ("BL", rmax, cmin)]
    cells: list[tuple[int, int]] = []
    for name, r, c in anchors:
        for dr, dc in _L_OFFSETS[name]:
            cells.append((r + dr, c + dc))
    return cells


def _single_l_cells(name: str, r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in _L_OFFSETS[name]]


def _l_pair_valid_bbox(h: int, w: int) -> bool:
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _pick_bbox_size(H: int, W: int, *, min_h: int = 2, min_w: int = 2) -> tuple[int, int] | None:
    shape_choice = random.choices(
        ["both_ge3", "h2_w_ge4", "h_ge4_w2"],
        weights=BBOX_SHAPE_WEIGHTS,
    )[0]
    if shape_choice == "both_ge3":
        bh_lo = max(3, min_h)
        bw_lo = max(3, min_w)
        bh_hi = min(BBOX_H_MAX, H)
        bw_hi = min(BBOX_W_MAX, W)
        if bh_lo > bh_hi or bw_lo > bw_hi:
            return None
        bh = random.randint(bh_lo, bh_hi)
        bw = random.randint(bw_lo, bw_hi)
    elif shape_choice == "h2_w_ge4":
        bh = 2
        bw_lo = max(4, min_w)
        bw_hi = min(BBOX_W_MAX, W)
        if min_h > 2 or bw_lo > bw_hi:
            return None
        bw = random.randint(bw_lo, bw_hi)
    else:
        bw = 2
        bh_lo = max(4, min_h)
        bh_hi = min(BBOX_H_MAX, H)
        if min_w > 2 or bh_lo > bh_hi:
            return None
        bh = random.randint(bh_lo, bh_hi)
    if bh > H or bw > W or not _l_pair_valid_bbox(bh, bw):
        return None
    return bh, bw


def _build_legend(H: int, W: int, bg: int):
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)

    fill_pool = [c for c in range(10) if c != bg and c not in marker_colors]
    if len(fill_pool) < max(1, n_entries - 1):
        return None

    fills = [marker_colors[1]]
    extra_fills = random.sample(fill_pool, n_entries - 1)
    fills.extend(extra_fills)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = fills[i]
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows):
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _can_place_cells(cells, H, W, occupied, forbidden) -> bool:
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if occupied[r, c] or forbidden[r, c]:
            return False
    return True


def _can_reserve_box(rmin: int, cmin: int, bh: int, bw: int, region_reserved: np.ndarray) -> bool:
    return not bool(region_reserved[rmin : rmin + bh, cmin : cmin + bw].any())


def _reserve_box(region_reserved: np.ndarray, rmin: int, cmin: int, bh: int, bw: int) -> None:
    region_reserved[rmin : rmin + bh, cmin : cmin + bw] = True


def _place_cells(grid: np.ndarray, occupied: np.ndarray, cells, color: int) -> None:
    for r, c in cells:
        grid[r, c] = color
        occupied[r, c] = True


def _try_place_regular_l_pair(
    H: int,
    W: int,
    color: int,
    occupied: np.ndarray,
    forbidden: np.ndarray,
    region_reserved: np.ndarray,
    grid: np.ndarray,
) -> bool:
    for _ in range(LPAIR_RETRIES):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        if not _can_reserve_box(rmin, cmin, bh, bw, region_reserved):
            continue
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
        if not _can_place_cells(cells, H, W, occupied, forbidden):
            continue
        _place_cells(grid, occupied, cells, color)
        _reserve_box(region_reserved, rmin, cmin, bh, bw)
        return True
    return False


def _try_place_block(
    H: int,
    W: int,
    color: int,
    occupied: np.ndarray,
    forbidden: np.ndarray,
    region_reserved: np.ndarray,
    grid: np.ndarray,
) -> bool:
    for _ in range(BLOCK_RETRIES):
        r0 = random.randint(0, H - 2)
        c0 = random.randint(0, W - 2)
        if not _can_reserve_box(r0, c0, 2, 2, region_reserved):
            continue
        cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
        if not _can_place_cells(cells, H, W, occupied, forbidden):
            continue
        _place_cells(grid, occupied, cells, color)
        _reserve_box(region_reserved, r0, c0, 2, 2)
        return True
    return False


def _try_place_chain_pair(
    H: int,
    W: int,
    color_a: int,
    color_b: int,
    occupied: np.ndarray,
    forbidden: np.ndarray,
    region_reserved: np.ndarray,
    grid: np.ndarray,
) -> bool:
    for _ in range(CHAIN_RETRIES):
        size_b = _pick_bbox_size(H, W, min_h=4, min_w=4)
        if size_b is None:
            continue
        bh_b, bw_b = size_b
        rmin_b = random.randint(0, H - bh_b)
        cmin_b = random.randint(0, W - bw_b)
        if not _can_reserve_box(rmin_b, cmin_b, bh_b, bw_b, region_reserved):
            continue

        max_ah = bh_b - 1
        max_aw = bw_b - 1
        if max_ah < 3 or max_aw < 3:
            continue
        ah = random.randint(3, max_ah)
        aw = random.randint(3, max_aw)
        if not _l_pair_valid_bbox(ah, aw):
            continue

        rmin_a = rmin_b
        cmin_a = cmin_b
        rmax_a = rmin_a + ah - 1
        cmax_a = cmin_a + aw - 1
        rmax_b = rmin_b + bh_b - 1
        cmax_b = cmin_b + bw_b - 1

        cells_a = _l_pair_cells(rmin_a, cmin_a, rmax_a, cmax_a, "TRBL")
        cells_b = _single_l_cells("BR", rmax_b, cmax_b)
        if not _can_place_cells(cells_a, H, W, occupied, forbidden):
            continue
        if not _can_place_cells(cells_b, H, W, occupied, forbidden):
            continue

        _place_cells(grid, occupied, cells_a, color_a)
        _place_cells(grid, occupied, cells_b, color_b)
        _reserve_box(region_reserved, rmin_b, cmin_b, bh_b, bw_b)
        return True
    return False


def _variant_signal_ok(grid: np.ndarray, out: np.ndarray, legend_mask: np.ndarray, bg: int) -> bool:
    metrics = _iterative_metrics(grid)
    if metrics["activated_entries"] < 1:
        return False
    if metrics["gained_marker_pixels"] < 4:
        return False
    if metrics["output_delta"] < 4:
        return False
    if metrics["visible_fill_pixels"] < 8:
        return False
    if metrics["erased_input_pixels"] < 6:
        return False
    temporary_mask = (grid != bg) & (~legend_mask)
    if int(np.count_nonzero(temporary_mask)) < 9:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()
    region_reserved = forbidden.copy()

    # Guarantee one chain: row 0 paints marker color of row 1, row 1 is initially only half-present.
    if not _try_place_chain_pair(
        H,
        W,
        marker_colors[0],
        marker_colors[1],
        occupied,
        forbidden,
        region_reserved,
        grid,
    ):
        return None

    n_entries = len(legend_rows)
    types = ["chain_seed", "chain_target"] + ["L"] * (n_entries - 2)
    idx_pool = list(range(2, n_entries))
    random.shuffle(idx_pool)
    if idx_pool and random.random() < NOOP_PROB:
        types[idx_pool.pop()] = "noop"
    if idx_pool and random.random() < BLOCK_PROB:
        types[idx_pool.pop()] = "block"

    for i in range(2, n_entries):
        marker = marker_colors[i]
        kind = types[i]
        if kind == "noop":
            continue
        if kind == "block":
            if not _try_place_block(H, W, marker, occupied, forbidden, region_reserved, grid):
                return None
            continue
        if not _try_place_regular_l_pair(H, W, marker, occupied, forbidden, region_reserved, grid):
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_regular_l_pair(
                H,
                W,
                random.choice(avail),
                occupied,
                forbidden,
                region_reserved,
                grid,
            )

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not _variant_signal_ok(grid, out, legend_mask, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(
        "generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES
    )
