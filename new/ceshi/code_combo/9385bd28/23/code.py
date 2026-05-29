# Combo variant of 9385bd28 (index 23)
# combo_spec: D11/0+D15/0+D16/0
# Name: 无干扰+小网格+禁止填充色等于背景色

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _solver_dir = _parent / "solvers"
    if (_solver_dir / "arc_utils.py").exists():
        if str(_solver_dir) not in sys.path:
            sys.path.append(str(_solver_dir))
        break
else:
    raise RuntimeError("could not locate solvers/arc_utils.py from combo variant path")

from arc_utils import get_bg


# Design analysis for combo D11/0 + D15/0 + D16/0
# a) Concretization and behavior trace:
#    Base task: read the bottom-left legend, gather each marker color outside the legend,
#    infer which bbox diagonal contains the two L-corners, then fill the opposite-diagonal
#    diamond while protecting legend cells and later markers.
#    Fused variant: keep exactly that geometric transform, but tighten three meta degrees
#    of freedom simultaneously:
#    - no distractor colors at all,
#    - the whole canvas shrinks to about 10x10..12x12,
#    - every legend fill color differs from the background so each fill stays visible.
#    Concrete trace:
#    - Original sample possibility: bg=0, legend has 2->0 and 3->7, plus an unrelated
#      color-8 L-pair elsewhere. Processing color 2 may "fill" with bg and become visually
#      invisible, while color 8 must be ignored.
#    - Fused sample: bg=0, legend may have 2->6 and 3->7, and every non-legend foreground
#      pixel is either color 2 or color 3. The solver still reads the same diagonals, but
#      now both outputs are visible recolors and there is no mystery color to filter out.
#    - Because the grid is only 10-12 in each dimension, the placed L-pair bboxes are forced
#      to live close together; at least one pair of bboxes overlaps in area, so the compact
#      layout is not just a size cap but an actually visible crowding effect.
# b) Complexity calibration:
#    - Keeping the original L-pair rule avoids inventing a new geometry family just to make
#      the small grid noticeable.
#    - Distractors are removed, but multiple legend entries remain and their bboxes are packed
#      together, so the task does not collapse into a single isolated rectangle.
#    - The generator emits only active L-pairs in this combo. On a 10-12 grid, no-op rows or
#      rare 2x2 blocks would consume scarce sample budget without reinforcing the fused cue.
# c) Hidden-assumption audit of `main()`:
#    - `_find_legend(...)` assumes the first lower-left non-bg pixels are the legend. The
#      generator still reserves the scan corridor at and left of the legend column.
#    - `marker_pts = all pixels of one legend color outside the legend` assumes a legend color
#      does not also label unrelated clutter. D11/0 makes this stronger by banning distractors.
#    - L-anchor scoring assumes bbox shapes avoid the ambiguous (2,2), (2,3), (3,2) cases.
#      The generator still rejects those bbox sizes.
#    - The fill operation assumes later markers must survive earlier fills. That remains true,
#      and compact placement makes that protection behavior more relevant, not less.
#    - The fused variant adds one new semantic expectation: legend fill colors should not equal
#      the background. The generator enforces it directly; `main()` does not need color-id logic.
# d) Placement legality:
#    - Marker pixels never overlap each other or the legend.
#    - All marker cells stay in bounds, and the forbidden mask protects legend parsing.
#    - Compactness is achieved by steering later bboxes near the current cluster and requiring
#      at least one pair of active bboxes to intersect in area.
# e) Visual observability:
#    - A human now sees only legend-explained colors, in a visibly small canvas, with every
#      fill region appearing in a non-background color. Those are all directly inspectable.
#    - Because at least two bboxes overlap, the small-grid constraint has repeated structural
#      evidence rather than being an invisible numeric range change.
# f) Visual cleanliness:
#    - Even though bboxes overlap, only marker pixels are collision-forbidden, so shapes remain
#      individually legible while their future fills can interact through the normal overwrite rule.
#    - Fill colors avoid the background, which removes "did anything happen?" ambiguity.
# g) Change inferability:
#    - No behavior depends on hidden color ordering. The legend still fully explains color mapping.
#    - The fused changes are inferable from 2-4 examples because every sample repeatedly shows
#      the same three constraints: no stray colors, small dense layout, visible fills.


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


def _build_legend_mask(
    shape: tuple[int, int],
    legend_c: int,
    entries: list[tuple[int, int, int]],
) -> np.ndarray:
    h, w = shape
    legend_mask = np.zeros((h, w), dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True
    return legend_mask


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

    This is more robust than checking only corner membership, especially on tight bboxes
    where one arm can touch another bbox corner.
    """
    dr1, dc1 = arm1
    dr2, dc2 = arm2
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
    Mask for all cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neigh path
    between corners a and b.
    """
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    mask = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                mask[r, c] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor tightly packed legend-coded L-pairs on a small decoy-free canvas.

    Input structure:
    - The grid has one dominant background color and a two-column legend stacked near the bottom-left;
      each legend row is `(marker_color, fill_color)` and rows are read from bottom to top.
    - Outside the legend, every foreground pixel belongs to one of those legend marker colors; this
      fused variant removes non-legend-color distractors and packs the active L-pairs into a small grid.
    - Each active marker color appears as one opposite-corner L-pair whose bbox is close to the others.

    Rule:
    - For each legend row, gather that marker color's non-legend pixels and infer which bbox diagonal
      carries the two L anchors.
    - Fill the opposite diagonal's Manhattan diamond with the legend fill color while preserving the
      legend itself and all marker pixels needed by later legend rows.
    - In generated variant instances the fill color is never the background, so every fill stays visible.

    Key insight:
    - The L-corners mark one diagonal, but the paint goes on the other diagonal's shortest-path diamond.
      Because there are no decoy colors in this variant, the real work is reading each tight bbox and
      applying the legend's visible color mapping in the correct bottom-to-top order.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = _build_legend_mask((h, w), legend_c, entries)
    out = g.copy()
    marker_order = [marker for _, marker, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for future_marker in future_markers:
            protect |= (g == future_marker) & (~legend_mask)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            block = np.zeros_like(g, dtype=bool)
            for r, c in marker_pts:
                block[r, c] = True
            out[block & (~protect)] = int(fill)
            continue

        protect |= (g == marker) & (~legend_mask)
        if rmin == rmax or cmin == cmax:
            continue

        tl = (rmin, cmin)
        tr = (rmin, cmax)
        bl = (rmax, cmin)
        br = (rmax, cmax)

        tl_anchor = _l_anchor_present(marker_pts, tl[0], tl[1], (0, 1), (1, 0))
        tr_anchor = _l_anchor_present(marker_pts, tr[0], tr[1], (0, -1), (1, 0))
        bl_anchor = _l_anchor_present(marker_pts, bl[0], bl[1], (0, 1), (-1, 0))
        br_anchor = _l_anchor_present(marker_pts, br[0], br[1], (0, -1), (-1, 0))

        score_tlbr = int(tl_anchor) + int(br_anchor)
        score_trbl = int(tr_anchor) + int(bl_anchor)
        assert score_tlbr != 0 or score_trbl != 0, "at least one diagonal must have an L-anchor"
        assert score_tlbr != score_trbl, "diagonal scores must not tie"

        if score_tlbr > score_trbl:
            a, b = tr, bl
        else:
            a, b = tl, br

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill)

    return out


# --------------- generator (compact no-distractor visible-fill variant) ---------------

GRID_H_RANGE = (10, 12)
GRID_W_RANGE = (10, 12)

LEGEND_COL_RANGE = (0, 1)
LEGEND_ENTRY_RANGE = (2, 4)

BBOX_H_MAX = 7
BBOX_W_MAX = 7
BBOX_SHAPE_WEIGHTS = [0.85, 0.10, 0.05]  # [both>=3, h=2/w>=4, h>=4/w=2]

LPAIR_RETRIES = 220
GEN_RETRIES = 500

MAX_ACTIVE_BBOX_RATIO = 0.62
MIN_OVERLAP_PAIRS = 1

_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _l_pair_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    """Return the 6 cells for an L-pair at opposite bbox corners. diag in {'TLBR','TRBL'}."""
    if diag == "TLBR":
        anchors = [("TL", rmin, cmin), ("BR", rmax, cmax)]
    else:
        anchors = [("TR", rmin, cmax), ("BL", rmax, cmin)]
    cells: list[tuple[int, int]] = []
    for name, r, c in anchors:
        for dr, dc in _L_OFFSETS[name]:
            cells.append((r + dr, c + dc))
    return cells


def _l_pair_valid_bbox(h: int, w: int) -> bool:
    """True if (h, w) avoids an L-anchor score tie."""
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    """Pick a random valid L-pair bbox size that fits in (H, W)."""
    shape_choice = random.choices(
        ["both_ge3", "h2_w_ge4", "h_ge4_w2"],
        weights=BBOX_SHAPE_WEIGHTS,
    )[0]
    if shape_choice == "both_ge3":
        bh = random.randint(3, min(BBOX_H_MAX, H))
        bw = random.randint(3, min(BBOX_W_MAX, W))
    elif shape_choice == "h2_w_ge4":
        bh = 2
        w_hi = min(BBOX_W_MAX, W)
        if w_hi < 4:
            return None
        bw = random.randint(4, w_hi)
    else:
        h_hi = min(BBOX_H_MAX, H)
        if h_hi < 4:
            return None
        bh = random.randint(4, h_hi)
        bw = 2
    if bh > H or bw > W or not _l_pair_valid_bbox(bh, bw):
        return None
    return bh, bw


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _bboxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _union_bbox(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not bboxes:
        return None
    return (
        min(r0 for r0, _, _, _ in bboxes),
        min(c0 for _, c0, _, _ in bboxes),
        max(r1 for _, _, r1, _ in bboxes),
        max(c1 for _, _, _, c1 in bboxes),
    )


def _planned_diagonals(n_entries: int) -> list[str]:
    """
    Mix both diagonal branches inside each sample so the compact variant does not
    collapse into a single repeatedly-seen orientation.
    """
    n_tlbr = (n_entries + 1) // 2
    diags = ["TLBR"] * n_tlbr + ["TRBL"] * (n_entries - n_tlbr)
    random.shuffle(diags)
    return diags


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
    existing_bboxes: list[tuple[int, int, int, int]],
    require_overlap: bool,
    diag: str,
) -> tuple[int, int, int, int] | None:
    """Try to place one compact L-pair and return its bbox."""
    focus_bbox = _union_bbox(existing_bboxes)
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size

        if focus_bbox is None:
            r_lo, r_hi = 0, H - bh
            c_lo, c_hi = 0, W - bw
        else:
            fr0, fc0, fr1, fc1 = focus_bbox
            r_lo = max(0, fr0 - 2)
            r_hi = min(H - bh, fr1)
            c_lo = max(0, fc0 - 2)
            c_hi = min(W - bw, fc1)
            if r_lo > r_hi or c_lo > c_hi:
                r_lo, r_hi = 0, H - bh
                c_lo, c_hi = 0, W - bw

        rmin = random.randint(r_lo, r_hi)
        cmin = random.randint(c_lo, c_hi)
        bbox = (rmin, cmin, rmin + bh - 1, cmin + bw - 1)
        if require_overlap and not any(_bboxes_intersect(bbox, prev) for prev in existing_bboxes):
            continue

        cells = _l_pair_cells(rmin, cmin, bbox[2], bbox[3], diag)
        if can_place(cells):
            do_place(cells, color)
            return bbox
    return None


def _build_legend(H: int, W: int, bg: int):
    """Create legend entries and the corresponding grid and mask arrays."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None

    marker_pool = [c for c in range(10) if c != bg]
    if len(marker_pool) < n_entries:
        return None
    marker_colors = random.sample(marker_pool, n_entries)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill_pool = [c for c in range(10) if c != bg and c != marker]
        fill = random.choice(fill_pool)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(
    H: int,
    W: int,
    legend_c: int,
    legend_bot: int,
    legend_rows: list[tuple[int, int, int]],
) -> np.ndarray:
    """Protect the lower-left legend scan corridor."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _variant_metrics(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    marker_colors: list[int],
    legend_rows: list[tuple[int, int, int]],
    bboxes: list[tuple[int, int, int, int]],
) -> dict[str, float | int]:
    """Quantify the fused constraints for auditing and rejection."""
    bg = get_bg(grid)
    visible_colors = set(map(int, np.unique(grid[~legend_mask])))
    visible_colors.discard(bg)
    union = _union_bbox(bboxes)
    union_ratio = _bbox_area(union) / float(grid.shape[0] * grid.shape[1]) if union else 0.0
    overlap_pairs = 0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _bboxes_intersect(bboxes[i], bboxes[j]):
                overlap_pairs += 1

    return {
        "small_h": int(GRID_H_RANGE[0] <= grid.shape[0] <= GRID_H_RANGE[1]),
        "small_w": int(GRID_W_RANGE[0] <= grid.shape[1] <= GRID_W_RANGE[1]),
        "no_distractors": int(visible_colors <= set(marker_colors)),
        "fills_visible": int(all(fill != bg for _, _, fill in legend_rows)),
        "overlap_pairs": overlap_pairs,
        "union_ratio": union_ratio,
        "active_entries": len(bboxes),
    }


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
    placed_bboxes: list[tuple[int, int, int, int]] = []
    planned_diags = _planned_diagonals(len(legend_rows))

    def can_place(cells: list[tuple[int, int]]) -> bool:
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells: list[tuple[int, int]], color: int) -> None:
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    for i, (_, marker, _) in enumerate(legend_rows):
        bbox = _try_place_l_pair(
            H,
            W,
            marker,
            LPAIR_RETRIES,
            can_place,
            do_place,
            existing_bboxes=placed_bboxes,
            require_overlap=(i == 1),
            diag=planned_diags[i],
        )
        if bbox is None:
            return None
        placed_bboxes.append(bbox)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    metrics = _variant_metrics(grid, legend_mask, marker_colors, legend_rows, placed_bboxes)
    if not metrics["no_distractors"]:
        return None
    if not metrics["fills_visible"]:
        return None
    if metrics["overlap_pairs"] < MIN_OVERLAP_PAIRS:
        return None
    if metrics["union_ratio"] > MAX_ACTIVE_BBOX_RATIO:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(
        f"generate_input: failed to generate a valid compact visible-fill input after {GEN_RETRIES} attempts"
    )
