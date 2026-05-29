# Combo variant of 9385bd28 (index 18)
# combo_spec: D3/3+D7/2+D8/2
# Name: 仅保留填充区域+标记轨迹线+对角带状填充

import random
from collections import Counter

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg


# Design analysis
# ---------------
# fusion_summary:
#   - Keep the original input grammar: a bottom-left 2-column legend maps marker_color -> fill_color,
#     and each active marker color is still realized as a diagonal pair of L-corners.
#   - D3 changes the destination geometry from a full diamond/rectangle fill to a thin diagonal band
#     on the *opposite* diagonal of the marker anchors.
#   - D8 is fused in a stricter way than the single-dimension version: raw L markers are not preserved.
#     Instead, each anchor corner leaves a 1-cell trail that runs from the marker corner to the nearest
#     endpoint of the diagonal band, i.e. along the top and bottom edges of the inferred bbox.
#   - D7 then discards the full scene: legend, distractors, and unused marker arms vanish, and the
#     output is tightly cropped to the union of band pixels plus trail pixels.
# behavior_trace:
#   - Example: suppose a marker color forms L anchors at TL=(2,4) and BR=(6,9).
#   - Original task: infer that the fill belongs on the opposite diagonal TR=(2,9) to BL=(6,4),
#     then fill every shortest-path cell inside the bbox, leaving the L markers and legend in place.
#   - Combo task: infer the same hidden bbox, but draw only the supercover diagonal band from TR to BL,
#     then draw a marker-colored top trail from TL to TR and a bottom trail from BR to BL.
#     Finally, erase all scaffolding and crop to the painted union.
#   - Visible consequence: compared with the original, the interior rectangle disappears; compared with
#     D3+D7 alone, extra marker-colored edge traces remain; compared with the D8 single variant, those
#     traces are off the band rather than overlaid on the band itself.
# complexity_calibration:
#   - The generator keeps 2-4 legend rows so the legend mapping repeats enough to be learnable without
#     creating an unreadable cropped union.
#   - It guarantees at least two genuine L-pairs and avoids 2x2 blocks in generated examples, because
#     the combo's distinctive evidence is the interaction between diagonal bands and edge trails.
#   - Fill colors exclude bg and all legend marker colors so the trail-vs-fill distinction is obvious.
#   - At least one L-pair must have a bbox whose rectangle area exceeds band area by >= 3 cells, so the
#     output cannot be mistaken for the original rectangle/diamond fill.
# implicit_assumption_audit:
#   - `_find_legend` is still valid because the legend geometry is unchanged.
#   - Bounding-box extraction and L-anchor tests still rely on opposite bbox corners being occupied by
#     one marker color; generator preserves that invariant.
#   - The original full-scene solver assumed `out` started from the input and current markers remained
#     visible. That no longer holds, so rendering now starts from a blank bg canvas and explicitly draws
#     only the produced band and trails.
#   - Protection against future markers is still required even though the output removes markers: without
#     it, early entries would backfill through cells that are occupied by later markers in the source scene.
#   - Current marker cells must be protected from the fill band but not from the new trails, otherwise the
#     trail could not start at the anchor corner.
# placement_legality:
#   - Marker cells never overlap each other or the legend, and all placements remain inside bounds.
#   - The forbidden zone left of / below the legend is preserved so bottom-left legend scanning stays
#     unambiguous.
#   - Output overlaps between different entries are allowed and use the same bottom-to-top overwrite order
#     as the source task.
# observability_and_cleanliness:
#   - Human-solvable evidence comes from three repeated cues in every valid sample: thin diagonal fill,
#     off-band marker trails touching the band only at endpoints, and tight cropping that removes the legend.
#   - The generator rejects samples where the combo output collapses to the band-only crop, ensuring the
#     D8 contribution remains visible instead of silently degenerating into D3+D7.


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

    This is more robust than checking only corner membership, especially when the bbox is 2-wide/2-tall
    where L arms can overlap other bbox corners.
    """
    dr1, dc1 = arm1
    dr2, dc2 = arm2
    return (r, c) in pts and (r + dr1, c + dc1) in pts and (r + dr2, c + dc2) in pts


def _segment_hits_cell(a: tuple[int, int], b: tuple[int, int], r: int, c: int) -> bool:
    """True iff the center-to-center segment from a to b intersects cell (r,c)'s unit square."""
    y0, x0 = float(a[0]), float(a[1])
    y1, x1 = float(b[0]), float(b[1])
    xmin, xmax = c - 0.5, c + 0.5
    ymin, ymax = r - 0.5, r + 0.5
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0
    eps = 1e-9
    for p, q in (
        (-dx, x0 - xmin),
        (dx, xmax - x0),
        (-dy, y0 - ymin),
        (dy, ymax - y0),
    ):
        if abs(p) < eps:
            if q < -eps:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 - t1 > eps:
            return False
    return True


def _diag_band_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """Mask for the 1-2 pixel diagonal band between opposite fill corners a and b."""
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _segment_hits_cell(a, b, r, c):
                m[r, c] = True
    return m


def _axis_segment_mask(h: int, w: int, a: tuple[int, int], b: tuple[int, int]) -> np.ndarray:
    """Inclusive horizontal/vertical 1-cell segment."""
    m = np.zeros((h, w), dtype=bool)
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        m[r0, lo : hi + 1] = True
        return m
    assert c0 == c1, "trail segments in this combo are axis-aligned"
    lo, hi = sorted((r0, r1))
    m[lo : hi + 1, c0] = True
    return m


def _infer_geometry(
    marker_pts: set[tuple[int, int]],
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]]:
    """Infer fill-band endpoints and the two marker-to-band trail segments."""
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
    assert score_tlbr != score_trbl, "diagonal scores must not tie (tie-break path is unreachable)"

    if score_tlbr > score_trbl:
        return tr, bl, [(tl, tr), (br, bl)]
    return tl, br, [(tr, tl), (bl, br)]


def _render_result(grid: np.ndarray, with_trails: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """Render the combo result on a full-size bg canvas before tight cropping."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = int(get_bg(g))
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    out = np.full_like(g, bg)
    painted = np.zeros_like(g, dtype=bool)
    marker_order = [m for _, m, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        eff_fill = int(fill)
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        future_markers = set(marker_order[i + 1 :])
        base_protect = legend_mask.copy()
        for fm in future_markers:
            base_protect |= (g == fm) & (~legend_mask)

        current_marker_mask = (g == marker) & (~legend_mask)
        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            block_mask = current_marker_mask & (~base_protect)
            out[block_mask] = eff_fill
            painted |= block_mask
            continue

        if rmin == rmax or cmin == cmax:
            continue

        fill_a, fill_b, trail_segments = _infer_geometry(marker_pts, rmin, rmax, cmin, cmax)

        band = _diag_band_mask(h, w, rmin, rmax, cmin, cmax, fill_a, fill_b)
        fill_mask = band & (~base_protect) & (~current_marker_mask)
        out[fill_mask] = eff_fill
        painted |= fill_mask

        if with_trails:
            for s0, s1 in trail_segments:
                trail = _axis_segment_mask(h, w, s0, s1)
                trail_mask = trail & (~base_protect)
                out[trail_mask] = int(marker)
                painted |= trail_mask

    return out, painted, bg


def _render_cropped(grid: np.ndarray, with_trails: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    out, painted, bg = _render_result(grid, with_trails=with_trails)
    bb = bbox(painted)
    assert bb is not None, "variant expects at least one painted pixel"
    return crop_bbox(out, bb), painted, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Render only the diagonal-band results of the marker puzzle, with marker-colored trails left behind.

    Input structure:
    - A single background color fills the canvas, and a vertical two-column legend sits near the bottom-left.
    - Each legend row maps one marker color to one fill color, read from bottom to top.
    - Away from the legend, active marker colors appear as diagonal pairs of L-shaped corners; optional non-legend
      distractors may still exist, but they are ignored because they are absent from the legend.

    Rule:
    - For each legend entry, find all non-legend pixels of that marker color and recover the bounding box they define.
    - Detect which bbox diagonal carries the L anchors; the opposite diagonal determines the fill band.
    - Paint only the thin diagonal band between those opposite corners using the legend fill color.
    - From each anchor corner, draw a 1-cell marker-colored trail along the bbox edge until it touches the band's
      nearest endpoint, then discard the original marker shapes, legend, and all untouched background context.
    - Tight-crop the output to the union of all band and trail pixels.

    Key insight: The legend and L-corners still identify the same hidden bbox as in the source task, but the answer is
    no longer a full-scene rectangle fill. It is the cropped transform-only residue: a diagonal band plus two edge
    traces showing where the markers fed into that band.
    """
    cropped, _, _ = _render_cropped(grid, with_trails=True)
    return cropped


# --------------- generator (combo) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 4)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.18
BLOCK_PROB = 0.0
MIN_ACTIVE_LPAIRS = 2

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.72, 0.14, 0.14]
TLBR_PROB = 0.8
MIN_BAND_SAVINGS = 3
MIN_EXTRA_TRAIL_PIXELS = 4

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.2

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 150
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 400

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


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
    desired_diag: str | None = None,
) -> dict[str, int | str] | None:
    """Try to place an L-pair of `color`; return bbox metadata on success."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = desired_diag if desired_diag is not None else ("TLBR" if random.random() < TLBR_PROB else "TRBL")
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
        if can_place(cells):
            do_place(cells, color)
            return {
                "rmin": rmin,
                "cmin": cmin,
                "rmax": rmin + bh - 1,
                "cmax": cmin + bw - 1,
                "diag": diag,
            }
    return None


def _build_legend(H: int, W: int, bg: int):
    """Create legend entries and the corresponding grid/mask arrays."""
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
    if not fill_pool:
        return None

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = random.choice(fill_pool)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows):
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _fill_corners_from_anchor_diag(
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    anchor_diag: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if anchor_diag == "TLBR":
        return (rmin, cmax), (rmax, cmin)
    return (rmin, cmin), (rmax, cmax)


def _combo_metrics(grid: np.ndarray) -> dict[str, int | bool | tuple[int, int]]:
    combo_crop, combo_painted, _ = _render_cropped(grid, with_trails=True)
    band_crop, band_painted, _ = _render_cropped(grid, with_trails=False)
    extra_trail_pixels = int(np.count_nonzero(combo_painted & (~band_painted)))
    return {
        "extra_trail_pixels": extra_trail_pixels,
        "crop_changed": combo_crop.shape != grid.shape,
        "combo_differs_from_band_only": not np.array_equal(combo_crop, band_crop),
        "combo_shape": tuple(int(x) for x in combo_crop.shape),
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

    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and n_entries > MIN_ACTIVE_LPAIRS:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"
    while sum(t == "L" for t in types) < MIN_ACTIVE_LPAIRS:
        for idx, t in enumerate(types):
            if t != "L":
                types[idx] = "L"
                break

    l_indices = [i for i, t in enumerate(types) if t == "L"]
    diag_plan: dict[int, str] = {}
    if len(l_indices) >= 2:
        random.shuffle(l_indices)
        for pos, idx in enumerate(l_indices):
            diag_plan[idx] = "TLBR" if pos % 2 == 0 else "TRBL"
    elif len(l_indices) == 1:
        diag_plan[l_indices[0]] = "TLBR" if random.random() < TLBR_PROB else "TRBL"

    def can_place(cells):
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells, color):
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    l_infos: list[dict[str, int | str]] = []
    for i, (_, marker, fill) in enumerate(legend_rows):
        t = types[i]
        if t == "noop":
            continue
        if t == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, marker)
                    placed = True
                    break
            if not placed:
                return None
        else:
            info = _try_place_l_pair(
                H,
                W,
                marker,
                LPAIR_RETRIES,
                can_place,
                do_place,
                desired_diag=diag_plan.get(i),
            )
            if info is None:
                return None
            info["fill"] = int(fill)
            l_infos.append(info)

    if len(l_infos) < MIN_ACTIVE_LPAIRS:
        return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

    strong_band_count = 0
    for info in l_infos:
        rmin = int(info["rmin"])
        cmin = int(info["cmin"])
        rmax = int(info["rmax"])
        cmax = int(info["cmax"])
        fill_a, fill_b = _fill_corners_from_anchor_diag(rmin, rmax, cmin, cmax, str(info["diag"]))
        band = _diag_band_mask(H, W, rmin, rmax, cmin, cmax, fill_a, fill_b)
        bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
        if bbox_area - int(band.sum()) >= MIN_BAND_SAVINGS:
            strong_band_count += 1
    if strong_band_count == 0:
        return None

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(grid)
        metrics = _combo_metrics(grid)
    except Exception:
        return None

    if not metrics["crop_changed"]:
        return None
    if not metrics["combo_differs_from_band_only"]:
        return None
    if int(metrics["extra_trail_pixels"]) < MIN_EXTRA_TRAIL_PIXELS:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
