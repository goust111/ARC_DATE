# Combo variant of 9385bd28 (index 21)
# combo_spec: D2/1+D12/2+D18/1
# Name: 十字形标记+多尺寸变体+方向标记像素

"""
design_analysis:
  fused_rule:
    original_task:
      - Each legend color usually marked one rectangle via two opposite 3-pixel L-corners.
      - A rare 2x2 solid block skipped rectangle filling and was only recolored.
      - The old diagonal-choice logic depended on detecting which bbox corners contained L-corners.
    fused_task:
      - The main marker family is now a pair of disconnected same-color plus-shapes.
      - Each plus has a base arm length k in {1, 2, 3}. Its four orthogonal arms all have length k,
        except one chosen arm has one extra pixel, so that arm has length k+1.
      - The two plus centers still define the candidate rectangle corners, but the extra arm family
        carries the new directional cue:
          vertical long arms  -> fill the TL-BR-oriented slanted half of the center-bbox
          horizontal long arms -> fill the TR-BL-oriented slanted half
      - A secondary marker family remains as recolor-only solid rectangles, but now with several
        visible sizes such as 2x2, 2x3, 3x2, and 3x3.

  behavior_trace:
    concrete_example:
      - Suppose legend says marker 4 -> fill 7.
      - Non-legend color-4 pixels form two directed crosses with base arm length 2.
      - Their centers are at (4, 5) and (10, 12), so the center-bbox is rows 4..10 and cols 5..12.
      - The top cross has an upward extra tip; the bottom cross has a downward extra tip.
    old_flow_breakage:
      - The original solver would search for L-corners on the bbox corners and fail immediately,
        because a directed cross is not an L-shape.
      - Even a simpler "just use the two cross centers and fill the whole rectangle" rule would still
        ignore the added direction cue, so D18 would become visually inert.
      - The original 2x2-only special case would also miss a 2x3 or 3x3 recolor block.
    fused_flow:
      - First classify whether the marker pixels are a full solid rectangle; if so, recolor it only.
      - Otherwise split same-color pixels into 4-connected components and decode each component as one
        directed cross: recover its center, base arm length, and which arm is longer.
      - Use the two centers to get the axis-aligned bbox, and use the long-arm family to choose which
        diagonal-oriented half of that bbox to fill.
      - Preserve legend cells, current cross pixels, and future marker pixels exactly as in the base task.
    visible_difference:
      - Cross markers are larger and come in multiple scales.
      - The long-arm cue changes the output shape, not just an internal branch, so samples no longer
        collapse back to "fill the whole rectangle between centers".

  complexity_calibration:
    controls:
      - Arm lengths are limited to {1, 2, 3}; the long-arm cue has only two semantic families
        (vertical vs horizontal); block sizes come from a small repeated set.
      - Each grid keeps at least two cross-pair legend entries when possible, and generator pressure
        encourages both direction families and at least two arm lengths to appear across those entries.
      - Only one legend entry may become a recolor-only solid block, so the special family stays legible
        without overwhelming the main cross rule.
    why_not_simpler:
      - Removing blocks would discard the D12 contribution entirely.
      - Using long arms but still filling the whole center-bbox would make the direction cue irrelevant.
      - Letting every cross choose arbitrary unrelated long-arm patterns would create too much visual noise,
        so the pair shares one semantic direction family.

  implicit_assumption_review:
    old_main_lines_broken:
      - `_l_anchor_present(...)` assumed sparse 3-pixel L-corners at bbox corners. Directed crosses violate that.
      - `rmin/rmax/cmin/cmax` over all marker pixels no longer identify the fill rectangle, because larger crosses
        extend beyond their centers by k or k+1 cells.
      - `if rmax-rmin == 1 and cmax-cmin == 1 and len(marker_pts) == 4` assumed the only dense special case was 2x2.
      - Any solver path that ignores the new directional cue degenerates to a weaker center-only task.
    new_assumptions:
      - Every active cross marker color has exactly two disconnected directed-cross components.
      - Both components share the same base arm length.
      - Their long-arm families agree semantically: either both are vertical-family or both are horizontal-family.
      - Cross centers differ in both row and column, so they define a genuine diagonal.
      - Recolor-only block markers are full rectangles with height >= 2 and width >= 2.
    generator_guarantees:
      - Placement keeps all marker pixels in bounds and non-overlapping.
      - Cross centers sit far enough apart that the two components never merge under 4-connectivity.
      - At least one cross-pair per grid has `anchor_diag != fill_diag`, so the long-arm cue actually matters.

  placement_legality:
    crosses:
      - A base arm length k needs margin k+1 from every border to allow the extra tip in any direction.
      - The center-bbox is chosen so both centers remain diagonal and sufficiently separated for that k.
      - The forbidden legend corridor from the base task is retained, so legend scan integrity stays intact.
    blocks:
      - Block markers are placed as standalone full rectangles from a small legal size set.
      - Blocks never reuse non-legend colors outside their own legend entry.
    overlap_policy:
      - Marker pixels never overlap each other or legend cells.
      - Filled regions may overlap across entries, matching the original overwrite-order idea.

  visual_observability:
    human_cues:
      - A person sees repeated two-column legend rows, repeated same-color cross pairs, and occasional solid blocks.
      - Among cross pairs, arm length repeats visibly across both components of the same color.
      - The extra-long arm is locally visible and only needs a two-way interpretation:
        vertical-family long arms vs horizontal-family long arms.
      - Outputs show two distinct slanted fill halves plus a recolor-only block family, so 2-4 examples
        are enough to infer the structure without relying on numeric color ids.

  visual_cleanliness_and_color_inference:
    cleanliness:
      - Marker overlap is forbidden, and only one special solid block can appear.
      - Cross sizes are varied but bounded, preventing one sample from turning into an unreadable tangle.
    color_rule:
      - Colors still change only through the explicit bottom-left legend mapping.
      - No hidden ordering by color number or background value is used anywhere.
"""

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


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


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
VERTICAL_DIRS = {"U", "D"}
HORIZONTAL_DIRS = {"L", "R"}


def _split_components4(pts: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Split a point set into 4-connected components."""
    remaining = set(pts)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = {start}
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _solid_rect_mask(h: int, w: int, marker_pts: set[tuple[int, int]]) -> np.ndarray | None:
    """Return a mask for a full rectangular block, else None."""
    if len(marker_pts) < 4:
        return None
    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    bh = rmax - rmin + 1
    bw = cmax - cmin + 1
    if bh < 2 or bw < 2 or len(marker_pts) != bh * bw:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            if (r, c) not in marker_pts:
                return None
            mask[r, c] = True
    return mask


def _directed_cross_info(comp: set[tuple[int, int]]) -> tuple[tuple[int, int], int, str] | None:
    """
    Decode a directed cross component.

    Valid component:
    - center plus four orthogonal arms
    - base arm length k >= 1 in every direction
    - exactly one arm has length k+1 and the other three have length k
    """
    if len(comp) < 6:
        return None
    for center in comp:
        r, c = center
        lengths: dict[str, int] = {}
        expected = {center}
        valid = True
        for name, (dr, dc) in DIRS.items():
            arm_len = 0
            step = 1
            while (r + dr * step, c + dc * step) in comp:
                arm_len += 1
                expected.add((r + dr * step, c + dc * step))
                step += 1
            lengths[name] = arm_len
        if expected != comp:
            continue
        base = min(lengths.values())
        if base < 1:
            continue
        max_len = max(lengths.values())
        if max_len != base + 1:
            continue
        long_dirs = [name for name, value in lengths.items() if value == max_len]
        if len(long_dirs) != 1:
            continue
        if any(value not in (base, base + 1) for value in lengths.values()):
            continue
        return center, base, long_dirs[0]
    return None


def _fill_diag_from_long_dirs(long_dirs: list[str]) -> str:
    families = []
    for name in long_dirs:
        if name in VERTICAL_DIRS:
            families.append("V")
        elif name in HORIZONTAL_DIRS:
            families.append("H")
        else:
            raise AssertionError("unknown direction")
    assert len(set(families)) == 1, "both directed crosses must use the same long-arm family"
    return "TLBR" if families[0] == "V" else "TRBL"


def _anchor_diag_from_centers(a: tuple[int, int], b: tuple[int, int]) -> str:
    (r0, c0), (r1, c1) = a, b
    assert r0 != r1 and c0 != c1, "cross centers must differ in both row and col"
    return "TLBR" if (r1 - r0) * (c1 - c0) > 0 else "TRBL"


def _directional_fill_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    diag: str,
) -> np.ndarray:
    """
    Fill the slanted half of the bbox that is closer to the chosen diagonal.
    """
    m = np.zeros((h, w), dtype=bool)
    hh = r1 - r0
    ww = c1 - c0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            d_tlbr = abs((r - r0) * ww - (c - c0) * hh)
            d_trbl = abs((r - r0) * ww - (c1 - c) * hh)
            if diag == "TLBR":
                take = d_tlbr <= d_trbl
            else:
                take = d_trbl <= d_tlbr
            if take:
                m[r, c] = True
    return m


def _solve_grid(grid: np.ndarray, use_long_arm_direction: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    out = g.copy()
    marker_order = [marker for _, marker, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if not marker_pts:
            continue

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        block_mask = _solid_rect_mask(h, w, marker_pts)
        if block_mask is not None:
            out[block_mask & (~protect)] = int(fill)
            continue

        protect |= (g == marker) & (~legend_mask)
        comps = _split_components4(marker_pts)
        infos: list[tuple[tuple[int, int], int, str]] = []
        for comp in comps:
            info = _directed_cross_info(comp)
            assert info is not None, "every non-block marker component must be a directed cross"
            infos.append(info)

        assert len(infos) == 2, "every active non-block marker must contain exactly two directed crosses"
        centers = [info[0] for info in infos]
        bases = [info[1] for info in infos]
        long_dirs = [info[2] for info in infos]

        assert len(set(bases)) == 1, "paired directed crosses must share one base arm length"
        (r0, c0), (r1, c1) = centers
        rmin, rmax = sorted((r0, r1))
        cmin, cmax = sorted((c0, c1))
        assert rmin < rmax and cmin < cmax, "cross centers must be diagonal corners"

        anchor_diag = _anchor_diag_from_centers(centers[0], centers[1])
        fill_diag = (
            _fill_diag_from_long_dirs(long_dirs) if use_long_arm_direction else anchor_diag
        )

        fill_mask = _directional_fill_mask(h, w, rmin, rmax, cmin, cmax, fill_diag)
        out[fill_mask & (~protect)] = int(fill)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor legend-coded regions using directed crosses and multi-size solid blocks.

    Input structure:
    - A mostly uniform background contains the usual two-column legend stacked near the bottom-left.
    - Away from the legend, each active marker color is either a solid rectangular block or a pair of
      disconnected same-color plus-shapes.
    - A plus-shape has a shared base arm length and one extra-long arm that acts as the direction cue.

    Rule:
    - Read legend rows from bottom to top and collect all non-legend pixels of that marker color.
    - If the pixels form a full rectangle of size at least 2x2, recolor only that block to the legend fill color.
    - Otherwise decode the two connected components as directed crosses, use their centers as the bbox corners,
      and choose the slanted fill half from the long-arm family: vertical long arms mean TL-BR, horizontal
      long arms mean TR-BL.
    - As in the base task, keep legend cells, current cross pixels, and marker pixels of later legend entries protected.

    Key insight:
    - The cross centers provide the geometry, but the extra-long arm is the decisive cue. Ignoring that cue
      collapses the task to a weaker center-only rule, so correct solving requires reading both scale and direction.
    """
    return _solve_grid(grid, use_long_arm_direction=True)


# --------------- generator (combo) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.25
BLOCK_PROB = 0.35
BLOCK_DIM_OPTIONS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_DIM_WEIGHTS = [0.15, 0.30, 0.20, 0.35]

# ── CFG: directed cross geometry ───────────────────────────────────
ARM_LENGTH_OPTIONS = [1, 2, 3]
ARM_LENGTH_WEIGHTS = [0.45, 0.35, 0.20]
CENTER_BBOX_H_MAX = 13
CENTER_BBOX_W_MAX = 13
CENTER_BBOX_SHAPE_WEIGHTS = [0.70, 0.15, 0.15]  # [both>=4, h=3/w>=5, h>=5/w=3]
ANCHOR_FILL_MISMATCH_PROB = 0.65

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
CROSS_RETRIES = 180
DISTRACTOR_RETRIES = 80
GEN_RETRIES = 400


def _cross_cells(center: tuple[int, int], base_arm: int, long_dir: str) -> list[tuple[int, int]]:
    r, c = center
    cells = {(r, c)}
    for name, (dr, dc) in DIRS.items():
        reach = base_arm + (1 if name == long_dir else 0)
        for step in range(1, reach + 1):
            cells.add((r + dr * step, c + dc * step))
    return list(cells)


def _cross_pair_cells(
    rmin: int,
    cmin: int,
    rmax: int,
    cmax: int,
    anchor_diag: str,
    fill_diag: str,
    base_arm: int,
) -> list[tuple[int, int]]:
    if anchor_diag == "TLBR":
        centers = [(rmin, cmin), (rmax, cmax)]
    else:
        centers = [(rmin, cmax), (rmax, cmin)]

    top_idx = 0 if centers[0][0] < centers[1][0] else 1
    bottom_idx = 1 - top_idx
    left_idx = 0 if centers[0][1] < centers[1][1] else 1
    right_idx = 1 - left_idx

    long_dirs = ["", ""]
    if fill_diag == "TLBR":
        long_dirs[top_idx] = "U"
        long_dirs[bottom_idx] = "D"
    else:
        long_dirs[left_idx] = "L"
        long_dirs[right_idx] = "R"

    cells: list[tuple[int, int]] = []
    for center, long_dir in zip(centers, long_dirs):
        cells.extend(_cross_cells(center, base_arm, long_dir))
    return cells


def _pick_center_bbox_size(H: int, W: int, base_arm: int) -> tuple[int, int] | None:
    margin = base_arm + 1
    h_hi = min(CENTER_BBOX_H_MAX, H - 2 * margin)
    w_hi = min(CENTER_BBOX_W_MAX, W - 2 * margin)
    h_lo = max(3, base_arm + 3)
    w_lo = max(3, base_arm + 3)
    if h_hi < h_lo or w_hi < w_lo:
        return None

    options: list[tuple[str, float]] = []
    if h_hi >= max(4, h_lo) and w_hi >= max(4, w_lo):
        options.append(("both_ge4", CENTER_BBOX_SHAPE_WEIGHTS[0]))
    if h_hi >= max(3, h_lo) and w_hi >= max(5, w_lo):
        options.append(("h3_w_ge5", CENTER_BBOX_SHAPE_WEIGHTS[1]))
    if h_hi >= max(5, h_lo) and w_hi >= max(3, w_lo):
        options.append(("h_ge5_w3", CENTER_BBOX_SHAPE_WEIGHTS[2]))

    if not options:
        return random.randint(h_lo, h_hi), random.randint(w_lo, w_hi)

    shape = random.choices(
        [name for name, _ in options],
        weights=[weight for _, weight in options],
    )[0]
    if shape == "both_ge4":
        return random.randint(max(4, h_lo), h_hi), random.randint(max(4, w_lo), w_hi)
    if shape == "h3_w_ge5":
        return max(3, h_lo), random.randint(max(5, w_lo), w_hi)
    return random.randint(max(5, h_lo), h_hi), max(3, w_lo)


def _try_place_cross_pair(
    H: int,
    W: int,
    color: int,
    base_arm: int,
    anchor_diag: str,
    fill_diag: str,
    retries: int,
    can_place,
    do_place,
) -> bool:
    for _ in range(retries):
        size = _pick_center_bbox_size(H, W, base_arm)
        if size is None:
            continue
        bh, bw = size
        margin = base_arm + 1
        rmin_lo, rmin_hi = margin, H - margin - bh
        cmin_lo, cmin_hi = margin, W - margin - bw
        if rmin_lo > rmin_hi or cmin_lo > cmin_hi:
            continue
        rmin = random.randint(rmin_lo, rmin_hi)
        cmin = random.randint(cmin_lo, cmin_hi)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        cells = _cross_pair_cells(rmin, cmin, rmax, cmax, anchor_diag, fill_diag, base_arm)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _pick_block_size(H: int, W: int) -> tuple[int, int] | None:
    fitting = [
        (dims, weight)
        for dims, weight in zip(BLOCK_DIM_OPTIONS, BLOCK_DIM_WEIGHTS)
        if dims[0] <= H and dims[1] <= W
    ]
    if not fitting:
        return None
    dims = random.choices([d for d, _ in fitting], weights=[w for _, w in fitting])[0]
    return int(dims[0]), int(dims[1])


def _solid_block_cells(r0: int, c0: int, bh: int, bw: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)]


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
    if not fill_pool:
        return None

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        # Keep the legend's two columns visibly informative: the fill swatch should not
        # disappear into the background or masquerade as any marker-family color.
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
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _assign_types(n_entries: int) -> list[str]:
    """
    Keep at least two directed-cross entries whenever possible so the core fused rule remains visible.
    """
    types = ["cross"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    special_budget = max(0, n_entries - 2)
    if special_budget > 0 and random.random() < NOOP_PROB and idx_pool:
        types[idx_pool.pop()] = "noop"
        special_budget -= 1
    if special_budget > 0 and random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"
    return types


def _opposite_diag(diag: str) -> str:
    return "TRBL" if diag == "TLBR" else "TLBR"


def _assign_cross_specs(types: list[str]) -> dict[int, tuple[int, str, str]]:
    """
    Returns {entry_index: (base_arm, anchor_diag, fill_diag)} for cross entries.
    Forces at least one anchor-vs-fill mismatch and usually shows both fill directions
    and multiple arm lengths when the grid has enough cross entries.
    """
    cross_indices = [i for i, t in enumerate(types) if t == "cross"]
    specs: dict[int, tuple[int, str, str]] = {}
    if not cross_indices:
        return specs

    fill_diags = {i: random.choice(["TLBR", "TRBL"]) for i in cross_indices}
    if len(cross_indices) >= 2 and len({fill_diags[i] for i in cross_indices}) == 1:
        pivot = random.choice(cross_indices)
        fill_diags[pivot] = _opposite_diag(fill_diags[pivot])

    arm_lengths = {
        i: random.choices(ARM_LENGTH_OPTIONS, weights=ARM_LENGTH_WEIGHTS)[0]
        for i in cross_indices
    }
    if len(cross_indices) >= 2 and len({arm_lengths[i] for i in cross_indices}) == 1:
        pivot = random.choice(cross_indices)
        choices = [k for k in ARM_LENGTH_OPTIONS if k != arm_lengths[pivot]]
        arm_lengths[pivot] = random.choice(choices)

    mismatches = set()
    for i in cross_indices:
        fill_diag = fill_diags[i]
        if random.random() < ANCHOR_FILL_MISMATCH_PROB:
            anchor_diag = _opposite_diag(fill_diag)
            mismatches.add(i)
        else:
            anchor_diag = fill_diag
        specs[i] = (arm_lengths[i], anchor_diag, fill_diag)

    if not mismatches:
        pivot = random.choice(cross_indices)
        base_arm, anchor_diag, fill_diag = specs[pivot]
        specs[pivot] = (base_arm, _opposite_diag(anchor_diag), fill_diag)

    return specs


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

    types = _assign_types(len(legend_rows))
    cross_specs = _assign_cross_specs(types)
    if not cross_specs:
        return None

    def can_place(cells: list[tuple[int, int]]) -> bool:
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if forbidden[r, c] or occupied[r, c]:
                return False
        return True

    def do_place(cells: list[tuple[int, int]], color: int) -> None:
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    for i, (_, marker, _) in enumerate(legend_rows):
        t = types[i]
        if t == "noop":
            continue
        if t == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                dims = _pick_block_size(H, W)
                if dims is None:
                    break
                bh, bw = dims
                r0 = random.randint(0, H - bh)
                c0 = random.randint(0, W - bw)
                cells = _solid_block_cells(r0, c0, bh, bw)
                if can_place(cells):
                    do_place(cells, marker)
                    placed = True
                    break
            if not placed:
                return None
            continue

        base_arm, anchor_diag, fill_diag = cross_specs[i]
        if not _try_place_cross_pair(
            H,
            W,
            marker,
            base_arm,
            anchor_diag,
            fill_diag,
            CROSS_RETRIES,
            can_place,
            do_place,
        ):
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            color = random.choice(avail)
            base_arm = random.choices(ARM_LENGTH_OPTIONS, weights=ARM_LENGTH_WEIGHTS)[0]
            fill_diag = random.choice(["TLBR", "TRBL"])
            anchor_diag = random.choice(["TLBR", "TRBL"])
            _try_place_cross_pair(
                H,
                W,
                color,
                base_arm,
                anchor_diag,
                fill_diag,
                DISTRACTOR_RETRIES,
                can_place,
                do_place,
            )

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        new_out = main(grid)
        baseline_out = _solve_grid(grid, use_long_arm_direction=False)
    except Exception:
        return None

    if np.array_equal(new_out, baseline_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(
        f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts"
    )
