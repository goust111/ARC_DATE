# Combo variant of 9385bd28 (index 1)
# combo_spec: D1/1+D2/1+D18/1
# Name: 连接器配对十字形标记+方向标记

"""
design_analysis:
  concrete_rule:
    legend:
      Keep the original lower-left vertical 2-column legend. Each row is still
      (marker_color, fill_color), processed bottom-to-top.
    regular_object:
      A regular object now occupies the four logical corners of one rectangle with
      four same-color directional crosses.
      Each corner marker is a 5-pixel plus sign with one extra inward pixel, so one
      arm is length 2. That extra pixel is the D18 direction cue.
      Exactly one diagonal pair of corner crosses is joined by a thin same-color
      rectilinear connector. That is the D1 pairing cue.
      The connected diagonal's two crosses must share the same extension family:
      horizontal inward extensions mean TLBR-oriented fill; vertical inward
      extensions mean TRBL-oriented fill.
      The unconnected diagonal is a decoy and is generated with the opposite
      extension family so the connector is genuinely needed.

  behavior_trace:
    small_example:
      Suppose the four cross centers sit at (2,3), (2,9), (8,3), (8,9).
      TL and BR are connected by a thin path, while TR and BL stay isolated.
      If the connected TL and BR crosses have their extra pixels on the horizontal
      inward arms, the solver must fill the cells in rows 2..8 and cols 3..9 that
      are closer to the TL-BR diagonal than to the other diagonal.
      The disconnected TR and BL crosses may show the opposite extension family,
      but they must be ignored because they are not the connector-selected pair.
    old_vs_new:
      Original code assumed one marker color provided two oriented L-corners and
      the corner geometry alone revealed which diagonal mattered.
      In the fusion variant, four same-color corner candidates exist, so color and
      rectangle extrema are not enough. The solver must:
      1. detect the four cross centers,
      2. use connectivity to decide which diagonal is active,
      3. read the active diagonal's extended arm family,
      4. convert that family into the visible slanted fill direction.

  complexity_calibration:
    visibility:
      Four large plus-shaped corner candidates make the connector visually
      indispensable. The longer arm is also easy to spot in 2-4 examples.
    ambiguity_control:
      The connector runs only through the interior corridor at least 2 cells away
      from non-active corners, which prevents accidental merges and fake centers.
      The decoy diagonal always carries the opposite extension family, so "read all
      corners and majority vote" is not a valid shortcut.
    freedom_control:
      Direction has only two observable categories: horizontal-arm extension vs
      vertical-arm extension. Pairing has only two observable categories: TLBR vs
      TRBL connector. That keeps the task inferable from a small sample.

  implicit_assumption_audit:
    broken_old_assumptions:
      - The bbox of all marker pixels no longer directly gives the fill rectangle,
        because each cross sticks out by one cell and one arm by two cells.
      - L-corner template checks are invalid because markers are symmetric crosses.
      - A marker color can now split into three components: one connected diagonal
        pair plus two isolated decoy crosses.
      - Reading orientation from all same-color pixels is wrong because only the
        connector-linked diagonal contributes the true direction cue.
    new_solver_assumptions:
      - Every active marker color has exactly four cross centers at the logical
        rectangle corners.
      - Exactly one diagonal pair shares a 4-neighbor component.
      - The two connected corners share one hint family, and the two decoy corners
        share the opposite family.
      - Connector segments never create extra 4-neighbor cross centers.
      generate_input() enforces all of these, and oracle checks reject violations.

  placement_legality:
    bounds:
      Cross centers are placed at least one cell from the grid border so the outer
      arms remain in-bounds. Logical rectangles are at least 5x5 so there is a safe
      interior connector corridor and room for a length-2 inward arm.
    collision_rules:
      Marker pixels may not overlap each other or the legend protection zone.
      Fill regions may overlap, matching the original overwrite style.
    connector_safety:
      Horizontal-family active pairs use a row-bridge path whose vertical legs stay
      on columns cmin+2 / cmax-2; vertical-family pairs use a col-bridge path on
      rows rmin+2 / rmax-2. These corridors stay clear of the decoy corners.

  visual_observability:
    human_view:
      A human sees four same-color plus-like corner candidates, one diagonal tied
      together by a thin line, and one of each active cross's inward arms extended.
      Across examples, the connected diagonal decides which two corners count, while
      the longer-arm orientation decides which slanted half of the rectangle is
      colored. No hidden use of color ids or ordering is required.

  visual_cleanliness:
    controls:
      The connector is a single-bend monotone polyline, there is at most one
      distractor object, and decoy corners remain disconnected from the active pair.
      This keeps the causal structure readable despite allowing fill overlap.

  color_inference:
    cue:
      Colors still behave exactly as legend mappings from marker color to fill color.
      The connector and longer arm only disambiguate geometry; they do not introduce
      extra color semantics.
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


_CROSS_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
_HINT_TO_FILL_DIAG = {"H": "TLBR", "V": "TRBL"}


def _opposite_diag(diag: str) -> str:
    return "TRBL" if diag == "TLBR" else "TLBR"


def _opposite_family(family: str) -> str:
    return "V" if family == "H" else "H"


def _components4(pts: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    comps: list[set[tuple[int, int]]] = []
    unseen = set(pts)
    while unseen:
        start = unseen.pop()
        stack = [start]
        comp = {start}
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


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


def _cross_centers(pts: set[tuple[int, int]]) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for r, c in pts:
        if all((r + dr, c + dc) in pts for dr, dc in _CROSS_OFFSETS):
            centers.append((r, c))
    return sorted(centers)


def _corner_positions(rmin: int, rmax: int, cmin: int, cmax: int) -> dict[str, tuple[int, int]]:
    return {
        "TL": (rmin, cmin),
        "TR": (rmin, cmax),
        "BL": (rmax, cmin),
        "BR": (rmax, cmax),
    }


def _hint_endpoint(center: tuple[int, int], corner_name: str, family: str) -> tuple[int, int]:
    r, c = center
    if family == "H":
        if corner_name in ("TL", "BL"):
            return r, c + 2
        return r, c - 2
    if corner_name in ("TL", "TR"):
        return r + 2, c
    return r - 2, c


def _hint_family_for_corner(
    pts: set[tuple[int, int]],
    corner_name: str,
    center: tuple[int, int],
) -> str:
    h_pt = _hint_endpoint(center, corner_name, "H")
    v_pt = _hint_endpoint(center, corner_name, "V")
    has_h = h_pt in pts
    has_v = v_pt in pts
    assert has_h != has_v, "each corner cross must have exactly one extended inward arm"
    return "H" if has_h else "V"


def _parse_regular_object(marker_pts: set[tuple[int, int]]) -> dict[str, object]:
    centers = _cross_centers(marker_pts)
    assert len(centers) == 4, "regular object must contain exactly four cross centers"

    center_rows = sorted({r for r, _ in centers})
    center_cols = sorted({c for _, c in centers})
    assert len(center_rows) == 2 and len(center_cols) == 2, "cross centers must form rectangle corners"
    rmin, rmax = center_rows
    cmin, cmax = center_cols
    corners = _corner_positions(rmin, rmax, cmin, cmax)
    assert set(centers) == set(corners.values()), "cross centers must occupy all logical rectangle corners"

    comps = _components4(marker_pts)
    comp_id: dict[tuple[int, int], int] = {}
    for cid, comp in enumerate(comps):
        for pt in comp:
            comp_id[pt] = cid

    tlbr_linked = comp_id[corners["TL"]] == comp_id[corners["BR"]]
    trbl_linked = comp_id[corners["TR"]] == comp_id[corners["BL"]]
    assert tlbr_linked != trbl_linked, "exactly one diagonal pair must be connector-linked"
    active_diag = "TLBR" if tlbr_linked else "TRBL"

    families = {
        name: _hint_family_for_corner(marker_pts, name, center)
        for name, center in corners.items()
    }
    active_names = ("TL", "BR") if active_diag == "TLBR" else ("TR", "BL")
    decoy_names = ("TR", "BL") if active_diag == "TLBR" else ("TL", "BR")
    active_family = families[active_names[0]]
    assert families[active_names[1]] == active_family, "connected pair must agree on direction family"
    assert families[decoy_names[0]] == families[decoy_names[1]], "decoy pair must also agree on family"

    return {
        "rmin": rmin,
        "rmax": rmax,
        "cmin": cmin,
        "cmax": cmax,
        "corners": corners,
        "families": families,
        "active_diag": active_diag,
        "fill_diag": _HINT_TO_FILL_DIAG[active_family],
    }


def _solve_grid(grid: np.ndarray, use_connected_pair: bool) -> np.ndarray:
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
        protect |= (g == marker) & (~legend_mask)

        parsed = _parse_regular_object(marker_pts)
        fill_diag: str
        if use_connected_pair:
            fill_diag = str(parsed["fill_diag"])
        else:
            active_diag = str(parsed["active_diag"])
            other_names = ("TR", "BL") if active_diag == "TLBR" else ("TL", "BR")
            families = parsed["families"]
            other_family = str(families[other_names[0]])
            assert str(families[other_names[1]]) == other_family
            fill_diag = _HINT_TO_FILL_DIAG[other_family]

        fill_mask = _directional_fill_mask(
            h,
            w,
            int(parsed["rmin"]),
            int(parsed["rmax"]),
            int(parsed["cmin"]),
            int(parsed["cmax"]),
            fill_diag,
        )
        out[fill_mask & (~protect)] = int(fill)

    return out


def _solve_grid_fixed_relation(grid: np.ndarray, use_same_relation: bool) -> np.ndarray:
    """
    Auxiliary oracle for generator validation.
    Pretend the fill diagonal is always either the connected diagonal itself
    or always the opposite one, ignoring the D18 direction cue.
    """
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
        protect |= (g == marker) & (~legend_mask)

        parsed = _parse_regular_object(marker_pts)
        active_diag = str(parsed["active_diag"])
        fill_diag = active_diag if use_same_relation else _opposite_diag(active_diag)
        fill_mask = _directional_fill_mask(
            h,
            w,
            int(parsed["rmin"]),
            int(parsed["rmax"]),
            int(parsed["cmin"]),
            int(parsed["cmax"]),
            fill_diag,
        )
        out[fill_mask & (~protect)] = int(fill)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor connector-selected cross-corner rectangles using legend colors and arm-direction cues.

    Input structure:
    - The lower-left still contains a vertical 2-column legend whose rows are (marker_color, fill_color).
    - Each active marker color forms one rectangle with four same-color corner crosses at the logical corners.
    - One diagonal pair of those corner crosses is joined by a thin same-color connector, and each cross has one
      extra inward pixel so one arm is longer than the others.

    Rule:
    - Read legend entries from bottom to top.
    - For each marker color, ignore the legend cells, detect the four cross centers, and determine which diagonal
      pair belongs to the same 4-neighbor component because of the connector.
    - Read the longer-arm direction only from that connected diagonal: horizontal inward extensions select the
      TL-BR-oriented slanted fill, while vertical inward extensions select the TR-BL-oriented slanted fill.
    - Fill the corresponding region inside the rectangle spanned by the four cross centers, while preserving
      legend cells, current marker cells, and marker pixels of later legend entries.

    Key insight:
    - The connector answers "which two corners count"; the longer arm answers "which diagonal-oriented half of the
      rectangle should be colored." Neither cue is sufficient by itself in the generated inputs.
    """
    return _solve_grid(grid, use_connected_pair=True)


# --------------- generator (combo D1+D2+D18) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 4)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.25

# ── CFG: rectangle / connector geometry ────────────────────────────
RECT_H_MIN = 5
RECT_W_MIN = 5
RECT_H_MAX = 11
RECT_W_MAX = 11
RECT_SHAPE_WEIGHTS = [0.6, 0.2, 0.2]  # [balanced, taller, wider]
ACTIVE_DIAG_PROB = 0.5
FILL_DIAG_PROB = 0.5

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.2

# ── CFG: retry budgets ─────────────────────────────────────────────
PAIR_RETRIES = 220
DISTRACTOR_RETRIES = 80
GEN_RETRIES = 500


def _cross_cells(center_r: int, center_c: int) -> list[tuple[int, int]]:
    return [(center_r + dr, center_c + dc) for dr, dc in _CROSS_OFFSETS]


def _directional_cross_cells(
    center: tuple[int, int],
    corner_name: str,
    family: str,
) -> list[tuple[int, int]]:
    cells = _cross_cells(center[0], center[1])
    cells.append(_hint_endpoint(center, corner_name, family))
    return cells


def _connector_cells(
    rmin: int,
    cmin: int,
    rmax: int,
    cmax: int,
    active_diag: str,
    family: str,
) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    if active_diag == "TLBR":
        start_corner, end_corner = "TL", "BR"
        start_center, end_center = (rmin, cmin), (rmax, cmax)
    else:
        start_corner, end_corner = "TR", "BL"
        start_center, end_center = (rmin, cmax), (rmax, cmin)

    start = _hint_endpoint(start_center, start_corner, family)
    end = _hint_endpoint(end_center, end_corner, family)

    if family == "H":
        bridge_r = random.randint(rmin + 2, rmax - 2)
        c0, c1 = start[1], end[1]
        step = 1 if c1 >= c0 else -1
        for r in range(start[0], bridge_r + 1):
            pts.append((r, start[1]))
        for c in range(c0, c1 + step, step):
            pts.append((bridge_r, c))
        step_r = 1 if end[0] >= bridge_r else -1
        for r in range(bridge_r, end[0] + step_r, step_r):
            pts.append((r, end[1]))
    else:
        bridge_c = random.randint(cmin + 2, cmax - 2)
        r0, r1 = start[0], end[0]
        step = 1 if bridge_c >= start[1] else -1
        for c in range(start[1], bridge_c + step, step):
            pts.append((start[0], c))
        step_r = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step_r, step_r):
            pts.append((r, bridge_c))
        step = 1 if end[1] >= bridge_c else -1
        for c in range(bridge_c, end[1] + step, step):
            pts.append((end[0], c))

    return sorted(set(pts))


def _build_object_cells(
    rmin: int,
    cmin: int,
    rmax: int,
    cmax: int,
    active_diag: str,
    fill_diag: str,
) -> list[tuple[int, int]]:
    active_family = "H" if fill_diag == "TLBR" else "V"
    decoy_family = _opposite_family(active_family)
    corner_centers = _corner_positions(rmin, rmax, cmin, cmax)
    cells: list[tuple[int, int]] = []
    for corner_name, center in corner_centers.items():
        on_active_diag = (
            corner_name in ("TL", "BR") if active_diag == "TLBR" else corner_name in ("TR", "BL")
        )
        family = active_family if on_active_diag else decoy_family
        cells.extend(_directional_cross_cells(center, corner_name, family))
    cells.extend(_connector_cells(rmin, cmin, rmax, cmax, active_diag, active_family))
    return sorted(set(cells))


def _pick_rect_size(H: int, W: int) -> tuple[int, int] | None:
    h_hi = min(RECT_H_MAX, H - 2)
    w_hi = min(RECT_W_MAX, W - 2)
    if h_hi < RECT_H_MIN or w_hi < RECT_W_MIN:
        return None

    shape_choice = random.choices(
        ["balanced", "taller", "wider"],
        weights=RECT_SHAPE_WEIGHTS,
    )[0]
    if shape_choice == "balanced":
        bh = random.randint(RECT_H_MIN, h_hi)
        bw = random.randint(RECT_W_MIN, w_hi)
    elif shape_choice == "taller":
        bh = random.randint(max(RECT_H_MIN + 1, 6), h_hi)
        bw = random.randint(RECT_W_MIN, min(w_hi, 7))
    else:
        bh = random.randint(RECT_H_MIN, min(h_hi, 7))
        bw = random.randint(max(RECT_W_MIN + 1, 6), w_hi)
    if bh < RECT_H_MIN or bw < RECT_W_MIN:
        return None
    return bh, bw


def _try_place_object(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
    active_diag: str | None = None,
    fill_diag: str | None = None,
) -> bool:
    for _ in range(retries):
        size = _pick_rect_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin_lo, rmin_hi = 1, H - bh - 1
        cmin_lo, cmin_hi = 1, W - bw - 1
        if rmin_lo > rmin_hi or cmin_lo > cmin_hi:
            continue
        rmin = random.randint(rmin_lo, rmin_hi)
        cmin = random.randint(cmin_lo, cmin_hi)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        obj_active_diag = active_diag
        if obj_active_diag is None:
            obj_active_diag = "TLBR" if random.random() < ACTIVE_DIAG_PROB else "TRBL"
        obj_fill_diag = fill_diag
        if obj_fill_diag is None:
            obj_fill_diag = "TLBR" if random.random() < FILL_DIAG_PROB else "TRBL"
        cells = _build_object_cells(rmin, cmin, rmax, cmax, obj_active_diag, obj_fill_diag)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


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

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        legend_mask[r, legend_c] = True
        grid[r, legend_c + 1] = fill
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


def _random_diag(p_tlbr: float = 0.5) -> str:
    return "TLBR" if random.random() < p_tlbr else "TRBL"


def _assign_entry_specs(n_entries: int) -> list[dict[str, str]]:
    """
    Keep at least two real objects and force them to disagree on the
    active-vs-fill relation so the D18 cue is needed on every accepted grid.
    """
    specs: list[dict[str, str]] = [{"kind": "pair"} for _ in range(n_entries)]
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if n_entries > 2 and random.random() < NOOP_PROB:
        specs[idx_pool.pop()] = {"kind": "noop"}

    pair_indices = [i for i, spec in enumerate(specs) if spec["kind"] == "pair"]
    assert len(pair_indices) >= 2, "generator must leave at least two real objects"

    forced_pair_indices = random.sample(pair_indices, 2)
    if random.random() < 0.5:
        forced_specs = [
            {"kind": "pair", "active_diag": "TLBR", "fill_diag": "TLBR"},
            {"kind": "pair", "active_diag": "TRBL", "fill_diag": "TLBR"},
        ]
    else:
        forced_specs = [
            {"kind": "pair", "active_diag": "TLBR", "fill_diag": "TRBL"},
            {"kind": "pair", "active_diag": "TRBL", "fill_diag": "TRBL"},
        ]
    random.shuffle(forced_specs)
    for idx, spec in zip(forced_pair_indices, forced_specs):
        specs[idx] = spec

    for idx in pair_indices:
        if specs[idx]["kind"] != "pair" or "active_diag" in specs[idx]:
            continue
        specs[idx] = {
            "kind": "pair",
            "active_diag": _random_diag(ACTIVE_DIAG_PROB),
            "fill_diag": _random_diag(FILL_DIAG_PROB),
        }

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

    entry_specs = _assign_entry_specs(len(legend_rows))

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
        spec = entry_specs[i]
        if spec["kind"] == "noop":
            continue
        if not _try_place_object(
            H,
            W,
            marker,
            PAIR_RETRIES,
            can_place,
            do_place,
            active_diag=spec["active_diag"],
            fill_diag=spec["fill_diag"],
        ):
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_object(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        new_out = main(grid)
        wrong_pair_out = _solve_grid(grid, use_connected_pair=False)
        same_relation_out = _solve_grid_fixed_relation(grid, use_same_relation=True)
        opposite_relation_out = _solve_grid_fixed_relation(grid, use_same_relation=False)
    except Exception:
        return None

    if np.array_equal(new_out, wrong_pair_out):
        return None
    if np.array_equal(new_out, same_relation_out):
        return None
    if np.array_equal(new_out, opposite_relation_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
