# Combo variant of a395ee82 (index 4)
# combo_spec: D3/1+D8/1+D11/0
# Name: 孔洞补集重心对齐

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path(__file__).resolve().parents[4] / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, components_mask, find_components, get_bg


# Design analysis
# ---------------
# a) Concrete behavior and trace
# - The input still contains one large template component in color T plus a lattice
#   of singleton pixels, with exactly one singleton also colored T and the others O.
# - D3 changes the template from an ordinary sparse polyomino into a hole-aware mask:
#   inside its 3-4 by 3 bbox there must be 1-2 enclosed bg pixels. Ordinary copies
#   stamp only the template's filled cells; holes remain bg.
# - D8 changes only the origin copy: instead of recoloring the same filled mask, the
#   origin uses the template's bbox-level complement. Every local bg cell in the bbox
#   becomes O there, including enclosed holes and any edge-connected notches, while
#   original template cells become bg.
# - D11 changes the placement anchor: the lattice step (dr*h, dc*w) now targets the
#   centroid of the filled template pixels. The rendered bbox top-left is therefore
#   round((r0 + dr*h, c0 + dc*w) - centroid(mask)), so the output array usually shifts
#   relative to the original input template.
# - Example: if a 4x3 template has filled centroid (1.5, 1.0), then the origin copy
#   is drawn with top-left at (round(r0 - 1.5), round(c0 - 1.0)) = (r0 - 1, c0 - 1).
#   Ordinary copies keep the positive holey mask at similarly centroid-aligned sites,
#   but the origin copy paints the inverse support of that same bbox. Compared with the
#   source task, both "what is stamped" and "where the bbox lands" change together.
#
# b) Complexity calibration
# - The task keeps only three colors and one distinguished lattice origin, so the
#   solver still has a single reference correspondence to infer.
# - To keep the complement visually legible, the generator only uses holey templates
#   whose bbox complement has at least 3 cells. This avoids tiny 1-2 pixel negatives
#   that would be hard to parse from a few ARC examples.
# - The generator also rejects samples where centroid anchoring is too close to plain
#   bbox anchoring, so D11 is observable instead of collapsing into D3+D8 only.
# - Under the inherited fill-ratio bounds, 3x3 cannot realize an enclosed-hole template
#   while still spanning all four bbox edges, so the live height branch collapses to 4x3.
#   The generator therefore balances coverage across 1-hole vs 2-hole masks instead.
#
# c) Implicit-assumption audit
# - "Largest component is the template" still holds because the generator places only
#   one multi-pixel T component and all lattice markers remain area-1.
# - The original solver assumed every copy shares the same foreground support. D8 breaks
#   that, so the new solver counts non-origin positive stamps separately and renders the
#   origin complement afterwards.
# - The original solver also assumed bbox top-left anchoring. D11 breaks that, so all
#   translation math now flows through `_stamp_top_left(...)`.
# - Hole handling matters in two places: centroid computation must ignore holes because
#   it uses only filled pixels, while the origin complement must use the full bbox
#   complement. That is the key interaction between D3 and D8+D11.
#
# d) Placement legality
# - The generator enumerates only templates with 1-2 enclosed holes and valid fill ratio.
# - Every centroid-aligned bbox is kept fully inside the 22x22 grid.
# - Output bboxes are required to be pairwise disjoint; this keeps the positive copies,
#   the complement origin, and the hole signal visually separable.
# - The singleton cluster stays disjoint from the input template bbox with a bg gap, and
#   singleton pixels are forbidden from landing on any future colored output pixel.
#
# e) Observability / cleanliness / color inference
# - A human can see one holey template in the input and later see several copies of the
#   same geometry, except that exactly one copy becomes the local negative image.
# - Because the negative copy is also centroid-shifted, its bbox typically no longer
#   coincides with the input template. This makes D11 visible on the page rather than
#   being hidden behind the original object.
# - Color use stays structured: T marks the input template and ordinary output copies;
#   O marks non-origin lattice points and reappears as the origin-complement color.


def _round_nearest(x: float) -> int:
    """Round to the nearest integer using half-up behavior."""
    return int(math.floor(float(x) + 0.5))


def _cells_to_mask(cells: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _enclosed_bg_mask(filled_mask: np.ndarray) -> np.ndarray:
    """Return bbox-internal bg cells that are not 4-connected to the bbox edge."""
    h, w = filled_mask.shape
    holes = np.zeros((h, w), dtype=bool)
    for comp in components_mask(~filled_mask.astype(bool), connectivity=4):
        if comp.size == 0:
            continue
        touches_edge = any(
            int(r) == 0 or int(r) == h - 1 or int(c) == 0 or int(c) == w - 1
            for r, c in comp
        )
        if not touches_edge:
            holes[comp[:, 0], comp[:, 1]] = True
    return holes


def _template_centroid(templ_rel: list[tuple[int, int]]) -> tuple[float, float]:
    rs = [r for r, _c in templ_rel]
    cs = [c for _r, c in templ_rel]
    return float(sum(rs) / len(rs)), float(sum(cs) / len(cs))


def _stamp_top_left(
    r0: int,
    c0: int,
    dr: int,
    dc: int,
    h: int,
    w: int,
    centroid_r: float,
    centroid_c: float,
    *,
    anchor_mode: str,
) -> tuple[int, int]:
    anchor_r = int(r0 + dr * h)
    anchor_c = int(c0 + dc * w)
    if anchor_mode == "bbox":
        return anchor_r, anchor_c
    if anchor_mode == "centroid":
        return _round_nearest(anchor_r - centroid_r), _round_nearest(anchor_c - centroid_c)
    raise ValueError(f"Unknown anchor_mode: {anchor_mode}")


def _parse_instance(
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    list[tuple[int, int]],
    set[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int, int]],
]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_col = int(template["color"])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))

    templ_mask = (g[r0 : r1 + 1, c0 : c1 + 1] == int(templ_col))
    hole_mask = _enclosed_bg_mask(templ_mask)
    hole_pixels = int(hole_mask.sum())
    assert 1 <= hole_pixels <= 2

    templ_rel = [tuple(map(int, rc)) for rc in np.argwhere(templ_mask)]
    templ_set = set(templ_rel)
    comp_rel = [(rr, cc) for rr in range(h) for cc in range(w) if (rr, cc) not in templ_set]
    assert len(comp_rel) >= 1

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(comp["color"])))

    assert singletons
    return g, bg, templ_col, r0, c0, h, w, hole_pixels, templ_rel, templ_set, comp_rel, singletons


def _solve_with_modes(
    grid: np.ndarray,
    *,
    anchor_mode: str,
    origin_mode: str,
) -> np.ndarray:
    (
        g,
        bg,
        templ_col,
        r0,
        c0,
        h,
        w,
        _hole_pixels,
        templ_rel,
        templ_set,
        comp_rel,
        singletons,
    ) = _parse_instance(grid)
    H, W = g.shape

    centroid_r, centroid_c = _template_centroid(templ_rel)

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _ in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    positive_total = np.zeros((H, W), dtype=int)
    origin_positive = np.zeros((H, W), dtype=int)
    origin_top_left: tuple[int, int] | None = None

    for dr, dc, col in deltas:
        top_r, top_c = _stamp_top_left(
            r0, c0, dr, dc, h, w, centroid_r, centroid_c, anchor_mode=anchor_mode
        )
        is_origin = int(col) == int(templ_col)

        if is_origin and origin_mode == "complement":
            origin_top_left = (top_r, top_c)
            continue

        for rr, cc in templ_rel:
            R = int(top_r + rr)
            C = int(top_c + cc)
            if 0 <= R < H and 0 <= C < W:
                positive_total[R, C] += 1
                if is_origin:
                    origin_positive[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[positive_total > 0] = int(templ_col)
    out[positive_total >= 2] = int(overlap_col)

    if origin_mode == "recolor":
        out[(positive_total == 1) & (origin_positive > 0)] = int(overlap_col)
        return out

    assert origin_mode == "complement"
    assert origin_top_left is not None
    top_r, top_c = origin_top_left
    for rr, cc in comp_rel:
        R = int(top_r + rr)
        C = int(top_c + cc)
        if 0 <= R < H and 0 <= C < W:
            out[R, C] = int(overlap_col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a holey template from a singleton lattice, but place every copy by centroid alignment and render the origin as a complement.

    Input structure:
    - The grid has one large monochrome template in color T; its bbox is 3-4 by 3 and contains 1-2 enclosed bg holes.
    - Separate singleton pixels form a uniformly spaced lattice; exactly one singleton is also T and the rest are color O.
    - The hole pattern belongs to the template shape itself, so ordinary copies preserve those empty cells.

    Rule:
    - Measure every singleton offset relative to the unique T-colored singleton and normalize by the lattice spacing.
    - Use each offset to move a centroid anchor by (dr*h, dc*w), where h and w are the template bbox dimensions.
    - Convert that anchor back to a bbox top-left by subtracting the template centroid and rounding to grid cells.
    - Non-origin lattice points stamp the template's filled mask in T. The origin point instead stamps the bbox-level complement of the template in O, and any overlap among positive stamps is also O.

    Key insight:
    - The copied object is not just "the bbox" and not just "the positive sprite": holes change the centroid used for placement, while the distinguished origin copy flips filled-vs-empty inside the same bbox.
    """
    return _solve_with_modes(grid, anchor_mode="centroid", origin_mode="complement")


GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
TEMPLATE_HOLE_PIXELS = (1, 2)
MIN_COMPLEMENT_AREA = 3
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

SPANNING_RETRIES = 40
SINGLETON_RETRIES = 60
POSITION_RETRIES = 80
GENERATE_RETRIES = 400


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of n cells inside a rows x cols box."""
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.append((nr, nc))
    while len(cells) < n and frontier:
        cell = random.choice(frontier)
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                frontier.append((nr, nc))
    return list(cells) if len(cells) == n else None


def _spanning_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Polyomino of n cells that touches all four edges of the box."""
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _template_candidates(h: int, w: int) -> dict[int, list[list[tuple[int, int]]]]:
    """Enumerate all legal holey templates grouped by enclosed-hole count."""
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))

    grouped: dict[int, list[list[tuple[int, int]]]] = {1: [], 2: []}
    for bits in range(1, 1 << total):
        cells = [(idx // w, idx % w) for idx in range(total) if (bits >> idx) & 1]
        area = len(cells)
        if area < lo or area > hi:
            continue

        mask = _cells_to_mask(cells, h, w)
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        if not mask[0, :].any() or not mask[h - 1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, w - 1].any():
            continue

        hole_pixels = int(_enclosed_bg_mask(mask).sum())
        complement_area = int(total - area)
        if hole_pixels not in grouped:
            continue
        if complement_area < MIN_COMPLEMENT_AREA:
            continue
        grouped[hole_pixels].append(cells)

    return grouped


TEMPLATE_LIBRARY = {h: _template_candidates(h, TEMPLATE_W) for h in TEMPLATE_H_CHOICES}
ACTIVE_TEMPLATE_HEIGHTS = [h for h in TEMPLATE_H_CHOICES if any(TEMPLATE_LIBRARY[h].values())]


def _gen_template() -> tuple[int, int, list[tuple[int, int]], int] | None:
    hole_choices = list(TEMPLATE_HOLE_PIXELS)
    random.shuffle(hole_choices)
    for desired_holes in hole_choices:
        valid_heights = [h for h in ACTIVE_TEMPLATE_HEIGHTS if TEMPLATE_LIBRARY[h][desired_holes]]
        if not valid_heights:
            continue
        h = random.choice(valid_heights)
        cells = random.choice(TEMPLATE_LIBRARY[h][desired_holes])
        return h, TEMPLATE_W, cells, desired_holes
    return None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Lattice polyomino of n cells with a strictly interior origin."""
    ext_r = random.randint(*LATTICE_EXTENT_RANGE)
    ext_c = random.randint(*LATTICE_EXTENT_RANGE)
    cells = _spanning_polyomino(n, ext_r, ext_c)
    if cells is None:
        return None
    interior = [(r, c) for r, c in cells if 0 < r < ext_r - 1 and 0 < c < ext_c - 1]
    if not interior:
        return None
    return cells, random.choice(interior)


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _valid_template_origins(
    deltas: list[tuple[int, int]],
    h: int,
    w: int,
    centroid_r: float,
    centroid_c: float,
) -> tuple[list[int], list[int]]:
    valid_r0: list[int] = []
    valid_c0: list[int] = []

    for r0 in range(GRID_SIZE - h + 1):
        ok = True
        for dr, _dc in deltas:
            top_r = _stamp_top_left(
                r0, 0, dr, 0, h, w, centroid_r, centroid_c, anchor_mode="centroid"
            )[0]
            if top_r < 0 or top_r + h - 1 >= GRID_SIZE:
                ok = False
                break
        if ok:
            valid_r0.append(r0)

    for c0 in range(GRID_SIZE - w + 1):
        ok = True
        for _dr, dc in deltas:
            top_c = _stamp_top_left(
                0, c0, 0, dc, h, w, centroid_r, centroid_c, anchor_mode="centroid"
            )[1]
            if top_c < 0 or top_c + w - 1 >= GRID_SIZE:
                ok = False
                break
        if ok:
            valid_c0.append(c0)

    return valid_r0, valid_c0


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _build_stamp_layout(
    deltas: list[tuple[int, int]],
    templ_cells: list[tuple[int, int]],
    comp_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    centroid_r: float,
    centroid_c: float,
    *,
    anchor_mode: str,
) -> list[dict]:
    layout: list[dict] = []
    for dr, dc in deltas:
        top_r, top_c = _stamp_top_left(
            r0, c0, dr, dc, h, w, centroid_r, centroid_c, anchor_mode=anchor_mode
        )
        is_origin = (dr == 0 and dc == 0)
        support = comp_cells if is_origin else templ_cells
        bbox = (top_r, top_c, top_r + h - 1, top_c + w - 1)
        pixels = {(top_r + rr, top_c + cc) for rr, cc in support}
        layout.append(
            {
                "dr": int(dr),
                "dc": int(dc),
                "is_origin": bool(is_origin),
                "bbox": bbox,
                "pixels": pixels,
            }
        )
    return layout


def _layout_ok(layout: list[dict]) -> bool:
    bboxes = [item["bbox"] for item in layout]
    for r0, c0, r1, c1 in bboxes:
        if r0 < 0 or c0 < 0 or r1 >= GRID_SIZE or c1 >= GRID_SIZE:
            return False
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _boxes_overlap(bboxes[i], bboxes[j]):
                return False
    return True


def _future_pixels(layout: list[dict]) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    for item in layout:
        pixels.update(item["pixels"])
    return pixels


def _place_singletons(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    future_fg: set[tuple[int, int]],
    min_centroid_diff: int,
    min_legacy_diff: int,
) -> np.ndarray | None:
    """Place the singleton lattice while preserving input/output separability."""
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = orr + min_dr * GUNIT
        sr1 = orr + max_dr * GUNIT
        sc0 = orc + min_dc * GUNIT
        sc1 = orc + max_dc * GUNIT
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        singleton_pts = {(orr + dr * GUNIT, orc + dc * GUNIT) for dr, dc in deltas}
        if singleton_pts & future_fg:
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = templ_col
        for dr, dc in deltas:
            rr = orr + dr * GUNIT
            cc = orc + dc * GUNIT
            grid[rr, cc] = templ_col if (dr == 0 and dc == 0) else overlap_col

        try:
            fused_out = main(grid)
            bbox_complement_out = _solve_with_modes(
                grid, anchor_mode="bbox", origin_mode="complement"
            )
            legacy_out = _solve_with_modes(grid, anchor_mode="bbox", origin_mode="recolor")
        except Exception:
            continue

        centroid_diff = int(np.count_nonzero(fused_out != bbox_complement_out))
        legacy_diff = int(np.count_nonzero(fused_out != legacy_out))
        if centroid_diff < min_centroid_diff or legacy_diff < min_legacy_diff:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    templ = _gen_template()
    if templ is None:
        return None
    h, w, templ_cells, hole_pixels = templ
    templ_set = set(templ_cells)
    comp_cells = [(rr, cc) for rr in range(h) for cc in range(w) if (rr, cc) not in templ_set]

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(r - origin[0], c - origin[1]) for r, c in lattice_cells]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    centroid_r, centroid_c = _template_centroid(templ_cells)
    valid_r0, valid_c0 = _valid_template_origins(deltas, h, w, centroid_r, centroid_c)
    if not valid_r0 or not valid_c0:
        return None

    origin_shift = (
        _stamp_top_left(r0=0, c0=0, dr=0, dc=0, h=h, w=w, centroid_r=centroid_r, centroid_c=centroid_c, anchor_mode="centroid")[0],
        _stamp_top_left(r0=0, c0=0, dr=0, dc=0, h=h, w=w, centroid_r=centroid_r, centroid_c=centroid_c, anchor_mode="centroid")[1],
    )
    if origin_shift == (0, 0):
        return None

    for _ in range(POSITION_RETRIES):
        r0 = random.choice(valid_r0)
        c0 = random.choice(valid_c0)
        layout = _build_stamp_layout(
            deltas,
            templ_cells,
            comp_cells,
            r0,
            c0,
            h,
            w,
            centroid_r,
            centroid_c,
            anchor_mode="centroid",
        )
        if not _layout_ok(layout):
            continue

        bbox_layout = _build_stamp_layout(
            deltas,
            templ_cells,
            comp_cells,
            r0,
            c0,
            h,
            w,
            centroid_r,
            centroid_c,
            anchor_mode="bbox",
        )
        centroid_layout_diff = len(_future_pixels(layout) ^ _future_pixels(bbox_layout))
        if centroid_layout_diff < max(4, len(comp_cells)):
            continue

        future_fg = _future_pixels(layout)
        return _place_singletons(
            deltas,
            min_dr,
            max_dr,
            min_dc,
            max_dc,
            templ_cells,
            r0,
            c0,
            h,
            w,
            bg,
            templ_col,
            overlap_col,
            future_fg,
            min_centroid_diff=max(4, len(comp_cells)),
            min_legacy_diff=max(6, len(comp_cells) + hole_pixels),
        )

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
