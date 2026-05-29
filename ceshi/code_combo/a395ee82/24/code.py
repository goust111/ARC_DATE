# Combo variant of a395ee82 (index 24)
# combo_spec: D15/2+D10/0
# Name: 边缘模板紧凑裁剪

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D15/2 + D10/0 combo
# ============================================
# a) Concrete behavior and trace
#    The source task reads one monochrome template in color T plus a coarse
#    singleton lattice with one T-colored origin pixel and several O-colored
#    support pixels. The solver interprets each singleton offset as one template
#    translation measured in units of the template bbox size.
#    The D15 change forces the visible template sample to sit flush against one
#    grid border. With the original lattice spec, the origin singleton is still
#    strictly interior, so there are always offsets on both sides of the origin.
#    That means some translated copies lie outside the 22x22 canvas on the
#    outward side. The original solver already clips every stamped cell to
#    in-bounds coordinates, so these outward copies simply disappear. The D10
#    change then crops the rendered result to the minimal bbox of the visible
#    stamped cells. Example: template at the top edge with lattice deltas
#    including dr in {-1, 0, 1}. The dr=-1 copy is entirely above the grid and
#    vanishes, while dr in {0, 1} remain. The crop removes unused left/right and
#    bottom background, but not the already-flush top side.
# b) Complexity calibration
#    The combo is kept to a single clipped axis. The template touches exactly
#    one border, while the orthogonal axis keeps an interior margin. This makes
#    the visual change obvious without introducing multi-sided clipping chaos.
#    A solver can still reuse the original lattice reconstruction idea; the new
#    burden is inferring that only in-canvas stamped pixels survive and then the
#    result is tightly cropped.
# c) Hidden assumptions audit in main()
#    - `template = max(comps, key=area)` still works because the input still has
#      one large template component and the rest are singletons.
#    - Reading `templ_col` from that component still works because the template
#      remains monochrome.
#    - Detecting the origin singleton by `col == templ_col` still works because
#      exactly one singleton keeps T.
#    - The gcd-based lattice spacing recovery still works because singleton
#      spacing is unchanged.
#    - The key changed assumption is output coverage. In the source generator all
#      translated copies were fully in-bounds. Here that is deliberately false on
#      one side, so the solver must rely on the existing bounds check during
#      rendering. Tight-crop must run after this clipped render, not before.
# d) Placement legality in the generator
#    The generator guarantees: the input template bbox is fully in-bounds and
#    touches exactly one border; singleton points are in-bounds, mutually
#    separated on the intended lattice, and stay disjoint from the template with
#    at least one background cell gap; the visible stamped union touches only the
#    same border as the template; at least one lattice offset produces a fully
#    invisible copy so the edge effect is real rather than nominal.
# e) Visual observability
#    Across examples, a human sees the template sample used as a border-anchored
#    reference piece and notices that the output is the reconstructed stamp field
#    with one side missing, then tightly cropped. Because the touched border can
#    vary among top/bottom/left/right while the clipping rule stays consistent,
#    the border relation is supported by repeated structure instead of hidden
#    numeric facts.
# f) Visual cleanliness
#    Only one clipped axis is allowed, and the orthogonal axis keeps margins, so
#    the cropped output is a clean asymmetric stamp band rather than a cluttered
#    fragment. Singletons are kept away from the template sample to preserve the
#    reference-vs-lattice reading.
# g) Color-change inference
#    Colors stay simple: T for the template and visible non-origin copies, O for
#    the visible origin copy and any overlaps. The combo adds no new color rule,
#    so the only inference jump is geometric: off-canvas copies disappear, and
#    the surviving figure is cropped tightly.


GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2
ORTHO_MARGIN = 2
CLUSTER_BORDER_MARGIN = 1
EDGE_OPTIONS = ("top", "bottom", "left", "right")
_EDGE_BAG: list[str] = []

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
SINGLETON_RETRIES = 60
GENERATE_RETRIES = 400


def _foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touching_edges(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[str, ...]:
    r0, c0, r1, c1 = map(int, bbox)
    H, W = map(int, shape)
    edges: list[str] = []
    if r0 == 0:
        edges.append("top")
    if r1 == H - 1:
        edges.append("bottom")
    if c0 == 0:
        edges.append("left")
    if c1 == W - 1:
        edges.append("right")
    return tuple(edges)


def _extract_problem(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_coords = template["coords"]
    templ_col = int(g[int(templ_coords[0, 0]), int(templ_coords[0, 1])])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))
    templ_rel = [(int(r) - int(r0), int(c) - int(c0)) for r, c in templ_coords]

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    assert singletons
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

    return {
        "grid": g,
        "bg": bg,
        "shape": (int(H), int(W)),
        "templ_col": int(templ_col),
        "overlap_col": int(overlap_col),
        "anchor": (int(r0), int(c0)),
        "size": (int(h), int(w)),
        "templ_rel": templ_rel,
        "deltas": deltas,
        "singleton_count": int(len(singletons)),
    }


def _render_uncropped_from_info(info: dict[str, object]) -> np.ndarray:
    H, W = info["shape"]
    bg = int(info["bg"])
    templ_col = int(info["templ_col"])
    overlap_col = int(info["overlap_col"])
    r0, c0 = info["anchor"]
    h, w = info["size"]
    templ_rel = info["templ_rel"]
    deltas = info["deltas"]

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        is_origin = int(col) == int(templ_col)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


def _solve_uncropped(grid: np.ndarray) -> np.ndarray:
    return _render_uncropped_from_info(_extract_problem(grid))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct an edge-anchored template stamp layout from a singleton lattice, then tight-crop the visible result.

    Input structure:
    - One large monochrome template component in color `T` sits flush against exactly one border of the 22x22 grid.
    - Elsewhere, singleton pixels form a uniformly spaced lattice; exactly one singleton is also `T`, and the rest are `O`.
    - The `T` singleton is the lattice origin, while the border template is the reference sprite to be translated.

    Rule:
    - Measure every singleton offset relative to the `T` singleton and normalize by the lattice spacing.
    - Use those normalized offsets, scaled by the template bbox height/width, to place translated copies of the template from the border anchor position.
    - Keep only in-bounds stamped pixels; the origin copy is recolored to `O`, and any multiply covered cells are also recolored to `O`.
    - After rendering on the original 22x22 canvas, crop away every all-background border row and column.

    Key insight:
    - The singleton lattice still tells you the full translation pattern, but because the sample template starts on a border, the outward-side copies fall completely off-canvas and disappear; the answer is the tight crop of the surviving in-grid stamp pattern.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = _solve_uncropped(g)
    r0, c0, r1, c1 = _foreground_bbox(out != int(bg))
    return out[r0 : r1 + 1, c0 : c1 + 1]


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier = []
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
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    if lo > hi:
        return None
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(lo, hi)
        cells = _spanning_polyomino(area, h, w)
        if cells is not None:
            return cells
    return None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
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


def _stamp_metrics(
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
    shape: tuple[int, int] = (GRID_SIZE, GRID_SIZE),
) -> dict[str, object]:
    H, W = map(int, shape)
    occupied: set[tuple[int, int]] = set()
    visible_count = 0
    for dr, dc in deltas:
        has_visible = False
        for tr, tc in templ_cells:
            R = int(r0 + dr * h + tr)
            C = int(c0 + dc * w + tc)
            if 0 <= R < H and 0 <= C < W:
                occupied.add((R, C))
                has_visible = True
        visible_count += int(has_visible)

    assert occupied
    rs = [r for r, _c in occupied]
    cs = [c for _r, c in occupied]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    return {
        "occupied": occupied,
        "visible_count": int(visible_count),
        "hidden_count": int(len(deltas) - visible_count),
        "bbox": tuple(map(int, bbox)),
        "touch_edges": _touching_edges(bbox, shape),
    }


def _choose_edge_anchor(
    h: int,
    w: int,
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
) -> tuple[str, int, int] | None:
    candidates: list[tuple[str, int, int, int]] = []

    c_lo = ORTHO_MARGIN - min_dc * w
    c_hi = GRID_SIZE - ORTHO_MARGIN - max_dc * w - w
    if c_lo <= c_hi:
        candidates.append(("top", 0, c_lo, c_hi))
        candidates.append(("bottom", GRID_SIZE - h, c_lo, c_hi))

    r_lo = ORTHO_MARGIN - min_dr * h
    r_hi = GRID_SIZE - ORTHO_MARGIN - max_dr * h - h
    if r_lo <= r_hi:
        candidates.append(("left", r_lo, 0, r_hi))
        candidates.append(("right", r_lo, GRID_SIZE - w, r_hi))

    if not candidates:
        return None

    candidate_map = {edge: (edge, fixed_a, fixed_b, span_hi) for edge, fixed_a, fixed_b, span_hi in candidates}
    global _EDGE_BAG
    if not _EDGE_BAG:
        _EDGE_BAG = list(EDGE_OPTIONS)
        random.shuffle(_EDGE_BAG)

    edge = None
    for idx, preferred in enumerate(_EDGE_BAG):
        if preferred in candidate_map:
            edge = preferred
            del _EDGE_BAG[idx]
            break

    if edge is None:
        edge = random.choice(list(candidate_map))

    edge, fixed_a, fixed_b, span_hi = candidate_map[edge]
    if edge in ("top", "bottom"):
        c0 = random.randint(fixed_b, span_hi)
        r0 = fixed_a
    else:
        r0 = random.randint(fixed_a, span_hi)
        c0 = fixed_b
    return edge, int(r0), int(c0)


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
    T: int,
    O: int,
    visible_occupied: set[tuple[int, int]],
    edge: str,
) -> np.ndarray | None:
    orr_lo = max(-min_dr * GUNIT, CLUSTER_BORDER_MARGIN - min_dr * GUNIT)
    orr_hi = min(
        GRID_SIZE - 1 - max_dr * GUNIT,
        GRID_SIZE - 1 - CLUSTER_BORDER_MARGIN - max_dr * GUNIT,
    )
    orc_lo = max(-min_dc * GUNIT, CLUSTER_BORDER_MARGIN - min_dc * GUNIT)
    orc_hi = min(
        GRID_SIZE - 1 - max_dc * GUNIT,
        GRID_SIZE - 1 - CLUSTER_BORDER_MARGIN - max_dc * GUNIT,
    )
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

        if any((orr + dr * GUNIT, orc + dc * GUNIT) in visible_occupied for dr, dc in deltas):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = T
        for dr, dc in deltas:
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = T if (dr == 0 and dc == 0) else O

        try:
            cropped = main(grid)
            metrics = _variant_metrics(grid)
            if cropped.shape == (GRID_SIZE, GRID_SIZE):
                continue
            if metrics["template_edges"] != (edge,):
                continue
            if metrics["touch_edges"] != (edge,):
                continue
            if int(metrics["hidden_count"]) < 1:
                continue
        except Exception:
            return None
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    bg, T, O = random.sample(range(10), 3)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(r - origin[0], c - origin[1]) for r, c in lattice_cells]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    anchor = _choose_edge_anchor(h, w, min_dr, max_dr, min_dc, max_dc)
    if anchor is None:
        return None
    edge, r0, c0 = anchor

    metrics = _stamp_metrics(templ_cells, r0, c0, h, w, deltas)
    if metrics["touch_edges"] != (edge,):
        return None
    if int(metrics["hidden_count"]) < 1:
        return None

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
        T,
        O,
        metrics["occupied"],
        edge,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")


def _variant_metrics(grid: np.ndarray) -> dict[str, object]:
    info = _extract_problem(grid)
    r0, c0 = info["anchor"]
    h, w = info["size"]
    metrics = _stamp_metrics(
        info["templ_rel"],
        r0,
        c0,
        h,
        w,
        [(dr, dc) for dr, dc, _col in info["deltas"]],
        info["shape"],
    )
    template_bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
    metrics["template_edges"] = _touching_edges(template_bbox, info["shape"])
    metrics["crop_shape"] = tuple(map(int, main(grid).shape))
    metrics["singleton_count"] = int(info["singleton_count"])
    return metrics
