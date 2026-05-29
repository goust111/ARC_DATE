# Combo variant of a395ee82 (index 2)
# combo_spec: D2/2+D5/1+D10/0
# Name: 不均匀消失裁剪

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) concretization + behavior trace
#    - Input still has one large monochrome template in color T and 5-6 isolated
#      singleton markers. Exactly one singleton is also T and marks the source
#      origin; the others are O.
#    - D2/2 changes the source scaffold: singleton rows/cols are no longer
#      uniformly spaced. Adjacent visible gaps are only 2 or 3 pixels, which
#      decode locally to 1 or 2 stamp-steps.
#    - D5/1 changes the visible interaction rule. To make overlap observable, a
#      decoded step of one template unit now moves by (h-1, w-1) instead of
#      (h, w), so neighboring copies share a strip. Multiply covered cells
#      disappear to bg instead of turning into O.
#    - D10/0 changes the final canvas: after rendering and cancellation, crop to
#      the tight bbox of the surviving foreground.
#    - Concrete trace: if visible singleton rows are [5, 7, 10], their gaps are
#      [2, 3], so decoded row steps are [0, 1, 3]. With origin at row 7, those
#      become offsets [-1, 0, +2]. If h=4 and w=3, the fused variant translates
#      by row stride 3 and col stride 2; the source task would have used a
#      single global unit plus stride (4, 3), which removes the interference and
#      returns a full 22x22 canvas. Here the same decoded offsets create shared
#      bands, those shared cells vanish, and the answer is cropped around the
#      remaining T/O pixels.
#
# b) complexity calibration
#    - Only two source gap categories are used: 2->1 step and 3->2 steps. That
#      keeps the decoding human-visible and repeated across samples.
#    - The generator requires both gap categories to appear, so the irregular
#      scaffold cannot silently collapse back to the uniform original task.
#    - It also requires visible cancellation, visible origin-only O cells, and
#      visible non-origin T cells. This prevents degenerate all-bg or single-
#      color crops and makes the D5 behavior obvious.
#
# c) hidden assumptions audited in main()
#    - Original gcd-based normalization is invalid: with mixed 2/3 gaps, one
#      global lattice unit no longer recovers the intended source offsets.
#    - Original stride `(h, w)` is too large for D5 observability. The combo
#      must instead use `(h-1, w-1)` once decoded offsets are known.
#    - Original `out[total >= 2] = O` is wrong here: overlap must revert to bg.
#    - Original full-canvas return is wrong here: after rendering, only the
#      minimal non-bg bbox is returned.
#    - New assumptions introduced by the fused solver:
#      * sorted unique singleton rows/cols expose all warped axis levels
#      * every adjacent source gap is in {2, 3}
#      * the T-colored singleton is unique
#      * at least one rendered pixel survives cancellation
#    - The generator explicitly guarantees each of these.
#
# d) placement legality
#    - Template and singleton cluster stay disjoint with at least one bg cell
#      between their bboxes.
#    - All singleton pixels fit inside the 22x22 input, and all decoded stamp
#      copies fit inside the 22x22 working canvas before cropping.
#    - Singletons are also kept off the full stamp footprint for cleanliness.
#
# e) visual observability
#    - Humans can see the irregular source because singleton rows/cols contain
#      both short and long gaps.
#    - Humans can see the destructive overlap because internal holes/strips turn
#      into bg rather than O.
#    - Humans can see the crop because outputs are smaller than 22x22 and sized
#      to the surviving stamped figure.
#
# f) cleanliness
#    - The generator bounds decoded span so the figure stays compact.
#    - Only three color roles remain: bg, T, O. No hidden ordering by color id.
#    - Minimum visibility checks keep the cropped result legible instead of
#      turning into a noisy fringe.
#
# g) color-change inference
#    - O has a single structural role: pixels seen only in the origin copy.
#      T marks visible non-origin-only pixels. bg marks untouched cells and
#      canceled overlaps. The color behavior is therefore inferable from the
#      geometry rather than from arbitrary color identities.
#
# h) fusion-specific non-degeneration checks
#    - Non-uniform source is enforced by requiring both 2-pixel and 3-pixel
#      source gaps.
#    - Cancellation is enforced by requiring at least `MIN_CANCEL_CELLS`.
#    - Tight crop is enforced by placing the full rendered footprint away from
#      every border and rejecting uncropped 22x22 outputs.
#
# i) new main() assumption review vs generator
#    - `_axis_step_map()` assumes a complete sorted list of warped axis levels;
#      the spanning coarse polyomino guarantees every intermediate coarse row
#      and col is occupied by at least one singleton.
#    - `_coverage_maps()` assumes the decoded deltas describe one origin copy at
#      (0,0); the generator builds deltas relative to a chosen interior origin.
#    - `_tight_crop()` assumes some foreground survives; the generator enforces
#      positive counts for both origin-visible and non-origin-visible pixels.


GAP_TO_STEP = {2: 1, 3: 2}
SOURCE_GAP_CHOICES = tuple(sorted(GAP_TO_STEP))

# --------------- generator ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
OUTPUT_MARGIN = 2

MIN_CANCEL_CELLS = 2
MIN_ORIGIN_VISIBLE = 2
MIN_TEMPLATE_VISIBLE = 4

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
SINGLETON_RETRIES = 50
GAP_PATTERN_RETRIES = 80
GENERATE_RETRIES = 1000


def _axis_step_map(values: list[int]) -> dict[int, int]:
    """Decode visible warped coordinates into cumulative coarse steps."""
    vals = sorted({int(v) for v in values})
    assert len(vals) >= 2

    out = {vals[0]: 0}
    acc = 0
    for prev, cur in zip(vals, vals[1:]):
        gap = int(cur - prev)
        assert gap in GAP_TO_STEP
        acc += int(GAP_TO_STEP[gap])
        out[cur] = acc
    return out


def _stamp_steps(h: int, w: int) -> tuple[int, int]:
    """Reduced stride makes adjacent decoded copies overlap and cancel."""
    return max(1, int(h) - 1), max(1, int(w) - 1)


def _coverage_maps(
    shape: tuple[int, int],
    base_r: int,
    base_c: int,
    templ_rel: list[tuple[int, int]],
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return total coverage and origin-copy coverage for decoded deltas."""
    H, W = map(int, shape)
    step_r, step_c = _stamp_steps(h, w)
    total = np.zeros((H, W), dtype=int)
    origin = np.zeros((H, W), dtype=int)
    for dr, dc in deltas:
        off_r = int(dr * step_r)
        off_c = int(dc * step_c)
        is_origin = int(dr) == 0 and int(dc) == 0
        for rr, cc in templ_rel:
            R = int(base_r + off_r + rr)
            C = int(base_c + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    origin[R, C] += 1
    return total, origin


def _render_uncropped(
    bg: int,
    templ_col: int,
    overlap_col: int,
    total: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Render interference result before the final crop."""
    out = np.full(total.shape, int(bg), dtype=int)
    out[(total == 1) & (origin == 0)] = int(templ_col)
    out[(total == 1) & (origin > 0)] = int(overlap_col)
    return out


def _foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _foreground_bbox(grid != int(bg))
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _variant_stats(
    shape: tuple[int, int],
    base_r: int,
    base_c: int,
    templ_rel: list[tuple[int, int]],
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
) -> dict[str, int]:
    total, origin = _coverage_maps(shape, base_r, base_c, templ_rel, h, w, deltas)
    return {
        "cancel_cells": int(np.sum(total >= 2)),
        "origin_visible": int(np.sum((total == 1) & (origin > 0))),
        "templ_visible": int(np.sum((total == 1) & (origin == 0))),
        "visible_cells": int(np.sum(total == 1)),
    }


def _solve_uncropped(grid: np.ndarray) -> np.ndarray:
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
        if comp is template or int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    row_to_step = _axis_step_map([r for r, _c, _col in singletons])
    col_to_step = _axis_step_map([c for _r, c, _col in singletons])
    origin_sr = int(row_to_step[int(orr)])
    origin_sc = int(col_to_step[int(orc)])
    deltas = [
        (
            int(row_to_step[int(r)] - origin_sr),
            int(col_to_step[int(c)] - origin_sc),
        )
        for (r, c, _col) in singletons
    ]

    total, origin_cover = _coverage_maps((H, W), r0, c0, templ_rel, h, w, deltas)
    return _render_uncropped(bg, templ_col, overlap_col, total, origin_cover)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated template copies from an irregular singleton scaffold, erase overlaps, then crop tightly.

    Input structure:
    - `bg` is the background. One large monochrome component in color `T` is the template sprite.
    - Elsewhere are 5-6 isolated singleton pixels: exactly one singleton is also `T` and marks the source origin; the rest are `O`.
    - Singleton rows and columns are warped rather than uniform: each adjacent visible gap is either 2 or 3 pixels.

    Rule:
    - Decode the sorted singleton rows and columns as local source axes where gap `2 -> 1` stamp-step and gap `3 -> 2` stamp-steps.
    - Convert each singleton into a decoded offset from the unique `T` singleton, then place a template copy at that offset.
    - Use a stride of `(template height - 1, template width - 1)` so neighboring decoded copies can overlap.
    - Pixels covered only by the origin copy become `O`; pixels covered only by non-origin copies stay `T`; pixels covered by two or more copies disappear to `bg`.
    - After rendering, crop away every all-background border row and column.

    Key insight: The singleton scaffold no longer reveals one global lattice unit. Instead, short/long source gaps decode local step counts, and those decoded copies interfere destructively before the final answer is tight-cropped.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    uncropped = _solve_uncropped(g)
    return _tight_crop(uncropped, bg)


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    """Grow a random 4-connected polyomino of n cells in a rows x cols grid."""
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
    """Polyomino of n cells that spans exactly rows x cols."""
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
    """4-connected template shape in h x w, spanning bbox, within area/fill bounds."""
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


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int], int, int] | None:
    """Coarse polyomino of n cells with a strictly interior origin."""
    ext_r = random.randint(*LATTICE_EXTENT_RANGE)
    ext_c = random.randint(*LATTICE_EXTENT_RANGE)
    cells = _spanning_polyomino(n, ext_r, ext_c)
    if cells is None:
        return None
    interior = [(r, c) for r, c in cells if 0 < r < ext_r - 1 and 0 < c < ext_c - 1]
    if not interior:
        return None
    return cells, random.choice(interior), ext_r, ext_c


def _gen_gap_layout(ext_r: int, ext_c: int) -> tuple[list[int], list[int]] | None:
    """Choose visible source gaps while keeping the irregular decoding human-visible."""
    for _ in range(GAP_PATTERN_RETRIES):
        row_gaps = [random.choice(SOURCE_GAP_CHOICES) for _ in range(ext_r - 1)]
        col_gaps = [random.choice(SOURCE_GAP_CHOICES) for _ in range(ext_c - 1)]
        if set(row_gaps + col_gaps) != set(SOURCE_GAP_CHOICES):
            continue
        if len(set(row_gaps)) == 1 and len(set(col_gaps)) == 1:
            continue
        if sum(GAP_TO_STEP[g] for g in row_gaps) > 4:
            continue
        if sum(GAP_TO_STEP[g] for g in col_gaps) > 4:
            continue
        return row_gaps, col_gaps
    return None


def _levels_from_gaps(gaps: list[int]) -> tuple[list[int], list[int]]:
    """Return raw visible coordinates and decoded coarse-step coordinates."""
    src = [0]
    steps = [0]
    for gap in gaps:
        src.append(int(src[-1] + int(gap)))
        steps.append(int(steps[-1] + GAP_TO_STEP[int(gap)]))
    return src, steps


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return int(min_dr), int(max_dr), int(min_dc), int(max_dc)


def _footprint_points(
    base_r: int,
    base_c: int,
    templ_rel: list[tuple[int, int]],
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Cells touched by at least one decoded stamp before cancellation."""
    step_r, step_c = _stamp_steps(h, w)
    pts: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        for rr, cc in templ_rel:
            pts.add((int(base_r + dr * step_r + rr), int(base_c + dc * step_c + cc)))
    return pts


def _cropped_expected(
    bg: int,
    templ_col: int,
    overlap_col: int,
    shape: tuple[int, int],
    base_r: int,
    base_c: int,
    templ_rel: list[tuple[int, int]],
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
) -> np.ndarray:
    total, origin = _coverage_maps(shape, base_r, base_c, templ_rel, h, w, deltas)
    out = _render_uncropped(bg, templ_col, overlap_col, total, origin)
    return _tight_crop(out, bg)


def _place_singletons(
    raw_deltas: list[tuple[int, int]],
    raw_bounds: tuple[int, int, int, int],
    templ_rel: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    footprint: set[tuple[int, int]],
    expected: np.ndarray,
) -> np.ndarray | None:
    """Place the warped singleton scaffold away from the template and footprint."""
    min_dr, max_dr, min_dc, max_dc = raw_bounds
    orr_lo, orr_hi = -min_dr, GRID_SIZE - 1 - max_dr
    orc_lo, orc_hi = -min_dc, GRID_SIZE - 1 - max_dc
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = orr + min_dr
        sr1 = orr + max_dr
        sc0 = orc + min_dc
        sc1 = orc + max_dc
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        src_points = [(int(orr + dr), int(orc + dc)) for dr, dc in raw_deltas]
        if any(pt in footprint for pt in src_points):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for rr, cc in templ_rel:
            grid[int(r0 + rr), int(c0 + cc)] = int(templ_col)
        for rr, cc in src_points:
            grid[int(rr), int(cc)] = int(overlap_col)
        grid[int(orr), int(orc)] = int(templ_col)

        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt for the fused irregular/cancel/crop variant."""
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_rel = _gen_template(h, w)
    if templ_rel is None:
        return None

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing)
    if lat is None:
        return None
    lattice_cells, origin, ext_r, ext_c = lat

    gaps = _gen_gap_layout(ext_r, ext_c)
    if gaps is None:
        return None
    row_gaps, col_gaps = gaps
    row_src, row_steps = _levels_from_gaps(row_gaps)
    col_src, col_steps = _levels_from_gaps(col_gaps)

    raw_deltas: list[tuple[int, int]] = []
    step_deltas: list[tuple[int, int]] = []
    for lr, lc in lattice_cells:
        raw_deltas.append(
            (
                int(row_src[lr] - row_src[origin[0]]),
                int(col_src[lc] - col_src[origin[1]]),
            )
        )
        step_deltas.append(
            (
                int(row_steps[lr] - row_steps[origin[0]]),
                int(col_steps[lc] - col_steps[origin[1]]),
            )
        )

    min_sdr, max_sdr, min_sdc, max_sdc = _delta_bounds(step_deltas)
    step_r, step_c = _stamp_steps(h, w)
    r0_lo = OUTPUT_MARGIN - min_sdr * step_r
    r0_hi = GRID_SIZE - OUTPUT_MARGIN - max_sdr * step_r - h
    c0_lo = OUTPUT_MARGIN - min_sdc * step_c
    c0_hi = GRID_SIZE - OUTPUT_MARGIN - max_sdc * step_c - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)

    stats = _variant_stats((GRID_SIZE, GRID_SIZE), r0, c0, templ_rel, h, w, step_deltas)
    if stats["cancel_cells"] < MIN_CANCEL_CELLS:
        return None
    if stats["origin_visible"] < MIN_ORIGIN_VISIBLE:
        return None
    if stats["templ_visible"] < MIN_TEMPLATE_VISIBLE:
        return None

    expected = _cropped_expected(
        bg,
        templ_col,
        overlap_col,
        (GRID_SIZE, GRID_SIZE),
        r0,
        c0,
        templ_rel,
        h,
        w,
        step_deltas,
    )
    if expected.shape == (GRID_SIZE, GRID_SIZE):
        return None

    footprint = _footprint_points(r0, c0, templ_rel, h, w, step_deltas)
    return _place_singletons(
        raw_deltas=raw_deltas,
        raw_bounds=_delta_bounds(raw_deltas),
        templ_rel=templ_rel,
        r0=r0,
        c0=c0,
        h=h,
        w=w,
        bg=bg,
        templ_col=templ_col,
        overlap_col=overlap_col,
        footprint=footprint,
        expected=expected,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
