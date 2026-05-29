# Combo variant of a395ee82 (index 8)
# combo_spec: D2/2+D3/1+D14/1
# Name: 不均匀孔洞翻转

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, components_mask, find_components, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace:
# - Preserved backbone from the base task:
#   one large monochrome template in color T, plus detached singleton markers with
#   one T-colored origin and the rest in color O.
# - D2/2 changes how singleton offsets are decoded:
#   the visible singleton rows/cols form warped axes whose adjacent gaps are either
#   2 or 3 pixels. Gap 2 means one stamp-step; gap 3 means two stamp-steps.
# - D3/1 changes what gets stamped:
#   the template mask contains 1-2 enclosed bg holes, and those hole pixels remain
#   bg in every copy.
# - D14/1 changes how each copy is oriented:
#   points left of the origin use a horizontal mirror, points above the origin use
#   a vertical mirror, and upper-left points use both.
# - Example:
#   suppose visible source rows are [4, 6, 9] and cols are [10, 13, 15].
#   Their decoded steps are rows [0, 1, 3] and cols [0, 2, 3].
#   If the T-colored origin is at raw position (6, 13), then:
#     raw point (4, 10) decodes to step delta (-1, -2), so its copy moves up by
#     1 template height and left by 2 template widths, and is flipped both ways.
#     raw point (9, 13) decodes to (+2, 0), so it moves downward twice with no flip.
#   The old solver would divide by one global gcd and miss those locally decoded
#   step sizes; it would also reuse one orientation for every copy.
#
# b) Complexity calibration:
# - Source gaps use only two visible categories (2 and 3), repeated on both axes,
#   so the irregular spacing is inferable instead of being a hidden numeric trick.
# - The lattice is fixed to the smallest 6-point witness pattern that always shows
#   left, up, upper-left, right, and down neighbors around the origin.
# - The template library is restricted to masks that both contain real holes and
#   visibly change under horizontal, vertical, and double reflection. This keeps the
#   flip rule observable on every sample.
#
# c) Implicit-assumption audit against the original main():
# - Original `gcd` normalization assumed one global lattice unit; D2/2 breaks that.
#   This file replaces it with per-axis cumulative decoding from sorted raw coords.
# - Original `templ_rel` could be stamped unchanged at every target; D14/1 breaks
#   that. This file precomputes four reflected versions of the exact filled mask.
# - Original multicolor-component parsing was acceptable for solid templates, but
#   D3/1 benefits from monochromatic component parsing so the holey template can be
#   reasoned about explicitly and its enclosed bg pixels can be asserted.
# - Original stamping-by-foreground-coordinates already preserved holes implicitly;
#   this combo keeps that behavior but now treats hole presence as a required input
#   feature rather than an accidental allowance.
#
# d) Placement legality:
# - Output copies are still placed on template-height/template-width lattice steps,
#   so reflections stay inside each copy bbox and never enlarge it.
# - Generator chooses template placement from decoded step bounds, so all copies fit.
# - Singleton placement uses raw warped-coordinate bounds, stays disjoint from the
#   template bbox with a bg gap, and avoids landing on any stamped output pixel.
#
# e) Visual observability:
# - Every sample contains both short and long source gaps on each axis.
# - Every sample contains at least one left, one up, and one upper-left witness
#   relative to the origin, so all reflection cases are shown directly.
# - The template always has 1-2 enclosed holes and is reflection-asymmetric, so the
#   flip rule changes visible pixels rather than only invisible metadata.
#
# f) Visual cleanliness:
# - There are still only two object families in the input: one template component and
#   isolated singletons.
# - Stamped copies are generated without filled-pixel overlap, so the output remains
#   easy to parse even after reflections.
#
# g) Rule inferability / non-degeneration:
# - Colors remain only bg / T / O; there is no extra color mapping to infer.
# - The generator forbids regression to the original task by requiring mixed gap
#   categories, real holes, and visible reflection asymmetry in every instance.
#
# h) New main() assumptions and generator guarantees:
# - main() assumes singleton rows/cols lie on consecutive warped axis levels whose
#   adjacent gaps belong to {2, 3}; generator enforces that exactly.
# - main() assumes the largest monochromatic component is the holey template and that
#   there is exactly one T-colored singleton origin; generator enforces both.
# - main() assumes the template has 1-2 enclosed holes. The template library is
#   built only from masks satisfying that constraint.


GAP_TO_STEP = {2: 1, 3: 2}
SOURCE_GAP_CHOICES = tuple(sorted(GAP_TO_STEP))

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
TEMPLATE_HOLE_PIXELS = (1, 2)

# The lattice is fixed to the minimal witness set that exposes left / up / upper-left
# flips while still showing right / down unflipped references.
LATTICE_CELLS = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
LATTICE_ORIGIN = (1, 1)

SINGLETON_RETRIES = 80
GENERATE_RETRIES = 600


def _enclosed_bg_mask(filled_mask: np.ndarray) -> np.ndarray:
    """Return bg cells inside the bbox that are not 4-connected to the bbox edge."""
    h, w = map(int, filled_mask.shape)
    bg_mask = ~filled_mask.astype(bool)
    holes = np.zeros((h, w), dtype=bool)
    for coords in components_mask(bg_mask, connectivity=4):
        if coords.size == 0:
            continue
        touches_edge = any(
            int(r) == 0 or int(r) == h - 1 or int(c) == 0 or int(c) == w - 1
            for r, c in coords
        )
        if not touches_edge:
            holes[coords[:, 0], coords[:, 1]] = True
    return holes


def _transform_cells(
    cells: list[tuple[int, int]], h: int, w: int, flip_v: bool, flip_h: bool
) -> list[tuple[int, int]]:
    """Reflect relative template cells inside their h x w bbox."""
    out: list[tuple[int, int]] = []
    for rr, cc in cells:
        nr = int(h - 1 - rr) if flip_v else int(rr)
        nc = int(w - 1 - cc) if flip_h else int(cc)
        out.append((nr, nc))
    return out


def _axis_step_map(values: list[int]) -> dict[int, int]:
    """Decode a warped source axis into cumulative stamp steps."""
    vals = sorted({int(v) for v in values})
    assert len(vals) >= 2

    out = {int(vals[0]): 0}
    acc = 0
    for prev, cur in zip(vals, vals[1:]):
        gap = int(cur - prev)
        assert gap in GAP_TO_STEP
        acc += int(GAP_TO_STEP[gap])
        out[int(cur)] = int(acc)
    return out


def _cells_to_mask(cells: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _cell_set(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in cells}


def _has_visible_flips(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    """Require H, V, and HV reflections to visibly change the filled mask."""
    base = _cell_set(cells)
    return all(
        _cell_set(_transform_cells(cells, h, w, flip_v, flip_h)) != base
        for flip_v, flip_h in [(False, True), (True, False), (True, True)]
    )


def _template_candidates(h: int, w: int) -> dict[int, list[list[tuple[int, int]]]]:
    """Enumerate legal holey, reflection-visible templates grouped by hole count."""
    total = int(h * w)
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
        if not _has_visible_flips(cells, h, w):
            continue

        hole_pixels = int(_enclosed_bg_mask(mask).sum())
        if hole_pixels in grouped:
            grouped[hole_pixels].append(cells)
    return grouped


TEMPLATE_LIBRARY = {
    h: _template_candidates(h, TEMPLATE_W)
    for h in TEMPLATE_H_CHOICES
}


def _sample_template() -> tuple[int, int, list[tuple[int, int]], int] | None:
    """Sample a template with 1-2 enclosed holes and visible asymmetry."""
    hole_choices = list(TEMPLATE_HOLE_PIXELS)
    random.shuffle(hole_choices)
    for desired_holes in hole_choices:
        valid_heights = [h for h in TEMPLATE_H_CHOICES if TEMPLATE_LIBRARY[h][desired_holes]]
        if not valid_heights:
            continue
        h = int(random.choice(valid_heights))
        cells = random.choice(TEMPLATE_LIBRARY[h][desired_holes])
        return h, TEMPLATE_W, cells, int(desired_holes)
    return None


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(int(dr) for dr, _dc in deltas)
    max_dr = max(int(dr) for dr, _dc in deltas)
    min_dc = min(int(dc) for _dr, dc in deltas)
    max_dc = max(int(dc) for _dr, dc in deltas)
    return int(min_dr), int(max_dr), int(min_dc), int(max_dc)


def _levels_from_gaps(gaps: list[int]) -> tuple[list[int], list[int]]:
    """Return raw source coordinates and decoded step coordinates for one axis."""
    src = [0]
    steps = [0]
    for gap in gaps:
        src.append(int(src[-1] + gap))
        steps.append(int(steps[-1] + GAP_TO_STEP[int(gap)]))
    return src, steps


def _sample_gap_layout() -> tuple[list[int], list[int]]:
    """Generate non-uniform row/col gap patterns, both showing short and long gaps."""
    row_gaps = list(SOURCE_GAP_CHOICES)
    col_gaps = list(SOURCE_GAP_CHOICES)
    random.shuffle(row_gaps)
    random.shuffle(col_gaps)
    return row_gaps, col_gaps


def _lattice_deltas(
    row_src: list[int], row_steps: list[int], col_src: list[int], col_steps: list[int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return raw singleton deltas and decoded stamp-step deltas from the fixed lattice."""
    raw_deltas: list[tuple[int, int]] = []
    step_deltas: list[tuple[int, int]] = []
    orr, orc = LATTICE_ORIGIN
    for lr, lc in LATTICE_CELLS:
        raw_deltas.append(
            (
                int(row_src[lr] - row_src[orr]),
                int(col_src[lc] - col_src[orc]),
            )
        )
        step_deltas.append(
            (
                int(row_steps[lr] - row_steps[orr]),
                int(col_steps[lc] - col_steps[orc]),
            )
        )
    return raw_deltas, step_deltas


def _stamp_points(
    templ_cells: list[tuple[int, int]],
    h: int,
    w: int,
    r0: int,
    c0: int,
    deltas: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Real filled pixels occupied by all reflected output copies."""
    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        stamp = _transform_cells(templ_cells, h, w, int(dr) < 0, int(dc) < 0)
        for tr, tc in stamp:
            pt = (int(r0 + dr * h + tr), int(c0 + dc * w + tc))
            if pt in occupied:
                raise AssertionError("reflected stamps unexpectedly overlap")
            occupied.add(pt)
    return occupied


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp reflected copies of a holey template using an irregular singleton scaffold.

    Input structure:
    - bg is the dominant background color.
    - One large monochrome component of color T is the template; its bbox is 3-4 by 3.
    - The template contains 1-2 enclosed bg holes that are part of the replicated mask.
    - Detached singleton pixels form a warped lattice: exactly one singleton is color T
      and marks the origin, while the others share color O.

    Rule:
    - Read the sorted singleton rows and cols as warped axes whose adjacent gaps decode
      locally: 2 pixels -> 1 stamp-step, 3 pixels -> 2 stamp-steps.
    - Convert each singleton's decoded offset from the origin into translations by the
      template bbox height and width.
    - Stamp only the template's filled pixels at each translated position, preserving
      holes as bg; mirror the copy horizontally when it is left of the origin and
      vertically when it is above the origin.
    - Recolor the origin copy's filled pixels to O, and recolor any filled overlap to O.

    Key insight: The singleton scaffold controls two things at once: local gap categories
    determine how far each copy moves, and the sign of the decoded offset determines which
    reflection of the holey template is used there.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_col = int(template["color"])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))

    templ_mask = g[r0 : r1 + 1, c0 : c1 + 1] == int(templ_col)
    hole_mask = _enclosed_bg_mask(templ_mask)
    hole_pixels = int(hole_mask.sum())
    assert 1 <= hole_pixels <= 2

    templ_rel = [tuple(map(int, rc)) for rc in np.argwhere(templ_mask)]
    transformed = {
        (flip_v, flip_h): _transform_cells(templ_rel, h, w, flip_v, flip_h)
        for flip_v in (False, True)
        for flip_h in (False, True)
    }

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(comp["color"])))

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
            int(col),
        )
        for (r, c, col) in singletons
    ]

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        stamp_cells = transformed[(int(dr) < 0, int(dc) < 0)]
        is_origin = int(col) == int(templ_col)
        for rr, cc in stamp_cells:
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


def _place_singletons(
    raw_deltas: list[tuple[int, int]],
    raw_bounds: tuple[int, int, int, int],
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    T: int,
    O: int,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    """Place the warped singleton scaffold away from the template."""
    min_dr, max_dr, min_dc, max_dc = raw_bounds
    orr_lo, orr_hi = -int(min_dr), GRID_SIZE - 1 - int(max_dr)
    orc_lo, orc_hi = -int(min_dc), GRID_SIZE - 1 - int(max_dc)
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(r0), int(r0 + h - 1)
    tc0, tc1 = int(c0), int(c0 + w - 1)

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = int(orr + min_dr)
        sr1 = int(orr + max_dr)
        sc0 = int(orc + min_dc)
        sc1 = int(orc + max_dc)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        src_points = [(int(orr + dr), int(orc + dc)) for dr, dc in raw_deltas]
        if any(pt in occupied for pt in src_points):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
        for tr, tc in templ_cells:
            grid[int(r0 + tr), int(c0 + tc)] = int(T)
        for rr, cc in src_points:
            grid[int(rr), int(cc)] = int(O)
        grid[int(orr), int(orc)] = int(T)

        try:
            main(grid)
        except Exception:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt for the irregular-hole-reflection combo."""
    bg, T, O = random.sample(range(10), 3)

    templ = _sample_template()
    if templ is None:
        return None
    h, w, templ_cells, _hole_pixels = templ

    row_gaps, col_gaps = _sample_gap_layout()
    row_src, row_steps = _levels_from_gaps(row_gaps)
    col_src, col_steps = _levels_from_gaps(col_gaps)
    raw_deltas, step_deltas = _lattice_deltas(row_src, row_steps, col_src, col_steps)

    min_sdr, max_sdr, min_sdc, max_sdc = _delta_bounds(step_deltas)
    r0_lo, r0_hi = -int(min_sdr) * h, GRID_SIZE - int(max_sdr) * h - h
    c0_lo, c0_hi = -int(min_sdc) * w, GRID_SIZE - int(max_sdc) * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)
    occupied = _stamp_points(templ_cells, h, w, r0, c0, step_deltas)

    return _place_singletons(
        raw_deltas=raw_deltas,
        raw_bounds=_delta_bounds(raw_deltas),
        templ_cells=templ_cells,
        r0=r0,
        c0=c0,
        h=h,
        w=w,
        bg=bg,
        T=T,
        O=O,
        occupied=occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
