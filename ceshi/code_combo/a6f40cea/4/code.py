# Combo variant of a6f40cea (index 4)
# combo_spec: D2/1+D10/0
# Name: 多色循环长度相位
#
# Design analysis for the fused variant
# ====================================
# 0. Scope and source-task grounding
#    - The original task keeps one closed rectangular frame and places 2-3 open rectangles outside
#      it; their ring-touch endpoints project to Manhattan paths inside the frame.
#    - D2/1 changes the non-monochrome rule from a 2-color toggle to a visible multi-color cycle.
#    - D10/0 removes endpoint-color anchoring and makes the interior phase depend on path length.
#    - The fused rule therefore should not merely "read the endpoint colors of a 3-color trace".
#      It must recover an ordered 3-color cycle and then shift the interior start phase from a
#      visible structural variable: the interior path length modulo 3.
#
# 1. Concrete behavior and trace
#    - I restrict ring-touching components to L-shapes. Each sample uses one shared 3-color cycle
#      such as (2, 8, 6), reused by every ring-touching component in that sample.
#    - The external open rectangle is painted in canonical geometric order:
#        top/bottom ring touch -> far outside -> left/right ring touch.
#      Along that order the colors repeat A, B, C, A, B, C, ...
#    - The interior path keeps the same canonical direction and the same cycle order, but its first
#      color is shifted by `(len(inner_path) - 1) % 3`:
#        residue 0 -> start at A
#        residue 1 -> start at B
#        residue 2 -> start at C
#    - Small trace example:
#        outer cycle = (2, 8, 6)
#        inner path length = 5 cells -> residue 1 -> output colors = 8, 6, 2, 8, 6
#        original/D2-only logic would instead try to align one endpoint with the ring touch color.
#      The visible difference is that the interior no longer "starts where the ring starts".
#
# 2. Complexity calibration
#    - Allowing each component to pick an unrelated 3-color permutation would make the puzzle noisy
#      and hard to infer from 2-4 examples. I therefore share one 3-color cycle across all L
#      components inside a sample.
#    - I removed U-pairs for this combo. They are geometrically valid in the base task, but they do
#      not help express the new phase rule and would force extra exception-handling in the solver.
#    - The generator requires at least two different inner-path residues in each sample, so the
#      length-phase rule has repeated evidence inside a single input/output pair.
#
# 3. Original-main assumption audit
#    - Original/D2 endpoint-hint logic assumed the two ring-touch colors determine the interior
#      phase. That assumption is invalid here: the ring-touch colors are now just consequences of
#      the outer path length, not the phase constraint.
#    - Original `sorted(set(...))` color handling is insufficient because it erases cycle order.
#      The solver must traverse each external component as an ordered simple path and extract the
#      primitive repeating cycle from that order.
#    - Original code tolerated multiple component families (solid L, U-stripes, 2-color L). This
#      fused solver now assumes every ring-touching non-distractor is a simple L-path with exactly
#      one top/bottom touch and one left/right touch. The generator guarantees that.
#
# 4. Placement legality
#    - Components are generated with the original frame-size and margin bounds.
#    - Outer paths are still simple 4-connected paths, in bounds, non-overlapping, and not 4-adjacent
#      to each other except within the same component.
#    - Inner paths are also kept disjoint so projected outputs remain legible.
#    - Distractors are small monochrome blobs that never touch the frame ring, so the solver can
#      safely ignore them when reconstructing interior paths.
#
# 5. Visual observability
#    - The cycle length is fixed to 3. That is enough to make the color rule visibly different from
#      binary alternation without exploding the number of cases.
#    - Because all ring-touching components in one sample share the same ordered cycle, the solver
#      sees repeated evidence for the order A->B->C.
#    - Because samples must contain at least two different length residues, the user can compare two
#      L-paths with the same outer cycle but different interior starting colors and infer the
#      residue-based phase shift.
#
# 6. Cleanliness constraints
#    - The generator rejects samples where all cyclic components accidentally collapse to the old
#      endpoint-anchored behavior. At least one component must differ from the old "match both ring
#      endpoint colors" rule.
#    - The generator also rejects samples where all inner lengths share the same residue, since that
#      would hide the new control variable.
#
# 7. New solver assumptions and matching generator guarantees
#    - Solver assumption: each ring-touching component is a simple path whose only ring cells are
#      its two endpoints. Generator guarantee: every L-component is drawn exactly that way.
#    - Solver assumption: the component has one projected touch on a top/bottom side and one on a
#      left/right side. Generator guarantee: only L-components are created.
#    - Solver assumption: the observed outer colors form a primitive 3-color cycle in canonical
#      geometric order. Generator guarantee: all ring-touching components use the shared 3-color
#      cycle with distinct colors and canonical phase 0 on the top/bottom touch.
#    - Solver assumption: the interior phase is `start_idx = (len(path) - 1) % 3`. Generator
#      guarantee: expected outputs are painted exactly with that rule.

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with uniform interior fill.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None
    for bc in sorted(set(map(int, g.flatten())) - {int(bg)}):
        lab, n = label(g == int(bc), structure=structure)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(len(pts)) < 8:
                continue

            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue

            per = int(2 * (h + w) - 4)
            if int(len(pts)) != per:
                continue
            if not (g[r0, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r1, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c0] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c1] == int(bc)).all():
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            fill = int(Counter(inner.flatten()).most_common(1)[0][0])
            if fill == int(bc):
                continue
            if not (inner == int(fill)).all():
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _ordered_component_path(mask: np.ndarray) -> list[tuple[int, int]] | None:
    """Return endpoint-to-endpoint order for a simple 4-connected path component."""
    pts = np.argwhere(mask)
    if int(len(pts)) < 2:
        return None

    cells = {tuple(map(int, p)) for p in pts}
    deg: dict[tuple[int, int], int] = {}
    for r, c in cells:
        nbs = [(nr, nc) for nr, nc in neighbors4(r, c) if (nr, nc) in cells]
        if len(nbs) > 2:
            return None
        deg[(r, c)] = len(nbs)

    ends = [p for p, d in deg.items() if d == 1]
    if len(ends) != 2:
        return None

    ordered = [ends[0]]
    prev = None
    cur = ends[0]
    while cur != ends[1]:
        nbs = [(nr, nc) for nr, nc in neighbors4(*cur) if (nr, nc) in cells and (nr, nc) != prev]
        if len(nbs) != 1:
            return None
        nxt = nbs[0]
        ordered.append(nxt)
        prev, cur = cur, nxt

    if len(ordered) != len(cells):
        return None
    return ordered


def _primitive_cycle(seq: list[int]) -> tuple[int, ...] | None:
    """Extract the shortest repeating color cycle from an ordered color sequence."""
    n = len(seq)
    for k in range(1, n + 1):
        pat = tuple(int(x) for x in seq[:k])
        if k > 1 and len(set(pat)) != k:
            continue
        if all(int(seq[i]) == int(pat[i % k]) for i in range(n)):
            return pat
    return None


def _rotate_cycle(cycle: tuple[int, ...], shift: int) -> tuple[int, ...]:
    """Rotate a cycle left by shift."""
    k = len(cycle)
    shift %= k
    return tuple(int(x) for x in cycle[shift:] + cycle[:shift])


def _rotate_cycle_to_start(cycle: tuple[int, ...], start_color: int) -> tuple[int, ...] | None:
    """Rotate a cycle so that the first color equals start_color."""
    idxs = [i for i, col in enumerate(cycle) if int(col) == int(start_color)]
    if len(idxs) != 1:
        return None
    return _rotate_cycle(cycle, idxs[0])


def _paint_cycle(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    cycle: tuple[int, ...],
    *,
    start_idx: int = 0,
) -> None:
    """Paint path cells by repeating cycle from a chosen phase."""
    if not path:
        return
    rs, cs = zip(*path)
    vals = [int(cycle[(start_idx + i) % len(cycle)]) for i in range(len(path))]
    grid[rs, cs] = vals


def _canonical_l_touchpoints(
    pts: list[tuple[int, int]],
    outH: int,
    outW: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return (top/bottom touch, left/right touch) for an L-component."""
    tb = [p for p in pts if p[0] in (0, outH - 1) and 0 < p[1] < outW - 1]
    lr = [p for p in pts if p[1] in (0, outW - 1) and 0 < p[0] < outH - 1]
    if len(tb) == 1 and len(lr) == 1:
        return (int(tb[0][0]), int(tb[0][1])), (int(lr[0][0]), int(lr[0][1]))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the cropped frame interior by reading 3-color cycles on the external open rectangles.

    Input structure:
    - A background grid contains one closed 1px rectangular frame with uniform interior fill.
    - Outside the frame are simple L-shaped open rectangles whose two path endpoints touch the ring
      immediately beyond the frame, plus optional distractor blobs that do not touch the ring.

    Rule:
    - Output only the interior of the closed frame, initialized to the frame fill color.
    - For each ring-touching L-component, project its two touched frame sides into the interior and
      draw the corresponding Manhattan L-path from the top/bottom side to the left/right side.
    - Read the external component in that same canonical geometric direction to recover its
      repeating 3-color cycle.
    - Paint the interior path with the same cycle, but rotate the start phase by
      `(interior_path_length - 1) % 3` instead of matching either ring-touch endpoint color.

    Key insight:
    - The outer trace reveals the ordered color cycle, while the interior path length reveals which
      phase of that cycle should appear first. Endpoint colors are no longer the anchoring clue.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    def build_manhattan(
        p1: tuple[int, int],
        p2: tuple[int, int],
        cornerA: bool,
    ) -> list[tuple[int, int]]:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb:
            step = 1 if cb >= ca else -1
            return [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
        if ca == cb:
            step = 1 if rb >= ra else -1
            return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
        if cornerA:
            step = 1 if rb >= ra else -1
            path = [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
            step = 1 if cb >= ca else -1
            path += [(int(rb), int(c)) for c in range(int(ca) + step, int(cb) + step, step)]
            return path
        step = 1 if cb >= ca else -1
        path = [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
        step = 1 if rb >= ra else -1
        path += [(int(r), int(cb)) for r in range(int(ra) + step, int(rb) + step, step)]
        return path

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> bool:
        if p1[0] == p2[0] or p1[1] == p2[1]:
            return True
        rr, cc = (p2[0], p1[1]) if cornerA else (p1[0], p2[1])
        return 0 < int(rr) < outH - 1 and 0 < int(cc) < outW - 1

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring: set[tuple[int, int]] = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                ring.add((int(rr), int(cc)))

    for cid in range(1, int(cn) + 1):
        ordered = _ordered_component_path(clab == int(cid))
        if ordered is None:
            continue

        touch_idx = [i for i, p in enumerate(ordered) if p in ring]
        if sorted(touch_idx) != [0, len(ordered) - 1]:
            continue

        src_a, src_b = ordered[0], ordered[-1]
        a = map_to_out(*src_a)
        b = map_to_out(*src_b)
        touch = _canonical_l_touchpoints([a, b], outH, outW)
        if touch is None:
            continue
        tb_pt, lr_pt = touch

        if a == tb_pt and b == lr_pt:
            outer_order = ordered
        elif b == tb_pt and a == lr_pt:
            outer_order = list(reversed(ordered))
        else:
            continue

        colors_seq = [int(g[r, c]) for r, c in outer_order]
        cycle = _primitive_cycle(colors_seq)
        if cycle is None or len(cycle) != 3:
            continue

        pathA = build_manhattan(tb_pt, lr_pt, True)
        pathB = build_manhattan(tb_pt, lr_pt, False)
        path = pathA if inside_corner(tb_pt, lr_pt, True) else pathB
        start_idx = (len(path) - 1) % len(cycle)
        _paint_cycle(out, path, cycle, start_idx=start_idx)

    return out


# --------------- generator (a6 / combo D2/1 + D10/0) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_CYCLE_LEN = 3
_L_SPEC_TRIES = 120
_MIN_INNER_PATH_LEN = 4
_MIN_OUTER_PATH_LEN = 7
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 800


def _line(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    """Check that coords are in-bounds, on bg, and not 4-adjacent to existing components."""
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and bool(comp_mask[nr, nc]):
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    cycle: tuple[int, ...],
    bg: int,
    *,
    start_idx: int = 0,
) -> bool:
    """Validate, paint, and register one component."""
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_cycle(grid, path, cycle, start_idx=start_idx)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build canonical inner and outer paths for one L-component."""
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r = random.randint(1, max(1, H // 2)) if is_top else random.randint(max(1, H // 2), H - 2)
    c = random.randint(1, max(1, W // 2)) if is_left else random.randint(max(1, W // 2), W - 2)

    lo, hi = _OUTER_OFFSET_RANGE
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + c)
    ring_b = (r0 + 1 + r, c0 - 1 if is_left else c1 + 1)

    edge_r = 0 if is_top else H - 1
    edge_c = 0 if is_left else W - 1
    inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer


def _old_endpoint_match(
    inner: list[tuple[int, int]],
    outer: list[tuple[int, int]],
    cycle: tuple[int, ...],
) -> bool:
    """Check whether old endpoint-anchored cyclic logic would paint the same inner result."""
    k = len(cycle)
    ha = int(cycle[0])
    hb = int(cycle[(len(outer) - 1) % k])

    new_start = (len(inner) - 1) % k
    new_vals = [int(cycle[(new_start + i) % k]) for i in range(len(inner))]
    new_map = {p: int(v) for p, v in zip(inner, new_vals)}

    for path_pts, start_hint, end_hint in ((inner, ha, hb), (list(reversed(inner)), hb, ha)):
        for cyc in (cycle, tuple(reversed(cycle))):
            aligned = _rotate_cycle_to_start(cyc, int(start_hint))
            if aligned is None:
                continue
            if int(aligned[(len(path_pts) - 1) % k]) != int(end_hint):
                continue
            old_vals = [int(aligned[i % k]) for i in range(len(path_pts))]
            old_map = {p: int(v) for p, v in zip(path_pts, old_vals)}
            if old_map == new_map:
                return True
    return False


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    """Try to place a small monochrome blob away from the frame."""
    H, W = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r, c = random.randint(1, H - 2), random.randint(1, W - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (nr, nc) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add(p)
            coords.append(p)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid for the fused rule."""
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 1, c0 + W + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((H, W), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    used_inner = np.zeros((H, W), dtype=bool)

    avail = [x for x in range(10) if x not in (bg, bc, fill)]
    if len(avail) < _CYCLE_LEN:
        return None
    cycle = tuple(int(x) for x in random.sample(avail, _CYCLE_LEN))
    distractor_pool = [int(x) for x in avail if x not in cycle]
    random.shuffle(distractor_pool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    corners = random.sample(["tl", "tr", "bl", "br"], k=n_components)

    specs: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    residues: set[int] = set()
    differs_from_old = 0

    for kind in corners:
        placed = False
        for _ in range(_L_SPEC_TRIES):
            inner, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
            if len(inner) < _MIN_INNER_PATH_LEN or len(outer) < _MIN_OUTER_PATH_LEN:
                continue
            irs, ics = zip(*inner)
            if used_inner[irs, ics].any():
                continue
            if not _place_component(grid, comp_mask, outer, cycle, bg, start_idx=0):
                continue
            used_inner[irs, ics] = True
            specs.append((inner, outer))
            residues.add((len(inner) - 1) % _CYCLE_LEN)
            if not _old_endpoint_match(inner, outer, cycle):
                differs_from_old += 1
            placed = True
            break
        if not placed:
            return None

    if len(residues) < 2:
        return None
    if differs_from_old == 0:
        return None

    for inner, _outer in specs:
        _paint_cycle(expected, inner, cycle, start_idx=(len(inner) - 1) % _CYCLE_LEN)

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if not distractor_pool:
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), int(distractor_pool.pop()))

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
