# Combo variant of a6f40cea (index 26)
# combo_spec: D2/1+D17/0
# Name: 图例定义多色循环

# Design analysis for the combo variant
# ====================================
# 1. Concrete fused behavior
#    - Original a6f40cea infers interior Manhattan paths from open rectangles outside a closed frame.
#      Non-monochrome traces use their own visible colors directly.
#    - This combo inserts a corner legend and splits the color logic into two layers:
#        * outer ring-touching traces use a 3-color source cycle S0,S1,S2,...
#        * the corner legend is a 3x2 block whose rows define ordered pairs
#          (S0->T0), (S1->T1), (S2->T2)
#        * the interior path uses T0,T1,T2,... in the same cyclic phase.
#    - Small trace example:
#        legend rows: (8->1), (2->6), (7->9)
#        outer path order: 2,7,8,2,7
#        original task would copy visible colors inward.
#        fused task instead maps through the legend and paints 6,9,1,6,9 inward.
#      The geometry is still read from the open rectangle, but the output colors are not present on
#      the component itself except in the legend.
#
# 2. Complexity calibration
#    - The legend always has exactly 3 rows. That is enough to make the new rule visually obvious
#      while avoiding a noisy 4+ color palette.
#    - Every ring-touching component in a sample reuses the same source/target cycle pair, but each
#      component may start at a different phase. This gives repeated evidence for the legend rule.
#    - U-pairs and two-color alternators are removed from generation for this combo. Keeping them
#      would dilute the legend-defined cycle and create mixed local rules in too few examples.
#
# 3. Implicit-assumption audit against the original main()
#    - Original code assumes the set of component colors is enough to infer the path coloring rule.
#      That breaks here because the true output colors come from the legend, not from the component.
#    - Original `comp_cols[cid] = sorted(set(...))` loses path order. That already fails for D2/1
#      and is even less acceptable once the ordered outer cycle must be translated through a legend.
#    - Original code never had to ignore a structured non-frame object in the corner. The combo
#      solver must detect the legend first and exclude its cells from component labeling.
#    - Original binary phase logic (`len(cols) == 2`) assumes a toggle. The combo needs ordered path
#      traversal, primitive-cycle recovery, and phase alignment after legend translation.
#
# 4. Placement legality
#    - The generator reserves a 3x2 corner block plus a one-cell moat in all four corners. One
#      reserved corner receives the legend; the other three stay empty. This prevents accidental
#      false legends and keeps distractors away from the legend signature.
#    - Ring-touching components remain simple non-self-intersecting paths, do not overlap each other,
#      do not touch the legend reserve, and do not reuse interior cells.
#    - For each L-component, generation only accepts specs where inner and outer path lengths are
#      congruent modulo 3. That guarantees the same cycle phase can anchor both endpoints.
#
# 5. Visual observability
#    - Humans can see three distinct mapping rows in the corner and repeated three-color rhythms on
#      multiple external traces. The output then consistently uses a different repeated triple.
#    - Source-cycle colors and target-cycle colors are chosen disjointly. Therefore the legend is
#      necessary: a solver that ignores it and copies outer colors inward produces the wrong palette.
#
# 6. Visual cleanliness
#    - Only L-components are used as signal carriers, so the scene does not mix stripes and cycles.
#    - The legend is compact, anchored to a corner, and separated by a moat from all other objects.
#      This keeps the frame region visually distinct from the instruction region.
#
# 7. Generator-side guarantees for solver assumptions
#    - Each ring-touching component is a simple 4-connected path whose only ring cells are its two
#      endpoints. The solver relies on endpoint-to-endpoint traversal.
#    - Every non-distractor ring-touching component uses only legend source colors, with a primitive
#      cycle of length 3 and a path long enough to show at least one wrap.
#    - Distractors never touch the ring or any legend-reserve zone, so the solver can ignore them.
#
# 8. Post-edit audit of new assumptions
#    - `_find_corner_legend()` assumes exactly one reserved 3x2 corner block is non-background and
#      that the surrounding moat is background. The generator enforces both.
#    - The cyclic branch in `main()` assumes translated endpoint colors determine a valid direction
#      and phase for the interior path. The modulo-compatibility check in generation guarantees this.
#    - The solver assumes target-cycle colors do not need to be inferred from geometry alone; the
#      legend supplies them explicitly. The generator always paints the legend and keeps its rows
#      intact and unique.

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


_LEGEND_LEN = 3


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """Return the largest 1px rectangular frame with uniform interior fill."""
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
    """Return endpoint-to-endpoint order for a simple 4-connected path."""
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
    """Extract the shortest repeating color cycle from a sequence."""
    n = len(seq)
    for k in range(1, n + 1):
        pat = tuple(int(x) for x in seq[:k])
        if k > 1 and len(set(pat)) != k:
            continue
        if all(int(seq[i]) == int(pat[i % k]) for i in range(n)):
            return pat
    return None


def _rotate_cycle_to_start(cycle: tuple[int, ...], start_color: int) -> tuple[int, ...] | None:
    """Rotate cycle so its first color is start_color."""
    idxs = [i for i, col in enumerate(cycle) if int(col) == int(start_color)]
    if len(idxs) != 1:
        return None
    i = idxs[0]
    return tuple(int(x) for x in cycle[i:] + cycle[:i])


def _paint_cycle(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    cycle: tuple[int, ...],
    *,
    start_idx: int = 0,
) -> None:
    """Paint a path by repeating cycle from start_idx."""
    if not path:
        return
    rs, cs = zip(*path)
    vals = [int(cycle[(start_idx + i) % len(cycle)]) for i in range(len(path))]
    grid[rs, cs] = vals


def _legend_geom(shape: tuple[int, int], corner: str) -> tuple[list[int], int, int, np.ndarray]:
    """Return legend row order, source col, target col, and the legend mask for a corner."""
    H, W = map(int, shape)
    mask = np.zeros((H, W), dtype=bool)
    if corner == "tl":
        rows = [0, 1, 2]
        c_src, c_dst = 0, 1
    elif corner == "tr":
        rows = [0, 1, 2]
        c_src, c_dst = W - 2, W - 1
    elif corner == "bl":
        rows = [H - 1, H - 2, H - 3]
        c_src, c_dst = 0, 1
    else:
        rows = [H - 1, H - 2, H - 3]
        c_src, c_dst = W - 2, W - 1
    for r in rows:
        mask[int(r), int(c_src)] = True
        mask[int(r), int(c_dst)] = True
    return rows, int(c_src), int(c_dst), mask


def _corner_reserve_mask(shape: tuple[int, int]) -> np.ndarray:
    """Reserve legend-sized corner zones and a 1-cell inward moat in all four corners."""
    H, W = map(int, shape)
    mask = np.zeros((H, W), dtype=bool)
    for corner in ("tl", "tr", "bl", "br"):
        _rows, _c_src, _c_dst, legend = _legend_geom((H, W), corner)
        pts = np.argwhere(legend)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rr0 = max(0, int(r0) - 1)
        cc0 = max(0, int(c0) - 1)
        rr1 = min(H - 1, int(r1) + 1)
        cc1 = min(W - 1, int(c1) + 1)
        mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _find_corner_legend(g: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...], str]:
    """Find the unique 3x2 legend block in one corner."""
    H, W = g.shape
    reserve = _corner_reserve_mask((H, W))
    found = []
    for corner in ("tl", "tr", "bl", "br"):
        rows, c_src, c_dst, legend_mask = _legend_geom((H, W), corner)
        if not legend_mask.any():
            continue
        if not np.all(g[legend_mask] != int(bg)):
            continue

        pts = np.argwhere(legend_mask)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rr0 = max(0, int(r0) - 1)
        cc0 = max(0, int(c0) - 1)
        rr1 = min(H - 1, int(r1) + 1)
        cc1 = min(W - 1, int(c1) + 1)

        moat = np.zeros((H, W), dtype=bool)
        moat[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        moat &= ~legend_mask
        if np.any(g[moat] != int(bg)):
            continue
        # No other reserved corner zone may contain colored cells.
        other_reserved = reserve & (~moat) & (~legend_mask)
        if np.any((g != int(bg)) & other_reserved):
            continue

        src_cycle = tuple(int(g[int(r), int(c_src)]) for r in rows)
        dst_cycle = tuple(int(g[int(r), int(c_dst)]) for r in rows)
        if len(set(src_cycle)) != _LEGEND_LEN or len(set(dst_cycle)) != _LEGEND_LEN:
            continue
        found.append((legend_mask, src_cycle, dst_cycle, corner))

    assert len(found) == 1, "expected exactly one corner legend"
    return found[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct frame-interior paths by reading a corner legend and translating external color cycles.

    Input structure:
    - A uniform bg field contains one closed 1px rectangular frame with a single fill color inside.
    - A 3x2 legend sits in one grid corner; each row is an ordered source->target color pair.
    - Outside the frame are 4-connected open-rectangle traces whose two free ends touch the frame ring.

    Rule:
    - Output only the frame interior, initialized with the frame's fill color.
    - For each ring-touching component, project its two ring endpoints onto interior-edge coordinates.
    - Recover the ordered outer color cycle along the component, translate each cycle color through the
      legend, and paint the corresponding interior Manhattan L-path with the translated 3-color cycle.

    Key insight:
    - The visible rhythm on the external trace gives path direction and phase, but the legend supplies
      the actual output palette; the correct interior colors are obtained only after translating the
      recovered cycle through the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    legend_mask, src_cycle, dst_cycle, _corner = _find_corner_legend(g, bg)
    legend_map = {int(s): int(t) for s, t in zip(src_cycle, dst_cycle)}

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

    def build_manhattan(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> list[tuple[int, int]]:
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
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if cornerA else (ra, cb)
        rr, cc = corner
        return (0 < rr < outH - 1) and (0 < cc < outW - 1)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc)) & (~legend_mask)
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
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

        idx_touch = [i for i, p in enumerate(ordered) if p in ring]
        if len(idx_touch) != 2 or sorted(idx_touch) != [0, len(ordered) - 1]:
            continue

        source_seq = [int(g[r, c]) for r, c in ordered]
        if not all(int(col) in legend_map for col in source_seq):
            continue
        translated_seq = [int(legend_map[int(col)]) for col in source_seq]
        translated_cycle = _primitive_cycle(translated_seq)
        assert translated_cycle is not None and len(translated_cycle) == _LEGEND_LEN

        a_src = ordered[0]
        b_src = ordered[-1]
        a = map_to_out(*a_src)
        b = map_to_out(*b_src)

        pathA = build_manhattan(a, b, True)
        pathB = build_manhattan(a, b, False)
        path = pathB if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else pathA

        start_hint = int(translated_seq[0])
        end_hint = int(translated_seq[-1])
        chosen: tuple[list[tuple[int, int]], tuple[int, ...]] | None = None
        for path_pts, sh, eh in ((path, start_hint, end_hint), (list(reversed(path)), end_hint, start_hint)):
            for cyc in (translated_cycle, tuple(reversed(translated_cycle))):
                aligned = _rotate_cycle_to_start(cyc, int(sh))
                if aligned is None:
                    continue
                if int(aligned[(len(path_pts) - 1) % len(aligned)]) != int(eh):
                    continue
                chosen = (path_pts, aligned)
                break
            if chosen is not None:
                break

        assert chosen is not None
        path_pts, aligned = chosen
        _paint_cycle(out, path_pts, aligned)

    return out


# --------------- generator ---------------

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
_L_SPEC_TRIES = 100
_MIN_INNER_LEN = 5
_DISTRACTOR_COUNT_RANGE = (0, 1)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 700


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


def _can_place(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    bg: int,
    comp_mask: np.ndarray,
    blocked_mask: np.ndarray,
) -> bool:
    """All coords must be in-bounds, on bg, off blocked zones, and isolated from other components."""
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if bool(blocked_mask[r, c]):
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
    blocked_mask: np.ndarray,
    *,
    start_idx: int = 0,
) -> bool:
    """Validate, paint, and register a simple path component."""
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask, blocked_mask):
        return False
    _paint_cycle(grid, path, cycle, start_idx=start_idx)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build one interior L-path and its matching external open rectangle."""
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


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
    blocked_mask: np.ndarray,
) -> None:
    """Try to place a small monochrome blob away from the frame and the legend reserve."""
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
        if _place_component(grid, comp_mask, coords, (int(color),), bg, blocked_mask):
            return


def _draw_legend(grid: np.ndarray, corner: str, src_cycle: tuple[int, ...], dst_cycle: tuple[int, ...]) -> np.ndarray:
    """Paint the 3x2 corner legend and return its mask."""
    rows, c_src, c_dst, legend_mask = _legend_geom(grid.shape, corner)
    for i, r in enumerate(rows):
        grid[int(r), int(c_src)] = int(src_cycle[i])
        grid[int(r), int(c_dst)] = int(dst_cycle[i])
    return legend_mask


def _pick_cycles(avail: list[int]) -> tuple[tuple[int, ...], tuple[int, ...], list[int]]:
    """Pick disjoint source and target cycles plus leftover colors."""
    src_cycle = tuple(int(x) for x in random.sample(avail, _LEGEND_LEN))
    rest = [int(x) for x in avail if int(x) not in src_cycle]
    dst_cycle = tuple(int(x) for x in random.sample(rest, _LEGEND_LEN))
    leftovers = [int(x) for x in rest if int(x) not in dst_cycle]
    return src_cycle, dst_cycle, leftovers


def _find_cyclic_l(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    used_inner: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int] | None:
    """Retry until we find an L component that shows and preserves the 3-cycle phase."""
    for _ in range(_L_SPEC_TRIES):
        inner, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
        if len(inner) < _MIN_INNER_LEN:
            continue
        if (len(inner) - 1) % _LEGEND_LEN != (len(outer) - 1) % _LEGEND_LEN:
            continue
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            continue
        return inner, outer, random.randrange(_LEGEND_LEN)
    return None


def _gen_attempt() -> np.ndarray | None:
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

    comp_mask = np.zeros_like(grid, dtype=bool)
    used_inner = np.zeros((H, W), dtype=bool)
    expected = np.full((H, W), fill, dtype=int)

    avail = [x for x in range(10) if x not in (bg, bc, fill)]
    if len(avail) < 6:
        return None
    src_cycle, dst_cycle, leftovers = _pick_cycles(avail)

    blocked_mask = _corner_reserve_mask(grid.shape)
    legend_corner = random.choice(["tl", "tr", "bl", "br"])
    _draw_legend(grid, legend_corner, src_cycle, dst_cycle)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    corners = random.sample(["tl", "tr", "bl", "br"], k=n_components)
    for kind in corners:
        spec = _find_cyclic_l(kind, H, W, r0, c0, r1, c1, used_inner)
        if spec is None:
            return None
        inner, outer, start_idx = spec
        if not _place_component(grid, comp_mask, outer, src_cycle, bg, blocked_mask, start_idx=start_idx):
            return None
        _paint_cycle(expected, inner, dst_cycle, start_idx=start_idx)
        irs, ics = zip(*inner)
        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if not leftovers:
            break
        color = int(leftovers.pop())
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), color, blocked_mask)

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
