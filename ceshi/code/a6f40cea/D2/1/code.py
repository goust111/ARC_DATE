# Variant of a6f40cea
# Dimension: D2 — 轨迹着色模式 (trajectory)
# Change: 单色组件用纯色填充路径；双色组件用两色交替填充，相位由端点颜色锚定... → 颜色循环（多于2色）

# Design analysis for the D2 variant
# ==================================
# 0. Reference-task read status
#    - The prompt cites source task `195c6913` for the color-cycle idea.
#    - I searched the provided workspace for a standalone solver file for `195c6913` and found
#      only the ARC JSON examples, not a solver implementation. The design below therefore uses
#      the source examples as the baseline for "3+ color repeating cycle" behavior.
#
# 1. Concrete behavior and trace
#    - Original a6f40cea uses one closed rectangular frame and 2-3 external open-rect components.
#      Each ring-touching L-component projects to an interior Manhattan L-path. Monochrome paths
#      stay solid; bichrome paths alternate ABAB... with endpoint colors fixing the phase.
#    - New D2 behavior keeps the same geometry, but replaces the non-monochrome rule with a
#      repeating cycle of length 3 or 4: e.g. outer trace colors may be 2,8,6,2,8,6,... and the
#      corresponding interior path repeats the same cycle in the same stepwise manner.
#    - Small trace example:
#      original: endpoint colors = (2,8), inner length = 5 -> path becomes 2,8,2,8,2.
#      variant: cycle = (2,8,6), endpoint colors = (2,8), inner length = 5 -> path becomes
#      2,8,6,2,8. The solver must recover the ordered cycle, not just a 2-color toggle.
#
# 2. Complexity calibration
#    - If every component had its own unrelated 3-4 colors, the puzzle would look noisy and the
#      color rule would be hard to infer from 2-4 examples.
#    - To keep the change visible but learnable, each generated grid uses a single shared cycle
#      palette (length 3 or 4) for all cyclic L-components in that sample. Components may start at
#      different phases, but the cyclic order is reused. This provides repeated evidence.
#    - U-pairs remain monochrome. That preserves the original stripe behavior and avoids dragging
#      unrelated dimensions into the variant.
#
# 3. Implicit-assumption audit against the original main()
#    - Original `comp_cols[cid] = sorted(set(...))` silently discards path order. That was harmless
#      for 2 colors because {a,b} plus endpoint hints is enough to reconstruct alternation.
#      It fails for 3+ colors because sorting by color id is not a visible rule and loses the cycle.
#    - Original `if len(cols) == 2:` hard-codes a binary toggle. In the variant the successor color
#      depends on the ordered cycle, so the solver must traverse the component and extract a
#      primitive repeating pattern.
#    - Original endpoint logic assumes choosing start color from {ca,cb} is sufficient. For 3+ colors
#      the start phase is a rotation of the recovered cycle, and direction can also reverse.
#    - Original generator paints outer and inner traces independently with the same two-color tuple.
#      For cycles of length k>=3, endpoint compatibility is no longer automatic. The generator must
#      only accept L-specs where inner and outer path lengths agree modulo k, so the same cycle phase
#      can anchor both endpoints visibly.
#
# 4. Placement legality
#    - External components still must be non-overlapping, in-bounds, and not 4-adjacent to each
#      other except within the same component. The existing `_can_place` / `_place_component`
#      machinery still works after generalizing path painting.
#    - Cyclic L-components are generated with retry loops until:
#        * the interior path does not overlap already-used interior cells,
#        * the outer path can be placed legally,
#        * the inner and outer lengths are congruent modulo the cycle length,
#        * the inner path is long enough to display the cycle clearly.
#
# 5. Visual observability / cleanliness
#    - The geometric rule remains identical to the original task, so the only new visual cue is the
#      repeated multi-color cycle on the outer open frame and mirrored cycle on the interior path.
#    - Using one shared cycle per sample keeps the scene readable: multiple components reinforce the
#      same order instead of introducing many unrelated color permutations.
#    - The number of cycle colors is capped at 3 or 4. More than that would make the pattern too
#      hard to infer from a few examples.
#
# 6. New solver assumptions and generator guarantees
#    - Solver assumption: every non-distractor ring-touching component is a simple 4-connected path
#      with exactly two endpoints on the frame ring. Generator guarantee: L and U components are
#      constructed as simple paths; distractors do not touch the ring.
#    - Solver assumption: cyclic components use a primitive cycle with distinct colors. Generator
#      guarantee: the shared cycle palette is sampled without repeats.
#    - Solver assumption: for cyclic L-components, at least one direction/orientation of the cycle
#      is compatible with the two endpoint colors and the interior path length. Generator guarantee:
#      only modulo-compatible specs are accepted.

from __future__ import annotations

from collections import Counter, defaultdict
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
    """Return the unique endpoint-to-endpoint order for a simple 4-connected path component."""
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
    """Extract the shortest repeating color cycle used by a path."""
    n = len(seq)
    for k in range(1, n + 1):
        pat = tuple(int(x) for x in seq[:k])
        if k > 1 and len(set(pat)) != k:
            continue
        if all(int(seq[i]) == int(pat[i % k]) for i in range(n)):
            return pat
    return None


def _rotate_cycle_to_start(cycle: tuple[int, ...], start_color: int) -> tuple[int, ...] | None:
    """Rotate a cycle so that its first color is start_color."""
    idxs = [i for i, col in enumerate(cycle) if int(col) == int(start_color)]
    if len(idxs) != 1:
        return None
    i = idxs[0]
    return tuple(int(x) for x in cycle[i:] + cycle[:i])


def _paint_cycle(grid: np.ndarray, path: list[tuple[int, int]], cycle: tuple[int, ...]) -> None:
    """Paint a path by repeating cycle from its first point onward."""
    if not path:
        return
    rs, cs = zip(*path)
    if len(cycle) == 1:
        grid[rs, cs] = int(cycle[0])
        return
    vals = [int(cycle[i % len(cycle)]) for i in range(len(path))]
    grid[rs, cs] = vals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct interior paths by reading the color rhythm of external traces around the frame.

    Input structure:
    - A bg field containing one 1px closed rectangle with uniform fill.
    - Outside the frame sit 4-connected open-rect traces whose free ends touch the frame ring.

    Rule:
    - Output the frame interior only, starting from the fill color everywhere.
    - Project each ring-touching component's two free ends onto interior-edge coordinates.
    - Monochrome traces still produce solid stripes or solid L-paths.
    - Multi-color traces produce interior Manhattan paths painted by the same repeating color cycle.

    Key insight:
    - For cyclic traces, the visible order of colors along the external component reveals a primitive
      cycle; the endpoint colors then select the correct direction and phase for the interior path.
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

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
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

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid in range(1, int(cn) + 1):
        ordered = _ordered_component_path(clab == int(cid))
        if ordered is None:
            continue

        idx_touch = [i for i, p in enumerate(ordered) if p in ring]
        if len(idx_touch) != 2:
            continue
        if sorted(idx_touch) != [0, len(ordered) - 1]:
            continue

        colors_seq = [int(g[r, c]) for r, c in ordered]
        cycle = _primitive_cycle(colors_seq)
        if cycle is None:
            continue

        a_src = ordered[0]
        b_src = ordered[-1]
        a = map_to_out(*a_src)
        b = map_to_out(*b_src)
        ha = int(g[a_src])
        hb = int(g[b_src])

        if len(cycle) == 1:
            colv = int(cycle[0])
            if a[0] == b[0] or a[1] == b[1]:
                deferred[colv].add((int(a[0]), int(a[1])))
                deferred[colv].add((int(b[0]), int(b[1])))
                continue

            pathA = build_manhattan(a, b, True)
            pathB = build_manhattan(a, b, False)
            chosen = pathB if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else pathA
            _paint_cycle(out, chosen, (colv,))
            continue

        pathA = build_manhattan(a, b, True)
        pathB = build_manhattan(a, b, False)
        path = pathB if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else pathA

        chosen: tuple[list[tuple[int, int]], tuple[int, ...]] | None = None
        for path_pts, start_hint, end_hint in ((path, ha, hb), (list(reversed(path)), hb, ha)):
            for cyc in (cycle, tuple(reversed(cycle))):
                aligned = _rotate_cycle_to_start(cyc, int(start_hint))
                if aligned is None:
                    continue
                if int(aligned[(len(path_pts) - 1) % len(aligned)]) != int(end_hint):
                    continue
                chosen = (path_pts, aligned)
                break
            if chosen is not None:
                break

        assert chosen is not None
        path_pts, aligned = chosen
        _paint_cycle(out, path_pts, aligned)

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        tops = [p for p in pts if p[0] == 0]
        bots = [p for p in pts if p[0] == outH - 1]
        if tops and bots:
            for cc in sorted({c for _r, c in pts}):
                rs = [r for r, c in pts if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == outW - 1]
        if lefts and rights:
            for rr in sorted({r for r, _c in pts}):
                cs = [c for r, c in pts if r == rr]
                for cc in range(min(cs), max(cs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        if len(pts) == 1:
            rr, cc = pts[0]
            out[int(rr), int(cc)] = int(col)
            continue
        best = None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
                if best is None or d > best[0]:
                    best = (d, pts[i], pts[j])
        if best is None:
            continue
        _d, p1, p2 = best
        pathA = build_manhattan(p1, p2, True)
        pathB = build_manhattan(p1, p2, False)
        chosen = pathB if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else pathA
        _paint_cycle(out, chosen, (int(col),))

    return out


# --------------- generator (a6, D2 color-cycle variant) ---------------

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
_U_PAIR_PROB = 0.2
_CYCLE_LEN_CHOICES = (3, 4)
_L_SPEC_TRIES = 80
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 600


def _line(a: int, b: int) -> range:
    """Inclusive range from a to b."""
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    """Vertical segment from (r0, c) to (r1, c) inclusive."""
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    """Horizontal segment from (r, c0) to (c1) inclusive."""
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Concatenate path segments, deduplicating shared endpoints."""
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    *,
    start_idx: int = 0,
) -> None:
    """Paint a path with a solid color or a repeating color cycle."""
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    vals = [int(colors[(start_idx + i) % len(colors)]) for i in range(len(path))]
    grid[rs, cs] = vals


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    """Check that all coords are in-bounds, on bg, and not 4-adjacent to existing components."""
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
    colors: tuple[int, ...],
    bg: int,
    *,
    start_idx: int = 0,
) -> bool:
    """Validate, paint, and register a component."""
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors, start_idx=start_idx)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build inner and outer specs for one L-component."""
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


def _u_pair_specs(orientation: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build inner stripes and two outer U-shaped monochrome components."""
    lo, hi = _OUTER_OFFSET_RANGE
    if orientation == "vertical":
        c_a, c_b = sorted(random.sample(range(1, W - 1), 2))
        if c_b - c_a < 2:
            c_b = min(W - 2, c_a + 2)
        top_r = r0 - random.randint(lo, hi)
        bot_r = r1 + random.randint(lo, hi)
        a_top, b_top = (r0 - 1, c0 + 1 + c_a), (r0 - 1, c0 + 1 + c_b)
        a_bot, b_bot = (r1 + 1, c0 + 1 + c_a), (r1 + 1, c0 + 1 + c_b)
        outer_a = _join(
            _vpath(a_top[0], top_r, a_top[1]),
            _hpath(top_r, a_top[1], b_top[1]),
            _vpath(top_r, b_top[0], b_top[1]),
        )
        outer_b = _join(
            _vpath(a_bot[0], bot_r, a_bot[1]),
            _hpath(bot_r, a_bot[1], b_bot[1]),
            _vpath(bot_r, b_bot[0], b_bot[1]),
        )
        inner = [[(r, c_a) for r in range(H)], [(r, c_b) for r in range(H)]]
    else:
        r_a, r_b = sorted(random.sample(range(1, H - 1), 2))
        if r_b - r_a < 2:
            r_b = min(H - 2, r_a + 2)
        left_c = c0 - random.randint(lo, hi)
        right_c = c1 + random.randint(lo, hi)
        a_left, b_left = (r0 + 1 + r_a, c0 - 1), (r0 + 1 + r_b, c0 - 1)
        a_right, b_right = (r0 + 1 + r_a, c1 + 1), (r0 + 1 + r_b, c1 + 1)
        outer_a = _join(
            _hpath(a_left[0], a_left[1], left_c),
            _vpath(a_left[0], b_left[0], left_c),
            _hpath(b_left[0], left_c, b_left[1]),
        )
        outer_b = _join(
            _hpath(a_right[0], a_right[1], right_c),
            _vpath(a_right[0], b_right[0], right_c),
            _hpath(b_right[0], right_c, b_right[1]),
        )
        inner = [[(r_a, c) for c in range(W)], [(r_b, c) for c in range(W)]]
    return inner, outer_a, outer_b


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


def _pick_cycle_palette(avail: list[int]) -> tuple[int, ...]:
    """Choose one shared 3- or 4-color cycle palette for the whole sample."""
    cycle_len_opts = [k for k in _CYCLE_LEN_CHOICES if len(avail) >= k]
    assert cycle_len_opts
    k = random.choice(cycle_len_opts)
    cols = random.sample(avail, k)
    return tuple(int(x) for x in cols)


def _find_cyclic_l(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    used_inner: np.ndarray,
    cycle_palette: tuple[int, ...],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int] | None:
    """Retry until we find an L spec compatible with the shared cycle."""
    k = len(cycle_palette)
    for _ in range(_L_SPEC_TRIES):
        inner, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
        if len(inner) < k or len(outer) < k:
            continue
        if (len(inner) - 1) % k != (len(outer) - 1) % k:
            continue
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            continue
        start_idx = random.randrange(k)
        return inner, outer, int(start_idx)
    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
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
    cycle_palette = _pick_cycle_palette(avail)
    mono_pool = [x for x in avail if x not in cycle_palette]
    random.shuffle(mono_pool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components

    if remaining >= 3 and mono_pool and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 5:
            orientation = "vertical"
        color = int(mono_pool.pop())
        inner_lines, outer_a, outer_b = _u_pair_specs(orientation, H, W, r0, c0, r1, c1)
        all_inner = [p for line in inner_lines for p in line]
        irs, ics = zip(*all_inner)
        if used_inner[irs, ics].any():
            return None
        if not _place_component(grid, comp_mask, outer_a, (color,), bg):
            return None
        if not _place_component(grid, comp_mask, outer_b, (color,), bg):
            return None
        expected[irs, ics] = int(color)
        used_inner[irs, ics] = True
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        spec = _find_cyclic_l(kind, H, W, r0, c0, r1, c1, used_inner, cycle_palette)
        if spec is None:
            return None
        inner, outer, start_idx = spec
        if not _place_component(grid, comp_mask, outer, cycle_palette, bg, start_idx=start_idx):
            return None
        _paint_path(expected, inner, cycle_palette, start_idx=start_idx)
        irs, ics = zip(*inner)
        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if not mono_pool:
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), int(mono_pool.pop()))

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
    raise RuntimeError("failed to generate a valid a6f40cea D2 color-cycle input")
