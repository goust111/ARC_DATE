# Variant of a6f40cea
# Dimension: D7 -- 组件数量与配置 (meta)
# Change: 2-3个开放框架组件（L形或U形对），每个独立处理... -> 大量组件（4-6个）

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# Design analysis for the D7 variant
# ----------------------------------
# a) Concrete behavior:
#    The closed rectangle stays the same, but the outside now carries 4-6 open
#    L-shaped rectangular traces instead of only 2-3 components. Each outside
#    component still maps to one Manhattan path inside the frame, so the visual
#    novelty is not a new shape family but the much denser superposition of many
#    familiar paths. The key local process change is the paint stage:
#    original task: 2-3 paths, usually disjoint, so write order is mostly
#    harmless. Variant: 4-6 paths, frequent same-cell overlaps, so the solver
#    must use a visible ordering rule. We choose the clockwise order of first
#    ring contact along the outer ring around the frame; later components in
#    that order overwrite earlier ones.
# b) Complexity calibration:
#    The variant keeps only one visible ordering axis (clockwise ring order),
#    so the rule is still inferable from a few examples. Complexity comes from
#    quantity and overlap density, not from many unrelated subcases. Components
#    still use one or two colors, and the generator keeps at most two dual-color
#    traces so the palette does not explode.
# c) Implicit assumptions audited from the original main():
#    - Arbitrary component iteration was harmless when paths rarely overlapped;
#      it is no longer safe, so ordering is made explicit and deterministic.
#    - Original generator prevented inner overlap via used_inner; that must be
#      removed, otherwise D7 does not visibly change the task.
#    - Deferred stripe aggregation for same-row/same-col monochrome touches is
#      retained as compatibility logic, but the new generator does not depend on
#      it; all primary components are L-shapes with two adjacent frame sides.
# d) Placement legality:
#    Outside components still may not overlap or even touch in 4-neighborhood,
#    so the generator assigns side touch positions with spacing >= 2 and uses
#    per-corner offset layers for duplicate corners. Grid and frame margins are
#    enlarged to make 4-6 components physically placeable.
# e) Visual observability:
#    Humans can see more components attached all around the frame and can also
#    see that intersections are resolved by a perimeter traversal order rather
#    than by hidden color ids. No sorting uses color ids or other invisible
#    metadata.
# f) Visual cleanliness:
#    The outside remains sparse and non-overlapping; only the cropped interior is
#    dense. Distractors are reduced so the added D7 complexity is not masked by
#    unrelated clutter.
# g) New solver assumptions and generator guarantees:
#    - Each intended primary component is a separate 4-connected component.
#    - Each primary component reaches the frame ring at exactly two positions on
#      adjacent sides.
#    - Side touch positions are distinct enough that the solver can recover one
#      component per outside trace and the generator can place them without
#      collisions.


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


def _clockwise_ring(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> list[tuple[int, int]]:
    """Outer ring cells adjacent to the frame border, scanned clockwise."""
    ring: list[tuple[int, int]] = []
    for cc in range(c0, c1 + 1):
        rr = r0 - 1
        if 0 <= rr < H:
            ring.append((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        cc = c1 + 1
        if 0 <= cc < W:
            ring.append((int(rr), int(cc)))
    for cc in range(c1, c0 - 1, -1):
        rr = r1 + 1
        if 0 <= rr < H:
            ring.append((int(rr), int(cc)))
    for rr in range(r1, r0 - 1, -1):
        cc = c0 - 1
        if 0 <= cc < W:
            ring.append((int(rr), int(cc)))
    return ring


def _build_manhattan(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        step = 1 if cb >= ca else -1
        return [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
    if ca == cb:
        step = 1 if rb >= ra else -1
        return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
    if corner_a:
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


def _inside_corner(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
    outH: int,
    outW: int,
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if corner_a else (ra, cb)
    rr, cc = corner
    return (0 < rr < outH - 1) and (0 < cc < outW - 1)


def _choose_inside_path(
    p1: tuple[int, int],
    p2: tuple[int, int],
    outH: int,
    outW: int,
) -> list[tuple[int, int]]:
    path_a = _build_manhattan(p1, p2, True)
    path_b = _build_manhattan(p1, p2, False)
    if _inside_corner(p1, p2, False, outH, outW) and not _inside_corner(p1, p2, True, outH, outW):
        return path_b
    return path_a


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    """Paint a polyline with one color or an alternating two-color sequence."""
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
        return
    a, b = colors
    vals = [a if i % 2 == 0 else b for i in range(len(path))]
    grid[rs, cs] = vals


def _extract_component_actions(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    bc: int,
    fill: int,
) -> tuple[list[dict[str, object]], dict[int, set[tuple[int, int]]], int, int]:
    """
    Recover inside painting actions from outside components.

    Direct actions are sorted by clockwise first-contact order along the frame's
    outer ring. Deferred monochrome touches that lie on one side are kept for
    compatibility with same-side stripe patterns.
    """
    H, W = g.shape
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        mr = max(0, min(outH - 1, int(rr - r_int0)))
        mc = max(0, min(outW - 1, int(cc - c_int0)))
        return int(mr), int(mc)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = _clockwise_ring(r0, c0, r1, c1, H, W)
    comp_hits: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for order_key, (rr, cc) in enumerate(ring):
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        mr, mc = map_to_out(rr, cc)
        comp_hits[cid].append((int(order_key), int(mr), int(mc), int(g[rr, cc])))

    comp_cols: dict[int, list[int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    actions: list[dict[str, object]] = []
    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid in range(1, int(cn) + 1):
        hits = comp_hits.get(cid, [])
        if len(hits) < 2:
            continue
        cols = comp_cols.get(cid, [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        order_key = min(order for order, _r, _c, _col in hits)
        for _order, rr, cc, col in hits:
            hist[(int(rr), int(cc))][int(col)] += 1
        coords = list(hist.keys())
        if len(coords) < 2:
            continue

        best = None
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                if best is None or d > best[0]:
                    best = (d, coords[i], coords[j])
        if best is None:
            continue
        _dist, a, b = best

        if len(cols) == 1:
            colv = int(cols[0])
            if a[0] == b[0] or a[1] == b[1]:
                for p in coords:
                    deferred[colv].add((int(p[0]), int(p[1])))
                continue
            actions.append(
                {
                    "order": int(order_key),
                    "mode": "solid",
                    "path": _choose_inside_path(a, b, outH, outW),
                    "color": int(colv),
                }
            )
            continue

        if len(cols) == 2:
            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])
            base_path = _choose_inside_path(a, b, outH, outW)
            chosen = None
            for path_pts in (base_path, list(reversed(base_path))):
                for start_color in (ca, cb):
                    if len(path_pts) % 2 == 1:
                        end_color = int(start_color)
                    else:
                        end_color = int(cb) if int(start_color) == int(ca) else int(ca)
                    if int(start_color) == int(ha) and int(end_color) == int(hb):
                        chosen = (path_pts, int(start_color))
                        break
                if chosen is not None:
                    break
            assert chosen is not None
            actions.append(
                {
                    "order": int(order_key),
                    "mode": "alt",
                    "path": chosen[0],
                    "colors": (int(ca), int(cb)),
                    "start": int(chosen[1]),
                }
            )

    actions.sort(key=lambda item: int(item["order"]))
    return actions, deferred, outH, outW


def _paint_actions(out: np.ndarray, actions: list[dict[str, object]]) -> None:
    for action in actions:
        path = list(action["path"])
        if action["mode"] == "solid":
            col = int(action["color"])
            for rr, cc in path:
                out[int(rr), int(cc)] = int(col)
            continue
        ca, cb = map(int, action["colors"])
        cur = int(action["start"])
        for rr, cc in path:
            out[int(rr), int(cc)] = int(cur)
            cur = int(cb) if int(cur) == int(ca) else int(ca)


def _paint_deferred(out: np.ndarray, deferred: dict[int, set[tuple[int, int]]]) -> None:
    outH, outW = out.shape
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
        _dist, p1, p2 = best
        path = _choose_inside_path(p1, p2, outH, outW)
        for rr, cc in path:
            out[int(rr), int(cc)] = int(col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the framed interior by tracing many outside open-rectangle contacts inward.

    Input structure:
    - One 1px closed rectangular frame with a uniform interior fill sits in a larger background.
    - Around that frame are 4-6 colored open-rectangle traces; each trace touches the ring just
      outside the frame at two positions and may be monochrome or two-color alternating.

    Rule:
    - Output only the frame interior, initialized to the frame's fill color.
    - Convert each outside component's ring contacts into interior edge coordinates and rebuild the
      corresponding Manhattan path or stripe.
    - Paint those reconstructed paths in clockwise first-contact order around the frame so later
      components overwrite earlier ones where the denser variant forces overlaps.

    Key insight:
    - The outside traces still encode the same endpoints and alternation phase as in the original
      task, but the increased component count makes write order visible; the reliable order is the
      perimeter scan order of the components' first ring contacts.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    actions, deferred, outH, outW = _extract_component_actions(g, r0, c0, r1, c1, bg, bc, fill)

    out = np.full((outH, outW), int(fill), dtype=int)
    _paint_actions(out, actions)
    _paint_deferred(out, deferred)
    return out


# --------------- generator (a6 D7 variant) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.7
_INTERIOR_H_RANGE = (11, 18)
_INTERIOR_W_RANGE = (11, 18)
_GRID_SIZE_RANGE = (40, 56)
_FRAME_MARGIN = 10
_FRAME_MARGIN_FAR = 16
_COMPONENT_COUNT_RANGE = (4, 4)
_OUTER_OFFSET_BASE = 2
_OUTER_OFFSET_STEP = 3
_DUAL_COMPONENT_CAP = 2
_DUAL_REQUIRED_PROB = 0.75
_DUAL_COLOR_PROB = 0.5
_DISTRACTOR_COUNT_RANGE = (0, 1)
_DISTRACTOR_SIZE_RANGE = (1, 3)
_DISTRACTOR_TRIES = 50
_DISTRACTOR_CLEARANCE = 3
_GEN_RETRIES = 800


def _line(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    cells = set(coords)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in cells and 0 <= nr < H and 0 <= nc < W and comp_mask[nr, nc]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _center_spaced_positions(count: int, size: int) -> list[int] | None:
    """Pick interior positions in 1..size-2 with gap >= 2."""
    if count == 0:
        return []
    slack = int(size - 3 - 2 * (count - 1))
    if slack < 0:
        return None
    center_offset = slack // 2
    wiggle = min(1, center_offset, slack - center_offset)
    offset = center_offset
    if wiggle > 0:
        offset = center_offset + random.randint(-wiggle, wiggle)
    return [int(1 + offset + 2 * i) for i in range(count)]


def _component_endpoints(kind: str, row: int, col: int, H: int, W: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if kind == "tl":
        return (0, int(col)), (int(row), 0)
    if kind == "tr":
        return (0, int(col)), (int(row), int(W - 1))
    if kind == "bl":
        return (int(H - 1), int(col)), (int(row), 0)
    if kind == "br":
        return (int(H - 1), int(col)), (int(row), int(W - 1))
    raise ValueError(kind)


def _l_specs(
    kind: str,
    row: int,
    col: int,
    layer: int,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    """Return inner path, outer path, and the two ring-touch cells for one L component."""
    row_offset = int(_OUTER_OFFSET_BASE + (0 if kind.endswith("l") else 3) + layer * _OUTER_OFFSET_STEP)
    col_offset = int(_OUTER_OFFSET_BASE + (0 if kind.startswith("t") else 3) + layer * _OUTER_OFFSET_STEP)
    if kind == "tl":
        ring_a = (int(r0 - 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c0 - 1))
        outer = _join(
            _vpath(ring_a[0], r0 - row_offset, ring_a[1]),
            _hpath(r0 - row_offset, ring_a[1], c0 - col_offset),
            _vpath(r0 - row_offset, ring_b[0], c0 - col_offset),
            _hpath(ring_b[0], c0 - col_offset, ring_b[1]),
        )
    elif kind == "tr":
        ring_a = (int(r0 - 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c1 + 1))
        outer = _join(
            _vpath(ring_a[0], r0 - row_offset, ring_a[1]),
            _hpath(r0 - row_offset, ring_a[1], c1 + col_offset),
            _vpath(r0 - row_offset, ring_b[0], c1 + col_offset),
            _hpath(ring_b[0], c1 + col_offset, ring_b[1]),
        )
    elif kind == "bl":
        ring_a = (int(r1 + 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c0 - 1))
        outer = _join(
            _vpath(ring_a[0], r1 + row_offset, ring_a[1]),
            _hpath(r1 + row_offset, ring_a[1], c0 - col_offset),
            _vpath(r1 + row_offset, ring_b[0], c0 - col_offset),
            _hpath(ring_b[0], c0 - col_offset, ring_b[1]),
        )
    elif kind == "br":
        ring_a = (int(r1 + 1), int(c0 + 1 + col))
        ring_b = (int(r0 + 1 + row), int(c1 + 1))
        outer = _join(
            _vpath(ring_a[0], r1 + row_offset, ring_a[1]),
            _hpath(r1 + row_offset, ring_a[1], c1 + col_offset),
            _vpath(r1 + row_offset, ring_b[0], c1 + col_offset),
            _hpath(ring_b[0], c1 + col_offset, ring_b[1]),
        )
    else:
        raise ValueError(kind)

    p1, p2 = _component_endpoints(kind, row, col, H, W)
    inner = _choose_inside_path(p1, p2, H, W)
    return inner, outer, ring_a, ring_b


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    H, W = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r = random.randint(1, H - 2)
        c = random.randint(1, W - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _step in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (nr, nc) not in used
            ]
            if not opts:
                break
            pick = random.choice(opts)
            used.add(pick)
            coords.append(pick)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _component_overlap_stats(actions: list[dict[str, object]]) -> tuple[int, int]:
    cover = Counter()
    for action in actions:
        for cell in action["path"]:
            cover[tuple(cell)] += 1
    overlap_cells = sum(1 for v in cover.values() if v >= 2)
    overwrite_events = sum(v - 1 for v in cover.values() if v >= 2)
    return int(overlap_cells), int(overwrite_events)


def _gen_attempt(debug: Counter | None = None, raise_main: bool = False) -> np.ndarray | None:
    def fail(reason: str) -> None:
        if debug is not None:
            debug[reason] += 1

    bg = int(random.choice(_BG_CHOICES))
    bc = int(_BORDER_COLOR)
    fill_choices = [x for x in range(10) if x not in (bg, bc)]
    fill = int(bg if random.random() < _FILL_EQ_BG_PROB else random.choice(fill_choices))

    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows_min = max(_GRID_SIZE_RANGE[0], H + 2 + _FRAME_MARGIN + _FRAME_MARGIN_FAR)
    cols_min = max(_GRID_SIZE_RANGE[0], W + 2 + _FRAME_MARGIN + _FRAME_MARGIN_FAR)
    if rows_min > _GRID_SIZE_RANGE[1] or cols_min > _GRID_SIZE_RANGE[1]:
        fail("grid")
        return None
    rows = random.randint(rows_min, _GRID_SIZE_RANGE[1])
    cols = random.randint(cols_min, _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - (H + 2) - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - (W + 2) - _FRAME_MARGIN_FAR)
    r1, c1 = int(r0 + H + 1), int(c0 + W + 1)

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    n_components = random.randint(*_COMPONENT_COUNT_RANGE)
    base_kinds = ["tl", "tr", "bl", "br"]
    extra_kinds = [random.choice(base_kinds) for _ in range(n_components - 4)]
    kinds = base_kinds + extra_kinds
    random.shuffle(kinds)

    top_count = sum(1 for kind in kinds if kind in ("tl", "tr"))
    bot_count = sum(1 for kind in kinds if kind in ("bl", "br"))
    left_count = sum(1 for kind in kinds if kind in ("tl", "bl"))
    right_count = sum(1 for kind in kinds if kind in ("tr", "br"))
    top_cols = _center_spaced_positions(top_count, W)
    bot_cols = _center_spaced_positions(bot_count, W)
    left_rows = _center_spaced_positions(left_count, H)
    right_rows = _center_spaced_positions(right_count, H)
    if top_cols is None or bot_cols is None or left_rows is None or right_rows is None:
        fail("spacing")
        return None

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    max_dual = min(_DUAL_COMPONENT_CAP, max(0, len(palette) - n_components))
    if max_dual > 0 and n_components >= 5 and random.random() < _DUAL_REQUIRED_PROB:
        dual_count = random.randint(1, max_dual)
    else:
        dual_count = random.randint(0, max_dual)
    dual_ids = set(random.sample(range(n_components), k=dual_count))

    color_need = n_components + dual_count
    if color_need > len(palette):
        fail("palette")
        return None

    components: list[dict[str, object]] = []
    kind_layers = Counter()
    top_ids = [idx for idx, kind in enumerate(kinds) if kind in ("tl", "tr")]
    bot_ids = [idx for idx, kind in enumerate(kinds) if kind in ("bl", "br")]
    left_ids = [idx for idx, kind in enumerate(kinds) if kind in ("tl", "bl")]
    right_ids = [idx for idx, kind in enumerate(kinds) if kind in ("tr", "br")]
    random.shuffle(top_ids)
    random.shuffle(bot_ids)
    random.shuffle(left_ids)
    random.shuffle(right_ids)
    top_map = {idx: int(top_cols[pos]) for pos, idx in enumerate(top_ids)}
    bot_map = {idx: int(bot_cols[pos]) for pos, idx in enumerate(bot_ids)}
    left_map = {idx: int(left_rows[pos]) for pos, idx in enumerate(left_ids)}
    right_map = {idx: int(right_rows[pos]) for pos, idx in enumerate(right_ids)}

    color_i = 0
    for idx, kind in enumerate(kinds):
        layer = int(kind_layers[kind])
        kind_layers[kind] += 1
        row = int(left_map[idx] if kind in ("tl", "bl") else right_map[idx])
        col = int(top_map[idx] if kind in ("tl", "tr") else bot_map[idx])
        if idx in dual_ids:
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        else:
            colors = (int(palette[color_i]),)
            color_i += 1
        inner, outer, ring_a, ring_b = _l_specs(kind, row, col, layer, H, W, r0, c0, r1, c1)
        components.append(
            {
                "kind": kind,
                "layer": layer,
                "row": row,
                "col": col,
                "colors": colors,
                "inner": inner,
                "outer": outer,
                "ring_a": ring_a,
                "ring_b": ring_b,
            }
        )

    comp_mask = np.zeros_like(grid, dtype=bool)
    for comp_i, comp in enumerate(components):
        if not _place_component(grid, comp_mask, list(comp["outer"]), tuple(comp["colors"]), bg):
            fail(f"place:{comp_i}:{comp['kind']}:{comp['layer']}")
            return None

    for extra_color in palette[color_i : color_i + random.randint(*_DISTRACTOR_COUNT_RANGE)]:
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), int(extra_color))

    try:
        actual = main(grid)
    except Exception as exc:
        if raise_main:
            raise
        fail(f"main:{type(exc).__name__}")
        return None

    found = _find_framed_rect_uniform_fill(grid, bg=bg)
    if found is None:
        fail("frame")
        return None
    rr0, cc0, rr1, cc1, bbc, ffill = found
    actions, deferred, outH, outW = _extract_component_actions(grid, rr0, cc0, rr1, cc1, bg, bbc, ffill)
    if deferred:
        fail("deferred")
        return None
    if len(actions) != n_components:
        fail("action_count")
        return None
    if actual.shape != (outH, outW):
        fail("shape")
        return None

    if dual_count > 0 and not any(action["mode"] == "alt" for action in actions):
        fail("alt_missing")
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea D7 input")
