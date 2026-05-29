# Combo variant of a6f40cea (index 1)
# combo_spec: D1/1+D12/0
# Name: 障碍绕行路径

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# -----------------------------------------------------------------------------
# Design analysis for the combo D1/1 + D12/0 variant
#
# a) Concrete behavior and trace
#    - The outer-language of the task stays the same: a single closed frame is the
#      crop anchor, and 2-3 open rectangular traces outside the frame still define
#      which interior border points must be connected and which colors to use.
#    - The change is entirely inside the frame. The old "ignore tiny distractors"
#      rule is removed; instead the former distractor blobs are copied into the
#      frame interior and become forbidden cells for path reconstruction.
#    - Small traced example: suppose one outside L-trace maps to endpoints
#      (0,5) and (4,0). The original solver paints the single-bend interior L
#      using the preferred elbow near (4,5). In the combo variant a 2x1 blob can
#      sit on the top border route and another 1-cell blob can sit on the left
#      border route, so neither canonical one-bend L is available. The recovered
#      route must snake through the remaining gap, producing a shortest BFS path
#      with at least two bends. The output still preserves the obstacle cells.
#
# b) Complexity calibration
#    - Only one new semantic object class is introduced: 1-2 small obstacle blobs
#      of size 1-4, reusing the old distractor visual scale.
#    - The outside components remain L-only in this combo so the main novelty is
#      obstacle-aware routing, not mixing several orthogonal sub-rules at once.
#    - Obstacles are forced to matter: the generator rejects samples unless at
#      least one path deviates from the original single-bend L and has >=2 bends.
#
# c) Broken original assumptions
#    - The frame interior is no longer perfectly uniform, so frame detection must
#      infer the dominant fill color rather than require every interior cell to
#      match it.
#    - The old solver initialized a clean fill canvas, which would erase internal
#      blobs; the new solver starts from the observed interior crop.
#    - The old path builder assumed one Manhattan elbow was always enough. That
#      assumption fails once former distractors become blocking cells, so shortest
#      path search replaces direct elbow construction for L-components.
#    - The old inside-corner preference still matters only as BFS tie-breaking.
#
# d) Placement legality
#    - The generator first samples non-overlapping L-style endpoint tasks and the
#      matching outside open-rect shapes.
#    - One selected task is converted into the detour showcase: two tiny
#      rectangular blobs (each size <=4) are placed so they block the two old
#      one-bend candidates while leaving one shortest 2-bend corridor open.
#    - Remaining tasks are accepted only if their solved paths stay disjoint from
#      both those blobs and the showcased detour path. Final acceptance still uses
#      an oracle check: generated input must satisfy main(input) == expected.
#
# e) Visual observability
#    - Humans can see two consistent cues across examples: the same outer traces
#      still indicate endpoint/color information, and the newly visible interior
#      blobs remain in the output while paths bend around them.
#    - No hidden ordering by color id or similar non-visual convention is used;
#      only geometry and observed endpoint colors matter.
#
# f) Visual cleanliness
#    - Obstacles stay tiny, use one dedicated color, and are limited to 1-2 blobs.
#      Paths are rejected if they overlap each other, so the cropped interiors stay
#      readable rather than degenerating into tangled paint.
#
# g) Inferable color behavior
#    - Path colors still come directly from each outside component: monochrome
#      traces paint solid paths, two-color traces alternate with phase anchored by
#      the ring-touch endpoint colors.
#    - Obstacle color has a separate role and is never reused by any outside
#      component, so a viewer can separate "thing to keep" from "thing to route".
#
# Post-change solver assumptions and generator guarantees
#    - Solver assumption: every internal non-fill cell belongs to at most two
#      small 4-connected blobs. Generator guarantee: exactly 1-2 such blobs are
#      created, each size 1-4, and they never touch each other orthogonally.
#    - Solver assumption: deterministic BFS with the encoded endpoint/color hints
#      is enough to recover the intended path. Generator guarantee: every sample
#      is accepted only if all paths are solvable, pairwise disjoint, and the
#      solver reproduces the constructed output exactly.
#    - Solver assumption: the dominant interior color is the frame fill. Generator
#      guarantee: obstacle area is tiny compared with the interior, so the fill
#      remains the strict majority color inside the frame.
# -----------------------------------------------------------------------------

_MAX_OBSTACLE_BLOBS = 2
_MAX_OBSTACLE_SIZE = 4


def _find_framed_rect_with_sparse_obstacles(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """Detect the main 1px frame and infer its dominant interior fill."""
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, fill_count, r0, c0, r1, c1, border_color, fill_color)
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
            fill, fill_count = Counter(map(int, inner.flatten())).most_common(1)[0]
            fill = int(fill)
            fill_count = int(fill_count)
            if fill == int(bc):
                continue
            if fill_count * 2 <= int(inner.size):
                continue

            diff = inner != int(fill)
            if diff.any():
                dlab, dn = label(diff, structure=structure)
                if int(dn) > _MAX_OBSTACLE_BLOBS:
                    continue
                ok = True
                for did in range(1, int(dn) + 1):
                    mask = dlab == int(did)
                    if int(mask.sum()) > _MAX_OBSTACLE_SIZE:
                        ok = False
                        break
                    cols = set(map(int, inner[mask].flatten()))
                    if len(cols) != 1 or int(fill) in cols or int(bc) in cols:
                        ok = False
                        break
                if not ok:
                    continue

            area = int(h * w)
            cand = (area, fill_count, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[:2] > best[:2]:
                best = cand

    if best is None:
        return None
    _area, _fill_count, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _line(a: int, b: int) -> range:
    step = 1 if int(b) >= int(a) else -1
    return range(int(a), int(b) + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r in _line(int(r0), int(r1))]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in _line(int(c0), int(c1))]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            q = (int(p[0]), int(p[1]))
            if not out or out[-1] != q:
                out.append(q)
    return out


def _build_manhattan(
    p1: tuple[int, int], p2: tuple[int, int], prefer_vertical_first: bool
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        return [(int(ra), int(c)) for c in _line(int(ca), int(cb))]
    if ca == cb:
        return [(int(r), int(ca)) for r in _line(int(ra), int(rb))]
    if prefer_vertical_first:
        return _join(_vpath(int(ra), int(rb), int(ca)), _hpath(int(rb), int(ca), int(cb)))
    return _join(_hpath(int(ra), int(ca), int(cb)), _vpath(int(ra), int(rb), int(cb)))


def _inside_corner(
    p1: tuple[int, int], p2: tuple[int, int], prefer_vertical_first: bool, out_h: int, out_w: int
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if prefer_vertical_first else (ra, cb)
    rr, cc = corner
    return 0 < int(rr) < int(out_h) - 1 and 0 < int(cc) < int(out_w) - 1


def _preferred_vertical_first(
    p1: tuple[int, int], p2: tuple[int, int], out_h: int, out_w: int
) -> bool:
    use_v = True
    use_h = False
    if _inside_corner(p1, p2, use_h, out_h, out_w) and not _inside_corner(p1, p2, use_v, out_h, out_w):
        return False
    return True


def _ordered_dirs(
    pos: tuple[int, int], end: tuple[int, int], prefer_vertical_first: bool
) -> list[tuple[int, int]]:
    r, c = map(int, pos)
    er, ec = map(int, end)
    dr = 1 if er > r else -1 if er < r else 0
    dc = 1 if ec > c else -1 if ec < c else 0
    ordered: list[tuple[int, int]] = []
    if prefer_vertical_first:
        if dr != 0:
            ordered.append((int(dr), 0))
        if dc != 0:
            ordered.append((0, int(dc)))
    else:
        if dc != 0:
            ordered.append((0, int(dc)))
        if dr != 0:
            ordered.append((int(dr), 0))
    for step in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
        if step not in ordered:
            ordered.append(step)
    return ordered


def _shortest_path(
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: np.ndarray,
    *,
    prefer_vertical_first: bool,
) -> list[tuple[int, int]] | None:
    """Shortest 4-neighbor path with deterministic tie-breaking."""
    h, w = blocked.shape
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))
    if blocked[start] or blocked[end]:
        return None

    q: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for dr, dc in _ordered_dirs(cur, end, prefer_vertical_first):
            nr, nc = int(cur[0] + dr), int(cur[1] + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (int(nr), int(nc))
            if blocked[nxt] or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)

    if end not in prev:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append((int(cur[0]), int(cur[1])))
        cur = prev[cur]
    path.reverse()
    return path


def _count_bends(path: list[tuple[int, int]]) -> int:
    bends = 0
    for i in range(1, len(path) - 1):
        dr1 = int(path[i][0] - path[i - 1][0])
        dc1 = int(path[i][1] - path[i - 1][1])
        dr2 = int(path[i + 1][0] - path[i][0])
        dc2 = int(path[i + 1][1] - path[i][1])
        if (dr1, dc1) != (dr2, dc2):
            bends += 1
    return int(bends)


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    start_color: int | None = None,
) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    cur = int(a if start_color is None else start_color)
    vals = []
    for _ in path:
        vals.append(int(cur))
        cur = int(b) if int(cur) == int(a) else int(a)
    grid[rs, cs] = vals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the cropped frame interior by routing the encoded connections around visible blockers.

    Input structure:
    - One 1px closed rectangular frame encloses an interior whose dominant color is the base fill.
    - The interior may also contain 1-2 small obstacle blobs copied from former distractors.
    - Colored open-rect components remain outside the frame and touch its ring, encoding endpoint positions and mono/alternating path colors.

    Rule:
    - Output only the frame interior, preserving every obstacle blob exactly where it appears.
    - Project each ring-touching outside component onto interior border coordinates.
    - Connect the chosen endpoints by a shortest 4-connected path that avoids all obstacle cells.
    - Monochrome components paint solid paths; two-color components paint alternating paths whose phase is fixed by the endpoint colors observed on the ring.

    Key insight:
    - The outside traces still tell you which border points and colors belong together; the new reasoning step is that every internal non-fill cell is now functional terrain, so the intended path is the deterministic BFS route around those blockers rather than a fixed single-bend L.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_with_sparse_obstacles(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    inner = np.array(g[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    out_h, out_w = inner.shape
    out = inner.copy()
    obstacle_mask = inner != int(fill)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        mr = max(0, min(out_h - 1, int(rr - r_int0)))
        mc = max(0, min(out_w - 1, int(cc - c_int0)))
        return int(mr), int(mc)

    outside_mask = np.ones_like(g, dtype=bool)
    outside_mask[r0 + 1 : r1, c0 + 1 : c1] = False
    comp_mask = outside_mask & (g != int(bg)) & (g != int(fill)) & (g != int(bc))
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

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[int(rr), int(cc)])
        if cid == 0:
            continue
        mr, mc = map_to_out(int(rr), int(cc))
        comp_pts[int(cid)].append((int(mr), int(mc), int(g[int(rr), int(cc)])))

    comp_cols: dict[int, list[int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[int(cid)] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid in sorted(comp_pts):
        pts = comp_pts[int(cid)]
        if len(pts) < 2:
            continue
        cols = comp_cols.get(int(cid), [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1
        coords = list(hist.keys())

        if len(cols) == 1:
            colv = int(cols[0])
            rows = {int(r) for r, _c in coords}
            cols2 = {int(c) for _r, c in coords}
            if len(rows) == 1 or len(cols2) == 1:
                for p in coords:
                    deferred[colv].add((int(p[0]), int(p[1])))
                continue

            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(int(coords[i][0]) - int(coords[j][0])) + abs(int(coords[i][1]) - int(coords[j][1]))
                    if best is None or d > best[0]:
                        best = (int(d), coords[i], coords[j])
            if best is None:
                continue
            _d, p1, p2 = best
            prefer_vertical_first = _preferred_vertical_first(p1, p2, out_h, out_w)
            path = _shortest_path(
                p1, p2, obstacle_mask, prefer_vertical_first=prefer_vertical_first
            )
            assert path is not None
            for rr, cc in path:
                out[int(rr), int(cc)] = int(colv)
            continue

        if len(cols) == 2:
            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(int(coords[i][0]) - int(coords[j][0])) + abs(int(coords[i][1]) - int(coords[j][1]))
                    if best is None or d > best[0]:
                        best = (int(d), coords[i], coords[j])
            if best is None:
                continue
            _d, a, b = best
            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])
            prefer_vertical_first = _preferred_vertical_first(a, b, out_h, out_w)
            path = _shortest_path(
                a, b, obstacle_mask, prefer_vertical_first=prefer_vertical_first
            )
            assert path is not None

            chosen = None
            for path_pts in (path, list(reversed(path))):
                for start_color in (int(ca), int(cb)):
                    end_color = int(start_color) if ((len(path_pts) - 1) % 2 == 0) else (int(cb) if int(start_color) == int(ca) else int(ca))
                    if int(start_color) != int(ha):
                        continue
                    if int(end_color) != int(hb):
                        continue
                    chosen = (path_pts, int(start_color))
                    break
                if chosen is not None:
                    break
            assert chosen is not None

            path_pts, start_color = chosen
            _paint_path(out, path_pts, (int(ca), int(cb)), start_color=int(start_color))
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue

        top_cols = {int(c) for r, c in pts if int(r) == 0}
        bot_cols = {int(c) for r, c in pts if int(r) == out_h - 1}
        if top_cols and bot_cols:
            for cc in sorted(top_cols & bot_cols):
                path = _shortest_path(
                    (0, int(cc)),
                    (out_h - 1, int(cc)),
                    obstacle_mask,
                    prefer_vertical_first=True,
                )
                assert path is not None
                for rr, cc2 in path:
                    out[int(rr), int(cc2)] = int(col)
            continue

        left_rows = {int(r) for r, c in pts if int(c) == 0}
        right_rows = {int(r) for r, c in pts if int(c) == out_w - 1}
        if left_rows and right_rows:
            for rr in sorted(left_rows & right_rows):
                path = _shortest_path(
                    (int(rr), 0),
                    (int(rr), out_w - 1),
                    obstacle_mask,
                    prefer_vertical_first=False,
                )
                assert path is not None
                for rr2, cc in path:
                    out[int(rr2), int(cc)] = int(col)
            continue

        if len(pts) == 1:
            rr, cc = pts[0]
            out[int(rr), int(cc)] = int(col)
            continue

        best = None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = abs(int(pts[i][0]) - int(pts[j][0])) + abs(int(pts[i][1]) - int(pts[j][1]))
                if best is None or d > best[0]:
                    best = (int(d), pts[i], pts[j])
        if best is None:
            continue
        _d, p1, p2 = best
        prefer_vertical_first = _preferred_vertical_first(p1, p2, out_h, out_w)
        path = _shortest_path(
            p1, p2, obstacle_mask, prefer_vertical_first=prefer_vertical_first
        )
        assert path is not None
        for rr, cc in path:
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (a6 combo D1/1 + D12/0) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (6, 13)
_INTERIOR_W_RANGE = (8, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.45
_COMPONENT_TRIES = 80
_GEN_RETRIES = 400


def _can_place(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    bg: int,
    comp_mask: np.ndarray,
) -> bool:
    H, W = grid.shape
    s = {(int(r), int(c)) for r, c in coords}
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) not in s and 0 <= nr < H and 0 <= nc < W and comp_mask[int(nr), int(nc)]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
    start_color: int | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors, start_color=start_color)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _choose_outer_coloring(
    outer: list[tuple[int, int]],
    colors: tuple[int, int],
    need_a: int,
    need_b: int,
) -> tuple[list[tuple[int, int]], int] | None:
    ring_a = (int(outer[0][0]), int(outer[0][1]))
    ring_b = (int(outer[-1][0]), int(outer[-1][1]))
    for cand_path in (outer, list(reversed(outer))):
        for start_color in colors:
            cur = int(start_color)
            first_col = None
            last_col = None
            for idx, p in enumerate(cand_path):
                if p == ring_a:
                    first_col = int(cur)
                if p == ring_b:
                    last_col = int(cur)
                if idx != len(cand_path) - 1:
                    cur = int(colors[1]) if int(cur) == int(colors[0]) else int(colors[0])
            if int(first_col) == int(need_a) and int(last_col) == int(need_b):
                return cand_path, int(start_color)
    return None


def _rect_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, rect)
    if r0 > r1 or c0 > c1:
        return []
    return [(int(r), int(c)) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)]


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int) -> dict[str, object]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    if is_top:
        edge_r = 0
        side_r = random.randint(2, H - 2)
    else:
        edge_r = H - 1
        side_r = random.randint(1, H - 3)

    if is_left:
        edge_c = 0
        side_c = random.randint(2, W - 2)
    else:
        edge_c = W - 1
        side_c = random.randint(1, W - 3)

    p1 = (int(edge_r), int(side_c))
    p2 = (int(side_r), int(edge_c))
    prefer_vertical_first = _preferred_vertical_first(p1, p2, H, W)
    preferred_path = _build_manhattan(p1, p2, prefer_vertical_first)

    lo, hi = _OUTER_OFFSET_RANGE
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))
    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + side_c)
    ring_b = (r0 + 1 + side_r, c0 - 1 if is_left else c1 + 1)
    outer = _join(
        _vpath(int(ring_a[0]), int(R), int(ring_a[1])),
        _hpath(int(R), int(ring_a[1]), int(C)),
        _vpath(int(R), int(ring_b[0]), int(C)),
        _hpath(int(ring_b[0]), int(C), int(ring_b[1])),
    )

    return {
        "kind": kind,
        "p1": (int(p1[0]), int(p1[1])),
        "p2": (int(p2[0]), int(p2[1])),
        "prefer_vertical_first": bool(prefer_vertical_first),
        "preferred_path": [(int(r), int(c)) for r, c in preferred_path],
        "outer": [(int(r), int(c)) for r, c in outer],
    }


def _detour_l_spec(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int) -> dict[str, object] | None:
    if H < 6 or W < 8:
        return None

    span_r = random.choice([2, 3])
    span_c = random.choice([2, 3])
    gate = random.randint(1, span_c - 1)
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    if kind == "tl":
        end_r = int(span_r)
        end_c = int(span_c)
        gate_c = int(gate)
        p1 = (0, end_c)
        p2 = (end_r, 0)
        detour_path = _join(_hpath(0, end_c, gate_c), _vpath(0, end_r, gate_c), _hpath(end_r, gate_c, 0))
        rects = [(1, 0, end_r - 1, gate_c - 1), (1, gate_c + 1, end_r - 1, end_c)]
    elif kind == "tr":
        end_r = int(span_r)
        end_c = int(W - 1 - span_c)
        gate_c = int(W - 1 - gate)
        p1 = (0, end_c)
        p2 = (end_r, W - 1)
        detour_path = _join(_hpath(0, end_c, gate_c), _vpath(0, end_r, gate_c), _hpath(end_r, gate_c, W - 1))
        rects = [(1, end_c, end_r - 1, gate_c - 1), (1, gate_c + 1, end_r - 1, W - 1)]
    elif kind == "bl":
        end_r = int(H - 1 - span_r)
        end_c = int(span_c)
        gate_c = int(gate)
        p1 = (H - 1, end_c)
        p2 = (end_r, 0)
        detour_path = _join(_hpath(H - 1, end_c, gate_c), _vpath(H - 1, end_r, gate_c), _hpath(end_r, gate_c, 0))
        rects = [(end_r + 1, 0, H - 2, gate_c - 1), (end_r + 1, gate_c + 1, H - 2, end_c)]
    else:
        end_r = int(H - 1 - span_r)
        end_c = int(W - 1 - span_c)
        gate_c = int(W - 1 - gate)
        p1 = (H - 1, end_c)
        p2 = (end_r, W - 1)
        detour_path = _join(_hpath(H - 1, end_c, gate_c), _vpath(H - 1, end_r, gate_c), _hpath(end_r, gate_c, W - 1))
        rects = [(end_r + 1, end_c, H - 2, gate_c - 1), (end_r + 1, gate_c + 1, H - 2, W - 1)]

    obstacle_blobs = []
    for rect in rects:
        cells = _rect_cells(rect)
        if not cells or len(cells) > _MAX_OBSTACLE_SIZE:
            return None
        obstacle_blobs.append(cells)

    prefer_vertical_first = _preferred_vertical_first(p1, p2, H, W)
    preferred_path = _build_manhattan(p1, p2, prefer_vertical_first)
    if _count_bends(detour_path) < 2 or detour_path == preferred_path:
        return None

    lo, hi = _OUTER_OFFSET_RANGE
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))
    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + end_c)
    ring_b = (r0 + 1 + end_r, c0 - 1 if is_left else c1 + 1)
    outer = _join(
        _vpath(int(ring_a[0]), int(R), int(ring_a[1])),
        _hpath(int(R), int(ring_a[1]), int(C)),
        _vpath(int(R), int(ring_b[0]), int(C)),
        _hpath(int(ring_b[0]), int(C), int(ring_b[1])),
    )
    return {
        "kind": kind,
        "p1": (int(p1[0]), int(p1[1])),
        "p2": (int(p2[0]), int(p2[1])),
        "prefer_vertical_first": bool(prefer_vertical_first),
        "preferred_path": [(int(r), int(c)) for r, c in preferred_path],
        "detour_path": [(int(r), int(c)) for r, c in detour_path],
        "outer": [(int(r), int(c)) for r, c in outer],
        "obstacle_blobs": [[(int(r), int(c)) for r, c in blob] for blob in obstacle_blobs],
    }


def _solve_tasks_with_obstacles(
    tasks: list[dict[str, object]],
    obstacle_mask: np.ndarray,
) -> list[list[tuple[int, int]]] | None:
    paths: list[list[tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()
    for task in tasks:
        path = _shortest_path(
            tuple(task["p1"]),
            tuple(task["p2"]),
            obstacle_mask,
            prefer_vertical_first=bool(task["prefer_vertical_first"]),
        )
        if path is None:
            return None
        if any((int(r), int(c)) in occupied for r, c in path):
            return None
        occupied.update((int(r), int(c)) for r, c in path)
        paths.append([(int(r), int(c)) for r, c in path])
    return paths


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

    available = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(available)
    if len(available) < 4:
        return None
    obstacle_color = int(available[0])
    palette = [int(x) for x in available[1:]]

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    corners = random.sample(["tl", "tr", "bl", "br"], k=int(n_components))
    detour_kind = corners[0]
    detour_spec = _detour_l_spec(detour_kind, H, W, r0, c0, r1, c1)
    if detour_spec is None:
        return None

    tasks: list[dict[str, object]] = [detour_spec]
    blocked = np.zeros((H, W), dtype=bool)
    for blob in detour_spec["obstacle_blobs"]:
        for rr, cc in blob:
            blocked[int(rr), int(cc)] = True

    reserved = blocked.copy()
    target_path = [(int(r), int(c)) for r, c in detour_spec["detour_path"]]
    rs, cs = zip(*target_path)
    reserved[rs, cs] = True

    for kind in corners[1:]:
        placed = False
        for _ in range(_COMPONENT_TRIES):
            spec = _l_specs(kind, H, W, r0, c0, r1, c1)
            path = [(int(r), int(c)) for r, c in spec["preferred_path"]]
            prs, pcs = zip(*path)
            if reserved[prs, pcs].any():
                continue
            tasks.append(spec)
            reserved[prs, pcs] = True
            placed = True
            break
        if not placed:
            return None

    solved_paths = _solve_tasks_with_obstacles(tasks, blocked)
    if solved_paths is None:
        return None
    if solved_paths[0] != target_path or _count_bends(solved_paths[0]) < 2:
        return None
    obstacle_blobs = [[(int(r), int(c)) for r, c in blob] for blob in detour_spec["obstacle_blobs"]]

    inner = np.full((H, W), fill, dtype=int)
    expected = np.full((H, W), fill, dtype=int)
    for blob in obstacle_blobs:
        for rr, cc in blob:
            inner[int(rr), int(cc)] = int(obstacle_color)
            expected[int(rr), int(cc)] = int(obstacle_color)

    grid[r0 + 1 : r1, c0 + 1 : c1] = inner

    comp_mask = np.zeros_like(grid, dtype=bool)
    color_i = 0
    for task, path in zip(tasks, solved_paths):
        if color_i >= len(palette):
            return None
        outer_path = [(int(r), int(c)) for r, c in task["outer"]]
        want_dual = random.random() < _DUAL_COLOR_PROB and color_i + 1 < len(palette)
        if want_dual:
            colors2 = (int(palette[color_i]), int(palette[color_i + 1]))
            options: list[tuple[int, list[tuple[int, int]], int]] = []
            for inner_start in colors2:
                inner_end = int(inner_start) if ((len(path) - 1) % 2 == 0) else (int(colors2[1]) if int(inner_start) == int(colors2[0]) else int(colors2[0]))
                chosen_outer = _choose_outer_coloring(outer_path, colors2, int(inner_start), int(inner_end))
                if chosen_outer is not None:
                    options.append((int(inner_start), chosen_outer[0], int(chosen_outer[1])))
            if options:
                inner_start, chosen_outer_path, outer_start = random.choice(options)
                if not _place_component(
                    grid, comp_mask, chosen_outer_path, colors2, bg, start_color=int(outer_start)
                ):
                    return None
                _paint_path(expected, path, colors2, start_color=int(inner_start))
                color_i += 2
                continue

        color = int(palette[color_i])
        if not _place_component(grid, comp_mask, outer_path, (int(color),), bg):
            return None
        _paint_path(expected, path, (int(color),))
        color_i += 1

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
    raise RuntimeError("failed to generate a valid a6f40cea combo D1/1+D12/0 input")
