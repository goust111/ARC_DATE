# Variant of a6f40cea
# Dimension: D12 — 干扰项角色与处理 (obstacle)
# Change: 0-2个小色块干扰项（1-4个像素），距框架至少2像素，不触碰框边，不影响输出... → 干扰项作为路径障碍

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# -----------------------------------------------------------------------------
# Design analysis for the D12 obstacle variant
#
# a) Concrete behavior
#    - The closed frame stays 1px thick and the outside open-rect components still
#      define which interior endpoints must be connected.
#    - The change is entirely inside the frame: 0-2 small blobs now sit inside the
#      cropped output area, remain visible in the output, and act as forbidden
#      cells for every reconstructed path.
#    - Small traced example:
#        Original: a top+left L-component maps to endpoints (0,4) and (3,0), so
#        the solver paints the canonical Manhattan L inside the frame.
#        Variant: if a 2-cell blob occupies two cells on that canonical L, the
#        solver must still connect the same endpoints, but by a different shortest
#        orthogonal route that bends around the blob. The obstacle cells themselves
#        stay unchanged in the output.
#
# b) Complexity calibration
#    - Obstacles are still only 0-2 connected blobs of size 1-4, so the visual
#      vocabulary stays small and repeated across examples.
#    - They are placed on or immediately beside canonical paths often enough that
#      they visibly matter, but the interior remains sparse and readable.
#    - U-pairs are kept, but colors stay unique across unrelated components so
#      obstacle handling does not create grouping ambiguity.
#
# c) Original assumptions that break
#    - The old frame detector required the entire interior to be uniform. That is
#      no longer valid; it must now allow a majority fill plus a few tiny internal
#      exceptions.
#    - The old solver initialized the output as a flat fill canvas, so any internal
#      non-fill pixels would be lost. The new solver starts from the observed
#      interior so obstacles survive into the output.
#    - The old path logic assumed each path could be represented by one fixed
#      Manhattan elbow or straight stripe. That fails once a blob blocks that
#      route, so the new logic uses shortest-path search with deterministic tie
#      breaking.
#
# d) Placement legality
#    - The generator still places outside components with the original non-overlap
#      checks.
#    - Obstacles are restricted to the strict interior (never on the inner border),
#      never cover endpoints, never overlap each other, and are accepted only if
#      every required path remains solvable and pairwise disjoint.
#
# e) Visual observability
#    - The interior blobs are directly visible in the input and remain visible in
#      the output, while the new paths visibly bend around them. This makes the
#      new rule inferable from a few examples.
#    - Zero-obstacle examples are still allowed, but the generator biases toward
#      1-2 obstacles so the new behavior is usually demonstrated.
#
# f) Visual cleanliness
#    - Obstacles stay small and off the frame border, and generated paths are
#      rejected if they would intersect each other. This avoids tangled interiors.
#
# g) Color inference
#    - Obstacle colors are just object identities; no ordering or hidden color-id
#      semantics are used. Path colors are still read directly from the outside
#      components, exactly as before.
#
# Post-change solver assumptions and generator guarantees
#    - The solver assumes every internal non-fill cell belongs to at most two small
#      4-connected blobs that do not touch the inner border; the generator enforces
#      that.
#    - The solver assumes each required connection has at least one shortest route
#      around the blobs and that deterministic BFS will not force two paths to
#      overlap; invalid generations are rejected by an oracle check.
#    - The reference source-task id cb2d8a2c was provided as inspiration, but no
#      solver code for it exists in the allowed local files, so the implementation
#      uses an in-file pathfinding baseline instead of borrowing unseen code.
# -----------------------------------------------------------------------------


def _find_framed_rect_with_sparse_obstacles(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame whose interior is mostly uniform
    fill plus up to two tiny obstacle blobs.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest match.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, r0, c0, r1, c1, bc, fill)
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
            fill = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
            if fill == int(bc):
                continue

            diff = inner != int(fill)
            if diff.any():
                if diff[0, :].any() or diff[-1, :].any() or diff[:, 0].any() or diff[:, -1].any():
                    continue
                dlab, dn = label(diff, structure=structure)
                if int(dn) > 2:
                    continue
                ok = True
                for did in range(1, int(dn) + 1):
                    mask = dlab == int(did)
                    if int(mask.sum()) > 4:
                        ok = False
                        break
                    cols = set(map(int, inner[mask].flatten()))
                    if len(cols) != 1 or int(fill) in cols or int(bc) in cols:
                        ok = False
                        break
                if not ok:
                    continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


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


def _preferred_vertical_first(p1: tuple[int, int], p2: tuple[int, int], out_h: int, out_w: int) -> bool:
    use_v = True
    use_h = False
    if _inside_corner(p1, p2, use_h, out_h, out_w) and not _inside_corner(p1, p2, use_v, out_h, out_w):
        return False
    return True


def _ordered_dirs(
    pos: tuple[int, int], end: tuple[int, int], prefer_vertical_first: bool
) -> list[tuple[int, int]]:
    r, c = pos
    er, ec = end
    dr = 1 if er > r else -1 if er < r else 0
    dc = 1 if ec > c else -1 if ec < c else 0
    ordered: list[tuple[int, int]] = []
    if prefer_vertical_first:
        if dr != 0:
            ordered.append((dr, 0))
        if dc != 0:
            ordered.append((0, dc))
    else:
        if dc != 0:
            ordered.append((0, dc))
        if dr != 0:
            ordered.append((dr, 0))
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
            nxt = (nr, nc)
            if blocked[nxt] or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)

    if end not in prev:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    vals = [a if i % 2 == 0 else b for i in range(len(path))]
    grid[rs, cs] = vals


def _solve_tasks_with_obstacles(
    tasks: list[dict[str, object]], obstacle_mask: np.ndarray
) -> list[list[tuple[int, int]]] | None:
    """Solve all intended interior connections; reject overlapping paths."""
    paths: list[list[tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()
    for task in tasks:
        p1 = task["p1"]
        p2 = task["p2"]
        prefer_vertical_first = bool(task["prefer_vertical_first"])
        path = _shortest_path(
            p1, p2, obstacle_mask, prefer_vertical_first=prefer_vertical_first
        )
        if path is None:
            return None
        if any(p in occupied for p in path):
            return None
        occupied.update(path)
        paths.append(path)
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the frame interior by routing projected traces around internal blockers.

    Input structure:
    - A uniform background, one 1px closed rectangular frame, and colored open-rect
      components outside the frame that touch its ring.
    - The frame interior is mostly a single fill color, but may contain 0-2 tiny
      colored blobs entirely inside the frame. These blobs are obstacles.

    Rule:
    - Output is the cropped frame interior itself: keep the fill and keep every
      internal obstacle blob exactly where it appears.
    - Each outside component still determines one interior connection between the
      frame-touch coordinates obtained by projection through the border.
    - Monochrome components paint solid routes; two-color components paint
      alternating routes whose endpoint phase is fixed by the observed ring colors.
      Every route must avoid the obstacle cells.

    Key insight:
    - The outside shapes still tell you only the endpoints and colors; the new
      information that changes the route is the set of internal non-fill cells,
      which must be treated as forbidden cells during path reconstruction.
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
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        col = int(g[rr, cc])
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((mr, mc, col))

    comp_cols: dict[int, list[int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue
        cols = comp_cols.get(cid, [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1
        coords = list(hist.keys())

        if len(cols) == 1:
            colv = int(cols[0])
            rows = {r for r, _c in coords}
            cols2 = {c for _r, c in coords}
            if len(rows) == 1 or len(cols2) == 1:
                for p in coords:
                    deferred[colv].add((int(p[0]), int(p[1])))
                continue

            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if best is None or d > best[0]:
                        best = (d, coords[i], coords[j])
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
                    d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if best is None or d > best[0]:
                        best = (d, coords[i], coords[j])
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
                for start_color in (ca, cb):
                    end_color = (
                        start_color
                        if (len(path_pts) - 1) % 2 == 0
                        else (cb if start_color == ca else ca)
                    )
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
            cur = int(start_color)
            for rr, cc in path_pts:
                out[int(rr), int(cc)] = int(cur)
                cur = int(cb) if cur == int(ca) else int(ca)
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue

        tops = {c for r, c in pts if r == 0}
        bots = {c for r, c in pts if r == out_h - 1}
        if tops and bots:
            for cc in sorted(tops & bots):
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

        lefts = {r for r, c in pts if c == 0}
        rights = {r for r, c in pts if c == out_w - 1}
        if lefts and rights:
            for rr in sorted(lefts & rights):
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
                d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
                if best is None or d > best[0]:
                    best = (d, pts[i], pts[j])
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


# --------------- generator (a6 D12 obstacle variant) ---------------

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
_U_PAIR_PROB = 0.25
_DUAL_COLOR_PROB = 0.45
_OBSTACLE_COUNT_OPTIONS = [0, 1, 1, 2, 2]
_OBSTACLE_TRIES = 80
_GEN_RETRIES = 400
_OBSTACLE_SHAPES = [
    [(0, 0)],
    [(0, 0), (1, 0)],
    [(0, 0), (-1, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (0, -1)],
    [(0, 0), (1, 0), (0, 1)],
    [(0, 0), (-1, 0), (0, 1)],
    [(0, 0), (1, 0), (0, -1)],
    [(0, 0), (-1, 0), (0, -1)],
    [(0, 0), (1, 0), (-1, 0)],
    [(0, 0), (0, 1), (0, -1)],
    [(0, 0), (1, 0), (0, 1), (1, 1)],
]


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if grid[r, c] != bg:
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and comp_mask[nr, nc]:
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


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r = random.randint(1, max(1, H // 2)) if is_top else random.randint(max(1, H // 2), H - 2)
    c = random.randint(1, max(1, W // 2)) if is_left else random.randint(max(1, W // 2), W - 2)

    lo, hi = _OUTER_OFFSET_RANGE
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + c)
    ring_b = (r0 + 1 + r, c0 - 1 if is_left else c1 + 1)

    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )

    p1 = (0 if is_top else H - 1, c)
    p2 = (r, 0 if is_left else W - 1)
    return (p1, p2), outer


def _u_pair_specs(orientation: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
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
        tasks = [
            {"p1": (0, c_a), "p2": (H - 1, c_a), "prefer_vertical_first": True},
            {"p1": (0, c_b), "p2": (H - 1, c_b), "prefer_vertical_first": True},
        ]
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
        tasks = [
            {"p1": (r_a, 0), "p2": (r_a, W - 1), "prefer_vertical_first": False},
            {"p1": (r_b, 0), "p2": (r_b, W - 1), "prefer_vertical_first": False},
        ]
    return tasks, outer_a, outer_b


def _canonical_task_path(task: dict[str, object]) -> list[tuple[int, int]]:
    return _build_manhattan(
        task["p1"],
        task["p2"],
        bool(task["prefer_vertical_first"]),
    )


def _path_cells_strictly_inside(path: list[tuple[int, int]], H: int, W: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for rr, cc in path[1:-1]:
        if 0 < int(rr) < H - 1 and 0 < int(cc) < W - 1:
            out.append((int(rr), int(cc)))
    return out


def _blob_coords_from_shape(
    anchor: tuple[int, int], shape: list[tuple[int, int]], pivot: tuple[int, int]
) -> list[tuple[int, int]]:
    ar, ac = anchor
    pr, pc = pivot
    return [(int(ar + dr - pr), int(ac + dc - pc)) for dr, dc in shape]


def _plan_obstacles(
    tasks: list[dict[str, object]],
    H: int,
    W: int,
    want_count: int,
) -> tuple[list[list[tuple[int, int]]], np.ndarray, list[list[tuple[int, int]]]] | None:
    base_blocked = np.zeros((H, W), dtype=bool)
    base_paths = _solve_tasks_with_obstacles(tasks, base_blocked)
    if base_paths is None:
        return None
    if want_count == 0:
        return [], base_blocked, base_paths

    endpoints = {
        tuple(task["p1"]) for task in tasks
    } | {
        tuple(task["p2"]) for task in tasks
    }
    canonical = [_canonical_task_path(task) for task in tasks]
    obstacle_blobs: list[list[tuple[int, int]]] = []
    blocked = np.zeros((H, W), dtype=bool)
    changed_any = False

    for _ in range(want_count):
        placed = False
        task_order = list(range(len(tasks)))
        random.shuffle(task_order)
        for idx in task_order:
            candidates = _path_cells_strictly_inside(canonical[idx], H, W)
            random.shuffle(candidates)
            if not candidates:
                continue
            shapes = list(_OBSTACLE_SHAPES)
            random.shuffle(shapes)
            for cell in candidates:
                for shape in shapes:
                    pivots = list(shape)
                    random.shuffle(pivots)
                    for pivot in pivots:
                        coords = _blob_coords_from_shape(cell, shape, pivot)
                        unique = set(coords)
                        if len(unique) != len(coords):
                            continue
                        if any(not (0 < r < H - 1 and 0 < c < W - 1) for r, c in coords):
                            continue
                        if any((r, c) in endpoints for r, c in coords):
                            continue
                        if any(blocked[r, c] for r, c in coords):
                            continue

                        test_blocked = blocked.copy()
                        for r, c in coords:
                            test_blocked[r, c] = True
                        solved = _solve_tasks_with_obstacles(tasks, test_blocked)
                        if solved is None:
                            continue
                        if solved[idx] == canonical[idx]:
                            continue

                        blocked = test_blocked
                        obstacle_blobs.append(coords)
                        changed_any = True
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    if not changed_any:
        return None

    solved_paths = _solve_tasks_with_obstacles(tasks, blocked)
    if solved_paths is None:
        return None
    return obstacle_blobs, blocked, solved_paths


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice(
        [x for x in range(10) if x not in (bg, bc)]
    )
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
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    color_i = 0
    tasks: list[dict[str, object]] = []
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components

    if remaining >= 2 and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 5:
            orientation = "vertical"
        if color_i >= len(palette):
            return None
        color = int(palette[color_i])
        color_i += 1
        stripe_tasks, outer_a, outer_b = _u_pair_specs(orientation, H, W, r0, c0, r1, c1)
        stripe_paths = [_canonical_task_path(task) for task in stripe_tasks]
        for path in stripe_paths:
            rs, cs = zip(*path)
            if used_inner[rs, cs].any():
                return None
        for task in stripe_tasks:
            task["colors"] = (color,)
            tasks.append(task)
        for path in stripe_paths:
            rs, cs = zip(*path)
            used_inner[rs, cs] = True
        if not _place_component(grid, comp_mask, outer_a, (color,), bg):
            return None
        if not _place_component(grid, comp_mask, outer_b, (color,), bg):
            return None
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        if color_i >= len(palette) - 1:
            return None
        (p1, p2), outer = _l_specs(kind, H, W, r0, c0, r1, c1)
        if random.random() < _DUAL_COLOR_PROB:
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        else:
            colors = (int(palette[color_i]),)
            color_i += 1
        task = {
            "p1": p1,
            "p2": p2,
            "prefer_vertical_first": _preferred_vertical_first(p1, p2, H, W),
            "colors": colors,
        }
        path = _canonical_task_path(task)
        rs, cs = zip(*path)
        if used_inner[rs, cs].any():
            return None
        tasks.append(task)
        used_inner[rs, cs] = True
        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None

    want_obstacles = random.choice(_OBSTACLE_COUNT_OPTIONS)
    if want_obstacles > 0 and color_i + want_obstacles > len(palette):
        return None
    planned = _plan_obstacles(tasks, H, W, want_obstacles)
    if planned is None:
        return None
    obstacle_blobs, obstacle_mask, solved_paths = planned
    if len(obstacle_blobs) != want_obstacles:
        return None

    for blob in obstacle_blobs:
        if color_i >= len(palette):
            return None
        color = int(palette[color_i])
        color_i += 1
        for rr, cc in blob:
            grid[r0 + 1 + rr, c0 + 1 + cc] = color
            expected[rr, cc] = color

    for task, path in zip(tasks, solved_paths):
        _paint_path(expected, path, tuple(task["colors"]))

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
    raise RuntimeError("failed to generate a valid a6f40cea D12 obstacle input")
