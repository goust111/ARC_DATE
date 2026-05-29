# Variant of a6f40cea
# Dimension: D1 — 路径形态 (path)
# Change: 两端点间的曼哈顿L形路径（一个拐角，由水平段+垂直段组成），拐角选择使其严格位于内部而非边缘... → 螺旋路径

# Design analysis
# ===============
# 1. Source-task check
#    - The prompt names `da515329` as the spiral reference.
#    - I searched the current workspace for a readable solver source file for that task and only
#      found the JSON examples in the ARC data directories, not a `da515329.py` implementation.
#      The spiral mechanism below therefore uses the prompt's stated behavior plus the visible
#      sample pattern ("walk straight, turn 90 degrees when blocked by the boundary or by already
#      visited path cells"), without borrowing code from an unavailable solver.
#
# 2. Concrete behavior change
#    - Original a6f40cea maps each outside open-rect component to two boundary endpoints of the
#      framed interior, then fills a Manhattan L path or a straight stripe between them.
#    - In this variant the outside components still only tell us:
#      1) which two frame sides are connected,
#      2) which colors are involved,
#      3) the endpoint-color hints for alternating components.
#    - The changed part is the interior reconstruction rule:
#      instead of drawing a one-elbow Manhattan path, each usable endpoint pair now defines a
#      deterministic spiral walk. The walk starts from one endpoint, keeps moving forward when
#      possible, and otherwise rotates 90 degrees in a fixed handedness. The walk stops exactly
#      when it first reaches the other endpoint.
#
# 3. Why the generator cannot keep the original corner sampling unchanged
#    - A pure spiral between adjacent frame sides is much longer than the original L path.
#    - If every corner were allowed at once, the resulting paths would frequently overlap, because
#      a top-left spiral wants a tall left strip while a bottom-left spiral wants a wide bottom
#      band. Those domains collide even when the original L-shapes would not.
#    - To keep D1 as the only semantic change and avoid introducing hidden order-dependent
#      overwrites, the generator restricts itself to two mirrored top-anchored components:
#      one top-to-left spiral and one top-to-right spiral. Their strip domains are disjoint and
#      still clearly show the new spiral-vs-L distinction.
#    - The solver remains more permissive: it can still reconstruct same-side stripes exactly like
#      the original, and it also supports mirrored bottom-band spirals if such inputs are ever
#      provided.
#
# 4. Small traced example
#    - Suppose the mapped endpoints are top `(0, 4)` and left `(3, 0)` in a `7x8` output.
#    - Original logic: choose an elbow and paint `(0,4)->(3,4)->(3,0)` or `(0,4)->(0,0)->(3,0)`.
#      Exactly one turn.
#    - Variant logic: use the full-height left strip `rows 0..6, cols 0..4`, start at `(0,4)`,
#      move downward, then turn left at the strip bottom, then turn upward at the strip's left
#      edge, and stop on first arrival at `(3,0)`. The prefix is
#      `(0,4)...(6,4)->(6,3)...(6,0)->(5,0)->(4,0)->(3,0)`.
#      This is still fully determined by the endpoints, but it has two turns and a visibly wrapped
#      shape instead of a single elbow.
#
# 5. Complexity calibration / observability
#    - The path family is intentionally small: top-left and top-right spirals in the generator,
#      plus mirrored bottom-band support in the solver. This avoids a many-case puzzle with only
#      2-3 training examples.
#    - Every generated sample contains two spiral components. There is no branch where all ring
#      components collapse back to straight stripes or original L paths.
#    - Alternating two-color components are retained, but only when the outer open-rect parity can
#      support the same endpoint-color relation as the spiral prefix. This preserves the original
#      "endpoint hints anchor the phase" idea without hidden recoloring.
#
# 6. Implicit-assumption audit against the original solver
#    - Original `build_manhattan(...)` assumed the interior path was fully determined by the two
#      endpoints and, for non-straight cases, by choosing one of two elbows. That is no longer
#      true; the replacement must also know a start orientation and a turn handedness.
#    - Original `inside_corner(...)` only existed to choose between the two elbow placements.
#      That assumption disappears entirely.
#    - Original "farthest pair" selection is still safe because the outside open-rect still exposes
#      exactly two salient ring-touch coordinates for each spiral component.
#    - Original unconditional painting was harmless because a one-elbow path occupied a local
#      corner region. With spirals that can span tall strips, overlap would become destructive, so
#      the generator now reserves the entire spiral prefix in `used_inner` before accepting a
#      component.
#
# 7. Placement legality
#    - Top-left spiral domain: full-height strip `[0:outH, 0:start_col+1]`.
#    - Top-right spiral domain: full-height strip `[0:outH, start_col:outW]`.
#    - The two generated spirals are accepted only if their painted cells are disjoint.
#    - The chosen start columns leave at least one untouched center column, so the strips do not
#      overlap and the exterior open-rects can also be placed without touching each other.
#
# 8. Visual cleanliness / color inference
#    - Colors are still direct component colors; there is no color-id-based sorting or remapping.
#    - Distractors remain small isolated blobs away from the frame, so the main cue is still the
#      pair of open-rect components feeding wrapped paths into the cropped output.
#    - Two-color spirals alternate along the entire spiral prefix; the phase is inferable from the
#      ring-touch endpoint colors exactly as in the original task.

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
    """Detect the largest 1px rectangular frame with a uniform interior fill."""
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


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
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


def _spiral_walk_local(
    height: int,
    width: int,
    start: tuple[int, int],
    dir0: str,
    turn_delta: int,
) -> list[tuple[int, int]]:
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
    idx = {"U": 0, "R": 1, "D": 2, "L": 3}[dir0]
    visited = {tuple(start)}
    path = [tuple(start)]
    while True:
        moved = False
        for cand in (idx, (idx + turn_delta) % 4, (idx + 2 * turn_delta) % 4, (idx + 3 * turn_delta) % 4):
            dr, dc = dirs[cand]
            nr = int(path[-1][0] + dr)
            nc = int(path[-1][1] + dc)
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                idx = cand
                path.append((nr, nc))
                visited.add((nr, nc))
                moved = True
                break
        if not moved:
            return path


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev = None
    for i in range(1, len(path)):
        d = (int(path[i][0] - path[i - 1][0]), int(path[i][1] - path[i - 1][1]))
        if prev is not None and d != prev:
            turns += 1
        prev = d
    return int(turns)


def _spiral_path_between(
    p1: tuple[int, int],
    p2: tuple[int, int],
    outH: int,
    outW: int,
) -> list[tuple[int, int]] | None:
    """Recover the deterministic spiral path for supported endpoint-side pairs."""
    a = tuple(map(int, p1))
    b = tuple(map(int, p2))

    if a[0] == 0 and b[1] == 0:
        start, target = a, b
        local = _spiral_walk_local(outH, start[1] + 1, (0, start[1]), "D", +1)
        target_local = (target[0], 0)
        prefix = local[: local.index(target_local) + 1]
        return [(r, c) for r, c in prefix]

    if b[0] == 0 and a[1] == 0:
        return _spiral_path_between(b, a, outH, outW)

    if a[0] == 0 and b[1] == outW - 1:
        start, target = a, b
        width = int(outW - start[1])
        local = _spiral_walk_local(outH, width, (0, 0), "D", -1)
        target_local = (target[0], width - 1)
        prefix = local[: local.index(target_local) + 1]
        return [(r, start[1] + c) for r, c in prefix]

    if b[0] == 0 and a[1] == outW - 1:
        return _spiral_path_between(b, a, outH, outW)

    if a[1] == 0 and b[0] == outH - 1:
        start, target = a, b
        height = int(outH - start[0])
        local = _spiral_walk_local(height, outW, (0, 0), "R", +1)
        target_local = (height - 1, target[1])
        prefix = local[: local.index(target_local) + 1]
        return [(start[0] + r, c) for r, c in prefix]

    if b[1] == 0 and a[0] == outH - 1:
        return _spiral_path_between(b, a, outH, outW)

    if a[1] == outW - 1 and b[0] == outH - 1:
        start, target = a, b
        height = int(outH - start[0])
        local = _spiral_walk_local(height, outW, (0, outW - 1), "L", -1)
        target_local = (height - 1, target[1])
        prefix = local[: local.index(target_local) + 1]
        return [(start[0] + r, c) for r, c in prefix]

    if b[1] == outW - 1 and a[0] == outH - 1:
        return _spiral_path_between(b, a, outH, outW)

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct wrapped spiral traces inside a framed rectangle by reading the outside open-rect clues.

    Input structure:
    - A background field, one complete 1px rectangular frame, and a uniform fill inside that frame.
    - Colored components outside the frame touch the one-cell ring around the frame on two sides.

    Rule:
    - Output only the framed interior, initially filled with the frame's fill color.
    - Map each ring-touch coordinate onto the interior boundary.
    - Same-side monochrome components still become straight stripes, but adjacent-side components now
      expand into deterministic spiral prefixes that turn 90 degrees whenever forward motion is blocked.
    - Two-color components alternate along the recovered spiral, with the phase anchored by the two
      endpoint colors observed on the ring-touch cells.

    Key insight:
    - The outside component still gives only endpoints plus colors; the whole interior path is recovered
      because each supported endpoint-side pair implies a unique spiral domain, start direction, and
      turn handedness.
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
            spiral = _spiral_path_between(p1, p2, outH, outW)
            if spiral is None:
                continue
            for rr, cc in spiral:
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
            path = _spiral_path_between(a, b, outH, outW)
            assert path is not None

            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])

            chosen_start = None
            for start_color in (ca, cb):
                end_color = start_color if ((len(path) - 1) % 2 == 0) else (cb if start_color == ca else ca)
                if int(start_color) == int(ha) and int(end_color) == int(hb):
                    chosen_start = int(start_color)
                    break
            assert chosen_start is not None

            cur = int(chosen_start)
            for rr, cc in path:
                out[int(rr), int(cc)] = int(cur)
                cur = int(cb) if cur == int(ca) else int(ca)
            continue

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
        spiral = _spiral_path_between(p1, p2, outH, outW)
        if spiral is None:
            continue
        for rr, cc in spiral:
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (a6 D1 spiral variant) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (6, 13)
_INTERIOR_W_RANGE = (9, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400
_COMPONENT_TRIES = 80
_OUTER_TRIES = 30


def _outer_open_rect_top(
    side: str,
    start_col: int,
    target_row: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> list[tuple[int, int]]:
    lo, hi = _OUTER_OFFSET_RANGE
    ring_a = (r0 - 1, c0 + 1 + start_col)
    if side == "left":
        ring_b = (r0 + 1 + target_row, c0 - 1)
        R = r0 - random.randint(lo, hi)
        C = c0 - random.randint(lo, hi)
    else:
        ring_b = (r0 + 1 + target_row, c1 + 1)
        R = r0 - random.randint(lo, hi)
        C = c1 + random.randint(lo, hi)
    return _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )


def _pick_top_spiral(
    kind: str,
    H: int,
    W: int,
    used_inner: np.ndarray,
) -> tuple[list[tuple[int, int]], int, int] | None:
    """Return (inner_path, start_col, target_row) for kind tl/tr."""
    for _ in range(_COMPONENT_TRIES):
        if kind == "tl":
            start_lo = 2
            start_hi = max(2, W // 2 - 1)
            if start_lo > start_hi:
                return None
            start_col = random.randint(start_lo, start_hi)
        else:
            start_lo = max(W // 2 + 1, 2)
            start_hi = W - 3
            if start_lo > start_hi:
                return None
            start_col = random.randint(start_lo, start_hi)

        target_row = random.randint(1, H - 2)
        if kind == "tl":
            path = _spiral_path_between((0, start_col), (target_row, 0), H, W)
        else:
            path = _spiral_path_between((0, start_col), (target_row, W - 1), H, W)
        if path is None:
            continue
        if _path_turns(path) < 2:
            continue
        irs, ics = zip(*path)
        if used_inner[irs, ics].any():
            continue
        return path, int(start_col), int(target_row)
    return None


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
        if _place_component(grid, comp_mask, coords, (color,), bg):
            return


def _legacy_l_path(
    p1: tuple[int, int],
    p2: tuple[int, int],
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        return [(ra, c) for c in _line(ca, cb)]
    if ca == cb:
        return [(r, ca) for r in _line(ra, rb)]
    path = [(r, ca) for r in _line(ra, rb)]
    path += [(rb, c) for c in _line(ca, cb)][1:]
    return path


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

    expected = np.full((H, W), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)

    if len(palette) < 2:
        return None

    left_spec = _pick_top_spiral("tl", H, W, used_inner)
    if left_spec is None:
        return None
    left_inner, left_start_col, left_target_row = left_spec
    irs, ics = zip(*left_inner)
    used_inner[irs, ics] = True

    right_spec = _pick_top_spiral("tr", H, W, used_inner)
    if right_spec is None:
        return None
    right_inner, right_start_col, right_target_row = right_spec
    irs, ics = zip(*right_inner)
    used_inner[irs, ics] = True

    specs = [
        ("left", left_inner, left_start_col, left_target_row),
        ("right", right_inner, right_start_col, right_target_row),
    ]

    color_i = 0
    legacy_diff = False
    for side, inner, start_col, target_row in specs:
        if color_i >= len(palette):
            return None
        outer = None
        colors: tuple[int, ...] | None = None

        want_dual = color_i + 1 < len(palette) and random.random() < _DUAL_COLOR_PROB
        for _ in range(_OUTER_TRIES):
            cand_outer = _outer_open_rect_top(side, start_col, target_row, r0, c0, r1, c1)
            if want_dual:
                cand_colors = (palette[color_i], palette[color_i + 1])
                if ((len(cand_outer) - 1) % 2) != ((len(inner) - 1) % 2):
                    continue
            else:
                cand_colors = (palette[color_i],)
            if not _can_place(grid, cand_outer, bg, comp_mask):
                continue
            outer = cand_outer
            colors = cand_colors
            break

        if outer is None or colors is None:
            return None

        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None
        _paint_path(expected, inner, colors)
        if len(colors) == 2:
            color_i += 2
        else:
            color_i += 1

        if side == "left":
            legacy = _legacy_l_path((0, start_col), (target_row, 0))
        else:
            legacy = _legacy_l_path((0, start_col), (target_row, W - 1))
        if legacy != inner:
            legacy_diff = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), palette[color_i])
        color_i += 1

    if not legacy_diff:
        return None

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
    raise RuntimeError("failed to generate a valid a6f40cea spiral-variant input")
