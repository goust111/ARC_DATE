# Combo variant of a6f40cea (index 16)
# combo_spec: D8/1+D10/2
# Name: 外角决定拐点与色相
#
# Design analysis
# - Concrete behavior:
#   The closed frame is unchanged, but every generated ring-touching component is
#   an L-shaped open rectangle. The solver reads which top/bottom side and which
#   left/right side a component touches, recovers the component bbox, and projects
#   that exterior bbox corner into the cropped interior. That projected corner is
#   the output elbow. For two-color L components, the same exterior corner also
#   names color A by showing color A on the outer-corner cell itself; color B is
#   the other component color. The interior path is traversed in a fixed geometric
#   order from the top/bottom-side touch to the left/right-side touch, and its
#   starting color is chosen from the projected elbow height: upper half => A,
#   lower half => B.
# - Behavior trace:
#   Example 1: a top-left component maps to interior touch points (0, 5) and
#   (3, 0). Original a6f40cea picks the strictly-inside elbow (3, 5), then anchors
#   alternation from ring endpoint colors. This combo instead projects the outer
#   corner to (0, 0), so the path runs across the top edge then down the left edge.
#   Because that corner is in the upper half, the interior path starts with color A.
#   Example 2: a bottom-right component with mapped touches (H-1, 4) and (6, W-1)
#   projects its outer corner to (H-1, W-1). The path hugs the bottom and right
#   borders, and because the corner is in the lower half the path starts with
#   color B. The same visible outside corner therefore controls both geometry and
#   phase.
# - Complexity calibration:
#   If we kept U-pairs or allowed the old inside-corner fallback for generated L
#   traces, many samples would still be solvable by the original rule and the D8
#   change would be visually weak. The generator therefore emits only 2-3 L
#   components and forces at least one top-corner two-color L plus one bottom-
#   corner two-color L. That gives repeated support for both the shape cue and the
#   upper-vs-lower phase cue without creating too many cases.
# - Implicit assumption audit:
#   The original solver assumed the elbow could be inferred solely from the two
#   mapped ring endpoints and that endpoint colors were the trustworthy phase
#   anchor. Both assumptions are false here. The combo solver must additionally
#   recover the outside bbox corner, must use a fixed top/bottom -> left/right
#   traversal order for two-color L paths, and must ignore ring endpoint colors
#   when deciding phase for those L components. Non-L fallback branches keep the
#   original behavior for robustness, but the generator no longer relies on them.
# - Placement legality:
#   The generator stays output-first. It samples top L rows from the upper half and
#   bottom L rows from the lower half so corner-hugging paths usually occupy
#   disjoint edge segments. It still validates every outer component with the same
#   non-overlap and no-adjacency checks as the source task, and retries if an
#   extra component would collide. Distractors remain isolated from the frame ring.
# - Visual observability:
#   A human can compare a top-corner and a bottom-corner two-color example and see
#   that the outside corner location predicts both which output corner the path
#   hugs and which color appears first on the interior path. No color-id ordering
#   or hidden metadata is used: the relevant cues are the visible outside corner,
#   the touched sides, and the corner's vertical half.
# - Visual cleanliness and color inference:
#   The number of traces stays small, each path uses at most two colors, and the
#   color-role marker is embedded in the existing outside corner instead of adding
#   extra tokens. Because the generator forces both upper and lower dual-color
#   cases, the A/B phase rule has repeated support across a small sample set.

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


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = a, b
    if ra == rb:
        return [(int(ra), int(c)) for c in _line(int(ca), int(cb))]
    if ca == cb:
        return [(int(r), int(ca)) for r in _line(int(ra), int(rb))]
    raise ValueError("segment endpoints must align")


def _build_corner_path(
    p_tb: tuple[int, int],
    p_lr: tuple[int, int],
    corner: tuple[int, int],
) -> list[tuple[int, int]]:
    head = _segment(p_tb, corner)
    tail = _segment(corner, p_lr)
    return head + tail[1:]


def _build_manhattan(
    p1: tuple[int, int], p2: tuple[int, int], corner_a: bool
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return _segment(p1, p2)
    if corner_a:
        return _segment(p1, (int(rb), int(ca))) + _segment((int(rb), int(ca)), p2)[1:]
    return _segment(p1, (int(ra), int(cb))) + _segment((int(ra), int(cb)), p2)[1:]


def _inside_corner(
    p1: tuple[int, int], p2: tuple[int, int], corner_a: bool, out_h: int, out_w: int
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if corner_a else (ra, cb)
    rr, cc = corner
    return (0 < rr < out_h - 1) and (0 < cc < out_w - 1)


def _paint_alternating_linear(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    start_color: int,
    other_color: int,
) -> None:
    cur = int(start_color)
    nxt = int(other_color)
    for rr, cc in path:
        grid[int(rr), int(cc)] = int(cur)
        cur, nxt = nxt, cur


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    *,
    anchor: tuple[int, int] | None = None,
) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
        return
    a, b = map(int, colors)
    if anchor is None:
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
    else:
        idx = path.index(anchor)
        vals = [a if abs(i - idx) % 2 == 0 else b for i in range(len(path))]
    grid[rs, cs] = vals


def _dominant_coord(entries: list[tuple[int, int, int]]) -> tuple[int, int]:
    r, c = Counter((int(r), int(c)) for r, c, _col in entries).most_common(1)[0][0]
    return int(r), int(c)


def _other_color(colors: tuple[int, int], color: int) -> int:
    a, b = map(int, colors)
    return int(b) if int(color) == int(a) else int(a)


def _start_color_from_corner(colors: tuple[int, int], corner_r: int, out_h: int) -> int:
    return int(colors[0]) if 2 * int(corner_r) < int(out_h) else int(colors[1])


def _choose_half_rows(h: int, upper: bool) -> list[int]:
    vals = [r for r in range(1, h - 1) if (2 * r < h) == upper]
    if vals:
        return vals
    return [1] if upper else [h - 2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the cropped frame interior by extending the outside open rectangles inward.

    Input structure:
    - A uniform background contains one closed 1px rectangular frame with a uniform fill.
    - Colored open-rectangle components lie outside the frame and touch exactly the ring just beyond it.

    Rule:
    - Output only the frame interior, initialized to the fill color.
    - Monochrome traces still produce solid stripes or solid L paths between their projected touch points.
    - For an L component, recover which top/bottom side and which left/right side it touches, then project the component's outside bbox corner into the cropped output and force the interior Manhattan path to pass through that projected corner.
    - For a two-color L component, read color A from that outside corner cell, let the other component color be B, and start the alternating interior path with A for an upper-half projected corner or with B for a lower-half projected corner.

    Key insight:
    The same visible outside corner now drives both decisions at once: its projected position chooses the output elbow, and its vertical half chooses the alternation phase while the corner cell itself identifies color A.
    """
    g = np.array(grid, dtype=int)
    grid_h, grid_w = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    out = np.full((out_h, out_w), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        mr = max(0, min(out_h - 1, int(rr - r_int0)))
        mc = max(0, min(out_w - 1, int(cc - c_int0)))
        return int(mr), int(mc)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < grid_h:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < grid_w:
                ring.add((int(rr), int(cc)))

    comp_pts: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((int(rr), int(cc), int(mr), int(mc), int(g[rr, cc])))

    comp_cols: dict[int, list[int]] = {}
    comp_boxes: dict[int, tuple[int, int, int, int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))
        pts = np.argwhere(clab == int(cid))
        if pts.size == 0:
            continue
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        comp_boxes[cid] = (int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1]))

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue
        cols = comp_cols.get(cid, [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        side_hits: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        for rr, cc, mr, mc, col in pts:
            hist[(int(mr), int(mc))][int(col)] += 1
            if rr == r0 - 1:
                side_hits["top"].append((int(mr), int(mc), int(col)))
            if rr == r1 + 1:
                side_hits["bottom"].append((int(mr), int(mc), int(col)))
            if cc == c0 - 1:
                side_hits["left"].append((int(mr), int(mc), int(col)))
            if cc == c1 + 1:
                side_hits["right"].append((int(mr), int(mc), int(col)))

        coords = list(hist.keys())
        tb_sides = [side for side in ("top", "bottom") if side_hits[side]]
        lr_sides = [side for side in ("left", "right") if side_hits[side]]
        is_l = len(tb_sides) == 1 and len(lr_sides) == 1 and cid in comp_boxes

        if is_l:
            tb_side = tb_sides[0]
            lr_side = lr_sides[0]
            p_tb = _dominant_coord(side_hits[tb_side])
            p_lr = _dominant_coord(side_hits[lr_side])
            min_r, min_c, max_r, max_c = comp_boxes[cid]
            outer_r = min_r if tb_side == "top" else max_r
            outer_c = min_c if lr_side == "left" else max_c
            corner = map_to_out(outer_r, outer_c)
            path = _build_corner_path(p_tb, p_lr, corner)

            if len(cols) == 1:
                for rr, cc in path:
                    out[int(rr), int(cc)] = int(cols[0])
                continue

            if len(cols) == 2:
                color_a = int(g[outer_r, outer_c])
                assert color_a in cols
                color_b = _other_color((int(cols[0]), int(cols[1])), color_a)
                start_color = _start_color_from_corner((color_a, color_b), int(corner[0]), out_h)
                _paint_alternating_linear(out, path, start_color, _other_color((color_a, color_b), start_color))
                continue

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
            path_a = _build_manhattan(p1, p2, True)
            path_b = _build_manhattan(p1, p2, False)
            chosen = path_b if (_inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w)) else path_a
            for rr, cc in chosen:
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
            path_a = _build_manhattan(a, b, True)
            path_b = _build_manhattan(a, b, False)
            path = path_b if (_inside_corner(a, b, False, out_h, out_w) and not _inside_corner(a, b, True, out_h, out_w)) else path_a

            chosen = None
            for path_pts in (path, list(reversed(path))):
                for start_color in (ca, cb):
                    end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else (cb if start_color == ca else ca)
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
            _paint_alternating_linear(out, path_pts, start_color, cb if start_color == ca else ca)
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        tops = [p for p in pts if p[0] == 0]
        bots = [p for p in pts if p[0] == out_h - 1]
        if tops and bots:
            for cc in sorted({c for _r, c in pts}):
                rs = [r for r, c in pts if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == out_w - 1]
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
        path_a = _build_manhattan(p1, p2, True)
        path_b = _build_manhattan(p1, p2, False)
        chosen = path_b if (_inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w)) else path_a
        for rr, cc in chosen:
            out[int(rr), int(cc)] = int(col)

    return out


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
_DUAL_COLOR_PROB = 0.3
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 500


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    grid_h, grid_w = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < grid_h and 0 <= c < grid_w):
            return False
        if grid[r, c] != bg:
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < grid_h and 0 <= nc < grid_w and comp_mask[nr, nc]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
    *,
    anchor: tuple[int, int] | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors, anchor=anchor)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _l_specs(kind: str, h: int, w: int, r0: int, c0: int, r1: int, c1: int):
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r = random.choice(_choose_half_rows(h, is_top))
    if is_left:
        c = random.randint(1, max(1, w // 2))
    else:
        c = random.randint(max(1, w // 2), w - 2)

    lo, hi = _OUTER_OFFSET_RANGE
    outer_r = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    outer_c = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))

    ring_tb = (r0 - 1 if is_top else r1 + 1, c0 + 1 + c)
    ring_lr = (r0 + 1 + r, c0 - 1 if is_left else c1 + 1)

    corner = (0 if is_top else h - 1, 0 if is_left else w - 1)
    inner = _join(
        _hpath(corner[0], c, corner[1]),
        _vpath(corner[0], r, corner[1]),
    )
    outer = _join(
        _vpath(ring_tb[0], outer_r, ring_tb[1]),
        _hpath(outer_r, ring_tb[1], outer_c),
        _vpath(outer_r, ring_lr[0], outer_c),
        _hpath(ring_lr[0], outer_c, ring_lr[1]),
    )
    return inner, outer, corner, (outer_r, outer_c)


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    grid_h, grid_w = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r, c = random.randint(1, grid_h - 2), random.randint(1, grid_w - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < grid_h - 1 and 1 <= nc < grid_w - 1 and (nr, nc) not in used
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
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    h = random.randint(*_INTERIOR_H_RANGE)
    w = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], h + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], w + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - h - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - w - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + h + 1, c0 + w + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((h, w), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((h, w), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    top_dual = random.choice(["tl", "tr"])
    bottom_dual = random.choice(["bl", "br"])
    forced_duals = [top_dual, bottom_dual]
    remaining_corners = [kind for kind in ["tl", "tr", "bl", "br"] if kind not in forced_duals]
    corners = forced_duals + random.sample(remaining_corners, k=n_components - 2)
    random.shuffle(corners)

    color_i = 0
    for kind in corners:
        inner, outer, corner, outer_corner = _l_specs(kind, h, w, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None

        need_dual = kind in forced_duals
        use_dual = need_dual or (random.random() < _DUAL_COLOR_PROB)
        if use_dual:
            if color_i >= len(palette) - 1:
                return None
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
            start_color = _start_color_from_corner(colors, int(corner[0]), h)
            if not _place_component(grid, comp_mask, outer, colors, bg, anchor=outer_corner):
                return None
            _paint_alternating_linear(expected, inner, start_color, _other_color(colors, start_color))
        else:
            if color_i >= len(palette):
                return None
            colors = (int(palette[color_i]),)
            color_i += 1
            if not _place_component(grid, comp_mask, outer, colors, bg):
                return None
            _paint_path(expected, inner, colors)

        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), int(palette[color_i]))
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
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
