# Variant of a6f40cea
# Dimension: D3 — 源对象类型 (src)
# Change: 外部开放矩形框架（L形：2条完整边+2条截断stub；或U形：3条完整边），通过触碰闭合框边来标记端点位置... → 箭头形状指示方向

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# -----------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# a) Concrete behavior and trace
#    Original source objects were open rectangular frames that touched the closed
#    frame in two places. Those two ring touches directly encoded the two output
#    endpoints. In this variant, each meaningful source object becomes a single
#    bent arrow:
#    - exactly one ring-touching tip marks the output startpoint
#    - the tip side still tells which boundary point is used
#    - the arrow body bends around the same outer corner region as before
#    - the free tail stops one cell short of another frame side, and its row/col
#      alignment encodes the output endpoint on that adjacent side
#    Example: top-left arrow with tip on the top ring at column c and free tail
#    ending at column c0-2 on row r0+1+r. The solver maps the tip to interior
#    start (0, c) and maps the tail alignment to interior end (r, 0), then draws
#    the L path from the start downward first and then leftward.
#
# b) Complexity calibration
#    The variant keeps the closed frame, fill, cropping rule, color palette, and
#    Manhattan path idea unchanged. The only structural change is the external
#    object type and how endpoints are decoded. To keep the change visible, U-pair
#    sources are removed: they depend on dual frame touches and would dilute the
#    arrow reading rule. Visual repetition is preserved by generating 2-3 arrows
#    per grid and by keeping corner-local placement categories tl/tr/bl/br.
#
# c) Hidden assumptions in the original solver that no longer hold
#    - Original code assumed each path component contributes >=2 ring-touch points.
#      That fails here because each arrow has exactly one ring touch.
#    - Original two-color phase was anchored by the colors seen at both ring ends.
#      That also fails here because there is only one ring end. The new solver
#      anchors alternation from the tip color instead.
#    - Original deferred monochrome logic handled same-row/same-col multi-touch
#      aggregates. That becomes irrelevant because arrows encode one L each.
#
# d) Placement legality
#    The generator uses output-first construction. For each arrow it first chooses
#    an interior elbow (r, c), then derives:
#    - the tip on one frame side
#    - the free tail endpoint one cell short of the adjacent frame side
#    - an outer corner offset 3-4 cells away from the frame
#    The offset is intentionally larger than the tip-head offset (2), so the tail
#    decoding lines r0-2/r1+2/c0-2/c1+2 stay unique and do not collide with the
#    outer corner row/col. Components must be in-bounds, on bg, non-overlapping,
#    and not 4-adjacent to earlier components.
#
# e) Visual observability
#    Across examples, humans can see that the old open-rectangle cue is gone:
#    each useful object now has a pointed head on the frame ring and a bent tail
#    ending near a neighboring side. The tip position visibly marks where the path
#    begins; the tail alignment visibly marks where it should end; the output is
#    still the cropped frame interior. Two-color arrows are common so start-order
#    affects the output coloring and is not a purely hidden convention.
#
# f) Visual cleanliness
#    Each arrow occupies a dedicated outer quadrant around one frame corner, and
#    the interior elbows are also corner-local. This keeps the input legible:
#    arrows are separated from each other, the frame remains readable, and small
#    distractors stay away from the ring.
#
# g) Color inference
#    No invisible color-id sorting is used to define behavior. For monochrome
#    arrows the path is solid. For two-color arrows the solver uses only a visible
#    cue: the tip cell color is the first color of the interior alternation. The
#    generator reuses that same tip-anchored order when painting the expected
#    output.
#
# h) Non-degeneracy target
#    The old solver should not solve these inputs because its key primitive was
#    "two ring touches per component". Here each meaningful component has one ring
#    touch and one free tail alignment, so the old decoding path is broken by
#    construction.
# -----------------------------------------------------------------------------


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with uniform interior fill.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such
    rectangle, else None.
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


def _component_side_of_ring_cell(rr: int, cc: int, r0: int, c0: int, r1: int, c1: int) -> str | None:
    if rr == r0 - 1 and c0 <= cc <= c1:
        return "top"
    if rr == r1 + 1 and c0 <= cc <= c1:
        return "bottom"
    if cc == c0 - 1 and r0 <= rr <= r1:
        return "left"
    if cc == c1 + 1 and r0 <= rr <= r1:
        return "right"
    return None


def _tip_head_cells(tip: tuple[int, int], side: str) -> list[tuple[int, int]]:
    tr, tc = tip
    if side == "top":
        return [(tr - 1, tc - 1), (tr - 1, tc + 1)]
    if side == "bottom":
        return [(tr + 1, tc - 1), (tr + 1, tc + 1)]
    if side == "left":
        return [(tr - 1, tc - 1), (tr + 1, tc - 1)]
    if side == "right":
        return [(tr - 1, tc + 1), (tr + 1, tc + 1)]
    raise ValueError(side)


def _has_arrow_tip(coords: set[tuple[int, int]], tip: tuple[int, int], side: str) -> bool:
    tr, tc = tip
    if side == "top":
        needed = {(tr - 1, tc), (tr - 1, tc - 1), (tr - 1, tc + 1)}
    elif side == "bottom":
        needed = {(tr + 1, tc), (tr + 1, tc - 1), (tr + 1, tc + 1)}
    elif side == "left":
        needed = {(tr, tc - 1), (tr - 1, tc - 1), (tr + 1, tc - 1)}
    elif side == "right":
        needed = {(tr, tc + 1), (tr - 1, tc + 1), (tr + 1, tc + 1)}
    else:
        return False
    return needed.issubset(coords)


def _map_tip(rr: int, cc: int, side: str, outH: int, outW: int, r0: int, c0: int) -> tuple[int, int]:
    if side == "top":
        return 0, int(cc - (c0 + 1))
    if side == "bottom":
        return outH - 1, int(cc - (c0 + 1))
    if side == "left":
        return int(rr - (r0 + 1)), 0
    if side == "right":
        return int(rr - (r0 + 1)), outW - 1
    raise ValueError(side)


def _decode_tail(
    coords: set[tuple[int, int]],
    start_side: str,
    *,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    outH: int,
    outW: int,
) -> tuple[tuple[int, int], str] | None:
    if start_side in ("top", "bottom"):
        left_rows = sorted(rr for rr, cc in coords if cc == c0 - 2 and (r0 + 1) <= rr <= (r1 - 1))
        right_rows = sorted(rr for rr, cc in coords if cc == c1 + 2 and (r0 + 1) <= rr <= (r1 - 1))
        if bool(left_rows) == bool(right_rows):
            return None
        if left_rows:
            return (int(left_rows[0] - (r0 + 1)), 0), "left"
        return (int(right_rows[0] - (r0 + 1)), outW - 1), "right"

    top_cols = sorted(cc for rr, cc in coords if rr == r0 - 2 and (c0 + 1) <= cc <= (c1 - 1))
    bot_cols = sorted(cc for rr, cc in coords if rr == r1 + 2 and (c0 + 1) <= cc <= (c1 - 1))
    if bool(top_cols) == bool(bot_cols):
        return None
    if top_cols:
        return (0, int(top_cols[0] - (c0 + 1))), "top"
    return (outH - 1, int(bot_cols[0] - (c0 + 1))), "bottom"


def _build_ordered_path(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    start_axis: str,
) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    if sr == er:
        return [(sr, c) for c in _line(sc, ec)]
    if sc == ec:
        return [(r, sc) for r in _line(sr, er)]
    if start_axis == "v":
        return _join(_vpath(sr, er, sc), _hpath(er, sc, ec))
    return _join(_hpath(sr, sc, ec), _vpath(sr, er, ec))


def _paint_alt_path(
    out: np.ndarray,
    path: list[tuple[int, int]],
    start_color: int,
    other_color: int,
) -> None:
    cur = int(start_color)
    for rr, cc in path:
        out[int(rr), int(cc)] = int(cur)
        cur = int(other_color) if cur == int(start_color) else int(start_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct interior corner paths from external bent arrows around a framed rectangle.

    Input structure:
    - A uniform background with one 1px closed rectangle whose interior is uniformly filled.
    - Outside the frame, useful components are bent arrows: one pointed tip touches the frame's
      ring, while the free tail stops one cell short of an adjacent frame side.

    Rule:
    - Output only the rectangle interior, initialized with the interior fill color.
    - Map the arrow tip to the interior boundary startpoint.
    - Read the tail's row/column alignment to recover the endpoint on the adjacent boundary.
    - Draw the corresponding Manhattan L-path inside the frame; monochrome arrows paint solidly,
      and two-color arrows alternate starting from the tip color.

    Key insight:
    - The external source no longer gives two direct frame touches. The tip gives the start and
      the tail alignment gives the end, so solving depends on decoding both the arrow geometry and
      its position relative to the frame.
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

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        if 0 <= r0 - 1 < H:
            ring.add((r0 - 1, cc))
        if 0 <= r1 + 1 < H:
            ring.add((r1 + 1, cc))
    for rr in range(r0, r1 + 1):
        if 0 <= c0 - 1 < W:
            ring.add((rr, c0 - 1))
        if 0 <= c1 + 1 < W:
            ring.add((rr, c1 + 1))

    for cid in range(1, int(cn) + 1):
        coords_arr = np.argwhere(clab == int(cid))
        if coords_arr.size == 0:
            continue
        coords = {(int(r), int(c)) for r, c in coords_arr}
        ring_hits = [(rr, cc) for rr, cc in coords if (rr, cc) in ring]
        if len(ring_hits) != 1:
            continue

        tip = tuple(map(int, ring_hits[0]))
        start_side = _component_side_of_ring_cell(tip[0], tip[1], r0, c0, r1, c1)
        if start_side is None:
            continue
        if not _has_arrow_tip(coords, tip, start_side):
            continue

        start = _map_tip(tip[0], tip[1], start_side, outH, outW, r0, c0)
        decoded = _decode_tail(coords, start_side, r0=r0, c0=c0, r1=r1, c1=c1, outH=outH, outW=outW)
        if decoded is None:
            continue
        end, end_side = decoded
        assert start_side != end_side
        assert (start_side in ("top", "bottom")) != (end_side in ("top", "bottom"))

        path = _build_ordered_path(start, end, start_axis=("v" if start_side in ("top", "bottom") else "h"))
        cols = sorted(set(map(int, g[clab == int(cid)].flatten())))
        assert 1 <= len(cols) <= 2
        if len(cols) == 1:
            out[tuple(np.array(path).T)] = int(cols[0])
            continue

        start_color = int(g[tip[0], tip[1]])
        other = cols[0] if int(cols[1]) == int(start_color) else cols[1]
        _paint_alt_path(out, path, int(start_color), int(other))

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
_OUTER_OFFSET_RANGE = (3, 4)
_N_COMPONENTS_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.65
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
    else:
        a, b = colors
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
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and bool(comp_mask[nr, nc]):
                return False
    return True


def _color_lookup_along_backbone(path: list[tuple[int, int]], colors: tuple[int, ...]) -> dict[tuple[int, int], int]:
    if len(colors) == 1:
        return {p: int(colors[0]) for p in path}
    a, b = map(int, colors)
    return {p: (a if i % 2 == 0 else b) for i, p in enumerate(path)}


def _place_arrow_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    backbone: list[tuple[int, int]],
    head_cells: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
) -> bool:
    coords = list(dict.fromkeys(backbone + head_cells))
    if len(coords) != len(set(coords)):
        return False
    if not _can_place(grid, coords, bg, comp_mask):
        return False

    color_map = _color_lookup_along_backbone(backbone, colors)
    for rr, cc in backbone:
        grid[int(rr), int(cc)] = int(color_map[(rr, cc)])
    shaft = backbone[1] if len(backbone) >= 2 else backbone[0]
    head_color = int(color_map[shaft])
    for rr, cc in head_cells:
        grid[int(rr), int(cc)] = int(head_color)

    rs, cs = zip(*coords)
    comp_mask[rs, cs] = True
    return True


def _corner_rc(kind: str, H: int, W: int) -> tuple[int, int]:
    if kind[0] == "t":
        r = random.randint(1, max(1, H // 2))
    else:
        r = random.randint(max(1, H // 2), H - 2)
    if kind[1] == "l":
        c = random.randint(1, max(1, W // 2))
    else:
        c = random.randint(max(1, W // 2), W - 2)
    return int(r), int(c)


def _arrow_specs(kind: str, start_side: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    r, c = _corner_rc(kind, H, W)
    lo, hi = _OUTER_OFFSET_RANGE
    R_top = r0 - random.randint(lo, hi)
    R_bot = r1 + random.randint(lo, hi)
    C_left = c0 - random.randint(lo, hi)
    C_right = c1 + random.randint(lo, hi)

    top_tip = (r0 - 1, c0 + 1 + c)
    bot_tip = (r1 + 1, c0 + 1 + c)
    left_tip = (r0 + 1 + r, c0 - 1)
    right_tip = (r0 + 1 + r, c1 + 1)

    top_tail = (r0 - 2, c0 + 1 + c)
    bot_tail = (r1 + 2, c0 + 1 + c)
    left_tail = (r0 + 1 + r, c0 - 2)
    right_tail = (r0 + 1 + r, c1 + 2)

    if kind == "tl" and start_side == "top":
        start = (0, c)
        end = (r, 0)
        tip = top_tip
        backbone = _join(
            _vpath(tip[0], R_top, tip[1]),
            _hpath(R_top, tip[1], C_left),
            _vpath(R_top, left_tail[0], C_left),
            _hpath(left_tail[0], C_left, left_tail[1]),
        )
    elif kind == "tl" and start_side == "left":
        start = (r, 0)
        end = (0, c)
        tip = left_tip
        backbone = _join(
            _hpath(tip[0], tip[1], C_left),
            _vpath(tip[0], R_top, C_left),
            _hpath(R_top, C_left, top_tail[1]),
            _vpath(R_top, top_tail[0], top_tail[1]),
        )
    elif kind == "tr" and start_side == "top":
        start = (0, c)
        end = (r, W - 1)
        tip = top_tip
        backbone = _join(
            _vpath(tip[0], R_top, tip[1]),
            _hpath(R_top, tip[1], C_right),
            _vpath(R_top, right_tail[0], C_right),
            _hpath(right_tail[0], C_right, right_tail[1]),
        )
    elif kind == "tr" and start_side == "right":
        start = (r, W - 1)
        end = (0, c)
        tip = right_tip
        backbone = _join(
            _hpath(tip[0], tip[1], C_right),
            _vpath(tip[0], R_top, C_right),
            _hpath(R_top, C_right, top_tail[1]),
            _vpath(R_top, top_tail[0], top_tail[1]),
        )
    elif kind == "bl" and start_side == "bottom":
        start = (H - 1, c)
        end = (r, 0)
        tip = bot_tip
        backbone = _join(
            _vpath(tip[0], R_bot, tip[1]),
            _hpath(R_bot, tip[1], C_left),
            _vpath(R_bot, left_tail[0], C_left),
            _hpath(left_tail[0], C_left, left_tail[1]),
        )
    elif kind == "bl" and start_side == "left":
        start = (r, 0)
        end = (H - 1, c)
        tip = left_tip
        backbone = _join(
            _hpath(tip[0], tip[1], C_left),
            _vpath(tip[0], R_bot, C_left),
            _hpath(R_bot, C_left, bot_tail[1]),
            _vpath(R_bot, bot_tail[0], bot_tail[1]),
        )
    elif kind == "br" and start_side == "bottom":
        start = (H - 1, c)
        end = (r, W - 1)
        tip = bot_tip
        backbone = _join(
            _vpath(tip[0], R_bot, tip[1]),
            _hpath(R_bot, tip[1], C_right),
            _vpath(R_bot, right_tail[0], C_right),
            _hpath(right_tail[0], C_right, right_tail[1]),
        )
    elif kind == "br" and start_side == "right":
        start = (r, W - 1)
        end = (H - 1, c)
        tip = right_tip
        backbone = _join(
            _hpath(tip[0], tip[1], C_right),
            _vpath(tip[0], R_bot, C_right),
            _hpath(R_bot, C_right, bot_tail[1]),
            _vpath(R_bot, bot_tail[0], bot_tail[1]),
        )
    else:
        raise ValueError((kind, start_side))

    head = _tip_head_cells(tip, start_side)
    inner = _build_ordered_path(start, end, start_axis=("v" if start_side in ("top", "bottom") else "h"))
    return inner, backbone, head


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
        r, c = random.randint(1, H - 2), random.randint(1, W - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [(nr, nc) for nr, nc in neighbors4(*base) if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (nr, nc) not in used]
            if not opts:
                break
            p = random.choice(opts)
            used.add(p)
            coords.append(p)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _can_place(grid, coords, bg, comp_mask):
            rs, cs = zip(*coords)
            grid[rs, cs] = int(color)
            comp_mask[rs, cs] = True
            return


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
    used_inner = np.zeros((H, W), dtype=bool)

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    color_i = 0

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    kinds = random.sample(["tl", "tr", "bl", "br"], k=n_components)

    dual_count = 0
    for k_i, kind in enumerate(kinds):
        start_side = random.choice({"tl": ["top", "left"], "tr": ["top", "right"], "bl": ["bottom", "left"], "br": ["bottom", "right"]}[kind])
        inner, backbone, head = _arrow_specs(kind, start_side, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        if color_i >= len(palette):
            return None

        need_dual = (k_i == 0 and dual_count == 0) or (random.random() < _DUAL_COLOR_PROB)
        if need_dual and color_i + 1 < len(palette):
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
            dual_count += 1
        else:
            colors = (palette[color_i],)
            color_i += 1

        if not _place_arrow_component(grid, comp_mask, backbone, head, colors, bg):
            return None
        _paint_path(expected, inner, colors)
        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), palette[color_i])
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
    raise RuntimeError("failed to generate a valid a6f40cea D3 input")
