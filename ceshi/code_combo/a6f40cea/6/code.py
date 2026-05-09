# Combo variant of a6f40cea (index 6)
# combo_spec: D3/0+D4/3
# Name: 箭头射线填充

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# -----------------------------------------------------------------------------
# Design analysis for combo D3/0 + D4/3
#
# a) Concrete behavior and trace
#    Original task:
#    - A useful outer component was an open rectangular frame with two ring touches.
#    - Those two touches mapped to two interior boundary points.
#    - The output drew a Manhattan L or straight segment between those endpoints.
#    Fused task:
#    - The useful outer component is now a bent arrow, not an open rectangle.
#    - It has exactly one ring-touching tip. The tip marks the interior startpoint.
#    - The arrow's pointing side is read from the tip orientation:
#         top-ring tip  -> arrow points down
#         bottom-ring   -> arrow points up
#         left-ring     -> arrow points right
#         right-ring    -> arrow points left
#    - Instead of connecting to a second decoded endpoint, the output fills from
#      that startpoint all the way to the opposite side of the cropped interior.
#    Small trace:
#    - Suppose the frame interior is 6x9 and an arrow tip touches the top ring at
#      column c=3. In the original D3-only variant, the tail alignment would pick
#      an adjacent side and yield one L path such as (0,3)->(2,0).
#    - In this combo, the same tip maps to interior start (0,3), but the arrow is
#      read only for direction. Because the tip is on the top ring, the path runs
#      downward through the whole column: (0,3),(1,3),...,(5,3).
#    - If the arrow uses two colors, the tip color is still the first color and
#      the alternation continues until the opposite side is reached.
#
# b) Complexity calibration
#    A naive version could let arbitrary horizontal and vertical full-span rays
#    coexist, but then every row/column pair would cross and hide the rule behind
#    overwrite order. To keep the variant legible, each generated grid chooses one
#    ray family only:
#    - vertical family: top/bottom arrows create full columns
#    - horizontal family: left/right arrows create full rows
#    This preserves visible repetition inside one sample while still making the
#    change from "corner L completion" to "opposite-side fill" obvious.
#
# c) Hidden assumptions from the original solver that no longer hold
#    - "Each component contributes at least two ring touches" is false. Arrows
#      again have exactly one ring touch.
#    - "A second endpoint must be recovered from geometry" is false. The second
#      endpoint is now implicit: it is always the far boundary in the arrow's
#      inward direction.
#    - "Elbow choice matters" is false. The combo always outputs a straight ray.
#    - "Monochrome multi-touch deferred aggregation" is irrelevant because useful
#      components are one-arrow-per-ray with a single startpoint.
#    New assumptions introduced here:
#    - A useful component must have exactly one ring hit and a valid pointed tip.
#    - Alternating phase is anchored by the visible tip color.
#    - To avoid ambiguous full-span crossings, the generator must keep all rays in
#      one orientation family per example.
#
# d) Placement legality
#    The generator remains output-first. It first decides which rows or columns of
#    the cropped interior will be painted, then constructs corner-local bent arrows
#    outside the frame whose tips project to those starts. Each arrow:
#    - stays fully on background outside the frame
#    - is 4-connected
#    - touches the ring in exactly one cell
#    - is not 4-adjacent to another component
#    - uses a corner-local outer bend so different arrows stay separated
#    Small distractor blobs are kept away from the frame ring exactly as before.
#
# e) Visual observability
#    A human sees two simultaneous differences from the source task:
#    - the external cues are arrows rather than open rectangular frames
#    - the interior result is a full row/column to the far side, not a local L
#    The repeated presence of both top and bottom arrows (or left and right arrows)
#    in the same sample makes direction visible; the repeated full-span stripes
#    make the new stop rule visible.
#
# f) Visual cleanliness
#    Using only one orientation family per grid prevents dense crosshatching inside
#    the frame. The outer arrows remain parked in separate corner bands around the
#    frame, so the frame border, startpoints, and arrowheads stay readable.
#
# g) Change inferability
#    Color behavior remains structural rather than numeric. Monochrome arrows yield
#    solid rays. Two-color arrows alternate starting from the tip color, so the
#    only color-order cue needed by the solver is directly visible on the arrow.
#    This also makes direction matter: a top arrow and a bottom arrow on different
#    columns can use the same two colors but produce opposite phase along the full
#    column because their starts are on opposite ends.
#
# h) Non-degeneracy target
#    These inputs are not solvable by the original solver family:
#    - original a6 expects two ring touches and therefore misses the arrows
#    - D3-only would decode the same bent arrow into an adjacent-side L endpoint,
#      while this combo ignores that tail endpoint and fills to the far side
#    The generator therefore always builds bent arrows whose adjacent-side tail
#    would disagree with the combo's full-span output.
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


def _build_fill_path(
    start: tuple[int, int],
    start_side: str,
    outH: int,
    outW: int,
) -> list[tuple[int, int]]:
    sr, sc = start
    if start_side == "top":
        return [(r, sc) for r in range(int(sr), int(outH))]
    if start_side == "bottom":
        return [(r, sc) for r in range(int(sr), -1, -1)]
    if start_side == "left":
        return [(sr, c) for c in range(int(sc), int(outW))]
    if start_side == "right":
        return [(sr, c) for c in range(int(sc), -1, -1)]
    raise ValueError(start_side)


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
    Rebuild full-span interior rays from external bent arrows around a framed rectangle.

    Input structure:
    - A single 1px closed rectangle sits on a uniform background and has a uniformly filled interior.
    - Useful outer components are bent arrows with exactly one ring-touching tip on the frame.
    - The tip location marks an interior boundary startpoint, and the tip orientation gives one of
      the four inward directions: down from top, up from bottom, right from left, left from right.

    Rule:
    - Crop to the frame interior and initialize it with the interior fill color.
    - For each valid arrow, map its tip to the corresponding interior boundary cell.
    - Starting from that cell, extend a straight row or column in the arrow's direction until the
      opposite side of the interior is reached.
    - Monochrome arrows paint a solid ray; two-color arrows alternate along the ray beginning with
      the visible color at the tip.

    Key insight:
    - The old second endpoint has disappeared. The only explicit boundary cue is the tip, and the
      arrow direction determines both the ray orientation and which far side acts as the stop.
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
        path = _build_fill_path(start, start_side, outH, outW)
        cols = sorted(set(map(int, g[clab == int(cid)].flatten())))
        assert 1 <= len(cols) <= 2

        if len(cols) == 1:
            rs, cs = zip(*path)
            out[rs, cs] = int(cols[0])
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
_GEN_RETRIES = 500


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
    else:
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


def _sample_axis(kind: str, H: int, W: int) -> tuple[int, int]:
    max_r = H - 2
    max_c = W - 2
    top_hi = max(1, max_r // 2)
    bot_lo = min(max_r, top_hi + 1)
    left_hi = max(1, max_c // 2)
    right_lo = min(max_c, left_hi + 1)

    if kind[0] == "t":
        r = random.randint(1, top_hi)
    else:
        r = random.randint(bot_lo, max_r)
    if kind[1] == "l":
        c = random.randint(1, left_hi)
    else:
        c = random.randint(right_lo, max_c)
    return int(r), int(c)


def _fill_specs(start_side: str, r: int, c: int, H: int, W: int) -> list[tuple[int, int]]:
    if start_side == "top":
        return [(rr, c) for rr in range(H)]
    if start_side == "bottom":
        return [(rr, c) for rr in range(H - 1, -1, -1)]
    if start_side == "left":
        return [(r, cc) for cc in range(W)]
    if start_side == "right":
        return [(r, cc) for cc in range(W - 1, -1, -1)]
    raise ValueError(start_side)


def _arrow_specs(kind: str, start_side: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    r, c = _sample_axis(kind, H, W)
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
        tip = top_tip
        backbone = _join(
            _vpath(tip[0], R_top, tip[1]),
            _hpath(R_top, tip[1], C_left),
            _vpath(R_top, left_tail[0], C_left),
            _hpath(left_tail[0], C_left, left_tail[1]),
        )
    elif kind == "tl" and start_side == "left":
        tip = left_tip
        backbone = _join(
            _hpath(tip[0], tip[1], C_left),
            _vpath(tip[0], R_top, C_left),
            _hpath(R_top, C_left, top_tail[1]),
            _vpath(R_top, top_tail[0], top_tail[1]),
        )
    elif kind == "tr" and start_side == "top":
        tip = top_tip
        backbone = _join(
            _vpath(tip[0], R_top, tip[1]),
            _hpath(R_top, tip[1], C_right),
            _vpath(R_top, right_tail[0], C_right),
            _hpath(right_tail[0], C_right, right_tail[1]),
        )
    elif kind == "tr" and start_side == "right":
        tip = right_tip
        backbone = _join(
            _hpath(tip[0], tip[1], C_right),
            _vpath(tip[0], R_top, C_right),
            _hpath(R_top, C_right, top_tail[1]),
            _vpath(R_top, top_tail[0], top_tail[1]),
        )
    elif kind == "bl" and start_side == "bottom":
        tip = bot_tip
        backbone = _join(
            _vpath(tip[0], R_bot, tip[1]),
            _hpath(R_bot, tip[1], C_left),
            _vpath(R_bot, left_tail[0], C_left),
            _hpath(left_tail[0], C_left, left_tail[1]),
        )
    elif kind == "bl" and start_side == "left":
        tip = left_tip
        backbone = _join(
            _hpath(tip[0], tip[1], C_left),
            _vpath(tip[0], R_bot, C_left),
            _hpath(R_bot, C_left, bot_tail[1]),
            _vpath(R_bot, bot_tail[0], bot_tail[1]),
        )
    elif kind == "br" and start_side == "bottom":
        tip = bot_tip
        backbone = _join(
            _vpath(tip[0], R_bot, tip[1]),
            _hpath(R_bot, tip[1], C_right),
            _vpath(R_bot, right_tail[0], C_right),
            _hpath(right_tail[0], C_right, right_tail[1]),
        )
    elif kind == "br" and start_side == "right":
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
    inner = _fill_specs(start_side, r, c, H, W)
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


def _pick_component_slots(n_components: int):
    family = random.choice(["vertical", "horizontal"])
    if family == "vertical":
        top_choice = random.choice([("tl", "top"), ("tr", "top")])
        bottom_choice = random.choice([("bl", "bottom"), ("br", "bottom")])
        slots = [top_choice, bottom_choice]
        remaining = [slot for slot in [("tl", "top"), ("tr", "top"), ("bl", "bottom"), ("br", "bottom")] if slot not in slots]
    else:
        left_choice = random.choice([("tl", "left"), ("bl", "left")])
        right_choice = random.choice([("tr", "right"), ("br", "right")])
        slots = [left_choice, right_choice]
        remaining = [slot for slot in [("tl", "left"), ("bl", "left"), ("tr", "right"), ("br", "right")] if slot not in slots]

    if n_components == 3:
        slots.append(random.choice(remaining))
    random.shuffle(slots)
    return family, slots


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
    dual_count = 0

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    _family, slots = _pick_component_slots(n_components)

    for k_i, (kind, start_side) in enumerate(slots):
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
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
