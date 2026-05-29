# Variant of a6f40cea
# Dimension: D4 — 终点/停止条件 (dst)
# Change: 路径连接同一开放矩形框的两个触碰点（即补全开放框的缺失部分）... → 路径延伸固定长度

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# Design analysis for the D4 variant
# ----------------------------------
# concretization:
# - We keep the original large closed frame, the same outside "open rectangle" family,
#   and the same crop-to-interior output.
# - The original hidden object was the full missing L-path between the two ring touch
#   points of one outside component. The variant changes only the stopping rule:
#   the interior path now starts at the touch point on the top/bottom side of the
#   frame, follows the same implied hidden L route, but stops after a counted number
#   of cells instead of reaching the second touch point.
# - The counted quantity is made visible on the same outside component: the straight
#   exposed leg aligned with that start touch point. For top-start components this is
#   the vertical run above the frame; for bottom-start components it is the vertical
#   run below the frame. Its pixel count is copied as the interior prefix length.
#
# behavior trace on one concrete corner:
# - Original tl case: ring_a is on the top side, ring_b is on the left side. The
#   solver mapped both touch points into interior coordinates and painted the full
#   route top -> elbow -> left.
# - Variant tl case: the same two touch points still define the hidden route, but the
#   top touch point is now the start. If the visible exterior vertical leg above that
#   touch point has length 3, only the first 3 cells of the hidden route are painted:
#   row-0 touch, next cell downward, next cell downward. The path stops there even
#   though the second touch point is still visible outside and still helps identify
#   the elbow direction.
# - Difference: the second touch point no longer fixes the destination directly; it
#   only fixes the route geometry. The destination is determined by counting.
#
# complexity calibration:
# - The counted leg length is restricted to a small range (2..5 after placement
#   filtering), so the task shows repeated evidence across examples instead of too
#   many unrelated lengths.
# - We keep 2-3 components and keep mono / two-color alternation to preserve the
#   original visual richness. We intentionally drop the original rare U-pair branch
#   from generation because its same-side geometry gives a weak start-point cue under
#   a prefix rule; restricting generation to corner-L components makes the new D4
#   change readable in 2-3 examples.
#
# implicit-assumption audit of the original main():
# - Original code assumed each useful component should connect its farthest mapped ring
#   pair all the way to the other end. That assumption breaks here: using the whole
#   path would silently regress to the old task.
# - Original two-color handling anchored alternation by both endpoint colors. The new
#   rule only guarantees the start touch color matters, because the path may stop
#   before reaching the second touch point.
# - Original deferred same-row / same-col aggregation existed for U-pairs. Since the
#   generator no longer emits them, the new main() should not rely on that branch.
#
# placement legality:
# - The generator still validates that outside components do not overlap or touch
#   each other 4-neighbor-wise, and that distractors stay away from the frame ring.
# - For each L component we require target_len < full hidden L length, so the variant
#   really stops early and does not collapse back to the original completion rule.
# - We also reserve the full hidden L route in the generator, not just the visible
#   prefix, so two components cannot imply conflicting ring contacts or hidden elbows.
#
# observability / visual cleanliness:
# - The start side is always the horizontal-side touch (top for t*, bottom for b*),
#   so humans can infer a consistent "count the exposed vertical leg, then continue
#   inward" rule without invisible tie-breaks such as color-id ordering.
# - The counted leg is a clean straight run separated from the frame, while the rest
#   of the outside open rectangle still shows the corner orientation. This makes the
#   new stop condition visibly different from the original full-connection behavior.
# - Colors are not remapped by id. Monochrome paths stay solid; two-color paths reuse
#   the component's alternating palette starting from the start touch color.


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
    return [(int(r), int(c)) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append((int(p[0]), int(p[1])))
    return out


def _paint_alt_prefix(
    out: np.ndarray,
    path_pts: list[tuple[int, int]],
    colors: tuple[int, int],
    start_color: int,
) -> None:
    ca, cb = map(int, colors)
    cur = int(start_color)
    for rr, cc in path_pts:
        out[int(rr), int(cc)] = int(cur)
        cur = int(cb) if cur == int(ca) else int(ca)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover partial hidden L-paths inside a framed rectangle by counting an exposed outer leg.

    Input structure:
    - A uniform background, one closed 1px frame, and a uniformly filled frame interior.
    - Outside the frame are 2-3 corner-shaped open rectangles that touch one horizontal frame side and one vertical frame side.
    - The touch on the top or bottom side continues outward as a straight visible leg whose length is countable.

    Rule:
    - Output only the cropped frame interior, initially filled with the frame's fill color.
    - Each outside component still determines a hidden Manhattan L route between its two mapped touch points.
    - Start from the touch point on the top/bottom side and follow that route inward.
    - Stop after N cells, where N is the pixel count of the visible straight leg extending away from that same touch point.
    - Monochrome components paint solid prefixes; two-color components paint alternating prefixes starting from the start-touch color.

    Key insight: the second touch point still reveals the route's bend, but the endpoint now comes from counting the exposed leg rather than completing the whole missing side.
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
        mr = max(0, min(outH - 1, int(rr - r_int0)))
        mc = max(0, min(outW - 1, int(cc - c_int0)))
        return int(mr), int(mc)

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
        coords = np.argwhere(clab == int(cid))
        if coords.size == 0:
            continue

        ring_hits = [(int(rr), int(cc)) for rr, cc in coords if (int(rr), int(cc)) in ring]
        if len(ring_hits) < 2:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc in ring_hits:
            mr, mc = map_to_out(int(rr), int(cc))
            if not (0 <= mr < outH and 0 <= mc < outW):
                continue
            hist[(int(mr), int(mc))][int(g[int(rr), int(cc)])] += 1
        if len(hist) != 2:
            continue

        mapped = list(hist.keys())
        start_candidates = [p for p in mapped if p[0] in (0, outH - 1)]
        end_candidates = [p for p in mapped if p[1] in (0, outW - 1)]
        if len(start_candidates) != 1 or len(end_candidates) != 1:
            continue
        start = start_candidates[0]
        end = end_candidates[0]
        if start == end:
            continue

        full_path = _join(_vpath(start[0], end[0], start[1]), _hpath(end[0], start[1], end[1]))
        if not full_path:
            continue

        if start[0] == 0:
            leg_r, leg_c = int(r0 - 1), int(c_int0 + start[1])
            dr = -1
        else:
            leg_r, leg_c = int(r1 + 1), int(c_int0 + start[1])
            dr = 1

        visible_len = 0
        rr = int(leg_r)
        while 0 <= rr < H and int(clab[int(rr), int(leg_c)]) == int(cid):
            visible_len += 1
            rr += int(dr)

        if visible_len <= 0:
            continue
        visible_len = min(int(visible_len), int(len(full_path)))
        path = full_path[:visible_len]

        cols = sorted(set(map(int, g[clab == int(cid)].flatten())))
        if len(cols) == 1:
            col = int(cols[0])
            for rr, cc in path:
                out[int(rr), int(cc)] = int(col)
            continue

        if len(cols) == 2:
            start_color = int(hist[start].most_common(1)[0][0])
            _paint_alt_prefix(out, path, (int(cols[0]), int(cols[1])), int(start_color))
            continue

    return out


# --------------- generator (D4 variant) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (5, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 5
_FRAME_MARGIN_FAR = 5
_TARGET_LEN_RANGE = (2, 5)
_SIDE_OFFSET_RANGE = (2, 4)
_N_COMPONENTS_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.45
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
        grid[rs, cs] = colors[0]
    else:
        a, b = colors
        vals = [int(a) if i % 2 == 0 else int(b) for i in range(len(path))]
        grid[rs, cs] = vals


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    pts = set((int(r), int(c)) for r, c in coords)
    for r, c in pts:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) not in pts and 0 <= nr < H and 0 <= nc < W and comp_mask[int(nr), int(nc)]:
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
        coords = [(int(r), int(c))]
        used = {(int(r), int(c))}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (int(nr), int(nc))
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (int(nr), int(nc)) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add((int(p[0]), int(p[1])))
            coords.append((int(p[0]), int(p[1])))
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), int(bg)):
            return


def _l_specs(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    target_len: int,
    side_offset: int,
):
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r = random.randint(1, max(1, H // 2)) if is_top else random.randint(max(1, H // 2), H - 2)
    c = random.randint(1, max(1, W // 2)) if is_left else random.randint(max(1, W // 2), W - 2)

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + c)
    ring_b = (r0 + 1 + r, c0 - 1 if is_left else c1 + 1)

    edge_r = 0 if is_top else H - 1
    edge_c = 0 if is_left else W - 1
    inner_full = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))
    if len(inner_full) <= int(target_len):
        return None

    R = (r0 - int(target_len)) if is_top else (r1 + int(target_len))
    C = (c0 - int(side_offset)) if is_left else (c1 + int(side_offset))
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    inner_prefix = inner_full[: int(target_len)]
    return inner_full, inner_prefix, outer


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

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    expected = np.full((H, W), int(fill), dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    corners = random.sample(["tl", "tr", "bl", "br"], k=n_components)
    color_i = 0

    for kind in corners:
        if color_i >= len(palette) - 1:
            return None

        max_target = _TARGET_LEN_RANGE[1]
        if kind[0] == "t":
            max_target = min(int(max_target), int(r0))
        else:
            max_target = min(int(max_target), int(rows - 1 - r1))
        if max_target < _TARGET_LEN_RANGE[0]:
            return None
        target_len = random.randint(_TARGET_LEN_RANGE[0], max_target)

        max_side = _SIDE_OFFSET_RANGE[1]
        if kind[1] == "l":
            max_side = min(int(max_side), int(c0))
        else:
            max_side = min(int(max_side), int(cols - 1 - c1))
        if max_side < _SIDE_OFFSET_RANGE[0]:
            return None
        side_offset = random.randint(_SIDE_OFFSET_RANGE[0], max_side)

        spec = _l_specs(kind, H, W, r0, c0, r1, c1, int(target_len), int(side_offset))
        if spec is None:
            return None
        inner_full, inner_prefix, outer = spec

        irs, ics = zip(*inner_full)
        if used_inner[irs, ics].any():
            return None

        if random.random() < _DUAL_COLOR_PROB:
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        else:
            colors = (int(palette[color_i]),)
            color_i += 1

        if not _place_component(grid, comp_mask, outer, colors, int(bg)):
            return None
        _paint_path(expected, inner_prefix, colors)
        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, int(bg), (r0, c0, r1, c1), int(palette[color_i]))
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
    raise RuntimeError("failed to generate a valid a6f40cea D4 input")
