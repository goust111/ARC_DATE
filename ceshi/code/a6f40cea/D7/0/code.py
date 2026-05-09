# Variant of a6f40cea
# Dimension: D7 — 组件数量与配置 (meta)
# Change: 2-3个开放框架组件（L形或U形对），每个独立处理... → 固定单组件

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# Design analysis for the D7 variant
# a) Concrete behavior:
#    The closed frame remains unchanged. Outside it there is now exactly one
#    ring-touching open rectangle, always an L-shaped component whose two free
#    ends touch two adjacent sides of the frame. The output is only the frame
#    interior, filled with the interior color plus one Manhattan L-path. In the
#    original task, multiple ring-touching components could each write an
#    independent path, and a U-pair could create two full-span stripes. Here the
#    changed step is the component-processing stage: instead of iterating over
#    2-3 ring-touching components and aggregating deferred monochrome stripes, we
#    identify the single ring-touching component and reconstruct one path from
#    its two projected touch points.
#    Example trace: original input might contain a top-left L plus a bottom-right
#    L, so the solver paints two disjoint paths. In this variant only the
#    top-left L exists, so the same projection logic yields only one interior
#    path, making the one-to-one correspondence visually obvious.
# b) Complexity calibration:
#    The task is simpler because there is one component, but not trivial: the
#    learner still has to infer the framed crop, endpoint projection, Manhattan
#    elbow choice, and the mono vs alternating color rule. Keeping four corner
#    placements and both color modes preserves visible variation without bringing
#    back multi-component ambiguity.
# c) Hidden-assumption audit:
#    The original main() implicitly tolerated many component IDs, a possible
#    U-pair, and a deferred same-color aggregation step. Those branches depended
#    on the old "2-3 components" regime. In this variant they are no longer
#    valid assumptions, so main() now asserts that there is exactly one
#    ring-touching component and directly paints from that component alone.
#    Alternating-color phase anchoring still relies on ring endpoint colors, and
#    the generator guarantees those remain observable.
# d) Placement legality:
#    The generator still constructs the output path first, then builds one outer
#    L-shaped open rectangle whose cells stay on background, remain in-bounds,
#    and do not overlap the frame. Optional distractors never touch the frame
#    ring and are rejected if they collide with existing components.
# e) Visual observability:
#    Across 2-4 examples, the change is visible because every input has only one
#    salient open frame around the rectangle, and every output has exactly one
#    non-fill path. No hidden ordering by color ID is used; corner placement is
#    determined geometrically and color alternation is read directly from the
#    visible endpoints.
# f) Visual cleanliness:
#    Removing extra ring-touching components eliminates overlapping path logic
#    and makes the crop easier to parse. Distractors remain small and separated
#    from the frame so they do not compete with the sole teaching signal.
# g) Color inference:
#    The color rule is unchanged and remains local: one-color components paint a
#    solid path, two-color components alternate along the path with the endpoint
#    colors fixing the phase. The palette excludes bg/frame/fill to keep the
#    signal legible.
# 9) New main() assumptions checked against generate_input():
#    - Exactly one ring-touching component exists: guaranteed by generating one
#      L-component and placing distractors away from the frame ring.
#    - That component has exactly one or two colors: guaranteed by choosing
#      either a singleton color tuple or a 2-color tuple for the component.
#    - Its two mapped ring touch points lie on adjacent interior sides so a
#      proper L-path exists with an interior elbow: guaranteed by _l_specs(),
#      which chooses the elbow strictly away from the boundary corner.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a single interior path from one open rectangle outside a framed box.

    Input structure:
    - A background grid with one 1px rectangular frame and a uniformly filled interior.
    - Exactly one colored open-rectangle component outside the frame touching two adjacent frame sides.

    Rule:
    - Output only the interior of the closed frame, initialized with the fill color.
    - Project the outside component's ring-touch points onto the interior boundary and connect them with a Manhattan L-path.
    - If the outside component is monochrome, paint the path solid; if it uses two colors, alternate them using the visible endpoint colors to fix phase.

    Key insight:
    - The only ring-touching component already identifies the unique path, and its endpoint colors determine the alternating pattern without needing any global component ordering.
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

    def build_manhattan(p1: tuple[int, int], p2: tuple[int, int], corner_a: bool) -> list[tuple[int, int]]:
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

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], corner_a: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if corner_a else (ra, cb)
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

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        comp_pts[cid].append((*map_to_out(rr, cc), int(g[rr, cc])))

    ring_ids = sorted(comp_pts)
    assert len(ring_ids) == 1
    cid = int(ring_ids[0])

    cols = sorted(set(map(int, g[clab == int(cid)].flatten())))
    assert 1 <= len(cols) <= 2

    hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for r, c, col in comp_pts[cid]:
        hist[(int(r), int(c))][int(col)] += 1

    coords = list(hist.keys())
    assert len(coords) >= 2

    best = None
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
            if best is None or d > best[0]:
                best = (d, coords[i], coords[j])
    assert best is not None
    _d, a, b = best

    path_a = build_manhattan(a, b, True)
    path_b = build_manhattan(a, b, False)
    path = path_b if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else path_a

    if len(cols) == 1:
        out[tuple(np.array(path).T)] = int(cols[0])
        return out

    ca, cb = map(int, cols)
    ha = int(hist[a].most_common(1)[0][0])
    hb = int(hist[b].most_common(1)[0][0])

    chosen = None
    for path_pts in (path, list(reversed(path))):
        for start_color in (ca, cb):
            if int(start_color) != int(ha):
                continue
            end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else (cb if start_color == ca else ca)
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
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 300


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
        grid[rs, cs] = colors[0]
        return
    a, b = colors
    vals = [a if i % 2 == 0 else b for i in range(len(path))]
    grid[rs, cs] = vals


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

    inner, outer = _l_specs(random.choice(["tl", "tr", "bl", "br"]), H, W, r0, c0, r1, c1)
    irs, ics = zip(*inner)
    if used_inner[irs, ics].any():
        return None

    if random.random() < _DUAL_COLOR_PROB:
        colors = (palette[0], palette[1])
        next_color_i = 2
    else:
        colors = (palette[0],)
        next_color_i = 1

    if not _place_component(grid, comp_mask, outer, colors, bg):
        return None
    _paint_path(expected, inner, colors)
    used_inner[irs, ics] = True

    for idx in range(next_color_i, min(len(palette), next_color_i + random.randint(*_DISTRACTOR_COUNT_RANGE))):
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), palette[idx])

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
    raise RuntimeError("failed to generate a valid a6f40cea D7 input")
