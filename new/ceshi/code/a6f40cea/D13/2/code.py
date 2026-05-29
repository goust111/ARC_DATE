# Variant of a6f40cea
# Dimension: D13 — U形配对机制 (path)
# Change: 可选的U形对（两个U形开放框在框架对侧，产生贯穿内部的全跨直线条纹）... → 取消U形对，仅允许L形

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# Design analysis for the D13 variant
# a) Concrete behavior:
#    The original family mixed two external cue types: corner-open L frames and optional opposite-side U pairs.
#    U pairs were recognizable because two same-colored components touched opposite sides of the big frame and
#    the solver merged them into a full-span straight stripe. In this variant every ring-touching component is
#    an L-shaped open frame with exactly two touchpoints on adjacent sides, so every output mark comes from one
#    component and bends once inside the crop. Example: a top-left component touches the top ring at column c
#    and the left ring at row r; originally a top U plus a bottom U of the same color could be aggregated into
#    a vertical stripe, but now only the single top-left component exists, so the output contains the L path
#    from (0, c) to (r, 0) with the interior elbow chosen by geometry.
# b) Complexity calibration:
#    Removing U components would be too subtle if we kept the old 2-3 component budget, because some samples
#    would still look close to the original L-only subset. To make the change visible while staying on-task,
#    the generator uses 3-4 L components from distinct corners. This keeps the object vocabulary simple but
#    increases packing pressure inside the frame, which is the intended replacement challenge.
# c) Hidden-assumption audit:
#    The original `main()` used `deferred` to merge monochrome ring contacts that landed on one row/column.
#    That logic depended on U pairs existing and on multiple components of the same color being intentionally
#    combined. Under L-only inputs that branch is dead, and if left in place it would silently preserve the old
#    D13 behavior on accidental same-axis inputs. The new solver therefore paints each component independently
#    and never aggregates across components. The alternating-color endpoint anchoring remains valid because a
#    two-color L still exposes one ring-touch color at each end.
# d) Placement legality:
#    Components still cannot overlap the frame, each other, or distractors. The generator keeps one component
#    per chosen corner, retries elbow choices until both the interior L and the external open frame fit, and
#    oracle-checks `main(input) == expected` before accepting a sample.
# e) Visual observability:
#    The rule difference is visible from a few examples because outputs no longer contain full-width/full-height
#    stripes sourced from paired U cues; instead, every colored output mark bends at an interior elbow, and
#    there are usually 3-4 such bends per sample. No sorting or mapping depends on color ids.
# f) Visual cleanliness:
#    Corner-specific elbow bands keep top and bottom turns, and left and right turns, separated enough that
#    multiple L paths remain readable. The output is still just the cropped frame interior, so there is no
#    extra annotation layer competing with the main paths.
# g) Color inferability:
#    Colors keep the original semantics: monochrome components paint solid Ls, two-color components alternate
#    along the path with phase fixed by the visible ring-touch endpoint colors. Colors are not remapped.


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
    Reconstruct interior L paths from corner-open traces placed outside a framed rectangle.

    Input structure:
    - A background grid with one 1px closed rectangle whose border color differs from its uniform fill.
    - 3-4 colored components outside the frame, each touching the ring around the frame at two adjacent sides.

    Rule:
    - Output only the interior of the closed frame, initialized with the fill color.
    - For each ring-touching component, map its two touchpoints onto the interior boundary.
    - Draw one Manhattan path per component: solid for one-color components, alternating for two-color ones.

    Key insight:
    - In this variant there is no cross-component U-pair merge; every output mark comes from one component,
      and the ring-touch endpoint colors of a two-color trace fix the alternating phase.
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

    def build_manhattan(
        p1: tuple[int, int], p2: tuple[int, int], cornerA: bool
    ) -> list[tuple[int, int]]:
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
    for cc2 in range(c0, c1 + 1):
        for rr2 in (r0 - 1, r1 + 1):
            if 0 <= rr2 < H:
                ring.add((int(rr2), int(cc2)))
    for rr2 in range(r0, r1 + 1):
        for cc2 in (c0 - 1, c1 + 1):
            if 0 <= cc2 < W:
                ring.add((int(rr2), int(cc2)))

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((mr, mc, int(g[rr, cc])))

    comp_cols: dict[int, list[int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    def farthest_pair(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
        best = None
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                if best is None or d > best[0]:
                    best = (d, coords[i], coords[j])
        if best is None:
            return None
        return best[1], best[2]

    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue
        cols = comp_cols.get(cid, [])
        if len(cols) not in (1, 2):
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1
        coords = list(hist.keys())
        pair = farthest_pair(coords)
        if pair is None:
            continue
        a, b = pair

        pathA = build_manhattan(a, b, True)
        pathB = build_manhattan(a, b, False)
        path = pathB if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else pathA

        if len(cols) == 1:
            colv = int(cols[0])
            for rr, cc in path:
                out[int(rr), int(cc)] = int(colv)
            continue

        ca, cb2 = map(int, cols)
        ha = int(hist[a].most_common(1)[0][0])
        hb = int(hist[b].most_common(1)[0][0])

        chosen = None
        for path_pts in (path, list(reversed(path))):
            for start_color in (ca, cb2):
                end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else (cb2 if start_color == ca else ca)
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
            cur = int(cb2) if cur == int(ca) else int(ca)

    return out


# --------------- generator (a6 D13 L-only) ---------------

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
_N_COMPONENTS_RANGE = (3, 4)
_DUAL_COLOR_PROB = 0.45
_L_COMPONENT_TRIES = 60
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400


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
    else:
        a, b = colors
        grid[rs, cs] = [a if i % 2 == 0 else b for i in range(len(path))]


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


def _sample_elbow(kind: str, H: int, W: int) -> tuple[int, int]:
    top_max = max(1, (H - 2) // 2)
    bottom_min = min(H - 2, max(1, top_max + 1))
    left_max = max(1, (W - 2) // 2)
    right_min = min(W - 2, max(1, left_max + 1))

    if kind[0] == "t":
        r = random.randint(1, top_max)
    else:
        r = random.randint(bottom_min, H - 2)
    if kind[1] == "l":
        c = random.randint(1, left_max)
    else:
        c = random.randint(right_min, W - 2)
    return int(r), int(c)


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    r, c = _sample_elbow(kind, H, W)

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

    n_components = 4 if (H >= 6 and W >= 8 and random.random() < 0.5) else 3
    n_components = max(_N_COMPONENTS_RANGE[0], min(_N_COMPONENTS_RANGE[1], n_components))
    corners = random.sample(["tl", "tr", "bl", "br"], k=n_components)

    extra_budget = len(palette) - n_components
    wanted_dual = [i for i in range(n_components) if random.random() < _DUAL_COLOR_PROB]
    random.shuffle(wanted_dual)
    dual_set = set(wanted_dual[: max(0, extra_budget)])

    colors_for_components: list[tuple[int, ...]] = []
    color_i = 0
    for i in range(n_components):
        if i in dual_set:
            if color_i + 1 >= len(palette):
                return None
            colors_for_components.append((palette[color_i], palette[color_i + 1]))
            color_i += 2
        else:
            if color_i >= len(palette):
                return None
            colors_for_components.append((palette[color_i],))
            color_i += 1

    for kind, colors in zip(corners, colors_for_components):
        placed = False
        for _ in range(_L_COMPONENT_TRIES):
            inner, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
            irs, ics = zip(*inner)
            if used_inner[irs, ics].any():
                continue
            if not _place_component(grid, comp_mask, outer, colors, bg):
                continue
            _paint_path(expected, inner, colors)
            used_inner[irs, ics] = True
            placed = True
            break
        if not placed:
            return None

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
    raise RuntimeError("failed to generate a valid a6f40cea D13 input")
