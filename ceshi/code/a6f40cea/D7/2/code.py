# Variant of a6f40cea
# Dimension: D7 — 组件数量与配置 (meta)
# Change: 2-3个开放框架组件（L形或U形对），每个独立处理... → 组件间有依赖顺序

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# Design analysis
# ----------------
# a) Concrete behavior:
#    - We keep the original closed frame + external open-rect vocabulary, but we narrow the
#      generator to L-components so the new D7 signal is visually obvious.
#    - Each external component still projects to one interior Manhattan L-path.
#    - New rule: components are not independent. Each one has an external elbow outside one
#      frame corner; the farther that elbow sits from its frame corner, the later that path is
#      painted. Later paths overwrite earlier ones at crossings.
#    - Small trace:
#      original: component A paints its path, component B paints its path, but overlap does not
#      matter because generator avoids overlap.
#      variant: A and B are allowed to cross; if A has reach 4 and B has reach 7, A is painted
#      first, B second, so the crossing cell shows B's color sequence.
# b) Complexity calibration:
#    - Ordering depends on one visible scalar only: geometric reach from the relevant frame
#      corner to the component's outer elbow.
#    - We keep 2-3 components, and generator forces at least one real overwrite, so the ordering
#      is neither trivial nor hidden.
#    - We preserve monochrome vs alternating 2-color traces from the base task.
# c) Old assumptions that break:
#    - The original `main()` writes directly into the output while iterating connected
#      components, implicitly assuming order is irrelevant.
#    - The original generator uses `used_inner` to ban internal overlap, which suppresses any
#      D7 dependency.
#    - In this variant we must first decode every component into a draw operation, sort those
#      operations by reach, and only then paint.
# d) Placement legality:
#    - Outer components must still be non-overlapping, in-bounds, and non-touching.
#    - Internal paths may overlap; this is now intentional.
#    - Generator enforces distinct reach values so the priority order is unambiguous.
# e) Visual observability:
#    - Ordering never depends on color ids or hidden metadata.
#    - The user can compare how far each external elbow protrudes from its frame corner.
#    - Samples are rejected unless at least one crossing cell is actually overwritten by a later
#      component with a different visible color.
# f) Visual cleanliness:
#    - Only the interior paths overlap; the outer evidence remains separated around the frame.
#    - We keep the same frame/object vocabulary and avoid stacked markers or annotations.
# g) Color inference:
#    - Colors keep the original semantics: solid for 1-color components, endpoint-anchored
#      alternation for 2-color components.
#    - The new logic does not add any extra color remapping.
#
# New `main()` assumptions and generator guarantees:
# - Meaningful ring-touching components are L-shaped and touch one horizontal and one vertical
#   frame side; generator guarantees this.
# - Their reach values are distinct; generator guarantees this to avoid ambiguous ordering.
# - At least one pair of paths overlaps with a real visible overwrite; generator guarantees this.


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


def _build_manhattan(
    p1: tuple[int, int], p2: tuple[int, int], corner_a: bool
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        return [(int(ra), int(c)) for c in _line(ca, cb)]
    if ca == cb:
        return [(int(r), int(ca)) for r in _line(ra, rb)]
    if corner_a:
        return _join(_vpath(ra, rb, ca), _hpath(rb, ca, cb))
    return _join(_hpath(ra, ca, cb), _vpath(ra, rb, cb))


def _inside_corner(
    p1: tuple[int, int], p2: tuple[int, int], corner_a: bool, out_h: int, out_w: int
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    rr, cc = ((rb, ca) if corner_a else (ra, cb))
    return (0 < rr < out_h - 1) and (0 < cc < out_w - 1)


def _component_order_key(
    pts: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> tuple[int, int, int, int]:
    rows = set(map(int, pts[:, 0]))
    cols = set(map(int, pts[:, 1]))
    min_r, min_c = map(int, pts.min(axis=0))
    max_r, max_c = map(int, pts.max(axis=0))

    touch_top = int(r0 - 1) in rows
    touch_bottom = int(r1 + 1) in rows
    touch_left = int(c0 - 1) in cols
    touch_right = int(c1 + 1) in cols

    if touch_top and touch_left:
        outer = (min_r, min_c)
        frame_corner = (r0, c0)
        rank = 0
    elif touch_top and touch_right:
        outer = (min_r, max_c)
        frame_corner = (r0, c1)
        rank = 1
    elif touch_bottom and touch_right:
        outer = (max_r, max_c)
        frame_corner = (r1, c1)
        rank = 2
    elif touch_bottom and touch_left:
        outer = (max_r, min_c)
        frame_corner = (r1, c0)
        rank = 3
    else:
        depth = max(r0 - min_r, max_r - r1, c0 - min_c, max_c - c1)
        return (int(depth), 9, int(min_r), int(min_c))

    reach = abs(int(outer[0]) - int(frame_corner[0])) + abs(int(outer[1]) - int(frame_corner[1]))
    return (int(reach), int(rank), int(outer[0]), int(outer[1]))


def _extract_draw_ops(
    g: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int, int, int], list[dict[str, object]]]:
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
        rr2 = max(0, min(out_h - 1, int(rr - r_int0)))
        cc2 = max(0, min(out_w - 1, int(cc - c_int0)))
        return int(rr2), int(cc2)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < g.shape[0]:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < g.shape[1]:
                ring.add((int(rr), int(cc)))

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((int(mr), int(mc), int(g[rr, cc])))

    ops: list[dict[str, object]] = []
    for cid in range(1, int(cn) + 1):
        pts = comp_pts.get(int(cid), [])
        if len(pts) < 2:
            continue

        all_pts = np.argwhere(clab == int(cid))
        order_key = _component_order_key(all_pts, r0, c0, r1, c1)
        cols = sorted(set(map(int, g[clab == int(cid)].flatten())))
        if len(cols) not in (1, 2):
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1
        coords = list(hist.keys())
        if len(coords) < 2:
            continue

        best = None
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                if best is None or d > best[0]:
                    best = (int(d), coords[i], coords[j])
        if best is None:
            continue

        _dist, p1, p2 = best
        path_a = _build_manhattan(p1, p2, True)
        path_b = _build_manhattan(p1, p2, False)
        chosen_path = (
            path_b
            if (_inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w))
            else path_a
        )

        if len(cols) == 1:
            vals = [int(cols[0])] * len(chosen_path)
            ops.append({"cid": int(cid), "key": order_key, "path": chosen_path, "vals": vals})
            continue

        ca, cb = map(int, cols)
        ha = int(hist[p1].most_common(1)[0][0])
        hb = int(hist[p2].most_common(1)[0][0])
        chosen = None
        for path_pts in (chosen_path, list(reversed(chosen_path))):
            for start_color in (ca, cb):
                if len(path_pts) % 2 == 1:
                    end_color = int(start_color)
                else:
                    end_color = int(cb) if int(start_color) == int(ca) else int(ca)
                if int(start_color) == int(ha) and int(end_color) == int(hb):
                    vals = []
                    cur = int(start_color)
                    for _ in path_pts:
                        vals.append(int(cur))
                        cur = int(cb) if int(cur) == int(ca) else int(ca)
                    chosen = (path_pts, vals)
                    break
            if chosen is not None:
                break
        assert chosen is not None
        path_pts, vals = chosen
        ops.append({"cid": int(cid), "key": order_key, "path": path_pts, "vals": vals})

    return out, (r0, c0, r1, c1, bc, fill), ops


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct interior paths from external open rectangles, with order-dependent overwriting.

    Input structure:
    - A uniform background, one closed 1px rectangular frame, and 2-3 colored L-components outside it.
    - Each component touches two adjacent frame sides through ring cells and may be monochrome or 2-color alternating.

    Rule:
    - Crop to the frame interior and start from the uniform fill color.
    - Decode each external component into its corresponding interior Manhattan L-path.
    - Sort the decoded paths by how far the component's outer elbow sits from its frame corner, then paint in that order.
    - Later paths overwrite earlier ones wherever they cross.

    Key insight:
    - The only new dependency is geometric priority: the farther an outside elbow protrudes from its corner, the later its path is drawn.
    """
    g = np.array(grid, dtype=int)
    out, _frame, ops = _extract_draw_ops(g)
    for op in sorted(ops, key=lambda item: item["key"]):
        path = op["path"]
        vals = op["vals"]
        for (rr, cc), col in zip(path, vals):
            out[int(rr), int(cc)] = int(col)
    return out


# --------------- generator ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 11
_FRAME_MARGIN = 5
_FRAME_MARGIN_FAR = 5
_OUTER_OFFSET_RANGE = (2, 4)
_N_COMPONENTS_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 600


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
    h, w = grid.shape
    uniq = set(coords)
    if len(uniq) != len(coords):
        return False
    for r, c in uniq:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in uniq and 0 <= nr < h and 0 <= nc < w and comp_mask[nr, nc]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
) -> bool:
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _component_colors(palette: list[int], color_i: int) -> tuple[tuple[int, ...], int]:
    if random.random() < _DUAL_COLOR_PROB and color_i + 1 < len(palette):
        return (int(palette[color_i]), int(palette[color_i + 1])), int(color_i + 2)
    return (int(palette[color_i]),), int(color_i + 1)


def _l_specs(
    kind: str,
    h: int,
    w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    turn_r: int,
    turn_c: int,
    off_r: int,
    off_c: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if kind == "tl":
        ring_a = (r0 - 1, c0 + 1 + turn_c)
        ring_b = (r0 + 1 + turn_r, c0 - 1)
        outer_corner = (r0 - off_r, c0 - off_c)
        inner = _join(_vpath(0, turn_r, turn_c), _hpath(turn_r, turn_c, 0))
    elif kind == "tr":
        ring_a = (r0 - 1, c0 + 1 + turn_c)
        ring_b = (r0 + 1 + turn_r, c1 + 1)
        outer_corner = (r0 - off_r, c1 + off_c)
        inner = _join(_vpath(0, turn_r, turn_c), _hpath(turn_r, turn_c, w - 1))
    elif kind == "bl":
        ring_a = (r1 + 1, c0 + 1 + turn_c)
        ring_b = (r0 + 1 + turn_r, c0 - 1)
        outer_corner = (r1 + off_r, c0 - off_c)
        inner = _join(_vpath(h - 1, turn_r, turn_c), _hpath(turn_r, turn_c, 0))
    else:
        ring_a = (r1 + 1, c0 + 1 + turn_c)
        ring_b = (r0 + 1 + turn_r, c1 + 1)
        outer_corner = (r1 + off_r, c1 + off_c)
        inner = _join(_vpath(h - 1, turn_r, turn_c), _hpath(turn_r, turn_c, w - 1))

    outer = _join(
        _vpath(ring_a[0], outer_corner[0], ring_a[1]),
        _hpath(outer_corner[0], ring_a[1], outer_corner[1]),
        _vpath(outer_corner[0], ring_b[0], outer_corner[1]),
        _hpath(ring_b[0], outer_corner[1], ring_b[1]),
    )
    return inner, outer


def _pick_kinds(n_components: int) -> list[str]:
    diag = random.choice([("tl", "br"), ("tr", "bl")])
    kinds = [diag[0], diag[1]]
    if n_components == 3:
        kinds.append(random.choice([k for k in ("tl", "tr", "bl", "br") if k not in kinds]))
    return kinds


def _turn_specs_for_kinds(h: int, w: int, kinds: list[str]) -> dict[str, tuple[int, int]] | None:
    specs: dict[str, tuple[int, int]] = {}
    kind_set = set(kinds)

    if {"tl", "br"} <= kind_set:
        hi_r = random.randint(max(2, h // 2), h - 2)
        hi_c = random.randint(max(2, w // 2), w - 2)
        lo_r = random.randint(1, hi_r - 1)
        lo_c = random.randint(1, hi_c - 1)
        specs["tl"] = (hi_r, hi_c)
        specs["br"] = (lo_r, lo_c)
    elif {"tr", "bl"} <= kind_set:
        hi_r = random.randint(max(2, h // 2), h - 2)
        lo_r = random.randint(1, hi_r - 1)
        lo_c = random.randint(1, max(1, w // 2 - 1))
        hi_c = random.randint(lo_c + 1, w - 2)
        specs["tr"] = (hi_r, lo_c)
        specs["bl"] = (lo_r, hi_c)

    for kind in kinds:
        if kind in specs:
            continue
        if kind == "tl":
            rr = random.randint(1, max(1, h // 2))
            cc = random.randint(1, max(1, w // 2))
        elif kind == "tr":
            rr = random.randint(1, max(1, h // 2))
            cc = random.randint(max(1, w // 2), w - 2)
        elif kind == "bl":
            rr = random.randint(max(1, h // 2), h - 2)
            cc = random.randint(1, max(1, w // 2))
        else:
            rr = random.randint(max(1, h // 2), h - 2)
            cc = random.randint(max(1, w // 2), w - 2)
        specs[kind] = (int(rr), int(cc))

    if len(specs) != len(kinds):
        return None
    return specs


def _sample_offsets(kinds: list[str]) -> dict[str, tuple[int, int, int]] | None:
    lo, hi = _OUTER_OFFSET_RANGE
    sums_used: set[int] = set()
    out: dict[str, tuple[int, int, int]] = {}
    for kind in kinds:
        choices = [(a, b) for a in range(lo, hi + 1) for b in range(lo, hi + 1) if (a + b) not in sums_used]
        if not choices:
            return None
        off_r, off_c = random.choice(choices)
        reach = int(off_r + off_c)
        sums_used.add(reach)
        out[kind] = (int(off_r), int(off_c), int(reach))
    return out


def _ordered_expected(
    h: int, w: int, fill: int, ops: list[dict[str, object]]
) -> tuple[np.ndarray, int, int]:
    expected = np.full((h, w), int(fill), dtype=int)
    seen: dict[tuple[int, int], int] = {}
    overlap_count = 0
    overwrite_count = 0
    for op in sorted(ops, key=lambda item: item["key"]):
        path = op["path"]
        vals = op["vals"]
        for cell, col in zip(path, vals):
            cell = (int(cell[0]), int(cell[1]))
            col = int(col)
            if cell in seen:
                overlap_count += 1
                if int(seen[cell]) != int(col):
                    overwrite_count += 1
            expected[cell[0], cell[1]] = int(col)
            seen[cell] = int(col)
    return expected, int(overlap_count), int(overwrite_count)


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    rows, cols = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r = random.randint(1, rows - 2)
        c = random.randint(1, cols - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and (nr, nc) not in used
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


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])

    h = random.randint(*_INTERIOR_H_RANGE)
    w = random.randint(*_INTERIOR_W_RANGE)
    min_rows = max(_GRID_SIZE_RANGE[0], h + _GRID_INTERIOR_PAD)
    min_cols = max(_GRID_SIZE_RANGE[0], w + _GRID_INTERIOR_PAD)
    rows = random.randint(min_rows, _GRID_SIZE_RANGE[1])
    cols = random.randint(min_cols, _GRID_SIZE_RANGE[1])

    r0_hi = rows - h - _FRAME_MARGIN_FAR - 2
    c0_hi = cols - w - _FRAME_MARGIN_FAR - 2
    if r0_hi < _FRAME_MARGIN or c0_hi < _FRAME_MARGIN:
        return None
    r0 = random.randint(_FRAME_MARGIN, r0_hi)
    c0 = random.randint(_FRAME_MARGIN, c0_hi)
    r1, c1 = r0 + h + 1, c0 + w + 1

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    color_i = 0

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    kinds = _pick_kinds(n_components)
    turn_specs = _turn_specs_for_kinds(h, w, kinds)
    offsets = _sample_offsets(kinds)
    if turn_specs is None or offsets is None:
        return None

    ops: list[dict[str, object]] = []
    for kind in kinds:
        if color_i >= len(palette):
            return None
        colors, color_i = _component_colors(palette, color_i)
        turn_r, turn_c = turn_specs[kind]
        off_r, off_c, reach = offsets[kind]
        inner, outer = _l_specs(kind, h, w, r0, c0, r1, c1, turn_r, turn_c, off_r, off_c)
        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None
        vals: list[int]
        if len(colors) == 1:
            vals = [int(colors[0])] * len(inner)
        else:
            a, b = map(int, colors)
            vals = [a if i % 2 == 0 else b for i in range(len(inner))]
        ops.append({"kind": kind, "key": (int(reach),), "path": inner, "vals": vals})

    expected, overlap_count, overwrite_count = _ordered_expected(h, w, fill, ops)
    if overlap_count == 0 or overwrite_count == 0:
        return None

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
    raise RuntimeError("failed to generate a valid a6f40cea D7 input")
