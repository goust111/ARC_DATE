# Combo variant of a6f40cea (index 35)
# combo_spec: D2/1+D17/0+D18/0
# Name: 图例映射多色循环

# Design analysis for the combo variant
# ====================================
# 1. Concrete behavior and trace
#    - Original a6f40cea: each external open-rect component touches the frame ring twice and
#      projects to an interior Manhattan path. Monochrome components stay solid; bichrome ones
#      alternate with endpoint colors fixing the phase.
#    - This combo keeps the geometry but inserts a visible color-translation step:
#        * a 3x2 legend block sits in the top-left corner,
#        * each legend row is (external_color, internal_color),
#        * every ring-touching L-component uses the same 3-color external cycle,
#        * the interior path repeats the mapped 3-color cycle in the same phase.
#    - Small trace:
#        outer cycle = [2, 8, 6], legend rows include 2->4, 8->1, 6->7.
#        If a component shows ...2,8,6,2,8... on its external path, the output path becomes
#        ...4,1,7,4,1... after projecting its endpoints into the frame interior.
#      Relative to the original task, the geometry is unchanged but the path colors are no longer
#      copied directly; they must be transformed through the legend first.
#
# 2. Complexity calibration
#    - The mapping must be obvious from only 2-4 examples, so the generator fixes exactly one
#      shared 3-color external cycle per sample and exactly one shared 3-entry legend.
#    - All ring-touching components in a sample reuse that same cycle, but may start at different
#      phases. This gives repeated support for both the cycle order and the color mapping.
#    - The mapped colors are chosen disjoint from the external colors, and identity pairs are
#      forbidden, so the fusion cannot visually collapse back to the original "copy colors inward"
#      rule.
#
# 3. Implicit-assumption audit of the original solver
#    - The original solver stored only sorted component color sets. That destroys the ordered cycle,
#      so this version must traverse each component as an ordered simple path and extract its
#      primitive repeating pattern.
#    - The original two-color branch assumed a binary toggle. Here the successor depends on a
#      3-color cycle, so phase selection must try both path directions and both cycle orientations.
#    - The original solver treated every non-frame colored cell as geometry. That would incorrectly
#      absorb the legend into the component set, so this version first detects and masks out the
#      dedicated 3x2 legend block.
#    - The original "outer colors equal inner colors" premise is broken. Every path color now goes
#      through `legend[external_color]` before painting the interior.
#
# 4. Placement legality
#    - The generator keeps the closed frame, 1px border, and external L-shapes from the base task.
#    - A reserved top-left zone prevents legend/component overlap or 4-neighbor touching.
#    - L-components are accepted only when:
#        * the external path is placeable and disjoint from other objects,
#        * the interior path does not overlap earlier interior paths,
#        * inner and outer path lengths match modulo 3 so the mapped endpoint colors stay visible.
#
# 5. Visual observability and cleanliness
#    - The legend is a compact 3x2 block in a grid corner, visually separate from the frame scene.
#    - Using one shared external cycle and one shared mapping avoids noisy, one-off permutations.
#    - Removing U-pairs and distractors keeps the sample focused on the three-step reasoning chain:
#      read legend -> map colors -> continue the 3-color cycle inside the frame.
#
# 6. New solver assumptions and generator guarantees
#    - Solver assumption: the legend is the unique full 3x2 colored block disconnected from the
#      frame scene. Generator guarantee: it always creates exactly one such block in the reserved
#      top-left region and no other object shares that signature.
#    - Solver assumption: each ring-touching component is a simple 4-connected path whose only ring
#      cells are its two endpoints. Generator guarantee: every external L-component is constructed
#      that way.
#    - Solver assumption: every ring-touching non-legend component uses a primitive 3-color cycle
#      with distinct colors that are all present in the legend. Generator guarantee: all such
#      components use the same 3 distinct external colors, and the legend maps each of them.

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


_LEGEND_H = 3
_LEGEND_W = 2


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


def _ordered_component_path(mask: np.ndarray) -> list[tuple[int, int]] | None:
    """Return the unique endpoint-to-endpoint order for a simple 4-connected path component."""
    pts = np.argwhere(mask)
    if int(len(pts)) < 2:
        return None
    cells = {tuple(map(int, p)) for p in pts}
    deg: dict[tuple[int, int], int] = {}
    for r, c in cells:
        nbs = [(nr, nc) for nr, nc in neighbors4(r, c) if (nr, nc) in cells]
        if len(nbs) > 2:
            return None
        deg[(r, c)] = len(nbs)

    ends = [p for p, d in deg.items() if d == 1]
    if len(ends) != 2:
        return None

    ordered = [ends[0]]
    prev = None
    cur = ends[0]
    while cur != ends[1]:
        nbs = [(nr, nc) for nr, nc in neighbors4(*cur) if (nr, nc) in cells and (nr, nc) != prev]
        if len(nbs) != 1:
            return None
        nxt = nbs[0]
        ordered.append(nxt)
        prev, cur = cur, nxt

    if len(ordered) != len(cells):
        return None
    return ordered


def _primitive_cycle(seq: list[int]) -> tuple[int, ...] | None:
    """Extract the shortest repeating color cycle used by a path."""
    n = len(seq)
    for k in range(1, n + 1):
        pat = tuple(int(x) for x in seq[:k])
        if k > 1 and len(set(pat)) != k:
            continue
        if all(int(seq[i]) == int(pat[i % k]) for i in range(n)):
            return pat
    return None


def _rotate_cycle_to_start(cycle: tuple[int, ...], start_color: int) -> tuple[int, ...] | None:
    """Rotate a cycle so that its first color is start_color."""
    idxs = [i for i, col in enumerate(cycle) if int(col) == int(start_color)]
    if len(idxs) != 1:
        return None
    i = idxs[0]
    return tuple(int(x) for x in cycle[i:] + cycle[:i])


def _paint_cycle(grid: np.ndarray, path: list[tuple[int, int]], cycle: tuple[int, ...]) -> None:
    """Paint a path by repeating cycle from its first point onward."""
    if not path:
        return
    rs, cs = zip(*path)
    vals = [int(cycle[i % len(cycle)]) for i in range(len(path))]
    grid[rs, cs] = vals


def _find_legend_block(g: np.ndarray, bg: int, bc: int, fill: int) -> tuple[dict[int, int], np.ndarray]:
    """Find the unique 3x2 legend block and return (mapping, mask)."""
    mask = (g != int(bg)) & (g != int(bc))
    if int(fill) != int(bg):
        mask &= (g != int(fill))
    lab, n = label(mask, structure=STRUCT4)

    candidates: list[tuple[int, int, dict[int, int], np.ndarray]] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if len(pts) != _LEGEND_H * _LEGEND_W:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if (r1 - r0 + 1, c1 - c0 + 1) != (_LEGEND_H, _LEGEND_W):
            continue
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        if (block == int(bg)).any() or (block == int(bc)).any() or (block == int(fill)).any():
            continue
        mapping = {int(block[r, 0]): int(block[r, 1]) for r in range(_LEGEND_H)}
        if len(mapping) != _LEGEND_H or len(set(mapping.values())) != _LEGEND_H:
            continue
        comp_mask = np.zeros_like(g, dtype=bool)
        comp_mask[lab == int(cid)] = True
        candidates.append((r0, c0, mapping, comp_mask))

    assert candidates, "legend block not found"
    candidates.sort(key=lambda x: (x[0] + x[1], x[0], x[1]))
    _r0, _c0, mapping, legend_mask = candidates[0]
    return mapping, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the frame interior by reading a corner legend and remapping the outer color cycle.

    Input structure:
    - A bg field contains one 1px closed rectangle with uniform fill plus several external
      open-rect traces whose endpoints touch the frame ring.
    - A separate 3x2 legend block in a grid corner lists three horizontal color pairs:
      left cell = outer trace color, right cell = interior path color.

    Rule:
    - Output only the frame interior, initialized with the frame's fill color.
    - Ignore the legend geometrically, but use it as a lookup table from external colors to
      internal colors.
    - For each ring-touching external path, recover its ordered repeating color cycle, project its
      two ring endpoints into the frame interior, and draw the corresponding Manhattan path.
    - Paint that interior path with the mapped cycle, preserving the same phase and direction up to
      the endpoint constraints.

    Key insight:
    - The outer component reveals the cycle order, while the legend says how to translate each
      cycle color before painting inside the frame.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    legend_map, legend_mask = _find_legend_block(g, int(bg), int(bc), int(fill))

    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    def build_manhattan(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> list[tuple[int, int]]:
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

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc)) & (~legend_mask)
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

    for cid in range(1, int(cn) + 1):
        ordered = _ordered_component_path(clab == int(cid))
        if ordered is None:
            continue

        idx_touch = [i for i, p in enumerate(ordered) if p in ring]
        if len(idx_touch) != 2:
            continue
        if sorted(idx_touch) != [0, len(ordered) - 1]:
            continue

        colors_seq = [int(g[r, c]) for r, c in ordered]
        cycle = _primitive_cycle(colors_seq)
        assert cycle is not None
        assert all(int(col) in legend_map for col in cycle)
        mapped_cycle = tuple(int(legend_map[int(col)]) for col in cycle)

        a_src = ordered[0]
        b_src = ordered[-1]
        a = map_to_out(*a_src)
        b = map_to_out(*b_src)
        ha = int(legend_map[int(g[a_src])])
        hb = int(legend_map[int(g[b_src])])

        pathA = build_manhattan(a, b, True)
        pathB = build_manhattan(a, b, False)
        path = pathB if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else pathA

        chosen: tuple[list[tuple[int, int]], tuple[int, ...]] | None = None
        for path_pts, start_hint, end_hint in ((path, ha, hb), (list(reversed(path)), hb, ha)):
            for cyc in (mapped_cycle, tuple(reversed(mapped_cycle))):
                aligned = _rotate_cycle_to_start(cyc, int(start_hint))
                if aligned is None:
                    continue
                if int(aligned[(len(path_pts) - 1) % len(aligned)]) != int(end_hint):
                    continue
                chosen = (path_pts, aligned)
                break
            if chosen is not None:
                break

        assert chosen is not None
        path_pts, aligned = chosen
        _paint_cycle(out, path_pts, aligned)

    return out


# --------------- generator (combo variant) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.5
_INTERIOR_H_RANGE = (5, 12)
_INTERIOR_W_RANGE = (8, 14)
_GRID_SIZE_RANGE = (22, 30)
_GRID_INTERIOR_PAD = 12
_FRAME_MIN_RC = 8
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_L_SPEC_TRIES = 100
_GEN_RETRIES = 800
_LEGEND_TOP = 1
_LEGEND_LEFT = 1
_LEGEND_PAD = 1
_RESERVED_BOTTOM = _LEGEND_TOP + _LEGEND_H + _LEGEND_PAD
_RESERVED_RIGHT = _LEGEND_LEFT + _LEGEND_W + _LEGEND_PAD


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


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    *,
    start_idx: int = 0,
) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    vals = [int(colors[(start_idx + i) % len(colors)]) for i in range(len(path))]
    grid[rs, cs] = vals


def _can_place(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    bg: int,
    comp_mask: np.ndarray,
    blocked_mask: np.ndarray,
) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if bool(blocked_mask[r, c]):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and bool(comp_mask[nr, nc]):
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
    blocked_mask: np.ndarray,
    *,
    start_idx: int = 0,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask, blocked_mask):
        return False
    _paint_path(grid, path, colors, start_idx=start_idx)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _legend_mask(shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    r0 = max(0, _LEGEND_TOP - _LEGEND_PAD)
    c0 = max(0, _LEGEND_LEFT - _LEGEND_PAD)
    r1 = min(H, _LEGEND_TOP + _LEGEND_H + _LEGEND_PAD)
    c1 = min(W, _LEGEND_LEFT + _LEGEND_W + _LEGEND_PAD)
    mask[r0:r1, c0:c1] = True
    return mask


def _paint_legend(grid: np.ndarray, comp_mask: np.ndarray, rows: list[tuple[int, int]]) -> None:
    for i, (outer_col, inner_col) in enumerate(rows):
        rr = _LEGEND_TOP + i
        grid[rr, _LEGEND_LEFT] = int(outer_col)
        grid[rr, _LEGEND_LEFT + 1] = int(inner_col)
        comp_mask[rr, _LEGEND_LEFT] = True
        comp_mask[rr, _LEGEND_LEFT + 1] = True


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


def _find_cyclic_l(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    used_inner: np.ndarray,
    cycle_palette: tuple[int, ...],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int] | None:
    k = len(cycle_palette)
    for _ in range(_L_SPEC_TRIES):
        inner, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
        if len(inner) < k + 2 or len(outer) < k + 2:
            continue
        if (len(inner) - 1) % k != (len(outer) - 1) % k:
            continue
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            continue
        return inner, outer, random.randrange(k)
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill_choices = [x for x in range(10) if x != bc]
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in fill_choices if x != bg])

    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])

    r0_hi = rows - H - _FRAME_MARGIN_FAR
    c0_hi = cols - W - _FRAME_MARGIN_FAR
    if r0_hi < _FRAME_MIN_RC or c0_hi < _FRAME_MIN_RC:
        return None

    r0 = random.randint(_FRAME_MIN_RC, r0_hi)
    c0 = random.randint(_FRAME_MIN_RC, c0_hi)
    r1, c1 = r0 + H + 1, c0 + W + 1

    if r0 <= _RESERVED_BOTTOM or c0 <= _RESERVED_RIGHT:
        return None

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((H, W), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    blocked_mask = _legend_mask(grid.shape)
    used_inner = np.zeros((H, W), dtype=bool)

    avail = [x for x in range(10) if x not in (bg, bc, fill)]
    if len(avail) < 6:
        return None
    outer_cycle = tuple(int(x) for x in random.sample(avail, 3))
    inner_pool = [x for x in avail if x not in outer_cycle]
    inner_cycle = tuple(int(x) for x in random.sample(inner_pool, 3))
    legend_rows = list(zip(outer_cycle, inner_cycle))
    random.shuffle(legend_rows)
    _paint_legend(grid, comp_mask, legend_rows)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    corners = random.sample(["tl", "tr", "bl", "br"], k=n_components)
    for kind in corners:
        spec = _find_cyclic_l(kind, H, W, r0, c0, r1, c1, used_inner, outer_cycle)
        if spec is None:
            return None
        inner, outer, start_idx = spec
        if not _place_component(grid, comp_mask, outer, outer_cycle, bg, blocked_mask, start_idx=start_idx):
            return None
        _paint_path(expected, inner, inner_cycle, start_idx=start_idx)
        irs, ics = zip(*inner)
        used_inner[irs, ics] = True

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
