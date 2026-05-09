# Combo variant of a6f40cea (index 29)
# combo_spec: D1/3+D3/0+D4/3
# Name: 箭头直线贯穿射线

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    Original a6f40cea used external open rectangles that touched the frame ring
#    in two places, and those two touches defined a Manhattan path inside the
#    cropped interior. This combo changes three linked stages together:
#    - D3/0: the external source object becomes an arrow sprite with one ring tip
#    - D1/3: the interior path is forced to be straight, never an L
#    - D4/3: the line no longer connects to another external touch; it keeps
#      going until it reaches the opposite interior side
#    Example: an arrow touching the top ring at column c points downward into the
#    frame. The solver maps that tip to interior cell (0, c), then paints the
#    entire interior column from top to bottom. If the arrow is two-colored, the
#    tip color is the first interior color and the second color alternates after
#    it. A bottom-touching arrow at the same column would paint the same geometry
#    but in the reverse order, so the visible phase flips.
#
# b) Complexity calibration
#    The variant removes the old two-endpoint decoding but keeps enough structure
#    that the rule is still nontrivial:
#    - there are 2-3 arrows per grid, not just one
#    - arrows can start from either of two opposite sides of the chosen axis
#    - monochrome and two-color arrows both appear
#    - at least one arrow per sample is two-color so direction is behaviorally
#      relevant, not merely decorative
#    To keep the scene legible, each sample uses one axis family only:
#    either vertical rays (top/bottom arrows) or horizontal rays (left/right
#    arrows). That avoids row/column crossings that would otherwise create noisy
#    overwrite interactions at intersections.
#
# c) Hidden assumptions broken from the original solver
#    The original main() depended on several premises that fail here:
#    - useful components no longer have >=2 ring touches; each arrow has exactly 1
#    - there is no second endpoint to pair or a corner to choose
#    - deferred mono aggregation for same-row/same-col ring points is obsolete
#    - U-pairs are incompatible with "one arrow -> one through-line", so they are
#      removed entirely
#    The new solver therefore detects one ring tip, validates that the component
#    has the local arrow-head geometry for that side, then derives the full line
#    directly from side + tip position.
#
# d) Placement legality
#    The generator stays output-first. It first chooses the interior line set,
#    then converts each line into a single external arrow with:
#    - one ring-touching tip
#    - a short straight shaft extending away from the frame
#    - a 3-cell head bar just behind the tip, making direction visible
#    Arrows are separated with a minimum gap so their heads never merge into one
#    component. They are also kept away from the grid edge by frame margins large
#    enough to fit the longest shaft.
#
# e) Visual observability
#    A human can infer the rule from a few examples because the visual cue is
#    direct: the old box-like sources are gone, and each meaningful source is now
#    plainly an inward-pointing arrow attached to one side of the frame. The
#    output contains whole rows or whole columns rather than corner L-shapes, so
#    the D1 and D4 changes are both visible in the crop itself.
#
# f) Visual cleanliness
#    Restricting each sample to one axis family keeps the input and output clean.
#    Useful objects sit on opposite sides of the frame with clear spacing; small
#    distractors stay away from the ring and cannot accidentally become solver
#    candidates.
#
# g) Color inference
#    Colors are structural, not id-based. Monochrome arrows produce solid rays.
#    Two-color arrows alternate along the shaft and continue that alternation in
#    the output, anchored only by a visible cue: the ring tip's color is the
#    first output color on the touched side.
#
# h) Degeneracy target
#    These inputs should not collapse back to the original task. The original
#    solver expects two touch points from open rectangles; here each useful
#    component has one touch plus an arrow head, and the output is full-span
#    straight rays rather than missing-corner completions.
#
# i) New implicit assumptions introduced by the new solver
#    The new main() assumes:
#    - every meaningful component has exactly one ring cell
#    - that ring cell belongs to a valid arrow-head pattern for one side
#    - each useful component uses one or two non-frame colors only
#    - useful arrows do not intersect in the cropped output
#    The generator guarantees these by construction: spaced positions, one axis
#    per sample, dedicated palette colors, and explicit arrow-shape validation in
#    the oracle loop.
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


def _straight_path_from_start(
    start: tuple[int, int],
    side: str,
    outH: int,
    outW: int,
) -> list[tuple[int, int]]:
    sr, sc = start
    if side == "top":
        return [(r, sc) for r in range(0, outH)]
    if side == "bottom":
        return [(r, sc) for r in range(outH - 1, -1, -1)]
    if side == "left":
        return [(sr, c) for c in range(0, outW)]
    if side == "right":
        return [(sr, c) for c in range(outW - 1, -1, -1)]
    raise ValueError(side)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct full-span interior rays from external arrows around a framed rectangle.

    Input structure:
    - A uniform background with one 1px closed rectangle whose interior is uniformly filled.
    - Useful external components are inward-pointing arrows. Each arrow has exactly one tip cell
      on the frame ring and a short shaft extending straight away from the frame.

    Rule:
    - Output only the rectangle interior, initialized with the interior fill color.
    - Map each arrow tip to the corresponding interior boundary cell.
    - Use the touched side as the ray direction and draw a straight row or column all the way to
      the opposite interior side.
    - Monochrome arrows paint solid rays; two-color arrows alternate starting from the tip color.

    Key insight:
    - The arrow no longer encodes two endpoints. Its single ring tip fixes the line position, and
      the side/orientation of the arrow determines which full-span straight ray must be drawn.
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
        side = _component_side_of_ring_cell(tip[0], tip[1], r0, c0, r1, c1)
        if side is None or not _has_arrow_tip(coords, tip, side):
            continue

        start = _map_tip(tip[0], tip[1], side, outH, outW, r0, c0)
        path = _straight_path_from_start(start, side, outH, outW)

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
_FRAME_MARGIN = 6
_FRAME_MARGIN_FAR = 6
_ARROW_LEN_RANGE = (3, 5)
_N_COMPONENTS_RANGE = (2, 3)
_MIN_ARROW_GAP = 4
_DUAL_COLOR_PROB = 0.65
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 500


def _max_spaced_count(size: int) -> int:
    return 1 + (max(0, int(size) - 1) // _MIN_ARROW_GAP)


def _sample_spaced_positions(size: int, count: int) -> list[int] | None:
    positions = list(range(int(size)))
    for _ in range(80):
        chosen = sorted(random.sample(positions, count))
        if all(chosen[i + 1] - chosen[i] >= _MIN_ARROW_GAP for i in range(len(chosen) - 1)):
            return [int(x) for x in chosen]
    for offset in range(_MIN_ARROW_GAP):
        pool = list(range(offset, int(size), _MIN_ARROW_GAP))
        if len(pool) >= count:
            chosen = sorted(random.sample(pool, count))
            return [int(x) for x in chosen]
    return None


def _straight_inner_path(side: str, idx: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(r, idx) for r in range(H)]
    if side == "bottom":
        return [(r, idx) for r in range(H - 1, -1, -1)]
    if side == "left":
        return [(idx, c) for c in range(W)]
    if side == "right":
        return [(idx, c) for c in range(W - 1, -1, -1)]
    raise ValueError(side)


def _sample_color_modes(n_components: int) -> list[str]:
    """
    Build a per-sample color-mode plan with explicit branch coverage.

    Every sample contains at least one monochrome arrow and one two-color arrow
    so both painting branches are behaviorally visible to a human solver.
    """
    modes = ["dual", "mono"]
    while len(modes) < int(n_components):
        modes.append("dual" if random.random() < 0.5 else "mono")
    random.shuffle(modes)
    return modes


def _arrow_specs(side: str, idx: int, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    shaft_len = random.randint(*_ARROW_LEN_RANGE)

    if side == "top":
        tip = (r0 - 1, c0 + 1 + idx)
        backbone = _vpath(tip[0], tip[0] - (shaft_len - 1), tip[1])
    elif side == "bottom":
        tip = (r1 + 1, c0 + 1 + idx)
        backbone = _vpath(tip[0], tip[0] + (shaft_len - 1), tip[1])
    elif side == "left":
        tip = (r0 + 1 + idx, c0 - 1)
        backbone = _hpath(tip[0], tip[1], tip[1] - (shaft_len - 1))
    elif side == "right":
        tip = (r0 + 1 + idx, c1 + 1)
        backbone = _hpath(tip[0], tip[1], tip[1] + (shaft_len - 1))
    else:
        raise ValueError(side)

    head = _tip_head_cells(tip, side)
    inner = _straight_inner_path(side, idx, H, W)
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
    rows = random.randint(
        max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD, H + 1 + _FRAME_MARGIN + _FRAME_MARGIN_FAR),
        _GRID_SIZE_RANGE[1],
    )
    cols = random.randint(
        max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD, W + 1 + _FRAME_MARGIN + _FRAME_MARGIN_FAR),
        _GRID_SIZE_RANGE[1],
    )
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

    axis_caps = []
    if _max_spaced_count(W) >= 2:
        axis_caps.append(("vertical", min(3, _max_spaced_count(W))))
    if _max_spaced_count(H) >= 2:
        axis_caps.append(("horizontal", min(3, _max_spaced_count(H))))
    if not axis_caps:
        return None

    axis, max_n = random.choice(axis_caps)
    n_components = random.randint(_N_COMPONENTS_RANGE[0], min(_N_COMPONENTS_RANGE[1], max_n))

    if axis == "vertical":
        positions = _sample_spaced_positions(W, n_components)
        side_pair = ["top", "bottom"]
    else:
        positions = _sample_spaced_positions(H, n_components)
        side_pair = ["left", "right"]
    if positions is None:
        return None

    sides = [side_pair[i % 2] for i in range(n_components)]
    random.shuffle(sides)
    color_modes = _sample_color_modes(n_components)

    for i, idx in enumerate(positions):
        side = sides[i]
        inner, backbone, head = _arrow_specs(side, idx, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        if color_i >= len(palette):
            return None

        if color_modes[i] == "dual" and color_i + 1 < len(palette):
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
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
