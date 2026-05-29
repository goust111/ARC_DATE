# Combo variant of 35ab12c3 (index 20)
# combo_spec: D6/0+D13/3+D19/2
# Name: 按颜色值排序处理+全覆盖绘制+阴影优先

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
------------------------------
Original 35ab12c3 treats each color-group as an effectively independent object:
main edges are painted first, shadows are painted second, and both only fill
background cells. With that rule, cross-object ordering is invisible because
separated objects never compete and even self-overlap keeps the main shape color.

This combo makes three changes work together:
1. Colors are processed in ascending color-id order, and that order now matters.
2. Every draw is overwrite-capable, so later shapes can cover earlier ones.
3. Within one shape, the shadow is still drawn after the main edges, so the shadow
   now overwrites the shape on overlap cells.

Small canonical trace used by the generator:
- color 1: a rectangle border with a red shadow anchor at diagonal offset (+1,+1);
- color 4: a long vertical line crossing the rectangle;
- color 8: a long horizontal line crossing both of them.

Execution:
- Step A: color-1 rectangle border is painted; then its shifted shadow is painted and
  recolors the self-overlap cells from color 1 to color 2.
- Step B: color-4 line is processed later, so it overwrites any color-1 / color-2
  cells on its path.
- Step C: color-8 line is processed last, so it overwrites the color-4 crossing cell
  and any earlier rectangle/shadow cells on its path.

The visible result is a three-layer stack:
shadow beats its own base shape locally, but a later color-group still beats both.

b) Complexity calibration
-------------------------
The risky failure mode is degeneration back to the original task:
- if no shadow self-overlap occurs, D6 is invisible;
- if different shapes do not overlap, D13 and D19 are invisible.

To prevent that, every generated grid contains one mandatory interaction patch:
- a shadowed rectangle with guaranteed self-overlap;
- one later-processed vertical line crossing it;
- one still-later horizontal line crossing both.

This keeps the free variables small and repeated across examples. The palette roles
are fixed for the interaction patch, so the precedence chain is repeatedly visible
instead of being hidden behind too many random color combinations.

c) Implicit-assumption audit against the original main()
--------------------------------------------------------
Relevant original assumptions that no longer hold:

1. `_draw_line_bg_only(...)` for main edges:
   original assumption: edge drawing never needs to overwrite an existing color.
   combo status: invalid. Later color-groups must cover earlier ones.

2. `_draw_line_bg_only(...)` for shadows:
   original assumption: shape color wins when shape and shadow overlap.
   combo status: invalid. Shadow must repaint those cells.

3. `colors = sorted(...)`:
   original assumption: harmless iteration detail because objects stay disjoint.
   combo status: now a real rendering rule. The generator must deliberately create
   cross-shape overlap so this ordering is observable.

4. `_clear_in_original(...)`:
   original assumption: checking only the sparse input is sufficient because later
   drawing never changes eligibility.
   combo status: still valid. The combo changes write precedence, not geometric
   eligibility. Segments are still inferred only from the original sparse input.

5. Object separation in the generator:
   original assumption: final output footprints should not overlap.
   combo status: invalid for the mandatory patch. Controlled overlap is required,
   but only inside that patch; unrelated extras remain separated.

d) Placement legality
---------------------
The generator uses one pre-validated local interaction patch and optionally places
isolated extras elsewhere with `random_free_location(..., padding=1)`.

Legality guarantees:
- all mandatory overlapping happens inside the local patch by construction;
- sparse input points from different mandatory groups never coincide;
- shadow anchors are 8-adjacent to exactly one vertex and never lie on a shape cell;
- optional extra shapes / distractors are placed disjointly from the patch output.

e) Visual observability
-----------------------
The combo contains one inherently non-local rule: color-id order. To keep that
inferable in practice, the generator repeats the same visible precedence chain across
examples instead of randomizing the interacting shape colors. A human sees:
- the same shadow color eating into the rectangle border;
- the same later vertical line cutting through that earlier layer;
- the same final horizontal line sitting on top of the crossing stack.

That repeated structure makes the layered rendering rule observable from a few
examples rather than hiding it behind one-off color accidents.

f) Visual cleanliness
---------------------
Only one dense overlap zone is forced. Optional extras are isolated and use the
original sparse vocabulary. This avoids turning the whole grid into unreadable
multicolor noise while still making the fusion behavior unmistakable.

g) Color inference
------------------
No arbitrary color remapping is introduced. Colors remain object identities, but the
mandatory interaction patch reuses the same identities across examples so the user can
visually infer a stable precedence chain. The shadow color is distinct from all shape
colors and only appears via the usual nearby anchor cue plus the shifted copy.

New solver assumptions introduced by the combo and generator support
--------------------------------------------------------------------
1. Cross-group overlap is allowed and meaningful.
   Generator support: the mandatory patch guarantees several such overlaps.

2. Shadow/self overlap must be visible.
   Generator support: the rectangle uses a diagonal shadow offset that always creates
   repaint cells on the border.

3. The chosen shadow anchor must remain unambiguous after random transpose / flips.
   Generator support: the anchor point is transformed together with the shape and is
   checked to be 8-adjacent to exactly one vertex.
"""

from __future__ import annotations

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg as _bg_color
from arc_utils import get_bg, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    if dr != 0 and abs(dr) == abs(dc):
        return True
    return False


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_line_with_policy(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    col: int,
    bg: int,
    *,
    overwrite: bool,
) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if overwrite or out[rr, cc] == bg:
            out[rr, cc] = col


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _solve(grid: np.ndarray, *, main_overwrite: bool, shadow_overwrite: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        seed_idxs = []
        for i, (r, c) in enumerate(cycle):
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != bg and v != col:
                        seed_idxs.append(i)
                        break

        if seed_idxs:
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
        cycle = cycle[start:] + cycle[:start]

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_with_policy(out, r0, c0, r1, c1, col, bg, overwrite=main_overwrite)

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_with_policy(out, rr0, cc0, rr1, cc1, ncol, bg, overwrite=shadow_overwrite)

    return out


def _original_reference(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, main_overwrite=False, shadow_overwrite=False)


def _shadow_shape_wins_reference(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, main_overwrite=True, shadow_overwrite=False)


def _no_cross_overwrite_reference(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, main_overwrite=False, shadow_overwrite=True)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    final = main(grid)
    shadow_ref = _shadow_shape_wins_reference(grid)
    cross_ref = _no_cross_overwrite_reference(grid)
    original_ref = _original_reference(grid)
    return {
        "shadow_priority_diff": int(np.count_nonzero(final != shadow_ref)),
        "cross_overwrite_diff": int(np.count_nonzero(final != cross_ref)),
        "original_diff": int(np.count_nonzero(final != original_ref)),
    }


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (inp != BG) | (out != BG)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return inp.copy(), out.copy()
    r0, c0, r1, c1 = bbox_coords(coords)
    return inp[r0 : r1 + 1, c0 : c1 + 1].copy(), out[r0 : r1 + 1, c0 : c1 + 1].copy()


"""
Overview:
    Reconstruct every drawable H/V/45-degree shape in ascending color-id order and
    treat that order as a real painter's algorithm.

Input structure:
    Each shape color appears as sparse vertices. If one vertex has an adjacent pixel
    of another color, that neighbor encodes a shadow offset/color for the whole shape.
    Distractors remain as isolated pixels.

Rule:
    For each non-background color from small to large, infer its polygon/polyline
    edges exactly as in the base task. Draw the main edges with overwrite enabled,
    then draw every carried parallel shadow edge with overwrite enabled as well.
    Because shadows are drawn after their parent shape, shadow cells beat shape cells;
    because later colors are processed later, higher-priority later groups cover
    earlier groups on overlap.

Key insight:
    The geometry inference is unchanged; only rendering precedence changed. The final
    image is therefore determined by a layered write order:
    earlier color-group < its shadow < later color-group < that later group's shadow.
"""
def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, main_overwrite=True, shadow_overwrite=True)


# --------------- generator ---------------

BG = 0
MANDATORY_SHAPE_COLORS = (1, 4, 8)
MANDATORY_SHADOW_COLOR = 2
EXTRA_COLORS = (3, 6, 7)

CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

DISTRACTOR_PROB = 0.45
EXTRA_SHAPE_PROB = 0.55
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8
GENERATE_RETRIES = 300
SHAPE_PLACEMENT_RETRIES = 40

INTERACTION_RECT_H_RANGE = (7, 11)
INTERACTION_RECT_W_RANGE = (9, 14)
MIN_CROSS_OVERWRITE_DIFF = 5

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 7)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 6)
SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _transform_named(named: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    keys = list(named)
    vals = _transform_coords([named[k] for k in keys])
    return {k: vals[i] for i, k in enumerate(keys)}


def _build_interaction_patch() -> tuple[np.ndarray, np.ndarray] | None:
    rect_h = int(np.random.randint(*INTERACTION_RECT_H_RANGE))
    rect_w = int(np.random.randint(*INTERACTION_RECT_W_RANGE))
    cross_c = int(np.random.randint(3, rect_w))
    cross_r = int(np.random.randint(3, rect_h))

    named = {
        "rect_tl": (1, 1),
        "rect_tr": (1, rect_w),
        "rect_br": (rect_h, rect_w),
        "rect_bl": (rect_h, 1),
        "shadow_anchor": (2, 2),
        "vert_top": (-1, cross_c),
        "vert_bot": (rect_h + 2, cross_c),
        "hori_left": (cross_r, -1),
        "hori_right": (cross_r, rect_w + 2),
    }
    named = _transform_named(named)

    h = max(r for r, _ in named.values()) + 1
    w = max(c for _, c in named.values()) + 1
    inp = np.full((h, w), BG, dtype=int)

    for key in ("rect_tl", "rect_tr", "rect_br", "rect_bl"):
        rr, cc = named[key]
        inp[rr, cc] = MANDATORY_SHAPE_COLORS[0]

    rr, cc = named["shadow_anchor"]
    if inp[rr, cc] != BG:
        return None
    inp[rr, cc] = MANDATORY_SHADOW_COLOR

    for key in ("vert_top", "vert_bot"):
        rr, cc = named[key]
        if inp[rr, cc] != BG:
            return None
        inp[rr, cc] = MANDATORY_SHAPE_COLORS[1]

    for key in ("hori_left", "hori_right"):
        rr, cc = named[key]
        if inp[rr, cc] != BG:
            return None
        inp[rr, cc] = MANDATORY_SHAPE_COLORS[2]

    out = main(inp)
    metrics = _fusion_metrics(inp)
    if metrics["shadow_priority_diff"] <= 0:
        return None
    if metrics["cross_overwrite_diff"] < MIN_CROSS_OVERWRITE_DIFF:
        return None
    if metrics["original_diff"] <= 0:
        return None
    return _crop_pair(inp, out)


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_triangle() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRIANGLE_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    h = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (h, 0), (h, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _make_isolated_shape_patch(color: int) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    cells = _shape_cells(vertices)
    coords = np.array(list(cells) + vertices, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _shape_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=color)
    for r, c in shifted_vertices:
        inp[r, c] = color

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(h)), int(np.random.randint(w)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]):
                return None
    return coords


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    mandatory = _build_interaction_patch()
    if mandatory is None:
        return None
    if not _try_place(inp, out, *mandatory):
        return None

    remaining = list(EXTRA_COLORS)
    np.random.shuffle(remaining)

    if remaining and np.random.rand() < EXTRA_SHAPE_PROB:
        placed = False
        color = int(remaining.pop())
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_isolated_shape_patch(color)
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if remaining and np.random.rand() < DISTRACTOR_PROB:
        color = int(remaining.pop())
        if not _try_place(inp, out, *_make_distractor_patch(color)):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if not np.array_equal(main(inp), out):
        return None

    metrics = _fusion_metrics(inp)
    if metrics["shadow_priority_diff"] <= 0:
        return None
    if metrics["cross_overwrite_diff"] < MIN_CROSS_OVERWRITE_DIFF:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
