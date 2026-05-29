# Variant of 35ab12c3
# Dimension: D13 — 多步处理顺序 (multi_step)
# Change: 三步流程：Step1识别同色像素对 -> Step2画连线 -> Step3画平行阴影线。主线先画、阴影后画（避免阴影阻挡主线） → 迭代扩展——逐步增长

"""
Design analysis for the D13 variant
==================================

a) Concrete behavior and trace
------------------------------
Original 35ab12c3 is static: once same-color vertices are grouped into a polyline /
polygon, every eligible edge is drawn in one shot, and only after that the parallel
shadow copies are drawn. Because all edge eligibility is checked only against the
original input, freshly drawn lines never affect later lines.

This variant keeps the same visible vocabulary:
- same-color vertices define a shape edge set;
- one 8-neighbor pixel of another color can define a carried shadow offset;
- distractors still stay unchanged.

The change is only the execution order:
- every eligible main edge becomes a frontier;
- at each tick, every unfinished main edge can advance by at most one pixel;
- after the main step of that tick lands, its shadow copy may place the parallel pixel
  at the same progress index;
- any non-background pixel already present in the evolving output is now an obstacle.

Canonical small trace used by the generator:
- winner shape: two vertical endpoints with distance 3 from its start to the crossing;
- loser shape: two horizontal endpoints with distance 4 from its start to the same crossing;
- static/original behavior would draw a full cross;
- dynamic behavior reaches the crossing with the vertical line first, so the horizontal
  frontier stops one cell early and never reaches its far endpoint. Its optional shadow
  stops in parallel as well.

b) Complexity calibration
-------------------------
If the generator kept the original "fully separated output footprints" placement rule,
the new solver would usually produce the same output as the old static solver, so D13
would be visually invisible. To avoid that degeneration while keeping the task readable:
- each generated grid contains at least one deliberate blocking pair;
- only one such contested interaction is forced, so the scene remains easy to parse;
- optional extra shapes remain isolated and follow the original static-looking behavior,
  preserving the original object vocabulary without adding too many simultaneous races.

c) Implicit-assumption audit against the original main()
--------------------------------------------------------
Relevant original assumptions that break under D13:

1. `colors = sorted(...)` followed by per-color drawing was harmless before because the
   final output did not depend on processing order. Under D13 that would become a hidden
   tie-break. Therefore this variant still parses per color, but the drawing itself is
   simulated globally and simultaneously across all shapes.

2. `_clear_in_original(...)` was sufficient to decide the whole edge forever. Under D13
   that is only an initial eligibility test. It is still used to define candidate edges,
   but final completion is decided incrementally by the evolving output state.

3. `_draw_line_bg_only(...)` assumed "all remaining cells are still background enough".
   That is no longer true. The replacement is a frontier state with `progress`, `blocked`,
   and per-tick movement.

4. The original "draw every main edge, then every shadow edge" guaranteed that shadows
   never blocked mains. In the variant, shadows no longer run as a later global phase.
   Instead, each shadow is attached to its parent main edge and only attempts the pixel
   corresponding to the main edge's newly reached progress. This preserves local
   main-before-shadow priority without reverting to the old static two-phase fill.

d) Placement legality
---------------------
The generator still places patches using the filled output footprint, not only the sparse
input vertices. This keeps unrelated objects disjoint and inside bounds. The one intended
interaction is local to a single pre-built blocking patch whose own input/output are
 validated before global placement.

e) Visual observability
-----------------------
The dynamic rule is inferable from 2-3 ARC examples because:
- the same kind of sparse endpoint input can sometimes yield a complete line and
  sometimes a truncated one;
- the truncation happens exactly where another line reached first;
- when a loser carries a shadow, the shadow truncates in the same parallel fashion,
  reinforcing that the scene is "growing" rather than "batch drawn".
No color-id-based ordering is used as part of the intended rule.

f) Visual cleanliness
---------------------
Only one contested corridor is forced per grid. Other shapes, if present, are placed with
padding from the final output footprint. This keeps the dynamic interaction readable
instead of turning the grid into dense overdraw.

g) Color inference
------------------
Colors remain structural labels only: one color per shape, optional one color per shadow,
optional one distractor color. There is no hidden color remapping or color-ranked logic.

New assumptions introduced by this variant and how the generator guarantees them
-----------------------------------------------------------------------
1. The intended blocking examples rely on unequal arrival times to a shared cell. The
   generator enforces this geometrically in a canonical blocking patch.
2. Simultaneous same-tick collisions between different shapes are intentionally avoided in
   generated data. The solver still has a deterministic visible-coordinate fallback, but
   it is not relied upon by the dataset.
3. Optional extra shapes are isolated so they do not create ambiguous multi-way races.
"""

from __future__ import annotations

import math
import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

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


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
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


def _extract_edge_specs(g: np.ndarray) -> list[dict]:
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    specs: list[dict] = []

    for col in colors_present(g, exclude={bg}):
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == int(col))]
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
                    if v != bg and v != int(col):
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
                if v != bg and v != int(col):
                    carried.append((drn, dcn, v))

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        for edge_idx, ((r0, c0), (r1, c1)) in enumerate(edges):
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={int(col)}):
                continue

            path = list(_segment_cells(r0, c0, r1, c1))
            drs, dcs = _step_dir(r0, c0, r1, c1)
            shadows = []
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
                shadows.append(
                    {
                        "color": int(ncol),
                        "path": [(rr + drn, cc + dcn) for rr, cc in path],
                        "blocked": False,
                    }
                )

            specs.append(
                {
                    "color": int(col),
                    "path": path,
                    "progress": 0,
                    "blocked": False,
                    "done": len(path) <= 1,
                    "shadows": shadows,
                    "tie_key": (path[0][0], path[0][1], path[-1][0], path[-1][1], edge_idx),
                }
            )

    return specs


def _copy_specs(specs: list[dict]) -> list[dict]:
    copied = []
    for edge in specs:
        copied.append(
            {
                "color": int(edge["color"]),
                "path": list(edge["path"]),
                "progress": 0,
                "blocked": bool(edge["blocked"]),
                "done": bool(edge["done"]),
                "shadows": [
                    {
                        "color": int(sh["color"]),
                        "path": list(sh["path"]),
                        "blocked": bool(sh["blocked"]),
                    }
                    for sh in edge["shadows"]
                ],
                "tie_key": tuple(edge["tie_key"]),
            }
        )
    return copied


def _simulate_edges(g: np.ndarray, specs: list[dict], *, dynamic: bool) -> np.ndarray:
    g = np.asarray(g, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    if not dynamic:
        for edge in specs:
            (r0, c0), (r1, c1) = edge["path"][0], edge["path"][-1]
            _draw_line_bg_only(out, r0, c0, r1, c1, int(edge["color"]), bg)
        for edge in specs:
            for sh in edge["shadows"]:
                (r0, c0), (r1, c1) = sh["path"][0], sh["path"][-1]
                _draw_line_bg_only(out, r0, c0, r1, c1, int(sh["color"]), bg)
        return out

    state = _copy_specs(specs)

    while True:
        snapshot = out.copy()
        proposals: dict[tuple[int, int], list[int]] = {}
        pending = 0

        for idx, edge in enumerate(state):
            if edge["blocked"] or edge["done"]:
                continue
            if edge["progress"] >= len(edge["path"]) - 1:
                edge["done"] = True
                continue
            pending += 1
            rr, cc = edge["path"][edge["progress"] + 1]
            v = int(snapshot[rr, cc])
            if v == bg or v == int(edge["color"]):
                proposals.setdefault((rr, cc), []).append(idx)
            else:
                edge["blocked"] = True

        if pending == 0:
            break
        if not proposals:
            break

        winners: list[int] = []
        for cell, idxs in proposals.items():
            if len(idxs) == 1:
                winners.append(idxs[0])
                continue
            idxs = sorted(idxs, key=lambda i: state[i]["tie_key"])
            winners.append(idxs[0])
            for loser in idxs[1:]:
                state[loser]["blocked"] = True

        moved: list[int] = []
        for idx in sorted(set(winners), key=lambda i: state[i]["tie_key"]):
            edge = state[idx]
            if edge["blocked"] or edge["done"]:
                continue
            rr, cc = edge["path"][edge["progress"] + 1]
            v = int(out[rr, cc])
            if v != bg and v != int(edge["color"]):
                edge["blocked"] = True
                continue
            if v == bg:
                out[rr, cc] = int(edge["color"])
            edge["progress"] += 1
            if edge["progress"] >= len(edge["path"]) - 1:
                edge["done"] = True
            moved.append(idx)

        if not moved:
            break

        for idx in moved:
            edge = state[idx]
            prog = edge["progress"]
            for sh in edge["shadows"]:
                if sh["blocked"] or prog >= len(sh["path"]):
                    continue
                rr, cc = sh["path"][prog]
                v = int(out[rr, cc])
                if v == bg:
                    out[rr, cc] = int(sh["color"])
                elif v != int(sh["color"]):
                    sh["blocked"] = True

    return out


def _static_reference(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return _simulate_edges(g, _extract_edge_specs(g), dynamic=False)


def _dynamic_difference_count(grid: np.ndarray) -> int:
    dyn = main(grid)
    sta = _static_reference(grid)
    return int(np.count_nonzero(dyn != sta))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    specs = _extract_edge_specs(g)
    return _simulate_edges(g, specs, dynamic=True)


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8
GENERATE_RETRIES = 400
DISTRACTOR_PROB = 0.45

EXTRA_SHAPE_COUNT_WEIGHTS = np.array([0.45, 0.45, 0.10], dtype=float)
BLOCK_SHADOW_PROB = 0.7
EXTRA_SHADOW_PROB = 0.4

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
SHAPE_PLACEMENT_RETRIES = 40


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


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


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int]) -> None:
    dr, dc = offset
    h, w = out.shape
    src = out == shape_color
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _make_isolated_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    cells = _shape_cells(vertices)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _shape_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        _paint_shadow(out, shape_color, int(shadow_color), offset)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)
        nb_hits = sum(
            1
            for rr, cc in neighbors8(anchor[0], anchor[1])
            if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color
        )
        if nb_hits != 1:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _make_blocking_pair_patch(
    winner_color: int,
    loser_color: int,
    shadow_color: int | None,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    named = {
        "w0": (0, 4),
        "w1": (6, 4),
        "l0": (3, 0),
        "l1": (3, 8),
    }
    if shadow_color is not None:
        named["s0"] = (2, 0)
    named = _transform_named(named)

    pts = list(named.values())
    r0, c0, r1, c1 = bbox_coords(np.array(pts, dtype=int))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    wr0, wc0 = named["w0"][0] - r0, named["w0"][1] - c0
    wr1, wc1 = named["w1"][0] - r0, named["w1"][1] - c0
    lr0, lc0 = named["l0"][0] - r0, named["l0"][1] - c0
    lr1, lc1 = named["l1"][0] - r0, named["l1"][1] - c0
    inp[wr0, wc0] = int(winner_color)
    inp[wr1, wc1] = int(winner_color)
    inp[lr0, lc0] = int(loser_color)
    inp[lr1, lc1] = int(loser_color)
    if shadow_color is not None:
        sr, sc = named["s0"][0] - r0, named["s0"][1] - c0
        if inp[sr, sc] != BG:
            return None
        inp[sr, sc] = int(shadow_color)

    out = main(inp)
    static = _static_reference(inp)
    diff = int(np.count_nonzero(out != static))
    if diff <= 0:
        return None
    return inp, out, {"dynamic_gap": diff, "has_block_shadow": shadow_color is not None}


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
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    extra_shape_count = int(np.random.choice([0, 1, 2], p=EXTRA_SHAPE_COUNT_WEIGHTS / EXTRA_SHAPE_COUNT_WEIGHTS.sum()))
    has_block_shadow = bool(np.random.rand() < BLOCK_SHADOW_PROB)
    extra_shadow_flags = [bool(np.random.rand() < EXTRA_SHADOW_PROB) for _ in range(extra_shape_count)]
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)

    total_colors = 2 + int(has_block_shadow) + extra_shape_count + sum(extra_shadow_flags) + int(has_distractor)
    if total_colors > len(COLORS):
        return None

    palette = list(np.random.permutation(COLORS)[:total_colors])
    idx = 0
    winner_color = int(palette[idx])
    idx += 1
    loser_color = int(palette[idx])
    idx += 1
    block_shadow_color = int(palette[idx]) if has_block_shadow else None
    idx += int(has_block_shadow)

    block_patch = _make_blocking_pair_patch(winner_color, loser_color, block_shadow_color)
    if block_patch is None:
        return None
    if not _try_place(inp, out, block_patch[0], block_patch[1]):
        return None

    for use_shadow in extra_shadow_flags:
        shape_color = int(palette[idx])
        idx += 1
        shadow_color = int(palette[idx]) if use_shadow else None
        idx += int(use_shadow)
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_isolated_shape_patch(shape_color, shadow_color)
            if patch is None:
                continue
            if _try_place(inp, out, patch[0], patch[1]):
                placed = True
                break
        if not placed:
            return None

    if has_distractor:
        distractor_color = int(palette[idx])
        idx += 1
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if _dynamic_difference_count(inp) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
