# Combo variant of 35ab12c3 (index 9)
# combo_spec: D9/1+D13/0
# Name: 迭代扩展+干扰项障碍物
#
# Design analysis for this combo
# --------------------------------
# a) Concrete behavior and trace
#    Original 35ab12c3 treats each eligible edge statically: either the whole straight
#    edge is drawable from the sparse vertices, or it is skipped. This combo changes
#    two linked stages together.
#    - First, foreign-color distractor pixels become obstacle cells. A straight edge
#      that meets its first blocker does not vanish; it tries a fixed one-cell sidestep
#      and then continues on the parallel lane before rejoining the target endpoint.
#    - Second, these routed edges are not painted in one shot. Every edge advances only
#      one cell per tick, globally and simultaneously. Already-painted line cells and
#      obstacle cells can stop later progress.
#    Small trace used by the generator:
#      vertical winner: (0,4)->(6,4)
#      horizontal loser: (3,0)->(3,8)
#      obstacle: (3,3)
#      original task: obstacle would simply kill the whole horizontal edge.
#      obstacle-only routing: loser bends to row 2 (or row 4) and finishes there.
#      fused combo: loser bends, but while moving along the detour it later reaches a
#      cell already claimed by the vertical line, so it truncates mid-detour.
#    The visible difference is therefore "bend first, then race and stop", not merely
#    "skip edge" or "draw full detour".
#
# b) Complexity calibration
#    The detour rule is intentionally small: only the first blocker matters, sidestep
#    amount is exactly one cell, and only the two unit normals are tried. This keeps
#    the reroute human-readable from 2-4 examples.
#    To make the combo visible, every generated grid contains exactly one forced
#    obstacle+racing interaction. Optional extra shapes are isolated and follow the
#    original vocabulary, so the scene stays legible instead of becoming a maze.
#
# c) Implicit-assumption audit against the original solver
#    - Original assumption: `_clear_in_original(...)` decides an edge forever.
#      Broken: now it only decides whether the straight edge is obstacle-free; if not,
#      `_route_segment_in_original(...)` constructs a one-cell detour instead.
#    - Original assumption: drawing per color is harmless.
#      Broken: under iterative growth, later edges depend on earlier painted cells, so
#      the simulation must be global across all shapes. Per-color order is used only
#      for parsing, not for execution priority.
#    - Original assumption: every different-color neighbor of the chosen start vertex
#      is a shadow anchor.
#      Broken by obstacles: obstacle colors could otherwise be misread as shadows.
#      New rule: only singleton colors can be carried as shadow anchors.
#    - Original assumption: shadows can be painted as a separate late phase.
#      Broken: the combo attaches each shadow to its parent edge frontier and attempts
#      only the newly reached progress cell after the main cell lands.
#    - New assumption introduced here: true shadow colors are singleton groups and
#      obstacle pixels are never 8-neighboring shape vertices. The generator enforces
#      both, so parsing remains unambiguous.
#
# d) Placement legality
#    The generator builds local patches whose outputs are produced by the combo solver
#    itself, then places those output footprints on the global canvas with one-cell
#    padding. This guarantees in-bounds placement and prevents unrelated objects from
#    overlapping or accidentally participating in the forced race.
#
# e) Visual observability
#    The combo is inferable because the sparse input still shows the same object types
#    as the original task, but one color group of 1 pixel is now positioned directly in
#    a candidate corridor and the resulting output shows both a detour and a truncation.
#    No hidden color ranking is used; the only priorities are visible geometry and a
#    deterministic coordinate tie-break for rare same-tick collisions.
#
# f) Visual cleanliness
#    Only one contested region is forced per grid. Extra shapes, if any, are placed
#    using their final output footprints, so they remain visually separate. Shadows are
#    kept optional and sparse to avoid layered clutter near the blocking patch.
#
# g) Change inferability / color logic
#    Colors remain purely structural labels:
#    - one color per shape,
#    - optional singleton shadow color attached to one start vertex,
#    - one obstacle color for the distractor group.
#    There is no color remapping. The obstacle role is inferred from location and the
#    resulting routed/truncated geometry, not from the numeric color id.

from __future__ import annotations

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if int(out[rr, cc]) == bg:
            out[rr, cc] = int(col)


def _draw_path_bg_only(out: np.ndarray, path: list[tuple[int, int]], col: int, bg: int) -> None:
    for rr, cc in path:
        if int(out[rr, cc]) == bg:
            out[rr, cc] = int(col)


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _cycle_edges(cycle: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _color_counts(g: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(g, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _ordered_cycle_for_color(
    g: np.ndarray,
    col: int,
    bg: int,
    pts_rc: list[tuple[int, int]],
    color_counts: dict[int, int],
) -> list[tuple[int, int]]:
    cycle = _angle_cycle(pts_rc)
    if len(cycle) < 2:
        return cycle

    seed_idxs: list[int] = []
    h, w = g.shape
    for i, (r, c) in enumerate(cycle):
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col and color_counts.get(v, 0) == 1:
                    seed_idxs.append(i)
                    break

    if seed_idxs:
        start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
    else:
        start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
    return cycle[start:] + cycle[:start]


def _cell_is_clear(g: np.ndarray, r: int, c: int, bg: int, allow: set[int]) -> bool:
    v = int(g[r, c])
    return v == bg or v in allow


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        if not _cell_is_clear(g, rr, cc, bg, allow):
            return False
    return True


def _perp_offsets(dr: int, dc: int) -> list[tuple[int, int]]:
    return [(-dc, dr), (dc, -dr)]


def _first_blocker(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow: set[int],
) -> tuple[int, tuple[int, int]] | None:
    cells = list(_segment_cells(r0, c0, r1, c1))
    for idx, (rr, cc) in enumerate(cells[1:-1], start=1):
        if not _cell_is_clear(g, rr, cc, bg, allow):
            return idx, (rr, cc)
    return None


def _route_segment_in_original(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow: set[int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if not _aligned_8(r0, c0, r1, c1):
        return []

    block = _first_blocker(g, r0, c0, r1, c1, bg, allow)
    if block is None:
        return [((r0, c0), (r1, c1))]

    cells = list(_segment_cells(r0, c0, r1, c1))
    block_idx, _ = block
    prev = cells[block_idx - 1]
    drs, dcs = _step_dir(r0, c0, r1, c1)

    for nr, nc in _perp_offsets(drs, dcs):
        mid0 = (prev[0] + nr, prev[1] + nc)
        mid1 = (r1 + nr, c1 + nc)
        if not (
            0 <= mid0[0] < g.shape[0]
            and 0 <= mid0[1] < g.shape[1]
            and 0 <= mid1[0] < g.shape[0]
            and 0 <= mid1[1] < g.shape[1]
        ):
            continue
        if int(g[mid0]) != bg or int(g[mid1]) != bg:
            continue
        if not _clear_in_original(g, mid0[0], mid0[1], mid1[0], mid1[1], bg, allow=set()):
            continue
        routed: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if (r0, c0) != prev:
            routed.append(((r0, c0), prev))
        routed.append((prev, mid0))
        routed.append((mid0, mid1))
        routed.append((mid1, (r1, c1)))
        return routed

    if (r0, c0) != prev:
        return [((r0, c0), prev)]
    return []


def _segments_to_path(segments: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for a, b in segments:
        cells = list(_segment_cells(a[0], a[1], b[0], b[1]))
        if not path:
            path.extend(cells)
        else:
            path.extend(cells[1:])
    return path


def _shifted_segments_valid(
    g: np.ndarray,
    routed: list[tuple[tuple[int, int], tuple[int, int]]],
    offset: tuple[int, int],
    bg: int,
    shadow_color: int,
) -> tuple[bool, list[tuple[tuple[int, int], tuple[int, int]]]]:
    drn, dcn = offset
    shifted: list[tuple[tuple[int, int], tuple[int, int]]] = []
    h, w = g.shape
    for (r0, c0), (r1, c1) in routed:
        rr0, cc0 = r0 + drn, c0 + dcn
        rr1, cc1 = r1 + drn, c1 + dcn
        if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
            return False, []
        if not _aligned_8(rr0, cc0, rr1, cc1):
            return False, []
        if int(g[rr0, cc0]) not in {bg, shadow_color} or int(g[rr1, cc1]) not in {bg, shadow_color}:
            return False, []
        if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
            return False, []
        shifted.append(((rr0, cc0), (rr1, cc1)))
    return True, shifted


def _extract_edge_specs(g: np.ndarray, *, route_obstacles: bool) -> list[dict]:
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    color_counts = _color_counts(g)
    specs: list[dict] = []

    for col in colors_present(g, exclude={bg}):
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == int(col))]
        if len(pts_rc) < 2:
            continue

        cycle = _ordered_cycle_for_color(g, int(col), bg, pts_rc, color_counts)
        if len(cycle) < 2:
            continue

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != int(col) and color_counts.get(v, 0) == 1:
                    carried.append((drn, dcn, v))

        for edge_idx, ((r0, c0), (r1, c1)) in enumerate(_cycle_edges(cycle)):
            if not _aligned_8(r0, c0, r1, c1):
                continue

            if route_obstacles:
                routed = _route_segment_in_original(g, r0, c0, r1, c1, bg, allow={int(col)})
            else:
                routed = [((r0, c0), (r1, c1))]
            if not routed:
                continue

            path = _segments_to_path(routed)
            if len(path) < 2:
                continue

            shadows = []
            for drn, dcn, ncol in carried:
                ok, shadow_segments = _shifted_segments_valid(g, routed, (drn, dcn), bg, int(ncol))
                if not ok:
                    continue
                shadow_path = _segments_to_path(shadow_segments)
                if len(shadow_path) != len(path):
                    continue
                shadows.append(
                    {
                        "color": int(ncol),
                        "path": shadow_path,
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
                    "segments": routed,
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
                "segments": list(edge["segments"]),
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
            _draw_path_bg_only(out, edge["path"], int(edge["color"]), bg)
        for edge in specs:
            for sh in edge["shadows"]:
                _draw_path_bg_only(out, sh["path"], int(sh["color"]), bg)
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

        if pending == 0 or not proposals:
            break

        winners: list[int] = []
        for idxs in proposals.values():
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


def _static_routed_reference(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return _simulate_edges(g, _extract_edge_specs(g, route_obstacles=True), dynamic=False)


def _dynamic_straight_reference(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return _simulate_edges(g, _extract_edge_specs(g, route_obstacles=False), dynamic=True)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    specs = _extract_edge_specs(g, route_obstacles=True)
    combo = _simulate_edges(g, specs, dynamic=True)
    static_routed = _simulate_edges(g, specs, dynamic=False)
    straight_dynamic = _dynamic_straight_reference(g)
    detoured_edges = sum(1 for edge in specs if len(edge["segments"]) > 1)
    return {
        "dynamic_gap": int(np.count_nonzero(combo != static_routed)),
        "obstacle_gap": int(np.count_nonzero(combo != straight_dynamic)),
        "detoured_edges": int(detoured_edges),
    }


"""
Overview:
Connect same-color vertices as in the original task, but obstacle-colored distractors
force a one-cell detour and all routed edges grow globally one pixel at a time.

Input structure:
- Background 0.
- Each shape color appears as 2+ sparse vertices that define straight H/V/45deg edges.
- Some shapes have a singleton neighboring pixel of another color that marks a shadow
  offset from the chosen start vertex.
- One obstacle color may appear as 1-3 sparse distractor pixels; these are not aligned
  into their own shape and remain unchanged.

Rule:
1. For every shape edge, try the original straight segment.
2. If the first foreign interior pixel blocks that segment, reroute with a fixed
   one-cell sidestep: prefix, sidestep, parallel run, rejoin. If both sidesteps fail,
   keep only the clear prefix.
3. Simulate all resulting main paths simultaneously. Each unfinished edge advances by
   at most one cell per tick. Any existing non-background pixel already in the evolving
   output blocks further progress.
4. After a main edge lands its new cell, each of its valid shadow paths attempts the
   corresponding progress cell in parallel. Main cells therefore have local priority
   over their own shadows.

Key insight:
The obstacle pixels change where an edge wants to go, while the iterative global growth
changes how far that routed edge actually survives. The intended output is determined by
both effects together, not by either one alone.
"""
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    specs = _extract_edge_specs(g, route_obstacles=True)
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
SHAPE_PLACEMENT_RETRIES = 40

EXTRA_SHAPE_COUNT_WEIGHTS = np.array([0.45, 0.40, 0.15], dtype=float)
EXTRA_SHADOW_PROB = 0.45
BLOCK_PATCH_RETRIES = 60
BLOCK_PATCH_KINDS = ("orthogonal", "diagonal")
BLOCK_PATCH_KIND_WEIGHTS = np.array([0.3, 0.7], dtype=float)

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


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


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
    return _cycle_edges(vertices)


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
        inp[r, c] = int(shape_color)

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        _paint_shadow(out, int(shape_color), int(shadow_color), offset)
        sr, sc = anchor[0] + offset[0], anchor[1] + offset[1]
        if not (0 <= sr < h and 0 <= sc < w) or int(inp[sr, sc]) != BG:
            return None
        inp[sr, sc] = int(shadow_color)
        nb_hits = sum(
            1
            for rr, cc in neighbors8(anchor[0], anchor[1])
            if 0 <= rr < h and 0 <= cc < w and int(inp[rr, cc]) == int(shadow_color)
        )
        if nb_hits != 1:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _make_orthogonal_blocking_patch(
    winner_color: int,
    loser_color: int,
    obstacle_color: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    named = _transform_named(
        {
            "w0": (0, 4),
            "w1": (6, 4),
            "l0": (3, 0),
            "l1": (3, 8),
            "o0": (3, 3),
        }
    )

    pts = list(named.values())
    r0, c0, r1, c1 = bbox_coords(np.array(pts, dtype=int))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    wr0, wc0 = named["w0"][0] - r0, named["w0"][1] - c0
    wr1, wc1 = named["w1"][0] - r0, named["w1"][1] - c0
    lr0, lc0 = named["l0"][0] - r0, named["l0"][1] - c0
    lr1, lc1 = named["l1"][0] - r0, named["l1"][1] - c0
    or0, oc0 = named["o0"][0] - r0, named["o0"][1] - c0

    inp[wr0, wc0] = int(winner_color)
    inp[wr1, wc1] = int(winner_color)
    inp[lr0, lc0] = int(loser_color)
    inp[lr1, lc1] = int(loser_color)
    inp[or0, oc0] = int(obstacle_color)

    metrics = _behavior_metrics(inp)
    out = main(inp)
    if metrics["detoured_edges"] < 1:
        return None
    if metrics["dynamic_gap"] <= 0 or metrics["obstacle_gap"] <= 0:
        return None
    return inp, out


def _make_diagonal_blocking_patch(
    winner_color: int,
    loser_color: int,
    obstacle_color: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    named = _transform_named(
        {
            "l0": (0, 0),
            "l1": (4, 4),
            "w0": (0, 1),
            "w1": (6, 1),
            "o0": (2, 2),
        }
    )

    pts = list(named.values())
    r0, c0, r1, c1 = bbox_coords(np.array(pts, dtype=int))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    lr0, lc0 = named["l0"][0] - r0, named["l0"][1] - c0
    lr1, lc1 = named["l1"][0] - r0, named["l1"][1] - c0
    wr0, wc0 = named["w0"][0] - r0, named["w0"][1] - c0
    wr1, wc1 = named["w1"][0] - r0, named["w1"][1] - c0
    or0, oc0 = named["o0"][0] - r0, named["o0"][1] - c0

    inp[lr0, lc0] = int(loser_color)
    inp[lr1, lc1] = int(loser_color)
    inp[wr0, wc0] = int(winner_color)
    inp[wr1, wc1] = int(winner_color)
    inp[or0, oc0] = int(obstacle_color)

    metrics = _behavior_metrics(inp)
    out = main(inp)
    if metrics["detoured_edges"] < 1:
        return None
    if metrics["dynamic_gap"] <= 0 or metrics["obstacle_gap"] <= 0:
        return None
    return inp, out


def _make_blocking_obstacle_patch(
    winner_color: int,
    loser_color: int,
    obstacle_color: int,
    kind: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if kind == "orthogonal":
        return _make_orthogonal_blocking_patch(winner_color, loser_color, obstacle_color)
    if kind == "diagonal":
        return _make_diagonal_blocking_patch(winner_color, loser_color, obstacle_color)
    raise ValueError(f"unknown blocking patch kind: {kind}")


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
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    extra_shape_count = int(
        np.random.choice([0, 1, 2], p=EXTRA_SHAPE_COUNT_WEIGHTS / EXTRA_SHAPE_COUNT_WEIGHTS.sum())
    )
    extra_shadow_flags = [bool(np.random.rand() < EXTRA_SHADOW_PROB) for _ in range(extra_shape_count)]

    total_colors = 3 + extra_shape_count + sum(extra_shadow_flags)
    if total_colors > len(COLORS):
        return None

    palette = list(np.random.permutation(COLORS)[:total_colors])
    idx = 0
    winner_color = int(palette[idx])
    idx += 1
    loser_color = int(palette[idx])
    idx += 1
    obstacle_color = int(palette[idx])
    idx += 1

    block_patch = None
    block_kind = str(
        np.random.choice(
            BLOCK_PATCH_KINDS,
            p=BLOCK_PATCH_KIND_WEIGHTS / BLOCK_PATCH_KIND_WEIGHTS.sum(),
        )
    )
    for _ in range(BLOCK_PATCH_RETRIES):
        block_patch = _make_blocking_obstacle_patch(winner_color, loser_color, obstacle_color, block_kind)
        if block_patch is not None:
            break
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

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if not np.array_equal(main(inp), out):
        return None
    metrics = _behavior_metrics(inp)
    if metrics["detoured_edges"] < 1 or metrics["dynamic_gap"] <= 0 or metrics["obstacle_gap"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
