# Combo variant of a47bf94d (index 25)
# combo_spec: D3/3+D7/1
# Name: 颜色渐变轨迹+碰撞停止
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace example
# - The source task already has a clean notion of "send one complementary token
#   through one routed wire to the opposite endpoint", so the fusion keeps the
#   original wire topology and the original solid/X <-> diamond/X decomposition.
# - The D7/1 change is applied to the transport stage, not to the static wire
#   parsing stage. Tokens are processed in visible reading order of their 3x3
#   centers. Earlier processed tokens can therefore create blockers for later
#   ones.
# - The D3/3 change is fused with that stop rule instead of being added as an
#   independent decoration: the painted trajectory always uses the *full*
#   intended route as its denominator, but only the actually traversed prefix is
#   colored. Therefore the last visible band on a blocked route directly encodes
#   how far along the original route the token managed to travel.
# - Concrete trace on one color:
#   1. A top solid is converted locally into a diamond.
#   2. It traces its full path toward the routed bottom endpoint.
#   3. If that bottom center is empty, the whole path is painted with the fixed
#      gradient [source color -> 5 -> 9 -> source color] and an X appears at
#      the far endpoint.
#   4. If that bottom center already contains a stopped token, the same full
#      route is still known, but only the traversed prefix is painted; no new
#      far-end X appears. Any visible trail cells laid down by earlier routes
#      also act as blockers, so later tokens can stop partway through a shared
#      corridor instead of only at an endpoint.
# - Relative to the source task, the crucial difference is that a route can now
#   fail to materialize its complementary shape, and the painted path length is
#   informative rather than merely decorative.
#
# b) Complexity calibration
# - The gradient palette stays intentionally tiny: only two repeated structural
#   middle colors (5 and 9) are inserted between identical source/destination
#   token colors. This is enough for humans to see direction/progress without
#   needing a hidden numeric color ordering.
# - The generator always creates both outcomes in the same sample:
#   at least one successful long top->bottom transport and at least one blocked
#   route created by a bottom X already waiting at the paired destination.
# - Reading-order dependence is kept simple and repeated: all channels have top
#   solids, while only a subset also have bottom X sources. This makes the
#   "top travels first, then bottom gets blocked" pattern recur several times.
#
# c) Hidden assumptions from the original main() that no longer hold
# - Original order independence is broken. The old endpoint loop could iterate
#   over an arbitrary set order because all routes were independent. Here the
#   solver must collect source tokens explicitly and sort them by center.
# - Original "far endpoint always receives a complement" is false. A routed path
#   may now terminate without creating any new 3x3 token because the receiving
#   center is already occupied.
# - Original route painting used the realized path length as its only notion of
#   distance. That is insufficient here because a blocked route would then look
#   like a complete shorter route. The fused painter therefore bands colors by
#   the full intended route and only truncates the painted prefix.
# - Original `comp_colors` filtering is no longer valid because a color may
#   legitimately appear twice in the input: once as a top solid and once as the
#   same-color bottom X that blocks it. The fused solver therefore reasons by
#   geometry and occupancy, not by assuming one token occurrence per color.
# - Original output writing could ignore later state because successful arrivals
#   did not affect subsequent logic. Here successful and blocked routes both
#   change what later signals can traverse, so the solver must maintain explicit
#   `occupied_centers` and previously stopped visible trail cells.
#
# d) Placement legality
# - The generator preserves the source layout constraints: 4-5 channel columns,
#   4-5 spacing, token centers exactly two cells outside degree-1 endpoints, and
#   0-3 embedded color-9 rectangles that satisfy the original routing checks.
# - Bottom X blockers are only placed on a strict subset of routed destinations,
#   so no two 3x3 tokens overlap and at least one destination remains free for a
#   successful emission.
# - Because collisions are expressed through already-valid endpoint tokens and
#   already-valid wire cells, no new off-wire geometry is introduced and the
#   original margins remain sufficient.
#
# e) Visual observability
# - A human can compare channels inside one sample:
#   some top solids produce full gradient trails and far-end Xs,
#   while other equally shaped top solids produce the same kind of trail but no
#   far-end X because a bottom X is already present there.
# - The paired bottom X then often produces no outbound token at all, while some
#   later top routes truncate inside previously painted corridors. This gives
#   repeated evidence for "long route", "destination blocked", "mid-route stop",
#   and "reverse route immediately blocked".
#
# f) Visual cleanliness
# - The fused variant avoids introducing interior parked 3x3 obstacles, which
#   would clutter narrow wire corridors. Collisions happen on already-painted
#   wire cells or at existing endpoint slots, so the only extra visual signal is
#   the truncated gradient itself.
# - Ambiguous shared junction cells are still left structural, exactly as in the
#   trajectory-only variant, so crossing centers and 9-rect interiors do not
#   accumulate conflicting path colors.
#
# g) Color-change inferability
# - No hidden mapping based on numeric color ids is used. Each active route uses
#   the same visible band order [token color, 5, 9, token color].
# - The only thing that changes between channels is how much of that fixed band
#   sequence becomes visible before the route stops. That amount is tied to the
#   visibly occupied destination slot and to earlier painted trails, both
#   inferable from the grids themselves.
#
# h) Quantitative anti-degeneration target
# - Generated samples are rejected unless the fused solver reports:
#   at least one successful transport, at least one destination-blocked route,
#   at least one mid-route trail collision, at least one start-blocked reverse
#   route, and visible hits in both middle gradient bands. This prevents the
#   combo from collapsing back to either the source task or the pure D3-only
#   variant.
#
# i) New solver assumptions introduced here and enforced by the generator
# - Every active token is still attached to exactly one endpoint and has a clear
#   3x3 center two cells outside that endpoint.
# - Top solids are always present on all channels, so bottom X routes always have
#   a visible top-side blocker after the local solid->diamond rewrite.
# - Bottom X blockers appear on at least one but not all destinations, ensuring
#   the sample contains both successful and blocked transports.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]
_TRAJ_MID_PALETTE = [5, 9]
_WIRE_STRUCT_COLORS = {5, 8, 9}


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0, c0))} for maximal isolated solid 3x3 blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            maximal = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                maximal = False
                                break
                    if not maximal:
                        break
                if not maximal:
                    break
            if maximal:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0, c0))} for isolated 3x3 X-patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _X_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangles and the wire-adjacent cells exposed on each face."""
    mask9 = g == 9
    if not mask9.any():
        return []
    h, w = g.shape
    rects = []
    for comp9 in components_mask(mask9, connectivity=4):
        cells = set((int(r), int(c)) for r, c in comp9)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        faces: dict[str, list[tuple[int, int]]] = {}
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in cells or not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) not in (5, 8):
                    continue
                face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _trace_wire_path(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    """
    Follow one logical routed strand and return its ordered visible wire path.

    The path uses the same logical routing as the source task: color-9 rectangles
    are traversed as pass-through junctions, but their interiors are not expanded
    into large painted areas in the returned path.
    """
    cur, d, prev = start, dir0, None
    path = [start]
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for _ in range(h * w * 4):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped unexpectedly")
        visited.add(state)

        jumped = False
        for cells9, faces9 in rects_9:
            if cur not in cells9:
                continue
            entry_face = _DIR2FACE[d]
            opp = _OPP[entry_face]
            if opp in faces9:
                exit_face = opp
            else:
                others = [f for f in faces9 if f != entry_face]
                if len(others) != 1:
                    raise AssertionError("invalid 9-rect routing")
                exit_face = others[0]
            pts = faces9[exit_face]
            er, ec = cur
            if entry_face in ("N", "S"):
                exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
            else:
                exit_cell = min(pts, key=lambda p: abs(p[0] - er))
            cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
            if path[-1] != cur:
                path.append(cur)
            jumped = True
            break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return path, cur, d

        rev = (-d[0], -d[1])
        nonrev = [o for o in opts if (o[1], o[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [o for o in opts2 if (o[1], o[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            perp = [o for o in opts2 if o[1] * d[0] + o[2] * d[1] == 0]
            if perp:
                p, dr, dc = sorted(perp, key=lambda x: (x[1], x[2]))[0]
            else:
                p, dr, dc = opts2[0]

        if p == prev:
            return path, cur, d

        prev = cur
        cur = p
        d = (dr, dc)
        path.append(cur)

    raise AssertionError("wire trace exceeded step limit")


def _stamp_offsets(out: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    h, w = out.shape
    cr, cc = center
    for dr, dc in offsets:
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < h and 0 <= cc2 < w:
            out[rr, cc2] = color


def _paintable_path_cells(
    g: np.ndarray,
    full_path: list[tuple[int, int]],
    deg: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    return [p for p in full_path if int(g[p]) != 9 and deg.get(p, 0) <= 2]


def _paint_trajectory(
    out: np.ndarray,
    g: np.ndarray,
    full_path: list[tuple[int, int]],
    used_steps: int,
    deg: dict[tuple[int, int], int],
    color: int,
) -> dict[str, object]:
    """
    Paint the traversed prefix, but band it according to the full intended route.

    This is the core D3/3 + D7/1 interaction: a blocked route shows only an
    initial prefix of the same full-gradient scale, so its final visible color
    reveals the progress ratio instead of looking like a separate shorter route.
    """
    full_paintable = _paintable_path_cells(g, full_path, deg)
    used_paintable = _paintable_path_cells(g, full_path[:used_steps], deg)
    palette = [color, *_TRAJ_MID_PALETTE, color]
    band_hits = {mid: 0 for mid in _TRAJ_MID_PALETTE}

    n_full = len(full_paintable)
    if n_full == 0 or not used_paintable:
        return {"count": 0, "band_hits": band_hits, "full_count": n_full, "used_cells": used_paintable}

    last_band = len(palette) - 1
    for idx, (r, c) in enumerate(full_paintable[: len(used_paintable)]):
        band = 0 if n_full == 1 else int(idx * last_band / (n_full - 1))
        col = palette[band]
        out[r, c] = col
        if col in band_hits:
            band_hits[col] += 1
    return {"count": len(used_paintable), "band_hits": band_hits, "full_count": n_full, "used_cells": used_paintable}


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for (_cr, _cc), (_col, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    # Solids split locally before any transport, so their top-side diamonds are
    # already available as blockers for later bottom->top routes.
    for center, (col, (r0, c0)) in solids.items():
        _ = center
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        _stamp_offsets(out, (r0 + 1, c0 + 1), col, _DIAMOND_OFF)

    occupied_centers = set(solids.keys()) | set(xs.keys())
    blocked_trail_cells: set[tuple[int, int]] = set()

    comp_infos = []
    sources = []
    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
            continue
        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = sorted(p for p in comp if deg[p] <= 1)
        if not endpoints:
            continue
        comp_idx = len(comp_infos)
        comp_infos.append({"comp": comp, "deg": deg, "endpoints": endpoints})
        seen_centers: set[tuple[int, int]] = set()
        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue
                if cen in seen_centers:
                    continue
                if cen in solids:
                    col, _ = solids[cen]
                    kind = "solid"
                elif cen in xs:
                    col, _ = xs[cen]
                    kind = "x"
                else:
                    continue
                seen_centers.add(cen)
                sources.append(
                    {
                        "center": cen,
                        "color": col,
                        "kind": kind,
                        "endpoint": ep,
                        "dir_in": (-odr, -odc),
                        "comp_idx": comp_idx,
                    }
                )

    sources.sort(key=lambda src: (src["center"][0], src["center"][1], src["kind"]))

    route_stats = []
    for src in sources:
        info = comp_infos[src["comp_idx"]]
        path, end, dd = _trace_wire_path(info["comp"], src["endpoint"], src["dir_in"], h, w, rects_9)
        dest_center = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])

        blocked_idx = next((i for i, p in enumerate(path) if p in blocked_trail_cells), None)
        success = False
        stop_kind = "trail" if blocked_idx is not None else "dest"
        used_steps = 0

        if blocked_idx is not None:
            used_steps = blocked_idx
        else:
            used_steps = len(path)
            if dest_center not in occupied_centers:
                success = True
                stop_kind = "arrive"
                offsets = _X_OFF if src["kind"] == "solid" else _DIAMOND_OFF
                _stamp_offsets(out, dest_center, int(src["color"]), offsets)
                occupied_centers.add(dest_center)

        paint_meta = _paint_trajectory(out, g, path, used_steps, info["deg"], int(src["color"]))
        blocked_trail_cells.update(paint_meta["used_cells"])
        route_stats.append(
            {
                "kind": src["kind"],
                "color": int(src["color"]),
                "success": success,
                "stop_kind": stop_kind,
                "painted_cells": int(paint_meta["count"]),
                "full_cells": int(paint_meta["full_count"]),
                "used_fraction": 0.0
                if int(paint_meta["full_count"]) == 0
                else float(paint_meta["count"]) / float(paint_meta["full_count"]),
                "band_hits": paint_meta["band_hits"],
            }
        )

    if not return_meta:
        return out

    band_totals = {mid: 0 for mid in _TRAJ_MID_PALETTE}
    for stat in route_stats:
        for mid in _TRAJ_MID_PALETTE:
            band_totals[mid] += int(stat["band_hits"][mid])

    meta = {
        "route_count": len(route_stats),
        "successful_routes": sum(1 for stat in route_stats if stat["success"]),
        "blocked_routes": sum(1 for stat in route_stats if not stat["success"]),
        "dest_collision_routes": sum(1 for stat in route_stats if stat["stop_kind"] == "dest"),
        "trail_collision_routes": sum(1 for stat in route_stats if stat["stop_kind"] == "trail"),
        "partial_trail_routes": sum(
            1
            for stat in route_stats
            if stat["stop_kind"] == "trail" and 0 < int(stat["painted_cells"]) < int(stat["full_cells"])
        ),
        "start_blocked_routes": sum(
            1 for stat in route_stats if stat["stop_kind"] == "trail" and int(stat["painted_cells"]) == 0
        ),
        "trajectory_cells": sum(int(stat["painted_cells"]) for stat in route_stats),
        "band_totals": band_totals,
        "used_fractions": [float(stat["used_fraction"]) for stat in route_stats],
        "route_lengths": [int(stat["painted_cells"]) for stat in route_stats],
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route endpoint tokens through the wire network, but let visible trajectory progress and collision order interact.

    Input structure:
    - Background cells are empty.
    - Colors 5/8/9 form the same routed pipe network as in the source task, with
      optional color-9 rectangular junction blocks embedded in the wires.
    - Colored endpoint tokens are 3x3 solids or 3x3 X-shapes whose centers sit
      exactly two cells outside degree-1 wire endpoints. In this fused variant a
      color may appear on both sides of the same routed channel pair.

    Rule:
    - Every solid becomes an in-place diamond.
    - Each source token then tries to send its complementary shape along the
      routed wire in reading order of token centers: solid emits X, X emits
      diamond.
    - If the receiving center is free, the complementary shape appears there. If
      that center is already occupied by an earlier or pre-existing stopped token,
      the transport stops without creating a new far-end shape.
    - Any already-painted visible trajectory cell also blocks later transports,
      so a later signal can stop mid-route as soon as it reaches an earlier
      stopped trail.
    - The output also paints the actually traversed wire prefix with the fixed
      gradient [source color -> 5 -> 9 -> source color], but the gradient bands
      are indexed by the full intended route length. Therefore an early stop
      reveals partial progress by ending on an earlier gradient band.

    Key insight:
    - The old task's routing and shape complement logic still identify what each
      token *wants* to do; the new thing is that route order matters. Earlier
      stopped tokens can deny later arrivals, and the truncated gradient shows
      how far the later signal got before that collision happened.
    """
    return _solve(grid)


# --------------- generator (gradient_trajectory_collision_stop) ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

BOTTOM_X_COUNT_WEIGHTS = [65, 35]  # choose 1 or 2 blockers, later clamped to N-1
GEN_MAX_RETRIES = 600
MIN_TRAJECTORY_CELLS = 18


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using the bypass column."""
    n = len(perm)
    visited = [False] * n
    crossovers: list[tuple[int, int]] = []
    for start in range(n):
        if visited[start] or perm[start] == start:
            visited[start] = True
            continue
        cycle = []
        i = start
        while not visited[i]:
            visited[i] = True
            cycle.append(i)
            i = perm[i]
        if len(cycle) < 2:
            continue
        crossovers.append((cycle[0], -1))
        for k in range(len(cycle) - 1, 0, -1):
            crossovers.append((cycle[k], cycle[(k + 1) % len(cycle)]))
        crossovers.append((-1, cycle[1]))
    return crossovers


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n_channels: int,
    placed_rects: list[tuple[int, int, int, int]],
):
    """Attempt to place one valid color-9 rectangle inside existing wire columns."""
    h, w = grid.shape
    target_wires = random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0]

    if target_wires >= 2 and random.random() < RECT_PARALLEL_PROB and n_channels >= 2:
        mode = "parallel"
    elif target_wires >= 2 and xover_rows:
        mode = "crossing"
    else:
        mode = "single"

    if mode == "parallel":
        j0 = random.randint(0, n_channels - 2)
        c_lo, c_hi = ch_cols[j0], ch_cols[j0 + 1]
        margin_l, margin_r = random.randint(0, 1), random.randint(0, 1)
    else:
        j0 = random.randint(0, n_channels - 1)
        c_lo = c_hi = ch_cols[j0]
        margin_l, margin_r = random.randint(0, 2), random.randint(0, 2)
        if margin_l + margin_r < 1:
            if random.random() < 0.5:
                margin_l = 1
            else:
                margin_r = 1
    rect_c0, rect_c1 = c_lo - margin_l, c_hi + margin_r
    if rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH or rect_c0 < 0 or rect_c1 >= w:
        return None

    rect_h = random.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4))
    if mode == "crossing" and xover_rows:
        xr = random.choice(xover_rows)
        r_start = xr - random.randint(0, rect_h - 1)
    else:
        r_start = random.randint(top_ep_row + 2, bot_ep_row - rect_h - 1)
    rect_r0, rect_r1 = r_start, r_start + rect_h - 1

    wire_cols = sorted(c for c in ch_cols if rect_c0 <= c <= rect_c1)
    if not wire_cols:
        return None
    for c in wire_cols:
        if not np.all(grid[rect_r0 : rect_r1 + 1, c] == 8):
            return None

    sub = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    if rect_r0 <= top_ep_row or rect_r1 >= bot_ep_row:
        return None
    for c in wire_cols:
        if grid[rect_r0 - 1, c] != 8 or grid[rect_r1 + 1, c] != 8:
            return None

    saved = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1].copy()
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = 9

    n_cols, s_cols, w_rows, e_rows = set(), set(), set(), set()
    for c in range(rect_c0, rect_c1 + 1):
        if rect_r0 - 1 >= 0 and grid[rect_r0 - 1, c] in (5, 8):
            n_cols.add(c)
        if rect_r1 + 1 < h and grid[rect_r1 + 1, c] in (5, 8):
            s_cols.add(c)
    for r in range(rect_r0, rect_r1 + 1):
        if rect_c0 - 1 >= 0 and grid[r, rect_c0 - 1] in (5, 8):
            w_rows.add(r)
        if rect_c1 + 1 < w and grid[r, rect_c1 + 1] in (5, 8):
            e_rows.add(r)
    total_wires = len(n_cols & s_cols) + len(w_rows & e_rows)

    if total_wires < 1 or total_wires > 3:
        grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = saved
        return None
    return (rect_r0, rect_c0, rect_r1, rect_c1)


def _build_wire_network(
    grid: np.ndarray,
    n_channels: int,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
) -> tuple[list[int | None], int | None]:
    """Draw the routed wire network and return the final wire->source mapping."""
    slot_wire: list[int | None] = list(range(n_channels))
    bypass_wire: int | None = None

    col_seg_start: dict[int, int] = {}
    for j in range(n_channels):
        col_seg_start[ch_cols[j]] = top_ep_row
        grid[top_ep_row, ch_cols[j]] = 8

    for xover_idx, (src_slot, dst_slot) in enumerate(crossovers):
        xrow = xover_rows[xover_idx]

        if src_slot == -1:
            src_col, dst_col = bypass_col, ch_cols[dst_slot]
        elif dst_slot == -1:
            src_col, dst_col = ch_cols[src_slot], bypass_col
        else:
            src_col, dst_col = ch_cols[src_slot], ch_cols[dst_slot]

        active_cols = {ch_cols[j] for j in range(n_channels) if slot_wire[j] is not None}
        if bypass_wire is not None:
            active_cols.add(bypass_col)

        for c in active_cols:
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= xrow - 1:
                    grid[start:xrow, c] = 8

        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min : c_max + 1] = 8
        for c in active_cols:
            grid[xrow, c] = 8

        for c in active_cols:
            if c == src_col or c == dst_col:
                continue
            if c_min < c < c_max:
                if c - 1 >= c_min:
                    grid[xrow, c - 1] = 5
                if c + 1 <= c_max:
                    grid[xrow, c + 1] = 5

        new_seg_start = {c: xrow for c in active_cols if c != src_col}
        new_seg_start[dst_col] = xrow
        col_seg_start = new_seg_start

        if src_slot == -1:
            slot_wire[dst_slot] = bypass_wire
            bypass_wire = None
        elif dst_slot == -1:
            bypass_wire = slot_wire[src_slot]
            slot_wire[src_slot] = None
        else:
            slot_wire[dst_slot] = slot_wire[src_slot]
            slot_wire[src_slot] = None

    for j in range(n_channels):
        if slot_wire[j] is not None:
            c = ch_cols[j]
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= bot_ep_row:
                    grid[start : bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _derive_input(
    grid: np.ndarray,
    n_channels: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> tuple[np.ndarray, set[int]]:
    """
    Build one input:
    - top solids on every channel
    - bottom X blockers on a strict subset of routed destinations
    """
    inp = grid.copy()
    max_blockers = max(1, min(2, n_channels - 1))
    n_bottom_x = random.choices([1, 2], weights=BOTTOM_X_COUNT_WEIGHTS)[0]
    n_bottom_x = min(n_bottom_x, max_blockers)
    blocked_sources = set(random.sample(range(n_channels), n_bottom_x))

    for i in range(n_channels):
        _stamp_token(inp, top_token_row, ch_cols[i], token_colors[i], _SOLID_OFF)

    for i in blocked_sources:
        _stamp_token(inp, bot_token_row, ch_cols[perm[i]], token_colors[i], _X_OFF)

    return inp, blocked_sources


def _combo_visible(meta: dict[str, object], n_channels: int, blocked_sources: set[int]) -> bool:
    expected_routes = n_channels + len(blocked_sources)
    if int(meta["route_count"]) != expected_routes:
        return False
    if int(meta["successful_routes"]) < 1:
        return False
    if int(meta["dest_collision_routes"]) < 1:
        return False
    if int(meta["trail_collision_routes"]) < 1:
        return False
    if int(meta["partial_trail_routes"]) < 1:
        return False
    if int(meta["start_blocked_routes"]) < 1:
        return False
    if int(meta["trajectory_cells"]) < MIN_TRAJECTORY_CELLS:
        return False
    band_totals = meta["band_totals"]
    if any(int(band_totals[mid]) == 0 for mid in _TRAJ_MID_PALETTE):
        return False
    used_fractions = [float(x) for x in meta["used_fractions"]]
    if not any(frac == 0.0 for frac in used_fractions):
        return False
    if not any(0.0 < frac < 0.95 for frac in used_fractions):
        return False
    if not any(0.95 <= frac <= 1.0 for frac in used_fractions):
        return False
    return True


def _gen_attempt():
    n_channels = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n_channels)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    w = max(bypass_col + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n_channels)

    while True:
        perm = list(range(n_channels))
        random.shuffle(perm)
        if perm != list(range(n_channels)):
            break

    crossovers = _decompose_to_crossovers(perm)
    n_xovers = len(crossovers)
    if n_xovers == 0:
        return None

    min_wire_rows = n_xovers * 2 + 2
    h_min = top_ep_row + min_wire_rows + top_token_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (n_xovers + 1))
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + s * 2 for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((h, w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        grid, n_channels, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
    )

    inv_perm = [0] * n_channels
    for i in range(n_channels):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n_channels)) or bypass_wire is not None:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n_channels, placed_rects)
        if result is not None:
            placed_rects.append(result)

    inp, blocked_sources = _derive_input(grid, n_channels, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        _out, meta = _solve(inp, return_meta=True)
    except Exception:
        return None
    if not _combo_visible(meta, n_channels, blocked_sources):
        return None
    return inp


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
