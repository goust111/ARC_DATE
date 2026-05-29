# Variant of a47bf94d
# Dimension: D7 — 传输目标与停止条件 (dst)
# Change: 管道对端端点（pipeline far endpoint）——token的互补形状传输到同一管道的另一端点，距端点2格处放置... → 碰撞停止

"""
Design analysis for the D7 collision-stop variant.

design:
  concrete_behavior:
    original:
      - A solid 3x3 source becomes a same-color diamond in place.
      - An X source stays in place.
      - The complementary token always appears at the opposite endpoint, with
        its center fixed 2 cells away from that endpoint.
    variant:
      - Keep the same wire graph, colors, token vocabulary, and 9-rect routing.
      - Make each endpoint ray support two visible landing slots:
        outer slot at gap=5 and inner slot at gap=2.
      - A transmitted complement first tries the outer slot. If that slot is
        already occupied by an existing token/diamond or by an earlier emitted
        complement, it stops in the inner slot instead.
      - This preserves the "same pipe, same opposite endpoint" idea, but D7 is
        now observable because identical routing can end at different visible
        distances from the endpoint.
    traced_example:
      - Suppose a top solid routes to a bottom endpoint whose outer slot is
        empty. Under the old rule the complement appears at the single fixed
        landing position; under the variant it reaches the outer slot.
      - If that bottom outer slot already contains a same-color resting
        diamond, the old rule would still write the complement at the far slot,
        while the variant leaves the blocker in place and stops the complement
        in the inner slot.
      - The visual difference is therefore not abstract routing metadata: it is
        a clear "far vs near" placement change along the endpoint ray.
  complexity_calibration:
    - Only two landing gaps are used, repeated across every grid, so humans can
      infer the rule from a few examples.
    - Input channels mix blocked and unblocked transmissions; otherwise the new
      rule would be invisible.
    - We keep the original solid/X/diamond shapes and the original wire syntax
      to avoid introducing unrelated ambiguity.
  implicit_assumptions_review:
    - The original solver assumed the destination center is always
      end + 2 * direction. That is no longer valid.
    - The original solver did not track pre-existing diamonds as blockers. The
      variant must detect them because generators now place them deliberately.
    - The original code iterated endpoint sets in hash order. Because stopping
      can depend on previously placed complements, this variant sorts connected
      components and endpoints explicitly to make the order deterministic.
    - Unconditional writes at the destination are no longer safe. Placement
      must check token occupancy before selecting outer vs inner slot.
  legality:
    - Endpoint rows are moved farther from the token bands so both landing slots
      fit fully inside the grid and do not overlap the wire.
    - Channel spacing remains 4-5 columns, which is enough for 3x3 tokens on
      adjacent channels.
    - The generator only places source tokens and blocker diamonds on the outer
      slot, so the solver can unambiguously interpret every input token as a
      source or a stationary blocker.
  observability:
    - The same color may appear as a source token on one side and a blocker
      diamond on the other; this gives a visible reason why some complements
      stop early and others travel farther.
    - Samples are rejected unless at least one transmission uses the inner stop
      slot and at least one uses the outer slot, and unless the collision-stop
      output differs from the fixed-far-slot baseline.
  visual_cleanliness:
    - Two landing slots per endpoint are enough to show the new behavior
      without filling the whole margin with queued copies.
    - The inner and outer slots are separated by 3 rows, so their 3x3 tokens
      remain distinct and readable.
  color_reasoning:
    - No color-based sorting or hidden color semantics are used.
    - Colors still only pair tokens with the routed wire endpoint; the variant
      changes distance-to-endpoint, not color identity.
"""

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

OUTER_SLOT_GAP = 5
INNER_SLOT_GAP = 2
SOURCE_GAPS = [OUTER_SLOT_GAP, INNER_SLOT_GAP]
LANDING_GAPS = [OUTER_SLOT_GAP, INNER_SLOT_GAP]


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col in (bg, 5, 8, 9) or np.any(sub != col):
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
    """Return {center: (color, (r0,c0))} for isolated 3x3 X patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col in (bg, 5, 8, 9):
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


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 diamond patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            vals = {int(sub[rr, cc]) for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]}
            if len(vals) != 1:
                continue
            col = vals.pop()
            if col in (bg, 5, 8, 9):
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _DIAMOND_OFF:
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
    """Find color-9 rectangular blocks and their wire connections per face."""
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
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in (5, 8):
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Follow one wire branch until its terminal endpoint."""
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped")
        visited.add((cur, d))

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                entry_face = _DIR2FACE[d]
                opp = _OPP[entry_face]
                if opp in faces9:
                    exit_face = opp
                else:
                    others = [f for f in faces9 if f != entry_face]
                    assert len(others) == 1
                    exit_face = others[0]
                pts = faces9[exit_face]
                er, ec = cur
                if entry_face in ("N", "S"):
                    exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
                else:
                    exit_cell = min(pts, key=lambda p: abs(p[0] - er))
                cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
                jumped = True
                break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d

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
            return cur, d

        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _shape_cells(center: tuple[int, int], offsets: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]] | None:
    cells = []
    cr, cc = center
    for dr, dc in offsets:
        rr, cc2 = cr + dr, cc + dc
        if not (0 <= rr < h and 0 <= cc2 < w):
            return None
        cells.append((rr, cc2))
    return cells


def _stamp_cells(out: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for rr, cc in cells:
        out[rr, cc] = color


def _find_source_for_endpoint(
    ep: tuple[int, int],
    solids: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    xs: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    h: int,
    w: int,
) -> tuple[str, int, tuple[int, int], tuple[int, int]] | None:
    er, ec = ep
    candidates = []
    for odr, odc in _N4:
        for gap in SOURCE_GAPS:
            cen = (er + gap * odr, ec + gap * odc)
            if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                continue
            if cen in solids:
                col, _ = solids[cen]
                candidates.append((gap, "solid", col, cen, (odr, odc)))
            elif cen in xs:
                col, _ = xs[cen]
                candidates.append((gap, "x", col, cen, (odr, odc)))
    if not candidates:
        return None
    gap, kind, col, cen, outward = max(candidates, key=lambda x: x[0])
    return kind, col, cen, outward


def _build_track_mask(
    g: np.ndarray,
    bg: int,
    solids: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    xs: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    diamonds: dict[tuple[int, int], tuple[int, tuple[int, int]]],
) -> np.ndarray:
    h, w = g.shape
    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False
    for cr, cc in diamonds.keys():
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False
    return track_mask


def _build_initial_output_and_occupancy(
    g: np.ndarray,
    bg: int,
    h: int,
    w: int,
    solids: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    xs: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    diamonds: dict[tuple[int, int], tuple[int, tuple[int, int]]],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = g.copy()
    occupied: set[tuple[int, int]] = set()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        cells = _shape_cells((cr, cc), _DIAMOND_OFF, h, w)
        assert cells is not None
        _stamp_cells(out, cells, col)
        occupied.update(cells)

    for (cr, cc), (col, _) in xs.items():
        cells = _shape_cells((cr, cc), _X_OFF, h, w)
        assert cells is not None
        occupied.update(cells)

    for (cr, cc), (col, _) in diamonds.items():
        cells = _shape_cells((cr, cc), _DIAMOND_OFF, h, w)
        assert cells is not None
        occupied.update(cells)

    return out, occupied


def _solve(
    grid: np.ndarray,
    *,
    collision_stop: bool,
    return_records: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    diamonds = _find_diamond_3x3(g, bg)

    track_mask = _build_track_mask(g, bg, solids, xs, diamonds)
    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    out, occupied = _build_initial_output_and_occupancy(g, bg, h, w, solids, xs, diamonds)
    records: list[dict[str, object]] = []

    for pts in sorted(comps, key=lambda comp: min(comp)):
        comp = set(pts)
        if len(comp) < 4:
            continue

        comp_colors = {int(g[r, c]) for r, c in comp}
        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = sorted(p for p in comp if deg[p] <= 1)
        if not endpoints:
            continue

        for ep in endpoints:
            source = _find_source_for_endpoint(ep, solids, xs, h, w)
            if source is None:
                continue
            kind, col, source_center, outward = source
            if col in comp_colors:
                continue
            dir_in = (-outward[0], -outward[1])
            end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)

            shape_off = _X_OFF if kind == "solid" else _DIAMOND_OFF
            chosen_gap = None
            chosen_cells = None
            gaps = LANDING_GAPS if collision_stop else [OUTER_SLOT_GAP]
            for gap in gaps:
                dest = (end[0] + gap * dd[0], end[1] + gap * dd[1])
                cells = _shape_cells(dest, shape_off, h, w)
                if cells is None:
                    continue
                if collision_stop and any(cell in occupied for cell in cells):
                    continue
                chosen_gap = gap
                chosen_cells = cells
                break

            if chosen_cells is not None:
                _stamp_cells(out, chosen_cells, col)
                occupied.update(chosen_cells)

            records.append(
                {
                    "source_kind": kind,
                    "color": col,
                    "source_center": source_center,
                    "endpoint": end,
                    "placed_gap": chosen_gap,
                    "blocked": chosen_gap != OUTER_SLOT_GAP,
                }
            )

    if return_records:
        return out, records
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route token complements through the wire network, but stop early when the far slot is already occupied.

    Input structure:
    - Non-background cells form the original teal/gray wire system plus 3x3 tokens near wire endpoints.
    - Source tokens are still solids or X patterns; resting diamonds may also already occupy endpoint slots.

    Rule:
    - Every solid source collapses into a diamond at its own location; every X source stays where it is.
    - The complementary shape travels to the routed opposite endpoint and tries to occupy the outer landing slot first.
    - If that outer slot already contains another token or an earlier emitted complement, the moving shape stops in the inner slot instead.

    Key insight:
    - D7 no longer means "always write the complement at one fixed far endpoint position"; the visible output depends on whether the far landing slot is free or blocked.
    """
    return _solve(grid, collision_stop=True)


# --------------- generator (collision-stop endpoint slots) ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOP_TOKEN_ROW = 2
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (22, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26

RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

X_SOURCE_COUNT_WEIGHTS = [35, 45, 20]
BLOCKED_CHANNEL_COUNT_WEIGHTS = [0, 35, 40, 25]
GEN_MAX_RETRIES = 400


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential crossover moves using the bypass column."""
    n = len(perm)
    visited = [False] * n
    crossovers = []
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
    n: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    target_wires = random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0]

    if target_wires >= 2 and random.random() < RECT_PARALLEL_PROB and n >= 2:
        mode = "parallel"
    elif target_wires >= 2 and xover_rows:
        mode = "crossing"
    else:
        mode = "single"

    if mode == "parallel":
        j0 = random.randint(0, n - 2)
        c_lo, c_hi = ch_cols[j0], ch_cols[j0 + 1]
        margin_l, margin_r = random.randint(0, 1), random.randint(0, 1)
    else:
        j0 = random.randint(0, n - 1)
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

    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + RECT_MARGIN_GAP and pr0 <= rect_r1 + RECT_MARGIN_GAP and rect_c0 <= pc1 + RECT_MARGIN_GAP and pc0 <= rect_c1 + RECT_MARGIN_GAP:
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
    n: int,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
) -> tuple[list[int | None], int | None]:
    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None

    col_seg_start: dict[int, int] = {}
    for j in range(n):
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

        active_cols = {ch_cols[j] for j in range(n) if slot_wire[j] is not None}
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

    for j in range(n):
        if slot_wire[j] is not None:
            c = ch_cols[j]
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= bot_ep_row:
                    grid[start : bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _make_input(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    bot_token_row: int,
) -> np.ndarray:
    inp = grid.copy()

    x_count = random.choices([0, 1, 2], weights=X_SOURCE_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(x_count, n)))

    blocked_count = random.choices([0, 1, 2, 3], weights=BLOCKED_CHANNEL_COUNT_WEIGHTS)[0]
    blocked_count = max(1, min(blocked_count, n - 1))
    blocked_channels = set(random.sample(range(n), blocked_count))

    sides = {i: random.choice(["top", "bottom"]) for i in range(n)}

    for i in range(n):
        col = token_colors[i]
        top_center = (TOP_TOKEN_ROW, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        src_center = top_center if sides[i] == "top" else bot_center
        far_center = bot_center if sides[i] == "top" else top_center

        if i in x_channels:
            _stamp_token(inp, src_center[0], src_center[1], col, _X_OFF)
        else:
            _stamp_token(inp, src_center[0], src_center[1], col, _SOLID_OFF)

        if i in blocked_channels:
            _stamp_token(inp, far_center[0], far_center[1], col, _DIAMOND_OFF)

    return inp


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    w = max(bypass_col + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_ep_row = TOP_TOKEN_ROW + OUTER_SLOT_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n)

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
            break

    crossovers = _decompose_to_crossovers(perm)
    if not crossovers:
        return None

    min_wire_rows = len(crossovers) * 2 + 2
    h_min = top_ep_row + min_wire_rows + TOP_TOKEN_ROW + OUTER_SLOT_GAP + 1
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = h - TOP_TOKEN_ROW - 1
    bot_ep_row = bot_token_row - OUTER_SLOT_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (len(crossovers) + 1))
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + s * 2 for s in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((h, w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows)

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        rect = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if rect is not None:
            placed_rects.append(rect)

    inp = _make_input(grid, n, ch_cols, perm, token_colors, bot_token_row)

    try:
        out, records = _solve(inp, collision_stop=True, return_records=True)
        baseline = _solve(inp, collision_stop=False)
    except Exception:
        return None

    blocked = sum(1 for rec in records if rec["blocked"])
    outer = sum(1 for rec in records if rec["placed_gap"] == OUTER_SLOT_GAP)
    if blocked < 1 or outer < 1:
        return None
    if np.array_equal(out, baseline):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
