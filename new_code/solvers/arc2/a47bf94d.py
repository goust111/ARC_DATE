import numpy as np
import heapq

from arc_utils import cc4 as _cc4, get_bg as _bg_color

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-neigh around center
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]  # diagonals + center
def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            # Reject if this 3x3 is part of a larger same-colored region.
            maximal = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for dr, dc in _N4:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                maximal = False
                                break
                    if not maximal:
                        break
                if not maximal:
                    break
            if not maximal:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 'X' patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])  # top-left must be colored for an X
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            # Ensure the X is isolated (no 4-neigh same-color outside the 3x3).
            isolated = True
            for dr0, dc0 in _X_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for dr, dc in _N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if not isolated:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return _cc4(mask)


def _route_to_opposite_endpoint(
    comp: set[tuple[int, int]],
    start_ep: tuple[int, int],
    dir_in: tuple[int, int],
    endpoints: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    """
    Route from a start endpoint to an endpoint on the *opposite* side of the component bbox.

    Components can contain cycles; a naive greedy walk can loop. To avoid that, we run
    a shortest-path search over state (cell, last_step_dir) where:
      - straight step cost = 0
      - turn cost = 1
      - reverse cost = 100

    We accept only goal states that arrive with `last_step_dir == dir_in`, which ensures
    the path exits the component in the intended direction.
    """
    mr, Mr, mc, Mc = bbox
    if dir_in == (1, 0):
        goals = {p for p in endpoints if p[0] == Mr}
    elif dir_in == (-1, 0):
        goals = {p for p in endpoints if p[0] == mr}
    elif dir_in == (0, 1):
        goals = {p for p in endpoints if p[1] == Mc}
    else:  # (0, -1)
        goals = {p for p in endpoints if p[1] == mc}
    if not goals:
        return None

    INF = 10**9
    dist: dict[tuple[tuple[int, int], tuple[int, int]], int] = { (start_ep, dir_in): 0 }
    pq: list[tuple[int, tuple[int, int], tuple[int, int]]] = [(0, start_ep, dir_in)]

    while pq:
        cost, (r, c), last_dir = heapq.heappop(pq)
        if cost != dist.get(((r, c), last_dir), INF):
            continue
        if (r, c) in goals and last_dir == dir_in:
            return (r, c)
        for dr, dc in _N4:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in comp:
                continue
            step_dir = (dr, dc)
            if step_dir == last_dir:
                w_cost = 0
            elif step_dir == (-last_dir[0], -last_dir[1]):
                w_cost = 100
            else:
                w_cost = 1
            ncost = cost + w_cost
            st = ((nr, nc), step_dir)
            if ncost < dist.get(st, INF):
                dist[st] = ncost
                heapq.heappush(pq, (ncost, (nr, nc), step_dir))

    return None


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
) -> tuple[tuple[int, int], tuple[int, int], bool]:
    """
    Greedy 'wire following': prefer going straight, otherwise turn (no immediate reverse).
    Returns (end_cell, last_direction, looped).
    """
    cur = start
    d = dir0
    prev = None
    seen: set[tuple[tuple[int, int], tuple[int, int], tuple[int, int] | None]] = set()
    for _ in range(h * w * 40):
        state = (cur, d, prev)
        if state in seen:
            return cur, d, True
        seen.add(state)

        r, c = cur
        opts = []
        for dr, dc in _N4:
            p = (r + dr, c + dc)
            if p in comp:
                opts.append((p, dr, dc))
        if not opts:
            return cur, d, False

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
            return cur, d, False

        prev = cur
        cur = p
        d = (dr, dc)

    return cur, d, True


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transmit colored tokens along wire networks, converting each to its complementary 3×3 pattern.

    Input structure:
    - bg = background. Non-bg cells form wire networks and 3×3 tokens (solid blocks or X/diagonal patterns) near wire endpoints.

    Rule:
    - Every solid 3×3 token is replaced in-place by a diamond (center + 4 cardinal neighbors).
    - Each token near a wire endpoint sends its complement to the opposite endpoint: solid emits X, X emits diamond.

    Key insight: Token shape cycles solid → X → diamond; wires route each complement to the far end.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    solids = _find_solid_3x3(g, bg)  # center -> (color, (r0,c0))
    xs = _find_x_3x3(g, bg)  # center -> (color, (r0,c0))

    # Build track mask: non-bg minus token cells.
    track_mask = g != bg
    for (cr, cc), (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for (cr, cc) in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)

    out = g.copy()

    # Solid tokens become diamonds in place.
    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = col

    # Route tokens across each track component.
    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
            continue

        comp_colors = {int(g[r, c]) for r, c in comp}
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        bbox = (min(rs), max(rs), min(cs), max(cs))

        # Degree within the component (4-neigh).
        deg = {p: 0 for p in comp}
        for r, c in comp:
            for dr, dc in _N4:
                if (r + dr, c + dc) in comp:
                    deg[(r, c)] += 1
        endpoints = [p for p in comp if deg[p] <= 1]
        if not endpoints:
            continue

        # For each endpoint, check if a token center exists 2 cells away in any direction.
        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue

                if cen in solids:
                    col, _ = solids[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd, looped = _trace_wire(comp, ep, dir_in, h, w)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    # If the greedy walk looped or the destination isn't empty background, fall back to routing.
                    if looped or any(
                        not (0 <= dest[0] + dr < h and 0 <= dest[1] + dc < w) or int(g[dest[0] + dr, dest[1] + dc]) != bg
                        for dr, dc in _X_OFF
                    ):
                        end2 = _route_to_opposite_endpoint(comp, ep, dir_in, endpoints, bbox)
                        if end2 is None:
                            continue
                        dest = (end2[0] + 2 * dir_in[0], end2[1] + 2 * dir_in[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd, looped = _trace_wire(comp, ep, dir_in, h, w)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    if looped or any(
                        not (0 <= dest[0] + dr < h and 0 <= dest[1] + dc < w) or int(g[dest[0] + dr, dest[1] + dc]) != bg
                        for dr, dc in _DIAMOND_OFF
                    ):
                        end2 = _route_to_opposite_endpoint(comp, ep, dir_in, endpoints, bbox)
                        if end2 is None:
                            continue
                        dest = (end2[0] + 2 * dir_in[0], end2[1] + 2 * dir_in[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out
