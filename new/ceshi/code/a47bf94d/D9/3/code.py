# Variant of a47bf94d
# Dimension: D9 — 形状循环关系 (transform)
# Change: 三形状循环: solid → (留十字, 发X) ; X → (留X, 发diamond) ; diamond → (已... → 可逆合并（反向操作）

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

# Design analysis for the D9 inversion variant.
#
# spec:
#   invariant substrate:
#     - Keep the original 8/5/9 wire language, top/bottom channel layout, permutation routing,
#       crossover rows, and embedded color-9 pass-through rectangles.
#     - Token colors still come from {1,2,3,4,6,7}, one color per routed channel.
#   visible change:
#     - Input no longer contains source solids. Instead every routed color appears twice:
#       one endpoint hosts an X token and the opposite endpoint hosts a diamond token.
#     - Output removes both endpoint fragments and creates a full 3x3 solid block of the same
#       color at a merge point near the middle of that route.
#   midpoint concretization:
#     - "Middle" is implemented as the route cell nearest the path midpoint whose 3x3 neighborhood
#       only touches that route and contains no 5/9 structure. This keeps the merge visible and
#       avoids covering crossings or maroon junction rectangles.
#
# behavior_trace:
#   - Original local rule on a single color route:
#       solid at one endpoint -> leave diamond there; emit X to the far endpoint.
#   - New local rule on the same route:
#       X at one endpoint + diamond at the far endpoint -> both are consumed; their route is traced
#       inward; a solid 3x3 appears at the route midpoint.
#   - Difference:
#       the old solver transformed one endpoint token and copied a complement outward; the new solver
#       requires a complementary pair and performs one inward aggregation event.
#
# complexity_calibration:
#   - Only two incomplete shapes participate, so the rule is learnable from 2-4 examples.
#   - Every sample repeats the same merge mechanic across 4-5 colors, which gives enough evidence.
#   - Side assignment of X vs diamond is randomized per color, so the task does not collapse to
#     "top always X, bottom always diamond".
#
# old_assumptions_that_break:
#   - Removing only solid/X token cells from the track mask is no longer enough; diamonds must be
#     excluded too or midpoint routing will absorb token pixels as if they were wire.
#   - The original unconditional "solid becomes local diamond" step is invalid because no local
#     remnant should survive after aggregation.
#   - The old far-end emission logic assumes a single source token per route; this variant instead
#     requires pairing two endpoint fragments of the same color and opposite kinds.
#
# placement_legality:
#   - Generator rejects layouts unless each routed path has a safe midpoint cell for a 3x3 solid.
#   - The chosen 3x3 merge squares must be in bounds, pairwise non-overlapping, and may overwrite
#     only background plus cells belonging to that same route.
#   - This guarantees the solver sees the intended endpoint pairings and can stamp clean outputs.
#
# visual_readability:
#   - The output change is obvious: fragments vanish from both extremes and a chunky 3x3 block
#     appears inside the wire network.
#   - By forbidding midpoint placement on 5/9 structures and foreign wires, the central result
#     remains visually distinct instead of becoming a cluttered pileup.


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            maximal = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        inside = r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3
                        if 0 <= nr < h and 0 <= nc < w and not inside and int(g[nr, nc]) == col:
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
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _X_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    inside = r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3
                    if 0 <= nr < h and 0 <= nc < w and not inside and int(g[nr, nc]) == col:
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
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 1])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _DIAMOND_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    inside = r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3
                    if 0 <= nr < h and 0 <= nc < w and not inside and int(g[nr, nc]) == col:
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
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    """Trace one deterministic route through a component and return its visited cells."""
    cur, d, prev = start, dir0, None
    path = [start]
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped — should not happen with 9-rect pass-through")
        visited.add((cur, d))

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
                if not others:
                    raise AssertionError("missing 9-rect exit")
                er, ec = cur
                def _face_score(face: str) -> tuple[int, str]:
                    pts = faces9[face]
                    if face in ("N", "S"):
                        return min(abs(p[1] - ec) for p in pts), face
                    return min(abs(p[0] - er) for p in pts), face
                exit_face = min(others, key=_face_score)
            pts = faces9[exit_face]
            er, ec = cur
            if entry_face in ("N", "S"):
                exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
            else:
                exit_cell = min(pts, key=lambda p: abs(p[0] - er))
            cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
            path.append(cur)
            jumped = True
            break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d, path

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
            return cur, d, path

        prev = cur
        cur = p
        d = (dr, dc)
        path.append(cur)

    raise AssertionError("wire trace exceeded step limit")


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _erase_token(grid: np.ndarray, center: tuple[int, int], offsets: list[tuple[int, int]], bg: int) -> None:
    cr, cc = center
    h, w = grid.shape
    for dr, dc in offsets:
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < h and 0 <= cc2 < w:
            grid[rr, cc2] = bg


def _token_offsets(kind: str) -> list[tuple[int, int]]:
    return _X_OFF if kind == "X" else _DIAMOND_OFF


def _choose_merge_center(
    path: list[tuple[int, int]],
    g: np.ndarray,
    bg: int,
) -> tuple[int, int] | None:
    """Pick the safest route cell nearest the path midpoint for the merged solid."""
    h, w = g.shape
    path_set = set(path)
    target = (len(path) - 1) / 2.0
    candidates = []
    for idx, (r, c) in enumerate(path):
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            continue
        if int(g[r, c]) != 8:
            continue
        ok = True
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                val = int(g[rr, cc])
                if val in (5, 9):
                    ok = False
                    break
                if val != bg and (rr, cc) not in path_set:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((abs(idx - target), idx, (r, c)))
    if candidates:
        return min(candidates)[2]

    fallback = []
    for idx, (r, c) in enumerate(path):
        if 1 <= r < h - 1 and 1 <= c < w - 1 and int(g[r, c]) == 8:
            fallback.append((abs(idx - target), idx, (r, c)))
    if fallback:
        return min(fallback)[2]
    return None


def _find_merge_plans(grid: np.ndarray) -> list[dict]:
    """Detect valid X+diamond endpoint pairs and compute their merge destinations."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    xs = _find_x_3x3(g, bg)
    diamonds = _find_diamond_3x3(g, bg)

    track_mask = g != bg
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            track_mask[cr + dr, cc + dc] = False
    for cr, cc in diamonds.keys():
        for dr, dc in _DIAMOND_OFF:
            track_mask[cr + dr, cc + dc] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)

    plans: list[dict] = []
    used_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()

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

        endpoint_tokens: dict[tuple[int, int], dict] = {}
        for ep in endpoints:
            er, ec = ep
            matches = []
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue
                if cen in xs:
                    matches.append({
                        "center": cen,
                        "kind": "X",
                        "color": xs[cen][0],
                        "dir_in": (-odr, -odc),
                    })
                elif cen in diamonds:
                    matches.append({
                        "center": cen,
                        "kind": "diamond",
                        "color": diamonds[cen][0],
                        "dir_in": (-odr, -odc),
                    })
            if len(matches) == 1:
                endpoint_tokens[ep] = matches[0]

        for ep, info in endpoint_tokens.items():
            end, _dd, _path = _trace_wire_path(comp, ep, info["dir_in"], h, w, rects_9)
            if end == ep or end not in endpoint_tokens:
                continue
            other = endpoint_tokens[end]
            if info["color"] != other["color"]:
                continue
            if {info["kind"], other["kind"]} != {"X", "diamond"}:
                continue

            key = tuple(sorted((ep, end)))
            if key in used_pairs:
                continue

            start_ep = key[0]
            start_info = endpoint_tokens[start_ep]
            dest_ep = key[1]
            traced_end, _dd2, path2 = _trace_wire_path(comp, start_ep, start_info["dir_in"], h, w, rects_9)
            if traced_end != dest_ep:
                continue

            merge_center = _choose_merge_center(path2, g, bg)
            if merge_center is None:
                continue

            plans.append({
                "color": info["color"],
                "merge_center": merge_center,
                "path": path2,
                "tokens": [
                    {"center": endpoint_tokens[key[0]]["center"], "kind": endpoint_tokens[key[0]]["kind"]},
                    {"center": endpoint_tokens[key[1]]["center"], "kind": endpoint_tokens[key[1]]["kind"]},
                ],
            })
            used_pairs.add(key)

    return plans


def _apply_merge_plans(grid: np.ndarray, plans: list[dict], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for plan in plans:
        for token in plan["tokens"]:
            _erase_token(out, token["center"], _token_offsets(token["kind"]), bg)
    for plan in plans:
        cr, cc = plan["merge_center"]
        _stamp_token(out, cr, cc, plan["color"], _SOLID_OFF)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge complementary endpoint fragments through wire networks into central solid blocks.

    Input structure:
    - Background cells are empty; non-background cells form the same wire substrate as the source task
      plus endpoint tokens that are now incomplete 3x3 shapes: X or diamond.
    - Each valid routed color appears twice, once at each extreme of the same wire path.

    Rule:
    - When the two endpoint tokens on a route are complementary fragments of the same color
      (one X and one diamond), both travel inward and disappear from the ends.
    - A full 3x3 solid block of that color is created at a safe midpoint on the connecting route.

    Key insight:
    - This is the inverse of the original shape-splitting dynamic: instead of one endpoint token
      decomposing and sending a complement outward, two partial shapes converge and recombine centrally.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    plans = _find_merge_plans(g)
    return _apply_merge_plans(g, plans, bg)


# --------------- generator (merge_inverse_midpoint) ---------------

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

GEN_MAX_RETRIES = 400


def _decompose_to_crossovers(perm):
    """Decompose a permutation into sequential crossover moves using a bypass column."""
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


def _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects):
    """Attempt to place one color-9 rectangular block. Returns (r0, c0, r1, c1) or None."""
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
        if not np.all(grid[rect_r0:rect_r1 + 1, c] == 8):
            return None

    sub = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1]
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

    saved = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1].copy()
    grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = 9

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
        grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = saved
        return None
    return rect_r0, rect_c0, rect_r1, rect_c1


def _build_wire_network(grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows):
    """Draw the wire network onto grid. Returns (slot_wire, bypass_wire) after all crossovers."""
    slot_wire = list(range(n))
    bypass_wire = None

    col_seg_start = {}
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
            if c not in col_seg_start:
                continue
            start = col_seg_start[c] + 1
            if start <= xrow - 1:
                grid[start:xrow, c] = 8

        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min:c_max + 1] = 8

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
        if slot_wire[j] is None:
            continue
        c = ch_cols[j]
        if c not in col_seg_start:
            continue
        start = col_seg_start[c] + 1
        if start <= bot_ep_row:
            grid[start:bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row):
    """Place one X and one diamond on each routed color pair."""
    inp = grid.copy()
    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        if random.choice([True, False]):
            _stamp_token(inp, top_center[0], top_center[1], col, _X_OFF)
            _stamp_token(inp, bot_center[0], bot_center[1], col, _DIAMOND_OFF)
        else:
            _stamp_token(inp, top_center[0], top_center[1], col, _DIAMOND_OFF)
            _stamp_token(inp, bot_center[0], bot_center[1], col, _X_OFF)
    return inp


def _solid_overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) <= 2 and abs(a[1] - b[1]) <= 2


def _gen_attempt():
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    w = max(bypass_col + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n)

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
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
    slot_wire, bypass_wire = _build_wire_network(grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows)

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if result is not None:
            placed_rects.append(result)

    inp = _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    plans = _find_merge_plans(inp)
    if len(plans) != n:
        return None

    centers = [plan["merge_center"] for plan in plans]
    if len(set(centers)) != n:
        return None
    for i in range(n):
        for j in range(i + 1, n):
            if _solid_overlaps(centers[i], centers[j]):
                return None

    expected = _apply_merge_plans(inp, plans, get_bg(inp))
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
