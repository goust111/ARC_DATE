# Combo variant of a47bf94d (index 26)
# combo_spec: D14/0+D20/1
# Name: 管道颜色配对+多子网络

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import components_mask, get_bg, neighbors4
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the combo variant
# - concretization and behavior trace:
#   take two channels inside one local subnetwork. In the original task both wire
#   paths were cyan and the solver used "token color is absent from the wire" to
#   decide what to route. In this combo, the left channel's pipe is drawn in the
#   left token's color and the right channel's pipe is drawn in the right token's
#   color, while grey crossing markers and maroon 9-rects stay neutral. A solid
#   sender on one endpoint still becomes a local diamond and emits an X to the
#   far endpoint; an X sender still emits a diamond. The difference is that the
#   endpoint cell touching a real sender now has the same color as that sender,
#   which is the new visible pairing cue.
# - complexity calibration:
#   the change is visible but not trivialized because the scene still contains
#   shared topology: each subnetwork may contain multiple colored routes that
#   cross or pass through neutral 9-rect blocks. The generator always creates
#   2-3 disconnected subnetworks and requires at least one component to carry
#   more than one pipe color, so the task does not collapse into "one isolated
#   pipe per channel".
# - hidden-assumption audit against the original solver:
#   `if col in comp_colors: continue` was valid only when wires never used token
#   colors; it is removed here. Instead the solver checks `grid[endpoint] ==
#   token_color`, because the generator guarantees that the degree-1 pipe cell
#   attached to a true sender uses that sender's color. The greedy path tracer
#   itself still works because routing is encoded by topology plus neutral 5/9
#   markers, not by the old uniform wire color.
# - placement legality:
#   every token center remains exactly two cells from a degree-1 endpoint; group
#   columns are separated enough that subnetworks do not touch; local permutations
#   are generated independently per group; bottom destinations stay unique inside
#   each group; optional 9-rects are only placed strictly between endpoint rows.
# - observability and cleanliness:
#   from 2-4 examples a human can see two simultaneous regularities: pipes that
#   share a color belong to the same route, and the full scene is split into 2-3
#   disconnected subnetworks. Neutral grey/maroon markers remain sparse, so the
#   added color richness does not bury the core endpoint/token relation.
# - generator-enforced new assumptions:
#   every real sender touches a same-colored endpoint cell, every track component
#   has at least one degree-1 endpoint, and at least one component is multi-color.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _is_pipe_cell(v: int, bg: int = 0) -> bool:
    return int(v) not in (bg, 5, 9)


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for solid 3x3 blocks, allowing one wire attachment."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            allowed = {
                (r0 - 1, c0 + 1),
                (r0 + 3, c0 + 1),
                (r0 + 1, c0 - 1),
                (r0 + 1, c0 + 3),
            }
            outside_same: set[tuple[int, int]] = set()
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                outside_same.add((nr, nc))
            if outside_same and (len(outside_same) > 1 or not outside_same.issubset(allowed)):
                continue
            if len(outside_same) <= 1:
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


def _find_9rects(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangles and the non-9 track cells touching each face."""
    mask9 = g == 9
    if not mask9.any():
        return []
    h, w = g.shape
    rects: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]] = []
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
                val = int(g[nr, nc])
                if val == bg or val == 9:
                    continue
                face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Follow one route through a mixed-color component.

    Neutral 5-cells and 9-rects preserve the old routing semantics; the current
    pipe color is not needed once the correct endpoint has been identified.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped")
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


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform sender tokens locally and route their complementary shape through colored subnetworks.

    Input structure:
    - Background cells are empty.
    - Pipe cells use the same color as the token they transport, while grey 5-cells and maroon 9-rects remain neutral routing aids.
    - The full network is split into 2-3 disconnected 4-connected subnetworks, and each sender token sits two cells away from a degree-1 endpoint of its matching-colored route.

    Rule:
    - A solid 3x3 sender turns into a local diamond and emits an X at the opposite endpoint of its route.
    - An X sender stays in place and emits a diamond at the opposite endpoint.
    - The route to follow is the one whose touching endpoint cell has the same color as the sender token; neutral 5/9 structures only guide the path.

    Key insight:
    - The old "token color is absent from the wires" cue is gone. Instead, color directly identifies the route channel, and disconnected subnetworks mean you first segment the scene into local routing problems, then follow each endpoint through the shared topology of that subnetwork.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g, bg)
    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = col

    for pts in comps:
        comp = set(pts)
        if len(comp) < 2:
            continue

        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = [p for p in comp if deg[p] <= 1]
        if not endpoints:
            continue

        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue
                dir_in = (-odr, -odc)

                if cen in solids:
                    col, _ = solids[cen]
                    if int(g[ep]) != col:
                        continue
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    if int(g[ep]) != col:
                        continue
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --------------- generator ---------------

CHANNEL_COUNTS = [4, 5, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 30
EP_TOKEN_GAP = 2

GROUP_PATTERNS = {
    4: [[2, 2]],
    5: [[2, 2, 1], [2, 1, 2], [1, 2, 2]],
}

RECT_COUNT_WEIGHTS = [55, 35, 10]
RECT_WIRE_COUNT_WEIGHTS = [55, 35, 10]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 4)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 6
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [35, 40, 25]
FAR_DIAMOND_PROB = 0.33
GEN_MAX_RETRIES = 800


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a local permutation into original-style crossover operations."""
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


def _random_nonidentity_perm(n: int) -> list[int]:
    if n == 1:
        return [0]
    if n == 2:
        return [1, 0]
    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
            return perm


def _assign_xover_rows(top_ep_row: int, bot_ep_row: int, n_xovers: int) -> list[int] | None:
    if n_xovers == 0:
        return []
    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (n_xovers + 1))
    rows = [top_ep_row + 1 + step * (s + 1) for s in range(n_xovers)]
    if rows[-1] >= bot_ep_row:
        rows = [top_ep_row + 2 + 2 * s for s in range(n_xovers)]
    if rows[-1] >= bot_ep_row:
        return None
    return rows


def _column_to_wire(ch_cols: list[int], slot_wire: list[int | None], bypass_col: int, bypass_wire: int | None) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for j, wid in enumerate(slot_wire):
        if wid is not None:
            mapping[ch_cols[j]] = wid
    if bypass_wire is not None:
        mapping[bypass_col] = bypass_wire
    return mapping


def _build_colored_network(
    grid: np.ndarray,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
    wire_colors: list[int],
) -> tuple[list[int | None], int | None]:
    """Draw one disconnected local subnetwork using route colors instead of color 8."""
    n = len(ch_cols)
    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None
    col_seg_start = {c: top_ep_row for c in ch_cols}
    for j, c in enumerate(ch_cols):
        grid[top_ep_row, c] = wire_colors[j]

    for xrow, (src_slot, dst_slot) in zip(xover_rows, crossovers):
        col_to_wire = _column_to_wire(ch_cols, slot_wire, bypass_col, bypass_wire)
        for c, wid in col_to_wire.items():
            start = col_seg_start[c] + 1
            if start <= xrow - 1:
                grid[start:xrow, c] = wire_colors[wid]

        if src_slot == -1:
            src_col, moving_wire = bypass_col, bypass_wire
        else:
            src_col, moving_wire = ch_cols[src_slot], slot_wire[src_slot]
        if dst_slot == -1:
            dst_col = bypass_col
        else:
            dst_col = ch_cols[dst_slot]
        if moving_wire is None:
            raise AssertionError("invalid moving wire")

        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min : c_max + 1] = wire_colors[moving_wire]

        for c, wid in col_to_wire.items():
            if c < c_min or c > c_max or (c != src_col and c != dst_col):
                grid[xrow, c] = wire_colors[wid]

        for c in col_to_wire:
            if c == src_col or c == dst_col or not (c_min < c < c_max):
                continue
            if c - 1 >= c_min:
                grid[xrow, c - 1] = 5
            if c + 1 <= c_max:
                grid[xrow, c + 1] = 5

        new_seg_start = {c: xrow for c in col_to_wire if c != src_col}
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

    final_map = _column_to_wire(ch_cols, slot_wire, bypass_col, bypass_wire)
    for c, wid in final_map.items():
        if c == bypass_col:
            continue
        start = col_seg_start.get(c, top_ep_row) + 1
        if start <= bot_ep_row:
            grid[start : bot_ep_row + 1, c] = wire_colors[wid]
    return slot_wire, bypass_wire


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one neutral 9-rect inside a colored subnetwork."""
    h, w = grid.shape
    target_wires = random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0]
    if target_wires >= 2 and random.random() < RECT_PARALLEL_PROB and len(ch_cols) >= 2:
        mode = "parallel"
    elif target_wires >= 2 and xover_rows:
        mode = "crossing"
    else:
        mode = "single"

    if mode == "parallel":
        j0 = random.randint(0, len(ch_cols) - 2)
        c_lo, c_hi = ch_cols[j0], ch_cols[j0 + 1]
        margin_l, margin_r = random.randint(0, 1), random.randint(0, 1)
    else:
        j0 = random.randint(0, len(ch_cols) - 1)
        c_lo = c_hi = ch_cols[j0]
        margin_l, margin_r = random.randint(0, 2), random.randint(0, 2)
        if margin_l + margin_r < 1:
            margin_l = 1 if random.random() < 0.5 else 0
            margin_r = 0 if margin_l else 1

    rect_c0, rect_c1 = c_lo - margin_l, c_hi + margin_r
    if rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH or rect_c0 < 0 or rect_c1 >= w:
        return None

    max_h = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4)
    if max_h < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], max_h)
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
        col_vals = grid[rect_r0 : rect_r1 + 1, c]
        if not np.all([_is_pipe_cell(int(v)) for v in col_vals]):
            return None

    sub = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
    if not np.all([(int(v) == 0) or (int(v) == 5) or _is_pipe_cell(int(v)) for v in sub.flat]):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    if rect_r0 <= top_ep_row or rect_r1 >= bot_ep_row:
        return None
    for c in wire_cols:
        if not _is_pipe_cell(int(grid[rect_r0 - 1, c])) or not _is_pipe_cell(int(grid[rect_r1 + 1, c])):
            return None

    saved = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1].copy()
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = 9

    n_cols, s_cols, w_rows, e_rows = set(), set(), set(), set()
    for c in range(rect_c0, rect_c1 + 1):
        if rect_r0 - 1 >= 0 and int(grid[rect_r0 - 1, c]) != 0:
            n_cols.add(c)
        if rect_r1 + 1 < h and int(grid[rect_r1 + 1, c]) != 0:
            s_cols.add(c)
    for r in range(rect_r0, rect_r1 + 1):
        if rect_c0 - 1 >= 0 and int(grid[r, rect_c0 - 1]) != 0:
            w_rows.add(r)
        if rect_c1 + 1 < w and int(grid[r, rect_c1 + 1]) != 0:
            e_rows.add(r)
    total_wires = len(n_cols & s_cols) + len(w_rows & e_rows)

    if total_wires < 1 or total_wires > 3:
        grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = saved
        return None
    return rect_r0, rect_c0, rect_r1, rect_c1


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """Measure whether the generated input actually exhibits the combo behavior."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < g.shape[0] and 0 <= cc2 < g.shape[1]:
                track_mask[rr, cc2] = False
    # Ignore isolated clue cells such as optional far-end diamonds: they are
    # not routing subnetworks and should not erase D20 coverage during filtering.
    comps = [set(pts) for pts in _components(track_mask) if len(pts) >= 2]
    multicolor = 0
    max_pipe_colors = 0
    for comp in comps:
        cols = {int(g[r, c]) for r, c in comp if _is_pipe_cell(int(g[r, c]), bg)}
        max_pipe_colors = max(max_pipe_colors, len(cols))
        if len(cols) >= 2:
            multicolor += 1
    return {
        "subnetwork_count": len(comps),
        "multicolor_component_count": multicolor,
        "max_pipe_colors_in_component": max_pipe_colors,
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)
    left_margin = random.randint(*LEFT_MARGIN_RANGE)

    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    width = max(ch_cols[-1] + 5, GRID_W_MIN)
    if width > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n)
    group_sizes = random.choice(GROUP_PATTERNS[n])

    groups: list[dict] = []
    start = 0
    max_wire_rows = 2
    for size in group_sizes:
        idxs = list(range(start, start + size))
        start += size
        perm = _random_nonidentity_perm(size)
        crossovers = _decompose_to_crossovers(perm)
        max_wire_rows = max(max_wire_rows, len(crossovers) * 2 + 2)
        groups.append({"indices": idxs, "perm": perm, "crossovers": crossovers})

    h_min = top_ep_row + max_wire_rows + top_token_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    height = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = height - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < max_wire_rows:
        return None

    grid = np.zeros((height, width), dtype=int)
    placed_rects: list[tuple[int, int, int, int]] = []
    channels: list[dict[str, int]] = [{} for _ in range(n)]

    for group in groups:
        idxs = group["indices"]
        local_cols = [ch_cols[i] for i in idxs]
        local_colors = [token_colors[i] for i in idxs]
        xover_rows = _assign_xover_rows(top_ep_row, bot_ep_row, len(group["crossovers"]))
        if xover_rows is None:
            return None
        slot_wire, bypass_wire = _build_colored_network(
            grid,
            local_cols,
            local_cols[-1] + 2,
            top_ep_row,
            bot_ep_row,
            group["crossovers"],
            xover_rows,
            local_colors,
        )
        inv_perm = [0] * len(local_cols)
        for i, dst in enumerate(group["perm"]):
            inv_perm[dst] = i
        if slot_wire != inv_perm or bypass_wire is not None:
            return None

        target_rects = random.choices([0, 1, 2], weights=RECT_COUNT_WEIGHTS)[0]
        for _ in range(target_rects * RECT_RETRY_FACTOR):
            if target_rects <= 0:
                break
            placed = _try_place_9rect(grid, local_cols, xover_rows, top_ep_row, bot_ep_row, placed_rects)
            if placed is not None:
                placed_rects.append(placed)
                target_rects -= 1

        for local_src, global_idx in enumerate(idxs):
            dst_local = group["perm"][local_src]
            channels[global_idx] = {
                "color": token_colors[global_idx],
                "top_col": local_cols[local_src],
                "bot_col": local_cols[dst_local],
            }

    output = grid.copy()
    for ch in channels:
        _stamp_token(output, top_token_row, ch["top_col"], ch["color"], _DIAMOND_OFF)
        _stamp_token(output, bot_token_row, ch["bot_col"], ch["color"], _X_OFF)

    inp = grid.copy()
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))
    for i, ch in enumerate(channels):
        top_center = (top_token_row, ch["top_col"])
        bot_center = (bot_token_row, ch["bot_col"])
        if i in x_channels:
            near, far = (top_center, bot_center) if random.random() < 0.5 else (bot_center, top_center)
            _stamp_token(inp, near[0], near[1], ch["color"], _X_OFF)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], ch["color"], _DIAMOND_OFF)
        else:
            side = top_center if random.random() < 0.5 else bot_center
            _stamp_token(inp, side[0], side[1], ch["color"], _SOLID_OFF)

    metrics = _variant_metrics(inp)
    if metrics["subnetwork_count"] < 2 or metrics["subnetwork_count"] > 3:
        return None
    if metrics["multicolor_component_count"] < 1 or metrics["max_pipe_colors_in_component"] < 2:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
