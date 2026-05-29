# Variant of a47bf94d
# Dimension: D2 — 管道交叉处理规则 (obstacle)
# Change: 交叉标记（crossing markers）——垂直管道通过水平管道的交叉处，用一对5号像素标记穿行管道的交叉点前后，表... → 单侧标记转向

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the D2 variant:
# - concrete_behavior:
#   The original crossing row had symmetric color-5 cells bracketing a vertical
#   through-wire. Those markers only signaled "keep going straight". In the
#   variant, each swap row is still a horizontal color-8 segment crossing several
#   vertical color-8 channels, but only the two endpoint columns of the swap are
#   marked: the left endpoint gets a single right-side 5, the right endpoint gets
#   a single left-side 5. That means "the vertical wire plugs into this specific
#   horizontal arm". Unmarked intersections stay straight.
#   Example: if columns 4 and 12 are swapped on row 9, then (9,5)=5 and
#   (9,11)=5. A token trace descending in column 4 turns into the row at column 4,
#   travels east, and turns back into column 12 when it reaches the left-side
#   marker there. Under the old solver, the same grid would usually keep going
#   straight or choose a fixed fallback turn, so the output changes.
# - complexity_calibration:
#   The rule has exactly three locally visible cases at an intersection:
#   left-marked, right-marked, or unmarked. That is small enough to infer from a
#   few examples, but rich enough that marker position matters. Every generated
#   puzzle uses a non-identity permutation, so at least one marked swap row is
#   present, and each swap row contains both left- and right-side markers.
# - implicit_assumptions_review:
#   The original _trace_wire relied on a greedy "prefer straight" rule and on the
#   fact that symmetric markers did not need directional interpretation. That no
#   longer holds: an unguarded straight preference would erase the new behavior,
#   and a fixed perpendicular fallback would misread left-vs-right. The new trace
#   logic therefore inspects the side of nearby 5-cells before choosing a turn and
#   keeps the last vertical travel direction so a wire can leave a horizontal span
#   on the correct side of the grid.
# - placement_legality:
#   Tokens remain centered two cells away from degree-1 endpoints, so token
#   detection is unchanged. Swap rows are separated vertically, never touch the
#   token bands, and stay within channel columns. Color-9 rectangles are only
#   embedded in plain vertical stretches that contain no swap rows, so they cannot
#   overwrite the single-sided markers or create ambiguous mixed semantics.
# - visual_observability:
#   The new rule is visible directly in the input: old puzzles showed mirrored
#   pairs of 5s around pass-through columns, while this variant shows isolated 5s
#   attached to one side of the swapped columns. Because the output permutation is
#   determined by those attachments, a solver must track the side, not just the
#   existence, of a marker.
# - visual_cleanliness:
#   Only one layer of wire meaning is active in any local region: 8/5 for swap
#   rows and 9 for wide vertical junctions away from swaps. This keeps the grid
#   legible and avoids markers being buried inside filled 9-rectangles.
# - color_inference:
#   No color remapping is introduced. Token colors still just identify which top
#   and bottom shapes correspond to one another, so the new challenge is purely
#   spatial and does not depend on hidden ordering by color id.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]


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
            if not isolated:
                continue
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


def _marker_sides(g: np.ndarray, comp: set[tuple[int, int]], r: int, c: int, w: int) -> tuple[bool, bool]:
    west = c - 1 >= 0 and (r, c - 1) in comp and int(g[r, c - 1]) == 5
    east = c + 1 < w and (r, c + 1) in comp and int(g[r, c + 1]) == 5
    return west, east


def _pick_vertical(
    opts: list[tuple[tuple[int, int], int, int]],
    flow_v: int,
) -> tuple[tuple[int, int], int, int] | None:
    vertical = [o for o in opts if o[1] != 0]
    if not vertical:
        return None
    if flow_v:
        for opt in vertical:
            if opt[1] == flow_v:
                return opt
    return sorted(vertical, key=lambda x: x[1])[0]


def _trace_wire(
    g: np.ndarray,
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Follow a wire with single-sided turn markers and 9-rect pass-through.

    A west-side 5 connects the vertical stem to the west horizontal arm.
    An east-side 5 connects the vertical stem to the east horizontal arm.
    Unmarked crossings keep vertical and horizontal traffic straight.
    """
    cur, d = start, dir0
    flow_v = d[0] if d[0] != 0 else 0
    visited: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for _ in range(h * w * 4):
        state = (cur, d, flow_v)
        if state in visited:
            raise AssertionError("wire trace looped under single-sided crossing rules")
        visited.add(state)

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
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
                cur = exit_cell
                d = _FACE2DIR[exit_face]
                if d[0] != 0:
                    flow_v = d[0]
                jumped = True
                break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d

        rev = (-d[0], -d[1])
        opts2 = [o for o in opts if (o[1], o[2]) != rev]
        if not opts2:
            return cur, d

        west_mark, east_mark = _marker_sides(g, comp, r, c, w)
        straight = next((o for o in opts2 if (o[1], o[2]) == d), None)
        chosen = None

        if d[0] != 0:
            if west_mark ^ east_mark:
                turn = (0, -1) if west_mark else (0, 1)
                chosen = next((o for o in opts2 if (o[1], o[2]) == turn), None)
            if chosen is None and straight is not None:
                chosen = straight
            if chosen is None:
                horiz = [o for o in opts2 if o[2] != 0]
                chosen = sorted(horiz, key=lambda x: x[2])[0] if horiz else opts2[0]
        else:
            entered_from_west = d == (0, 1)
            turn_here = (entered_from_west and west_mark) or ((not entered_from_west) and east_mark)
            if turn_here:
                chosen = _pick_vertical(opts2, flow_v)
            if chosen is None and straight is not None:
                chosen = straight
            if chosen is None:
                chosen = _pick_vertical(opts2, flow_v)
            if chosen is None:
                chosen = opts2[0]

        cur = chosen[0]
        d = (chosen[1], chosen[2])
        if d[0] != 0:
            flow_v = d[0]

    raise AssertionError("wire trace exceeded step limit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route colored tokens through wire networks whose crossings use one-sided turn markers.

    Input structure:
    - Background cells are empty. Non-background cells form color-8/5/9 wire networks plus 3x3 colored tokens near degree-1 wire endpoints.
    - A color-5 cell is not a generic crossing bracket anymore: its left/right placement beside a crossing tells which horizontal arm is connected to the vertical stem there.

    Rule:
    - Every solid 3x3 token collapses in place to a diamond.
    - Each endpoint token sends its complementary shape to the far endpoint of its wire: solid sends an X, X sends a diamond.
    - While tracing the wire, an unmarked crossing goes straight, a left-side marker joins the west arm, and a right-side marker joins the east arm; color-9 rectangles still act as wide pass-through blocks.

    Key insight:
    - The token-shape logic is unchanged; the new difficulty is that wire routing now depends on marker side, so you must read "which side is marked" rather than just "is there a crossing marker".
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
    rects_9 = _find_9rects(g)

    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = col

    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
            continue

        comp_colors = {int(g[r, c]) for r, c in comp}

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

                if cen in solids:
                    col, _ = solids[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(g, comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(g, comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --------------- generator (single_sided_cross_turn) ---------------

_DIAMOND = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 6
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33

GEN_MAX_RETRIES = 400


def _perm_to_swaps(perm: list[int]) -> list[tuple[int, int]]:
    """Return column swaps so bottom column j carries top wire inv_perm[j]."""
    n = len(perm)
    target = [0] * n
    for i, p in enumerate(perm):
        target[p] = i

    cur = list(range(n))
    swaps = []
    for j in range(n):
        if cur[j] == target[j]:
            continue
        k = next(idx for idx in range(j + 1, n) if cur[idx] == target[j])
        cur[j], cur[k] = cur[k], cur[j]
        swaps.append((j, k))
    return swaps


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _build_wire_network(
    grid: np.ndarray,
    ch_cols: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    swaps: list[tuple[int, int]],
    swap_rows: list[int],
) -> list[int]:
    """Draw vertical channels and swap rows. Returns bottom wire ownership per column."""
    for c in ch_cols:
        grid[top_ep_row : bot_ep_row + 1, c] = 8

    slot_wire = list(range(len(ch_cols)))
    for row, (a, b) in zip(swap_rows, swaps):
        c0, c1 = ch_cols[a], ch_cols[b]
        left, right = min(c0, c1), max(c0, c1)
        grid[row, left : right + 1] = 8
        grid[row, left + 1] = 5
        grid[row, right - 1] = 5
        slot_wire[a], slot_wire[b] = slot_wire[b], slot_wire[a]
    return slot_wire


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    swap_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Place a vertical pass-through 9-rectangle away from marked swap rows."""
    h, w = grid.shape
    max_wires = min(3, len(ch_cols))
    wire_count = random.choices([1, 2, 3][:max_wires], weights=RECT_WIRE_COUNT_WEIGHTS[:max_wires])[0]
    j0 = random.randint(0, len(ch_cols) - wire_count)
    wire_cols = ch_cols[j0 : j0 + wire_count]

    margin_l = random.randint(0, 1 if wire_count > 1 else 2)
    margin_r = random.randint(0, 1 if wire_count > 1 else 2)
    rect_c0 = wire_cols[0] - margin_l
    rect_c1 = wire_cols[-1] + margin_r
    if rect_c0 < 0 or rect_c1 >= w or rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH:
        return None

    max_h = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 2)
    if max_h < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], max_h)
    rect_r0 = random.randint(top_ep_row + 1, bot_ep_row - rect_h)
    rect_r1 = rect_r0 + rect_h - 1

    if any(rect_r0 <= sr <= rect_r1 for sr in swap_rows):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    for c in wire_cols:
        if not np.all(grid[rect_r0 : rect_r1 + 1, c] == 8):
            return None
        if grid[rect_r0 - 1, c] != 8 or grid[rect_r1 + 1, c] != 8:
            return None

    sub = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 8)):
        return None

    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = 9
    return (rect_r0, rect_c0, rect_r1, rect_c1)


def _derive_input(
    grid: np.ndarray,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> np.ndarray:
    inp = grid.copy()
    n = len(ch_cols)
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in x_channels:
            near, far = (top_center, bot_center) if random.choice([True, False]) else (bot_center, top_center)
            _stamp_token(inp, near[0], near[1], col, _X)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND)
        else:
            side = random.choice([top_center, bot_center])
            _stamp_token(inp, side[0], side[1], col, _SOLID)
    return inp


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    width = max(ch_cols[-1] + 3, GRID_W_MIN)
    if width > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n)

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
            break

    swaps = _perm_to_swaps(perm)
    if not swaps:
        return None

    min_wire_rows = len(swaps) * 2 + 2
    h_min = top_ep_row + min_wire_rows + top_token_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    height = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = height - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 1
    step = max(2, avail // (len(swaps) + 1))
    swap_rows = [top_ep_row + step * (k + 1) for k in range(len(swaps))]
    if swap_rows[-1] >= bot_ep_row:
        swap_rows = [top_ep_row + 2 + 2 * k for k in range(len(swaps))]
    if swap_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((height, width), dtype=int)
    slot_wire = _build_wire_network(grid, ch_cols, top_ep_row, bot_ep_row, swaps, swap_rows)

    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    if slot_wire != inv_perm:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(grid, ch_cols, swap_rows, top_ep_row, bot_ep_row, placed_rects)
        if result is not None:
            placed_rects.append(result)

    output = grid.copy()
    for i in range(n):
        _stamp_token(output, top_token_row, ch_cols[i], token_colors[i], _DIAMOND)
        _stamp_token(output, bot_token_row, ch_cols[perm[i]], token_colors[i], _X)

    inp = _derive_input(grid, ch_cols, perm, token_colors, top_token_row, bot_token_row)
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
