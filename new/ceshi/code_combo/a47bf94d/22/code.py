# Combo variant of a47bf94d (index 22)
# combo_spec: D2/3+D18/0+D19/0
# Name: 单侧标记+伪管道干扰+多标记编码

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the fused variant:
# a) Concrete behavior and trace:
#    - Keep the original token vocabulary and top/bottom endpoint layout: solid 3x3
#      blocks and X-tokens appear two cells outside degree-1 endpoints of a mostly
#      vertical wire network, and the solver still converts solid -> diamond in place
#      while routing the complementary shape to the opposite endpoint.
#    - The changed part is how a perpendicular crossing is read. Color-5 no longer
#      means "there is some crossing bracket". It is now an annotation layer placed
#      on only one side of the vertical stem:
#        * 3 markers on the left side  => the wire turns left / west
#        * 3 markers on the right side => the wire turns right / east
#        * 1 marker on either side     => the wire stays straight
#        * 0 markers                   => default straight
#      Example: if a swap row crosses columns 3, 8, 13 on row 10, then column 3 may
#      have a three-cell right-side cluster just above/right of the crossing meaning
#      "drop from col 3 into the eastward row", column 8 may have a single left-side
#      marker in the matching left-side cluster meaning "even though one side is
#      marked, keep going vertically", and column 13 may have the mirrored three-cell
#      left cluster meaning "leave the row and go vertical here".
#    - The grid also contains disconnected color-8 decoy pieces in the middle band.
#      They are real-looking pipes but do not touch any token endpoint, so routing must
#      ignore them even though they share the same visual language.
# b) Complexity calibration:
#    - The local crossing rule has only four visible count states per side (0/1/3 with
#      side location), which is enough to be learnable from a few examples without
#      exploding into many arbitrary codes.
#    - Every generated puzzle is forced to contain at least one 3-marker turning
#      crossing, at least one 1-marker straight crossing, and at least one decoy
#      component. That prevents the combo from collapsing back into the original task
#      or into the simpler D2-only "any one-sided marker means turn" rule.
# c) Implicit assumptions review against the original solver:
#    - Original track extraction treated every non-background non-token cell as a
#      traversable wire cell. That breaks here because stacked 5-markers are pure
#      annotations. The fused solver therefore traces only through colors 8 and 9 and
#      reads color-5 locally as metadata.
#    - Original crossing logic preferred straight unless a fixed perpendicular fallback
#      was needed. That would ignore both the left/right side distinction and the
#      1-versus-3 count distinction. The new trace logic must inspect both side and
#      count before deciding whether vertical flow enters a horizontal span or passes
#      through it.
#    - Original generator could place swap rows too close together. That would make the
#      off-wire three-cell side clusters visually overlap. The fused generator therefore
#      spaces swap rows at least three rows apart and keeps color-9 rectangles away from
#      the annotation bands above each crossing.
# d) Placement legality:
#    - Marker trios occupy a fixed three-cell cluster just above and to one side of a
#      crossing, so swap rows stay below the token band and at least three rows apart
#      from each other.
#    - Color-9 rectangles are inserted only on plain vertical stretches that do not
#      touch swap rows or their marker side-bands.
#    - Decoy pipes are placed only on empty background cells whose 8-neighborhood is
#      otherwise empty, so they cannot accidentally connect to the functional network
#      or to other decoys.
# e) Visual observability:
#    - Across examples a human sees three repeated signals: one-sided single markers
#      that do not cause turns, one-sided trios that do cause turns, and extra 8-only
#      components that never receive or emit tokens. That is enough support to infer
#      both "count matters" and "some pipes are fake".
# f) Visual cleanliness:
#    - The annotation layer is spatially separated from the traversable wire: markers
#      sit beside crossings rather than replacing them, so stacked trios remain legible.
#    - Decoys are restricted to sparse middle-band shapes and never carry 5 or 9, which
#      prevents the middle of the grid from becoming an unreadable pile of mixed codes.
# g) Change inferability:
#    - No hidden color ordering or numeric color rule is introduced. Colors still only
#      identify token correspondences. The new reasoning burden is purely geometric:
#      count the side markers, decide crossing behavior, and ignore disconnected pipes.
# h) New solver assumptions audited against generation:
#    - main() assumes all traversable wire cells are exactly colors 8/9; generate_input()
#      enforces that and uses color-5 only for crossing annotations.
#    - main() assumes any turning crossing has exactly one 3-marker side and any
#      single-marker side means straight; the generator writes only 0/1/3 counts.
#    - main() assumes decoy components have no token center at Manhattan distance 2 from
#      an endpoint; the decoy placer keeps them in the central band away from token rows.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WIRE_COLORS = (8, 9)

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


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
    """Find color-9 rectangular blocks and their color-8 connections per face."""
    mask9 = g == 9
    if not mask9.any():
        return []
    h, w = g.shape
    rects = []
    for comp9 in components_mask(mask9, connectivity=4):
        cells = {(int(r), int(c)) for r, c in comp9}
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        faces: dict[str, list[tuple[int, int]]] = {}
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == 8:
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


_LEFT_MARKER_OFF = [(-2, -1), (-1, -1), (-1, -2)]
_RIGHT_MARKER_OFF = [(-2, 1), (-1, 1), (-1, 2)]
_LEFT_SINGLE_IDX = 2
_RIGHT_SINGLE_IDX = 2


def _marker_counts(g: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = g.shape
    left = 0
    right = 0
    for dr, dc in _LEFT_MARKER_OFF:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == 5:
            left += 1
    for dr, dc in _RIGHT_MARKER_OFF:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == 5:
            right += 1
    return left, right


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
    Follow a wire whose crossings are controlled by one-sided marker counts.

    Traversable cells are only colors 8 and 9. Color-5 cells are annotations:
    count 3 on one side means turn to that side; count 1 or 0 means continue straight.
    """
    cur, d = start, dir0
    flow_v = d[0] if d[0] != 0 else 0
    visited: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for _ in range(h * w * 4):
        state = (cur, d, flow_v)
        if state in visited:
            raise AssertionError("wire trace looped under fused crossing rules")
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

        left_count, right_count = _marker_counts(g, r, c)
        turn_left = left_count == 3 and right_count != 3
        turn_right = right_count == 3 and left_count != 3
        straight = next((o for o in opts2 if (o[1], o[2]) == d), None)
        chosen = None

        if d[0] != 0:
            if turn_left:
                chosen = next((o for o in opts2 if (o[1], o[2]) == (0, -1)), None)
            elif turn_right:
                chosen = next((o for o in opts2 if (o[1], o[2]) == (0, 1)), None)
            if chosen is None and straight is not None:
                chosen = straight
            if chosen is None:
                horiz = [o for o in opts2 if o[2] != 0]
                chosen = sorted(horiz, key=lambda x: x[2])[0] if horiz else opts2[0]
        else:
            entered_from_west = d == (0, 1)
            should_turn = (entered_from_west and turn_left) or ((not entered_from_west) and turn_right)
            if should_turn:
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
    Route colored tokens through annotated pipe networks with decoy pipe fragments.

    Input structure:
    - Background cells are 0. Functional wires use color 8 plus optional color-9 rectangular
      junctions, while color-5 cells sit beside crossings as annotations rather than as wire.
    - Colored 3x3 tokens (solid or X) appear two cells beyond degree-1 endpoints of the real
      network, and extra disconnected color-8 pipe snippets may appear elsewhere as decoys.

    Rule:
    - Every solid 3x3 token collapses in place to a diamond.
    - Each endpoint token emits its complementary shape to the far endpoint of the same connected
      functional wire: solid emits X, X emits diamond.
    - Crossing behavior is decoded from one-sided marker counts: a left-side trio means turn left,
      a right-side trio means turn right, and a single marker or no marker means continue straight.
      Disconnected decoy pipes are ignored because they do not connect any endpoint token pair.

    Key insight:
    - The token-shape transformation is still the original complement rule; the new challenge is to
      separate real pipes from decoys and to read both marker side and marker count at each crossing.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    track_mask = np.isin(g, WIRE_COLORS)

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

                if cen in solids:
                    col, _ = solids[cen]
                    end, dd = _trace_wire(g, comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    end, dd = _trace_wire(g, comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --------------- generator (single_sided_counted_crossings_with_decoys) ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [5]
LEFT_MARGIN_RANGE = (2, 2)
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [15, 40, 30, 15]
RECT_WIRE_COUNT_WEIGHTS = [55, 35, 10]
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 6
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33

DECOY_COUNT_WEIGHTS = [55, 30, 15]
DECOY_RETRY_FACTOR = 20

GEN_MAX_RETRIES = 600


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


def _paint_marker(grid: np.ndarray, row: int, col: int, side: str, count: int) -> None:
    offsets = _LEFT_MARKER_OFF if side == "L" else _RIGHT_MARKER_OFF
    if count == 1:
        dr, dc = offsets[_LEFT_SINGLE_IDX if side == "L" else _RIGHT_SINGLE_IDX]
        grid[row + dr, col + dc] = 5
        return
    if count == 3:
        for dr, dc in offsets:
            grid[row + dr, col + dc] = 5
        return
    raise ValueError(f"unsupported marker count {count}")


def _build_wire_network(
    grid: np.ndarray,
    ch_cols: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    swaps: list[tuple[int, int]],
    swap_rows: list[int],
) -> tuple[list[int], list[tuple[int, int, str, int]], int]:
    """Draw vertical channels and swap rows. Returns bottom wire ownership plus marker specs."""
    for c in ch_cols:
        grid[top_ep_row : bot_ep_row + 1, c] = 8

    marker_specs: list[tuple[int, int, str, int]] = []
    straight_sites: list[tuple[int, int]] = []
    slot_wire = list(range(len(ch_cols)))
    for row, (a, b) in zip(swap_rows, swaps):
        left_col = ch_cols[a]
        right_col = ch_cols[b]
        grid[row, left_col : right_col + 1] = 8

        marker_specs.append((row, left_col, "R", 3))
        marker_specs.append((row, right_col, "L", 3))
        for c in ch_cols:
            if left_col < c < right_col:
                straight_sites.append((row, c))

        slot_wire[a], slot_wire[b] = slot_wire[b], slot_wire[a]

    if straight_sites:
        forced_one = random.randrange(len(straight_sites))
        forced_zero = random.randrange(len(straight_sites)) if len(straight_sites) >= 2 else None
        while forced_zero == forced_one:
            forced_zero = random.randrange(len(straight_sites))

        for idx, (row, col) in enumerate(straight_sites):
            if idx == forced_one:
                count = 1
            elif idx == forced_zero:
                count = 0
            else:
                count = random.choice([0, 1])
            if count == 1:
                marker_specs.append((row, col, random.choice(["L", "R"]), 1))

    for row, col, side, count in marker_specs:
        _paint_marker(grid, row, col, side, count)
    return slot_wire, marker_specs, len(straight_sites)


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    swap_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Place a vertical pass-through 9-rectangle away from annotated swap rows."""
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

    max_h = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4)
    if max_h < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], max_h)
    rect_r0 = random.randint(top_ep_row + 1, bot_ep_row - rect_h)
    rect_r1 = rect_r0 + rect_h - 1

    if any(rect_r0 <= sr and sr - 2 <= rect_r1 for sr in swap_rows):
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


_DECOY_SHAPES = [
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1)],
]


def _can_place_decoy(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != 0:
            return False
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and grid[nr, nc] != 0:
                    return False
    return True


def _add_decoys(grid: np.ndarray, top_ep_row: int, bot_ep_row: int) -> int:
    h, w = grid.shape
    target = random.choices([1, 2, 3], weights=DECOY_COUNT_WEIGHTS)[0]
    placed = 0
    if bot_ep_row - top_ep_row < 6:
        return 0

    for _ in range(target * DECOY_RETRY_FACTOR):
        if placed >= target:
            break
        shape = random.choice(_DECOY_SHAPES)
        max_r = bot_ep_row - 2
        min_r = top_ep_row + 2
        if max_r <= min_r:
            break
        anchor_r = random.randint(min_r, max_r)
        anchor_c = random.randint(1, w - 2)
        cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]
        if not _can_place_decoy(grid, cells):
            continue
        for r, c in cells:
            grid[r, c] = 8
        placed += 1
    return placed


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
            _stamp_token(inp, near[0], near[1], col, _X_OFF)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND_OFF)
        else:
            side = random.choice([top_center, bot_center])
            _stamp_token(inp, side[0], side[1], col, _SOLID_OFF)
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
    straight_site_budget = sum(max(0, b - a - 1) for a, b in swaps)
    if not swaps or straight_site_budget < 2:
        return None

    min_wire_rows = len(swaps) * 3 + 3
    h_min = top_ep_row + min_wire_rows + top_token_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    height = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = height - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 4
    step = max(3, avail // max(1, len(swaps)))
    swap_rows = [top_ep_row + 3 + step * k for k in range(len(swaps))]
    if swap_rows[-1] > bot_ep_row:
        swap_rows = [top_ep_row + 3 + 3 * k for k in range(len(swaps))]
    if swap_rows[-1] > bot_ep_row:
        return None

    grid = np.zeros((height, width), dtype=int)
    slot_wire, marker_specs, straight_site_count = _build_wire_network(
        grid, ch_cols, top_ep_row, bot_ep_row, swaps, swap_rows
    )

    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    if slot_wire != inv_perm:
        return None
    if not any(count == 1 for _, _, _, count in marker_specs):
        return None
    if not any(count == 3 for _, _, _, count in marker_specs):
        return None
    if straight_site_count >= 2 and straight_site_count == sum(count == 1 for _, _, _, count in marker_specs):
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(grid, ch_cols, swap_rows, top_ep_row, bot_ep_row, placed_rects)
        if result is not None:
            placed_rects.append(result)

    if _add_decoys(grid, top_ep_row, bot_ep_row) < 1:
        return None

    output = grid.copy()
    for i in range(n):
        _stamp_token(output, top_token_row, ch_cols[i], token_colors[i], _DIAMOND_OFF)
        _stamp_token(output, bot_token_row, ch_cols[perm[i]], token_colors[i], _X_OFF)

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
