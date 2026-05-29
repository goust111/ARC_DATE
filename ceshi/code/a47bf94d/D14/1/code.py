# Variant of a47bf94d
# Dimension: D14 — 管道颜色体系 (symbol_instructions)
# Change: 三色管道系统——主管道用8号色（青色），交叉标记用5号色（灰色），矩形连接块用9号色（栗色），三种结构色固定且不与token色冲突 → 管道使用单一颜色无区分

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4


# Design analysis
# ----------------
# spec:
#   variant_goal:
#     keep the original token behavior unchanged, but collapse every pipe-side structural color
#     into color 8. The only visible difference is the pipe topology itself: former 5-markers
#     disappear and former 9-rectangles become thick 8-colored blobs inside the same network.
#   concretization:
#     input/output still contain top and bottom token rows, one token color per channel, and a
#     single orthogonal wire network implementing a non-identity permutation.
#     The changed appearance is:
#       - every wire cell is 8
#       - plain crossings are now literal "+" intersections of 8-cells
#       - wide pass-through blocks are solid 8-rectangles, not color-9 rectangles
#     Example of the key local change:
#       - old: a rectangle bridge is a solid 9-box with 8/5 neighbors on its faces, so the solver
#         can detect "inside-rectangle" only by color
#       - new: the same region is a solid 8-box embedded in the 8-wire component, so the solver
#         must first detect maximal filled 8-rectangles (height>=2 and width>=2), then apply the
#         same face-to-face pass-through rule there
#       - difference: color no longer tells the solver where the special conduit is; only the thick
#         rectangular topology does
#   complexity_calibration:
#     remove only the D14 color distinction, but keep both kinds of special topology present:
#       - every generated sample uses at least one thick pipe rectangle, so color-9 removal is
#         visible and solver-side rectangle detection is exercised
#       - every sample already has a non-identity permutation, so unmarked 8-colored crossings are
#         also visible
#     token colors still avoid 8, preventing ambiguity between tokens and pipes
#   implicit_assumption_audit:
#     original main() relied on three facts that no longer hold:
#       1. any solid 3x3 non-background block is a token
#          - no longer true because an all-8 pipe rectangle may be 3x3
#          - fix: token detectors explicitly reject color 8
#       2. special pass-through areas can be found by searching color 9
#          - no longer true because rectangles are also color 8
#          - fix: scan the 8-network for maximal solid rectangles and keep only routable ones
#       3. track_mask can be built from all non-background cells and then subtract token cells
#          - still valid, but simpler and safer now to define track_mask as exactly the 8-colored
#            pipe cells, because all structure colors have been unified
#   placement_legality:
#     generator constraints guarantee:
#       - token colors are from {1,2,3,4,6,7}, never 8
#       - each thick rectangle contains at least one newly-filled 0-cell, so it is visually real
#       - rectangles do not overlap and keep a one-cell gap
#       - detected face sets are routable: opposite faces for straight-through wires or exactly one
#         adjacent partner for a turn; ambiguous 3-face T-shapes are rejected
#       - each sample must place at least one adjacent-face turn rectangle successfully, so the
#         monochrome rectangle logic is behaviorally necessary rather than just visually different
#   visual_observability:
#     humans can compare several examples and see that:
#       - the pipe network is entirely cyan/8, with no grey or maroon helper colors
#       - crossings still behave like "go straight unless the path actually bends"
#       - wide cyan blocks act like internal conduits because tokens emerge on matching far ends
#   cleanliness:
#     thick rectangles are separated by margin and token centers remain two cells away from wire
#     endpoints, so token shapes do not overlap pipes and the grid stays legible
#   color_reasoning:
#     there is no new color mapping. The only color-system change is structural: 5 and 9 vanish,
#     while token colors remain stable identifiers of channel identity.


PIPE_COLOR = 8
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid token blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col in (bg, PIPE_COLOR) or np.any(sub != col):
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
    """Return {center: (color, (r0,c0))} for isolated 3x3 token X patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col in (bg, PIPE_COLOR):
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


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    acc = mask.astype(int)
    return acc.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(ps[r1, c1])
    if r0 > 0:
        total -= int(ps[r0 - 1, c1])
    if c0 > 0:
        total -= int(ps[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(ps[r0 - 1, c0 - 1])
    return total


def _is_full_rect(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return _rect_sum(ps, r0, c0, r1, c1) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_maximal_full_rect(ps: np.ndarray, h: int, w: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    if r0 > 0 and _is_full_rect(ps, r0 - 1, c0, r1, c1):
        return False
    if r1 + 1 < h and _is_full_rect(ps, r0, c0, r1 + 1, c1):
        return False
    if c0 > 0 and _is_full_rect(ps, r0, c0 - 1, r1, c1):
        return False
    if c1 + 1 < w and _is_full_rect(ps, r0, c0, r1, c1 + 1):
        return False
    return True


def _collect_rect_faces(pipe_mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> dict[str, list[tuple[int, int]]]:
    h, w = pipe_mask.shape
    faces: dict[str, list[tuple[int, int]]] = {}
    for c in range(c0, c1 + 1):
        if r0 - 1 >= 0 and pipe_mask[r0 - 1, c]:
            faces.setdefault("N", []).append((r0 - 1, c))
        if r1 + 1 < h and pipe_mask[r1 + 1, c]:
            faces.setdefault("S", []).append((r1 + 1, c))
    for r in range(r0, r1 + 1):
        if c0 - 1 >= 0 and pipe_mask[r, c0 - 1]:
            faces.setdefault("W", []).append((r, c0 - 1))
        if c1 + 1 < w and pipe_mask[r, c1 + 1]:
            faces.setdefault("E", []).append((r, c1 + 1))
    return faces


def _count_rect_wires(faces: dict[str, list[tuple[int, int]]]) -> int:
    n_cols = {c for _, c in faces.get("N", [])}
    s_cols = {c for _, c in faces.get("S", [])}
    w_rows = {r for r, _ in faces.get("W", [])}
    e_rows = {r for r, _ in faces.get("E", [])}
    return len(n_cols & s_cols) + len(w_rows & e_rows)


def _faces_are_routable(faces: dict[str, list[tuple[int, int]]]) -> bool:
    if len(faces) < 2:
        return False
    for face in faces:
        opp = _OPP[face]
        if opp in faces:
            continue
        others = [f for f in faces if f != face]
        if len(others) != 1:
            return False
    return True


def _rect_has_turn_signature(faces: dict[str, list[tuple[int, int]]]) -> bool:
    if len(faces) != 2:
        return False
    face_set = set(faces)
    first = next(iter(face_set))
    return _OPP[first] not in face_set


def _find_pipe_rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find maximal solid 8-rectangles that act as thick wire conduits."""
    pipe_mask = g == PIPE_COLOR
    if not pipe_mask.any():
        return []
    h, w = g.shape
    ps = _prefix_sum(pipe_mask)
    rects: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for r0 in range(h - 1):
        for c0 in range(w - 1):
            if not pipe_mask[r0, c0]:
                continue
            for r1 in range(r0 + 1, h):
                for c1 in range(c0 + 1, w):
                    bbox = (r0, c0, r1, c1)
                    if bbox in seen:
                        continue
                    if not _is_full_rect(ps, r0, c0, r1, c1):
                        continue
                    if not _is_maximal_full_rect(ps, h, w, r0, c0, r1, c1):
                        continue
                    faces = _collect_rect_faces(pipe_mask, r0, c0, r1, c1)
                    if not _faces_are_routable(faces):
                        continue
                    total_wires = _count_rect_wires(faces)
                    if not (_rect_has_turn_signature(faces) or 1 <= total_wires <= 3):
                        continue
                    seen.add(bbox)
                    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
                    rects.append((cells, faces))
    return rects


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    pipe_rects: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Follow a monochrome wire network, using topology to jump across thick 8-rectangles.

    At a detected pipe rectangle:
    - continue to the opposite face when that face exists
    - otherwise use the unique adjacent face, which is the legal turn case
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped in monochrome pipe network")
        visited.add((cur, d))

        jumped = False
        for rect_cells, rect_faces in pipe_rects:
            if cur not in rect_cells:
                continue
            entry_face = _DIR2FACE[d]
            opp = _OPP[entry_face]
            if opp in rect_faces:
                exit_face = opp
            else:
                others = [face for face in rect_faces if face != entry_face]
                if len(others) != 1:
                    raise AssertionError("ambiguous entry into monochrome pipe rectangle")
                exit_face = others[0]
            candidates = rect_faces[exit_face]
            er, ec = cur
            if entry_face in ("N", "S"):
                exit_cell = min(candidates, key=lambda p: abs(p[1] - ec))
            else:
                exit_cell = min(candidates, key=lambda p: abs(p[0] - er))
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
        nonrev = [opt for opt in opts if (opt[1], opt[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [opt for opt in opts2 if (opt[1], opt[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            perp = [opt for opt in opts2 if opt[1] * d[0] + opt[2] * d[1] == 0]
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route colored tokens through a fully monochrome pipe network and emit their complementary shapes.

    Input structure:
    - Background cells are 0. Every pipe-side structure is color 8: thin wires, unmarked crossings,
      and thick rectangular conduit blocks. Tokens near degree-1 pipe endpoints use non-8 colors.

    Rule:
    - A solid 3x3 token turns into an in-place diamond and emits an X token to the far endpoint.
    - An X token stays as-is and emits a diamond to the far endpoint.
    - Thick 8-rectangles are not tokens; they are wide conduit regions that preserve the original
      face-to-face wire routing even though they share the same color as the rest of the pipes.

    Key insight:
    - Color no longer separates ordinary wire, crossing hints, and rectangular conduits; the solver
      must read the 8-network topologically: go straight through + intersections, and treat maximal
      filled 8-rectangles as special pass-through zones.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    track_mask = g == PIPE_COLOR
    comps = _components(track_mask)
    pipe_rects = _find_pipe_rects(g)

    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0:r0 + 3, c0:c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = col

    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
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
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, pipe_rects)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, pipe_rects)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --------------- generator ---------------

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

RECT_COUNT_WEIGHTS = [45, 35, 20]               # 1, 2, 3 rectangles
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 8
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33

GEN_MAX_RETRIES = 300


def _decompose_to_crossovers(perm):
    """Decompose permutation into crossover moves using a bypass column."""
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


def _stamp_token(grid, center_r, center_c, color, offsets):
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _try_place_pipe_rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects):
    """Attempt to thicken part of the 8-wire network into one solid rectangle."""
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
        if not np.all(grid[rect_r0:rect_r1 + 1, c] == PIPE_COLOR):
            return None

    sub = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1]
    if not np.all((sub == 0) | (sub == PIPE_COLOR)):
        return None
    if not np.any(sub == 0):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    if rect_r0 <= top_ep_row or rect_r1 >= bot_ep_row:
        return None
    for c in wire_cols:
        if grid[rect_r0 - 1, c] != PIPE_COLOR or grid[rect_r1 + 1, c] != PIPE_COLOR:
            return None

    saved = sub.copy()
    grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = PIPE_COLOR
    pipe_mask = grid == PIPE_COLOR
    ps = _prefix_sum(pipe_mask)
    faces = _collect_rect_faces(pipe_mask, rect_r0, rect_c0, rect_r1, rect_c1)
    total_wires = _count_rect_wires(faces)

    valid = (
        _is_maximal_full_rect(ps, h, w, rect_r0, rect_c0, rect_r1, rect_c1)
        and _faces_are_routable(faces)
        and 1 <= total_wires <= 3
    )
    if not valid:
        grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = saved
        return None
    return (rect_r0, rect_c0, rect_r1, rect_c1)


def _try_place_turn_rect(grid, crossovers, xover_rows, ch_cols, bypass_col, top_ep_row, bot_ep_row, placed_rects):
    """Place one mandatory adjacent-face turn rectangle around a crossover elbow."""
    h, w = grid.shape
    candidates = []
    for xrow, (src_slot, dst_slot) in zip(xover_rows, crossovers):
        if src_slot == -1:
            src_col, dst_col = bypass_col, ch_cols[dst_slot]
        elif dst_slot == -1:
            src_col, dst_col = ch_cols[src_slot], bypass_col
        else:
            src_col, dst_col = ch_cols[src_slot], ch_cols[dst_slot]
        direction = 1 if dst_col > src_col else -1
        max_width = min(3, abs(dst_col - src_col))
        if max_width < 2:
            continue
        for width in range(2, max_width + 1):
            for height in (2, 3):
                if direction > 0:
                    candidates.append((xrow - 1, src_col, xrow - 1 + height - 1, src_col + width - 1))
                    candidates.append((xrow, dst_col - width + 1, xrow + height - 1, dst_col))
                else:
                    candidates.append((xrow - 1, src_col - width + 1, xrow - 1 + height - 1, src_col))
                    candidates.append((xrow, dst_col, xrow + height - 1, dst_col + width - 1))

    random.shuffle(candidates)
    for rect_r0, rect_c0, rect_r1, rect_c1 in candidates:
        if rect_r0 <= top_ep_row or rect_r1 >= bot_ep_row:
            continue
        if rect_r0 < 0 or rect_c0 < 0 or rect_r1 >= h or rect_c1 >= w:
            continue
        if rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH:
            continue

        gap = RECT_MARGIN_GAP
        blocked = False
        for pr0, pc0, pr1, pc1 in placed_rects:
            if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
                blocked = True
                break
        if blocked:
            continue

        sub = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1]
        if not np.all((sub == 0) | (sub == PIPE_COLOR)):
            continue
        if not np.any(sub == 0):
            continue

        saved = sub.copy()
        grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = PIPE_COLOR
        pipe_mask = grid == PIPE_COLOR
        ps = _prefix_sum(pipe_mask)
        faces = _collect_rect_faces(pipe_mask, rect_r0, rect_c0, rect_r1, rect_c1)
        total_wires = _count_rect_wires(faces)
        valid = (
            _is_maximal_full_rect(ps, h, w, rect_r0, rect_c0, rect_r1, rect_c1)
            and _faces_are_routable(faces)
            and _rect_has_turn_signature(faces)
            and total_wires == 0
        )
        if valid:
            return (rect_r0, rect_c0, rect_r1, rect_c1)
        grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = saved
    return None


def _build_wire_network(grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows):
    """Draw the monochrome pipe network."""
    slot_wire = list(range(n))
    bypass_wire = None

    col_seg_start = {}
    for j in range(n):
        col_seg_start[ch_cols[j]] = top_ep_row
        grid[top_ep_row, ch_cols[j]] = PIPE_COLOR

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
                    grid[start:xrow, c] = PIPE_COLOR

        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min:c_max + 1] = PIPE_COLOR
        for c in active_cols:
            grid[xrow, c] = PIPE_COLOR

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
                    grid[start:bot_ep_row + 1, c] = PIPE_COLOR

    return slot_wire, bypass_wire


def _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row):
    inp = grid.copy()
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        if i in x_channels:
            if random.choice([True, False]):
                near, far = top_center, bot_center
            else:
                near, far = bot_center, top_center
            _stamp_token(inp, near[0], near[1], col, _X_OFF)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND_OFF)
        else:
            side = random.choice([top_center, bot_center])
            _stamp_token(inp, side[0], side[1], col, _SOLID_OFF)
    return inp


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
    slot_wire, bypass_wire = _build_wire_network(
        grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
    )

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    placed_rects: list[tuple[int, int, int, int]] = []
    turn_rect = _try_place_turn_rect(
        grid, crossovers, xover_rows, ch_cols, bypass_col, top_ep_row, bot_ep_row, placed_rects
    )
    if turn_rect is None:
        return None
    placed_rects.append(turn_rect)

    target_rects = random.choices([1, 2, 3], weights=RECT_COUNT_WEIGHTS)[0]
    for _ in range(max(0, target_rects - 1) * RECT_RETRY_FACTOR):
        if len(placed_rects) >= target_rects:
            break
        result = _try_place_pipe_rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if result is not None:
            placed_rects.append(result)

    output = grid.copy()
    for i in range(n):
        _stamp_token(output, top_token_row, ch_cols[i], token_colors[i], _DIAMOND_OFF)
        _stamp_token(output, bot_token_row, ch_cols[perm[i]], token_colors[i], _X_OFF)

    inp = _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        if np.array_equal(main(inp), output):
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
