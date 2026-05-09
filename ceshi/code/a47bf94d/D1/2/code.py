# Variant of a47bf94d
# Dimension: D1 — 管道路径追踪方式 (path)
# Change: 管道追踪（pipeline_trace）——沿青色(color 8)正交管道线段从一个端点追踪到另一个端点，遇到color-9矩形块时穿过
#         （从入口面到对面出口面）→ BFS链式激活扩散

"""
design_analysis:
  concrete_variant:
    preserved:
      - token vocabulary stays the same: solid 3x3, X, diamond
      - source token still sits exactly 2 cells outside a degree-1 wire endpoint
      - wire colors stay in the original family: 8 main wire, 5 crossing markers, 9 wire-embedded blocks
    changed_core:
      - the wire is no longer a permutation router with one far endpoint per source
      - instead, the network is a single branching tree: top stems join a top bus, bottom stems join a bottom bus, and one trunk links the buses
      - a source token injects a signal at its attached endpoint; the signal spreads by BFS through every connected branch
      - every other endpoint reached by that BFS receives the source token's complementary shape
    visible_effect:
      - old logic on the same grid would only populate one destination leaf
      - new logic populates all non-source leaves in the connected component

  behavior_trace:
    original_local_flow:
      - endpoint -> greedy trace along one path -> stop at one terminal -> stamp one complement
    new_local_flow:
      - endpoint -> BFS over the abstract wire graph -> visit branch junctions -> collect all reachable terminals -> stamp complements at each
    concrete_example:
      - if a solid block is attached to the top-left leaf, the old task would create one X at a single matched far leaf
      - here the same solid block becomes a diamond locally, while X copies appear at every other top/bottom leaf attached to the same trunk

  complexity_calibration:
    - one sample contains 6-8 leaves, so the branching rule is obvious and not reducible to a trivial 1-to-1 guess
    - only one source token is used per grid to avoid color collisions at shared destinations
    - the source alternates between solid and X so both complementary outcomes remain represented across examples
    - optional 9-rectangles are placed on the trunk, so pass-through logic remains relevant without introducing unrelated routing ambiguity

  implicit_assumption_audit:
    old_main_dependencies_that_break:
      - greedy direction state assumed each component had one meaningful continuation; branch nodes violate that
      - scanning all endpoint directions and tracing from each assumed tokens map to unique far destinations; BFS now requires a graph model and endpoint set
      - treating 9-cells as traversed positions with a current direction assumed single-path motion; BFS instead uses explicit outside-face edges through each 9-rect
    new_main_assumptions:
      - wire cells that matter are 5/8; 9-rectangles are modeled as teleport edges between face-adjacent 5/8 cells
      - source tokens appear only at degree-1 endpoints and there is at most one source per connected component
      - destination 3x3 zones are empty of other source tokens

  placement_legality:
    - token centers lie on fixed top/bottom rows, spaced by 4-5 columns, so 3x3 stamps never overlap
    - endpoints are exactly 2 rows inward from token centers, matching solver detection
    - the trunk rectangles never touch bus rows, so each 9-block has only north/south connections and remains solver-safe
    - buses and stems form a tree, so BFS connectivity is unambiguous and there are no cycles that could hide the branching rule

  observability_and_cleanliness:
    - all destinations share the source color, making the spread visually attributable to the unique source token
    - top and bottom buses produce repeated branching evidence in a compact but readable layout
    - color 5 remains sparse and only marks through-sections on the buses, preventing clutter
    - no hidden ordering uses color ids or off-screen metadata
"""

from collections import deque
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

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0, c0))} for maximal 3x3 solid blocks."""
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
    """Return {center: (color, (r0, c0))} for isolated 3x3 X patterns."""
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
            for dr, dc in _X_OFF:
                rr, cc = r0 + 1 + dr, c0 + 1 + dc
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


def _find_9rects(g: np.ndarray) -> list[dict]:
    """Find color-9 rectangles together with the 5/8 cells touching each face."""
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
                if (nr, nc) in cells or not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) not in (5, 8):
                    continue
                face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                faces.setdefault(face, []).append((nr, nc))
        rects.append({"cells": cells, "faces": faces})
    return rects


def _pick_face_match(face: str, src: tuple[int, int], dst_pts: list[tuple[int, int]]) -> tuple[int, int]:
    sr, sc = src
    if face in ("N", "S"):
        return min(dst_pts, key=lambda p: abs(p[1] - sc))
    return min(dst_pts, key=lambda p: abs(p[0] - sr))


def _build_wire_graph(g: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """Graph on 5/8 cells plus explicit pass-through edges around 9-rectangles."""
    nodes = {(int(r), int(c)) for r, c in np.argwhere((g == 5) | (g == 8))}
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {p: set() for p in nodes}

    for r, c in nodes:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in nodes:
                adj[(r, c)].add((nr, nc))

    for rect in _find_9rects(g):
        faces = rect["faces"]
        for face, pts in faces.items():
            opp = _OPP[face]
            if opp in faces:
                exit_face = opp
            else:
                others = [f for f in faces if f != face]
                if len(others) != 1:
                    continue
                exit_face = others[0]
            for p in pts:
                q = _pick_face_match(face, p, faces[exit_face])
                if p == q:
                    continue
                adj.setdefault(p, set()).add(q)
                adj.setdefault(q, set()).add(p)
    return adj


def _clear_box(out: np.ndarray, center: tuple[int, int], bg: int) -> None:
    cr, cc = center
    h, w = out.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = bg


def _paint_shape(out: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]], bg: int) -> None:
    _clear_box(out, center, bg)
    h, w = out.shape
    for dr, dc in offsets:
        rr, cc = center[0] + dr, center[1] + dc
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = color


def _endpoint_outdir(ep: tuple[int, int], adj: dict[tuple[int, int], set[tuple[int, int]]]) -> tuple[int, int] | None:
    nbrs = list(adj.get(ep, ()))
    if len(nbrs) != 1:
        return None
    nr, nc = nbrs[0]
    return ep[0] - nr, ep[1] - nc


def _attached_center(ep: tuple[int, int], outdir: tuple[int, int] | None) -> tuple[int, int] | None:
    if outdir is None:
        return None
    return ep[0] + 2 * outdir[0], ep[1] + 2 * outdir[1]


def _reachable(start: tuple[int, int], adj: dict[tuple[int, int], set[tuple[int, int]]]) -> set[tuple[int, int]]:
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread one token's signal through a branching pipe graph and stamp its complement at every reached leaf.

    Input structure:
    - Background 0 surrounds a connected 8/5 wire tree, sometimes interrupted by rectangular 9 pass-through blocks.
    - Exactly one 3x3 source token (solid or X) is attached two cells outside a degree-1 endpoint of that tree.

    Rule:
    - A solid source collapses in place to a diamond; an X source stays as an X.
    - The source endpoint injects a BFS signal into the wire graph. Whenever that signal reaches another endpoint, place the source token's complementary shape there.

    Key insight:
    - This is no longer a one-path transport puzzle. The wire must be understood as a branching connectivity graph, with 9-blocks acting as embedded pass-through links, so one source populates every reachable leaf.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    adj = _build_wire_graph(g)
    endpoints = [p for p, nbrs in adj.items() if len(nbrs) <= 1]
    endpoint_dirs = {ep: _endpoint_outdir(ep, adj) for ep in endpoints}

    out = g.copy()
    source_by_ep: dict[tuple[int, int], tuple[int, str]] = {}

    for ep in endpoints:
        center = _attached_center(ep, endpoint_dirs[ep])
        if center is None:
            continue
        if center in solids:
            col, _ = solids[center]
            source_by_ep[ep] = (col, "solid")
            _paint_shape(out, center, col, _DIAMOND_OFF, bg)
        elif center in xs:
            col, _ = xs[center]
            source_by_ep[ep] = (col, "x")

    for src_ep, (col, kind) in source_by_ep.items():
        reached = _reachable(src_ep, adj)
        dest_offsets = _X_OFF if kind == "solid" else _DIAMOND_OFF
        for ep in endpoints:
            if ep == src_ep or ep not in reached or ep in source_by_ep:
                continue
            center = _attached_center(ep, endpoint_dirs[ep])
            if center is None:
                continue
            _paint_shape(out, center, col, dest_offsets, bg)

    return out


# --------------- generator (branching_bfs_activation) ---------------

CHANNEL_COUNTS = [3, 4]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
TRUNK_COL_OFFSET = [2, 3]
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2
BUS_GAP_RANGE = (2, 4)
RECT_COUNT_WEIGHTS = [25, 45, 30]
RECT_H_RANGE = (2, 4)
RECT_W_RANGE = (2, 3)
RECT_MARGIN_GAP = 1
SOURCE_SHAPE_WEIGHTS = [65, 35]  # solid, x
GEN_MAX_RETRIES = 300


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _build_branching_wire(
    h: int,
    w: int,
    cols: list[int],
    trunk_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    top_bus_row: int,
    bot_bus_row: int,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)

    for c in cols:
        grid[top_ep_row:top_bus_row + 1, c] = 8
        grid[bot_bus_row:bot_ep_row + 1, c] = 8

    left_col = min(cols)
    grid[top_bus_row, left_col:trunk_col + 1] = 8
    grid[bot_bus_row, left_col:trunk_col + 1] = 8
    grid[top_bus_row:bot_bus_row + 1, trunk_col] = 8

    for bus_row in (top_bus_row, bot_bus_row):
        for c in cols:
            if left_col < c < trunk_col:
                grid[bus_row, c - 1] = 5
                grid[bus_row, c + 1] = 5

    return grid


def _try_place_trunk_rect(
    grid: np.ndarray,
    trunk_col: int,
    top_bus_row: int,
    bot_bus_row: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    if bot_bus_row - top_bus_row < 5:
        return None

    rect_h = random.randint(*RECT_H_RANGE)
    if top_bus_row + 2 > bot_bus_row - rect_h:
        return None
    rect_r0 = random.randint(top_bus_row + 2, bot_bus_row - rect_h)
    rect_r1 = rect_r0 + rect_h - 1

    rect_w = random.randint(*RECT_W_RANGE)
    shift = random.randint(0, rect_w - 1)
    rect_c0 = trunk_col - shift
    rect_c1 = rect_c0 + rect_w - 1
    if rect_c0 < 0 or rect_c1 >= w:
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    sub = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 8)):
        return None
    if not np.all(grid[rect_r0:rect_r1 + 1, trunk_col] == 8):
        return None
    if grid[rect_r0 - 1, trunk_col] != 8 or grid[rect_r1 + 1, trunk_col] != 8:
        return None

    grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = 9
    return (rect_r0, rect_c0, rect_r1, rect_c1)


def _expected_output(
    wire: np.ndarray,
    cols: list[int],
    top_token_row: int,
    bot_token_row: int,
    source_center: tuple[int, int],
    color: int,
    source_kind: str,
) -> np.ndarray:
    out = wire.copy()
    if source_kind == "solid":
        _paint_shape(out, source_center, color, _DIAMOND_OFF, 0)
        dest_offsets = _X_OFF
    else:
        _paint_shape(out, source_center, color, _X_OFF, 0)
        dest_offsets = _DIAMOND_OFF

    for r in (top_token_row, bot_token_row):
        for c in cols:
            center = (r, c)
            if center == source_center:
                continue
            _paint_shape(out, center, color, dest_offsets, 0)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)
    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    cols = [first_col + i * spacing for i in range(n)]
    trunk_col = cols[-1] + random.choice(TRUNK_COL_OFFSET)
    w = max(trunk_col + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    h = random.randint(*GRID_H_RANGE)
    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP

    top_bus_row = top_ep_row + random.randint(*BUS_GAP_RANGE)
    bot_bus_row = bot_ep_row - random.randint(*BUS_GAP_RANGE)
    if bot_bus_row - top_bus_row < 5:
        return None

    wire = _build_branching_wire(h, w, cols, trunk_col, top_ep_row, bot_ep_row, top_bus_row, bot_bus_row)

    placed_rects: list[tuple[int, int, int, int]] = []
    n_rects = random.choices([0, 1, 2], weights=RECT_COUNT_WEIGHTS)[0]
    for _ in range(n_rects * 5):
        if len(placed_rects) >= n_rects:
            break
        rect = _try_place_trunk_rect(wire, trunk_col, top_bus_row, bot_bus_row, placed_rects)
        if rect is not None:
            placed_rects.append(rect)

    color = random.choice(TOKEN_COLORS_POOL)
    source_kind = random.choices(["solid", "x"], weights=SOURCE_SHAPE_WEIGHTS)[0]
    source_side = random.choice(["top", "bottom"])
    source_col = random.choice(cols)
    source_center = (top_token_row, source_col) if source_side == "top" else (bot_token_row, source_col)

    inp = wire.copy()
    if source_kind == "solid":
        _stamp_token(inp, source_center[0], source_center[1], color, _SOLID_OFF)
    else:
        _stamp_token(inp, source_center[0], source_center[1], color, _X_OFF)

    expected = _expected_output(wire, cols, top_token_row, bot_token_row, source_center, color, source_kind)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
