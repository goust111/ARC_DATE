# Combo variant of a47bf94d (index 21)
# combo_spec: D1/2+D8/0+D13/2
# Name: BFS激活+信号变换节点+先分后路由

"""
design_analysis:
  concrete_variant:
    preserved:
      - token vocabulary stays 3x3-local and still uses a source two cells outside a degree-1 pipe endpoint
      - wire colors remain visually legible: 8 for ordinary pipe, sparse 5 markers on buses, 9 for filled rectangular embedded nodes
      - the remote outputs are still complementary fragments derived from a solid source
    changed_core:
      - the network is no longer a one-to-one permutation router; it is a branching tree with one solid source and many reachable leaves
      - every solid first decomposes in place into a visible cross while its moving residue becomes an X-signal
      - propagation is BFS over a pipe graph, so one source can populate every other leaf in the component
      - every path crossing a 9-rectangle flips the moving signal between X and cross, turning the rectangle into a transformation node instead of a neutral pass-through
    concrete_example:
      - suppose the source is the top-left solid on a tree with two other top leaves and three bottom leaves
      - original task: the solid would produce one far X at a unique matched endpoint after greedy routing
      - fused task step 1: the source instantly becomes a cross at its own location
      - fused task step 2: BFS sends an X-signal to all five other leaves; leaves whose unique path crosses a 9-rectangle receive a cross, while leaves on plain pipe receive an X
      - visible difference: the final board shows one local split, one-to-many spreading, and shape changes aligned with 9-block locations

  complexity_calibration:
    - each sample uses 3-4 columns, so a single source reaches 5-7 destinations; this makes BFS branching obvious and avoids accidental equivalence with 1-to-1 routing
    - there is exactly one source per component, which prevents ambiguous color overwrite while keeping the spread pattern easy to read
    - every sample contains at least one transformation rectangle and at least one plain destination, so the 9-node effect is always contrasted inside the same grid
    - the generator caps each source-to-leaf path at at most one rectangle, so the transformed-vs-plain distinction is visible without requiring humans to count long parity chains

  implicit_assumption_audit:
    broken_original_assumptions:
      - the original greedy tracer assumed a single continuation direction; branch points invalidate that and require a graph traversal
      - the old solver treated 9-rectangles as geometric tunnels only; the fused rule needs edges that also carry signal-state changes
      - the old endpoint loop assumed each source maps to one far endpoint; BFS now maps one source to many endpoints
      - the old in-place solid rewrite could be interleaved with routing because routing was independent; here the two-stage order must be explicit because the emitted signal kind depends on the prior split
    new_main_assumptions:
      - every connected pipe component has at most one source token in generated data, so two expanding colors never compete for the same destination box
      - destination 3x3 boxes are empty in the input, so painting routed shapes never destroys unrelated objects
      - 9-rectangles are embedded on vertical trunk or stem segments, which keeps each rectangle's face structure simple and makes endpoint shape changes attributable to path geometry
    generator_guarantees:
      - the source sits exactly two cells outside a degree-1 endpoint, matching solver-side endpoint attachment logic
      - stem spacing and token rows keep all 3x3 boxes disjoint and in bounds
      - rectangles never overlap and always retain one wire cell immediately outside their north and south faces, so the pass-through graph is well formed

  placement_legality:
    - grid sizes stay within the original family: 20-26 rows and 22-26 cols
    - token centers use uniform columns spaced by 4-5, so 3x3 source/output stamps do not touch
    - buses and trunk form a tree, and rectangles are placed only on long enough vertical segments with margin from tokens and bus junctions
    - every sample is oracle-checked by comparing the generated expected output with main(input)

  visual_observability:
    - BFS behavior is readable because one source color fans out to many leaves in one shot
    - the two-step split is visible because the source starts as a solid 3x3 block and ends as a thinner cross
    - the transformation-node rule is visible because some destinations behind 9-blocks end as crosses while sibling leaves on plain pipe end as X-shapes
    - no hidden ordering depends on color ids, scan order, or non-visual metadata

  cleanliness_and_inferability:
    - only one moving color appears per sample, reducing clutter and making all remote shapes attributable to the same source
    - 5-cells stay sparse and only decorate buses, so they cannot be mistaken for token fragments
    - the generator avoids stacking multiple rectangles on a single source-to-leaf path, preventing hard-to-infer parity cancellation
    - because transformed leaves are selected by path structure rather than by arbitrary color mapping, humans can infer the rule from 2-4 examples
"""

from collections import deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_CROSS_OFF = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}

KIND_X = 0
KIND_CROSS = 1


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


def _add_edge(
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    a: tuple[int, int],
    b: tuple[int, int],
    toggle: int,
) -> None:
    pair = (b, toggle)
    if pair not in adj[a]:
        adj[a].append(pair)
    pair_rev = (a, toggle)
    if pair_rev not in adj[b]:
        adj[b].append(pair_rev)


def _build_wire_graph(g: np.ndarray) -> dict[tuple[int, int], list[tuple[tuple[int, int], int]]]:
    """Graph on 5/8 cells; crossing a 9-rectangle toggles the moving shape."""
    nodes = {(int(r), int(c)) for r, c in np.argwhere((g == 5) | (g == 8))}
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {p: [] for p in nodes}

    for r, c in nodes:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in nodes:
                _add_edge(adj, (r, c), (nr, nc), 0)

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
                if p != q:
                    _add_edge(adj, p, q, 1)
    return adj


def _neighbor_set(
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    p: tuple[int, int],
) -> set[tuple[int, int]]:
    return {q for q, _ in adj.get(p, [])}


def _clear_box(out: np.ndarray, center: tuple[int, int], bg: int) -> None:
    cr, cc = center
    h, w = out.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = bg


def _paint_shape(
    out: np.ndarray,
    center: tuple[int, int],
    color: int,
    offsets: list[tuple[int, int]],
    bg: int,
) -> None:
    _clear_box(out, center, bg)
    h, w = out.shape
    for dr, dc in offsets:
        rr, cc = center[0] + dr, center[1] + dc
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = color


def _endpoint_outdir(
    ep: tuple[int, int],
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
) -> tuple[int, int] | None:
    nbrs = sorted(_neighbor_set(adj, ep))
    if len(nbrs) != 1:
        return None
    nr, nc = nbrs[0]
    return ep[0] - nr, ep[1] - nc


def _attached_center(ep: tuple[int, int], outdir: tuple[int, int] | None) -> tuple[int, int] | None:
    if outdir is None:
        return None
    return ep[0] + 2 * outdir[0], ep[1] + 2 * outdir[1]


def _propagate_signal(
    src_ep: tuple[int, int],
    init_kind: int,
    adj: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    endpoints: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """BFS over (node, kind), where toggle=1 on 9-rect transitions flips X <-> cross."""
    q = deque([(src_ep, init_kind)])
    seen = {(src_ep, init_kind)}
    reached: dict[tuple[int, int], int] = {}

    while q:
        cur, kind = q.popleft()
        if cur in endpoints and cur != src_ep:
            reached.setdefault(cur, kind)
        for nxt, toggle in adj.get(cur, []):
            nxt_kind = kind ^ toggle
            state = (nxt, nxt_kind)
            if state not in seen:
                seen.add(state)
                q.append(state)
    return reached


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split solid sources first, then spread their emitted signal through a branching pipe graph with transforming 9-nodes.

    Input structure:
    - Background 0 surrounds a connected 8/5 pipe tree whose leaves sit two cells away from 3x3 token positions.
    - The only source token in generated data is a solid 3x3 block; color-9 rectangles may interrupt vertical stems or the trunk.

    Rule:
    - Stage 1: every solid source is rewritten in place as a cross, and it emits an X-shaped moving signal from its attached endpoint.
    - Stage 2: that signal expands by BFS through the whole pipe component. Moving along ordinary wire preserves the signal; crossing a 9-rectangle flips it between X and cross. Each reached leaf receives the current signal shape.

    Key insight:
    - The task must be read as a graph process with explicit phase order. The source does not route to one matched endpoint anymore; it fans out to every reachable leaf, and the final shape at each leaf is determined by whether its path passed through a transforming 9-node.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    adj = _build_wire_graph(g)
    endpoints = {p for p in adj if len(_neighbor_set(adj, p)) <= 1}
    endpoint_dirs = {ep: _endpoint_outdir(ep, adj) for ep in endpoints}

    out = g.copy()
    source_by_ep: dict[tuple[int, int], tuple[int, int]] = {}

    # Stage 1: resolve all local solid splits before any routing.
    for ep in sorted(endpoints):
        center = _attached_center(ep, endpoint_dirs[ep])
        if center is None:
            continue
        if center in solids:
            col, _ = solids[center]
            source_by_ep[ep] = (col, KIND_X)
            _paint_shape(out, center, col, _CROSS_OFF, bg)

    # Stage 2: BFS-activate every emitted signal and stamp the reached leaves.
    for src_ep, (col, init_kind) in source_by_ep.items():
        reached = _propagate_signal(src_ep, init_kind, adj, endpoints)
        for ep, kind in reached.items():
            if ep in source_by_ep:
                continue
            center = _attached_center(ep, endpoint_dirs[ep])
            if center is None:
                continue
            offsets = _X_OFF if kind == KIND_X else _CROSS_OFF
            _paint_shape(out, center, col, offsets, bg)

    return out


# --------------- generator (bfs_transform_two_stage) ---------------

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
BUS_GAP_RANGE = (3, 5)
RECT_H_RANGE = (2, 4)
RECT_W_RANGE = (2, 3)
RECT_MARGIN_GAP = 1
GEN_MAX_RETRIES = 300

LAST_GENERATED_INFO: dict | None = None
# Alternate branch choices across calls so 50-sample self-tests cover both trunk and non-trunk cases.
_GEN_CALL_COUNT = 0


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


def _rect_overlaps(
    rect: tuple[int, int, int, int],
    placed_rects: list[tuple[int, int, int, int]],
    gap: int = RECT_MARGIN_GAP,
) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed_rects:
        if r0 <= pr1 + gap and pr0 <= r1 + gap and c0 <= pc1 + gap and pc0 <= c1 + gap:
            return True
    return False


def _try_place_vertical_rect(
    grid: np.ndarray,
    col: int,
    r_lo: int,
    r_hi: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    if r_hi - r_lo + 1 < RECT_H_RANGE[0]:
        return None

    rect_h = random.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], r_hi - r_lo + 1))
    rect_r0 = random.randint(r_lo, r_hi - rect_h + 1)
    rect_r1 = rect_r0 + rect_h - 1

    rect_w = random.randint(*RECT_W_RANGE)
    shift = random.randint(0, rect_w - 1)
    rect_c0 = col - shift
    rect_c1 = rect_c0 + rect_w - 1
    if rect_c0 < 0 or rect_c1 >= w:
        return None

    rect = (rect_r0, rect_c0, rect_r1, rect_c1)
    if _rect_overlaps(rect, placed_rects):
        return None

    if rect_r0 - 1 < 0 or rect_r1 + 1 >= h:
        return None
    if grid[rect_r0 - 1, col] != 8 or grid[rect_r1 + 1, col] != 8:
        return None

    sub = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 8)):
        return None
    if not np.all(grid[rect_r0:rect_r1 + 1, col] == 8):
        return None

    grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = 9
    return rect


def _leaf_ids(cols: list[int]) -> list[tuple[str, int]]:
    ids = [("top", c) for c in cols]
    ids.extend(("bottom", c) for c in cols)
    return ids


def _center_for_leaf(leaf_id: tuple[str, int], top_token_row: int, bot_token_row: int) -> tuple[int, int]:
    side, col = leaf_id
    return (top_token_row, col) if side == "top" else (bot_token_row, col)


def _place_rect_for_leaf(
    grid: np.ndarray,
    leaf_id: tuple[str, int],
    top_ep_row: int,
    bot_ep_row: int,
    top_bus_row: int,
    bot_bus_row: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    side, col = leaf_id
    if side == "top":
        return _try_place_vertical_rect(grid, col, top_ep_row + 1, top_bus_row - 1, placed_rects)
    return _try_place_vertical_rect(grid, col, bot_bus_row + 1, bot_ep_row - 1, placed_rects)


def _expected_output(
    wire: np.ndarray,
    cols: list[int],
    top_token_row: int,
    bot_token_row: int,
    source_leaf: tuple[str, int],
    color: int,
    transformed_leaves: set[tuple[str, int]],
) -> np.ndarray:
    out = wire.copy()
    source_center = _center_for_leaf(source_leaf, top_token_row, bot_token_row)
    _paint_shape(out, source_center, color, _CROSS_OFF, 0)

    for leaf_id in _leaf_ids(cols):
        if leaf_id == source_leaf:
            continue
        center = _center_for_leaf(leaf_id, top_token_row, bot_token_row)
        offsets = _CROSS_OFF if leaf_id in transformed_leaves else _X_OFF
        _paint_shape(out, center, color, offsets, 0)
    return out


def _gen_attempt(prefer_trunk: bool, preferred_source_side: str) -> tuple[np.ndarray, dict] | None:
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
    if bot_bus_row - top_bus_row < RECT_H_RANGE[0] + 2:
        return None

    wire = _build_branching_wire(h, w, cols, trunk_col, top_ep_row, bot_ep_row, top_bus_row, bot_bus_row)

    source_side = preferred_source_side
    source_col = random.choice(cols)
    source_leaf = (source_side, source_col)

    placed_rects: list[tuple[int, int, int, int]] = []
    transformed_leaves: set[tuple[str, int]] = set()

    use_trunk = prefer_trunk
    trunk_rect = None
    if use_trunk:
        trunk_rect = _try_place_vertical_rect(
            wire, trunk_col, top_bus_row + 1, bot_bus_row - 1, placed_rects
        )
        if trunk_rect is None:
            return None
        placed_rects.append(trunk_rect)
        opposite_side = "bottom" if source_side == "top" else "top"
        transformed_leaves.update((opposite_side, c) for c in cols)

    if use_trunk:
        stem_candidates = [(source_side, c) for c in cols if c != source_col]
        stem_pick_max = min(2, len(stem_candidates))
        stem_pick_count = 0 if stem_pick_max == 0 else random.randint(0, stem_pick_max)
    else:
        stem_candidates = [leaf for leaf in _leaf_ids(cols) if leaf != source_leaf]
        stem_pick_max = min(2, len(stem_candidates) - 1)
        stem_pick_count = 1 if stem_pick_max <= 1 else random.randint(1, stem_pick_max)

    random.shuffle(stem_candidates)
    chosen_stems = stem_candidates[:stem_pick_count]
    stem_rects = []
    for leaf_id in chosen_stems:
        rect = _place_rect_for_leaf(
            wire, leaf_id, top_ep_row, bot_ep_row, top_bus_row, bot_bus_row, placed_rects
        )
        if rect is None:
            return None
        placed_rects.append(rect)
        stem_rects.append(rect)
        transformed_leaves.add(leaf_id)

    dest_leaves = [leaf for leaf in _leaf_ids(cols) if leaf != source_leaf]
    if not transformed_leaves or len(transformed_leaves) >= len(dest_leaves):
        return None

    color = random.choice(TOKEN_COLORS_POOL)
    inp = wire.copy()
    source_center = _center_for_leaf(source_leaf, top_token_row, bot_token_row)
    _stamp_token(inp, source_center[0], source_center[1], color, _SOLID_OFF)

    expected = _expected_output(
        wire, cols, top_token_row, bot_token_row, source_leaf, color, transformed_leaves
    )
    if not np.array_equal(main(inp), expected):
        return None

    info = {
        "channels": n,
        "endpoint_count": 2 * n,
        "destination_count": len(dest_leaves),
        "transformed_destinations": len(transformed_leaves),
        "plain_destinations": len(dest_leaves) - len(transformed_leaves),
        "has_trunk_rect": trunk_rect is not None,
        "stem_rect_count": len(stem_rects),
        "source_side": source_side,
    }
    return inp, info


def generate_input() -> np.ndarray:
    global LAST_GENERATED_INFO, _GEN_CALL_COUNT
    call_idx = _GEN_CALL_COUNT
    prefer_trunk = (call_idx % 2) == 0
    preferred_source_side = "top" if ((call_idx // 2) % 2) == 0 else "bottom"
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt(prefer_trunk, preferred_source_side)
        if result is not None:
            inp, info = result
            LAST_GENERATED_INFO = info
            _GEN_CALL_COUNT += 1
            return inp
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
