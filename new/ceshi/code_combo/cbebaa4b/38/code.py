# Combo variant of cbebaa4b (index 38)
# combo_spec: D3/2+D7/3
# Name: 线段锚点 + 双向同步组装

from __future__ import annotations

from pathlib import Path
import sys
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Source task: keep one yellow anchor fixed, then translate the scattered
#      sprites until every red connector pixel overlaps exactly one matching red.
#    - D3/2 changes the anchor from a filled yellow block to a thin yellow line.
#      For this combo the line is used in its strongest "axial" form: the useful
#      attachment sites are the two line ends, not the broad sides.
#    - D7/3 changes the assembly policy from "grow anywhere a red port is free"
#      to "grow both branches in lockstep". The anchor therefore seeds exactly
#      two chains, one from each end, and each recursion level consumes a pair of
#      pieces, one on the left/top branch and one on the right/bottom branch.
#    - Small concrete trace:
#      - Original: a yellow 3x3 block may first accept one piece above it, then
#        another from the left, then a later piece can extend only one of those
#        branches while the other branch waits.
#      - Combo: a horizontal yellow line first accepts one piece at its left end
#        and one at its right end. If there is another stage, both of those new
#        tips must each accept one more piece. The visible result is two matched
#        outward chains with equal depth.
#
# b) Complexity calibration
#    - The generator fixes the number of non-anchor pieces at 4. That yields two
#      synchronized attachment stages after the anchor, which is enough to make
#      the dual-chain structure obvious without introducing long ambiguous walks.
#    - Every non-anchor piece has either one port pair (leaf) or two opposite port
#      pairs (internal chain segment). This removes branching hubs that would hide
#      the D7/3 effect and keeps the topology legible to humans.
#    - Each stage reuses the same non-red color on both branches. That provides a
#      repeated visual cue that the two chains advance in matched layers without
#      introducing any hidden color mapping rule.
#
# c) Hidden-assumption audit of the original main()
#    - `_extract_pieces(...)` still works unchanged because the input is still a
#      collection of disjoint 4-connected components on a black background.
#    - The original anchor test "piece contains yellow" is no longer sufficient.
#      The new solver must verify a straight yellow line and verify that its four
#      red cells sit on the two line ends.
#    - The original DFS implicitly allowed one branch to finish before another.
#      That contradicts D7/3, so the new DFS groups moves into synchronized pairs:
#      one placement for each active frontier at every depth.
#    - `_try_place(...)` in the seed permitted same-color non-red overlap. That is
#      unsafe once same-stage color reuse is introduced, because identical colors
#      could accidentally mask an illegal collision. The combo therefore enforces
#      the stricter physical rule: only red-on-red overlap is allowed.
#    - New assumption introduced by the combo: every non-anchor piece exposes red
#      ports only on the anchor axis, and the final topology is exactly two equal-
#      depth chains. `main()` checks the first part directly and relies on the
#      synchronized search plus uniqueness check for the second; `generate_input()`
#      constructs samples that satisfy both properties.
#
# d) Placement legality
#    - Generation remains output-first. A fully assembled target is created first,
#      then all non-anchor sprites are scattered away from the anchor.
#    - In the target, every child overlaps exactly one parent and only at a single
#      2-cell red port pair. `_candidate_fits(...)` rejects non-parent overlaps,
#      out-of-bounds placements, and assemblies that exceed the global size limit.
#    - The scattered input keeps the anchor at its solved position and places every
#      other sprite with Chebyshev padding, preserving the original "reassemble by
#      translation only" interaction pattern.
#
# e) Visual observability
#    - Across examples the yellow object is always a line, never a rectangle, so
#      D3/2 is immediately visible.
#    - D7/3 is visible because the solved outputs always have two outward branches
#      of equal depth, anchored at the two line ends. A human can see that the
#      whole structure advances in matched left/right or top/bottom layers.
#    - No decision depends on color ids, generation order, or hidden labels. The
#      rule is encoded in visible geometry: line ends, red port placement, and the
#      balanced two-branch topology.
#
# f) Visual cleanliness
#    - Restricting overlap to red/red keeps all colored bodies readable even when
#      the chains bend.
#    - The line anchor leaves more empty space around the center than the original
#      yellow block, so the two branches remain easy to distinguish.
#
# g) Change inferability
#    - There is no color remapping puzzle. Yellow still means anchor and red still
#      means connector.
#    - The inferable novelty is structural: identify the yellow line, notice that
#      docking starts from its two ends, then notice that both ends progress in
#      parallel stages until the whole shape is complete.
#
# h) Fusion metrics used after implementation
#    - Metric 1: chain length. Every accepted sample records synchronous depth 2,
#      so the total non-anchor piece count is always 4.
#    - Metric 2: two-ended usage. In every solved output there must be occupied
#      non-anchor pixels beyond both ends of the yellow line along the anchor axis.
#    - Metric 3: orientation mix. Over repeated trials both horizontal and vertical
#      line anchors should appear.
#
# i) New solver assumptions and generator guarantees
#    - `main()` assumes there is exactly one valid line-anchor with end ports.
#      `_make_anchor()` and `_line_anchor_info(...)` guarantee and verify that.
#    - `main()` assumes every ordinary piece has 2 or 4 red cells on the anchor
#      axis only. `_make_piece()` and `_piece_axis_info(...)` guarantee and verify
#      that.
#    - `main()` assumes the full assembly is unique under synchronized growth.
#      `_gen_attempt()` checks this by requiring `main(input) == assembled_output`
#      before accepting a generated sample.


SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}
AXIS_FRONTIERS = {"horizontal": ("left", "right"), "vertical": ("top", "bottom")}
INCOMING_SIDE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
            }
        )
    return pieces


def _try_place(
    canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int
) -> np.ndarray | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None

    sub = canvas[top : top + ph, left : left + pw]
    mask = arr != bg
    if not np.any(mask):
        return None

    conflict = mask & (sub != bg) & ~((arr == 2) & (sub == 2))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _line_anchor_info(arr: np.ndarray) -> dict | None:
    yellow = np.argwhere(arr == 4)
    if len(yellow) < 5:
        return None

    r0, c0, r1, c1 = bbox_coords(yellow)
    if (r0 == r1) == (c0 == c1):
        return None
    if np.any(arr[r0 : r1 + 1, c0 : c1 + 1] != 4):
        return None
    if int(np.sum(arr == 4)) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return None

    reds = np.argwhere(arr == 2)
    if len(reds) != 4:
        return None

    if r0 == r1:
        cols = sorted({int(c) for _, c in reds})
        if cols != [c0 - 1, c1 + 1]:
            return None
        for cc in cols:
            rows = sorted(int(r) for r, c in reds if int(c) == cc)
            if rows != [r0, r0 + 1]:
                return None
        return {"axis": "horizontal", "yellow_bbox": (r0, c0, r1, c1)}

    rows = sorted({int(r) for r, _ in reds})
    if rows != [r0 - 1, r1 + 1]:
        return None
    for rr in rows:
        cols = sorted(int(c) for r, c in reds if int(r) == rr)
        if cols != [c0, c0 + 1]:
            return None
    return {"axis": "vertical", "yellow_bbox": (r0, c0, r1, c1)}


def _piece_axis_info(arr: np.ndarray, expected_axis: str) -> dict | None:
    reds = np.argwhere(arr == 2)
    if len(reds) not in (2, 4):
        return None

    h, w = arr.shape
    groups: dict[str, list[tuple[int, int]]] = {}
    for rr, cc in reds:
        sides = []
        if int(rr) == 0:
            sides.append("top")
        if int(rr) == h - 1:
            sides.append("bottom")
        if int(cc) == 0:
            sides.append("left")
        if int(cc) == w - 1:
            sides.append("right")
        if len(sides) != 1:
            return None
        groups.setdefault(sides[0], []).append((int(rr), int(cc)))

    used_sides = []
    for side, cells in groups.items():
        if len(cells) != 2:
            return None
        rows = sorted(r for r, _ in cells)
        cols = sorted(c for _, c in cells)
        if side in ("top", "bottom"):
            if rows[0] != rows[1] or cols[1] != cols[0] + 1:
                return None
        else:
            if cols[0] != cols[1] or rows[1] != rows[0] + 1:
                return None
        used_sides.append(side)

    allowed = set(AXIS_FRONTIERS[expected_axis])
    if not set(used_sides).issubset(allowed):
        return None
    if len(used_sides) != len(reds) // 2:
        return None

    return {
        "axis": expected_axis,
        "used_sides": tuple(sorted(used_sides)),
        "port_count": int(len(reds)),
        "groups": {side: tuple(sorted(cells)) for side, cells in groups.items()},
    }


def _group_frontiers(rc: dict[tuple[int, int], int], axis: str) -> list[dict] | None:
    unpaired = sorted(coord for coord, count in rc.items() if count == 1)
    if not unpaired:
        return []

    groups = []
    if axis == "horizontal":
        by_col: dict[int, list[int]] = {}
        for rr, cc in unpaired:
            by_col.setdefault(cc, []).append(rr)
        for cc, rows in by_col.items():
            rows = sorted(rows)
            if len(rows) != 2 or rows[1] != rows[0] + 1:
                return None
            groups.append({"coord_key": cc, "coords": ((rows[0], cc), (rows[1], cc))})
    else:
        by_row: dict[int, list[int]] = {}
        for rr, cc in unpaired:
            by_row.setdefault(rr, []).append(cc)
        for rr, cols in by_row.items():
            cols = sorted(cols)
            if len(cols) != 2 or cols[1] != cols[0] + 1:
                return None
            groups.append({"coord_key": rr, "coords": ((rr, cols[0]), (rr, cols[1]))})

    if len(groups) != 2:
        return None

    groups.sort(key=lambda g: g["coord_key"])
    for label, grp in zip(AXIS_FRONTIERS[axis], groups):
        grp["label"] = label
    return groups


def _next_red_counts(
    rc: dict[tuple[int, int], int], piece: dict, top: int, left: int
) -> dict[tuple[int, int], int]:
    out = dict(rc)
    for rr, cc in piece["reds"]:
        key = (top + rr, left + cc)
        out[key] = out.get(key, 0) + 1
    return out


def _candidate_moves_for_frontier(
    cur: np.ndarray,
    remaining: list[dict],
    rc: dict[tuple[int, int], int],
    frontier: dict,
    target_port_count: int,
    bg: int,
) -> list[tuple[int, int, int]]:
    h, w = cur.shape
    label = frontier["label"]
    incoming = INCOMING_SIDE[label]
    target_coords = frontier["coords"]
    moves = []

    for idx, piece in enumerate(remaining):
        info = piece["axis_info"]
        if info["port_count"] != target_port_count:
            continue
        if incoming not in info["groups"]:
            continue

        local_group = info["groups"][incoming]
        top = target_coords[0][0] - local_group[0][0]
        left = target_coords[0][1] - local_group[0][1]
        translated = tuple((top + rr, left + cc) for rr, cc in local_group)
        if translated != target_coords:
            continue

        ph, pw = piece["shape"]
        if top < 0 or left < 0 or top + ph > h or left + pw > w:
            continue

        overlapped = []
        bad = False
        for rr, cc in piece["reds"]:
            gr, gc = top + rr, left + cc
            prev = rc.get((gr, gc), 0)
            if prev >= 2:
                bad = True
                break
            if prev == 1:
                overlapped.append((gr, gc))
        if bad or tuple(sorted(overlapped)) != target_coords:
            continue

        if _try_place(cur, piece["arr"], top, left, bg) is None:
            continue
        moves.append((idx, top, left))

    dedup = sorted(set(moves), key=lambda x: (-remaining[x[0]]["area"], x[1], x[2], x[0]))
    return dedup


def _anchor_end_extensions(out: np.ndarray, anchor: dict) -> tuple[bool, bool]:
    r0, c0, r1, c1 = anchor["bbox"]
    yr0, yc0, yr1, yc1 = anchor["anchor_info"]["yellow_bbox"]
    axis = anchor["anchor_info"]["axis"]
    g = out.copy()
    g[r0 + yr0 : r0 + yr1 + 1, c0 + yc0 : c0 + yc1 + 1] = 0
    mask = g != 0
    if axis == "horizontal":
        left_ok = bool(np.any(mask[:, : c0 + yc0]))
        right_ok = bool(np.any(mask[:, c0 + yc1 + 1 :]))
        return left_ok, right_ok
    top_ok = bool(np.any(mask[: r0 + yr0, :]))
    bottom_ok = bool(np.any(mask[r0 + yr1 + 1 :, :]))
    return top_ok, bottom_ok


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites around a fixed yellow line-anchor by extending two connector chains in synchronized stages.

    Input structure:
    - bg = black (0). Several disconnected colored sprites are scattered on the grid.
    - Exactly one sprite contains yellow (4) cells arranged as a straight horizontal or vertical line, with one 2-cell red (2) connector pair at each end of that line.
    - Every other sprite has either one or two red connector pairs, and all of those pairs lie on the same axis as the yellow line.

    Rule:
    - Keep the yellow line-anchor at its input position.
    - Translate the remaining sprites without rotation or reflection so red connector pairs overlap existing unpaired red pairs.
    - The assembly grows in synchronized two-piece stages: one piece extends one end-chain and one piece extends the opposite end-chain, until every original red pixel has been paired exactly once.

    Key insight: Red-pair matching still drives the puzzle, but the fixed seed is now a two-ended yellow line. Solving therefore means tracking two equally deep outward chains at once rather than letting a single branch grow arbitrarily ahead of the other.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = []
    for piece in pieces:
        info = _line_anchor_info(piece["arr"])
        if info is not None:
            piece["anchor_info"] = info
            anchors.append(piece)
    assert len(anchors) == 1, "input must have exactly one yellow line-anchor piece"
    anchor = anchors[0]
    axis = anchor["anchor_info"]["axis"]
    rest = [p for p in pieces if p is not anchor]

    assert len(rest) == 4, "combo samples use exactly two synchronized extension stages"

    leaf_count = 0
    for piece in rest:
        info = _piece_axis_info(piece["arr"], axis)
        assert info is not None, "all non-anchor pieces must use only the anchor axis for red ports"
        piece["axis_info"] = info
        if info["port_count"] == 2:
            leaf_count += 1
    assert leaf_count == 2, "synchronized two-chain topology requires exactly two leaves"

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        key = (ar0 + rr, ac0 + cc)
        red_counts[key] = red_counts.get(key, 0) + 1

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return
        if len(rc) > target_unique_red:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions.append(cur)
            return

        if len(remaining) % 2 != 0:
            return

        frontiers = _group_frontiers(rc, axis)
        if frontiers is None or len(frontiers) != 2:
            return

        frontier_map = {grp["label"]: grp for grp in frontiers}
        target_port_count = 2 if len(remaining) == 2 else 4

        move_lists = {
            label: _candidate_moves_for_frontier(cur, remaining, rc, grp, target_port_count, bg)
            for label, grp in frontier_map.items()
        }
        if any(len(v) == 0 for v in move_lists.values()):
            return

        first_label, second_label = sorted(
            frontier_map,
            key=lambda label: (len(move_lists[label]), label),
        )

        for idx1, top1, left1 in move_lists[first_label]:
            piece1 = remaining[idx1]
            cur1 = _try_place(cur, piece1["arr"], top1, left1, bg)
            if cur1 is None:
                continue
            rc1 = _next_red_counts(rc, piece1, top1, left1)
            rem1 = remaining[:idx1] + remaining[idx1 + 1 :]

            for idx2, top2, left2 in _candidate_moves_for_frontier(
                cur1,
                rem1,
                rc1,
                frontier_map[second_label],
                target_port_count,
                bg,
            ):
                piece2 = rem1[idx2]
                cur2 = _try_place(cur1, piece2["arr"], top2, left2, bg)
                if cur2 is None:
                    continue
                rc2 = _next_red_counts(rc1, piece2, top2, left2)
                rem2 = rem1[:idx2] + rem1[idx2 + 1 :]

                if not rem2:
                    if len(rc2) == target_unique_red and all(v == 2 for v in rc2.values()):
                        solutions.append(cur2)
                        if len(solutions) >= 2:
                            return
                    continue

                next_frontiers = _group_frontiers(rc2, axis)
                if next_frontiers is None or len(next_frontiers) != 2:
                    continue
                dfs(cur2, rem2, rc2)
                if len(solutions) >= 2:
                    return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "synchronized jigsaw assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
CHAIN_LENGTH_CHOICES = (2,)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 16
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
BAR_SKIP_PROB = 0.35
STAGE_PAIR_TRIES = 80

# Anchor
ANCHOR_ORIENTATIONS = ("horizontal", "vertical")
ANCHOR_LINE_LENGTHS = (5, 6, 7)

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 500

_LAST_GENERATION_META: dict | None = None


def _new_node(kind: str, h: int, w: int, color: int, axis: str, top: int = 0, left: int = 0) -> dict:
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "axis": axis,
        "top": int(top),
        "left": int(left),
        "sides": set(),
        "ports": {},
    }


def _ensure_port(node: dict, side: str) -> None:
    if node["kind"] == "anchor" or side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = (span - 2) // 2


def _port_cells(node: dict, side: str) -> list[tuple[int, int]]:
    if node["kind"] == "anchor":
        h, w = node["shape"]
        if node["axis"] == "horizontal":
            if side == "left":
                return [(1, 0), (2, 0)]
            return [(1, w - 1), (2, w - 1)]
        if side == "top":
            return [(0, 1), (0, 2)]
        return [(h - 1, 1), (h - 1, 2)]

    h, w = node["shape"]
    k = node["ports"][side]
    if side == "top":
        return [(0, k), (0, k + 1)]
    if side == "bottom":
        return [(h - 1, k), (h - 1, k + 1)]
    if side == "left":
        return [(k, 0), (k + 1, 0)]
    return [(k, w - 1), (k + 1, w - 1)]


def _global_port(node: dict, side: str) -> list[tuple[int, int]]:
    tr, tc = node["top"], node["left"]
    return [(tr + r, tc + c) for r, c in _port_cells(node, side)]


def _bbox_of(node: dict) -> tuple[int, int, int, int]:
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union_bbox(nodes: list[dict]) -> tuple[int, int, int, int]:
    boxes = [_bbox_of(node) for node in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _draw_path(arr: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int, order: int) -> None:
    r, c = start
    er, ec = end
    axes = (
        [(r, er, c, True), (c, ec, er, False)]
        if order == 0
        else [(c, ec, r, False), (r, er, ec, True)]
    )
    for a_start, a_end, fixed, is_row in axes:
        step = 1 if a_end >= a_start else -1
        for v in range(a_start, a_end + step, step):
            rr, cc = (v, fixed) if is_row else (fixed, v)
            if arr[rr, cc] != 2:
                arr[rr, cc] = color


def _paint_bar(arr: np.ndarray, side: str, color: int, used: set[str]) -> None:
    h, w = arr.shape
    is_horizontal = side in ("top", "bottom")
    fixed = 0 if side in ("top", "left") else (h - 1 if side == "bottom" else w - 1)
    span = w if is_horizontal else h
    lo_conflict = "left" if is_horizontal else "top"
    hi_conflict = "right" if is_horizontal else "bottom"
    for i in range(span):
        if (i == 0 and lo_conflict in used) or (i == span - 1 and hi_conflict in used):
            continue
        if is_horizontal:
            arr[fixed, i] = color
        else:
            arr[i, fixed] = color


def _bar_endpoint(shape: tuple[int, int], side: str, hub: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    hr, hc = hub
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        return (row, min(max(hc, 1), w - 2))
    col = 0 if side == "left" else w - 1
    return (min(max(hr, 1), h - 2), col)


def _make_piece(node: dict) -> np.ndarray | None:
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    bar_sides = [side for side in SIDES if side not in used]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
            rr, cc = r + dr, c + dc
            arr[rr, cc] = color
            terminals.append((rr, cc))

    arr[hub] = color
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == 2)) != 2 * len(used):
        return None
    return arr


def _make_anchor(node: dict) -> np.ndarray | None:
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    if node["axis"] == "horizontal":
        arr[1, 1 : 1 + node["line_len"]] = 4
    else:
        arr[1 : 1 + node["line_len"], 1] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if _line_anchor_info(arr) is None:
        return None
    return arr


def _candidate_fits(nodes: list[dict], parent_i: int, child: dict) -> bool:
    cb = _bbox_of(child)
    if not _rects_overlap(cb, _bbox_of(nodes[parent_i])):
        return False
    for i, other in enumerate(nodes):
        if i == parent_i:
            continue
        if _rects_overlap(cb, _bbox_of(other)):
            return False
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    return (r1 - r0 + 1) <= MAX_ASSEMBLY_DIM and (c1 - c0 + 1) <= MAX_ASSEMBLY_DIM


def _make_chain_child(
    axis: str,
    label: str,
    color: int,
    is_leaf: bool,
    parent: dict,
    parent_i: int,
    nodes: list[dict],
) -> dict | None:
    dims = [
        (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
        for _ in range(PIECE_DIM_CANDIDATES)
    ]
    random.shuffle(dims)
    incoming = INCOMING_SIDE[label]

    for ph, pw in dims:
        child = _new_node("piece", ph, pw, color, axis)
        child["sides"].add(incoming)
        if not is_leaf:
            child["sides"].add(label)
        for side in child["sides"]:
            _ensure_port(child, side)

        parent_port = _global_port(parent, label)[0]
        child_port = _port_cells(child, incoming)[0]
        child["top"] = parent_port[0] - child_port[0]
        child["left"] = parent_port[1] - child_port[1]
        if _candidate_fits(nodes, parent_i, child):
            return child
    return None


def _grow_synchronous_chains() -> tuple[list[dict], dict] | None:
    axis = random.choice(ANCHOR_ORIENTATIONS)
    line_len = random.choice(ANCHOR_LINE_LENGTHS)
    chain_len = random.choice(CHAIN_LENGTH_CHOICES)
    stage_colors = random.sample(PIECE_COLORS, chain_len)

    ah, aw = (3, line_len + 2) if axis == "horizontal" else (line_len + 2, 3)
    anchor = _new_node("anchor", ah, aw, 4, axis)
    anchor["line_len"] = line_len
    anchor["sides"] = set(AXIS_FRONTIERS[axis])
    nodes = [anchor]
    frontier_parents = {label: 0 for label in AXIS_FRONTIERS[axis]}

    for depth, color in enumerate(stage_colors):
        is_leaf = depth == chain_len - 1
        placed = False
        for _ in range(STAGE_PAIR_TRIES):
            order = list(AXIS_FRONTIERS[axis])
            random.shuffle(order)
            trial_nodes = nodes[:]
            created = {}
            ok = True
            for label in order:
                parent_i = frontier_parents[label]
                child = _make_chain_child(axis, label, color, is_leaf, trial_nodes[parent_i], parent_i, trial_nodes)
                if child is None:
                    ok = False
                    break
                trial_nodes.append(child)
                created[label] = len(trial_nodes) - 1
            if ok:
                nodes = trial_nodes
                if not is_leaf:
                    frontier_parents = {label: created[label] for label in AXIS_FRONTIERS[axis]}
                placed = True
                break
        if not placed:
            return None

    meta = {
        "axis": axis,
        "chain_length": chain_len,
        "piece_count": 2 * chain_len,
        "stage_colors": tuple(stage_colors),
    }
    return nodes, meta


def _place_node_on_canvas(out: np.ndarray, node: dict, sr: int, sc: int) -> dict | None:
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    red_counts = {}
    mask = arr != 0
    rows, cols = np.nonzero(mask)
    g_rows, g_cols = rows + top, cols + left
    vals = arr[rows, cols]
    cur_vals = out[g_rows, g_cols]
    conflicts = (cur_vals != 0) & ~((cur_vals == 2) & (vals == 2))
    if np.any(conflicts):
        return None
    red_mask = vals == 2
    for gr, gc in zip(g_rows[red_mask], g_cols[red_mask]):
        key = (int(gr), int(gc))
        red_counts[key] = red_counts.get(key, 0) + 1
    paint_mask = cur_vals == 0
    out[g_rows[paint_mask], g_cols[paint_mask]] = vals[paint_mask]
    return red_counts


def _assemble(nodes: list[dict], h: int, w: int) -> np.ndarray | None:
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for key, value in rc.items():
            all_red_counts[key] = all_red_counts.get(key, 0) + value

    total_ports = sum(int(np.sum(node["arr"] == 2)) for node in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    return out


def _scatter(nodes: list[dict], h: int, w: int) -> np.ndarray | None:
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    rest = nodes[1:]
    random.shuffle(rest)
    for node in rest:
        try:
            r, c = random_free_location(
                inp,
                node["arr"],
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, node["arr"], r, c, transparent=0)
    return inp


def _gen_attempt() -> np.ndarray | None:
    global _LAST_GENERATION_META

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grown = _grow_synchronous_chains()
    if grown is None:
        return None
    nodes, meta = grown
    out = _assemble(nodes, h, w)
    if out is None:
        return None
    inp = _scatter(nodes, h, w)
    if inp is None:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    anchor_piece = next(piece for piece in _extract_pieces(inp, 0) if _line_anchor_info(piece["arr"]) is not None)
    anchor_piece["anchor_info"] = _line_anchor_info(anchor_piece["arr"])
    end_usage = _anchor_end_extensions(out, anchor_piece)
    if not all(end_usage):
        return None

    _LAST_GENERATION_META = {
        **meta,
        "end_usage": end_usage,
        "grid_shape": (h, w),
    }
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
