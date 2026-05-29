# Combo variant of cbebaa4b (index 26)
# combo_spec: D1/3+D7/2+D10/2
# Name: 多色连接器 + 有向端口 + 链式传递

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "solvers"
    if (_candidate / "arc_utils.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) concretization + behavior trace:
# - Preserve the original scene shell: black background, one fixed yellow anchor, and 4-6
#   scattered non-anchor sprites that assemble by translation only.
# - Fuse the three requested changes into one connector system:
#     family 0: output=2, input=3
#     family 1: output=5, input=6
#     family 2: output=7, input=8
#   A legal seam must therefore satisfy three constraints at once:
#     1. it is the current chain tail (D7),
#     2. the existing tail cells are output ports and the incoming piece brings input ports (D10),
#     3. both sides belong to the same family (D1).
# - Small trace:
#     original: after the anchor is placed, any piece red pair may overlap any open red pair, and
#     later steps may branch from any remaining open red on the assembled object.
#     fused: anchor exposes exactly one output pair, say family 1 using color 5. The first piece
#     must place its family-1 input pair (color 6) on that exact pair. If that piece exposes a
#     family-2 output pair (color 7), then the next piece must specifically bring a family-2 input
#     pair (color 8) to that new tail. Neither another family nor the wrong role is legal, and no
#     piece may attach back to the anchor once its tail has been consumed.
# - Difference from the original: matching is no longer "same red anywhere". It is now ordered
#   relay assembly with a visible role split and per-edge family labels.
#
# b) complexity calibration:
# - Use exactly three possible connector families, but require at least two families in every
#   generated sample. This keeps the color rule observable without exploding the hypothesis space.
# - Restrict every connector group to the same simple 2-cell boundary segment as the source task.
#   That keeps the additional difficulty in the logic rather than in new geometry.
# - Keep the topology to a simple path: anchor -> piece1 -> ... -> pieceN. Intermediate pieces
#   therefore show one input side and one output side, which makes the directional relay visible.
# - Require at least one internal relay piece whose input family differs from its output family;
#   otherwise the multi-family requirement would be visually weaker.
#
# c) implicit assumption audit:
# - Original _extract_pieces assumed "all connectors are red". That fails here; extraction must
#   recover connector groups, then decode both family and role from the pixel colors.
# - Original _try_place allowed same-color overlap. That is now wrong twice over: same-color
#   connector overlap is illegal because roles differ, and same-color body overlap remains illegal.
#   The only allowed overlap is opposite-role / same-family connector overlap.
# - Original DFS state keyed ports only by occupancy count. That loses both family and direction.
#   The fused state stores (family, out_count, in_count) per assembled coordinate.
# - Original search expanded from any unpaired connector on the whole assembly. The chain variant
#   invalidates that assumption; only the unique open output group may be extended next.
# - Original generator allowed arbitrary trees and anchor degree 1-2. That would silently weaken
#   D7, so generation is restricted to a path with exactly one anchor output group.
#
# d) placement legality:
# - Output-first generation is kept. The solved chain is built in local coordinates first, shifted
#   into the grid, validated, and only then scattered into an input.
# - Every new child bbox must overlap only its predecessor bbox and no earlier non-parent bbox.
# - During assembly, only opposite-role connectors from the same family may overlap; every other
#   nonzero collision is rejected.
#
# e) visual observability:
# - The input visibly separates connector roles because each family uses two distinct colors:
#   one for output and one for input. A human can see that pieces are not attached by arbitrary
#   same-color bars; they need a role-complementary pair.
# - The solved output is a line-like relay, not a star, so the chaining rule is also visible.
# - Requiring at least two families and at least one family change along the chain makes the
#   multi-family constraint inferable from a few samples rather than hidden in rare cases.
#
# f) cleanliness:
# - Body colors exclude all connector colors and the yellow anchor color, so ports remain salient.
# - Connector groups stay on outer edges and are only 2 cells long, keeping overlaps localized.
# - The path topology avoids dense hubs and reduces unreadable overlap clusters near the anchor.
#
# g) inferability of the color change:
# - There are only three family labels and they always appear through the same visible role pairs.
# - The family condition is structurally supported by adjacency: a seam is valid only where an
#   output-colored pair and its matching input-colored pair coincide at the active tail.
# - Because at least one internal piece changes family between its incoming and outgoing side, the
#   examples show that the chain propagates direction while the family requirement is evaluated
#   separately at each seam.
#
# new solver assumptions and generator guarantees:
# - Assumption: the anchor has exactly one output connector group and no input connector group.
#   Generator guarantees this.
# - Assumption: every non-anchor piece has exactly one input group and at most one output group.
#   Generator guarantees this by building a simple path.
# - Assumption: every connector group is a 2-cell 4-connected boundary segment of uniform color.
#   Generator guarantees this via the side-port construction.
# - Assumption: the only unresolved frontier after each legal step is either a single output pair
#   or nothing at the end. Generator guarantees this path topology, and main() asserts it.


ANCHOR_COLOR = 4

MATCH_FAMILIES = (
    {"out": 2, "in": 3},
    {"out": 5, "in": 6},
    {"out": 7, "in": 8},
)
OUT_COLORS = tuple(pair["out"] for pair in MATCH_FAMILIES)
IN_COLORS = tuple(pair["in"] for pair in MATCH_FAMILIES)
CONNECTOR_COLORS = OUT_COLORS + IN_COLORS
CONNECTOR_SET = set(CONNECTOR_COLORS)
BODY_COLORS = [1, 9]

COLOR_TO_META = {
    pair["out"]: (family, "out") for family, pair in enumerate(MATCH_FAMILIES)
} | {
    pair["in"]: (family, "in") for family, pair in enumerate(MATCH_FAMILIES)
}


def _is_connector_color(v: np.ndarray | int) -> np.ndarray | np.bool_:
    return np.isin(v, CONNECTOR_COLORS)


def _sorted_group(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _group_is_pair(group: tuple[tuple[int, int], ...]) -> bool:
    if len(group) != 2:
        return False
    (r0, c0), (r1, c1) = group
    return abs(r0 - r1) + abs(c0 - c1) == 1


def _connector_color(family: int, role: str) -> int:
    return int(MATCH_FAMILIES[family][role])


def _compatible_connector_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(a, dtype=bool)
    for pair in MATCH_FAMILIES:
        out_color = int(pair["out"])
        in_color = int(pair["in"])
        mask |= ((a == out_color) & (b == in_color)) | ((a == in_color) & (b == out_color))
    return mask


def _find_connector_groups(arr: np.ndarray) -> list[dict]:
    groups = []
    for pts in components_mask(_is_connector_color(arr), connectivity=4):
        coords = _sorted_group([(int(r), int(c)) for r, c in pts])
        colors = {int(arr[r, c]) for r, c in coords}
        assert len(colors) == 1, "connector group must have uniform color"
        color = int(next(iter(colors)))
        family, role = COLOR_TO_META[color]
        groups.append(
            {
                "coords": coords,
                "family": int(family),
                "role": role,
                "color": color,
            }
        )
    groups.sort(key=lambda g: (g["role"], g["family"], g["coords"]))
    return groups


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        groups = _find_connector_groups(arr)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "groups": groups,
                "input_groups": [g for g in groups if g["role"] == "in"],
                "output_groups": [g for g in groups if g["role"] == "out"],
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

    compatible = _compatible_connector_overlap(sub, arr)
    conflict = mask & (sub != bg) & ~compatible
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    paint_mask = mask & (sub == bg)
    merge_mask = mask & compatible
    out_sub[paint_mask] = arr[paint_mask]
    for pair in MATCH_FAMILIES:
        out_color = int(pair["out"])
        in_color = int(pair["in"])
        fam_merge = merge_mask & (
            ((sub == out_color) & (arr == in_color))
            | ((sub == in_color) & (arr == out_color))
        )
        out_sub[fam_merge] = out_color
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _match_group_translations(
    local_group: tuple[tuple[int, int], ...],
    target_group: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    target = set(target_group)
    out: set[tuple[int, int]] = set()
    for lr, lc in local_group:
        for tr, tc in target_group:
            top = int(tr - lr)
            left = int(tc - lc)
            shifted = {(int(r + top), int(c + left)) for r, c in local_group}
            if shifted == target:
                out.add((top, left))
    return sorted(out)


def _abs_group(
    group: tuple[tuple[int, int], ...], top: int, left: int
) -> tuple[tuple[int, int], ...]:
    return _sorted_group([(r + top, c + left) for r, c in group])


def _add_piece_state(
    state: dict[tuple[int, int], tuple[int, int, int]],
    piece: dict,
    top: int,
    left: int,
) -> dict[tuple[int, int], tuple[int, int, int]] | None:
    nxt = dict(state)
    for group in piece["groups"]:
        family = int(group["family"])
        role = group["role"]
        for rr, cc in group["coords"]:
            key = (int(top + rr), int(left + cc))
            prev = nxt.get(key)
            if prev is None:
                out_n = 0
                in_n = 0
            else:
                prev_family, out_n, in_n = prev
                if prev_family != family:
                    return None
            if role == "out":
                if out_n >= 1:
                    return None
                nxt[key] = (family, out_n + 1, in_n)
            else:
                if in_n >= 1:
                    return None
                nxt[key] = (family, out_n, in_n + 1)
    return nxt


def _state_tail(
    state: dict[tuple[int, int], tuple[int, int, int]]
) -> tuple[bool, tuple[tuple[int, int], ...] | None, int | None]:
    unresolved_out: list[tuple[tuple[int, int], int]] = []
    for coord, (family, out_n, in_n) in state.items():
        if (out_n, in_n) == (1, 1):
            continue
        if (out_n, in_n) == (1, 0):
            unresolved_out.append((coord, int(family)))
            continue
        return False, None, None

    if not unresolved_out:
        return True, None, None
    if len(unresolved_out) != 2:
        return False, None, None
    families = {family for _coord, family in unresolved_out}
    if len(families) != 1:
        return False, None, None
    tail = _sorted_group([coord for coord, _family in unresolved_out])
    if not _group_is_pair(tail):
        return False, None, None
    return True, tail, int(next(iter(families)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered sprites into a directed multi-color relay chain rooted at the yellow anchor.

    Input structure:
    - bg = black (0). Several colored sprites are scattered without touching.
    - Exactly one anchor sprite contains a yellow (4) rectangle and one output connector pair.
    - Connector pixels come in three family-specific role pairs:
      2->3, 5->6, and 7->8 where the smaller color is an output port and the larger color is the matching input port.

    Rule:
    - The anchor stays fixed and starts the chain.
    - At each step, only the current tail output pair may be extended, and the next piece must place a matching-family input pair on that exact location.
    - If the piece has another connector pair, that pair becomes the new tail; continue until all pieces are consumed and every connector cell is paired exactly once.

    Key insight:
    - The open frontier is a single directed colored pair, not an arbitrary open connector. Solving requires tracking three things together: which pair is still open, which role it needs next, and which color family is allowed there.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if np.any(p["arr"] == ANCHOR_COLOR)]
    assert len(anchors) == 1, "input must have exactly one yellow anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    assert len(anchor["input_groups"]) == 0, "anchor must not expose input ports"
    assert len(anchor["output_groups"]) == 1, "anchor must expose exactly one output port group"
    assert _group_is_pair(anchor["output_groups"][0]["coords"]), "anchor connector must be a 2-cell pair"
    for p in rest:
        assert len(p["input_groups"]) == 1, "each non-anchor piece must have exactly one input group"
        assert len(p["output_groups"]) in (0, 1), "each non-anchor piece must have at most one output group"
        assert len(p["groups"]) in (1, 2), "each non-anchor piece must have one or two connector groups"
        for group in p["groups"]:
            assert _group_is_pair(group["coords"]), "each connector group must be a 2-cell pair"

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(2 * len(p["groups"]) for p in pieces))
    assert total_ports % 2 == 0, "total connector cells must be even"
    target_unique_ports = total_ports // 2

    state = _add_piece_state({}, anchor, ar0, ac0)
    assert state is not None, "anchor state must be valid"
    ok, tail, tail_family = _state_tail(state)
    assert ok and tail is not None and tail_family is not None, "anchor must create one initial tail"

    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        cur_state: dict[tuple[int, int], tuple[int, int, int]],
        active_tail: tuple[tuple[int, int], ...] | None,
        active_family: int | None,
    ) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if active_tail is None and len(cur_state) == target_unique_ports:
                if all((out_n, in_n) == (1, 1) for _family, out_n, in_n in cur_state.values()):
                    solutions.append(cur)
            return

        if active_tail is None or active_family is None:
            return
        if len(cur_state) > target_unique_ports:
            return

        moves: list[tuple[int, int, int, dict, tuple[tuple[int, int], ...] | None, int | None]] = []
        for idx, piece in enumerate(remaining):
            matching_inputs = [
                group for group in piece["input_groups"] if int(group["family"]) == int(active_family)
            ]
            if not matching_inputs:
                continue
            ph, pw = piece["shape"]
            for group in matching_inputs:
                for top, left in _match_group_translations(group["coords"], active_tail):
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    next_state = _add_piece_state(cur_state, piece, top, left)
                    if next_state is None or len(next_state) > target_unique_ports:
                        continue
                    valid_tail, next_tail, next_family = _state_tail(next_state)
                    if not valid_tail:
                        continue
                    if remaining[:idx] + remaining[idx + 1 :]:
                        if next_tail is None or next_family is None:
                            continue
                    elif next_tail is not None:
                        continue
                    moves.append((idx, top, left, next_state, next_tail, next_family))

        dedup: dict[tuple[int, int, int], tuple[dict, tuple[tuple[int, int], ...] | None, int | None]] = {}
        for idx, top, left, next_state, next_tail, next_family in moves:
            dedup[(idx, top, left)] = (next_state, next_tail, next_family)

        ordered = sorted(
            dedup.items(),
            key=lambda item: (-remaining[item[0][0]]["area"], item[0][1], item[0][2], item[0][0]),
        )
        for (idx, top, left), (next_state, next_tail, next_family) in ordered:
            piece = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            nxt = _try_place(cur, piece["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_state, next_tail, next_family)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, state, tail, tail_family)
    assert len(solutions) == 1, "fused chain assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
BAR_SKIP_PROB = 0.35

ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]

SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 500

SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind: str, h: int, w: int, color: int, top: int = 0, left: int = 0, cap: int = 2) -> dict:
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "sides": set(),
        "ports": {},
        "roles": {},
        "families": {},
    }


def _ensure_port(node: dict, side: str) -> None:
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node: dict, side: str) -> list[tuple[int, int]]:
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
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
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
            if int(arr[rr, cc]) not in CONNECTOR_SET:
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
        family = int(node["families"][side])
        role = node["roles"][side]
        port_color = _connector_color(family, role)
        dr, dc = INWARD[side]
        for r, c in _port_cells(node, side):
            arr[r, c] = port_color
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
    if int(np.sum(_is_connector_color(arr))) != 2 * len(used):
        return None
    return arr


def _make_anchor(node: dict) -> np.ndarray:
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = ANCHOR_COLOR
    for side in node["sides"]:
        family = int(node["families"][side])
        role = node["roles"][side]
        port_color = _connector_color(family, role)
        for r, c in _port_cells(node, side):
            arr[r, c] = port_color
    return arr


def _candidate_fits(nodes: list[dict], parent_i: int, child: dict) -> bool:
    child_box = _bbox_of(child)
    if not _rects_overlap(child_box, _bbox_of(nodes[parent_i])):
        return False
    for i, other in enumerate(nodes):
        if i == parent_i:
            continue
        if _rects_overlap(child_box, _bbox_of(other)):
            return False
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    return (r1 - r0 + 1) <= MAX_ASSEMBLY_DIM and (c1 - c0 + 1) <= MAX_ASSEMBLY_DIM


def _assign_edge_families(nodes: list[dict], edges: list[tuple[int, str, int, str]]) -> bool:
    family_count = random.randint(2, len(MATCH_FAMILIES))
    active_families = random.sample(range(len(MATCH_FAMILIES)), family_count)
    edge_families = active_families[:] + [
        random.choice(active_families) for _ in range(len(edges) - family_count)
    ]
    random.shuffle(edge_families)
    if len(set(edge_families)) < 2:
        return False

    for (parent_i, out_side, child_i, in_side), family in zip(edges, edge_families):
        nodes[parent_i]["roles"][out_side] = "out"
        nodes[parent_i]["families"][out_side] = int(family)
        nodes[child_i]["roles"][in_side] = "in"
        nodes[child_i]["families"][in_side] = int(family)

    relay_changes = 0
    for node in nodes[1:-1]:
        families = [int(node["families"][side]) for side in node["roles"]]
        if len(families) == 2 and families[0] != families[1]:
            relay_changes += 1
    return relay_changes > 0


def _grow_chain() -> list[dict] | None:
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, ANCHOR_COLOR, cap=1)
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    edges: list[tuple[int, str, int, str]] = []
    tail_i = 0
    n_pieces = random.randint(*PIECE_COUNT_RANGE)

    for _ in range(n_pieces):
        parent = nodes[tail_i]
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            sides = [side for side in SIDES if side not in parent["sides"]]
            random.shuffle(sides)
            for side in sides:
                child = _new_node("piece", ph, pw, random.choice(BODY_COLORS), cap=2)
                incoming = OPPOSITE[side]
                _ensure_port(parent, side)
                _ensure_port(child, incoming)
                pr, pc = _global_port(parent, side)[0]
                cr, cc = _port_cells(child, incoming)[0]
                child["top"] = int(pr - cr)
                child["left"] = int(pc - cc)
                child["sides"].add(incoming)
                if not _candidate_fits(nodes, tail_i, child):
                    continue
                parent["sides"].add(side)
                nodes.append(child)
                child_i = len(nodes) - 1
                edges.append((tail_i, side, child_i, incoming))
                tail_i = child_i
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if not _assign_edge_families(nodes, edges):
        return None
    return nodes


def _node_port_state(node: dict, top: int, left: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    state = {}
    for side, role in node["roles"].items():
        family = int(node["families"][side])
        for rr, cc in _port_cells(node, side):
            key = (int(top + rr), int(left + cc))
            prev = state.get(key)
            if prev is None:
                out_n = 0
                in_n = 0
            else:
                prev_family, out_n, in_n = prev
                if prev_family != family:
                    raise AssertionError("one node cannot place different families on one connector cell")
            if role == "out":
                state[key] = (family, out_n + 1, in_n)
            else:
                state[key] = (family, out_n, in_n + 1)
    return state


def _place_node_on_canvas(
    out: np.ndarray, node: dict, sr: int, sc: int
) -> dict[tuple[int, int], tuple[int, int, int]] | None:
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (int(top), int(left))

    ph, pw = arr.shape
    sub = out[top : top + ph, left : left + pw]
    mask = arr != 0
    compatible = _compatible_connector_overlap(sub, arr)
    conflicts = mask & (sub != 0) & ~compatible
    if np.any(conflicts):
        return None

    paint_mask = mask & (sub == 0)
    merge_mask = mask & compatible
    sub[paint_mask] = arr[paint_mask]
    for pair in MATCH_FAMILIES:
        out_color = int(pair["out"])
        in_color = int(pair["in"])
        fam_merge = merge_mask & (
            ((sub == out_color) & (arr == in_color))
            | ((sub == in_color) & (arr == out_color))
        )
        sub[fam_merge] = out_color
    out[top : top + ph, left : left + pw] = sub
    return _node_port_state(node, top, left)


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
    full_state: dict[tuple[int, int], tuple[int, int, int]] = {}
    for node in nodes:
        node_state = _place_node_on_canvas(out, node, sr, sc)
        if node_state is None:
            return None
        for key, (family, out_n, in_n) in node_state.items():
            prev = full_state.get(key)
            if prev is None:
                full_state[key] = (family, out_n, in_n)
                continue
            prev_family, prev_out, prev_in = prev
            if prev_family != family:
                return None
            full_state[key] = (family, prev_out + out_n, prev_in + in_n)

    total_ports = sum(2 * len(node["roles"]) for node in nodes)
    if len(full_state) * 2 != total_ports:
        return None
    if any((out_n, in_n) != (1, 1) for _family, out_n, in_n in full_state.values()):
        return None
    if len({int(family) for family, _out_n, _in_n in full_state.values()}) < 2:
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
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_chain()
    if nodes is None:
        return None
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
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused cbebaa4b input")
