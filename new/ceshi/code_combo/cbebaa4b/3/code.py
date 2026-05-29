# Combo variant of cbebaa4b (index 3)
# combo_spec: D1/4+D10/2
# Name: 互补色连接 + 有向连接端口

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


# Design analysis
# ---------------
# combo_variant:
#   concreteization:
#     - Keep the original output-first scaffold: a fixed yellow anchor stays in place,
#       4-6 other sprites are scattered in the input, and the output is the unique
#       assembled structure obtained by translation only.
#     - Realize "complementary color + directed port" with four connector colors:
#         output ports  : red(2), teal(8)
#         input ports   : blue(1), orange(7)
#         complement map: red(2)<->blue(1), teal(8)<->orange(7)
#     - A legal seam therefore needs two simultaneous checks:
#         1. roles must be opposite (output onto input),
#         2. colors must be the designated complementary pair.
#     - Completed seams are rendered in the output-side color, so a red/blue join
#       appears red in the final assembly and a teal/orange join appears teal.
#       This keeps the output deterministic and makes role visible across examples:
#       input colors disappear when consumed, output colors survive.
#   behavior_trace:
#     - Original local step:
#         if the assembly exposes a red cell, any remaining red cell may overlap it.
#     - D1-only local step:
#         if the assembly exposes red, only blue may overlap it.
#     - D10-only local step:
#         if the assembly exposes an output, only an input may overlap it.
#     - Fused local step:
#         if the assembly exposes red-output, only blue-input may overlap it;
#         if it exposes orange-input, only teal-output may overlap it.
#       Example: a piece with teal output cannot close an orange-input frontier of
#       the wrong pair, and a blue input cannot attach to another blue input even
#       though both are connector colors.
#   complexity_calibration:
#     - Use exactly two complementary pairs, repeated across samples, instead of an
#       arbitrary many-color relation. That is enough to force rule discovery while
#       remaining learnable from 2-4 examples.
#     - Body colors are restricted to {3,5,6,9}, so connector colors never double
#       as body paint and remain easy to isolate visually.
#     - The generator rejects samples unless both complementary pairs appear, the
#       anchor uses both roles, and at least one non-anchor piece mixes input and
#       output ports. This prevents degeneration into a single-pair or single-role
#       puzzle that would look too close to a simpler variant.
#   hidden_assumptions_review:
#     - Original _extract_pieces assumed all connectors were red; now it must record
#       local connector coordinates together with their role/color.
#     - Original _try_place allowed same-color red/red overlap; now overlap is legal
#       only when one side is output, the other is input, and their colors satisfy
#       the fixed complement map.
#     - Original DFS state tracked only "how many reds landed here". That collapses
#       under the fused rule, so the new state tracks per-coordinate output color and
#       input color separately.
#     - Original move generation aligned any red with any open red. That is no longer
#       sound: each open coordinate now demands one exact missing signature, such as
#       "blue input" or "teal output".
#   placement_legality:
#     - The generator still grows a tree in local coordinates, renders each node,
#       assembles the output, and only then scatters the non-anchor pieces.
#     - Candidate children must overlap only their parent bbox, avoid every other
#       bbox, stay within the global size cap, and produce only legal connector
#       overlaps during final rendering.
#     - Each assembled coordinate may receive at most one output contributor and at
#       most one input contributor; final seams must be exactly one of each.
#   visual_observability:
#     - Inputs visibly contain four connector colors, but the outputs retain only
#       red/teal seam colors. Across examples this shows two facts at once:
#         1. blue/orange are not body colors and disappear only when attached,
#         2. red pairs specifically with blue, while teal pairs specifically with
#            orange, not with any opposite-role color.
#     - Because some sprites have mixed roles, solvers cannot reduce the task to
#       "all warm colors attach outward" or any one-sided star assembly heuristic.
#   neatness_and_inference:
#     - No extra glyphs, labels, or hidden ordering cues are introduced.
#     - The fused change remains local to the connector pixels, so the output still
#       looks like a clean single connected jigsaw rather than a cluttered overlay.


ANCHOR_COLOR = 4
OUTPUT_COLORS = (2, 8)
INPUT_COLORS = (1, 7)
CONNECTOR_COLORS = OUTPUT_COLORS + INPUT_COLORS
COMP_COLOR = {2: 1, 1: 2, 8: 7, 7: 8}
ROLE_BY_COLOR = {2: "out", 8: "out", 1: "in", 7: "in"}


def _is_connector_color(v: np.ndarray | int) -> np.ndarray | bool:
    return np.isin(v, CONNECTOR_COLORS)


def _is_output_color(v: np.ndarray | int) -> np.ndarray | bool:
    return np.isin(v, OUTPUT_COLORS)


def _is_input_color(v: np.ndarray | int) -> np.ndarray | bool:
    return np.isin(v, INPUT_COLORS)


def _compatible_connector_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _is_connector_color(a) & _is_connector_color(b) & (a == _complement_grid(b))


def _complement_grid(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape, -1, dtype=int)
    for color, comp in COMP_COLOR.items():
        out[arr == color] = comp
    return out


def _merge_connector_colors(existing: np.ndarray, incoming: np.ndarray) -> np.ndarray:
    return np.where(_is_output_color(existing), existing, incoming)


def _scan_ports(arr: np.ndarray) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    outputs = [(int(r), int(c), int(arr[r, c])) for r, c in np.argwhere(_is_output_color(arr))]
    inputs = [(int(r), int(c), int(arr[r, c])) for r, c in np.argwhere(_is_input_color(arr))]
    return outputs, inputs


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        outputs, inputs = _scan_ports(arr)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "outputs": outputs,
                "inputs": inputs,
                "ports": outputs + inputs,
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
    out_sub[merge_mask] = _merge_connector_colors(sub[merge_mask], arr[merge_mask])
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _add_ports(
    state: dict[tuple[int, int], tuple[int, int]],
    ports: list[tuple[int, int, int]],
    top: int,
    left: int,
) -> dict[tuple[int, int], tuple[int, int]] | None:
    nxt = dict(state)
    for rr, cc, color in ports:
        key = (top + rr, left + cc)
        out_color, in_color = nxt.get(key, (0, 0))
        role = ROLE_BY_COLOR[color]
        if role == "out":
            if out_color != 0:
                return None
            if in_color != 0 and COMP_COLOR[color] != in_color:
                return None
            nxt[key] = (color, in_color)
        else:
            if in_color != 0:
                return None
            if out_color != 0 and COMP_COLOR[color] != out_color:
                return None
            nxt[key] = (out_color, color)
    return nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered sprites around the fixed yellow anchor by matching directed complementary ports.

    Input structure:
    - bg = black (0). Several colored sprites are separated on the grid.
    - Exactly one sprite is the anchor: it contains the yellow (4) core and stays fixed.
    - Boundary connector cells use four special colors: red (2) / teal (8) are output ports, blue (1) / orange (7) are input ports.

    Rule:
    - Translate each non-anchor sprite without rotating it.
    - A legal overlap happens only when one sprite contributes an output port and the other contributes the complementary input color at the same coordinate: red with blue, teal with orange.
    - Every connector cell must be consumed exactly once, so each final seam is one output cell plus one matching input cell.

    Key insight: An open frontier coordinate now carries two requirements at once: a missing role and a missing complementary color. Seeing an exposed orange input does not mean “any output fits”; it specifically demands a teal output.
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

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["ports"]) for p in pieces))
    assert total_ports % 2 == 0, "total connector cells must be even"
    target_unique_ports = total_ports // 2

    port_state = _add_ports({}, anchor["ports"], ar0, ac0)
    assert port_state is not None, "anchor ports must be self-consistent"

    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        state: dict[tuple[int, int], tuple[int, int]],
    ) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(state) == target_unique_ports and all(out_c and in_c for out_c, in_c in state.values()):
                solutions.append(cur)
            return

        if len(state) > target_unique_ports:
            return

        needs: list[tuple[tuple[int, int], str, int]] = []
        for coord, (out_color, in_color) in state.items():
            if out_color and in_color:
                continue
            if out_color:
                needs.append((coord, "in", COMP_COLOR[out_color]))
            elif in_color:
                needs.append((coord, "out", COMP_COLOR[in_color]))
            else:
                return
        if not needs:
            return

        best_moves = None
        for target, needed_role, needed_color in needs:
            tr, tc = target
            moves = []
            for idx, p in enumerate(remaining):
                local_ports = p["inputs"] if needed_role == "in" else p["outputs"]
                candidates = [port for port in local_ports if port[2] == needed_color]
                if not candidates:
                    continue
                ph, pw = p["shape"]
                for pr, pc, _color in candidates:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue

                    bad = False
                    overlap = 0
                    for rr, cc, port_color in p["ports"]:
                        gr, gc = top + rr, left + cc
                        out_color, in_color = state.get((gr, gc), (0, 0))
                        role = ROLE_BY_COLOR[port_color]
                        if role == "out":
                            if out_color:
                                bad = True
                                break
                            if in_color:
                                if COMP_COLOR[port_color] != in_color:
                                    bad = True
                                    break
                                overlap += 1
                        else:
                            if in_color:
                                bad = True
                                break
                            if out_color:
                                if COMP_COLOR[port_color] != out_color:
                                    bad = True
                                    break
                                overlap += 1
                    if bad:
                        continue
                    moves.append((idx, top, left, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible directed complementary move"

        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
            p = remaining[idx]
            next_state = _add_ports(state, p["ports"], top, left)
            if next_state is None:
                continue

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue

            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            dfs(nxt, next_remaining, next_state)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, port_state)
    assert len(solutions) == 1, "directed complementary jigsaw must have exactly one valid solution"
    return solutions[0]


# --------------- generator ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_BODY_COLORS = [3, 5, 6, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [2]

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 600

# Geometry
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP):
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
        "port_colors": {},
    }


def _ensure_port(node, side):
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node, side):
    h, w = node["shape"]
    k = node["ports"][side]
    if side == "top":
        return [(0, k), (0, k + 1)]
    if side == "bottom":
        return [(h - 1, k), (h - 1, k + 1)]
    if side == "left":
        return [(k, 0), (k + 1, 0)]
    return [(k, w - 1), (k + 1, w - 1)]


def _global_port(node, side):
    tr, tc = node["top"], node["left"]
    return [(tr + r, tc + c) for r, c in _port_cells(node, side)]


def _bbox_of(node):
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _rects_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union_bbox(nodes):
    boxes = [_bbox_of(n) for n in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _draw_path(arr, start, end, color, order):
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
            if not _is_connector_color(arr[rr, cc]):
                arr[rr, cc] = color


def _paint_bar(arr, side, color, used):
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


def _bar_endpoint(shape, side, hub):
    h, w = shape
    hr, hc = hub
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        return (row, min(max(hc, 1), w - 2))
    col = 0 if side == "left" else w - 1
    return (min(max(hr, 1), h - 2), col)


def _make_piece(node):
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    bar_sides = [s for s in SIDES if s not in used]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        port_color = node["port_colors"][side]
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


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = ANCHOR_COLOR
    for side in node["sides"]:
        port_color = node["port_colors"][side]
        for r, c in _port_cells(node, side):
            arr[r, c] = port_color
    return arr


def _candidate_fits(nodes, parent_i, child):
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


def _choose_parent_role(parent):
    existing = list(parent["roles"].values())
    if not existing:
        return random.choice(("in", "out"))
    if len(existing) == 1:
        return "out" if existing[0] == "in" else "in"
    return random.choice(("in", "out"))


def _assign_edge_signature(parent, side, child, incoming):
    edge_out_color = random.choice(OUTPUT_COLORS)
    parent_role = _choose_parent_role(parent)
    child_role = "in" if parent_role == "out" else "out"

    if parent_role == "out":
        parent_color = edge_out_color
        child_color = COMP_COLOR[edge_out_color]
    else:
        parent_color = COMP_COLOR[edge_out_color]
        child_color = edge_out_color

    child["roles"][incoming] = child_role
    child["port_colors"][incoming] = child_color
    return parent_role, parent_color


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node(
        "anchor", bh + 2, bw + 2, ANCHOR_COLOR, cap=random.choice(ANCHOR_CAP_CHOICES)
    )
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = [random.choice(PIECE_BODY_COLORS) for _ in range(n_pieces)]

    for i in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            if i < 2:
                parents = [0]
            elif i == 2:
                parents = [
                    j
                    for j in range(1, min(len(nodes), 3))
                    if len(nodes[j]["sides"]) < nodes[j]["cap"]
                ]
            else:
                parents = [j for j, n in enumerate(nodes) if len(n["sides"]) < n["cap"]]
            random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = [s for s in SIDES if s not in parent["sides"]]
                random.shuffle(sides)
                for side in sides:
                    child = _new_node("piece", ph, pw, colors[i])
                    incoming = OPPOSITE[side]
                    _ensure_port(parent, side)
                    _ensure_port(child, incoming)

                    pr, pc = _global_port(parent, side)[0]
                    cr, cc = _port_cells(child, incoming)[0]
                    child["top"] = pr - cr
                    child["left"] = pc - cc
                    child["sides"].add(incoming)

                    parent_role, parent_color = _assign_edge_signature(parent, side, child, incoming)
                    if not _candidate_fits(nodes, parent_i, child):
                        continue

                    parent["sides"].add(side)
                    parent["roles"][side] = parent_role
                    parent["port_colors"][side] = parent_color
                    nodes.append(child)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    all_roles = {role for node in nodes for role in node["roles"].values()}
    anchor_roles = set(anchor["roles"].values())
    mixed_piece_count = sum(
        len(set(node["roles"].values())) == 2 for node in nodes[1:] if node["roles"]
    )
    used_output_colors = {
        color
        for node in nodes
        for color in node["port_colors"].values()
        if color in OUTPUT_COLORS
    }
    if all_roles != {"in", "out"}:
        return None
    if anchor_roles != {"in", "out"}:
        return None
    if mixed_piece_count == 0:
        return None
    if used_output_colors != set(OUTPUT_COLORS):
        return None
    return nodes


def _place_node_on_canvas(out, node, sr, sc):
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)

    ph, pw = arr.shape
    sub = out[top : top + ph, left : left + pw]
    mask = arr != 0
    compatible = _compatible_connector_overlap(sub, arr)
    conflicts = mask & (sub != 0) & ~compatible
    if np.any(conflicts):
        return None

    piece_state = _add_ports({}, node["ports_list"], top, left)
    if piece_state is None:
        return None

    paint_mask = mask & (sub == 0)
    merge_mask = mask & compatible
    sub[paint_mask] = arr[paint_mask]
    sub[merge_mask] = _merge_connector_colors(sub[merge_mask], arr[merge_mask])
    out[top : top + ph, left : left + pw] = sub
    return piece_state


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None
        outputs, inputs = _scan_ports(node["arr"])
        node["ports_list"] = outputs + inputs

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    all_state: dict[tuple[int, int], tuple[int, int]] = {}
    for node in nodes:
        piece_state = _place_node_on_canvas(out, node, sr, sc)
        if piece_state is None:
            return None
        merged_state = _add_ports(all_state, node["ports_list"], node["pos"][0], node["pos"][1])
        if merged_state is None:
            return None
        all_state = merged_state

    total_ports = sum(len(node["ports_list"]) for node in nodes)
    if len(all_state) * 2 != total_ports:
        return None
    if any(not (out_c and in_c) for out_c, in_c in all_state.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    if not set(np.unique(out)).issubset({0, 2, 3, 4, 5, 6, 8, 9}):
        return None
    if not (set(np.unique(out)) & set(OUTPUT_COLORS)) == set(OUTPUT_COLORS):
        return None
    return out


def _scatter(nodes, h, w):
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


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_tree()
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
    raise RuntimeError("failed to generate a valid directed complementary cbebaa4b input")
