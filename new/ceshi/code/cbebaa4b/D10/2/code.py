# Variant of cbebaa4b
# Dimension: D10 — 连接器像素的数量与分布 (meta)
# Change: 每个精灵有少量红色连接器像素分布在边缘，锚点方块边缘也有红色像素... → 连接器分为输入端和输出端

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
# ----------------
# variant:
#   source_task: none
#   concreteization:
#     - Keep the original output-first assembly setup: one yellow anchor stays fixed,
#       4-6 other sprites are scattered in the input, and the output is the single
#       assembled structure obtained by translation only.
#     - Replace undirected red connectors with directed connectors:
#         output port = red (2), input port = green (3).
#       A legal join is an overlap between a red two-cell edge segment from one
#       sprite and a green two-cell edge segment from another sprite.
#     - The visible output keeps the original "single joined seam" look by drawing
#       every completed seam in red. This is the smallest compatible display change:
#       direction is visible in the input, but the final connected shape still looks
#       like the same kind of fused jigsaw structure.
#   behavior_trace:
#     - Original local step:
#         assembly has one unpaired red coordinate; any remaining red coordinate may
#         try to land there, because all connectors are interchangeable.
#     - Variant local step:
#         if the frontier cell currently contains only red, the next piece must bring
#         green there; if it currently contains only green, the next piece must bring
#         red there. Same-color overlaps are illegal.
#     - Difference:
#         matching is no longer symmetric. The search state must track connector role,
#         not just occupancy count.
#   complexity_calibration:
#     - Only two connector roles/colors are used, repeated across all samples.
#     - Port geometry stays as the original two adjacent boundary cells, so the visual
#       difference is clear without introducing many new freedoms.
#     - Piece motion is still translation-only; no rotation/flip is added.
#   implicit_assumption_audit:
#     - Original _extract_pieces assumed "all ports are red"; variant must record
#       red and green separately.
#     - Original _try_place allowed red/red overlap and treated all other overlaps as
#       conflicts; variant must allow only red/green overlap and must canonicalize
#       the merged seam color in the assembled output.
#     - Original DFS state stored one count per red coordinate; variant must store
#       per-coordinate counts for output-vs-input contributors so that completed
#       seams are exactly (1 output, 1 input).
#     - Original move generation aligned any red to any unpaired red; variant must
#       align a needed input only with a piece input port, and a needed output only
#       with a piece output port.
#   placement_legality:
#     - The generator still builds the final assembly first, then scatters pieces.
#     - Child placement is legal only if its bbox overlaps its parent bbox, does not
#       touch any other node bbox, and the total union bbox stays within bounds.
#     - During assembly, only opposite-role connector cells may overlap; all other
#       nonzero overlaps are rejected.
#   visual_observability:
#     - Humans can see the rule change directly in the input because connector colors
#       now split into red "providers" and green "receivers".
#     - Samples include both connector colors and at least one mixed-role non-anchor
#       piece, so the direction cue is not reducible to a trivial recoloring.
#   cleanliness:
#     - Output remains a single clean connected component with red seams only.
#     - Decorative bars stay in the piece body color, so connector markers remain
#       visually distinct from the piece interiors.


OUTPUT_COLOR = 2
INPUT_COLOR = 3
ANCHOR_COLOR = 4


def _is_connector_color(v: np.ndarray | int) -> np.ndarray | bool:
    return (v == OUTPUT_COLOR) | (v == INPUT_COLOR)


def _compatible_connector_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a == OUTPUT_COLOR) & (b == INPUT_COLOR)) | (
        (a == INPUT_COLOR) & (b == OUTPUT_COLOR)
    )


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        outputs = np.argwhere(arr == OUTPUT_COLOR)
        inputs = np.argwhere(arr == INPUT_COLOR)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "outputs": [(int(r), int(c)) for r, c in outputs],
                "inputs": [(int(r), int(c)) for r, c in inputs],
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
    out_sub[merge_mask] = OUTPUT_COLOR
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _add_port_counts(
    state: dict[tuple[int, int], tuple[int, int]],
    coords: list[tuple[int, int]],
    top: int,
    left: int,
    kind: str,
) -> dict[tuple[int, int], tuple[int, int]]:
    nxt = dict(state)
    out_inc, in_inc = (1, 0) if kind == "out" else (0, 1)
    for rr, cc in coords:
        key = (top + rr, left + cc)
        prev_out, prev_in = nxt.get(key, (0, 0))
        nxt[key] = (prev_out + out_inc, prev_in + in_inc)
    return nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered sprites around the fixed yellow anchor by matching directed edge ports.

    Input structure:
    - bg = black (0). Several colored sprites are separated on the grid.
    - One anchor sprite contains the yellow (4) block and remains at its original location.
    - Connector cells come in two colors: red (2) is an output port, green (3) is an input port.

    Rule:
    - Translate the non-anchor sprites one at a time onto the growing assembly.
    - A legal seam is formed only when a sprite's red connector cells overlap another sprite's green connector cells, or vice versa.
    - Every connector cell must end up paired exactly once, producing one connected assembled structure.

    Key insight: The frontier is not just "an unpaired connector" anymore; each frontier coordinate demands a specific role. Red-only frontier cells must be covered by green, and green-only frontier cells must be covered by red.
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

    total_ports = int(
        sum(len(p["outputs"]) + len(p["inputs"]) for p in pieces)
    )
    assert total_ports % 2 == 0, "total connector cells must be even"
    target_unique_ports = total_ports // 2

    port_state: dict[tuple[int, int], tuple[int, int]] = {}
    port_state = _add_port_counts(port_state, anchor["outputs"], ar0, ac0, "out")
    port_state = _add_port_counts(port_state, anchor["inputs"], ar0, ac0, "in")

    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        state: dict[tuple[int, int], tuple[int, int]],
    ) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(state) == target_unique_ports and all(v == (1, 1) for v in state.values()):
                solutions.append(cur)
            return

        if len(state) > target_unique_ports:
            return
        if any(out_n > 1 or in_n > 1 for out_n, in_n in state.values()):
            return

        needs: list[tuple[tuple[int, int], str]] = []
        for coord, (out_n, in_n) in state.items():
            if out_n == 1 and in_n == 0:
                needs.append((coord, "in"))
            elif out_n == 0 and in_n == 1:
                needs.append((coord, "out"))
            elif out_n == 1 and in_n == 1:
                continue
            else:
                return
        if not needs:
            return

        best_moves = None
        for target, needed_kind in needs:
            tr, tc = target
            moves = []
            for idx, p in enumerate(remaining):
                local_ports = p["inputs"] if needed_kind == "in" else p["outputs"]
                if not local_ports:
                    continue
                ph, pw = p["shape"]
                for pr, pc in local_ports:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue

                    bad = False
                    overlap = 0
                    for rr, cc in p["outputs"]:
                        gr, gc = top + rr, left + cc
                        prev_out, prev_in = state.get((gr, gc), (0, 0))
                        if prev_out >= 1:
                            bad = True
                            break
                        if prev_in == 1:
                            overlap += 1
                    if bad:
                        continue
                    for rr, cc in p["inputs"]:
                        gr, gc = top + rr, left + cc
                        prev_out, prev_in = state.get((gr, gc), (0, 0))
                        if prev_in >= 1:
                            bad = True
                            break
                        if prev_out == 1:
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

        assert best_moves is not None, "there must be at least one feasible directed move"

        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
            p = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_state = _add_port_counts(state, p["outputs"], top, left, "out")
            next_state = _add_port_counts(next_state, p["inputs"], top, left, "in")

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_state)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, port_state)
    assert len(solutions) == 1, "directed jigsaw assembly must have exactly one valid solution"
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
PIECE_COLORS = [1, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [2]

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 400

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


def _connector_color(role):
    return OUTPUT_COLOR if role == "out" else INPUT_COLOR


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
        port_color = _connector_color(node["roles"][side])
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


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = ANCHOR_COLOR
    for side in node["sides"]:
        port_color = _connector_color(node["roles"][side])
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


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node(
        "anchor", bh + 2, bw + 2, ANCHOR_COLOR, cap=random.choice(ANCHOR_CAP_CHOICES)
    )
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)

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
                parents = [j for j in range(1, min(len(nodes), 3)) if len(nodes[j]["sides"]) < nodes[j]["cap"]]
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

                    existing_roles = list(parent["roles"].values())
                    if not existing_roles:
                        parent_role = random.choice(["in", "out"])
                    elif len(set(existing_roles)) == 1:
                        parent_role = "out" if existing_roles[0] == "in" else "in"
                    else:
                        parent_role = random.choice(["in", "out"])
                    child_role = "in" if parent_role == "out" else "out"

                    child["sides"].add(incoming)
                    child["roles"][incoming] = child_role
                    if not _candidate_fits(nodes, parent_i, child):
                        continue
                    parent["sides"].add(side)
                    parent["roles"][side] = parent_role
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
    mixed_piece_count = sum(
        len(set(node["roles"].values())) == 2 for node in nodes[1:] if node["roles"]
    )
    anchor_roles = set(anchor["roles"].values())
    if all_roles != {"in", "out"}:
        return None
    if mixed_piece_count == 0:
        return None
    if len(anchor_roles) < 2:
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

    red_state: dict[tuple[int, int], tuple[int, int]] = {}
    for rr, cc in zip(*np.nonzero(arr == OUTPUT_COLOR)):
        key = (top + int(rr), left + int(cc))
        prev_out, prev_in = red_state.get(key, (0, 0))
        red_state[key] = (prev_out + 1, prev_in)
    for rr, cc in zip(*np.nonzero(arr == INPUT_COLOR)):
        key = (top + int(rr), left + int(cc))
        prev_out, prev_in = red_state.get(key, (0, 0))
        red_state[key] = (prev_out, prev_in + 1)

    paint_mask = mask & (sub == 0)
    merge_mask = mask & compatible
    sub[paint_mask] = arr[paint_mask]
    sub[merge_mask] = OUTPUT_COLOR
    out[top : top + ph, left : left + pw] = sub
    return red_state


def _assemble(nodes, h, w):
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
    all_port_counts: dict[tuple[int, int], tuple[int, int]] = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for key, (out_n, in_n) in rc.items():
            prev_out, prev_in = all_port_counts.get(key, (0, 0))
            all_port_counts[key] = (prev_out + out_n, prev_in + in_n)

    total_ports = sum(int(np.sum(_is_connector_color(n["arr"]))) for n in nodes)
    if len(all_port_counts) * 2 != total_ports:
        return None
    if any(v != (1, 1) for v in all_port_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
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
    raise RuntimeError("failed to generate a valid directed cbebaa4b input")
