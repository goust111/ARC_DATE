# Variant of cbebaa4b
# Dimension: D6 — 输出网格形状 (grid_shape)
# Change: 输出与输入网格尺寸相同... → 紧凑裁剪至组装结构

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location
import random


# Design analysis for the D6 variant
# spec:
#   core_change:
#     original_output: full input-sized canvas containing the assembled jigsaw
#     variant_output: tight crop of that same assembled jigsaw, removing all outer black padding
#   concrete_behavior:
#     input still shows one yellow anchor plus scattered colored pieces with red connector pairs.
#     solver still reconstructs the full assembly on the original canvas because the anchor keeps
#     its absolute position. Only after the unique assembly is found do we crop to the minimal
#     bbox of non-black cells.
#   trace_example:
#     original: a 24x25 input may assemble into rows 4..17 and cols 6..18, but the solver
#       returns the whole 24x25 canvas with empty margins preserved.
#     variant: the same reconstruction is performed, then only rows 4..17 and cols 6..18 are
#       returned, producing a 14x13 output.
#     difference: the pairing and placement logic is unchanged; only the final framing changes.
#   complexity_calibration:
#     the task must still be solved by understanding red-port pairing, not by trivial cropping.
#     therefore the generator keeps the original multi-piece tree assembly and only enforces that
#     the completed structure leaves at least a 1-cell black border inside the input canvas so the
#     crop is visually observable in every sample.
#   implicit_assumptions_audit:
#     _extract_pieces assumes sprites are separated 4-connected components in the input; scattering
#       still guarantees this.
#     anchor detection assumes exactly one piece contains yellow; unchanged.
#     dfs placement assumes pieces translate within the full input canvas and only red-red overlaps
#       are legal; unchanged because the assembly still lives in the original coordinates first.
#     the only new postcondition is that the final assembled structure is non-empty and can be
#       cropped; the generator guarantees this by always including the anchor and at least one piece.
#   placement_legality:
#     generator-side assembled placement now reserves a 1-cell border on every side. This ensures
#     the tight crop is smaller than the input in both dimensions while keeping the scattered input
#     valid and leaving enough room for random_free_location.
#   observability_and_cleanliness:
#     the human-visible change is consistent across examples: same sprite assembly, but the output
#     frame shrinks exactly to the connected object. No color remapping or hidden ordering signal is
#     introduced, so the variant remains readable from a few examples.


def _tight_crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(g != bg)
    if pts.size == 0:
        return g[:0, :0].copy()
    r0, c0, r1, c1 = bbox_coords(pts)
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


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

    conflict = mask & (sub != bg) & ~(((arr == 2) & (sub == 2)) | (arr == sub))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered jigsaw around the yellow anchor, then return only the tight crop.

    Input structure:
    - bg = black (0). Several colored sprites scattered on the grid.
    - One anchor sprite contains the yellow (4) block.
    - Each sprite has red (2) connector pixels along its edges.

    Rule:
    - The anchor stays in place; remaining sprites attach one at a time.
    - Each sprite is translated so that one of its red pixels overlaps an unpaired red pixel on the growing assembly.
    - After all red connectors are perfectly paired, discard the extra black border and keep only the minimal bounding box of the assembled structure.

    Key insight: The hard part is still the same connector-matching search; the new D6 change is that the solved canvas is not returned in full, only its exact non-background footprint.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    # Anchor is the unique piece containing the yellow (color 4) block.
    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one yellow (4) anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"
    target_unique_red = total_ports // 2

    # Track how many original red pixels map to each assembled red coordinate.
    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    # Backtracking assembly by pairing up red connector pixels.
    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions.append(cur)
            return

        if len(rc) > target_unique_red:
            return

        # Unpaired reds are exactly those with count==1; always pair one of them next.
        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            return

        rc1 = {k for k, v in rc.items() if v == 1}

        # Pick the most constrained unpaired red coordinate: fewest ways to cover it.
        best_moves = None
        for t in unpaired:
            moves = []
            tr, tc = t
            for idx, p in enumerate(remaining):
                ph, pw = p["shape"]
                for pr, pc in p["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    # quick red-count feasibility
                    bad = False
                    overlap = 0
                    for rr, cc in p["reds"]:
                        gr, gc = top + rr, left + cc
                        prev = rc.get((gr, gc), 0)
                        if prev >= 2:
                            bad = True
                            break
                        if (gr, gc) in rc1:
                            overlap += 1
                    if bad:
                        continue
                    # must overlap at least the target (guaranteed), but keep for scoring
                    moves.append((idx, top, left, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible connector move"

        # Deduplicate moves by (piece, top, left) keeping max overlap score.
        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            k = (idx, top, left)
            dedup[k] = max(dedup.get(k, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
            p = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_rc = dict(rc)
            for rr, cc in p["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    return _tight_crop_non_bg(solutions[0], bg)

# --------------- generator (o5x) ---------------

# ── Tunable configuration ──────────────────────────────────────────────
# Canvas
GRID_H_RANGE = (22, 26)               # output grid height bounds
GRID_W_RANGE = (22, 26)               # output grid width bounds
MAX_ASSEMBLY_DIM = 20                  # leave room so the final crop is visibly smaller

# Pieces
PIECE_COUNT_RANGE = (4, 6)            # non-anchor sprite count
PIECE_H_RANGE = (4, 7)               # piece bbox height bounds
PIECE_W_RANGE = (4, 8)               # piece bbox width bounds
PIECE_DIM_CANDIDATES = 12            # random (h,w) trials per piece placement
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9] # available non-red, non-yellow colors
PIECE_CAP = 2                         # max connections per piece node
BAR_SKIP_PROB = 0.35                  # probability of dropping one decorative bar

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]  # yellow block (h, w) options
ANCHOR_CAP_CHOICES = [1, 2]              # max connections for anchor node

# Scatter
SCATTER_PADDING = 2                   # Chebyshev gap between scattered sprites
ASSEMBLY_BORDER = 1                   # enforce visible black margin for the crop variant

# Generation
GEN_RETRY_LIMIT = 300                 # max attempts before raising RuntimeError

# ── Side geometry lookups ──────────────────────────────────────────────
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind, "shape": (int(h), int(w)), "color": int(color),
        "top": int(top), "left": int(left), "cap": int(cap),
        "sides": set(), "ports": {},
    }


def _ensure_port(node, side):
    """Lazily assign a random port offset for *side* if not already set."""
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node, side):
    """Return the two local (r, c) cells that form the red connector on *side*."""
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
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


def _draw_path(arr, start, end, color, order):
    """Draw an L-shaped path from *start* to *end*, preserving red (2) pixels."""
    r, c = start
    er, ec = end
    # order 0: row-first then col; order 1: col-first then row
    axes = [(r, er, c, True), (c, ec, er, False)] if order == 0 else \
           [(c, ec, r, False), (r, er, ec, True)]
    for a_start, a_end, fixed, is_row in axes:
        step = 1 if a_end >= a_start else -1
        for v in range(a_start, a_end + step, step):
            rr, cc = (v, fixed) if is_row else (fixed, v)
            if arr[rr, cc] != 2:
                arr[rr, cc] = color


def _paint_bar(arr, side, color, used):
    """Paint a decorative bar along *side*, skipping corners claimed by *used* sides."""
    h, w = arr.shape
    is_horizontal = side in ("top", "bottom")
    fixed = 0 if side in ("top", "left") else (h - 1 if side == "bottom" else w - 1)
    span = w if is_horizontal else h
    # corner neighbors: (index, conflicting side) pairs
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
    """Render a piece sprite array with red connectors and decorative bars."""
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


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    return arr


def _candidate_fits(nodes, parent_i, child):
    """Check child overlaps only its parent and total assembly stays within bounds."""
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
    """Build a tree of connected nodes (1 anchor + N pieces) in local coordinates."""
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)

    for i in range(n_pieces):
        placed = False
        dims = [(random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
                for _ in range(PIECE_DIM_CANDIDATES)]
        random.shuffle(dims)
        for ph, pw in dims:
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
                    if not _candidate_fits(nodes, parent_i, child):
                        continue
                    parent["sides"].add(side)
                    nodes.append(child)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None
    return nodes


def _place_node_on_canvas(out, node, sr, sc):
    """Stamp a single node's sprite onto *out*, tracking red overlaps.

    Returns red_counts dict on success, or None on conflict.
    """
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    red_counts = {}
    mask = arr != 0
    rows, cols = np.nonzero(mask)
    g_rows, g_cols = rows + top, cols + left
    vals = arr[rows, cols]
    cur_vals = out[g_rows, g_cols]
    # Conflict: non-bg destination that isn't a red-on-red overlap
    conflicts = (cur_vals != 0) & ~((cur_vals == 2) & (vals == 2))
    if np.any(conflicts):
        return None
    # Track red overlap counts
    red_mask = vals == 2
    for gr, gc in zip(g_rows[red_mask], g_cols[red_mask]):
        key = (int(gr), int(gc))
        red_counts[key] = red_counts.get(key, 0) + 1
    # Paint non-overlapping pixels
    paint_mask = cur_vals == 0
    out[g_rows[paint_mask], g_cols[paint_mask]] = vals[paint_mask]
    return red_counts


def _assemble(nodes, h, w):
    """Render all nodes, shift into grid, and validate the assembled output."""
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h - 2 * ASSEMBLY_BORDER or (c1 - c0 + 1) > w - 2 * ASSEMBLY_BORDER:
        return None
    sr_lo = ASSEMBLY_BORDER - r0
    sr_hi = h - ASSEMBLY_BORDER - 1 - r1
    sc_lo = ASSEMBLY_BORDER - c0
    sc_hi = w - ASSEMBLY_BORDER - 1 - c1
    if sr_lo > sr_hi or sc_lo > sc_hi:
        return None
    sr = random.randint(sr_lo, sr_hi)
    sc = random.randint(sc_lo, sc_hi)

    out = np.zeros((h, w), dtype=int)
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for k, v in rc.items():
            all_red_counts[k] = all_red_counts.get(k, 0) + v

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    cropped = _tight_crop_non_bg(out, 0)
    if cropped.shape[0] >= h or cropped.shape[1] >= w:
        return None
    return out


def _scatter(nodes, h, w):
    """Place anchor at its assembled position, then scatter remaining pieces."""
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    rest = nodes[1:]
    random.shuffle(rest)
    for node in rest:
        try:
            r, c = random_free_location(
                inp, node["arr"], transparent=0,
                padding=SCATTER_PADDING, padding_connectivity=8)
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
    out_cropped = _tight_crop_non_bg(out, 0)
    inp = _scatter(nodes, h, w)
    if inp is None:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out_cropped):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b input")
