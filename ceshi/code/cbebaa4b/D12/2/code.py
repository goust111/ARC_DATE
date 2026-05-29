# Variant of cbebaa4b
# Dimension: D12 — 精灵主体颜色的多样性 (meta)
# Change: 每个精灵有自己的主体颜色，不同精灵颜色可能不同... → 精灵为多色拼贴（内部含多种颜色）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis for the D12 variant
# -----------------------------------
# source_task: none
#
# concretization:
#   original pieces were single-color line/bar sprites with red seam pixels.
#   Here every non-anchor piece keeps the same geometric role, but its non-red
#   cells are recolored from one shared 3-color patchwork field defined in the
#   assembled coordinate system. Each piece is therefore a rigid textured shard:
#   when scattered it already contains 2-3 colors, and when translated back to
#   the correct location those local fragments line up into a larger continuous
#   patchwork around the yellow anchor.
#
# flow trace:
#   old generator path:
#     piece structure -> paint one body color everywhere -> scatter -> solver
#     only needs red connector matching.
#   new generator path:
#     piece structure -> paint body cells by a global 3-color pattern indexed by
#     assembled coordinates -> scatter -> solver still pairs red connectors, but
#     must preserve the internal patchwork exactly when placing pieces.
#   difference:
#     the geometry and motion rule do not change, but the rendered content of
#     each piece now carries visible internal structure that completes a larger
#     motif after assembly.
#
# complexity calibration:
#   - use exactly one shared palette of 3 visible colors per puzzle.
#   - require every non-anchor piece to realize 2 or 3 of those colors.
#   - keep only 3 pattern modes so samples vary, but the output remains legible.
#   This avoids the old single-color look without exploding the number of
#   latent cases humans would need to infer from 2-4 examples.
#
# implicit-assumption audit on main():
#   - _extract_pieces() only assumes "foreground cells belong to 4-connected
#     sprites"; this still holds with multicolor bodies.
#   - anchor detection only assumes "the unique anchor contains yellow=4"; still
#     valid.
#   - DFS state tracks red connector multiplicities only; still valid because
#     translation-only assembly is unchanged.
#   - original _try_place() allowed non-red overlap when colors happened to be
#     equal. That was mostly harmless before because piece body colors were
#     effectively unique. It becomes wrong here because many pieces share the
#     same patchwork palette, so accidental same-color overlaps could create
#     false placements. The variant therefore tightens placement legality to
#     allow overlaps only for red-on-red seam pixels.
#
# placement legality:
#   - assembled nodes are still generated first and must fit within the canvas.
#   - node rendering is validated as one 4-connected component.
#   - assembled stamping rejects any overlap except red/red seam coincidence.
#   - scattered placement still uses >=2 Chebyshev padding, so components stay
#     visually separate and solver extraction remains unambiguous.
#
# visual observability / cleanliness:
#   - red connectors and the yellow anchor retain their old roles.
#   - non-anchor sprites use only 3 body colors total, so the multicolor change
#     is obvious without becoming cluttered.
#   - because all pieces sample one shared patchwork field, the output shows
#     continuous stripes/checker-like bands across different pieces, making the
#     "reassemble the texture fragments" effect visible to a human observer.
#
# generator guarantees for solver assumptions:
#   - every piece is 4-connected and has exact red seam counts as before.
#   - every non-anchor piece contains 2-3 non-special colors.
#   - the assembled image has only red/red overlaps and remains uniquely solved
#     by the same connector-pairing DFS.


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

    # In the variant, many pieces reuse the same body colors, so same-color
    # non-red overlap must be rejected; only red seam pixels may coincide.
    conflict = mask & (sub != bg) & ~((arr == 2) & (sub == 2))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered patchwork sprites around the central yellow anchor by matching red seam pixels.

    Input structure:
    - The background is black (0), with one yellow anchor sprite and several detached sprites.
    - Each non-anchor sprite is a 4-connected multicolor patchwork using 2-3 body colors plus red (2) edge connectors.
    - The anchor is the unique piece containing the yellow (4) block and its own red connector pixels.

    Rule:
    - Keep the anchor fixed and translate the remaining sprites only.
    - Attach a sprite by making one of its red pixels land on an unpaired red pixel in the current assembly.
    - No non-red cells may overlap; after all sprites are placed, every red connector pixel must be paired exactly once.

    Key insight:
    - The multicolor interiors are rigid texture fragments rather than clues for recoloring.
    - The actual assembly logic still comes from pairing red seam pixels, and the larger patchwork appears automatically once those rigid fragments are moved back into alignment.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

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

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        key = (ar0 + rr, ac0 + cc)
        red_counts[key] = red_counts.get(key, 0) + 1

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

        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            return

        rc1 = {k for k, v in rc.items() if v == 1}

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
                    moves.append((idx, top, left, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible connector move"

        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
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
    return solutions[0]


# --------------- generator (o5x) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

# Shared patchwork
PATCHWORK_COLOR_COUNT = 3
PATCHWORK_PATTERN_TRIES = 8

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 300

# Side geometry lookups
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "sides": set(),
        "ports": {},
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
            if arr[rr, cc] != 2:
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


def _make_piece_structure(node):
    """Render a piece skeleton with placeholder body cells=1 and red connectors=2."""
    h, w = node["shape"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    bar_sides = [s for s in SIDES if s not in used]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, 1, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
            rr, cc = r + dr, c + dc
            arr[rr, cc] = 1
            terminals.append((rr, cc))

    arr[hub] = 1
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, 1, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), 1, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == 2)) != 2 * len(used):
        return None
    if int(np.sum(arr == 1)) < 4:
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
    anchor = _new_node("anchor", bh + 2, bw + 2, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)

    for _ in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parents = [j for j, n in enumerate(nodes) if len(n["sides"]) < n["cap"]]
            random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = [s for s in SIDES if s not in parent["sides"]]
                random.shuffle(sides)
                for side in sides:
                    child = _new_node("piece", ph, pw)
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


def _choose_patchwork_spec(r0, c0):
    return {
        "r0": int(r0),
        "c0": int(c0),
        "mode": random.randint(0, 2),
        "phase": random.randint(0, 2),
        "row_step": random.choice([1, 2]),
        "col_step": random.choice([1, 2]),
        "diag_step": random.choice([1, 2]),
        "palette": random.sample(PIECE_COLORS, PATCHWORK_COLOR_COUNT),
    }


def _patchwork_index(gr, gc, spec):
    rr = int(gr - spec["r0"])
    cc = int(gc - spec["c0"])
    row = rr // spec["row_step"]
    col = cc // spec["col_step"]
    diag = (rr + cc) // spec["diag_step"]
    phase = spec["phase"]
    mode = spec["mode"]
    if mode == 0:
        return (row + 2 * col + phase) % PATCHWORK_COLOR_COUNT
    if mode == 1:
        return (diag + col + phase) % PATCHWORK_COLOR_COUNT
    return (row + col + ((rr + cc + phase) & 1)) % PATCHWORK_COLOR_COUNT


def _render_piece_patchwork(template, node, spec):
    arr = template.copy()
    rows, cols = np.nonzero(arr == 1)
    palette = spec["palette"]
    for r, c in zip(rows, cols):
        idx = _patchwork_index(node["top"] + r, node["left"] + c, spec)
        arr[r, c] = palette[idx]
    return arr


def _piece_body_colors(arr):
    vals = sorted(set(int(v) for v in arr.flatten() if v not in (0, 2, 4)))
    return vals


def _place_node_on_canvas(out, node, sr, sc):
    """Stamp a single node's sprite onto *out*, tracking red overlaps."""
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


def _assign_rendered_arrays(nodes):
    for node in nodes:
        if node["kind"] == "anchor":
            node["template"] = _make_anchor(node)
        else:
            node["template"] = _make_piece_structure(node)
        if node["template"] is None:
            return False

    r0, c0, _, _ = _union_bbox(nodes)
    for _ in range(PATCHWORK_PATTERN_TRIES):
        spec = _choose_patchwork_spec(r0, c0)
        ok = True
        for node in nodes:
            if node["kind"] == "anchor":
                node["arr"] = node["template"].copy()
                continue
            node["arr"] = _render_piece_patchwork(node["template"], node, spec)
            ncols = len(_piece_body_colors(node["arr"]))
            if ncols < 2 or ncols > PATCHWORK_COLOR_COUNT:
                ok = False
                break
        if ok:
            return True
    return False


def _assemble(nodes, h, w):
    if not _assign_rendered_arrays(nodes):
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
        for key, val in rc.items():
            all_red_counts[key] = all_red_counts.get(key, 0) + val

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
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
    raise RuntimeError("failed to generate a valid cbebaa4b input")
