# Combo variant of cbebaa4b (index 42)
# combo_spec: D3/3+D13/2
# Name: 十字形锚点 + 连接方向由锚点决定

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior:
#    - The anchor is no longer a solid yellow rectangle. It is a thick cross or a
#      thick L mask, with 2-cell red ports only at branch tips.
#    - A tip port now encodes two facts at once: it marks that this branch exists
#      and it fixes the side from which the first attached piece must arrive.
#      Example: if the anchor has a red tip on its top branch, the directly docked
#      piece must use a bottom-side red port and therefore sits above the anchor.
#    - Relative to the original task, the old "match any red to any unpaired red"
#      rule still governs later piece-to-piece growth, but the first hop out of
#      the anchor is stricter: the whole 2-cell tip must be covered by the child
#      piece's opposite-facing 2-cell port.
# b) Complexity calibration:
#    - Only two anchor families are used (`cross`, rotated `L`) and each anchor
#      exposes a small subset of branch tips. This keeps the new rule visible.
#    - Cross anchors use exactly 3 of 4 tips; L anchors use both exposed tips.
#      That makes the encoded topology readable in a few training examples.
#    - We intentionally keep piece caps small (<= 2) so the anchor-side direction
#      signal is not drowned out by large hubs or many competing ports.
# c) Hidden assumptions audited from the original solver:
#    - "The anchor is any piece containing yellow" is no longer enough; we now
#      require the yellow/red occupancy mask to equal one supported cross/L mask.
#    - "Any red pixel can seed any first attachment" breaks under D13/2. Anchor
#      tip reds are now grouped by side, and a legal direct attachment must cover
#      both red cells of that tip with a piece port on the opposite side.
#    - The original solver tracked only red coordinates. The combo solver also
#      tracks each piece red's boundary side so it can reject wrong-direction
#      anchor attachments while leaving later piece-to-piece attachments unchanged.
# d) Placement legality:
#    - The generator remains output-first: build a full assembled tree, validate
#      non-red non-overlap, then scatter all non-anchor pieces with padding.
#    - Anchor ports are represented explicitly, because L anchors have branch tips
#      that are not describable as arbitrary offsets on a convex rectangle.
#    - Every accepted sample is re-solved by `main()` before returning.
# e) Visual observability:
#    - The yellow anchor geometry is directly visible, and the red tip locations
#      visibly explain why branches leave upward/leftward/rightward/downward.
#    - The generator rejects samples with zero "forbidden old-style anchor moves".
#      This ensures the direction rule is doing real work instead of being a
#      decorative restatement of the geometry.
# f) Visual cleanliness:
#    - Only red may overlap in the final assembly, exactly as in the source task.
#    - The anchor remains monochrome yellow except at active red tip cells, so the
#      reader can isolate the branch ends without extra markings or legend colors.
# g) Color inference:
#    - Non-anchor colors stay in the original palette family. The fusion changes
#      spatial reasoning only; it does not introduce hidden color remapping.
# h) Fusion metric used after implementation:
#    - We measure `forbidden_anchor_moves`: placements that the original "align one
#      red anywhere" logic would allow against the bare anchor, but that the combo
#      rule rejects because the piece approaches from the wrong side or covers only
#      half of an anchor tip. Generated samples require this metric to be >= 1.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes exactly one anchor piece matches a supported cross/L mask,
#      and that each red tip side is either fully active (2 red cells) or inactive.
#      The generator enforces this.
#    - `main()` assumes the direct anchor child for each active tip uses the
#      opposite-facing side port as a 2-cell unit. `_grow_tree()` and `_make_anchor`
#      enforce that structure, while later attachments still follow the ordinary
#      pairwise red-port assembly rule.


def _anchor_mask(kind: str, turns: int = 0) -> np.ndarray:
    if kind == "cross":
        mask = np.zeros((6, 6), dtype=bool)
        mask[2:4, :] = True
        mask[:, 2:4] = True
        return mask
    if kind == "el":
        mask = np.zeros((6, 6), dtype=bool)
        mask[:, :2] = True
        mask[-2:, :] = True
        return np.rot90(mask, k=turns)
    raise ValueError(f"unknown anchor kind: {kind}")


def _extract_anchor_ports(mask: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    h, w = mask.shape
    ports: dict[str, list[tuple[int, int]]] = {}

    top_cols = np.flatnonzero(mask[0, :])
    if len(top_cols) == 2 and top_cols[1] == top_cols[0] + 1:
        ports["top"] = [(0, int(top_cols[0])), (0, int(top_cols[1]))]

    bottom_cols = np.flatnonzero(mask[h - 1, :])
    if len(bottom_cols) == 2 and bottom_cols[1] == bottom_cols[0] + 1:
        ports["bottom"] = [(h - 1, int(bottom_cols[0])), (h - 1, int(bottom_cols[1]))]

    left_rows = np.flatnonzero(mask[:, 0])
    if len(left_rows) == 2 and left_rows[1] == left_rows[0] + 1:
        ports["left"] = [(int(left_rows[0]), 0), (int(left_rows[1]), 0)]

    right_rows = np.flatnonzero(mask[:, w - 1])
    if len(right_rows) == 2 and right_rows[1] == right_rows[0] + 1:
        ports["right"] = [(int(right_rows[0]), w - 1), (int(right_rows[1]), w - 1)]
    return ports


def _anchor_catalog() -> list[dict]:
    out = []
    cross_mask = _anchor_mask("cross")
    out.append(
        {
            "kind": "cross",
            "turns": 0,
            "mask": cross_mask,
            "ports": _extract_anchor_ports(cross_mask),
        }
    )
    for turns in range(4):
        mask = _anchor_mask("el", turns)
        out.append(
            {
                "kind": "el",
                "turns": turns,
                "mask": mask,
                "ports": _extract_anchor_ports(mask),
            }
        )
    return out


ANCHOR_SHAPES = _anchor_catalog()


def _analyze_anchor_piece(arr: np.ndarray) -> dict | None:
    if not np.any(arr == 4):
        return None
    if not np.all(np.isin(arr, [0, 2, 4])):
        return None

    occ = arr != 0
    reds = {tuple(map(int, rc)) for rc in np.argwhere(arr == 2)}
    for spec in ANCHOR_SHAPES:
        mask = spec["mask"]
        if occ.shape != mask.shape or not np.array_equal(occ, mask):
            continue
        port_cells = spec["ports"]
        all_port_cells = {cell for cells in port_cells.values() for cell in cells}
        if not reds.issubset(all_port_cells):
            continue
        used_sides = set()
        ok = True
        for side, cells in port_cells.items():
            flags = [arr[r, c] == 2 for r, c in cells]
            if any(flags) and not all(flags):
                ok = False
                break
            if all(flags):
                used_sides.add(side)
        if not ok or not used_sides:
            continue
        return {
            "kind": spec["kind"],
            "turns": spec["turns"],
            "mask": mask,
            "port_cells": port_cells,
            "used_sides": used_sides,
        }
    return None


def _red_side(arr: np.ndarray, r: int, c: int) -> str:
    h, w = arr.shape
    sides = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    if len(sides) == 1:
        return sides[0]
    if len(sides) == 2:
        candidates = []
        if "top" in sides and ((c > 0 and arr[r, c - 1] == 2) or (c + 1 < w and arr[r, c + 1] == 2)):
            candidates.append("top")
        if "bottom" in sides and ((c > 0 and arr[r, c - 1] == 2) or (c + 1 < w and arr[r, c + 1] == 2)):
            candidates.append("bottom")
        if "left" in sides and ((r > 0 and arr[r - 1, c] == 2) or (r + 1 < h and arr[r + 1, c] == 2)):
            candidates.append("left")
        if "right" in sides and ((r > 0 and arr[r - 1, c] == 2) or (r + 1 < h and arr[r + 1, c] == 2)):
            candidates.append("right")
        if len(candidates) == 1:
            return candidates[0]
    raise AssertionError("piece red connectors must lie on an identifiable bbox side")


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for rr, cc in pts:
            arr[int(rr) - int(r0), int(cc) - int(c0)] = g[int(rr), int(cc)]
        reds = np.argwhere(arr == 2)
        red_coords = [(int(r), int(c)) for r, c in reds]
        anchor_info = _analyze_anchor_piece(arr)
        if anchor_info is not None:
            side_map = {
                cell: side
                for side, cells in anchor_info["port_cells"].items()
                for cell in cells
            }
            red_sides = [side_map[(int(r), int(c))] for r, c in reds]
        else:
            red_sides = [_red_side(arr, int(r), int(c)) for r, c in reds]
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": red_coords,
                "red_sides": red_sides,
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
                "anchor_info": anchor_info,
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


def _anchor_global_ports(anchor: dict) -> dict[str, list[tuple[int, int]]]:
    ar0, ac0, _, _ = anchor["bbox"]
    return {
        side: [(ar0 + r, ac0 + c) for r, c in cells]
        for side, cells in anchor["anchor_info"]["port_cells"].items()
        if side in anchor["anchor_info"]["used_sides"]
    }


def _placement_respects_anchor_direction(
    piece: dict,
    top: int,
    left: int,
    anchor_side: str,
    anchor_ports: dict[str, list[tuple[int, int]]],
) -> bool:
    want_side = OPPOSITE[anchor_side]
    red_side_at = {
        (top + rr, left + cc): piece["red_sides"][i]
        for i, (rr, cc) in enumerate(piece["reds"])
    }
    return all(red_side_at.get(coord) == want_side for coord in anchor_ports[anchor_side])


def _forbidden_anchor_move_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pieces = _extract_pieces(g, 0)
    anchors = [p for p in pieces if p["anchor_info"] is not None]
    if len(anchors) != 1:
        return 0
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.zeros((h, w), dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, 0)
    if placed is None:
        return 0

    anchor_ports = _anchor_global_ports(anchor)
    blocked = set()
    for side, coords in anchor_ports.items():
        for tr, tc in coords:
            for idx, p in enumerate(rest):
                ph, pw = p["shape"]
                for red_i, (pr, pc) in enumerate(p["reds"]):
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    if _try_place(placed, p["arr"], top, left, 0) is None:
                        continue
                    if _placement_respects_anchor_direction(p, top, left, side, anchor_ports):
                        continue
                    blocked.add((idx, side, top, left, red_i))
    return len(blocked)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites by docking them onto a yellow cross/L anchor whose red tips dictate approach direction.

    Input structure:
    - bg = black (0). Several colored sprites are scattered on the grid.
    - Exactly one sprite is a yellow (4) anchor shaped like a thick cross or thick L.
    - Red (2) connector pairs appear on sprite boundaries; on the anchor they sit at branch tips.

    Rule:
    - The yellow anchor stays at its input position.
    - Remaining sprites translate without rotation. A direct anchor attachment is legal only if a piece covers an entire active anchor tip with the opposite-facing 2-cell port.
    - After a piece has attached to the anchor, later piece-to-piece growth continues by pairing every remaining unpaired red exactly once.

    Key insight:
    - Red pixels are still the assembly pegs, but anchor reds are no longer generic: the side of each anchor tip tells you which side of the first child piece is allowed to dock there.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if p["anchor_info"] is not None]
    assert len(anchors) == 1, "input must have exactly one cross/L yellow anchor piece"
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
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    anchor_ports = _anchor_global_ports(anchor)
    anchor_side_by_coord = {
        coord: side for side, coords in anchor_ports.items() for coord in coords
    }

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
            anchor_side = anchor_side_by_coord.get(t)
            for idx, p in enumerate(remaining):
                ph, pw = p["shape"]
                for red_i, (pr, pc) in enumerate(p["reds"]):
                    if anchor_side is not None and p["red_sides"][red_i] != OPPOSITE[anchor_side]:
                        continue
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    if anchor_side is not None and not _placement_respects_anchor_direction(
                        p, top, left, anchor_side, anchor_ports
                    ):
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


# --------------- generator (combo variant) ---------------

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
ANCHOR_VARIANTS = ("cross", "el")
ANCHOR_DIRECT_DEGREE = {"cross": 3, "el": 2}
MIN_FORBIDDEN_ANCHOR_MOVES = 1

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
    }


def _new_anchor_node() -> dict:
    kind = random.choice(ANCHOR_VARIANTS)
    turns = random.randint(0, 3) if kind == "el" else 0
    mask = _anchor_mask(kind, turns)
    port_cells = _extract_anchor_ports(mask)
    direct_degree = ANCHOR_DIRECT_DEGREE[kind]
    node = _new_node("anchor", mask.shape[0], mask.shape[1], 4, cap=direct_degree)
    node["anchor_kind"] = kind
    node["anchor_turns"] = turns
    node["mask"] = mask
    node["port_cells"] = port_cells
    if kind == "cross":
        node["direct_sides"] = set(random.sample(list(port_cells), direct_degree))
    else:
        node["direct_sides"] = set(port_cells)
    return node


def _ensure_port(node, side):
    if side in node.get("port_cells", {}):
        return
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _port_cells(node, side):
    explicit = node.get("port_cells", {})
    if side in explicit:
        return explicit[side]
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
    arr = np.zeros(node["mask"].shape, dtype=int)
    arr[node["mask"]] = 4
    for side in node["sides"]:
        for r, c in node["port_cells"][side]:
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


def _parent_available_sides(node):
    if node["kind"] == "anchor":
        return [s for s in node["direct_sides"] if s not in node["sides"]]
    return [s for s in SIDES if s not in node["sides"]]


def _grow_tree():
    anchor = _new_anchor_node()
    nodes = [anchor]
    anchor_child_indices: list[int] = []
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)

    for i in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        need_anchor = len(anchor["sides"]) < len(anchor["direct_sides"])
        need_extend_anchor_child = (
            not need_anchor
            and n_pieces > len(anchor["direct_sides"])
            and not any(len(nodes[idx]["sides"]) >= 2 for idx in anchor_child_indices)
        )
        for ph, pw in dims:
            if need_anchor:
                parents = [0]
            else:
                parents = [j for j, n in enumerate(nodes) if len(n["sides"]) < n["cap"]]
                anchor_parents = [j for j in anchor_child_indices if j in parents]
                other_parents = [j for j in parents if j not in anchor_parents]
                random.shuffle(anchor_parents)
                random.shuffle(other_parents)
                parents = anchor_parents + other_parents
                if need_extend_anchor_child:
                    parents = anchor_parents
                    if not parents:
                        break
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = _parent_available_sides(parent)
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
                    if parent_i == 0:
                        child["anchor_side"] = side
                    if not _candidate_fits(nodes, parent_i, child):
                        continue
                    parent["sides"].add(side)
                    nodes.append(child)
                    if parent_i == 0:
                        anchor_child_indices.append(len(nodes) - 1)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    if n_pieces > len(anchor["direct_sides"]) and not any(
        len(nodes[idx]["sides"]) >= 2 for idx in anchor_child_indices
    ):
        return None
    return nodes


def _place_node_on_canvas(out, node, sr, sc):
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
    if _forbidden_anchor_move_count(inp) < MIN_FORBIDDEN_ANCHOR_MOVES:
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
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
