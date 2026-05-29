# Combo variant of cbebaa4b (index 39)
# combo_spec: D1/2+D2/0+D14/0
# Name: 旋转匹配 + 矩形精灵 + 矩形目标
#
# Design analysis
# ---------------
# concretization_and_trace:
# - The assembled output is a single filled rectangle split into five rectangular bodies:
#   top strip, left strip, yellow anchor block, right strip, bottom strip.
# - Every movable sprite keeps a plain rectangular body in one solid color. Red connector cells are the
#   only irregular part: they sit on body edges and may protrude one cell beyond the rectangle into a
#   neighboring region's cells. This preserves the "rectangle sprite + tab" look required by D2/0.
# - Small trace:
#   original logic: an irregular piece with fixed orientation only needs translation so one red edge lands
#   on an exposed red edge of the growing assembly.
#   fused logic: a scattered 4x10 top slab can arrive as a 10x4 vertical slab, or as its mirror, because
#   the off-center red tabs move with the dihedral transform. The solver must test that reoriented tab
#   geometry first, then place the piece so red cells pair, and finally keep only assemblies whose whole
#   silhouette becomes one solid rectangle. The novelty is geometric packing, not path drawing.
#
# complexity_calibration:
# - The layout stays in a five-region rectangle around a fixed yellow anchor. That keeps the search small
#   enough for exhaustive DFS while still giving four movable pieces with different aspect ratios.
# - A non-star spanning tree on the region adjacency graph creates both leaf and degree-2 pieces, so tab
#   patterns vary. Off-center two-cell tabs and unequal strip sizes make the pieces visibly asymmetric and
#   give most of them 4-8 distinct dihedral variants.
# - Colors remain unique per movable piece. This keeps the scene readable for humans without introducing
#   any hidden color remapping rule.
#
# implicit_assumption_audit_of_original_main:
# - The original solver assumed each extracted piece had one fixed bbox and one fixed red set. That breaks
#   under D1/2, so every movable piece now carries a list of unique rotated/flipped variants.
# - The original `_try_place()` tolerated same-color solid overlap. With solid rectangular bodies from D2/0
#   and D14/0, that would create fake packings, so overlap is now legal only on red-red cells.
# - The original completion test only required all red pixels to be paired. Under D14/0 that is too weak:
#   the finished union must also fill its bounding box exactly, so rectangle validity is checked explicitly.
# - The original generator could scatter pieces in their solved pose, silently deactivating D1/2. Here all
#   movable sprites are scattered in non-identity dihedral poses, and generation rejects any case where a
#   no-rotation rectangle solver can still solve the puzzle.
#
# placement_legality:
# - Output-first generation works in absolute rectangle coordinates. Each region owns a disjoint rectangle
#   of solid cells; only chosen connector cells are shared, and they are shared as red-red overlaps only.
# - The child in each tree edge contributes red protrusions into the parent-owned cells, so every sprite
#   stays 4-connected and every overlap cell is intentional.
# - The assembled rectangle is shifted to a random in-bounds location, then only the non-anchor pieces are
#   scattered with Chebyshev padding so extracted input components are clean and non-overlapping.
#
# visual_observability_and_tidiness:
# - Across examples, humans see clean slabs with small red tabs that re-pack into one clean rectangle
#   around the yellow center. There is no hidden ranking by color id or coordinate parity.
# - At least one scattered piece uses a flip, and all movable pieces use non-identity poses, so the
#   rotation/flip rule is visually active instead of being a dormant possibility.
# - Generation also requires at least one non-rectangular red-valid assembly under the relaxed search. That
#   makes the rectangle target a causal rule rather than a property that happens to hold automatically.
#
# inference_and_generator_guarantees:
# - No color transformation exists: color only marks piece identity, yellow marks the anchor, red marks
#   tabs, black is background. The reasoning burden is entirely geometric.
# - `LAST_GEN_INFO` records changed-piece count, flip count, unique dihedral counts, whether no-rotation
#   solving fails, and how many relaxed solutions are non-rectangles. Those metrics are used after testing
#   to verify the fused controls are active instead of degenerating back to the base task.

from __future__ import annotations

import itertools
import os
import random
import sys

import numpy as np

_ROOT = "/home/lixinyu/snap/new"
_SOLVERS = os.path.join(_ROOT, "solvers")
if _SOLVERS not in sys.path:
    sys.path.append(_SOLVERS)

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _piece_record(arr: np.ndarray, bg: int) -> dict:
    reds = np.argwhere(arr == 2)
    return {
        "arr": arr.copy(),
        "reds": [(int(r), int(c)) for r, c in reds],
        "shape": (int(arr.shape[0]), int(arr.shape[1])),
        "area": int(np.sum(arr != bg)),
    }


def _variant_records(arr: np.ndarray, bg: int) -> list[dict]:
    variants = []
    seen = set()
    for rot_k in range(4):
        rot = np.rot90(arr, rot_k)
        for is_flip in (False, True):
            cur = np.fliplr(rot) if is_flip else rot
            key = (tuple(cur.shape), cur.tobytes())
            if key in seen:
                continue
            seen.add(key)
            rec = _piece_record(cur, bg)
            rec["rot_k"] = int(rot_k)
            rec["is_flip"] = bool(is_flip)
            rec["code"] = "id" if (rot_k == 0 and not is_flip) else (
                f"fr{90 * rot_k}" if is_flip else f"r{90 * rot_k}"
            )
            variants.append(rec)
    return variants


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        piece = _piece_record(arr, bg)
        piece["bbox"] = (int(r0), int(c0), int(r1), int(c1))
        pieces.append(piece)
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


def _is_solid_rectangle(g: np.ndarray, bg: int) -> bool:
    mask = g != bg
    if not np.any(mask):
        return False
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _enumerate_assemblies(
    grid: np.ndarray,
    *,
    rectangle_only: bool | None,
    allow_rotate: bool,
    max_solutions: int,
) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must contain at least one non-background piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must contain exactly one yellow anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]
    for piece in rest:
        piece["variants"] = _variant_records(piece["arr"], bg) if allow_rotate else [_piece_record(piece["arr"], bg)]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor must fit at its original location"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector count must be even"
    target_unique_red = total_ports // 2
    target_area = int(sum(p["area"] for p in pieces) - target_unique_red)

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        key = (ar0 + rr, ac0 + cc)
        red_counts[key] = red_counts.get(key, 0) + 1

    solutions: list[np.ndarray] = []
    seen: set[bytes] = set()

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= max_solutions:
            return

        cur_mask = cur != bg
        cur_area = int(np.sum(cur_mask))
        if cur_area > target_area:
            return
        if rectangle_only is True:
            pts = np.argwhere(cur_mask)
            r0, c0, r1, c1 = bbox_coords(pts)
            if (r1 - r0 + 1) * (c1 - c0 + 1) > target_area:
                return

        if not remaining:
            if len(rc) != target_unique_red or any(v != 2 for v in rc.values()):
                return
            is_rect = _is_solid_rectangle(cur, bg)
            if rectangle_only is True and not is_rect:
                return
            if rectangle_only is False and is_rect:
                return
            key = cur.tobytes()
            if key not in seen:
                seen.add(key)
                solutions.append(cur.copy())
            return

        if len(rc) > target_unique_red:
            return

        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            return

        rc1 = {k for k, v in rc.items() if v == 1}
        best_moves = None
        for tr, tc in unpaired:
            moves = []
            for idx, piece in enumerate(remaining):
                for vidx, variant in enumerate(piece["variants"]):
                    ph, pw = variant["shape"]
                    for pr, pc in variant["reds"]:
                        top = tr - pr
                        left = tc - pc
                        if top < 0 or left < 0 or top + ph > h or left + pw > w:
                            continue
                        bad = False
                        overlap = 0
                        for rr, cc in variant["reds"]:
                            gr, gc = top + rr, left + cc
                            prev = rc.get((gr, gc), 0)
                            if prev >= 2:
                                bad = True
                                break
                            if (gr, gc) in rc1:
                                overlap += 1
                        if bad:
                            continue
                        moves.append((idx, vidx, top, left, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible move"

        dedup: dict[tuple[int, int, int, int], int] = {}
        for idx, vidx, top, left, overlap in best_moves:
            key = (idx, vidx, top, left)
            dedup[key] = max(dedup.get(key, 0), overlap)
        moves2 = [(idx, vidx, top, left, ov) for (idx, vidx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[4], -remaining[x[0]]["area"], x[2], x[3], x[0], x[1]))

        for idx, vidx, top, left, _ov in moves2:
            piece = remaining[idx]
            variant = piece["variants"][vidx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_rc = dict(rc)
            for rr, cc in variant["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, variant["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= max_solutions:
                return

    dfs(placed, rest, red_counts)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack scattered rectangular tabbed sprites into one filled rectangle around the yellow anchor.

    Input structure:
    - bg = black (0). Several isolated sprites lie on the grid.
    - One fixed anchor sprite contains the yellow (4) rectangle and stays at its final location.
    - Every non-anchor sprite is a solid colored rectangle with red (2) edge tabs; in the input those
      rectangles may already be rotated by 90/180/270 degrees or mirrored.

    Rule:
    - Keep the anchor where it is.
    - For each remaining sprite, choose a translated position and, if needed, a rotated/flipped pose so
      that red tabs overlap existing unpaired red tabs.
    - Only red-red overlaps are legal, every red must end up paired exactly once, and the completed
      non-black union must be one solid rectangle.

    Key insight: This is no longer a free-form connector puzzle. The red tabs still define local matches,
    but the decisive reasoning is geometric packing: infer the correct orientation of each rectangular slab
    and keep only the assembly whose full silhouette becomes a rectangle.
    """
    sols = _enumerate_assemblies(
        np.asarray(grid, dtype=int),
        rectangle_only=True,
        allow_rotate=True,
        max_solutions=2,
    )
    assert len(sols) == 1, "assembly must have exactly one rectangle-valid rotated solution"
    return sols[0]


# --------------- generator (combo D1/2 + D2/0 + D14/0) ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
SIDE_THICKNESS_RANGE = (4, 6)
STRIP_THICKNESS_RANGE = (3, 5)
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 1800
MIN_UNIQUE_VARIANTS = 4
MAX_RELAXED_SOLUTIONS = 6

A, T, L, R, B = 0, 1, 2, 3, 4
REGION_NAMES = {A: "anchor", T: "top", L: "left", R: "right", B: "bottom"}
GRAPH_EDGES = [
    (A, T),
    (A, L),
    (A, R),
    (A, B),
    (T, L),
    (T, R),
    (L, B),
    (R, B),
]

LAST_GEN_INFO: dict = {}


def _ownership_grid(
    top_h: int,
    left_w: int,
    anchor_h: int,
    anchor_w: int,
    right_w: int,
    bottom_h: int,
) -> np.ndarray:
    h = top_h + anchor_h + bottom_h
    w = left_w + anchor_w + right_w
    own = np.full((h, w), -1, dtype=int)
    own[:top_h, :] = T
    own[top_h : top_h + anchor_h, :left_w] = L
    own[top_h : top_h + anchor_h, left_w : left_w + anchor_w] = A
    own[top_h : top_h + anchor_h, left_w + anchor_w :] = R
    own[top_h + anchor_h :, :] = B
    return own


def _region_cells(own: np.ndarray, region: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(own == region)]


def _adjacency_pairs(
    own: np.ndarray,
) -> dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]]:
    h, w = own.shape
    out: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    for r in range(h):
        for c in range(w):
            u = int(own[r, c])
            if r + 1 < h:
                v = int(own[r + 1, c])
                if u != v:
                    key = (min(u, v), max(u, v))
                    out.setdefault(key, []).append(((r, c), (r + 1, c)))
            if c + 1 < w:
                v = int(own[r, c + 1])
                if u != v:
                    key = (min(u, v), max(u, v))
                    out.setdefault(key, []).append(((r, c), (r, c + 1)))
    return out


def _tree_ok(edges: list[tuple[int, int]]) -> bool:
    if len(edges) != 4:
        return False
    adj = {k: set() for k in REGION_NAMES}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    seen = set()
    stack = [A]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj[cur] - seen)
    return len(seen) == 5


def _all_spanning_trees() -> list[list[tuple[int, int]]]:
    trees = []
    for edges in itertools.combinations(GRAPH_EDGES, 4):
        edge_list = list(edges)
        if not _tree_ok(edge_list):
            continue
        deg_anchor = sum(1 for u, v in edge_list if A in (u, v))
        if deg_anchor == 4:
            continue
        trees.append(edge_list)
    return trees


SPANNING_TREES = _all_spanning_trees()


def _orient_from_anchor(tree_edges: list[tuple[int, int]]) -> dict[int, int | None]:
    adj = {k: [] for k in REGION_NAMES}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    parent: dict[int, int | None] = {A: None}
    stack = [A]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt in parent:
                continue
            parent[nxt] = cur
            stack.append(nxt)
    return parent


def _choose_segment(
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    parent: int,
    own: np.ndarray,
) -> list[tuple[int, int]]:
    oriented = []
    for p0, p1 in pairs:
        if int(own[p0]) == parent:
            oriented.append((p0, p1))
        else:
            oriented.append((p1, p0))
    assert len(oriented) >= 2
    starts = list(range(len(oriented) - 1))
    if len(starts) >= 3:
        starts = starts[1:-1]
    start = random.choice(starts)
    return [pc for pc, _ in oriented[start : start + 2]]


def _cells_to_arr(cells: dict[tuple[int, int], int]) -> tuple[np.ndarray, tuple[int, int]]:
    pts = np.array(list(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(pts)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), col in cells.items():
        arr[r - r0, c - c0] = int(col)
    return arr, (int(r0), int(c0))


def _build_piece_layout() -> tuple[list[dict], int, int]:
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor_h, anchor_w = bh + 2, bw + 2
    top_h = random.randint(*STRIP_THICKNESS_RANGE)
    bottom_h = random.randint(*STRIP_THICKNESS_RANGE)
    left_w = random.randint(*SIDE_THICKNESS_RANGE)
    right_w = random.randint(*SIDE_THICKNESS_RANGE)

    own = _ownership_grid(top_h, left_w, anchor_h, anchor_w, right_w, bottom_h)
    interfaces = _adjacency_pairs(own)
    tree_edges = random.choice(SPANNING_TREES)
    parent = _orient_from_anchor(tree_edges)

    colors = random.sample(PIECE_COLORS, 4)
    region_color = {A: 4, T: colors[0], L: colors[1], R: colors[2], B: colors[3]}
    pieces_abs: dict[int, dict[tuple[int, int], int]] = {
        region: {cell: region_color[region] for cell in _region_cells(own, region)}
        for region in REGION_NAMES
    }

    for child, par in parent.items():
        if child == A:
            continue
        key = (min(child, par), max(child, par))
        for cell in _choose_segment(interfaces[key], par, own):
            pieces_abs[par][cell] = 2
            pieces_abs[child][cell] = 2

    pieces = []
    for region in [A, T, L, R, B]:
        arr, (r0, c0) = _cells_to_arr(pieces_abs[region])
        pieces.append(
            {
                "name": REGION_NAMES[region],
                "arr": arr,
                "offset": (r0, c0),
                "reds": [tuple(map(int, rc)) for rc in np.argwhere(arr == 2)],
            }
        )

    return pieces, int(own.shape[0]), int(own.shape[1])


def _render_assembled(
    pieces: list[dict],
    grid_h: int,
    grid_w: int,
    sr: int,
    sc: int,
) -> np.ndarray | None:
    out = np.zeros((grid_h, grid_w), dtype=int)
    red_counts: dict[tuple[int, int], int] = {}
    for piece in pieces:
        r0 = sr + piece["offset"][0]
        c0 = sc + piece["offset"][1]
        placed = _try_place(out, piece["arr"], r0, c0, 0)
        if placed is None:
            return None
        out = placed
        piece["pos"] = (r0, c0)
        for rr, cc in piece["reds"]:
            key = (r0 + rr, c0 + cc)
            red_counts[key] = red_counts.get(key, 0) + 1

    total_ports = sum(len(piece["reds"]) for piece in pieces)
    if len(red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    if not _is_solid_rectangle(out, 0):
        return None
    return out


def _choose_scatter_variants(pieces: list[dict]) -> bool:
    piece_infos = []
    for piece in pieces:
        if piece["name"] == "anchor":
            continue
        variants = _variant_records(piece["arr"], 0)
        non_identity = [v for v in variants if not np.array_equal(v["arr"], piece["arr"])]
        flip_choices = [v for v in non_identity if v["is_flip"]]
        if len(variants) < MIN_UNIQUE_VARIANTS or not non_identity:
            return False
        piece_infos.append((piece, variants, non_identity, flip_choices))

    if not piece_infos:
        return False

    for piece, variants, non_identity, _flip_choices in piece_infos:
        piece["all_variants"] = variants
        chosen = random.choice(non_identity)
        piece["scatter_arr"] = chosen["arr"]
        piece["scatter_variant"] = chosen["code"]
        piece["scatter_is_flip"] = bool(chosen["is_flip"])

    if not any(piece.get("scatter_is_flip") for piece, *_rest in piece_infos):
        candidates = [(piece, flips) for piece, _variants, _non_identity, flips in piece_infos if flips]
        if not candidates:
            return False
        piece, flips = random.choice(candidates)
        chosen = random.choice(flips)
        piece["scatter_arr"] = chosen["arr"]
        piece["scatter_variant"] = chosen["code"]
        piece["scatter_is_flip"] = True

    return True


def _scatter(
    pieces: list[dict],
    anchor_pos: tuple[int, int],
    grid_h: int,
    grid_w: int,
) -> np.ndarray | None:
    inp = np.zeros((grid_h, grid_w), dtype=int)
    anchor = next(piece for piece in pieces if piece["name"] == "anchor")
    paste(inp, anchor["arr"], *anchor_pos, transparent=0)
    rest = [piece for piece in pieces if piece["name"] != "anchor"]
    random.shuffle(rest)
    for piece in rest:
        try:
            r, c = random_free_location(
                inp,
                piece["scatter_arr"],
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, piece["scatter_arr"], r, c, transparent=0)
    return inp


def _record_last_gen_info(
    pieces: list[dict],
    rect_solutions: list[np.ndarray],
    no_rot_rect_solutions: list[np.ndarray],
    all_solutions: list[np.ndarray],
) -> None:
    global LAST_GEN_INFO
    movable = [piece for piece in pieces if piece["name"] != "anchor"]
    LAST_GEN_INFO = {
        "piece_count": len(movable),
        "changed_piece_count": int(sum(1 for p in movable if p.get("scatter_variant") not in (None, "id"))),
        "flip_piece_count": int(sum(1 for p in movable if p.get("scatter_is_flip"))),
        "variant_codes": [p.get("scatter_variant") for p in movable],
        "unique_variant_counts": [len(p.get("all_variants", [])) for p in movable],
        "rect_solution_count": len(rect_solutions),
        "no_rotation_rect_solution_count": len(no_rot_rect_solutions),
        "all_solution_count": len(all_solutions),
        "nonrect_solution_count": int(sum(not _is_solid_rectangle(sol, 0) for sol in all_solutions)),
    }


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    pieces, out_h, out_w = _build_piece_layout()
    if out_h > grid_h or out_w > grid_w:
        return None

    sr = random.randint(0, grid_h - out_h)
    sc = random.randint(0, grid_w - out_w)
    out = _render_assembled(pieces, grid_h, grid_w, sr, sc)
    if out is None:
        return None

    if not _choose_scatter_variants(pieces):
        return None

    anchor = next(piece for piece in pieces if piece["name"] == "anchor")
    anchor_pos = (sr + anchor["offset"][0], sc + anchor["offset"][1])
    inp = _scatter(pieces, anchor_pos, grid_h, grid_w)
    if inp is None:
        return None

    try:
        rect_solutions = _enumerate_assemblies(
            inp, rectangle_only=True, allow_rotate=True, max_solutions=2
        )
        no_rot_rect_solutions = _enumerate_assemblies(
            inp, rectangle_only=True, allow_rotate=False, max_solutions=2
        )
        all_solutions = _enumerate_assemblies(
            inp,
            rectangle_only=None,
            allow_rotate=True,
            max_solutions=MAX_RELAXED_SOLUTIONS,
        )
    except Exception:
        return None

    if len(rect_solutions) != 1:
        return None
    if not np.array_equal(rect_solutions[0], out):
        return None
    if len(no_rot_rect_solutions) != 0:
        return None
    if len(all_solutions) < 2:
        return None
    if sum(not _is_solid_rectangle(sol, 0) for sol in all_solutions) < 1:
        return None

    _record_last_gen_info(pieces, rect_solutions, no_rot_rect_solutions, all_solutions)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo-39 input")
