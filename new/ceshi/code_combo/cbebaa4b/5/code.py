# Combo variant of cbebaa4b (index 5)
# combo_spec: D2/3+D14/0
# Name: 带洞精灵 + 矩形目标形状
#
# Design analysis
# ===============
# a) Concrete behavior and flow impact
# - We keep the original family structure: one fixed yellow anchor plus several scattered colored sprites, and the
#   solver may only translate pieces, never rotate them.
# - D14 contributes the global layout: the assembled output must be one solid rectangle. We realize that rectangle as
#   five regions: top / left / anchor / right / bottom, exactly like a coarse tiling around the yellow center.
# - D2 contributes the local interface geometry: every non-anchor parent -> child edge is rendered as a slot/tab pair.
#   Concretely, the parent owns a 2-cell red overlap segment on the shared boundary and also has a 2-cell cavity one
#   step inside its body; the child overlaps the same two red cells and extends a same-color tongue into that cavity.
# - Small trace on one vertical piece-piece interface:
#   original cbebaa4b:
#     parent and child only share the two red cells; after pairing, the surrounding silhouette can stay jagged.
#   fused variant:
#     parent has red-red boundary cells plus two black cells directly behind them; child has the same red pair plus a
#     two-cell tongue entering those black cells; after placement, the tongue fills the hole and the union still has to
#     complete the surrounding rectangle.
#   difference:
#     a candidate move is not judged just by "reds can overlap". Its geometry must also avoid solid-solid collisions,
#     fill visible cavities, and still support a globally solid rectangular final silhouette.
#
# b) Complexity calibration
# - The rectangle is always partitioned into five visually interpretable regions, so humans can infer the macro-goal
#   from a few examples instead of learning many unrelated layout classes.
# - To keep the cavity behavior visible and repeated, we only sample spanning trees whose anchor degree is exactly 2.
#   That guarantees exactly two plain anchor links and exactly two piece-piece slot/tab links in every instance.
# - Region thicknesses come from small integer ranges (4-6), large enough to make enclosed holes obvious while keeping
#   the overall rectangle comfortably within the 22-26 canvas.
#
# c) Hidden assumptions audited from the original main()
# - Original `_try_place()` allowed same-color body overlap. That becomes unsafe once solid slabs and tongues exist, so
#   the fused solver allows overlap only on red-red cells.
# - Original completion accepted any all-paired red assembly. That no longer matches the task: completion must also be
#   a solid rectangle, and every interior cavity extracted from the input pieces must be covered in the final board.
# - Original extraction ignored interior black cells inside piece bounding boxes. Here those holes are semantic, so the
#   solver records them and uses them both for move scoring and for final validation.
# - Original generator assumed a free-form connection tree could be rendered independently per piece. Here the global
#   rectangle and the local slot/tab geometry must be designed together, so generation is output-first from a rectangle
#   ownership grid.
#
# d) Placement legality
# - Each region starts as a disjoint solid block inside the target rectangle.
# - Plain anchor links reuse the D14 trick: parent and child both own the same two red boundary cells.
# - Slot/tab links remain legal because the parent deletes the two cavity cells, while the child alone owns those two
#   cells as a protrusion; therefore the only overlaps are still exactly the two red cells.
# - After the assembled rectangle is shifted into the canvas, only non-anchor pieces are scattered with padding.
#
# e) Visual observability
# - Across examples, a human sees two simultaneous regularities:
#   1. the output is always one filled rectangle centered on the yellow region;
#   2. some scattered non-anchor pieces visibly contain enclosed black slits that disappear only when a neighboring
#      piece slides in with a matching tongue.
# - No invisible ordering uses numeric color ids. Colors are persistent sprite identities only.
#
# f) Visual cleanliness
# - The assembled output is a single clean rectangle.
# - Cavities are always short 2-cell slits inside thick rectangular pieces, so they are legible instead of noisy.
# - Overlap depth stays simple: only red-red overlap is ever permitted.
#
# g) Inference burden
# - There is no color remapping rule. The challenge is purely geometric: infer local slot/tab compatibility and the
#   global rectangular objective from repeated visual evidence.
#
# h) Generator-side anti-degeneration intent
# - Every generated sample contains exactly two slot/tab interfaces, hence at least four interior hole cells in the
#   scattered input. That guarantees the D2 behavior is not accidentally absent.
# - The assembled board is explicitly checked to be a solid rectangle, so the D14 behavior is always active.
#
# i) New solver assumptions guaranteed by the generator
# - Exactly one piece contains yellow (4): enforced by the five-region construction.
# - Every valid overlap is red-red only: enforced when interfaces are carved.
# - All interior holes come from deliberate slot geometry and are fillable by the matching child tongue.
# - Exactly one assembly satisfies the fused constraints: `generate_input()` verifies this by running `main()`.

from __future__ import annotations

import itertools
import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _interior_holes(arr: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Background cells inside the piece bbox that do not touch the bbox border."""
    h, w = arr.shape
    holes: list[tuple[int, int]] = []
    for comp in components_mask(arr == bg, connectivity=4):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        holes.extend((int(r), int(c)) for r, c in comp)
    return holes


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for idx, pts in enumerate(comps):
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        pieces.append(
            {
                "id": int(idx),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "holes": _interior_holes(arr, bg),
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


def _is_solid_rectangle(g: np.ndarray, bg: int) -> bool:
    mask = g != bg
    if not np.any(mask):
        return False
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _holes_filled(board: np.ndarray, placed: dict[int, tuple[dict, int, int]], bg: int) -> bool:
    for piece, top, left in placed.values():
        for rr, cc in piece["holes"]:
            if int(board[top + rr, left + cc]) == bg:
                return False
    return True


def _enumerate_assemblies(
    grid: np.ndarray,
    *,
    require_rectangle: bool | None,
    require_filled_holes: bool | None,
    max_solutions: int,
) -> list[np.ndarray]:
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
    assert total_ports % 2 == 0, "total red connector ports must be even"
    target_unique_red = total_ports // 2
    target_area = int(sum(p["area"] for p in pieces) - target_unique_red)

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        key = (ar0 + rr, ac0 + cc)
        red_counts[key] = red_counts.get(key, 0) + 1

    placed_map: dict[int, tuple[dict, int, int]] = {anchor["id"]: (anchor, ar0, ac0)}
    solutions: list[np.ndarray] = []
    seen: set[bytes] = set()

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        rc: dict[tuple[int, int], int],
        placed_now: dict[int, tuple[dict, int, int]],
    ) -> None:
        if len(solutions) >= max_solutions:
            return

        cur_mask = cur != bg
        cur_area = int(np.sum(cur_mask))
        if cur_area > target_area:
            return
        if require_rectangle is True and cur_area > 0:
            pts = np.argwhere(cur_mask)
            r0, c0, r1, c1 = bbox_coords(pts)
            if (r1 - r0 + 1) * (c1 - c0 + 1) > target_area:
                return

        if not remaining:
            if len(rc) != target_unique_red or any(v != 2 for v in rc.values()):
                return
            is_rect = _is_solid_rectangle(cur, bg)
            holes_ok = _holes_filled(cur, placed_now, bg)
            if require_rectangle is True and not is_rect:
                return
            if require_rectangle is False and is_rect:
                return
            if require_filled_holes is True and not holes_ok:
                return
            if require_filled_holes is False and holes_ok:
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

        assert best_moves is not None

        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)

        scored = []
        for (idx, top, left), ov in dedup.items():
            p = remaining[idx]
            hole_fill_gain = 0
            for piece, pr0, pc0 in placed_now.values():
                for hr, hc in piece["holes"]:
                    gr, gc = pr0 + hr, pc0 + hc
                    if top <= gr < top + p["shape"][0] and left <= gc < left + p["shape"][1]:
                        rr = gr - top
                        cc = gc - left
                        if int(p["arr"][rr, cc]) != bg:
                            hole_fill_gain += 1
            scored.append((idx, top, left, ov, hole_fill_gain))
        scored.sort(key=lambda x: (-x[4], -x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov, _fill in scored:
            p = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_rc = dict(rc)
            for rr, cc in p["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue

            next_placed = dict(placed_now)
            next_placed[p["id"]] = (p, top, left)
            dfs(nxt, next_remaining, next_rc, next_placed)
            if len(solutions) >= max_solutions:
                return

    dfs(placed, rest, red_counts, placed_map)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites into a filled rectangle around the fixed yellow anchor, while using tongues to plug visible cavities.

    Input structure:
    - The background is black (0), with one yellow anchor piece and four non-anchor colored sprites scattered apart.
    - Red (2) cells mark connector overlaps, and some non-anchor sprites contain enclosed black holes inside their own bounding boxes.
    - The anchor is already at its final location and stays fixed; every other sprite must be translated onto it.

    Rule:
    - Build the assembly by overlapping a piece's red connector cells with currently unpaired red cells in the growing structure.
    - Non-red cells may never overlap; the only legal overlaps are red-on-red connector matches.
    - A completed assembly is valid only if every red cell is paired exactly once, every enclosed hole from the input pieces is filled by a neighboring tongue, and the final non-black silhouette is one solid rectangle.

    Key insight:
    - The red markers propose candidate translations, but they do not finish the job on their own.
    - The correct move must also respect the slot/tab geometry of the pieces, and the full set of locally compatible joins must still close into one globally solid rectangle.
    """
    sols = _enumerate_assemblies(
        np.asarray(grid, dtype=int),
        require_rectangle=True,
        require_filled_holes=True,
        max_solutions=2,
    )
    assert len(sols) == 1, "assembly must have exactly one fused-valid solution"
    return sols[0]


# --------------- generator (D2/3 + D14/0) ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
SIDE_THICKNESS_RANGE = (4, 6)
STRIP_THICKNESS_RANGE = (4, 6)
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 1200

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


def _ownership_grid(
    top_h: int, left_w: int, anchor_h: int, anchor_w: int, right_w: int, bottom_h: int
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


def _adjacency_pairs(own: np.ndarray) -> dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]]:
    h, w = own.shape
    out: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    for r in range(h):
        for c in range(w):
            u = int(own[r, c])
            if r + 1 < h:
                v = int(own[r + 1, c])
                if u != v:
                    out.setdefault((min(u, v), max(u, v)), []).append(((r, c), (r + 1, c)))
            if c + 1 < w:
                v = int(own[r, c + 1])
                if u != v:
                    out.setdefault((min(u, v), max(u, v)), []).append(((r, c), (r, c + 1)))
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
        if deg_anchor != 2:
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
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    oriented = []
    for p0, p1 in pairs:
        if int(own[p0]) == parent:
            oriented.append((p0, p1))
        else:
            oriented.append((p1, p0))
    assert len(oriented) >= 4
    starts = list(range(len(oriented) - 1))
    if len(starts) >= 3:
        starts = starts[1:-1]
    start = random.choice(starts)
    seg = oriented[start : start + 2]
    parent_cells = [pc for pc, _ in seg]
    child_cells = [cc for _, cc in seg]
    return parent_cells, child_cells


def _cells_to_arr(cells: dict[tuple[int, int], int]) -> tuple[np.ndarray, tuple[int, int]]:
    pts = np.array(list(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(pts)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), col in cells.items():
        arr[r - r0, c - c0] = int(col)
    return arr, (int(r0), int(c0))


def _build_piece_layout():
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
        region: {cell: region_color[region] for cell in _region_cells(own, region)} for region in REGION_NAMES
    }

    slot_edges = 0
    for child, par in parent.items():
        if child == A:
            continue
        key = (min(child, par), max(child, par))
        parent_cells, child_cells = _choose_segment(interfaces[key], par, own)
        for cell in parent_cells:
            pieces_abs[par][cell] = 2
            pieces_abs[child][cell] = 2

        if par != A:
            step_r = child_cells[0][0] - parent_cells[0][0]
            step_c = child_cells[0][1] - parent_cells[0][1]
            hole_cells = [(pr - step_r, pc - step_c) for pr, pc in parent_cells]
            for hr, hc in hole_cells:
                if int(own[hr, hc]) != par:
                    return None
                pieces_abs[par].pop((hr, hc), None)
                pieces_abs[child][(hr, hc)] = region_color[child]
            slot_edges += 1

    pieces = []
    total_holes = 0
    for region in [A, T, L, R, B]:
        arr, (r0, c0) = _cells_to_arr(pieces_abs[region])
        if len(components_mask(arr != 0, connectivity=4)) != 1:
            return None
        holes = _interior_holes(arr, 0)
        total_holes += len(holes)
        pieces.append(
            {
                "name": REGION_NAMES[region],
                "arr": arr,
                "offset": (r0, c0),
                "reds": [tuple(map(int, rc)) for rc in np.argwhere(arr == 2)],
                "holes": holes,
            }
        )

    if slot_edges != 2:
        return None
    if total_holes < 4:
        return None
    return pieces, own.shape[0], own.shape[1], slot_edges, total_holes


def _place_piece(out: np.ndarray, piece: dict, top: int, left: int) -> tuple[np.ndarray, dict[tuple[int, int], int]] | None:
    placed = _try_place(out, piece["arr"], top, left, 0)
    if placed is None:
        return None
    red_counts = {}
    for rr, cc in piece["reds"]:
        key = (top + rr, left + cc)
        red_counts[key] = red_counts.get(key, 0) + 1
    return placed, red_counts


def _render_assembled(
    pieces: list[dict], grid_h: int, grid_w: int, sr: int, sc: int
) -> tuple[np.ndarray, tuple[int, int]] | None:
    out = np.zeros((grid_h, grid_w), dtype=int)
    red_counts: dict[tuple[int, int], int] = {}
    anchor_pos = None
    placed_meta: dict[str, tuple[int, int]] = {}
    for piece in pieces:
        r0 = sr + piece["offset"][0]
        c0 = sc + piece["offset"][1]
        res = _place_piece(out, piece, r0, c0)
        if res is None:
            return None
        out, rc = res
        placed_meta[piece["name"]] = (r0, c0)
        if piece["name"] == "anchor":
            anchor_pos = (r0, c0)
        for k, v in rc.items():
            red_counts[k] = red_counts.get(k, 0) + v

    assert anchor_pos is not None
    total_ports = sum(len(piece["reds"]) for piece in pieces)
    if len(red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    if not _is_solid_rectangle(out, 0):
        return None
    for piece in pieces:
        top, left = placed_meta[piece["name"]]
        for rr, cc in piece["holes"]:
            if int(out[top + rr, left + cc]) == 0:
                return None
    return out, anchor_pos


def _scatter(pieces: list[dict], anchor_pos: tuple[int, int], grid_h: int, grid_w: int) -> np.ndarray | None:
    inp = np.zeros((grid_h, grid_w), dtype=int)
    anchor = next(piece for piece in pieces if piece["name"] == "anchor")
    paste(inp, anchor["arr"], *anchor_pos, transparent=0)
    rest = [piece for piece in pieces if piece is not anchor]
    random.shuffle(rest)
    for piece in rest:
        try:
            r, c = random_free_location(
                inp,
                piece["arr"],
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, piece["arr"], r, c, transparent=0)
    return inp


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    layout = _build_piece_layout()
    if layout is None:
        return None
    pieces, out_h, out_w, _slot_edges, _hole_cells = layout
    if out_h > grid_h or out_w > grid_w:
        return None

    sr = random.randint(0, grid_h - out_h)
    sc = random.randint(0, grid_w - out_w)
    rendered = _render_assembled(pieces, grid_h, grid_w, sr, sc)
    if rendered is None:
        return None
    out, anchor_pos = rendered

    inp = _scatter(pieces, anchor_pos, grid_h, grid_w)
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
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
