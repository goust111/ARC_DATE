# Variant of cbebaa4b
# Dimension: D14 — 组装后结构的目标形状 (meta)
# Change: 组装后形成不规则的连通结构，无特定目标形状约束... → 组装后必须形成矩形
#
# Design analysis
# ===============
# a) Concrete behavior and flow impact
# - Input still contains one yellow anchor sprite and several scattered colored sprites with red connectors.
# - Output is no longer “any fully paired connected assembly”. The paired assembly must also occupy every cell of
#   its own bounding box, so the final non-black shape is exactly one solid rectangle.
# - To make this visible, the generator no longer uses sparse bracket/frame sprites. Each piece now has a solid body
#   plus red overlap tabs. In the assembled output, those bodies tile a rectangle around the anchor, while the red
#   tabs mark the overlap cells where pieces lock together.
# - Small example of the changed step:
#   original: a piece could be attached as soon as all red overlaps were pairwise valid, even if the growing union
#   ended as a jagged silhouette.
#   variant: the same local attachment is only acceptable if the complete union later fills its bounding box exactly.
#   During generation we deliberately keep instances where the relaxed “pair reds only” rule admits at least one
#   non-rectangular assembly, so D14 is an active constraint instead of a passive property.
#
# b) Complexity calibration
# - The rectangle is partitioned into five regions (top / left / anchor / right / bottom), but the connection tree is
#   sampled from many non-star spanning trees of that adjacency graph. This keeps the visual language simple while
#   preserving enough combinatorial freedom that local red matching alone is not always decisive.
# - Region sizes come from small integer ranges and the anchor stays one of two familiar sizes, so the task remains
#   inferable from a few examples instead of exploding into many unrelated layout classes.
#
# c) Hidden assumptions in the original main()
# - Original `_try_place()` allowed overlap when `arr == sub`, which was harmless when generated pieces never reused a
#   filled color at the same coordinate. With solid pieces this becomes dangerous: same-color body overlap would fake a
#   valid placement. The variant tightens this and now permits overlap only on red-red cells.
# - Original completion test only checked that every red ended up paired. That is insufficient for D14 because a fully
#   paired irregular union must now be rejected. The new solver adds an explicit solid-rectangle check at completion.
# - Original generator assumed the anchor could be mostly empty except for the yellow block. That would leave holes
#   inside the required rectangle. The new generator therefore uses a solid yellow anchor body with boundary red ports.
#
# d) Placement legality
# - Output-first generation works in absolute rectangle coordinates. Every region owns a disjoint solid block of cells.
# - For each tree edge, the parent owns the overlap cells and the child adds two red protrusion cells into them. This
#   preserves non-red non-overlap and guarantees all overlaps are exactly red-on-red.
# - The assembled rectangle is shifted to a random in-bounds position, then only non-anchor pieces are scattered with
#   padding. The anchor stays in its assembled position, matching the original task family.
#
# e) Visual observability
# - Across examples, the visible invariant is strong: the scattered pieces always reassemble into one filled rectangle
#   centered on the yellow anchor, not a free-form jigsaw. Red tabs still provide the local assembly cue.
# - No hidden ordering uses color ids. Colors are sampled only to distinguish sprites visually.
#
# f) Visual cleanliness
# - The output uses one clean rectangle, and each piece is a simple solid slab with at most a few red tabs. There are
#   no internal holes or dense overlaps beyond the intended connector cells.
#
# g) Color inference
# - Colors do not transform. They are persistent piece identities. The only special color roles remain the original
#   ones: yellow marks the anchor, red marks connector overlaps, black is background.
#
# h) Generator-side anti-degeneration check
# - `generate_input()` rejects candidates unless the relaxed old-style search finds at least one non-rectangular valid
#   assembly in addition to exactly one rectangular assembly. This makes the rectangle constraint causally necessary.
#
# i) New solver assumptions audited against the generator
# - `main()` assumes exactly one component contains yellow: generator enforces one yellow anchor piece only.
# - `main()` assumes all valid overlaps are red-red only: generator constructs overlaps exclusively that way.
# - `main()` assumes exactly one rectangle-valid assembly exists: generator verifies this before returning an input.

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


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

    # D14 variant: only red-red overlaps are legal. Solid-body overlap must fail even if colors match.
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
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(sub))


def _enumerate_assemblies(
    grid: np.ndarray, *, rectangle_only: bool | None, max_solutions: int
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
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    solutions: list[np.ndarray] = []
    seen: set[bytes] = set()

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= max_solutions:
            return

        cur_mask = cur != bg
        cur_area = int(np.sum(cur_mask))
        if cur_area > target_area:
            return
        if rectangle_only:
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

        assert best_moves is not None

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
            if len(solutions) >= max_solutions:
                return

    dfs(placed, rest, red_counts)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered tabbed sprites into one filled rectangle anchored by the yellow center piece.

    Input structure:
    - bg = black (0). Several colored sprites are scattered apart from each other.
    - Exactly one sprite is the yellow anchor: it contains the yellow (4) core and stays fixed in place.
    - Red (2) cells mark overlap tabs on sprite boundaries; those are the only cells allowed to coincide.

    Rule:
    - Translate the non-anchor sprites so red tabs pair up into red-on-red overlaps.
    - No solid body cell may overlap another solid body cell.
    - After every connector is paired, the completed non-black shape must be a single solid rectangle.

    Key insight: Local red matches are not enough in this variant — many connector pairings can be legal, but only the
    assembly whose full silhouette becomes a rectangle is the valid answer.
    """
    sols = _enumerate_assemblies(np.asarray(grid, dtype=int), rectangle_only=True, max_solutions=2)
    assert len(sols) == 1, "assembly must have exactly one rectangle-valid solution"
    return sols[0]


# --------------- generator (D14) ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
SIDE_THICKNESS_RANGE = (4, 6)
STRIP_THICKNESS_RANGE = (3, 5)
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 1500

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
    H = top_h + anchor_h + bottom_h
    W = left_w + anchor_w + right_w
    own = np.full((H, W), -1, dtype=int)
    own[:top_h, :] = T
    own[top_h : top_h + anchor_h, :left_w] = L
    own[top_h : top_h + anchor_h, left_w : left_w + anchor_w] = A
    own[top_h : top_h + anchor_h, left_w + anchor_w :] = R
    own[top_h + anchor_h :, :] = B
    return own


def _region_cells(own: np.ndarray, region: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(own == region)]


def _adjacency_pairs(own: np.ndarray) -> dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]]:
    H, W = own.shape
    out: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    for r in range(H):
        for c in range(W):
            u = int(own[r, c])
            if r + 1 < H:
                v = int(own[r + 1, c])
                if u != v:
                    key = (min(u, v), max(u, v))
                    out.setdefault(key, []).append(((r, c), (r + 1, c)))
            if c + 1 < W:
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
    child: int,
    own: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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
    seg = oriented[start : start + 2]
    parent_cells = [pc for pc, _ in seg]
    return parent_cells, [cc for _, cc in seg]


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

    for child, par in parent.items():
        if child == A:
            continue
        key = (min(child, par), max(child, par))
        parent_cells, _ = _choose_segment(interfaces[key], par, child, own)
        for cell in parent_cells:
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

    H, W = own.shape
    return pieces, H, W


def _render_assembled(pieces: list[dict], grid_h: int, grid_w: int, sr: int, sc: int) -> np.ndarray | None:
    out = np.zeros((grid_h, grid_w), dtype=int)
    red_counts: dict[tuple[int, int], int] = {}
    for piece in pieces:
        r0 = sr + piece["offset"][0]
        c0 = sc + piece["offset"][1]
        placed = _try_place(out, piece["arr"], r0, c0, 0)
        if placed is None:
            return None
        out = placed
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
    pieces, out_h, out_w = _build_piece_layout()
    if out_h > grid_h or out_w > grid_w:
        return None

    sr = random.randint(0, grid_h - out_h)
    sc = random.randint(0, grid_w - out_w)
    out = _render_assembled(pieces, grid_h, grid_w, sr, sc)
    if out is None:
        return None

    anchor = next(piece for piece in pieces if piece["name"] == "anchor")
    anchor_pos = (sr + anchor["offset"][0], sc + anchor["offset"][1])
    inp = _scatter(pieces, anchor_pos, grid_h, grid_w)
    if inp is None:
        return None

    try:
        rect_solutions = _enumerate_assemblies(inp, rectangle_only=True, max_solutions=2)
        all_solutions = _enumerate_assemblies(inp, rectangle_only=None, max_solutions=4)
    except Exception:
        return None

    if len(rect_solutions) != 1:
        return None
    if not np.array_equal(rect_solutions[0], out):
        return None
    if len(all_solutions) < 2:
        return None
    if all(_is_solid_rectangle(sol, 0) for sol in all_solutions):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b D14 input")
