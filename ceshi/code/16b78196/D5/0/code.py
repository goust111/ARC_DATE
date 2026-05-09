# Variant of 16b78196
# Dimension: D5 — 变换：放置后的颜色或形状变化 (transform)
# Change: 无变换——精灵保持原色原形直接放置到目标位置... → 放置后重新着色为杆色

from __future__ import annotations

import numpy as np

from arc_utils import components_mask, find_components, get_bg

# Design analysis for the D5 recolor variant
# ==========================================
# a) Concrete behavior and trace
# - Input stays structurally identical to the source task: a full-span bar with
#   several boundary notches, plus scattered monochrome sprites of distinct
#   non-bar colors.
# - Output changes only after successful placement. In the source task, each
#   placed sprite keeps its own color, so the finished chain is color-segmented.
#   In this variant, the same shapes are inserted into the same notch/strip
#   positions, but every placed sprite cell is recolored to the bar color.
# - Small trace on one root placement:
#   source: root candidate covers notch cells, solver writes piece color 6;
#   variant: root candidate still covers the same notch cells, but solver writes
#   bar color 3. The chain geometry is unchanged; only the rendered color differs.
#   The same applies to continuation pieces that fill the previous bbox cavity.
#
# b) Complexity calibration
# - The recolor rule is intentionally global and single-valued: "all placed chain
#   cells become the bar color". This is visually strong without adding extra
#   branching or many color mappings.
# - Input sprites remain distinctly colored and separate from the bar, so the
#   solver still has enough visible structure to identify the pieces. Output uses
#   fewer colors, but the chain silhouettes remain nontrivial because shape/bbox
#   cavities are unchanged.
#
# c) Hidden-assumption audit in main()
# - `pieces.append({"color": ...})`: source solver stored piece colors because it
#   later rendered them. In the variant the color is no longer used for output,
#   but keeping it is harmless and preserves data shape.
# - All `out_grid[r, c] = pieces[pi]["color"]` writes depended on the source
#   assumption "placed pieces keep their own color". That assumption no longer
#   holds, so every successful placement must instead write `bar_color`.
# - The cavity test `out_grid[r, c] == bg` still holds because placed cells are
#   non-bg regardless of whether they are piece-colored or bar-colored.
# - Undo logic that resets explored placements to `bg` remains valid because the
#   search only backtracks non-bar candidate piece cells; bar cells are restored
#   separately from `bar_occ`.
#
# d) Placement legality
# - Geometry generation is unchanged: no overlaps, in-bounds placement, fixed
#   strip width per chain, root covers a used notch, continuation pieces cover
#   the previous bbox cavity and extend outward.
# - Because input pieces keep non-bar colors, monochromatic component extraction
#   still separates the bar from each scattered piece exactly as in the source.
#
# e) Visual observability
# - Across 2-4 examples, the user still sees colorful loose sprites in the input
#   but a single-color merged bar+chain structure in the output. The only way to
#   explain the missing piece colors is that they were inserted and recolored to
#   match the bar.
# - The rule is inferable from few samples because it has one stable outcome
#   across all pieces and all chain positions.
#
# f) Visual cleanliness
# - No extra overlay is introduced. Recoloring actually simplifies the output
#   image by collapsing chain colors into the bar color while preserving clean
#   silhouettes of the assembled strips.
#
# g) Color-change inferability
# - The mapping is not arbitrary; it is structural: "piece color -> bar color of
#   the same instance". There is only one active target color per example, and it
#   is already visibly marked by the largest central component.
#
# New solver assumptions introduced by the variant
# ================================================
# - The bar remains the largest monochrome component in the input; the generator
#   preserves that by leaving only the input unchanged and recoloring only the
#   output.
# - Successful placements are rendered uniformly with `bar_color`, so the solver
#   must never rely on output colors to distinguish chains or piece order. The
#   search here relies only on geometry, strip alignment, and cavities, which the
#   generator preserves exactly.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble colored sprites into a central bar's notches, chaining successive
    pieces along fixed-width strips, then recolor every placed piece to the bar
    color so the finished chain visually merges with the bar.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    bar = comps[0]
    pieces_raw = comps[1:]

    out = np.full((H, W), int(bg), dtype=int)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    def shape_from_coords(coords: np.ndarray, bbox):
        r0, c0, r1, c1 = map(int, bbox)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        rel_set = set(rel)
        return {"rel": rel, "rel_set": rel_set, "h": h, "w": w}

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    def bar_notches(bar_mask, bar_r0, bar_c0):
        empty = ~bar_mask
        h, w = map(int, empty.shape)
        notches = []
        for pts in components_mask(empty, connectivity=4):
            r = pts[:, 0]
            c = pts[:, 1]
            top = int((r == 0).sum())
            bottom = int((r == h - 1).sum())
            left = int((c == 0).sum())
            right = int((c == w - 1).sum())
            if top + bottom + left + right == 0:
                continue

            hole_gl = {(bar_r0 + int(rr), bar_c0 + int(cc)) for rr, cc in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    def pixels(v, top, left):
        return {(top + lr, left + lc) for lr, lc in v["rel"]}

    def placement_pts(v, top, left, occ, must_cover=None, strip=None):
        if top < 0 or left < 0 or top + v["h"] > H or left + v["w"] > W:
            return None
        if strip is not None:
            axis, A, B = strip
            if axis == "cols":
                if left != A or v["w"] != B - A + 1:
                    return None
            else:
                if top != A or v["h"] != B - A + 1:
                    return None
        pts = pixels(v, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        return pts

    def aligned_positions(axis, A, cover, v):
        if axis == "cols":
            left = A
            tops = set()
            for r_n, c_n in cover:
                for lr, lc in v["rel"]:
                    if c_n == left + lc:
                        tops.add(r_n - lr)
            for top in tops:
                yield top, left
        else:
            top = A
            lefts = set()
            for r_n, c_n in cover:
                for lr, lc in v["rel"]:
                    if r_n == top + lr:
                        lefts.add(c_n - lc)
            for left in lefts:
                yield top, left

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, v):
        pos = set()
        for r_h, c_h in hole:
            for lr, lc in v["rel"]:
                pos.add((r_h - lr, c_h - lc))
        return pos

    root_cands = []
    for ni, n in enumerate(notches):
        axis = n["axis"]
        for pi, p in enumerate(pieces):
            v = p["shape"]
            for top, left in hole_positions(n["hole"], v):
                pts = placement_pts(v, top, left, bar_occ, must_cover=n["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + v["w"] - 1)
                else:
                    A, B = int(top), int(top + v["h"] - 1)
                mv = abs(int(top) - int(p["orig"][0])) + abs(int(left) - int(p["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), -int(n["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": v},
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:20]]

    best_out = out.copy()
    best_key = None

    def search(out_grid, occ_set, chains, remaining):
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (placed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        def gen_extend_moves(chain, remaining_set):
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = pixels(head["shape"], head["top"], head["left"])
            head_top, head_left = int(head["top"]), int(head["left"])
            head_h, head_w = int(head["shape"]["h"]), int(head["shape"]["w"])

            cavity = {
                (r, c)
                for r in range(head_top, head_top + head_h)
                for c in range(head_left, head_left + head_w)
                if out_grid[r, c] == bg
            }
            if not cavity:
                return []

            moves = []
            for pi in remaining_set:
                v = pieces[pi]["shape"]
                for top, left in aligned_positions(axis, A, cavity, v):
                    pts = placement_pts(v, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                    if pts is None:
                        continue
                    if not touches(pts, head_pts):
                        continue
                    moves.append((pi, top, left, pts, v))
            return moves

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for pi, top, left, pts, v in moves[:8]:
                for r, c in pts:
                    out_grid[r, c] = bar_color
                old_head = ch["head"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": v}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
        return False

    from itertools import combinations

    for k in range(1, min(3, len(cand_roots)) + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(bar_occ)
            out0 = np.full((H, W), int(bg), dtype=int)
            for r, c in bar_occ:
                out0[r, c] = bar_color

            chains = []
            ok = True
            for idx in roots_sel:
                root = cand_roots[idx]
                if root["pi"] in used_p or root["ni"] in used_n:
                    ok = False
                    break
                pts = pixels(root["shape"], root["top"], root["left"])
                if pts & occ0:
                    ok = False
                    break
                used_p.add(root["pi"])
                used_n.add(root["ni"])
                for r, c in pts:
                    out0[r, c] = bar_color
                occ0 |= pts
                chains.append({"axis": root["axis"], "A": root["A"], "B": root["B"], "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]}})

            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


def _connected(cells):
    """Check 4-connectivity of a cell set."""
    if len(cells) <= 1:
        return True
    cells_set = set(cells)
    start = next(iter(cells_set))
    visited = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb in cells_set and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(cells_set)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=3):
    """Grow a 4-connected shape of exactly *target* cells within *available*."""
    d4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 20):
            if len(cells) >= target:
                break
            frontier = [nb for r, c in cells for dr, dc in d4 if (nb := (r + dr, c + dc)) in available and nb not in cells]
            if not frontier:
                break
            if must_span_axis and span_range:
                A, B = span_range
                if must_span_axis == "cols":
                    cm, cx = min((c for _, c in cells), default=B), max((c for _, c in cells), default=A)
                    if cm > A:
                        p = [nb for nb in frontier if nb[1] < cm]
                        if p:
                            cells.add(rng.choice(p))
                            continue
                    if cx < B:
                        p = [nb for nb in frontier if nb[1] > cx]
                        if p:
                            cells.add(rng.choice(p))
                            continue
                else:
                    cm, cx = min((r for r, _ in cells), default=B), max((r for r, _ in cells), default=A)
                    if cm > A:
                        p = [nb for nb in frontier if nb[0] < cm]
                        if p:
                            cells.add(rng.choice(p))
                            continue
                    if cx < B:
                        p = [nb for nb in frontier if nb[0] > cx]
                        if p:
                            cells.add(rng.choice(p))
                            continue
            cells.add(rng.choice(frontier))
        if len(cells) < target:
            continue
        while len(cells) > target:
            removable = []
            for rc in cells:
                if rc in seeds:
                    continue
                if must_span_axis and span_range:
                    A, B = span_range
                    if must_span_axis == "cols":
                        if rc[1] == A and sum(1 for _, c2 in cells if c2 == A) <= 1:
                            continue
                        if rc[1] == B and sum(1 for _, c2 in cells if c2 == B) <= 1:
                            continue
                    else:
                        if rc[0] == A and sum(1 for r2, _ in cells if r2 == A) <= 1:
                            continue
                        if rc[0] == B and sum(1 for r2, _ in cells if r2 == B) <= 1:
                            continue
                if _connected(cells - {rc}):
                    removable.append(rc)
            if not removable:
                break
            cells.remove(rng.choice(removable))
        if len(cells) != target:
            continue
        if must_span_axis and span_range:
            A, B = span_range
            if must_span_axis == "cols":
                if min(c for _, c in cells) != A or max(c for _, c in cells) != B:
                    continue
            else:
                if min(r for r, _ in cells) != A or max(r for r, _ in cells) != B:
                    continue
        if _connected(cells):
            return cells
    return None


def _gen_attempt():
    """Build one (input, output) pair; returns (input, output) or None."""
    import random as _random

    rng = _random.Random()
    H, W = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    bar_color = rng.randint(1, 9)
    thickness = rng.randint(5, 7)

    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * H)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = bar_r0 + thickness - 1
        if bar_r1 >= H:
            bar_r1 = H - 1
            bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        space_before, space_after = bar_r0, H - 1 - bar_r1
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = bar_c0 + thickness - 1
        if bar_c1 >= W:
            bar_c1 = W - 1
            bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        space_before, space_after = bar_c0, W - 1 - bar_c1

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    num_chains = rng.randint(1, 2)
    chain_lens = [3] if num_chains == 1 else [rng.randint(2, 3) for _ in range(2)]
    if num_chains == 2:
        while not (3 <= sum(chain_lens) <= 6):
            chain_lens = [rng.randint(2, 3) for _ in range(2)]

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    color_idx = 0

    all_chain_pieces = []
    notch_holes: list[set] = []
    occ_set = set(bar_cells)

    for chain_idx in range(num_chains):
        chain_len = chain_lens[chain_idx]
        strip_width = rng.randint(4, 5)

        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "top" else space_after) >= chain_len * 2 + 3), None)
            if side is None:
                return None
            sc0 = rng.randint(1, W - strip_width - 1)
            sc1 = sc0 + strip_width - 1
            sA, sB = sc0, sc1
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "left" else space_after) >= chain_len * 2 + 3), None)
            if side is None:
                return None
            sr0 = rng.randint(1, H - strip_width - 1)
            sr1 = sr0 + strip_width - 1
            sA, sB = sr0, sr1

        notch_depth = rng.randint(1, 2)
        notch_area = rng.randint(1, min(4, notch_depth * strip_width))

        if orientation == "horizontal":
            nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
            nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
            n_avail = {(r, c) for r in range(nr0, nr1 + 1) for c in range(sc0, sc1 + 1) if (r, c) in bar_cells}
        else:
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            n_avail = {(r, c) for r in range(sA, sB + 1) for c in range(nc0, nc1 + 1) if (r, c) in bar_cells}

        all_holes = set().union(*notch_holes) if notch_holes else set()
        n_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
        if len(n_avail) < notch_area:
            return None

        notch = _grow_connected(rng, n_avail, notch_area, {rng.choice(list(n_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        notch_holes.append(notch)

        chain_pieces = []
        for piece_idx in range(chain_len):
            if color_idx >= len(avail_colors):
                return None
            piece_color = avail_colors[color_idx]
            color_idx += 1
            piece_area = rng.randint(8, 13)
            chain_dim = rng.randint(2, 3)

            if piece_idx == 0:
                if orientation == "horizontal":
                    if side == "top":
                        pr0, pr1 = max(0, bar_r0 - chain_dim), max(r for r, _ in notch)
                    else:
                        pr0, pr1 = min(r for r, _ in notch), min(H - 1, bar_r1 + chain_dim)
                    pc0, pc1 = sc0, sc1
                else:
                    if side == "left":
                        pc0, pc1 = max(0, bar_c0 - chain_dim), max(c for _, c in notch)
                    else:
                        pc0, pc1 = min(c for _, c in notch), min(W - 1, bar_c1 + chain_dim)
                    pr0, pr1 = sA, sB

                bbox_area = (pr1 - pr0 + 1) * (pc1 - pc0 + 1)
                target = max(max(8, int(bbox_area * 0.54)), min(piece_area, int(bbox_area * 0.80)))
                if target > bbox_area or target < 8:
                    return None

                avail = {(r, c) for r in range(pr0, pr1 + 1) for c in range(pc0, pc1 + 1) if (r, c) in notch or (r, c) not in occ_set}
                root = _grow_connected(
                    rng,
                    avail,
                    target,
                    notch,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=5,
                )
                if root is None:
                    return None
                if not any(not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1) for r, c in root):
                    return None
                piece_cells = root
            else:
                prev_coords = chain_pieces[-1]["coords"]
                prs = [r for r, _ in prev_coords]
                pcs = [c for _, c in prev_coords]
                prev_r0, prev_r1 = min(prs), max(prs)
                prev_c0, prev_c1 = min(pcs), max(pcs)

                if orientation == "horizontal":
                    cavity = {
                        (r, c)
                        for r in range(prev_r0, prev_r1 + 1)
                        for c in range(sc0, sc1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                else:
                    cavity = {
                        (r, c)
                        for r in range(sA, sB + 1)
                        for c in range(prev_c0, prev_c1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                if not cavity:
                    return None

                if orientation == "horizontal":
                    es = (max(0, prev_r0 - chain_dim), prev_r0 - 1) if side == "top" else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    ext = {(r, c) for r in range(es[0], es[1] + 1) for c in range(sc0, sc1 + 1) if (r, c) not in occ_set} if es[0] <= es[1] else set()
                else:
                    es = (max(0, prev_c0 - chain_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    ext = {(r, c) for r in range(sA, sB + 1) for c in range(es[0], es[1] + 1) if (r, c) not in occ_set} if es[0] <= es[1] else set()

                all_avail = cavity | ext
                target = min(13, max(len(cavity), min(piece_area, len(all_avail))))
                if len(cavity) > 13 or target < 8:
                    return None

                is_tail = piece_idx == chain_len - 1
                if is_tail:
                    target = min(13, len(all_avail))

                piece_cells = _grow_connected(
                    rng,
                    all_avail,
                    target,
                    cavity,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=5,
                )
                if piece_cells is None or not cavity.issubset(piece_cells):
                    return None

                if is_tail:
                    prs2 = [r for r, _ in piece_cells]
                    pcs2 = [c for _, c in piece_cells]
                    t_r0, t_r1 = min(prs2), max(prs2)
                    t_c0, t_c1 = min(pcs2), max(pcs2)
                    if orientation == "horizontal":
                        remaining = {(r, c) for r in range(t_r0, t_r1 + 1) for c in range(sc0, sc1 + 1) if (r, c) not in occ_set and (r, c) not in piece_cells}
                    else:
                        remaining = {(r, c) for r in range(sA, sB + 1) for c in range(t_c0, t_c1 + 1) if (r, c) not in occ_set and (r, c) not in piece_cells}
                    if remaining:
                        return None

            if not (8 <= len(piece_cells) <= 13):
                return None
            chain_pieces.append({"color": piece_color, "coords": piece_cells})
            occ_set |= piece_cells

        if len(chain_pieces) < chain_len:
            return None
        all_chain_pieces.extend(chain_pieces)

    if not (3 <= len(all_chain_pieces) <= 6):
        return None

    num_decoys = rng.randint(4, 7)
    decoys = 0
    for _ in range(500):
        if decoys >= num_decoys:
            break
        depth = rng.randint(1, 2)
        area = rng.randint(1, min(6, depth * 5))
        if orientation == "horizontal":
            ds = rng.choice(["top", "bottom"])
            dr0 = bar_r0 if ds == "top" else bar_r1 - depth + 1
            dr1 = bar_r0 + depth - 1 if ds == "top" else bar_r1
            dc0 = rng.randint(bar_c0, bar_c1)
            dc1 = min(bar_c1, dc0 + rng.randint(0, 4))
        else:
            ds = rng.choice(["left", "right"])
            dc0 = bar_c0 if ds == "left" else bar_c1 - depth + 1
            dc1 = bar_c0 + depth - 1 if ds == "left" else bar_c1
            dr0 = rng.randint(bar_r0, bar_r1)
            dr1 = min(bar_r1, dr0 + rng.randint(0, 4))

        d_avail = {(r, c) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        all_h = set().union(*notch_holes) if notch_holes else set()
        d_avail -= all_h | {(r + dr, c + dc) for r, c in all_h for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
        if len(d_avail) < 1:
            continue
        area = min(area, len(d_avail))
        dn = _grow_connected(rng, d_avail, area, {rng.choice(list(d_avail))})
        if dn is None:
            continue
        if not _connected(bar_cells - dn):
            continue
        bar_cells -= dn
        notch_holes.append(dn)
        decoys += 1
    if decoys < 4:
        return None

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr, cc = pts[:, 0], pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return None

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = bar_color
    for p in all_chain_pieces:
        for r, c in p["coords"]:
            output[r, c] = bar_color

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color

    placed_occ = set(bar_cells)
    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)

    for p in pieces_order:
        coords = p["coords"]
        color = p["color"]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        p_r0, p_c0 = min(rs), min(cs)
        rel = [(r - p_r0, c - p_c0) for r, c in coords]
        p_h, p_w = max(rs) - p_r0 + 1, max(cs) - p_c0 + 1

        placed = False
        for _ in range(500):
            gap = rng.randint(2, 7)
            if orientation == "horizontal":
                top = bar_r0 - gap - p_h if rng.random() < 0.5 else bar_r1 + gap + 1
                left = rng.randint(0, max(0, W - p_w))
            else:
                left = bar_c0 - gap - p_w if rng.random() < 0.5 else bar_c1 + gap + 1
                top = rng.randint(0, max(0, H - p_h))
            if top < 0 or top + p_h > H or left < 0 or left + p_w > W:
                continue
            pts = {(top + dr, left + dc) for dr, dc in rel}
            if pts & placed_occ:
                continue
            if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                continue
            for r, c in pts:
                input_grid[r, c] = color
            placed_occ |= pts
            placed = True
            break
        if not placed:
            return None

    return input_grid, output


def generate_input():
    """Generate a valid input grid for the D5 recolor variant of 16b78196."""
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
