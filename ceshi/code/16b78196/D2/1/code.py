# Variant of 16b78196
# Dimension: D2 — 源对象：散落精灵的类型与属性 (src)
# Change: 单色连通精灵，面积8-13，外接矩形填充率0.54-0.80，不旋转不翻转，散布在杆两侧... → 精灵带锚点标记
#
# Design analysis
# 1. Concretisation / behavior trace
#    - Original input: a full-span bar with 6-9 boundary notches, plus 3-6 scattered monochrome sprites.
#      Root placement was found by exact geometric coverage of a notch hole.
#    - Variant input: every sprite is now a 4-connected multicolor component:
#        * one body color filling 7-12 cells
#        * one special anchor-color pixel attached to the body
#      Each used notch additionally contains one isolated anchor-color marker pixel inside the notch hole.
#      Decoy notches remain unmarked.
#    - Output: assembled chains still grow exactly as before, but each placed piece preserves its anchor-color
#      pixel. For roots, that anchor lands exactly on the notch marker location.
#    - Small trace:
#        original root search: try all translations where shape covers hole -> choose best
#        variant root search: notch marker m and sprite anchor a give one primary translation
#          (top,left) = m - a; then shape-coverage is only checked as a legality filter
#        difference: placement is now controlled by visible marker alignment, not by free-form hole matching.
# 2. Complexity calibration
#    - Anchor color is global, not per-piece, to stay within ARC's 10-color palette while keeping the cue
#      repeated across all pieces and used notches.
#    - Body colors remain distinct across pieces when possible, preserving chain readability.
#    - Used notches are still only 1-2 among 6-9 total notches, so the user must still separate signal from
#      decoys; anchors reduce ambiguity but do not trivialise the whole chain logic.
# 3. Implicit-assumption review against the original solver
#    - `find_components(... monochromatic=True)` used to identify every sprite directly. That breaks now because
#      each sprite is bichromatic. The new solver still uses monochromatic components only to find the largest
#      bar and to infer the anchor color; actual sprites are extracted as multicolor connected components after
#      removing bar cells.
#    - Original `bar_notches` treated every non-bar cell in the bar bbox as bg-hole cells. That remains valid
#      here because the anchor marker is intentionally inside the hole and should be covered by the root piece.
#    - Original root candidate generation enumerated all hole-covering translations. That assumption no longer
#      matches the variant. It is replaced by anchor-marker alignment, then coverage checking.
#    - Continuation search relied only on occupancy/cavity geometry, not piece color homogeneity. That still
#      holds, so it is intentionally preserved.
# 4. Placement legality
#    - Generator first constructs the full output chains, then assigns one anchor cell per piece. Roots choose
#      an anchor cell from inside their notch, so the corresponding input marker lies in a legal hole cell.
#    - Scattered input placement preserves relative geometry and anchor position, forbids overlap with the bar
#      or other pieces, and keeps pieces non-touching.
#    - Input used-notch markers are isolated single pixels adjacent to bar cells; the solver explicitly removes
#      bar cells before grouping multicolor objects, so markers are not merged into the bar or into sprites.
# 5. Visual observability
#    - Across examples the human now sees a repeated pattern: every scattered piece contains the same odd-colored
#      anchor pixel, and only some notches contain the same odd-colored marker. In the output those marked
#      notches are exactly where roots dock, with the odd-colored pixel ending up on the marker.
#    - This is visible in 2-3 examples without hidden numeric ordering.
# 6. Visual cleanliness
#    - Only one extra pixel per sprite and one extra pixel per used notch are added; no dense overlays or
#      excessive recoloring are introduced.
# 7. Color inferability
#    - The anchor color has one structural role everywhere: it marks the alignment point.
#      No other rule depends on color identities.
# 8. Source-task note
#    - The prompt cites 4c416de3 for anchor_points, but within the allowed files there is no readable solver
#      source for 4c416de3, only task JSON plus repository notes mentioning anchor-point behavior. This variant
#      therefore implements the anchor mechanism directly while staying within the stated D2 change.

from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np

for _ARC_UTILS_DIR in [
    "/home/lixinyu/snap/new/solvers",
    "/home/lixinyu/snap/new/arc2_solver_select",
]:
    if os.path.isdir(_ARC_UTILS_DIR) and _ARC_UTILS_DIR not in sys.path:
        sys.path.append(_ARC_UTILS_DIR)

from arc_utils import components_mask, find_components, get_bg


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _connected(cells: set[tuple[int, int]]) -> bool:
    if len(cells) <= 1:
        return True
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=3):
    """Grow a 4-connected shape of exactly `target` cells inside `available`."""
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 20):
            if len(cells) >= target:
                break
            frontier = [
                nb
                for r, c in cells
                for dr, dc in DIR4
                if (nb := (r + dr, c + dc)) in available and nb not in cells
            ]
            if not frontier:
                break
            if must_span_axis and span_range:
                A, B = span_range
                if must_span_axis == "cols":
                    cm, cx = min((c for _, c in cells), default=B), max((c for _, c in cells), default=A)
                    if cm > A:
                        picks = [nb for nb in frontier if nb[1] < cm]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                    if cx < B:
                        picks = [nb for nb in frontier if nb[1] > cx]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                else:
                    cm, cx = min((r for r, _ in cells), default=B), max((r for r, _ in cells), default=A)
                    if cm > A:
                        picks = [nb for nb in frontier if nb[0] < cm]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                    if cx < B:
                        picks = [nb for nb in frontier if nb[0] > cx]
                        if picks:
                            cells.add(rng.choice(picks))
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble anchored sprites into the central bar.

    Variant rule:
    - Every scattered piece is a multicolor 4-connected sprite with exactly one anchor-color pixel.
    - Each used bar notch contains one isolated marker of that same anchor color.
    - Root placement is determined primarily by anchor-marker alignment; hole-coverage remains a validity check.
    - Once roots are placed, chain extension still follows the original cavity-filling rule.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    mono.sort(key=lambda d: -d["area"])
    assert mono

    bar = mono[0]
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1

    # Anchor color = non-bar, non-bg color whose monochromatic components are all singletons.
    color_to_areas: dict[int, list[int]] = {}
    for comp in mono[1:]:
        col = int(comp["color"])
        color_to_areas.setdefault(col, []).append(int(comp["area"]))
    anchor_candidates = [
        (len(areas), col)
        for col, areas in color_to_areas.items()
        if areas and max(areas) == 1
    ]
    assert anchor_candidates
    anchor_color = max(anchor_candidates)[1]

    out = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_occ:
        out[r, c] = bar_color

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True

    def shape_from_coords(coords: np.ndarray):
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        rel = []
        anchor_rel = None
        body_colors = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            col = int(g[rr, cc])
            rel.append((rr - r0, cc - c0, col))
            if col == anchor_color:
                assert anchor_rel is None
                anchor_rel = (rr - r0, cc - c0)
            else:
                body_colors.add(col)
        assert anchor_rel is not None
        assert len(body_colors) == 1
        return {
            "rel": rel,
            "rel_set": {(dr, dc) for dr, dc, _ in rel},
            "anchor": anchor_rel,
            "body_color": next(iter(body_colors)),
            "h": r1 - r0 + 1,
            "w": c1 - c0 + 1,
        }

    # Multicolor pieces are extracted after removing bar cells, so isolated notch markers stay separate.
    fg_mask = (g != bg)
    for r, c in bar_occ:
        fg_mask[r, c] = False
    mixed_components = components_mask(fg_mask, connectivity=4)

    pieces = []
    for pts in mixed_components:
        coords = [(int(r), int(c)) for r, c in pts]
        if len(coords) == 1 and int(g[coords[0]]) == anchor_color:
            continue
        if len(coords) < 8:
            continue
        coords_arr = np.array(coords, dtype=int)
        r0 = int(coords_arr[:, 0].min())
        c0 = int(coords_arr[:, 1].min())
        pieces.append({"shape": shape_from_coords(coords_arr), "orig": (r0, c0)})

    def bar_notches():
        hole_mask = ~bar_mask
        notches = []
        for pts in components_mask(hole_mask, connectivity=4):
            rr = pts[:, 0]
            cc = pts[:, 1]
            top = int((rr == 0).sum())
            bottom = int((rr == bar_h - 1).sum())
            left = int((cc == 0).sum())
            right = int((cc == bar_w - 1).sum())
            if top + bottom + left + right == 0:
                continue
            hole = {(bar_r0 + int(r), bar_c0 + int(c)) for r, c in pts}
            markers = {(r, c) for r, c in hole if int(g[r, c]) == anchor_color}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole, "markers": markers, "mouth_n": mouth_n, "side": side})
        return notches

    notches = bar_notches()

    def pixels(v, top, left):
        return {(top + dr, left + dc) for dr, dc, _ in v["rel"]}

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

    def apply_piece(out_grid, v, top, left):
        for dr, dc, col in v["rel"]:
            out_grid[top + dr, left + dc] = int(col)

    def clear_piece(out_grid, v, top, left):
        for dr, dc, _ in v["rel"]:
            out_grid[top + dr, left + dc] = int(bg)

    def aligned_positions(axis, A, cover, v):
        if axis == "cols":
            left = A
            tops = set()
            for rN, cN in cover:
                for lr, lc in v["rel_set"]:
                    if cN == left + lc:
                        tops.add(rN - lr)
            for top in tops:
                yield top, left
        else:
            top = A
            lefts = set()
            for rN, cN in cover:
                for lr, lc in v["rel_set"]:
                    if rN == top + lr:
                        lefts.add(cN - lc)
            for left in lefts:
                yield top, left

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    root_cands = []
    for ni, notch in enumerate(notches):
        if len(notch["markers"]) != 1:
            continue
        marker = next(iter(notch["markers"]))
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            v = piece["shape"]
            top = int(marker[0] - v["anchor"][0])
            left = int(marker[1] - v["anchor"][1])
            pts = placement_pts(v, top, left, bar_occ, must_cover=notch["hole"])
            if pts is None:
                continue
            if axis == "cols":
                A, B = int(left), int(left + v["w"] - 1)
            else:
                A, B = int(top), int(top + v["h"] - 1)
            mv = abs(top - int(piece["orig"][0])) + abs(left - int(piece["orig"][1]))
            extra = len(pts) - len(notch["hole"])
            root_cands.append(
                (
                    (mv, extra, -int(notch["mouth_n"])),
                    {"ni": ni, "pi": pi, "top": top, "left": left, "axis": axis, "A": A, "B": B, "shape": v},
                )
            )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [x[1] for x in root_cands[:24]]

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
            head_top = int(head["top"])
            head_left = int(head["left"])
            head_h = int(head["shape"]["h"])
            head_w = int(head["shape"]["w"])
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
            moves.sort(key=lambda x: (abs(x[1] - pieces[x[0]]["orig"][0]) + abs(x[2] - pieces[x[0]]["orig"][1]), len(x[3])))
            return moves

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for pi, top, left, pts, v in moves[:8]:
                apply_piece(out_grid, v, top, left)
                old_head = ch["head"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": v}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["head"] = old_head
                clear_piece(out_grid, v, top, left)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
        return False

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
                apply_piece(out0, pieces[root["pi"]]["shape"], root["top"], root["left"])
                occ0 |= pts
                chains.append(
                    {
                        "axis": root["axis"],
                        "A": root["A"],
                        "B": root["B"],
                        "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                    }
                )
            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


def _choose_anchor_cell(piece_cells, preferred, order_key):
    ordered = sorted(preferred, key=order_key) + sorted(set(piece_cells) - set(preferred), key=order_key)
    for rc in ordered:
        if len(piece_cells) <= 1 or _connected(set(piece_cells) - {rc}):
            return rc
    return ordered[0]


def _piece_anchor_key(orientation, side):
    if orientation == "horizontal":
        if side == "top":
            return lambda rc: (-rc[0], rc[1])
        return lambda rc: (rc[0], rc[1])
    if side == "left":
        return lambda rc: (-rc[1], rc[0])
    return lambda rc: (rc[1], rc[0])


def _gen_attempt():
    import random as _random

    rng = _random.Random()
    H, W = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    palette = list(range(1, 10))
    rng.shuffle(palette)
    bar_color = palette.pop()
    anchor_color = palette.pop()
    thickness = rng.randint(5, 7)

    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * H)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = min(H - 1, bar_r0 + thickness - 1)
        bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        space_before, space_after = bar_r0, H - 1 - bar_r1
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = min(W - 1, bar_c0 + thickness - 1)
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

    if len(palette) < sum(chain_lens):
        return None
    body_colors = palette[: sum(chain_lens)]
    color_idx = 0

    all_chain_pieces = []
    notch_holes: list[set[tuple[int, int]]] = []
    occ_set = set(bar_cells)

    for chain_len in chain_lens:
        strip_width = rng.randint(4, 5)
        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next(
                (
                    s
                    for s in sides
                    if (space_before if s == "top" else space_after) >= chain_len * 2 + 3
                ),
                None,
            )
            if side is None:
                return None
            sc0 = rng.randint(1, W - strip_width - 1)
            sc1 = sc0 + strip_width - 1
            sA, sB = sc0, sc1
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            side = next(
                (
                    s
                    for s in sides
                    if (space_before if s == "left" else space_after) >= chain_len * 2 + 3
                ),
                None,
            )
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
        n_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
        if len(n_avail) < notch_area:
            return None
        notch = _grow_connected(rng, n_avail, notch_area, {rng.choice(list(n_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        notch_holes.append(notch)

        chain_pieces = []
        for piece_idx in range(chain_len):
            piece_color = body_colors[color_idx]
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
                avail = {
                    (r, c)
                    for r in range(pr0, pr1 + 1)
                    for c in range(pc0, pc1 + 1)
                    if (r, c) in notch or (r, c) not in occ_set
                }
                piece_cells = _grow_connected(
                    rng,
                    avail,
                    target,
                    notch,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=5,
                )
                if piece_cells is None:
                    return None
                if not any(not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1) for r, c in piece_cells):
                    return None
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
                    ext = {
                        (r, c)
                        for r in range(es[0], es[1] + 1)
                        for c in range(sc0, sc1 + 1)
                        if (r, c) not in occ_set
                    } if es[0] <= es[1] else set()
                else:
                    es = (max(0, prev_c0 - chain_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    ext = {
                        (r, c)
                        for r in range(sA, sB + 1)
                        for c in range(es[0], es[1] + 1)
                        if (r, c) not in occ_set
                    } if es[0] <= es[1] else set()

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
                        remaining = {
                            (r, c)
                            for r in range(t_r0, t_r1 + 1)
                            for c in range(sc0, sc1 + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    else:
                        remaining = {
                            (r, c)
                            for r in range(sA, sB + 1)
                            for c in range(t_c0, t_c1 + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    if remaining:
                        return None

            if not (8 <= len(piece_cells) <= 13):
                return None
            chain_pieces.append(
                {
                    "body_color": piece_color,
                    "coords": set(piece_cells),
                    "orientation": orientation,
                    "side": side,
                    "notch": set(notch) if piece_idx == 0 else None,
                }
            )
            occ_set |= piece_cells

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
        d_avail -= all_h | {(r + dr, c + dc) for r, c in all_h for dr, dc in DIR4}
        if not d_avail:
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

    used_markers = []
    for piece in all_chain_pieces:
        pref = piece["coords"] & piece["notch"] if piece["notch"] is not None else set()
        anchor = _choose_anchor_cell(piece["coords"], pref, _piece_anchor_key(piece["orientation"], piece["side"]))
        piece["anchor"] = anchor
        if piece["notch"] is not None:
            if anchor not in piece["notch"]:
                return None
            used_markers.append(anchor)

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = bar_color
    for piece in all_chain_pieces:
        for r, c in piece["coords"]:
            output[r, c] = anchor_color if (r, c) == piece["anchor"] else piece["body_color"]

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color
    for r, c in used_markers:
        input_grid[r, c] = anchor_color

    placed_occ = set(bar_cells) | set(used_markers)
    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)

    for piece in pieces_order:
        coords = piece["coords"]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        p_r0, p_c0 = min(rs), min(cs)
        rel = []
        for r, c in coords:
            rel.append((r - p_r0, c - p_c0, anchor_color if (r, c) == piece["anchor"] else piece["body_color"]))
        p_h = max(rs) - p_r0 + 1
        p_w = max(cs) - p_c0 + 1

        placed = False
        for _ in range(500):
            gap = rng.randint(2, 7)
            if orientation == "horizontal":
                top = (bar_r0 - gap - p_h) if rng.random() < 0.5 else (bar_r1 + gap + 1)
                left = rng.randint(0, max(0, W - p_w))
            else:
                left = (bar_c0 - gap - p_w) if rng.random() < 0.5 else (bar_c1 + gap + 1)
                top = rng.randint(0, max(0, H - p_h))
            if top < 0 or top + p_h > H or left < 0 or left + p_w > W:
                continue
            pts = {(top + dr, left + dc) for dr, dc, _ in rel}
            if pts & placed_occ:
                continue
            if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in DIR4):
                continue
            for dr, dc, col in rel:
                input_grid[top + dr, left + dc] = col
            placed_occ |= pts
            placed = True
            break
        if not placed:
            return None

    return input_grid, output


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
