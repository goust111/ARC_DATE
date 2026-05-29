# Combo variant of 16b78196 (index 2)
# combo_spec: D1/0+D3/1
# Name: 颜色双重约束匹配

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

from arc_utils import components_mask, find_components, get_bg


"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
  Source task:
    - a central full-span bar has several open notches
    - the solver can try every scattered piece against every notch
    - geometry decides which piece becomes a root, then the rest of the chain
      grows by filling the previous piece's bbox cavity
  Combo task:
    - every notch is visibly marked by a colored inner rim on the bar itself
    - only 1-2 of those notch colors reappear on scattered root pieces
    - a root candidate is generated only when piece.color == notch.hint_color
    - after that color filter, geometry still verifies that the placement really
      covers the hole and fixes the chain strip
  Small trace:
    - suppose the bar has five colored notches: red, teal, yellow, maroon, grey
    - scattered pieces include one red root piece, one teal root piece, plus
      later-chain pieces in other colors
    - original logic would test the red piece against all five notches
    - combo logic first maps red -> red-notch and teal -> teal-notch, so the
      candidate set collapses to two notch-piece pairs; shape fitting becomes a
      legality check rather than the primary identification mechanism

b) Complexity calibration
  The change must stay visible. If only the used notches were colored, the task
  would look too close to a small cue puzzle. Therefore every notch is colored.
  At the same time, colors are limited to the ARC palette, so the generator uses
  4-6 total notches and makes all notch colors distinct. This keeps the mapping
  human-readable without creating impossible color-budget constraints.

c) Implicit assumption audit against the original main()
  - `bar = comps[0]` no longer means "the whole bar" because the bar is now
    multi-colored. We recover the largest monochrome base block, then absorb all
    4-neighbor non-bg components touching it into the union bar.
  - `pieces_raw = comps[1:]` is invalid because colored rim fragments would be
    mistaken for scattered pieces. After union-bar recovery, only non-bar
    components remain as piece candidates.
  - `bar_mask` cannot be built from one color only; it must cover the full union
    bar, otherwise rim cells would be misread as holes.
  - restoring the board after backtracking cannot repaint the bar with a single
    color. The solver stores per-cell bar colors and restores them exactly.
  - root enumeration in the original solver assumes every notch is geometrically
    eligible. Here the first filter is notch.hint_color == piece.color; geometry
    only validates placements inside that color-matched subset.
  - once every notch is colored, some colors are deliberate decoys. The solver
    must therefore require roots exactly for the notches whose colors are present
    among scattered pieces, not for all colored notches.

d) Placement legality
  The generator still works output-first:
    - build the bar
    - carve 1-2 used notches and 3-4 decoy notches
    - grow each chain outward in a fixed strip
    - compute a non-overlapping rim for each notch from the final bar geometry
    - paint each rim with a unique color
    - scatter pieces away from the bar and away from each other
  This guarantees in-bounds placement, non-overlap, and that each colored notch
  remains visually distinct.

e) Visual observability
  Across a few examples, a human sees that:
    - every notch has a colored rim
    - only some rim colors reappear on scattered pieces
    - those matching-colored pieces become the roots
    - later pieces still follow the original cavity-filling chain rule
  The color mapping is therefore directly visible; it does not depend on numeric
  color ids or hidden ordering.

f) Visual cleanliness
  The rim lives on the bar cells adjacent to each hole, so there is no extra
  overlay layer floating outside the bar. Scattered pieces are kept at least two
  cells away from the bar, which prevents accidental merging with the rim.

g) Change inferability
  All notch colors are unique within one sample. Used-notch colors equal root
  piece colors; decoy-notch colors are absent from every piece. This gives a
  crisp same-color relation that a human can infer from 2-4 examples.

Additional solver assumptions introduced by the combo and guaranteed by the
generator:
  1. every rim component touching the base bar belongs to the bar, not to a
     scattered piece; scattering distance >= 2 guarantees that
  2. each notch rim has a single dominant non-base color; the generator paints
     each rim uniformly
  3. every root color appears on exactly one notch rim; the generator uses
     distinct root colors and distinct notch colors
  4. every matched-color notch has at least one legal geometric root placement;
     output-first construction guarantees this
"""


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _shape_from_coords(coords: np.ndarray, bbox):
    r0, c0, r1, c1 = map(int, bbox)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}


def _pixels(shape, top, left):
    return {(top + dr, left + dc) for dr, dc in shape["rel"]}


def _connected(cells):
    if len(cells) <= 1:
        return True
    todo = [next(iter(cells))]
    seen = set()
    while todo:
        rc = todo.pop()
        if rc in seen:
            continue
        seen.add(rc)
        r, c = rc
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                todo.append(nb)
    return len(seen) == len(cells)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=4):
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
            if must_span_axis is not None and span_range is not None:
                A, B = span_range
                if must_span_axis == "cols":
                    cur_min = min((c for _, c in cells), default=B)
                    cur_max = max((c for _, c in cells), default=A)
                    if cur_min > A:
                        cand = [nb for nb in frontier if nb[1] < cur_min]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if cur_max < B:
                        cand = [nb for nb in frontier if nb[1] > cur_max]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                else:
                    cur_min = min((r for r, _ in cells), default=B)
                    cur_max = max((r for r, _ in cells), default=A)
                    if cur_min > A:
                        cand = [nb for nb in frontier if nb[0] < cur_min]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if cur_max < B:
                        cand = [nb for nb in frontier if nb[0] > cur_max]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
            cells.add(rng.choice(frontier))
        if len(cells) < target:
            continue

        while len(cells) > target:
            removable = []
            for rc in cells:
                if rc in seeds:
                    continue
                if must_span_axis is not None and span_range is not None:
                    A, B = span_range
                    if must_span_axis == "cols":
                        if rc[1] == A and sum(1 for _, c in cells if c == A) <= 1:
                            continue
                        if rc[1] == B and sum(1 for _, c in cells if c == B) <= 1:
                            continue
                    else:
                        if rc[0] == A and sum(1 for r, _ in cells if r == A) <= 1:
                            continue
                        if rc[0] == B and sum(1 for r, _ in cells if r == B) <= 1:
                            continue
                if _connected(cells - {rc}):
                    removable.append(rc)
            if not removable:
                break
            cells.remove(rng.choice(removable))
        if len(cells) != target:
            continue

        if must_span_axis is not None and span_range is not None:
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


def _collect_bar_and_pieces(g: np.ndarray, bg: int):
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    base = comps[0]
    bar_ids = {0}
    bar_occ = {(int(r), int(c)) for r, c in base["coords"]}

    changed = True
    while changed:
        changed = False
        for idx, comp in enumerate(comps[1:], start=1):
            if idx in bar_ids:
                continue
            coords = [(int(r), int(c)) for r, c in comp["coords"]]
            if any((r + dr, c + dc) in bar_occ for r, c in coords for dr, dc in DIR4):
                bar_ids.add(idx)
                bar_occ.update(coords)
                changed = True

    piece_comps = [comp for idx, comp in enumerate(comps) if idx not in bar_ids]
    bar_colors = {rc: int(g[rc]) for rc in bar_occ}
    rs = [r for r, _ in bar_occ]
    cs = [c for _, c in bar_occ]
    bar_bbox = (min(rs), min(cs), max(rs), max(cs))
    return base, bar_occ, bar_colors, bar_bbox, piece_comps


def _bar_notches(bar_mask, bar_r0, bar_c0, bar_occ, bar_colors, base_bar_color):
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
        ring = {
            (r0 + dr, c0 + dc)
            for r0, c0 in hole_gl
            for dr, dc in DIR4
            if (r0 + dr, c0 + dc) in bar_occ
        }
        rim_colors = [bar_colors[rc] for rc in ring if bar_colors[rc] != base_bar_color]
        hint_color = Counter(rim_colors).most_common(1)[0][0] if rim_colors else None
        notches.append(
            {
                "axis": axis,
                "side": side,
                "hole": hole_gl,
                "mouth_n": mouth_n,
                "hint_color": hint_color,
                "ring": ring,
            }
        )
    return notches


def _restore_bar(grid_out, bar_colors):
    for (r, c), col in bar_colors.items():
        grid_out[r, c] = int(col)


def _hole_positions(hole, shape):
    pos = set()
    for r_hole, c_hole in hole:
        for dr, dc in shape["rel"]:
            pos.add((r_hole - dr, c_hole - dc))
    return pos


def _placement_pts(shape, top, left, occ, H, W, must_cover=None, strip=None):
    if top < 0 or left < 0 or top + shape["h"] > H or left + shape["w"] > W:
        return None
    if strip is not None:
        axis, A, B = strip
        if axis == "cols":
            if left != A or shape["w"] != B - A + 1:
                return None
        else:
            if top != A or shape["h"] != B - A + 1:
                return None
    pts = _pixels(shape, top, left)
    if pts & occ:
        return None
    if must_cover is not None and not (must_cover <= pts):
        return None
    return pts


def _aligned_positions(axis, A, cover, shape):
    if axis == "cols":
        left = A
        tops = set()
        for r_need, c_need in cover:
            for dr, dc in shape["rel"]:
                if c_need == left + dc:
                    tops.add(r_need - dr)
        for top in tops:
            yield top, left
    else:
        top = A
        lefts = set()
        for r_need, c_need in cover:
            for dr, dc in shape["rel"]:
                if r_need == top + dr:
                    lefts.add(c_need - dc)
        for left in lefts:
            yield top, left


def _touches(a, b):
    return any((r + dr, c + dc) in b for r, c in a for dr, dc in DIR4)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered sprites into the central bar.

    Combo rule:
    - every bar notch has a colored rim on the bar itself
    - root selection is first constrained by color equality between piece and
      notch rim color
    - geometry then verifies the placement and determines the chain strip
    - later pieces follow the original cavity-filling rule
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    base_bar, bar_occ, bar_colors, bar_bbox, piece_comps = _collect_bar_and_pieces(g, bg)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar_bbox)
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    base_bar_color = int(base_bar["color"])

    out = np.full((H, W), int(bg), dtype=int)
    _restore_bar(out, bar_colors)

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True

    pieces = []
    for comp in piece_comps:
        if int(comp["area"]) < 8:
            continue
        r0, c0 = map(int, comp["bbox"][:2])
        pieces.append({"color": int(comp["color"]), "shape": _shape_from_coords(comp["coords"], comp["bbox"]), "orig": (r0, c0)})

    notches = _bar_notches(bar_mask, bar_r0, bar_c0, bar_occ, bar_colors, base_bar_color)
    piece_colors = {p["color"] for p in pieces}
    required_notches = [idx for idx, notch in enumerate(notches) if notch["hint_color"] in piece_colors]
    required_root_count = len(required_notches)

    root_cands = []
    for ni in required_notches:
        notch = notches[ni]
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            if piece["color"] != notch["hint_color"]:
                continue
            shape = piece["shape"]
            for top, left in _hole_positions(notch["hole"], shape):
                pts = _placement_pts(shape, top, left, bar_occ, H, W, must_cover=notch["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + shape["w"] - 1)
                else:
                    A, B = int(top), int(top + shape["h"] - 1)
                move_cost = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
                    (
                        (move_cost, -int(notch["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": shape},
                    )
                )

    root_cands.sort(key=lambda item: item[0])
    cand_roots = [item[1] for item in root_cands]
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
            head_pts = _pixels(head["shape"], head["top"], head["left"])
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
                shape = pieces[pi]["shape"]
                for top, left in _aligned_positions(axis, A, cavity, shape):
                    pts = _placement_pts(shape, top, left, occ_set, H, W, must_cover=cavity, strip=(axis, A, B))
                    if pts is None or not _touches(pts, head_pts):
                        continue
                    moves.append((pi, top, left, pts, shape))
            return moves

        for chain in chains:
            moves = gen_extend_moves(chain, remaining)
            if not moves:
                continue
            for pi, top, left, pts, shape in moves[:8]:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = chain["head"]
                chain["head"] = {"top": int(top), "left": int(left), "shape": shape}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                chain["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                _restore_bar(out_grid, bar_colors)
        return False

    if required_root_count == 0:
        return best_out

    for roots_sel in combinations(range(len(cand_roots)), required_root_count):
        used_p = set()
        used_n = set()
        occ0 = set(bar_occ)
        out0 = np.full((H, W), int(bg), dtype=int)
        _restore_bar(out0, bar_colors)
        chains = []
        ok = True
        for idx in roots_sel:
            root = cand_roots[idx]
            if root["pi"] in used_p or root["ni"] in used_n:
                ok = False
                break
            pts = _pixels(root["shape"], root["top"], root["left"])
            if pts & occ0:
                ok = False
                break
            used_p.add(root["pi"])
            used_n.add(root["ni"])
            for r, c in pts:
                out0[r, c] = int(pieces[root["pi"]]["color"])
            occ0 |= pts
            chains.append(
                {
                    "axis": root["axis"],
                    "A": root["A"],
                    "B": root["B"],
                    "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                }
            )
        if not ok or used_n != set(required_notches):
            continue
        remaining = set(range(len(pieces))) - used_p
        if search(out0, occ0, chains, remaining):
            return best_out

    return best_out


def _hint_ring(hole, bar_cells):
    return {
        (r + dr, c + dc)
        for r, c in hole
        for dr, dc in DIR4
        if (r + dr, c + dc) in bar_cells
    }


def _scatter_piece(input_grid, placed_occ, piece, color, orientation, bar_r0, bar_r1, bar_c0, bar_c1, H, W, rng):
    coords = piece["coords"]
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    p_r0, p_c0 = min(rs), min(cs)
    rel = [(r - p_r0, c - p_c0) for r, c in coords]
    p_h = max(rs) - p_r0 + 1
    p_w = max(cs) - p_c0 + 1

    for _ in range(500):
        gap = rng.randint(3, 8)
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
        if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in DIR4):
            continue
        for r, c in pts:
            input_grid[r, c] = int(color)
        placed_occ |= pts
        return True
    return False


def _gen_attempt():
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

    num_chains = 2 if rng.random() < 0.7 else 1
    if num_chains == 1:
        chain_lens = [3]
    else:
        chain_lens = [rng.choice([2, 3]), rng.choice([2, 3])]
        while not (4 <= sum(chain_lens) <= 5):
            chain_lens = [rng.choice([2, 3]), rng.choice([2, 3])]

    all_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(all_colors)
    color_idx = 0

    all_chain_pieces = []
    used_notches = []
    all_notch_holes = []
    occ_set = set(bar_cells)

    for chain_len in chain_lens:
        strip_width = rng.randint(4, 5)
        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "top" else space_after) >= chain_len * 2 + 3), None)
            if side is None:
                return None
            sA = rng.randint(1, W - strip_width - 1)
            sB = sA + strip_width - 1
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "left" else space_after) >= chain_len * 2 + 3), None)
            if side is None:
                return None
            sA = rng.randint(1, H - strip_width - 1)
            sB = sA + strip_width - 1

        notch_depth = rng.randint(1, 2)
        notch_area = rng.randint(1, min(4, notch_depth * strip_width))

        if orientation == "horizontal":
            nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
            nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
            notch_avail = {(r, c) for r in range(nr0, nr1 + 1) for c in range(sA, sB + 1) if (r, c) in bar_cells}
        else:
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            notch_avail = {(r, c) for r in range(sA, sB + 1) for c in range(nc0, nc1 + 1) if (r, c) in bar_cells}

        all_holes = set().union(*all_notch_holes) if all_notch_holes else set()
        notch_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
        if len(notch_avail) < notch_area:
            return None

        notch = _grow_connected(rng, notch_avail, notch_area, {rng.choice(list(notch_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        all_notch_holes.append(notch)

        chain_pieces = []
        for piece_idx in range(chain_len):
            if color_idx >= len(all_colors):
                return None
            piece_color = all_colors[color_idx]
            color_idx += 1
            piece_area = rng.randint(8, 13)
            chain_dim = rng.randint(2, 3)

            if piece_idx == 0:
                if orientation == "horizontal":
                    if side == "top":
                        pr0, pr1 = max(0, bar_r0 - chain_dim), max(r for r, _ in notch)
                    else:
                        pr0, pr1 = min(r for r, _ in notch), min(H - 1, bar_r1 + chain_dim)
                    pc0, pc1 = sA, sB
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
                prev_rs = [r for r, _ in prev_coords]
                prev_cs = [c for _, c in prev_coords]
                prev_r0, prev_r1 = min(prev_rs), max(prev_rs)
                prev_c0, prev_c1 = min(prev_cs), max(prev_cs)

                if orientation == "horizontal":
                    cavity = {
                        (r, c)
                        for r in range(prev_r0, prev_r1 + 1)
                        for c in range(sA, sB + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                    ext_span = (
                        (max(0, prev_r0 - chain_dim), prev_r0 - 1)
                        if side == "top"
                        else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    )
                    extension = (
                        {(r, c) for r in range(ext_span[0], ext_span[1] + 1) for c in range(sA, sB + 1) if (r, c) not in occ_set}
                        if ext_span[0] <= ext_span[1]
                        else set()
                    )
                else:
                    cavity = {
                        (r, c)
                        for r in range(sA, sB + 1)
                        for c in range(prev_c0, prev_c1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                    ext_span = (
                        (max(0, prev_c0 - chain_dim), prev_c0 - 1)
                        if side == "left"
                        else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    )
                    extension = (
                        {(r, c) for r in range(sA, sB + 1) for c in range(ext_span[0], ext_span[1] + 1) if (r, c) not in occ_set}
                        if ext_span[0] <= ext_span[1]
                        else set()
                    )

                if not cavity:
                    return None
                all_avail = cavity | extension
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
                    tail_r0, tail_r1 = min(prs2), max(prs2)
                    tail_c0, tail_c1 = min(pcs2), max(pcs2)
                    if orientation == "horizontal":
                        remaining = {
                            (r, c)
                            for r in range(tail_r0, tail_r1 + 1)
                            for c in range(sA, sB + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    else:
                        remaining = {
                            (r, c)
                            for r in range(sA, sB + 1)
                            for c in range(tail_c0, tail_c1 + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    if remaining:
                        return None

            if not (8 <= len(piece_cells) <= 13):
                return None
            chain_pieces.append({"color": piece_color, "coords": piece_cells})
            occ_set |= piece_cells

        all_chain_pieces.extend(chain_pieces)
        used_notches.append({"hole": notch, "root_color": int(chain_pieces[0]["color"])})

    if not (3 <= len(all_chain_pieces) <= 6):
        return None

    piece_colors = {piece["color"] for piece in all_chain_pieces}
    unique_notch_colors = [c for c in range(1, 10) if c != bar_color and c not in piece_colors]
    if len(unique_notch_colors) < 3:
        return None

    max_total_notches = min(6, len(unique_notch_colors) + len(used_notches))
    min_total_notches = max(4, len(used_notches) + 3)
    if min_total_notches > max_total_notches:
        return None
    total_notches = rng.randint(min_total_notches, max_total_notches)
    num_decoys = total_notches - len(used_notches)

    decoy_notches = []
    for _ in range(600):
        if len(decoy_notches) >= num_decoys:
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

        avail = {(r, c) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        all_holes = set().union(*all_notch_holes) if all_notch_holes else set()
        avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
        if not avail:
            continue
        area = min(area, len(avail))
        notch = _grow_connected(rng, avail, area, {rng.choice(list(avail))})
        if notch is None or not _connected(bar_cells - notch):
            continue
        bar_cells -= notch
        all_notch_holes.append(notch)
        decoy_notches.append({"hole": notch})
    if len(decoy_notches) != num_decoys:
        return None

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr = pts[:, 0]
        cc = pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return None

    all_notches = []
    hint_occ = set()
    for meta in used_notches:
        ring = _hint_ring(meta["hole"], bar_cells)
        if not ring or ring & hint_occ:
            return None
        hint_occ |= ring
        all_notches.append({"hole": meta["hole"], "hint_cells": ring, "hint_color": meta["root_color"]})
    for meta, decoy_color in zip(decoy_notches, unique_notch_colors[:num_decoys]):
        ring = _hint_ring(meta["hole"], bar_cells)
        if not ring or ring & hint_occ:
            return None
        hint_occ |= ring
        all_notches.append({"hole": meta["hole"], "hint_cells": ring, "hint_color": int(decoy_color)})

    if len({meta["hint_color"] for meta in all_notches}) != len(all_notches):
        return None

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = int(bar_color)
    for meta in all_notches:
        for r, c in meta["hint_cells"]:
            output[r, c] = int(meta["hint_color"])
    for piece in all_chain_pieces:
        for r, c in piece["coords"]:
            output[r, c] = int(piece["color"])

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = int(bar_color)
    for meta in all_notches:
        for r, c in meta["hint_cells"]:
            input_grid[r, c] = int(meta["hint_color"])

    placed_occ = set(bar_cells) | set(hint_occ)
    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)
    for piece in pieces_order:
        if not _scatter_piece(input_grid, placed_occ, piece, piece["color"], orientation, bar_r0, bar_r1, bar_c0, bar_c1, H, W, rng):
            return None

    return input_grid, output


def _analyze_fusion_signal(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether color constraints are doing real work on an input sample.

    Metrics:
      notches_total: total number of colored/uncolored bar notches
      matched_notches: notches whose rim color appears on a scattered piece
      geometric_pairs: notch-piece pairs with at least one legal root placement
      color_pairs: subset of geometric_pairs that also satisfy color equality
      wrong_color_geometric_pairs: geometric root matches that color would prune
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)
    base_bar, bar_occ, bar_colors, bar_bbox, piece_comps = _collect_bar_and_pieces(g, bg)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar_bbox)
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    base_bar_color = int(base_bar["color"])

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
    notches = _bar_notches(bar_mask, bar_r0, bar_c0, bar_occ, bar_colors, base_bar_color)

    pieces = []
    for comp in piece_comps:
        if int(comp["area"]) >= 8:
            pieces.append({"color": int(comp["color"]), "shape": _shape_from_coords(comp["coords"], comp["bbox"])})

    piece_colors = {p["color"] for p in pieces}
    geometric_pairs = 0
    color_pairs = 0
    for notch in notches:
        for piece in pieces:
            found = False
            for top, left in _hole_positions(notch["hole"], piece["shape"]):
                pts = _placement_pts(piece["shape"], top, left, bar_occ, H, W, must_cover=notch["hole"])
                if pts is not None:
                    found = True
                    break
            if found:
                geometric_pairs += 1
                if piece["color"] == notch["hint_color"]:
                    color_pairs += 1

    return {
        "notches_total": len(notches),
        "matched_notches": sum(1 for notch in notches if notch["hint_color"] in piece_colors),
        "geometric_pairs": geometric_pairs,
        "color_pairs": color_pairs,
        "wrong_color_geometric_pairs": geometric_pairs - color_pairs,
    }


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        metrics = _analyze_fusion_signal(inp)
        if metrics["wrong_color_geometric_pairs"] <= 0:
            continue
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 500 attempts")
