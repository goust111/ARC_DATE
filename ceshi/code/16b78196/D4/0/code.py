# Variant of 16b78196
# Dimension: D4 — 链式拼接的多步处理逻辑 (multi_step)
# Change: 1-2条独立链从不同缺口向外生长，每链2-3个精灵，后续精灵填充前一精灵bbox中的空腔并继续延伸... → 双向生长链

# Design analysis for the variant
# =================================
# a) Concrete behavior and flow change
#    Original task:
#    - A used opening is a one-sided notch on the bar boundary.
#    - Root piece plugs that notch and extends only to that same exterior side.
#    - Later pieces keep filling the previous bbox cavity and keep moving farther
#      away from the bar on that same side.
#
#    Variant here:
#    - A used opening is no longer a one-sided notch. It is a connected through-
#      channel hole that touches both opposite sides of the bar bbox:
#        horizontal bar -> the hole touches both the top row and bottom row
#        vertical bar   -> the hole touches both the left col and right col
#    - The used hole is a full through-channel, but the root piece does not fill
#      the whole channel. Instead it plugs one mouth of the channel, reaches into
#      the corridor, and already touches the far mouth while leaving part of the
#      corridor black.
#    - The next piece fills that remaining corridor cavity and continues out
#      through the opposite side of the bar, so the chain crosses the bar.
#    - If there is a third piece, it fills the second piece's cavity and extends
#      farther on the opposite exterior side.
#
#    Small horizontal example:
#    - Original:
#      root = covers top notch + a few cells above the bar
#      piece2 = fills root cavity + goes farther upward
#    - Variant:
#      root = covers the top mouth + a sparse path through the channel + one row above
#      piece2 = fills the root cavity + emerges below the bar
#      piece3 = fills piece2 cavity + goes farther below
#    - Difference:
#      the cavity-filling rule stays, but the chain direction flips after the
#      root because the root already spans the bar channel.
#
# b) Complexity calibration
#    - The visible change must be obvious, so every used chain in the generator
#      uses a through-channel. There is no mixture of old one-sided used notches.
#    - To keep the puzzle learnable, the strip width for used chains is fixed to
#      5. This makes the corridor visually stable across samples and reduces
#      solver ambiguity.
#    - Decoy notches remain one-sided and smaller. They preserve ambiguity, but
#      solver logic can distinguish them from the used through-channels.
#    - Chain count and multi-step depth remain close to the original task:
#      1 chain of length 3, or 2 chains of length 2-3, total 3-6 pieces.
#
# c) Implicit-assumption audit for main()
#    Relevant original assumptions and what changes here:
#    - `bar_notches(...)` assumed a used opening touched one boundary side only.
#      This is no longer valid. The new solver explicitly classifies holes that
#      touch both opposite sides as through-channels and prefers them.
#    - Original root candidates let the root piece define strip width from its
#      own bbox. In the variant, the through-hole itself already defines the
#      corridor, so the solver locks the strip directly from the hole bbox.
#    - Original search was accidentally direction-agnostic after the root: it
#      only filled the current bbox cavity. That still holds and is reused.
#    - Original root placements only needed to cover a one-sided notch. Here the
#      root must also cover the full through-hole and have at least one cell
#      outside the bar bbox, otherwise it would be a purely internal filler.
#    - Undo logic in search restores only solid bar pixels (`bar_occ`), not hole
#      cells. That remains correct: used channels are bar holes, so undoing a
#      piece must leave them black, not repaint them as bar.
#
# d) Placement legality
#    - Through-channels are carved before placing pieces in output space.
#    - Every piece uses the fixed strip width of its chain and never overlaps the
#      remaining bar component or another chain.
#    - While scattering the pieces for input, shapes are kept in original
#      orientation, placed away from the bar, and forbidden from touching other
#      objects orthogonally.
#    - Generator validates every sample with `main(input) == output`.
#
# e) Visual observability
#    - The variant is human-visible because the bar contains one or two full
#      channels that open on both sides, and the completed output shows chains
#      crossing the bar instead of stopping on one side.
#    - With 2-4 training examples, a solver can see that only the full
#      through-channels are used; one-sided decoy notches are left empty.
#
# f) Visual cleanliness
#    - Used channels are separated by at least one cell buffer, so different
#      chains do not visually merge.
#    - Decoy notches also respect the hole buffer, preventing accidental
#      multi-sided mega-holes that would muddy the rule.
#    - The bar remains a single large monochrome component after carving holes.
#
# g) Color inference
#    - Colors do not encode ordering or direction. They remain arbitrary per
#      sprite, as in the original task.
#    - The rule is inferred from geometry: same-width strips, cavity-filling, and
#      the new fact that used strips pass all the way through the bar.
#
# New solver assumptions introduced by the variant
# ------------------------------------------------
# - The bar may be split into multiple same-color components by used through-
#   channels. Solver therefore identifies the bar by dominant total color area,
#   not by a single largest connected component.
# - If at least one through-channel hole exists, those are the only used roots.
#   Generator guarantees this.
# - Every used chain keeps a constant strip width equal to the through-hole bbox
#   span. Generator guarantees this by fixing the used strip width.
# - The root piece covers one mouth of the through-hole, reaches the opposite
#   mouth, and has cells outside the bar, but it leaves some of the channel
#   unfilled for the next piece. Generator guarantees that too.

from __future__ import annotations

import numpy as np

from arc_utils import components_mask, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps

    area_by_color = {}
    for comp in comps:
        color = int(comp["color"])
        area_by_color[color] = area_by_color.get(color, 0) + int(comp["area"])
    bar_color = max(area_by_color.items(), key=lambda kv: kv[1])[0]
    bar_parts = [comp for comp in comps if int(comp["color"]) == bar_color]
    pieces_raw = [comp for comp in comps if int(comp["color"]) != bar_color]

    out = np.full((H, W), int(bg), dtype=int)
    bar_occ = {(int(r), int(c)) for part in bar_parts for r, c in part["coords"]}
    bar_rs = [r for r, _ in bar_occ]
    bar_cs = [c for _, c in bar_occ]
    bar_r0, bar_c0, bar_r1, bar_c1 = min(bar_rs), min(bar_cs), max(bar_rs), max(bar_cs)
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    def shape_from_coords(coords: np.ndarray, bbox):
        r0, c0, r1, c1 = map(int, bbox)
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        return {"rel": rel, "h": r1 - r0 + 1, "w": c1 - c0 + 1}

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    def pixels(v, top, left):
        return {(top + lr, left + lc) for lr, lc in v["rel"]}

    def in_bar_bbox(rc):
        r, c = rc
        return bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1

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
            for rN, cN in cover:
                for lr, lc in v["rel"]:
                    if cN == left + lc:
                        tops.add(rN - lr)
            for top in tops:
                yield top, left
        else:
            top = A
            lefts = set()
            for rN, cN in cover:
                for lr, lc in v["rel"]:
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

    def hole_positions(hole, v):
        pos = set()
        for rH, cH in hole:
            for lr, lc in v["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    def bar_openings(mask, top0, left0):
        empty = ~mask
        h, w = empty.shape
        openings = []
        for pts in components_mask(empty, connectivity=4):
            r = pts[:, 0]
            c = pts[:, 1]
            top = int((r == 0).sum())
            bottom = int((r == h - 1).sum())
            left = int((c == 0).sum())
            right = int((c == w - 1).sum())
            if top + bottom + left + right == 0:
                continue
            hole_gl = {(top0 + int(rr), left0 + int(cc)) for rr, cc in pts}
            through_tb = top > 0 and bottom > 0 and left == 0 and right == 0
            through_lr = left > 0 and right > 0 and top == 0 and bottom == 0
            if through_tb:
                axis = "cols"
                A, B = min(c) + left0, max(c) + left0
                mouths = {
                    "low": {(top0 + int(rr), left0 + int(cc)) for rr, cc in pts if int(rr) == 0},
                    "high": {(top0 + int(rr), left0 + int(cc)) for rr, cc in pts if int(rr) == h - 1},
                }
            elif through_lr:
                axis = "rows"
                A, B = min(r) + top0, max(r) + top0
                mouths = {
                    "low": {(top0 + int(rr), left0 + int(cc)) for rr, cc in pts if int(cc) == 0},
                    "high": {(top0 + int(rr), left0 + int(cc)) for rr, cc in pts if int(cc) == w - 1},
                }
            else:
                side, mouth_n = max(
                    [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                    key=lambda x: x[1],
                )
                axis = "cols" if side in ("top", "bottom") else "rows"
                A = B = None
                openings.append(
                    {
                        "axis": axis,
                        "hole": hole_gl,
                        "mouth_n": mouth_n,
                        "through": False,
                        "A": A,
                        "B": B,
                    }
                )
                continue
            openings.append(
                    {
                        "axis": axis,
                        "hole": hole_gl,
                        "mouth_n": len(hole_gl),
                        "through": True,
                        "A": int(A),
                        "B": int(B),
                        "mouths": mouths,
                    }
                )
        return openings

    openings = bar_openings(bar_mask, bar_r0, bar_c0)
    through_openings = [o for o in openings if o["through"]]
    notches = through_openings if through_openings else openings

    root_cands = []
    for ni, n in enumerate(notches):
        axis = n["axis"]
        for pi, p in enumerate(pieces):
            v = p["shape"]
            if n["through"]:
                A, B = int(n["A"]), int(n["B"])
                if axis == "cols" and v["w"] != B - A + 1:
                    continue
                if axis == "rows" and v["h"] != B - A + 1:
                    continue
                strip = (axis, A, B)
                for side, mouth in n["mouths"].items():
                    far_mouth = n["mouths"]["high" if side == "low" else "low"]
                    pos_iter = aligned_positions(axis, A, mouth, v)
                    for top, left in pos_iter:
                        pts = placement_pts(v, top, left, bar_occ, must_cover=mouth, strip=strip)
                        if pts is None:
                            continue
                        if not (pts & far_mouth):
                            continue
                        if not any(not in_bar_bbox(rc) for rc in pts):
                            continue
                        mv = abs(int(top) - int(p["orig"][0])) + abs(int(left) - int(p["orig"][1]))
                        root_cands.append(
                            (
                                (0, int(mv), -int(len(mouth))),
                                {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": v},
                            )
                        )
                continue
            else:
                pos_iter = hole_positions(n["hole"], v)
                strip = None
            for top, left in pos_iter:
                pts = placement_pts(v, top, left, bar_occ, must_cover=n["hole"], strip=strip)
                if pts is None:
                    continue
                if not any(not in_bar_bbox(rc) for rc in pts):
                    continue
                if n["through"]:
                    A, B = int(n["A"]), int(n["B"])
                elif axis == "cols":
                    A, B = int(left), int(left + v["w"] - 1)
                else:
                    A, B = int(top), int(top + v["h"] - 1)
                mv = abs(int(top) - int(p["orig"][0])) + abs(int(left) - int(p["orig"][1]))
                root_cands.append(
                    (
                        (0 if n["through"] else 1, int(mv), -int(n["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": v},
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands]

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
            for pi, top, left, pts, v in moves[:10]:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
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

    max_roots = min(2, len(cand_roots))
    for k in range(1, max_roots + 1):
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
            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


def _connected(cells):
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
                    cm = min((c for _, c in cells), default=B)
                    cx = max((c for _, c in cells), default=A)
                    if cm > A:
                        pref = [nb for nb in frontier if nb[1] < cm]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                    if cx < B:
                        pref = [nb for nb in frontier if nb[1] > cx]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                else:
                    cm = min((r for r, _ in cells), default=B)
                    cx = max((r for r, _ in cells), default=A)
                    if cm > A:
                        pref = [nb for nb in frontier if nb[0] < cm]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                    if cx < B:
                        pref = [nb for nb in frontier if nb[0] > cx]
                        if pref:
                            cells.add(rng.choice(pref))
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


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _bbox_cells(cells):
    r0, c0, r1, c1 = _bbox(cells)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _strip_rect(orientation, A, B, g0, g1):
    if g0 > g1:
        return set()
    if orientation == "horizontal":
        return {(r, c) for r in range(g0, g1 + 1) for c in range(A, B + 1)}
    return {(r, c) for r in range(A, B + 1) for c in range(g0, g1 + 1)}


def _growth_bounds(cells, orientation):
    r0, c0, r1, c1 = _bbox(cells)
    return (r0, r1) if orientation == "horizontal" else (c0, c1)


def _outside_on_side(cell, orientation, bar_low, bar_high, side):
    g = cell[0] if orientation == "horizontal" else cell[1]
    return g < bar_low if side == "low" else g > bar_high


def _through_hole(rng, orientation, bar_cells, hole_sets, bar_r0, bar_c0, bar_r1, bar_c1, A, B):
    span_axis = "cols" if orientation == "horizontal" else "rows"
    thickness = (bar_r1 - bar_r0 + 1) if orientation == "horizontal" else (bar_c1 - bar_c0 + 1)
    target_lo = thickness + (B - A + 1) - 1
    target_hi = min(11, len(_strip_rect(orientation, A, B, bar_r0 if orientation == "horizontal" else bar_c0, bar_r1 if orientation == "horizontal" else bar_c1)))
    if target_lo > target_hi:
        return None
    all_holes = set().union(*hole_sets) if hole_sets else set()
    blocked = set(all_holes)
    blocked |= {(r + dr, c + dc) for r, c in all_holes for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
    if orientation == "horizontal":
        avail = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(A, B + 1) if (r, c) in bar_cells and (r, c) not in blocked}
        top_choices = [(bar_r0, c) for c in range(A, B + 1) if (bar_r0, c) in avail]
        bot_choices = [(bar_r1, c) for c in range(A, B + 1) if (bar_r1, c) in avail]
    else:
        avail = {(r, c) for r in range(A, B + 1) for c in range(bar_c0, bar_c1 + 1) if (r, c) in bar_cells and (r, c) not in blocked}
        top_choices = [(r, bar_c0) for r in range(A, B + 1) if (r, bar_c0) in avail]
        bot_choices = [(r, bar_c1) for r in range(A, B + 1) if (r, bar_c1) in avail]
    if not top_choices or not bot_choices:
        return None
    for _ in range(20):
        seeds = {rng.choice(top_choices), rng.choice(bot_choices)}
        target = rng.randint(target_lo, target_hi)
        hole = _grow_connected(rng, avail, target, seeds, must_span_axis=span_axis, span_range=(A, B), retries=10)
        if hole is None:
            continue
        if orientation == "horizontal":
            if not any(r == bar_r0 for r, _ in hole) or not any(r == bar_r1 for r, _ in hole):
                continue
        else:
            if not any(c == bar_c0 for _, c in hole) or not any(c == bar_c1 for _, c in hole):
                continue
        return hole
    return None


def _build_chain(
    rng,
    orientation,
    bar_r0,
    bar_c0,
    bar_r1,
    bar_c1,
    A,
    B,
    start_side,
    chain_len,
    hole,
    occ_set,
):
    span_axis = "cols" if orientation == "horizontal" else "rows"
    strip_w = B - A + 1
    bar_low = bar_r0 if orientation == "horizontal" else bar_c0
    bar_high = bar_r1 if orientation == "horizontal" else bar_c1
    line_high = 29 if orientation == "horizontal" else 29
    travel_side = "high" if start_side == "low" else "low"
    mouth = {rc for rc in hole if _outside_on_side(rc, orientation, bar_low + 1, bar_high - 1, start_side)}
    far_mouth = {rc for rc in hole if _outside_on_side(rc, orientation, bar_low + 1, bar_high - 1, travel_side)}
    if not mouth or not far_mouth:
        return None

    start_ext = _strip_rect(
        orientation,
        A,
        B,
        bar_low - 1 if start_side == "low" else bar_high + 1,
        bar_low - 1 if start_side == "low" else bar_high + 1,
    )
    start_ext = {rc for rc in start_ext if 0 <= rc[0] <= 29 and 0 <= rc[1] <= 29 and rc not in occ_set}
    avail_root = set(hole) | start_ext
    if len(start_ext) != strip_w:
        return None
    seed_base = set(mouth) | {rng.choice(list(far_mouth))}
    root_targets = []
    for t in range(8, min(13, len(avail_root) - 1) + 1):
        if t > len(seed_base):
            root_targets.append(t)
    rng.shuffle(root_targets)
    pieces = []
    occ_now = set(occ_set)
    root_piece = None
    for target in root_targets:
        root_piece = _grow_connected(rng, avail_root, target, seed_base, must_span_axis=span_axis, span_range=(A, B), retries=12)
        if root_piece is None:
            continue
        if not any(_outside_on_side(rc, orientation, bar_low, bar_high, start_side) for rc in root_piece):
            continue
        if not mouth.issubset(root_piece):
            continue
        if not (set(root_piece) & far_mouth):
            continue
        occ_test = occ_now | root_piece
        cavity = _bbox_cells(root_piece) - occ_test
        if not cavity or not _connected(cavity):
            continue
        root_piece = set(root_piece)
        break
    if root_piece is None:
        return None

    pieces.append(root_piece)
    occ_now |= root_piece

    while len(pieces) < chain_len:
        prev = pieces[-1]
        cavity = _bbox_cells(prev) - occ_now
        if not cavity:
            return None
        is_tail = len(pieces) == chain_len - 1
        ext_len = 1 if is_tail else 2
        prev_low, prev_high = _growth_bounds(prev, orientation)
        if travel_side == "low":
            ext0, ext1 = max(0, prev_low - ext_len), prev_low - 1
        else:
            ext0, ext1 = prev_high + 1, min(line_high, prev_high + ext_len)
        ext = _strip_rect(orientation, A, B, ext0, ext1)
        ext = {rc for rc in ext if 0 <= rc[0] <= 29 and 0 <= rc[1] <= 29 and rc not in occ_now}
        all_avail = set(cavity) | ext
        if is_tail:
            if not (8 <= len(all_avail) <= 13):
                return None
            target = len(all_avail)
            target_opts = [target]
        else:
            target_opts = []
            for desired_cavity in (4, 3):
                target = len(all_avail) - desired_cavity
                if 8 <= target <= 13 and target > len(cavity):
                    target_opts.append(target)
            rng.shuffle(target_opts)
        if not target_opts:
            return None

        new_piece = None
        for target in target_opts:
            cand = _grow_connected(rng, all_avail, target, cavity, must_span_axis=span_axis, span_range=(A, B), retries=12)
            if cand is None or not cavity.issubset(cand):
                continue
            occ_test = occ_now | set(cand)
            rem = _bbox_cells(cand) - occ_test
            if is_tail:
                if rem:
                    continue
            else:
                if not (3 <= len(rem) <= 6):
                    continue
            new_piece = set(cand)
            break
        if new_piece is None:
            return None
        pieces.append(new_piece)
        occ_now |= new_piece

    return pieces


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

    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    chain_lens = [3]

    strip_width = 5
    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    color_idx = 0

    notch_holes: list[set] = []
    used_meta = []
    occ_set = set(bar_cells)

    for chain_len in chain_lens:
        built = None
        for _ in range(80):
            if orientation == "horizontal":
                if max(space_before, space_after) < 1 or min(space_before, space_after) < 3:
                    return None
                sc0 = rng.randint(1, W - strip_width - 1)
                sc1 = sc0 + strip_width - 1
                A, B = sc0, sc1
                side_opts = []
                if space_before >= 1 and space_after >= 3:
                    side_opts.append("low")
                if space_after >= 1 and space_before >= 3:
                    side_opts.append("high")
            else:
                if max(space_before, space_after) < 1 or min(space_before, space_after) < 3:
                    return None
                sr0 = rng.randint(1, H - strip_width - 1)
                sr1 = sr0 + strip_width - 1
                A, B = sr0, sr1
                side_opts = []
                if space_before >= 1 and space_after >= 3:
                    side_opts.append("low")
                if space_after >= 1 and space_before >= 3:
                    side_opts.append("high")
            if not side_opts:
                return None

            hole = _through_hole(rng, orientation, bar_cells, notch_holes, bar_r0, bar_c0, bar_r1, bar_c1, A, B)
            if hole is None:
                continue

            for _ in range(30):
                start_side = rng.choice(side_opts)
                chain_pieces = _build_chain(
                    rng,
                    orientation,
                    bar_r0,
                    bar_c0,
                    bar_r1,
                    bar_c1,
                    A,
                    B,
                    start_side,
                    chain_len,
                    hole,
                    occ_set,
                )
                if chain_pieces is not None:
                    built = (A, B, hole, chain_pieces)
                    break
            if built is not None:
                break
        if built is None:
            return None

        A, B, hole, chain_pieces = built
        bar_cells -= hole
        notch_holes.append(hole)

        colored_chain = []
        for coords in chain_pieces:
            if color_idx >= len(avail_colors):
                return None
            piece_color = avail_colors[color_idx]
            color_idx += 1
            if not (8 <= len(coords) <= 13):
                return None
            colored_chain.append({"color": piece_color, "coords": coords})
            occ_set |= coords
        used_meta.append({"A": A, "B": B, "pieces": colored_chain})

    all_chain_pieces = [p for ch in used_meta for p in ch["pieces"]]
    if not (3 <= len(all_chain_pieces) <= 6):
        return None

    num_decoys = rng.randint(4, 7)
    decoys = 0
    for _ in range(600):
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
        all_holes = set().union(*notch_holes) if notch_holes else set()
        d_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
        if len(d_avail) < 1:
            continue
        area = min(area, len(d_avail))
        dn = _grow_connected(rng, d_avail, area, {rng.choice(list(d_avail))})
        if dn is None:
            continue
        bar_cells -= dn
        notch_holes.append(dn)
        decoys += 1
    if decoys < 4:
        return None

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
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
            output[r, c] = p["color"]

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
        for _ in range(700):
            gap = rng.randint(2, 7)
            if orientation == "horizontal":
                top = (bar_r0 - gap - p_h) if rng.random() < 0.5 else (bar_r1 + gap + 1)
                left = rng.randint(0, max(0, W - p_w))
            else:
                left = (bar_c0 - gap - p_w) if rng.random() < 0.5 else (bar_c1 + gap + 1)
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
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
