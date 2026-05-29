# Variant of 16b78196
# Dimension: D1 — 匹配条件：形状嵌合 vs 其他匹配逻辑 (match_cond)
# Change: s.shape fits into d.notch.shape — 精灵形状恰好填充杆上缺口的形状... → 边缘轮廓匹配（拼图齿合）

from __future__ import annotations

import numpy as np

from arc_utils import components_mask, find_components, get_bg


# Design analysis for this variant
# --------------------------------
# spec_variant:
#   core_change:
#     old: "root piece must cover the whole used notch component"
#     new: "root piece only has to match the notch's boundary-facing edge profile"
#   concrete_pixel_behavior:
#     - used notches are always 2 cells deep
#     - the boundary layer of the notch (the mouth cells) has a gapped profile
#     - the root piece occupies exactly that mouth layer inside the bar, but leaves the
#       deeper notch cells empty
#     - visually, the root looks like a jigsaw tab that catches the notch lip instead of
#       completely filling the recess
#     - subsequent chain growth remains the original mechanism: later pieces fill the bg
#       cavity inside the previous piece's bbox and extend farther outward in the same strip
#   example_trace:
#     - original root check: piece_cells_inside_bar == whole_notch
#     - new root check: piece_cells_inside_bar == notch_mouth_only, and the piece's
#       attaching edge occupancy profile equals the notch mouth profile
#     - difference: a used notch keeps deeper bg cells in the output, so exact hole-cover
#       logic from the original solver no longer works
#   complexity_calibration:
#     - mouth profiles are width 4-5 with one internal gap, so the edge signal is visible
#     - decoy notches use contiguous mouths, preventing accidental equivalence
#     - only the root matching rule changes; chain count, strip locking, and continuation
#       mechanics stay close to the original task
#   implicit_assumption_audit:
#     - original `placement_pts(... must_cover=n["hole"])` assumed the whole notch must be
#       occupied; this is removed for roots
#     - original root candidate generation relied on arbitrary translations that cover the
#       hole; here root placement is fixed by bar side + strip span + edge profile
#     - continuation search still assumes cavity fill inside previous bbox; the generator
#       preserves that by constructing root bboxes with remaining bg cavities
#   legality:
#     - root pieces are allowed to enter the bar only on mouth cells of their matched notch
#     - no piece may overlap solid bar pixels, cross the grid border, or touch another
#       scattered piece in the input
#     - used notch deeper cells are intentionally left empty; this is a required feature,
#       not generator noise
#   observability:
#     - a human can see that the first piece hooks onto the notch lip while the notch still
#       has deeper black cells behind it
#     - this remains learnable from few examples because all used notches share the same
#       pattern family: gapped mouth + deeper residual cavity


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ATTACH_EDGE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
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

    def edge_profile(v, edge):
        if edge == "top":
            return tuple(int((0, c) in v["rel_set"]) for c in range(v["w"]))
        if edge == "bottom":
            rr = v["h"] - 1
            return tuple(int((rr, c) in v["rel_set"]) for c in range(v["w"]))
        if edge == "left":
            return tuple(int((r, 0) in v["rel_set"]) for r in range(v["h"]))
        cc = v["w"] - 1
        return tuple(int((r, cc) in v["rel_set"]) for r in range(v["h"]))

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        shape = shape_from_coords(p["coords"], p["bbox"])
        pieces.append(
            {
                "color": int(p["color"]),
                "shape": shape,
                "orig": (r0, c0),
                "profiles": {side: edge_profile(shape, edge) for side, edge in ATTACH_EDGE.items()},
            }
        )

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

            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            hole_gl = {(bar_r0 + int(rr), bar_c0 + int(cc)) for rr, cc in pts}

            if side == "top":
                A, B = int(c.min()), int(c.max())
                profile = tuple(int((0, cc) in {(int(rr), int(cc)) for rr, cc in pts}) for cc in range(A, B + 1))
                mouth = {(bar_r0, bar_c0 + cc) for cc in range(A, B + 1) if profile[cc - A]}
            elif side == "bottom":
                A, B = int(c.min()), int(c.max())
                profile = tuple(int((h - 1, cc) in {(int(rr), int(cc)) for rr, cc in pts}) for cc in range(A, B + 1))
                mouth = {(bar_r1, bar_c0 + cc) for cc in range(A, B + 1) if profile[cc - A]}
            elif side == "left":
                A, B = int(r.min()), int(r.max())
                profile = tuple(int((rr, 0) in {(int(rr), int(cc)) for rr, cc in pts}) for rr in range(A, B + 1))
                mouth = {(bar_r0 + rr, bar_c0) for rr in range(A, B + 1) if profile[rr - A]}
            else:
                A, B = int(r.min()), int(r.max())
                profile = tuple(int((rr, w - 1) in {(int(rr), int(cc)) for rr, cc in pts}) for rr in range(A, B + 1))
                mouth = {(bar_r0 + rr, bar_c1) for rr in range(A, B + 1) if profile[rr - A]}

            notches.append(
                {
                    "axis": axis,
                    "side": side,
                    "hole": hole_gl,
                    "mouth": mouth,
                    "profile": profile,
                    "A": (bar_c0 + A) if axis == "cols" else (bar_r0 + A),
                    "B": (bar_c0 + B) if axis == "cols" else (bar_r0 + B),
                    "mouth_n": mouth_n,
                }
            )
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

    def root_position(n, v):
        if n["side"] == "top":
            return bar_r0 - v["h"] + 1, int(n["A"])
        if n["side"] == "bottom":
            return bar_r1, int(n["A"])
        if n["side"] == "left":
            return int(n["A"]), bar_c0 - v["w"] + 1
        return int(n["A"]), bar_c1

    root_cands = []
    for ni, n in enumerate(notches):
        span = int(n["B"] - n["A"] + 1)
        for pi, p in enumerate(pieces):
            v = p["shape"]
            if p["profiles"][n["side"]] != n["profile"]:
                continue
            if (n["axis"] == "cols" and v["w"] != span) or (n["axis"] == "rows" and v["h"] != span):
                continue
            top, left = root_position(n, v)
            pts = placement_pts(v, int(top), int(left), bar_occ)
            if pts is None:
                continue
            inside_hole = pts & n["hole"]
            if inside_hole != n["mouth"]:
                continue
            mv = abs(int(top) - int(p["orig"][0])) + abs(int(left) - int(p["orig"][1]))
            root_cands.append(
                (
                    (int(mv), -int(n["mouth_n"])),
                    {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": n["axis"], "A": int(n["A"]), "B": int(n["B"]), "shape": v},
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
                    out0[r, c] = int(pieces[root["pi"]]["color"])
                occ0 |= pts
                chains.append({"axis": root["axis"], "A": root["A"], "B": root["B"], "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]}})

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
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells_set and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(cells_set)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=3):
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 20):
            if len(cells) >= target:
                break
            frontier = [nb for r, c in cells for dr, dc in DIR4 if (nb := (r + dr, c + dc)) in available and nb not in cells]
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


def _used_notch_shape(side, span_start, width, base_r0, base_r1, base_c0, base_c1, gap_idx):
    if side in ("top", "bottom"):
        cols = [span_start + i for i in range(width)]
        mouth_cols = [c for i, c in enumerate(cols) if i != gap_idx]
        deep_l = max(0, gap_idx - 1)
        deep_r = min(width - 1, gap_idx + 1)
        deep_cols = [span_start + i for i in range(deep_l, deep_r + 1)]
        if side == "top":
            hole = {(base_r0, c) for c in mouth_cols} | {(base_r0 + 1, c) for c in deep_cols}
        else:
            hole = {(base_r1, c) for c in mouth_cols} | {(base_r1 - 1, c) for c in deep_cols}
        profile = tuple(1 if i != gap_idx else 0 for i in range(width))
    else:
        rows = [span_start + i for i in range(width)]
        mouth_rows = [r for i, r in enumerate(rows) if i != gap_idx]
        deep_l = max(0, gap_idx - 1)
        deep_r = min(width - 1, gap_idx + 1)
        deep_rows = [span_start + i for i in range(deep_l, deep_r + 1)]
        if side == "left":
            hole = {(r, base_c0) for r in mouth_rows} | {(r, base_c0 + 1) for r in deep_rows}
        else:
            hole = {(r, base_c1) for r in mouth_rows} | {(r, base_c1 - 1) for r in deep_rows}
        profile = tuple(1 if i != gap_idx else 0 for i in range(width))
    return hole, profile


def _profile_signature_for_hole(hole, side):
    if side in ("top", "bottom"):
        cols = [c for _, c in hole]
        A, B = min(cols), max(cols)
        boundary_r = min(r for r, _ in hole) if side == "top" else max(r for r, _ in hole)
        profile = tuple(int((boundary_r, c) in hole) for c in range(A, B + 1))
    else:
        rows = [r for r, _ in hole]
        A, B = min(rows), max(rows)
        boundary_c = min(c for _, c in hole) if side == "left" else max(c for _, c in hole)
        profile = tuple(int((r, boundary_c) in hole) for r in range(A, B + 1))
    return side, profile


def _count_deeper_unused_notch_cells(output):
    g = np.array(output, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    if not comps:
        return 0
    bar = comps[0]
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar["coords"]:
        bar_mask[int(r) - bar_r0, int(c) - bar_c0] = True
    unused = 0
    for pts in components_mask(~bar_mask, connectivity=4):
        r = pts[:, 0]
        c = pts[:, 1]
        if (r == 0).any():
            unused += int((r > 0).sum())
        elif (r == bar_h - 1).any():
            unused += int((r < bar_h - 1).sum())
        elif (c == 0).any():
            unused += int((c > 0).sum())
        elif (c == bar_w - 1).any():
            unused += int((c < bar_w - 1).sum())
    return unused


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

    num_chains = 2
    chain_lens = [2, 2]

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    color_idx = 0

    all_chain_pieces = []
    notch_holes = []
    occ_set = set(bar_cells)
    used_profiles = set()

    for chain_idx in range(num_chains):
        chain_len = chain_lens[chain_idx]
        strip_width = rng.randint(4, 5)

        chosen_meta = None
        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            starts = list(range(1, W - strip_width))
            rng.shuffle(starts)
            gap_choices = [0, strip_width - 1]
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            starts = list(range(1, H - strip_width))
            rng.shuffle(starts)
            gap_choices = [0, strip_width - 1]

        for cand_side in sides:
            if (space_before if cand_side in ("top", "left") else space_after) < chain_len * 2 + 3:
                continue
            for span_start in starts:
                rng.shuffle(gap_choices)
                for gap_idx in gap_choices:
                    hole, profile = _used_notch_shape(cand_side, span_start, strip_width, bar_r0, bar_r1, bar_c0, bar_c1, gap_idx)
                    if (cand_side, profile) in used_profiles:
                        continue
                    all_holes = set().union(*notch_holes) if notch_holes else set()
                    pad = all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
                    if hole & pad:
                        continue
                    if not hole.issubset(bar_cells):
                        continue
                    if not _connected(bar_cells - hole):
                        continue
                    chosen_meta = (cand_side, span_start, span_start + strip_width - 1, hole, profile)
                    break
                if chosen_meta is not None:
                    break
            if chosen_meta is not None:
                break
        if chosen_meta is None:
            return None

        side, sA, sB, notch, root_profile = chosen_meta
        bar_cells -= notch
        notch_holes.append(notch)
        used_profiles.add((side, root_profile))

        chain_pieces = []
        for piece_idx in range(chain_len):
            if color_idx >= len(avail_colors):
                return None
            piece_color = avail_colors[color_idx]
            color_idx += 1
            piece_area = rng.randint(8, 13)
            chain_dim = 2

            if piece_idx == 0:
                if orientation == "horizontal":
                    if side == "top":
                        pr0, pr1 = max(0, bar_r0 - chain_dim), bar_r0
                    else:
                        pr0, pr1 = bar_r1, min(H - 1, bar_r1 + chain_dim)
                    pc0, pc1 = sA, sB
                else:
                    if side == "left":
                        pc0, pc1 = max(0, bar_c0 - chain_dim), bar_c0
                    else:
                        pc0, pc1 = bar_c1, min(W - 1, bar_c1 + chain_dim)
                    pr0, pr1 = sA, sB

                bbox_area = (pr1 - pr0 + 1) * (pc1 - pc0 + 1)
                target = 8
                if target >= bbox_area:
                    return None

                if side == "top":
                    seeds = {(bar_r0, c) for c in range(sA, sB + 1) if root_profile[c - sA]}
                elif side == "bottom":
                    seeds = {(bar_r1, c) for c in range(sA, sB + 1) if root_profile[c - sA]}
                elif side == "left":
                    seeds = {(r, bar_c0) for r in range(sA, sB + 1) if root_profile[r - sA]}
                else:
                    seeds = {(r, bar_c1) for r in range(sA, sB + 1) if root_profile[r - sA]}

                outside = {
                    (r, c)
                    for r in range(pr0, pr1 + 1)
                    for c in range(pc0, pc1 + 1)
                    if (r, c) not in occ_set and not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1)
                }
                avail = outside | seeds
                root = _grow_connected(
                    rng,
                    avail,
                    target,
                    seeds,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=6,
                )
                if root is None:
                    return None
                if not seeds.issubset(root):
                    return None
                if any((bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1 and (r, c) not in seeds) for r, c in root):
                    return None
                piece_cells = root
            else:
                prev_coords = chain_pieces[-1]["coords"]
                prs = [r for r, _ in prev_coords]
                pcs = [c for _, c in prev_coords]
                prev_r0, prev_r1 = min(prs), max(prs)
                prev_c0, prev_c1 = min(pcs), max(pcs)

                ext = set()
                if orientation == "horizontal":
                    cavity = {
                        (r, c)
                        for r in range(prev_r0, prev_r1 + 1)
                        for c in range(sA, sB + 1)
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

                all_avail = None
                for ext_dim in (1, 2):
                    if orientation == "horizontal":
                        es = (max(0, prev_r0 - ext_dim), prev_r0 - 1) if side == "top" else (prev_r1 + 1, min(H - 1, prev_r1 + ext_dim))
                        ext = {
                            (r, c)
                            for r in range(es[0], es[1] + 1)
                            for c in range(sA, sB + 1)
                            if (r, c) not in occ_set
                        } if es[0] <= es[1] else set()
                    else:
                        es = (max(0, prev_c0 - ext_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(W - 1, prev_c1 + ext_dim))
                        ext = {
                            (r, c)
                            for r in range(sA, sB + 1)
                            for c in range(es[0], es[1] + 1)
                            if (r, c) not in occ_set
                        } if es[0] <= es[1] else set()
                    candidate = cavity | ext
                    if 8 <= len(candidate) <= 13:
                        all_avail = candidate
                        break
                if all_avail is None:
                    return None
                if not (8 <= len(all_avail) <= 13):
                    return None

                is_tail = piece_idx == chain_len - 1
                target = len(all_avail) if is_tail else min(13, max(len(cavity), min(piece_area, len(all_avail))))

                piece_cells = _grow_connected(
                    rng,
                    all_avail,
                    target,
                    cavity,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=6,
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
                            for c in range(sA, sB + 1)
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
            chain_pieces.append({"color": piece_color, "coords": piece_cells})
            occ_set |= piece_cells

        if len(chain_pieces) < chain_len:
            return None
        all_chain_pieces.extend(chain_pieces)

    if not (3 <= len(all_chain_pieces) <= 6):
        return None

    decoy_targets = []
    if orientation == "horizontal":
        cols = [2, W // 3, (2 * W) // 3, W - 3]
        decoy_targets = [("top", c) for c in cols[:2]] + [("bottom", c) for c in cols[2:]]
    else:
        rows = [2, H // 3, (2 * H) // 3, H - 3]
        decoy_targets = [("left", r) for r in rows[:2]] + [("right", r) for r in rows[2:]]

    decoys = 0
    for ds, anchor in decoy_targets:
        if orientation == "horizontal":
            dn = {(bar_r0 if ds == "top" else bar_r1, anchor)}
        else:
            dn = {(anchor, bar_c0 if ds == "left" else bar_c1)}
        all_holes = set().union(*notch_holes) if notch_holes else set()
        pad = all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
        if dn & pad:
            continue
        if not dn.issubset(bar_cells):
            continue
        if not _connected(bar_cells - dn):
            continue
        bar_cells -= dn
        notch_holes.append(dn)
        decoys += 1
    if decoys < 4:
        return None
    if not _connected(bar_cells):
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
            output[r, c] = p["color"]

    if _count_deeper_unused_notch_cells(output) <= 0:
        return None

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
            if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in DIR4):
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
