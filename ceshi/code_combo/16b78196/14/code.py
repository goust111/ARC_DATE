# Combo variant of 16b78196 (index 14)
# combo_spec: D1/2+D2/1+D10/2
# Name: 边缘轮廓匹配+锚点精灵+高相似干扰

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


# Design analysis for the fused variant
# -------------------------------------
# 1. Concretisation / behavior trace
#    - Original root logic: monochrome root pieces were translated until they completely
#      covered a notch hole.
#    - Fused root logic:
#        * each sprite is now multicolor: one anchor-color pixel + one body color
#        * each used notch has one anchor marker pixel of that same anchor color
#        * the root no longer fills the whole notch; it only occupies the notch mouth
#          profile on the bar boundary, leaving the deeper notch cavity empty
#        * decoy notches mimic the same strip width and almost the same mouth shape,
#          differing by only 1-2 pixels, usually by filling the true profile's gap
#      Concrete trace for a top-side chain:
#        old: root covers the whole 2-deep hole
#        fused: root bottom edge = 1 1 0 1 1, notch mouth = 1 1 0 1 1, deeper cells
#        behind the gap remain black, and the root anchor lands on the notch marker
#        while a nearby decoy mouth 1 1 1 1 1 stays unused.
#    - Subsequent chain growth stays structurally close to the source task: the next
#      piece fills the background cavity inside the previous piece's bbox, constrained
#      to the same strip width, then extends farther away from the bar.
# 2. Complexity calibration
#    - The generator always creates two chains of length two. This gives repeated
#      evidence for all three cues: edge-profile fit, anchor alignment, and similar
#      distractors.
#    - Valid mouth profiles come from a small family: width 4-5 with a single visible
#      gap. Decoys usually add that missing mouth pixel, so the visual ambiguity is
#      strong but still human-auditable.
#    - A global anchor color is reused across all pieces and used notches to keep the
#      cue obvious without exhausting ARC's palette.
# 3. Implicit-assumption audit against the original solver
#    - `find_components(... monochromatic=True)` can still find the bar, but it can no
#      longer directly recover pieces because pieces are multicolor. The solver now
#      removes the bar and re-groups all remaining non-bg cells as mixed components.
#    - Original root search assumed `must_cover=hole`. That is false here because the
#      deeper notch cells must remain empty. The solver now checks that the translated
#      root occupies a subset of the notch hole equal to its attaching edge profile.
#    - Original arbitrary root translation from hole coverage is no longer appropriate.
#      Root position is fixed by notch side + strip span, while anchor-marker alignment
#      and mouth-profile mismatch are used as ranking signals.
#    - New solver assumption: the anchor pixel of a valid root lies on the mouth layer.
#      The generator guarantees this by choosing a removable mouth cell as the root
#      anchor before scattering the piece.
#    - New solver assumption: close decoys differ from a valid mouth by at most 2 cells
#      and are unmarked. The generator constructs exactly that family and rejects
#      degenerate samples with too few such decoys.
# 4. Placement legality
#    - Output-first construction is retained. Every hole carve is checked against bar
#      connectivity and against adjacency to previously carved holes.
#    - Scattered pieces preserve orientation, do not overlap the bar or each other, and
#      do not 4-touch existing occupied cells.
#    - Root anchors are chosen so that recoloring them does not disconnect the remaining
#      body-color component; this keeps anchor-color inference from singleton mono
#      components stable.
# 5. Visual observability / cleanliness / inferability
#    - A human sees the same odd color inside every piece and on exactly the used notches.
#      They also see that the inserted root only hooks onto the notch lip while the deep
#      recess stays black.
#    - Decoys are visually near-copies of real notches, so the anchor cue has a visible
#      purpose rather than acting as redundant decoration.
#    - Colors are structurally interpretable: one bar color, one anchor color, and one
#      body color per piece. No rule depends on the numeric identity of a color.


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ATTACH_EDGE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


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


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=5):
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 24):
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
                    mn = min((c for _, c in cells), default=B)
                    mx = max((c for _, c in cells), default=A)
                    if mn > A:
                        picks = [nb for nb in frontier if nb[1] < mn]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                    if mx < B:
                        picks = [nb for nb in frontier if nb[1] > mx]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                else:
                    mn = min((r for r, _ in cells), default=B)
                    mx = max((r for r, _ in cells), default=A)
                    if mn > A:
                        picks = [nb for nb in frontier if nb[0] < mn]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                    if mx < B:
                        picks = [nb for nb in frontier if nb[0] > mx]
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


def _edge_profile_from_shape(rel_set: set[tuple[int, int]], h: int, w: int, side: str) -> tuple[int, ...]:
    edge = ATTACH_EDGE[side]
    if edge == "top":
        return tuple(int((0, c) in rel_set) for c in range(w))
    if edge == "bottom":
        rr = h - 1
        return tuple(int((rr, c) in rel_set) for c in range(w))
    if edge == "left":
        return tuple(int((r, 0) in rel_set) for r in range(h))
    cc = w - 1
    return tuple(int((r, cc) in rel_set) for r in range(h))


def _piece_anchor_key(orientation: str, side: str):
    if orientation == "horizontal":
        if side == "top":
            return lambda rc: (-rc[0], rc[1])
        return lambda rc: (rc[0], rc[1])
    if side == "left":
        return lambda rc: (-rc[1], rc[0])
    return lambda rc: (rc[1], rc[0])


def _choose_anchor_cell(piece_cells, preferred, order_key):
    ordered = sorted(preferred, key=order_key) + sorted(set(piece_cells) - set(preferred), key=order_key)
    for rc in ordered:
        if len(piece_cells) <= 1 or _connected(set(piece_cells) - {rc}):
            return rc
    return ordered[0] if ordered else None


def _used_notch_shape(side, span_start, width, bar_r0, bar_r1, bar_c0, bar_c1, gap_idx):
    if side in ("top", "bottom"):
        cols = [span_start + i for i in range(width)]
        mouth_cols = [c for i, c in enumerate(cols) if i != gap_idx]
        deep_l = max(0, gap_idx - 1)
        deep_r = min(width - 1, gap_idx + 1)
        deep_cols = [span_start + i for i in range(deep_l, deep_r + 1)]
        if side == "top":
            hole = {(bar_r0, c) for c in mouth_cols} | {(bar_r0 + 1, c) for c in deep_cols}
        else:
            hole = {(bar_r1, c) for c in mouth_cols} | {(bar_r1 - 1, c) for c in deep_cols}
        mouth = {(bar_r0 if side == "top" else bar_r1, c) for c in mouth_cols}
    else:
        rows = [span_start + i for i in range(width)]
        mouth_rows = [r for i, r in enumerate(rows) if i != gap_idx]
        deep_l = max(0, gap_idx - 1)
        deep_r = min(width - 1, gap_idx + 1)
        deep_rows = [span_start + i for i in range(deep_l, deep_r + 1)]
        if side == "left":
            hole = {(r, bar_c0) for r in mouth_rows} | {(r, bar_c0 + 1) for r in deep_rows}
        else:
            hole = {(r, bar_c1) for r in mouth_rows} | {(r, bar_c1 - 1) for r in deep_rows}
        mouth = {(r, bar_c0 if side == "left" else bar_c1) for r in mouth_rows}
    profile = tuple(1 if i != gap_idx else 0 for i in range(width))
    return hole, mouth, profile


def _similar_decoy_notch_shape(side, span_start, width, bar_r0, bar_r1, bar_c0, bar_c1, gap_idx, add_extra_deep):
    hole, _, _ = _used_notch_shape(side, span_start, width, bar_r0, bar_r1, bar_c0, bar_c1, gap_idx)
    if side == "top":
        hole.add((bar_r0, span_start + gap_idx))
        if add_extra_deep and 0 <= gap_idx + 2 < width:
            hole.add((bar_r0 + 1, span_start + gap_idx + 2))
    elif side == "bottom":
        hole.add((bar_r1, span_start + gap_idx))
        if add_extra_deep and 0 <= gap_idx - 2 < width:
            hole.add((bar_r1 - 1, span_start + gap_idx - 2))
    elif side == "left":
        hole.add((span_start + gap_idx, bar_c0))
        if add_extra_deep and 0 <= gap_idx + 2 < width:
            hole.add((span_start + gap_idx + 2, bar_c0 + 1))
    else:
        hole.add((span_start + gap_idx, bar_c1))
        if add_extra_deep and 0 <= gap_idx - 2 < width:
            hole.add((span_start + gap_idx - 2, bar_c1 - 1))
    return hole


def _find_bar_anchor_and_notches(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = get_bg(g)
    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    mono.sort(key=lambda d: -d["area"])
    assert mono
    bar = mono[0]
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_color = int(bar["color"])
    color_to_areas: dict[int, list[int]] = {}
    for comp in mono[1:]:
        col = int(comp["color"])
        color_to_areas.setdefault(col, []).append(int(comp["area"]))
    anchor_candidates = [(len(areas), col) for col, areas in color_to_areas.items() if areas and max(areas) == 1]
    assert anchor_candidates
    anchor_color = max(anchor_candidates)[1]

    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True

    notches = []
    for pts in components_mask(~bar_mask, connectivity=4):
        rr = pts[:, 0]
        cc = pts[:, 1]
        top = int((rr == 0).sum())
        bottom = int((rr == bar_h - 1).sum())
        left = int((cc == 0).sum())
        right = int((cc == bar_w - 1).sum())
        if top + bottom + left + right == 0:
            continue
        side, mouth_n = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )
        axis = "cols" if side in ("top", "bottom") else "rows"
        hole = {(bar_r0 + int(r), bar_c0 + int(c)) for r, c in pts}
        hole_local = {(int(r), int(c)) for r, c in pts}
        markers = {(r, c) for r, c in hole if int(g[r, c]) == anchor_color}
        if side == "top":
            A, B = int(cc.min()), int(cc.max())
            profile = tuple(int((0, x) in hole_local) for x in range(A, B + 1))
            mouth = {(bar_r0, bar_c0 + x) for x in range(A, B + 1) if profile[x - A]}
        elif side == "bottom":
            A, B = int(cc.min()), int(cc.max())
            profile = tuple(int((bar_h - 1, x) in hole_local) for x in range(A, B + 1))
            mouth = {(bar_r1, bar_c0 + x) for x in range(A, B + 1) if profile[x - A]}
        elif side == "left":
            A, B = int(rr.min()), int(rr.max())
            profile = tuple(int((x, 0) in hole_local) for x in range(A, B + 1))
            mouth = {(bar_r0 + x, bar_c0) for x in range(A, B + 1) if profile[x - A]}
        else:
            A, B = int(rr.min()), int(rr.max())
            profile = tuple(int((x, bar_w - 1) in hole_local) for x in range(A, B + 1))
            mouth = {(bar_r0 + x, bar_c1) for x in range(A, B + 1) if profile[x - A]}
        notches.append(
            {
                "axis": axis,
                "side": side,
                "hole": hole,
                "mouth": mouth,
                "profile": profile,
                "A": (bar_c0 + A) if axis == "cols" else (bar_r0 + A),
                "B": (bar_c0 + B) if axis == "cols" else (bar_r0 + B),
                "mouth_n": mouth_n,
                "markers": markers,
            }
        )
    return {
        "grid": g,
        "bg": bg,
        "bar": bar,
        "bar_occ": bar_occ,
        "bar_color": bar_color,
        "anchor_color": anchor_color,
        "notches": notches,
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    parsed = _find_bar_anchor_and_notches(g)
    bg = parsed["bg"]
    bar = parsed["bar"]
    bar_occ = parsed["bar_occ"]
    bar_color = parsed["bar_color"]
    anchor_color = parsed["anchor_color"]
    notches = parsed["notches"]

    out = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_occ:
        out[r, c] = bar_color

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
        rel_set = {(dr, dc) for dr, dc, _ in rel}
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        return {
            "rel": rel,
            "rel_set": rel_set,
            "anchor": anchor_rel,
            "body_color": next(iter(body_colors)),
            "h": h,
            "w": w,
            "profiles": {side: _edge_profile_from_shape(rel_set, h, w, side) for side in ATTACH_EDGE},
        }

    fg_mask = g != bg
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
        arr = np.array(coords, dtype=int)
        pieces.append({"shape": shape_from_coords(arr), "orig": (int(arr[:, 0].min()), int(arr[:, 1].min()))})

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

    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])

    def root_position(notch, v):
        if notch["side"] == "top":
            return bar_r0 - v["h"] + 1, int(notch["A"])
        if notch["side"] == "bottom":
            return bar_r1, int(notch["A"])
        if notch["side"] == "left":
            return int(notch["A"]), bar_c0 - v["w"] + 1
        return int(notch["A"]), bar_c1

    root_cands = []
    for ni, notch in enumerate(notches):
        span = int(notch["B"] - notch["A"] + 1)
        for pi, piece in enumerate(pieces):
            v = piece["shape"]
            if (notch["axis"] == "cols" and v["w"] != span) or (notch["axis"] == "rows" and v["h"] != span):
                continue
            top, left = root_position(notch, v)
            pts = placement_pts(v, int(top), int(left), bar_occ)
            if pts is None:
                continue
            anchor_pos = (int(top + v["anchor"][0]), int(left + v["anchor"][1]))
            if anchor_pos not in notch["mouth"]:
                continue
            inside_hole = pts & notch["hole"]
            # Root pieces only hook onto the notch lip; the deeper recess must remain empty.
            if inside_hole != notch["mouth"]:
                continue
            profile = v["profiles"][notch["side"]]
            if profile != notch["profile"]:
                continue
            if len(notch["markers"]) != 1:
                continue
            marker = next(iter(notch["markers"]))
            if anchor_pos != marker:
                continue
            mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
            root_cands.append(
                (
                    (mv, -int(notch["mouth_n"])),
                    {
                        "ni": ni,
                        "pi": pi,
                        "top": int(top),
                        "left": int(left),
                        "axis": notch["axis"],
                        "A": int(notch["A"]),
                        "B": int(notch["B"]),
                        "shape": v,
                    },
                )
            )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:32]]

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
                    mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                    moves.append((mv, len(pts), pi, top, left, pts, v))
            moves.sort()
            return [(pi, top, left, pts, v) for _, _, pi, top, left, pts, v in moves]

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

    for k in range(1, min(2, len(cand_roots)) + 1):
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
            if search(out0, occ0, chains, remaining):
                return best_out

    return best_out


def _fusion_metrics(inp: np.ndarray) -> dict[str, int]:
    parsed = _find_bar_anchor_and_notches(inp)
    g = parsed["grid"]
    bg = parsed["bg"]
    anchor_color = parsed["anchor_color"]
    notches = parsed["notches"]
    marked = [n for n in notches if len(n["markers"]) == 1]
    unmarked = [n for n in notches if len(n["markers"]) == 0]

    near_decoys = 0
    for u in marked:
        for d in unmarked:
            if u["side"] != d["side"]:
                continue
            if len(u["profile"]) != len(d["profile"]):
                continue
            dist = sum(int(a != b) for a, b in zip(u["profile"], d["profile"]))
            if 1 <= dist <= 2:
                near_decoys += 1

    fg_mask = g != bg
    for r, c in parsed["bar_occ"]:
        fg_mask[r, c] = False
    comps = components_mask(fg_mask, connectivity=4)
    pieces = 0
    for pts in comps:
        coords = [(int(r), int(c)) for r, c in pts]
        if len(coords) == 1 and int(g[coords[0]]) == anchor_color:
            continue
        if len(coords) >= 8:
            pieces += 1

    return {
        "marked_notches": len(marked),
        "unmarked_notches": len(unmarked),
        "near_decoys": near_decoys,
        "pieces": pieces,
    }


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

    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    num_chains = 2
    chain_lens = [2, 2]
    if len(palette) < sum(chain_lens):
        return None
    body_colors = palette[: sum(chain_lens)]
    color_idx = 0

    all_chain_pieces = []
    chain_specs = []
    notch_holes: list[set[tuple[int, int]]] = []
    occ_set = set(bar_cells)

    for chain_len in chain_lens:
        strip_width = rng.randint(4, 5)
        gap_candidates = [0, strip_width - 1]
        chosen_meta = None

        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            starts = list(range(2, W - strip_width - 1))
            rng.shuffle(starts)
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            starts = list(range(2, H - strip_width - 1))
            rng.shuffle(starts)

        for side in sides:
            need = chain_len * 2 + 3
            if (space_before if side in ("top", "left") else space_after) < need:
                continue
            for span_start in starts:
                rng.shuffle(gap_candidates)
                for gap_idx in gap_candidates:
                    hole, mouth, profile = _used_notch_shape(side, span_start, strip_width, bar_r0, bar_r1, bar_c0, bar_c1, gap_idx)
                    all_holes = set().union(*notch_holes) if notch_holes else set()
                    pad = all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
                    if hole & pad:
                        continue
                    if not hole.issubset(bar_cells):
                        continue
                    if not _connected(bar_cells - hole):
                        continue
                    chosen_meta = {
                        "side": side,
                        "span_start": span_start,
                        "span_end": span_start + strip_width - 1,
                        "strip_width": strip_width,
                        "gap_idx": gap_idx,
                        "hole": hole,
                        "mouth": mouth,
                        "profile": profile,
                    }
                    break
                if chosen_meta is not None:
                    break
            if chosen_meta is not None:
                break
        if chosen_meta is None:
            return None

        side = chosen_meta["side"]
        sA = chosen_meta["span_start"]
        sB = chosen_meta["span_end"]
        notch = chosen_meta["hole"]
        mouth = chosen_meta["mouth"]
        gap_idx = chosen_meta["gap_idx"]
        bar_cells -= notch
        notch_holes.append(notch)

        chain_pieces = []
        for piece_idx in range(chain_len):
            body_color = body_colors[color_idx]
            color_idx += 1

            if piece_idx == 0:
                chain_dim = 2
                target = 8 if strip_width == 4 else 9
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

                if side == "top":
                    seeds = {(bar_r0, c) for c in range(sA, sB + 1) if chosen_meta["profile"][c - sA]}
                elif side == "bottom":
                    seeds = {(bar_r1, c) for c in range(sA, sB + 1) if chosen_meta["profile"][c - sA]}
                elif side == "left":
                    seeds = {(r, bar_c0) for r in range(sA, sB + 1) if chosen_meta["profile"][r - sA]}
                else:
                    seeds = {(r, bar_c1) for r in range(sA, sB + 1) if chosen_meta["profile"][r - sA]}

                outside = {
                    (r, c)
                    for r in range(pr0, pr1 + 1)
                    for c in range(pc0, pc1 + 1)
                    if (r, c) not in occ_set and not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1)
                }
                avail = outside | seeds
                piece_cells = _grow_connected(
                    rng,
                    avail,
                    target,
                    seeds,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=8,
                )
                if piece_cells is None:
                    return None
                if not seeds.issubset(piece_cells):
                    return None
                if any((bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1 and (r, c) not in seeds) for r, c in piece_cells):
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
                        if side == "top":
                            es = (max(0, prev_r0 - ext_dim), prev_r0 - 1)
                        else:
                            es = (prev_r1 + 1, min(H - 1, prev_r1 + ext_dim))
                        ext = {
                            (r, c)
                            for r in range(es[0], es[1] + 1)
                            for c in range(sA, sB + 1)
                            if (r, c) not in occ_set
                        } if es[0] <= es[1] else set()
                    else:
                        if side == "left":
                            es = (max(0, prev_c0 - ext_dim), prev_c0 - 1)
                        else:
                            es = (prev_c1 + 1, min(W - 1, prev_c1 + ext_dim))
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
                piece_cells = _grow_connected(
                    rng,
                    all_avail,
                    len(all_avail),
                    cavity,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=8,
                )
                if piece_cells is None or not cavity.issubset(piece_cells):
                    return None
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

            chain_pieces.append(
                {
                    "coords": set(piece_cells),
                    "body_color": body_color,
                    "orientation": orientation,
                    "side": side,
                    "mouth": set(mouth) if piece_idx == 0 else None,
                    "gap_idx": gap_idx,
                }
            )
            occ_set |= piece_cells

        all_chain_pieces.extend(chain_pieces)
        chain_specs.append(chosen_meta)

    if len(all_chain_pieces) != 4:
        return None

    for spec in chain_specs:
        side = spec["side"]
        width = spec["strip_width"]
        gap_idx = spec["gap_idx"]
        starts = list(range(2, (W if orientation == "horizontal" else H) - width - 1))
        rng.shuffle(starts)
        made = 0
        for add_extra_deep in [False, True]:
            placed = False
            for span_start in starts:
                hole = _similar_decoy_notch_shape(side, span_start, width, bar_r0, bar_r1, bar_c0, bar_c1, gap_idx, add_extra_deep)
                all_holes = set().union(*notch_holes) if notch_holes else set()
                pad = all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
                if hole & pad:
                    continue
                if not hole.issubset(bar_cells):
                    continue
                if not _connected(bar_cells - hole):
                    continue
                bar_cells -= hole
                notch_holes.append(hole)
                placed = True
                made += 1
                break
            if not placed:
                return None
        if made != 2:
            return None

    bar_h = bar_r1 - bar_r0 + 1
    bar_w = bar_c1 - bar_c0 + 1
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr, cc = pts[:, 0], pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return None

    root_markers = []
    for piece in all_chain_pieces:
        if piece["mouth"] is not None:
            preferred = piece["coords"] & piece["mouth"]
            if not preferred:
                return None
            order_key = _piece_anchor_key(piece["orientation"], piece["side"])
            anchor = _choose_anchor_cell(piece["coords"], preferred, order_key)
            if anchor not in preferred:
                anchor = sorted(preferred, key=order_key)[0]
            piece["anchor"] = anchor
            root_markers.append(anchor)
        else:
            order_key = _piece_anchor_key(piece["orientation"], piece["side"])
            anchor = _choose_anchor_cell(piece["coords"], set(piece["coords"]), order_key)
            if anchor is None:
                return None
            piece["anchor"] = anchor

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = bar_color
    for piece in all_chain_pieces:
        for r, c in piece["coords"]:
            output[r, c] = anchor_color if (r, c) == piece["anchor"] else piece["body_color"]

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color
    for r, c in root_markers:
        input_grid[r, c] = anchor_color

    placed_occ = set(bar_cells) | set(root_markers)
    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)
    for piece in pieces_order:
        coords = piece["coords"]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        p_r0, p_c0 = min(rs), min(cs)
        rel = [(r - p_r0, c - p_c0, anchor_color if (r, c) == piece["anchor"] else piece["body_color"]) for r, c in coords]
        p_h = max(rs) - p_r0 + 1
        p_w = max(cs) - p_c0 + 1
        placed = False
        for _ in range(600):
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

    metrics = _fusion_metrics(input_grid)
    if metrics["marked_notches"] != 2 or metrics["pieces"] != 4 or metrics["near_decoys"] < 2:
        return None

    return input_grid, output


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 400 attempts")
