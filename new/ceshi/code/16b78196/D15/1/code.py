# Variant of 16b78196
# Dimension: D15 — 符号指令：是否包含图例或提示标记 (symbol_instructions)
# Change: 无图例或符号指令——所有信息从几何关系推断... → 缺口处标记链方向
#
# Design analysis
# ---------------
# reference_status:
#   source_task: 9aaea919
#   source_solver: not present in the workspace; only the JSON task data exists.
#   consequence: this variant borrows only the visible "marker_pixels" idea
#     (small dedicated instruction pixels) and does not depend on any unseen code.
#
# concrete_variant:
#   marker:
#     color: 5 (reserved for markers only; bar and pieces never use it)
#     shape: 3-pixel monochrome L in a 2x2 bbox
#     count: exactly one marker per active chain root notch (1-2 total)
#     placement: one cell away from the bar, on the outward side of the active notch
#     semantics: the L orientation encodes one of {top, bottom, left, right}; that
#       side is the chain-growth direction and also selects which notch is active.
#   unchanged_core_rule:
#     the colored sprites still assemble exactly as in the original task:
#       root piece plugs the used notch, then later pieces fill the previous
#       piece's cavity while staying inside the same strip width.
#   output:
#     markers disappear; output contains only the bar plus assembled chains.
#
# behavior_trace:
#   original:
#     main() scans every boundary-opening notch and tries every scattered piece as
#     a possible root. Chain direction is inferred only from which bar side the
#     notch opens to.
#   variant:
#     main() first detects marker components, decodes their L orientation, and
#     maps each marker to the nearest notch on that same side of the bar. Only
#     those marked notches are allowed to start chains. Everything after root
#     selection uses the original cavity-filling search unchanged.
#   small_example:
#     if the bar has top, bottom, and bottom decoy notches but only the top notch
#     has a grey L marker above it, the solver must ignore the two unmarked
#     bottom notches even if some piece could geometrically fit there.
#
# complexity_calibration:
#   - The marker is tiny and repeated 1-2 times, so the symbolic cue is obvious
#     from 2-3 examples without flooding the board with extra objects.
#   - Decoy notches remain, so the marker is visually meaningful rather than
#     decorative: there are several possible holes, but only marked ones matter.
#   - The marker uses a fixed visible color and fixed size, keeping the symbolic
#     vocabulary small enough for ARC-style induction.
#
# implicit_assumption_review:
#   original_code_dependencies_changed:
#   - "pieces_raw = comps[1:]" assumed every non-bar component is a movable piece.
#     This is false now because marker components are also present; the variant
#     must split non-bar components into markers vs real pieces.
#   - root candidate generation previously iterated over every notch. That assumed
#     the set of valid roots was implicit in geometry alone. This is false now:
#     root candidates must be restricted to marker-selected notches.
#   - bar side / axis inference from the notch still works, but it is no longer
#     the only control signal; the marker side must agree with the chosen notch.
#   - output painting must never copy marker pixels. In the original task every
#     non-background pixel in the input belonged either to the bar or to pieces,
#     so "ignore the rest" was harmless; now the solver must explicitly discard
#     instruction markers when constructing the output.
#
# placement_legality:
#   - Markers are placed with a 1-cell gap from the bar so they remain separate
#     monochrome components and never interfere with notch geometry.
#   - Scattered pieces are placed away from both the bar and the markers, so the
#     marker remains an isolated symbol instead of visually merging with pieces.
#   - Marker color 5 is reserved, which avoids ambiguity with true pieces.
#
# visual_observability:
#   - Humans can see a few grey elbow marks hugging specific bar notches; those
#     notches are exactly where the final chains appear in the output.
#   - Because decoy notches remain unmarked, the symbolic cue is observable even
#     without mentally simulating the whole search procedure.
#
# new_main_assumptions_checked_against_generator:
#   - main() assumes marker components are color-5 L-shapes of area 3. The
#     generator guarantees this.
#   - main() assumes each marker sits outside the bar on the side encoded by its
#     orientation. The generator guarantees this with fixed offsets.
#   - main() assumes each marked side has a nearby notch mouth to map onto. The
#     generator creates the marker immediately after creating the active notch.

from __future__ import annotations

import numpy as np

from arc_utils import components_mask, find_components, get_bg


MARKER_COLOR = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MARKER_SHAPES = {
    "top": {(0, 0), (1, 0), (1, 1)},
    "bottom": {(0, 0), (0, 1), (1, 1)},
    "left": {(0, 1), (1, 0), (1, 1)},
    "right": {(0, 0), (0, 1), (1, 0)},
}
REL_TO_SIDE = {frozenset(v): k for k, v in MARKER_SHAPES.items()}


def _shape_from_coords(coords: np.ndarray, bbox):
    r0, c0, r1, c1 = map(int, bbox)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}


def _pixels(shape, top, left):
    return {(top + lr, left + lc) for lr, lc in shape["rel"]}


def _connected(cells):
    if len(cells) <= 1:
        return True
    cells_set = set(cells)
    start = next(iter(cells_set))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells_set)


def _touches_any(pts, other):
    for r, c in pts:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in other:
                return True
    return False


def _classify_marker(comp, shape):
    if int(comp["color"]) != MARKER_COLOR:
        return None
    if int(comp["area"]) != 3:
        return None
    if int(shape["h"]) > 2 or int(shape["w"]) > 2:
        return None
    return REL_TO_SIDE.get(frozenset(shape["rel_set"]))


def _bar_notches(bar_mask, bar_r0, bar_c0):
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
        if side == "top":
            mouth = {(bar_r0, bar_c0 + int(cc)) for cc in c[r == 0]}
        elif side == "bottom":
            mouth = {(bar_r0 + h - 1, bar_c0 + int(cc)) for cc in c[r == h - 1]}
        elif side == "left":
            mouth = {(bar_r0 + int(rr), bar_c0) for rr in r[c == 0]}
        else:
            mouth = {(bar_r0 + int(rr), bar_c0 + w - 1) for rr in r[c == w - 1]}
        notches.append({"axis": axis, "side": side, "hole": hole_gl, "mouth": mouth, "mouth_n": mouth_n})
    return notches


def _marker_outside_bar(marker_bbox, bar_bbox, side):
    mr0, mc0, mr1, mc1 = map(int, marker_bbox)
    br0, bc0, br1, bc1 = map(int, bar_bbox)
    if side == "top":
        return mr1 < br0
    if side == "bottom":
        return mr0 > br1
    if side == "left":
        return mc1 < bc0
    return mc0 > bc1


def _marker_notch_overlap(marker_bbox, notch):
    mr0, mc0, mr1, mc1 = map(int, marker_bbox)
    if notch["side"] in ("top", "bottom"):
        cols = [c for _, c in notch["mouth"]]
        return max(0, min(mc1, max(cols)) - max(mc0, min(cols)) + 1)
    rows = [r for r, _ in notch["mouth"]]
    return max(0, min(mr1, max(rows)) - max(mr0, min(rows)) + 1)


def _map_markers_to_notches(markers, notches, bar_bbox):
    if not markers:
        return list(range(len(notches)))

    picked = []
    used = set()
    for marker in markers:
        side = marker["side"]
        if not _marker_outside_bar(marker["bbox"], bar_bbox, side):
            continue
        scored = []
        marker_pts = {(int(r), int(c)) for r, c in marker["coords"]}
        for ni, notch in enumerate(notches):
            if ni in used or notch["side"] != side:
                continue
            overlap = _marker_notch_overlap(marker["bbox"], notch)
            if overlap <= 0:
                continue
            dist = min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in marker_pts for r2, c2 in notch["mouth"])
            scored.append(((overlap, -dist, int(notch["mouth_n"])), ni))
        if not scored:
            continue
        scored.sort(reverse=True)
        ni = scored[0][1]
        used.add(ni)
        picked.append(ni)
    return picked if picked else list(range(len(notches)))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    bar = comps[0]
    out = np.full((H, W), int(bg), dtype=int)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    pieces = []
    markers = []
    for comp in comps[1:]:
        shape = _shape_from_coords(comp["coords"], comp["bbox"])
        marker_side = _classify_marker(comp, shape)
        if marker_side is not None:
            markers.append(
                {
                    "color": int(comp["color"]),
                    "bbox": tuple(map(int, comp["bbox"])),
                    "coords": [(int(r), int(c)) for r, c in comp["coords"]],
                    "shape": shape,
                    "side": marker_side,
                }
            )
            continue
        r0, c0 = map(int, comp["bbox"][:2])
        pieces.append({"color": int(comp["color"]), "shape": shape, "orig": (r0, c0)})

    notches = _bar_notches(bar_mask, bar_r0, bar_c0)
    active_notch_ids = _map_markers_to_notches(markers, notches, tuple(map(int, bar["bbox"])))

    def placement_pts(shape, top, left, occ, must_cover=None, strip=None):
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

    def aligned_positions(axis, A, cover, shape):
        if axis == "cols":
            left = A
            tops = set()
            for rN, cN in cover:
                for lr, lc in shape["rel"]:
                    if cN == left + lc:
                        tops.add(rN - lr)
            for top in tops:
                yield top, left
        else:
            top = A
            lefts = set()
            for rN, cN in cover:
                for lr, lc in shape["rel"]:
                    if rN == top + lr:
                        lefts.add(cN - lc)
            for left in lefts:
                yield top, left

    def hole_positions(hole, shape):
        pos = set()
        for rH, cH in hole:
            for lr, lc in shape["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    root_cands = []
    for ni in active_notch_ids:
        notch = notches[ni]
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            shape = piece["shape"]
            for top, left in hole_positions(notch["hole"], shape):
                pts = placement_pts(shape, top, left, bar_occ, must_cover=notch["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + shape["w"] - 1)
                else:
                    A, B = int(top), int(top + shape["h"] - 1)
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), -int(notch["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": shape},
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
                for top, left in aligned_positions(axis, A, cavity, shape):
                    pts = placement_pts(shape, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                    if pts is None:
                        continue
                    if not _touches_any(pts, head_pts):
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

            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


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
                    cm = min((c for _, c in cells), default=B)
                    cx = max((c for _, c in cells), default=A)
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
                    cm = min((r for r, _ in cells), default=B)
                    cx = max((r for r, _ in cells), default=A)
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


def _make_marker_cells(side, bar_r0, bar_r1, bar_c0, bar_c1, A, B, rng):
    rel = MARKER_SHAPES[side]
    if side in ("top", "bottom"):
        start = A if rng.random() < 0.5 else B - 1
        row0 = bar_r0 - 3 if side == "top" else bar_r1 + 2
        pts = {(row0 + dr, start + dc) for dr, dc in rel}
    else:
        start = A if rng.random() < 0.5 else B - 1
        col0 = bar_c0 - 3 if side == "left" else bar_c1 + 2
        pts = {(start + dr, col0 + dc) for dr, dc in rel}
    return pts


def _gen_attempt():
    import random as _random

    rng = _random.Random()
    H, W = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    bar_color = rng.choice([c for c in range(1, 10) if c != MARKER_COLOR])
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

    avail_colors = [c for c in range(1, 10) if c not in (bar_color, MARKER_COLOR)]
    rng.shuffle(avail_colors)
    color_idx = 0

    all_chain_pieces = []
    notch_holes = []
    marker_cells_all = set()
    occ_set = set(bar_cells)

    for chain_len in chain_lens:
        strip_width = rng.randint(4, 5)

        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "top" else space_after) >= chain_len * 2 + 4), None)
            if side is None:
                return None
            sc0 = rng.randint(1, W - strip_width - 1)
            sc1 = sc0 + strip_width - 1
            sA, sB = sc0, sc1
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "left" else space_after) >= chain_len * 2 + 4), None)
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

                avail = {
                    (r, c)
                    for r in range(pr0, pr1 + 1)
                    for c in range(pc0, pc1 + 1)
                    if (r, c) in notch or (r, c) not in occ_set
                }
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
            chain_pieces.append({"color": piece_color, "coords": piece_cells})
            occ_set |= piece_cells

        marker_cells = _make_marker_cells(side, bar_r0, bar_r1, bar_c0, bar_c1, sA, sB, rng)
        if any(r < 0 or r >= H or c < 0 or c >= W for r, c in marker_cells):
            return None
        if marker_cells & marker_cells_all:
            return None
        if _touches_any(marker_cells, marker_cells_all):
            return None
        marker_cells_all |= marker_cells

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
    for piece in all_chain_pieces:
        for r, c in piece["coords"]:
            output[r, c] = piece["color"]

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color
    for r, c in marker_cells_all:
        input_grid[r, c] = MARKER_COLOR

    placed_occ = set(bar_cells) | set(marker_cells_all)
    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)

    for piece in pieces_order:
        coords = piece["coords"]
        color = piece["color"]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        p_r0, p_c0 = min(rs), min(cs)
        rel = [(r - p_r0, c - p_c0) for r, c in coords]
        p_h, p_w = max(rs) - p_r0 + 1, max(cs) - p_c0 + 1

        placed = False
        for _ in range(500):
            gap = rng.randint(3, 7)
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
            if _touches_any(pts, placed_occ):
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
