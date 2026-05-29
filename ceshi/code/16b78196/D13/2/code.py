# Variant of 16b78196
# Dimension: D13 — 链终止条件 (meta)
# Change: 尾部精灵的bbox在条带内无剩余空腔时链终止... → 由特殊颜色的终止精灵标记

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the D13 variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and flow trace
#    Original task:
#    - A chain grows along a fixed strip from a used bar notch.
#    - Every non-root piece covers the previous piece's cavity and extends farther.
#    - The chain stops only when the tail piece leaves zero cavity in its own bbox.
#
#    D13 variant:
#    - Keep the same bar, notch, strip-width lock, root matching, and cavity-filling
#      mechanics.
#    - Reserve a fixed visible color, GREY=5, as a terminal sprite color.
#    - Each chain still grows by covering the previous cavity, but the last placed
#      sprite is always grey. Once that grey sprite is inserted, the chain is closed
#      even if its bbox still contains background cells.
#    - Therefore output examples visibly differ from the original task: the grey tail
#      sits at the end of a chain and still has a residual black cavity inside its
#      strip-aligned bbox.
#
#    Small trace:
#    - Original: root(red) leaves cavity -> middle(blue) fills it and extends ->
#      tail(green) fills blue's cavity and leaves no cavity, so stop.
#    - Variant: root(red) leaves cavity -> middle(blue) fills it and extends ->
#      tail(grey) fills blue's cavity, extends, but still leaves 1-6 bg cells in its
#      own bbox; stop because color==5, not because cavity vanished.
#
# b) Complexity calibration
#    - Only one new degree of freedom is introduced: terminal color is fixed to grey.
#    - The rest of the puzzle remains the same, so ambiguity stays controlled.
#    - To make the change visible, generator enforces a positive residual cavity for
#      every grey tail. If grey tails also had zero cavity, the new control variable
#      would be unobservable and the task could degenerate to the original.
#
# c) Old-code assumptions that break under D13
#    - Old root search accepted every piece as a possible root. Under D13, grey pieces
#      must not start chains, so roots now exclude terminal-color components.
#    - Old `gen_extend_moves()` returned no moves when a head had no cavity; that was
#      effectively the only termination criterion. Under D13, a chain can be closed
#      even while cavity remains, so the search state must track `closed` explicitly.
#    - Old success condition `if not remaining: return True` assumed all placed chains
#      were complete once pieces were exhausted. Under D13, we must additionally
#      require that every chain has consumed a grey terminal.
#    - Old generator tail branch forced zero residual cavity. That directly encodes the
#      old D13 value and is replaced by a grey tail with positive residual cavity.
#
# d) Placement legality
#    - Pieces still never overlap the bar or each other, and scattered input pieces do
#      not touch the bar or one another.
#    - Every chain piece spans the same strip width as before. The grey tail still
#      covers the previous cavity and extends outward, so alignment logic remains
#      solver-friendly.
#
# e) Visual observability
#    - In 2-4 examples, the recurring fact is: grey pieces always appear at chain ends.
#    - More importantly, grey tails still contain black holes in the output; this makes
#      it visually clear that "no cavity remains" is no longer the stopping rule.
#
# f) Visual cleanliness
#    - No extra overlay markers are added. The marker is the whole tail piece color,
#      so the scene remains the same kind of bar + scattered sprites puzzle.
#    - Grey is reserved away from the bar and non-terminal sprites to keep the terminal
#      marker visually isolated.
#
# g) Color-change inferability
#    - There is no hidden numeric ordering on colors. Only a single fixed visible color
#      (grey=5) gets special semantics.
#    - Repetition across chains/examples supports the inference that grey means "stop".


MARKER_COLOR = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered sprites into a notched bar. Each chain grows along a locked
    strip, but chain termination is now controlled by a grey terminal sprite.
    """
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
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        return {"rel": rel, "rel_set": set(rel), "h": int(r1 - r0 + 1), "w": int(c1 - c0 + 1)}

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        color = int(p["color"])
        pieces.append(
            {
                "color": color,
                "is_terminal": color == MARKER_COLOR,
                "shape": shape_from_coords(p["coords"], p["bbox"]),
                "orig": (r0, c0),
            }
        )

    def bar_notches(mask, r0, c0):
        empty = ~mask
        h, w = map(int, empty.shape)
        notches = []
        for pts in components_mask(empty, connectivity=4):
            rr = pts[:, 0]
            cc = pts[:, 1]
            top = int((rr == 0).sum())
            bottom = int((rr == h - 1).sum())
            left = int((cc == 0).sum())
            right = int((cc == w - 1).sum())
            if top + bottom + left + right == 0:
                continue
            hole_gl = {(r0 + int(rp), c0 + int(cp)) for rp, cp in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    def pixels(shape, top, left):
        return {(top + lr, left + lc) for lr, lc in shape["rel"]}

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
        pts = pixels(shape, top, left)
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

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, shape):
        pos = set()
        for rH, cH in hole:
            for lr, lc in shape["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    def head_cavity(shape, top, left, occ_after):
        return {
            (r, c)
            for r in range(top, top + shape["h"])
            for c in range(left, left + shape["w"])
            if (r, c) not in occ_after
        }

    terminal_count = sum(p["is_terminal"] for p in pieces)

    root_cands = []
    seen_roots = set()
    for ni, notch in enumerate(notches):
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            if piece["is_terminal"]:
                continue
            shape = piece["shape"]
            for top, left in hole_positions(notch["hole"], shape):
                pts = placement_pts(shape, top, left, bar_occ, must_cover=notch["hole"])
                if pts is None:
                    continue
                if not head_cavity(shape, int(top), int(left), bar_occ | pts):
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + shape["w"] - 1)
                else:
                    A, B = int(top), int(top + shape["h"] - 1)
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_key = (ni, pi, int(top), int(left))
                if root_key in seen_roots:
                    continue
                seen_roots.add(root_key)
                root_cands.append(
                    (
                        (int(mv), -int(notch["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": shape},
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands]

    best_out = out.copy()
    best_key = None

    def search(out_grid, occ_set, chains, remaining):
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        closed_cnt = sum(1 for ch in chains if ch["closed"])
        key = (placed_cnt, closed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return all(ch["closed"] for ch in chains)

        def gen_extend_moves(chain, remaining_set):
            if chain["closed"]:
                return []
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = pixels(head["shape"], head["top"], head["left"])
            cavity = head_cavity(head["shape"], int(head["top"]), int(head["left"]), occ_set)
            if not cavity:
                return []

            moves = []
            seen_moves = set()
            for pi in remaining_set:
                piece = pieces[pi]
                shape = piece["shape"]
                for top, left in aligned_positions(axis, A, cavity, shape):
                    pts = placement_pts(shape, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                    if pts is None:
                        continue
                    if not touches(pts, head_pts):
                        continue
                    new_cavity = head_cavity(shape, int(top), int(left), occ_set | pts)
                    if (not piece["is_terminal"]) and not new_cavity:
                        continue
                    mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                    move_key = (pi, int(top), int(left))
                    if move_key in seen_moves:
                        continue
                    seen_moves.add(move_key)
                    moves.append(((int(piece["is_terminal"]), int(mv)), pi, int(top), int(left), pts, shape))
            moves.sort(key=lambda x: x[0])
            return [(pi, top, left, pts, shape) for _, pi, top, left, pts, shape in moves]

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for pi, top, left, pts, shape in moves:
                piece = pieces[pi]
                for r, c in pts:
                    out_grid[r, c] = int(piece["color"])
                old_head = ch["head"]
                old_closed = ch["closed"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": shape}
                ch["closed"] = bool(piece["is_terminal"])
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["head"] = old_head
                ch["closed"] = old_closed
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
        return False

    from itertools import combinations

    max_roots = min(3, len(cand_roots), terminal_count if terminal_count > 0 else 1)
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
                        "closed": False,
                    }
                )

            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            if search(out0, occ0, chains, remaining):
                return best_out

    return best_out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

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


def _piece_cavity(piece_cells, occ_before, orientation, A, B):
    prs = [r for r, _ in piece_cells]
    pcs = [c for _, c in piece_cells]
    r0, r1 = min(prs), max(prs)
    c0, c1 = min(pcs), max(pcs)
    if orientation == "horizontal":
        bbox_strip = {(r, c) for r in range(r0, r1 + 1) for c in range(A, B + 1)}
    else:
        bbox_strip = {(r, c) for r in range(A, B + 1) for c in range(c0, c1 + 1)}
    return {cell for cell in bbox_strip if cell not in occ_before and cell not in piece_cells}


def _gen_attempt(return_meta=False):
    import random as _random

    rng = _random.Random()
    H, W = 30, 30
    bg = 0
    residual_lo, residual_hi = 1, 6

    orientation = rng.choice(["horizontal", "vertical"])
    bar_color = rng.choice([c for c in range(1, 10) if c != MARKER_COLOR])
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

    avail_colors = [c for c in range(1, 10) if c not in (bar_color, MARKER_COLOR)]
    rng.shuffle(avail_colors)
    color_idx = 0

    all_chain_pieces = []
    notch_holes = []
    occ_set = set(bar_cells)
    chain_metas = []

    for chain_len in chain_lens:
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
            n_avail = {(r, c) for r in range(nr0, nr1 + 1) for c in range(sA, sB + 1) if (r, c) in bar_cells}
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
        chain_residuals = []
        for piece_idx in range(chain_len):
            is_terminal = piece_idx == chain_len - 1
            if is_terminal:
                piece_color = MARKER_COLOR
            else:
                if color_idx >= len(avail_colors):
                    return None
                piece_color = avail_colors[color_idx]
                color_idx += 1

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
                lo = max(8, int(bbox_area * 0.54))
                hi = min(13, int(bbox_area * 0.78))
                if lo > hi:
                    return None
                piece_cells = None
                for _ in range(20):
                    target = rng.randint(lo, hi)
                    avail = {(r, c) for r in range(pr0, pr1 + 1) for c in range(pc0, pc1 + 1) if (r, c) in notch or (r, c) not in occ_set}
                    cand = _grow_connected(
                        rng,
                        avail,
                        target,
                        notch,
                        must_span_axis="cols" if orientation == "horizontal" else "rows",
                        span_range=(sA, sB),
                        retries=5,
                    )
                    if cand is None:
                        continue
                    if not any(not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1) for r, c in cand):
                        continue
                    residual = _piece_cavity(cand, occ_set, orientation, sA, sB)
                    if not (residual_lo <= len(residual) <= residual_hi):
                        continue
                    piece_cells = cand
                    chain_residuals.append(len(residual))
                    break
                if piece_cells is None:
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
                    es = (max(0, prev_r0 - chain_dim), prev_r0 - 1) if side == "top" else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    ext = {(r, c) for r in range(es[0], es[1] + 1) for c in range(sA, sB + 1) if (r, c) not in occ_set} if es[0] <= es[1] else set()
                else:
                    cavity = {
                        (r, c)
                        for r in range(sA, sB + 1)
                        for c in range(prev_c0, prev_c1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                    es = (max(0, prev_c0 - chain_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    ext = {(r, c) for r in range(sA, sB + 1) for c in range(es[0], es[1] + 1) if (r, c) not in occ_set} if es[0] <= es[1] else set()

                if not cavity or not ext:
                    return None
                all_avail = cavity | ext
                if len(cavity) > 13 or len(all_avail) < max(8, len(cavity) + 1):
                    return None

                piece_cells = None
                lo = max(8, len(cavity) + 1)
                hi = min(13, len(all_avail))
                if lo > hi:
                    return None
                for _ in range(24):
                    target = rng.randint(lo, hi)
                    cand = _grow_connected(
                        rng,
                        all_avail,
                        target,
                        cavity,
                        must_span_axis="cols" if orientation == "horizontal" else "rows",
                        span_range=(sA, sB),
                        retries=5,
                    )
                    if cand is None or not cavity.issubset(cand):
                        continue
                    if not (cand & ext):
                        continue
                    residual = _piece_cavity(cand, occ_set, orientation, sA, sB)
                    if not (residual_lo <= len(residual) <= residual_hi):
                        continue
                    piece_cells = cand
                    chain_residuals.append(len(residual))
                    break
                if piece_cells is None:
                    return None

            if not (8 <= len(piece_cells) <= 13):
                return None
            chain_pieces.append({"color": piece_color, "coords": piece_cells, "terminal": is_terminal})
            occ_set |= piece_cells

        all_chain_pieces.extend(chain_pieces)
        chain_metas.append({"len": chain_len, "terminal_residual": chain_residuals[-1], "residuals": list(chain_residuals)})

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

    meta = {
        "orientation": orientation,
        "num_chains": num_chains,
        "chain_lens": list(chain_lens),
        "marker_color": MARKER_COLOR,
        "chain_metas": chain_metas,
    }
    return (input_grid, output, meta) if return_meta else (input_grid, output)


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
