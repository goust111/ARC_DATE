# Variant of 16b78196
# Dimension: D14 — 精灵数量与链配置 (meta)
# Change: 总共3-6个精灵，分布在1-2条链中，每链2-3个精灵... → 含干扰精灵（不属于任何链）
#
# Design analysis
# ---
# variant_spec:
#   concrete_behavior:
#     input:
#       - A full-span bar with 6-9 boundary-opening notches remains unchanged.
#       - Real chain sprites are still scattered away from the bar and later plug into
#         1-2 used notches exactly as in the original task.
#       - Extra 1-2 monochrome decoy sprites are added to the scattered pool.
#       - Decoys visually resemble normal sprites in area/connectivity, but they never
#         fit any bar notch.
#       - At least one decoy shares a real chain's strip width and can imitate a later
#         continuation candidate, yet it cannot belong to any complete chain because it
#         leaves a cavity that no remaining real sprite can fill.
#     output:
#       - Only the true chain sprites are assembled into the bar.
#       - Decoys disappear because they were never part of any valid chain.
#   behavior_trace:
#     original:
#       - main() assumed every non-bar component must be consumed.
#       - search() stopped only when `remaining` became empty.
#     variant:
#       - main() must score partial explanations and keep the best valid subset.
#       - Search now prefers states with more completed chains and fewer open cavities,
#         rather than requiring all scattered sprites to be used.
#       - generator builds the normal output first, then appends 1-2 extra input-only
#         decoys that fail the root/extension constraints.
#     concrete_example:
#       - Suppose the bar has two true chains with lengths 2 and 3, plus one decoy.
#       - Original logic would keep searching until the decoy was also forced somewhere,
#         which could only fail or create a wrong output.
#       - Variant logic accepts the 5-piece assembly once both chain heads have zero
#         cavity, leaves the decoy unused, and returns that output.
#   complexity_calibration:
#     - Decoys keep the same area range (8-13) and connectivity as real sprites, so
#       they are not trivially identifiable by size alone.
#     - Only 1-2 decoys are added, keeping the branch factor moderate and the visual
#       change obvious without making the board chaotic.
#     - One decoy is deliberately "near-valid": it can mimic a tail candidate and thus
#       forces the solver to reason about complete chains instead of raw piece count.
#     - Optional extra decoys stay passive and only increase the scattered candidate pool.
#   implicit_assumptions_review:
#     - `if not remaining: return True` depended on "all pieces are useful"; this is
#       invalid now and is replaced by best-state scoring.
#     - Root enumeration still assumes the true chain count is 1-2; that remains true.
#     - Extension matching still assumes fixed strip width inside a chain; unchanged.
#     - Generator-side validation now rejects samples where non-root sprites would also
#       fit any notch, because that ambiguity was harmless before but would be harmful
#       once extra sprites exist.
#   placement_legality:
#     - All scattered sprites, including decoys, are placed non-overlapping, in-bounds,
#       and 4-neighbor separated from the bar and from each other.
#     - Decoys are generated in local coordinates first, then scattered with the same
#       spacing policy as real sprites.
#   visual_observability:
#     - Across examples, the user sees more scattered sprites in the input than appear
#       in the output. The missing ones are exactly the sprites that cannot plug any gap.
#     - The rule is inferable from a few examples because the central assembly behavior
#       remains stable while the unmatched extras are consistently ignored.
#   visual_cleanliness:
#     - Decoys are isolated from other pieces and from the bar, so they do not overlap
#       or visually mask the chain structure.
#   color_reasoning:
#     - Colors remain per-piece and mostly distinct. No invisible ordering by color id
#       is introduced; the solver relies on geometry/placement only.
#   new_solver_assumptions_checked_against_generator:
#     - main() assumes decoys never fit a bar notch. _gen_attempt() enforces that.
#     - main() assumes any continuation-style decoy cannot lead to a fully completed
#       chain. _gen_attempt() validates that no real sprite can close the decoy's cavity.
#     - main() assumes at least one complete chain exists. _gen_attempt() still builds
#       1-2 genuine chains before adding decoys.

from __future__ import annotations

from itertools import combinations
import random

import numpy as np

from arc_utils import components_mask, find_components, get_bg


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _shape_from_coords(coords: np.ndarray, bbox):
    r0, c0, r1, c1 = map(int, bbox)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}


def _shape_from_cells(cells: set[tuple[int, int]]):
    coords = np.array(sorted(cells), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return _shape_from_coords(coords, (int(r0), int(c0), int(r1), int(c1)))


def _shape_signature(shape) -> tuple[int, int, frozenset[tuple[int, int]]]:
    return int(shape["h"]), int(shape["w"]), frozenset(shape["rel"])


def _pixels(shape, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top) + lr, int(left) + lc) for lr, lc in shape["rel"]}


def _placement_pts(shape, top, left, H, W, occ, must_cover=None, strip=None):
    top = int(top)
    left = int(left)
    if top < 0 or left < 0 or top + int(shape["h"]) > H or left + int(shape["w"]) > W:
        return None
    if strip is not None:
        axis, A, B = strip
        if axis == "cols":
            if left != int(A) or int(shape["w"]) != int(B) - int(A) + 1:
                return None
        else:
            if top != int(A) or int(shape["h"]) != int(B) - int(A) + 1:
                return None
    pts = _pixels(shape, top, left)
    if pts & occ:
        return None
    if must_cover is not None and not (must_cover <= pts):
        return None
    return pts


def _aligned_positions(axis, A, cover, shape):
    A = int(A)
    if axis == "cols":
        left = A
        tops = set()
        for r_need, c_need in cover:
            for lr, lc in shape["rel"]:
                if c_need == left + lc:
                    tops.add(int(r_need - lr))
        for top in tops:
            yield int(top), int(left)
    else:
        top = A
        lefts = set()
        for r_need, c_need in cover:
            for lr, lc in shape["rel"]:
                if r_need == top + lr:
                    lefts.add(int(c_need - lc))
        for left in lefts:
            yield int(top), int(left)


def _touches(pts, other):
    for r, c in pts:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in other:
                return True
    return False


def _hole_positions(hole, shape):
    pos = set()
    for r_hole, c_hole in hole:
        for lr, lc in shape["rel"]:
            pos.add((int(r_hole - lr), int(c_hole - lc)))
    return pos


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
        hole_gl = {(int(bar_r0) + int(rr), int(bar_c0) + int(cc)) for rr, cc in pts}
        side, mouth_n = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )
        axis = "cols" if side in ("top", "bottom") else "rows"
        notches.append({"axis": axis, "hole": hole_gl, "mouth_n": int(mouth_n), "side": side})
    return notches


def _cavity_for_head(out_grid: np.ndarray, bg: int, head) -> set[tuple[int, int]]:
    top = int(head["top"])
    left = int(head["left"])
    h = int(head["shape"]["h"])
    w = int(head["shape"]["w"])
    return {
        (r, c)
        for r in range(top, top + h)
        for c in range(left, left + w)
        if int(out_grid[r, c]) == int(bg)
    }


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


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=3):
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
                A, B = map(int, span_range)
                if must_span_axis == "cols":
                    cm = min((c for _, c in cells), default=B)
                    cx = max((c for _, c in cells), default=A)
                    if cm > A:
                        preferred = [nb for nb in frontier if nb[1] < cm]
                        if preferred:
                            cells.add(rng.choice(preferred))
                            continue
                    if cx < B:
                        preferred = [nb for nb in frontier if nb[1] > cx]
                        if preferred:
                            cells.add(rng.choice(preferred))
                            continue
                else:
                    rm = min((r for r, _ in cells), default=B)
                    rx = max((r for r, _ in cells), default=A)
                    if rm > A:
                        preferred = [nb for nb in frontier if nb[0] < rm]
                        if preferred:
                            cells.add(rng.choice(preferred))
                            continue
                    if rx < B:
                        preferred = [nb for nb in frontier if nb[0] > rx]
                        if preferred:
                            cells.add(rng.choice(preferred))
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
                    A, B = map(int, span_range)
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
            A, B = map(int, span_range)
            if must_span_axis == "cols":
                if min(c for _, c in cells) != A or max(c for _, c in cells) != B:
                    continue
            else:
                if min(r for r, _ in cells) != A or max(r for r, _ in cells) != B:
                    continue
        if _connected(cells):
            return cells
    return None


def _root_match_exists(shape, hole, H, W, bar_occ) -> bool:
    for top, left in _hole_positions(hole, shape):
        pts = _placement_pts(shape, top, left, H, W, bar_occ, must_cover=hole)
        if pts is not None:
            return True
    return False


def _decoy_bbox_options(orientation: str, forbidden_strip_widths: set[int]) -> list[tuple[int, int]]:
    options = []
    if orientation == "horizontal":
        widths = [2, 3, 6, 7]
        for w in widths:
            if w in forbidden_strip_widths:
                continue
            for h in range(2, 8):
                if h * w >= 8:
                    options.append((h, w))
    else:
        heights = [2, 3, 6, 7]
        for h in heights:
            if h in forbidden_strip_widths:
                continue
            for w in range(2, 8):
                if h * w >= 8:
                    options.append((h, w))
    return options


def _scatter_piece(rng, grid, placed_occ, piece, orientation, bar_r0, bar_r1, bar_c0, bar_c1):
    H, W = map(int, grid.shape)
    coords = piece["coords"]
    color = int(piece["color"])
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    p_r0, p_c0 = min(rs), min(cs)
    rel = [(r - p_r0, c - p_c0) for r, c in coords]
    p_h = max(rs) - p_r0 + 1
    p_w = max(cs) - p_c0 + 1

    for _ in range(700):
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
        if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in DIR4):
            continue
        for r, c in pts:
            grid[r, c] = color
        placed_occ |= pts
        return True
    return False


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

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": _shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    notches = _bar_notches(bar_mask, bar_r0, bar_c0)

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            shape = piece["shape"]
            for top, left in _hole_positions(notch["hole"], shape):
                pts = _placement_pts(shape, top, left, H, W, bar_occ, must_cover=notch["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + shape["w"] - 1)
                else:
                    A, B = int(top), int(top + shape["h"] - 1)
                move_cost = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
                    (
                        (int(move_cost), -int(notch["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": shape},
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [item[1] for item in root_cands[:24]]

    best_out = out.copy()
    best_key = None

    def state_key(out_grid, chains, placed_cnt):
        open_sizes = []
        for ch in chains:
            cavity = _cavity_for_head(out_grid, bg, ch["head"])
            if cavity:
                open_sizes.append(len(cavity))
        complete_chains = len(chains) - len(open_sizes)
        return (int(complete_chains), -int(len(open_sizes)), int(placed_cnt), -int(sum(open_sizes)))

    def gen_extend_moves(out_grid, occ_set, chain, remaining_set):
        axis = chain["axis"]
        A, B = int(chain["A"]), int(chain["B"])
        head = chain["head"]
        head_pts = _pixels(head["shape"], head["top"], head["left"])
        cavity = _cavity_for_head(out_grid, bg, head)
        if not cavity:
            return []

        moves = []
        for pi in remaining_set:
            piece = pieces[pi]
            shape = piece["shape"]
            for top, left in _aligned_positions(axis, A, cavity, shape):
                pts = _placement_pts(shape, top, left, H, W, occ_set, must_cover=cavity, strip=(axis, A, B))
                if pts is None or not _touches(pts, head_pts):
                    continue
                move_cost = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                moves.append((int(move_cost), pi, int(top), int(left), pts, shape))
        moves.sort(key=lambda x: x[0])
        return moves[:8]

    def search(out_grid, occ_set, chains, remaining):
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = state_key(out_grid, chains, placed_cnt)
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()

        moves_found = False
        for chi, chain in enumerate(chains):
            for _cost, pi, top, left, pts, shape in gen_extend_moves(out_grid, occ_set, chain, remaining):
                moves_found = True
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = chain["head"]
                chain["head"] = {"top": int(top), "left": int(left), "shape": shape}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                search(out_grid, occ2, chains, rem2)
                chain["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
        return moves_found

    max_roots = min(2, len(cand_roots), len(notches), len(pieces))
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
                pts = _pixels(root["shape"], root["top"], root["left"])
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


def _gen_attempt():
    rng = random.Random()
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
    chain_notches = []
    notch_holes = []
    chain_infos = []
    occ_set = set(bar_cells)
    chain_axis = "cols" if orientation == "horizontal" else "rows"

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
        chain_notches.append(notch)

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
                    must_span_axis=chain_axis,
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

                if orientation == "horizontal":
                    ext_span = (max(0, prev_r0 - chain_dim), prev_r0 - 1) if side == "top" else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    ext = {
                        (r, c)
                        for r in range(ext_span[0], ext_span[1] + 1)
                        for c in range(sA, sB + 1)
                        if (r, c) not in occ_set
                    } if ext_span[0] <= ext_span[1] else set()
                else:
                    ext_span = (max(0, prev_c0 - chain_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    ext = {
                        (r, c)
                        for r in range(sA, sB + 1)
                        for c in range(ext_span[0], ext_span[1] + 1)
                        if (r, c) not in occ_set
                    } if ext_span[0] <= ext_span[1] else set()

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
                    must_span_axis=chain_axis,
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

        all_chain_pieces.extend(chain_pieces)
        chain_infos.append({"axis": chain_axis, "A": sA, "B": sB, "pieces": chain_pieces, "notch": notch, "side": side})

    if not (3 <= len(all_chain_pieces) <= 6):
        return None

    num_decoy_notches = rng.randint(4, 7)
    carved_decoys = 0
    for _ in range(500):
        if carved_decoys >= num_decoy_notches:
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
        d_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
        if not d_avail:
            continue
        area = min(area, len(d_avail))
        dn = _grow_connected(rng, d_avail, area, {rng.choice(list(d_avail))})
        if dn is None or not _connected(bar_cells - dn):
            continue
        bar_cells -= dn
        notch_holes.append(dn)
        carved_decoys += 1
    if carved_decoys < 4:
        return None

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr, cc = pts[:, 0], pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return None

    bar_occ = set(bar_cells)
    for info in chain_infos:
        for piece_idx, piece in enumerate(info["pieces"]):
            shape = _shape_from_cells(piece["coords"])
            matches = [hole for hole in notch_holes if _root_match_exists(shape, hole, H, W, bar_occ)]
            if piece_idx == 0:
                if len(matches) != 1 or matches[0] != info["notch"]:
                    return None
            elif matches:
                return None

    decoy_count = rng.randint(1, 2)
    strip_widths = {int(info["B"] - info["A"] + 1) for info in chain_infos}
    bbox_options = _decoy_bbox_options(orientation, strip_widths)
    if not bbox_options:
        return None

    full_occ = set(bar_occ)
    for piece in all_chain_pieces:
        full_occ |= set(piece["coords"])

    chain_sigs = {_shape_signature(_shape_from_cells(piece["coords"])) for piece in all_chain_pieces}
    decoy_pieces = []

    active_decoy = None
    shuffled_infos = list(chain_infos)
    rng.shuffle(shuffled_infos)
    for info in shuffled_infos:
        tail = info["pieces"][-1]
        tail_coords = set(tail["coords"])
        if len(tail_coords) <= 8:
            continue
        prev = info["pieces"][-2]
        prev_coords = set(prev["coords"])
        prev_r = [r for r, _ in prev_coords]
        prev_c = [c for _, c in prev_coords]
        prev_r0, prev_r1 = min(prev_r), max(prev_r)
        prev_c0, prev_c1 = min(prev_c), max(prev_c)

        if orientation == "horizontal":
            cavity = {(r, c) for r, c in tail_coords if prev_r0 <= r <= prev_r1 and info["A"] <= c <= info["B"]}
            edge = min(r for r, _ in tail_coords) if info["side"] == "top" else max(r for r, _ in tail_coords)
            step = -1 if info["side"] == "top" else 1
            candidates = [(edge + step, c) for c in range(info["A"], info["B"] + 1) if (edge, c) in tail_coords]
        else:
            cavity = {(r, c) for r, c in tail_coords if info["A"] <= r <= info["B"] and prev_c0 <= c <= prev_c1}
        candidates = [cell for cell in tail_coords if cell not in cavity and _connected(tail_coords - {cell})]

        if not cavity:
            continue
        rng.shuffle(candidates)
        for removed in candidates:
            decoy_abs = set(tail_coords) - {removed}
            shape = _shape_from_cells(decoy_abs)
            if _shape_signature(shape) in chain_sigs:
                continue
            if any(_root_match_exists(shape, hole, H, W, bar_occ) for hole in notch_holes):
                continue

            dec_r = [r for r, _ in decoy_abs]
            dec_c = [c for _, c in decoy_abs]
            dec_top, dec_left = min(dec_r), min(dec_c)
            occ_without_tail = full_occ - tail_coords
            pts = _placement_pts(
                shape,
                dec_top,
                dec_left,
                H,
                W,
                occ_without_tail,
                must_cover=cavity,
                strip=(info["axis"], info["A"], info["B"]),
            )
            if pts is None or pts != decoy_abs or not _touches(decoy_abs, prev_coords):
                continue

            if orientation == "horizontal":
                open_cavity = {
                    (r, c)
                    for r in range(min(dec_r), max(dec_r) + 1)
                    for c in range(info["A"], info["B"] + 1)
                    if (r, c) not in decoy_abs and (r, c) not in occ_without_tail
                }
            else:
                open_cavity = {
                    (r, c)
                    for r in range(info["A"], info["B"] + 1)
                    for c in range(min(dec_c), max(dec_c) + 1)
                    if (r, c) not in decoy_abs and (r, c) not in occ_without_tail
                }
            if not open_cavity:
                continue

            occ_after = occ_without_tail | decoy_abs
            continuation_exists = False
            for real_piece in all_chain_pieces:
                real_shape = _shape_from_cells(real_piece["coords"])
                for top2, left2 in _aligned_positions(info["axis"], info["A"], open_cavity, real_shape):
                    pts2 = _placement_pts(
                        real_shape,
                        top2,
                        left2,
                        H,
                        W,
                        occ_after,
                        must_cover=open_cavity,
                        strip=(info["axis"], info["A"], info["B"]),
                    )
                    if pts2 is not None and _touches(pts2, decoy_abs):
                        continuation_exists = True
                        break
                if continuation_exists:
                    break
            if continuation_exists:
                continue

            active_decoy = {"coords": set(shape["rel"]), "signature": _shape_signature(shape)}
            break
        if active_decoy is not None:
            break

    if active_decoy is None or color_idx >= len(avail_colors):
        return None
    active_decoy["color"] = avail_colors[color_idx]
    color_idx += 1
    chain_sigs.add(active_decoy["signature"])
    decoy_pieces.append({"color": active_decoy["color"], "coords": active_decoy["coords"]})

    passive_needed = decoy_count - 1
    for _ in range(passive_needed):
        if color_idx >= len(avail_colors):
            return None
        piece_color = avail_colors[color_idx]
        color_idx += 1
        accepted = None
        for __ in range(250):
            h, w = rng.choice(bbox_options)
            local_avail = {(r, c) for r in range(h) for c in range(w)}
            target = rng.randint(8, min(13, h * w))
            cells = _grow_connected(rng, local_avail, target, {rng.choice(list(local_avail))}, retries=6)
            if cells is None:
                continue
            shape = _shape_from_cells(cells)
            if _shape_signature(shape) in chain_sigs:
                continue
            if orientation == "horizontal" and int(shape["w"]) in strip_widths:
                continue
            if orientation == "vertical" and int(shape["h"]) in strip_widths:
                continue
            if any(_root_match_exists(shape, hole, H, W, bar_occ) for hole in notch_holes):
                continue
            accepted = cells
            chain_sigs.add(_shape_signature(shape))
            break
        if accepted is None:
            return None
        decoy_pieces.append({"color": piece_color, "coords": accepted})

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = bar_color
    for piece in all_chain_pieces:
        for r, c in piece["coords"]:
            output[r, c] = int(piece["color"])

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color

    placed_occ = set(bar_cells)
    pieces_order = list(all_chain_pieces) + decoy_pieces
    rng.shuffle(pieces_order)
    for piece in pieces_order:
        if not _scatter_piece(rng, input_grid, placed_occ, piece, orientation, bar_r0, bar_r1, bar_c0, bar_c1):
            return None

    return input_grid, output


def generate_input():
    for _ in range(3000):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 3000 attempts")
