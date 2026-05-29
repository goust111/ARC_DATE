# Variant of 16b78196
# Dimension: D1 — 匹配条件：形状嵌合 vs 其他匹配逻辑 (match_cond)
# Change: s.shape fits into d.notch.shape — 精灵形状恰好填充杆上缺口的形状... → 旋转/翻转后形状匹配

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

for _UTILS_DIR in (
    Path("/home/lixinyu/snap/new/solvers"),
    Path("/home/lixinyu/snap/new/arc2_solver_select"),
):
    if str(_UTILS_DIR) not in sys.path:
        sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg


# Design analysis for this D1 variant:
# a) Concrete behavior and trace:
#    The bar, notch count, strip width, and cavity-chaining logic stay the same as 16b78196.
#    The changed object is the scattered monochrome sprite: in the original task the input
#    sprite silhouette already had the exact orientation needed in output, while here the
#    scattered sprite may be rotated by 90/180/270 degrees and may also be mirrored.
#    Small trace:
#      original root match: a 3x5 L-like root piece above the bar already has the same
#      3x5 orientation as the notch+extension footprint, so the solver only translates it.
#      variant root match: the same piece may appear in input as a mirrored 5x3 silhouette;
#      the solver must test unique dihedral variants, discover that "flip + rot90" produces
#      the notch-covering 3x5 form, then place that transformed variant into the bar.
#    The same change propagates to continuation pieces because later chain links also match
#    by shape against the previous bbox cavity while keeping the chain strip width fixed.
# b) Complexity calibration:
#    Purely allowing arbitrary transforms without generator constraints would often collapse
#    back to the old task, because many random connected polyominoes are accidentally close
#    enough to identity placements and some are symmetric. To keep the D1 change visible:
#    - every used root piece is required to have at least one non-identity unique variant;
#    - every used root is scattered using a non-identity variant;
#    - many non-root pieces are also scattered with non-identity variants when possible.
#    This preserves a clear "puzzle piece must be reoriented before docking" signal while
#    keeping the search space controlled by deduplicating symmetric variants.
# c) Hidden assumptions audit against the original main():
#    - `pieces.append(..., "shape": ...)` assumed one fixed orientation per piece. That is
#      invalid now; it is replaced by `variants`, the unique dihedral orientations.
#    - Root search iterated `v = p["shape"]` and continuation search did the same. Both are
#      now wrong under the variant because a piece can match only after transform; both loops
#      must iterate over `p["variants"]`.
#    - The strip lock `A..B` was computed from the chosen root bbox. This assumption remains
#      valid, but only for the winning transformed root variant, so the chain head stores the
#      transformed shape that was actually placed.
#    - Candidate explosion becomes real after adding transforms. The original root shortlist
#      of 20 worked because each piece had one orientation; here we cap candidates per
#      (notch, piece) pair and deduplicate symmetric transforms, otherwise search would slow
#      down enough to destabilize generation.
# d) Placement legality:
#    Output-first construction remains unchanged, so all assembled chains are still legal by
#    construction: no overlap, in-bounds, fixed-width strip, cavity fill, and tail with zero
#    remaining cavity. Input scattering now uses transformed bboxes, but still enforces:
#    - piece cells stay in-bounds;
#    - pieces do not overlap or touch the bar/other pieces by 4-neighborhood;
#    - root pieces are visibly reoriented before scattering.
# e) Visual observability:
#    A human now sees asymmetrical colored sprites whose orientations in the input do not
#    match the notch/cavity footprints in the output. Across 2-4 examples the invariant is
#    visible: color identity stays the same, chain geometry stays the same, but silhouettes
#    must be turned/flipped before insertion.
# f) Visual cleanliness:
#    No extra markers or overlays are added. The variant reuses the existing sparse layout:
#    one large central bar and 3-6 distant sprites. Reorientation happens only at the level
#    of silhouette, so the grid remains easy to parse.
# g) Color inference:
#    No color remapping is introduced. Colors remain one-per-piece exactly as in the source
#    task, so the only new reasoning burden is geometric transform choice.
# h) Source-task baseline:
#    The requested reference solver `a25697e4` is available in
#    `/home/lixinyu/snap/new_code/solvers/arc2/a25697e4.py`. Its relevant mechanism is the
#    "enumerate unique dihedral variants and remember whether reflection was used" pattern.
#    This variant borrows that transform-enumeration baseline, but keeps the 16b78196 chain
#    assembly rule unchanged and does not import the reference task's color-swap behavior.
# i) New solver assumptions and generator guarantees:
#    - Assumption: each used root piece has at least one visible non-identity variant.
#      Guarantee: generator rejects samples where a root is too symmetric.
#    - Assumption: at least one correct transformed placement survives candidate pruning.
#      Guarantee: generator uses 1-2 chains, 3-6 pieces, and shape/cavity constraints that
#      keep each piece strongly tied to its chain while solver keeps multiple top candidates
#      per (notch, piece) pair.


_LAST_GEN_INFO: dict[str, object] = {}


def _shape_from_coords(coords: np.ndarray, bbox) -> dict[str, object]:
    r0, c0, r1, c1 = map(int, bbox)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return {
        "rel": rel,
        "rel_set": set(rel),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
    }


def _shape_from_cells(cells) -> dict[str, object]:
    coords = np.array(sorted((int(r), int(c)) for r, c in cells), dtype=int)
    rs = coords[:, 0]
    cs = coords[:, 1]
    bbox = (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))
    return _shape_from_coords(coords, bbox)


def _shape_variants(shape: dict[str, object]) -> tuple[list[dict[str, object]], tuple[tuple[int, int], ...]]:
    mask = np.zeros((int(shape["h"]), int(shape["w"])), dtype=bool)
    for r, c in shape["rel"]:
        mask[int(r), int(c)] = True

    base_key = tuple(shape["rel"])
    variants: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rot = np.rot90(mask, int(k))
        for flip in (False, True):
            arr = np.fliplr(rot) if flip else rot
            rel = tuple((int(r), int(c)) for r, c in np.argwhere(arr))
            if rel in seen:
                continue
            seen.add(rel)
            variants.append(
                {
                    "rel": list(rel),
                    "rel_set": set(rel),
                    "h": int(arr.shape[0]),
                    "w": int(arr.shape[1]),
                    "rot": int(k),
                    "flip": bool(flip),
                    "is_base": rel == base_key,
                }
            )
    return variants, base_key


def _transform_name(rot: int, flip: bool) -> str:
    return ("f" if flip else "r") + str(int(rot) * 90)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble colored sprites into a central bar's notches, chaining successive pieces along
    fixed-width strips. In this D1 variant, each scattered sprite may first need a rotation
    and/or mirror transform before it can fill the matching notch/cavity.
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

    pieces = []
    for p in pieces_raw:
        shape = _shape_from_coords(p["coords"], p["bbox"])
        variants, _base_key = _shape_variants(shape)
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append(
            {
                "color": int(p["color"]),
                "orig": (int(r0), int(c0)),
                "variants": variants,
            }
        )

    def bar_notches(mask: np.ndarray, r0: int, c0: int) -> list[dict[str, object]]:
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
            hole_gl = {(int(r0 + r), int(c0 + c)) for r, c in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": int(mouth_n)})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    def pixels(shape: dict[str, object], top: int, left: int) -> set[tuple[int, int]]:
        return {(int(top + r), int(left + c)) for r, c in shape["rel"]}

    def placement_pts(shape: dict[str, object], top: int, left: int, occ, must_cover=None, strip=None):
        if top < 0 or left < 0 or top + int(shape["h"]) > H or left + int(shape["w"]) > W:
            return None
        if strip is not None:
            axis, A, B = strip
            if axis == "cols":
                if left != A or int(shape["w"]) != B - A + 1:
                    return None
            else:
                if top != A or int(shape["h"]) != B - A + 1:
                    return None
        pts = pixels(shape, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        return pts

    def aligned_positions(axis: str, A: int, cover, shape: dict[str, object]):
        if axis == "cols":
            left = int(A)
            tops = set()
            for rN, cN in cover:
                for lr, lc in shape["rel"]:
                    if cN == left + lc:
                        tops.add(int(rN - lr))
            for top in tops:
                yield int(top), int(left)
        else:
            top = int(A)
            lefts = set()
            for rN, cN in cover:
                for lr, lc in shape["rel"]:
                    if rN == top + lr:
                        lefts.add(int(cN - lc))
            for left in lefts:
                yield int(top), int(left)

    def touches(pts, other) -> bool:
        for r, c in pts:
            for dr, dc in DIR4:
                if (int(r + dr), int(c + dc)) in other:
                    return True
        return False

    def hole_positions(hole, shape: dict[str, object]):
        pos = set()
        for rH, cH in hole:
            for lr, lc in shape["rel"]:
                pos.add((int(rH - lr), int(cH - lc)))
        return pos

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = str(notch["axis"])
        for pi, piece in enumerate(pieces):
            local = []
            for variant in piece["variants"]:
                for top, left in hole_positions(notch["hole"], variant):
                    pts = placement_pts(variant, top, left, bar_occ, must_cover=notch["hole"])
                    if pts is None:
                        continue
                    if axis == "cols":
                        A, B = int(left), int(left + int(variant["w"]) - 1)
                    else:
                        A, B = int(top), int(top + int(variant["h"]) - 1)
                    mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                    local.append(
                        (
                            (int(mv), -int(notch["mouth_n"])),
                            {
                                "ni": int(ni),
                                "pi": int(pi),
                                "top": int(top),
                                "left": int(left),
                                "axis": axis,
                                "A": int(A),
                                "B": int(B),
                                "shape": variant,
                            },
                        )
                    )
            local.sort(key=lambda x: x[0])
            root_cands.extend(local[:4])

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:32]]

    best_out = out.copy()
    best_key = None

    def search(out_grid: np.ndarray, occ_set, chains, remaining) -> bool:
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (int(placed_cnt), -int(len(remaining)), -int(len(occ_set)))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        def gen_extend_moves(chain, remaining_set):
            axis = str(chain["axis"])
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = pixels(head["shape"], int(head["top"]), int(head["left"]))
            head_top, head_left = int(head["top"]), int(head["left"])
            head_h, head_w = int(head["shape"]["h"]), int(head["shape"]["w"])

            cavity = {
                (int(r), int(c))
                for r in range(head_top, head_top + head_h)
                for c in range(head_left, head_left + head_w)
                if int(out_grid[r, c]) == int(bg)
            }
            if not cavity:
                return []

            moves = []
            cav_len = int(len(cavity))
            for pi in remaining_set:
                piece_moves = []
                for variant in pieces[pi]["variants"]:
                    for top, left in aligned_positions(axis, A, cavity, variant):
                        pts = placement_pts(variant, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                        if pts is None or not touches(pts, head_pts):
                            continue
                        mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                        ext = int(len(pts) - cav_len)
                        piece_moves.append((( -ext, int(mv)), (int(pi), int(top), int(left), pts, variant)))
                piece_moves.sort(key=lambda x: x[0])
                moves.extend(piece_moves[:3])
            moves.sort(key=lambda x: x[0])
            return [m[1] for m in moves[:8]]

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for pi, top, left, pts, variant in moves:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = ch["head"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": variant}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(int(pi))
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
        return False

    from itertools import combinations

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
                pts = pixels(root["shape"], int(root["top"]), int(root["left"]))
                if pts & occ0:
                    ok = False
                    break
                used_p.add(int(root["pi"]))
                used_n.add(int(root["ni"]))
                for r, c in pts:
                    out0[r, c] = int(pieces[root["pi"]]["color"])
                occ0 |= pts
                chains.append(
                    {
                        "axis": root["axis"],
                        "A": int(root["A"]),
                        "B": int(root["B"]),
                        "head": {"top": int(root["top"]), "left": int(root["left"]), "shape": root["shape"]},
                    }
                )

            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


def _connected(cells) -> bool:
    if len(cells) <= 1:
        return True
    cells_set = set(cells)
    start = next(iter(cells_set))
    visited = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (int(r + dr), int(c + dc))
            if nb in cells_set and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(cells_set)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=3):
    _D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
                for dr, dc in _D4
                if (nb := (int(r + dr), int(c + dc))) in available and nb not in cells
            ]
            if not frontier:
                break
            if must_span_axis and span_range:
                A, B = span_range
                if must_span_axis == "cols":
                    cm = min((c for _, c in cells), default=B)
                    cx = max((c for _, c in cells), default=A)
                    if cm > A:
                        left_opts = [nb for nb in frontier if nb[1] < cm]
                        if left_opts:
                            cells.add(rng.choice(left_opts))
                            continue
                    if cx < B:
                        right_opts = [nb for nb in frontier if nb[1] > cx]
                        if right_opts:
                            cells.add(rng.choice(right_opts))
                            continue
                else:
                    cm = min((r for r, _ in cells), default=B)
                    cx = max((r for r, _ in cells), default=A)
                    if cm > A:
                        up_opts = [nb for nb in frontier if nb[0] < cm]
                        if up_opts:
                            cells.add(rng.choice(up_opts))
                            continue
                    if cx < B:
                        down_opts = [nb for nb in frontier if nb[0] > cx]
                        if down_opts:
                            cells.add(rng.choice(down_opts))
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
    bar_cells = {
        (int(r), int(c))
        for r in range(bar_r0, bar_r1 + 1)
        for c in range(bar_c0, bar_c1 + 1)
    }

    num_chains = rng.randint(1, 2)
    chain_lens = [3] if num_chains == 1 else [rng.randint(2, 3) for _ in range(2)]
    if num_chains == 2:
        while not (3 <= sum(chain_lens) <= 6):
            chain_lens = [rng.randint(2, 3) for _ in range(2)]

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    color_idx = 0

    all_chain_pieces = []
    notch_holes: list[set[tuple[int, int]]] = []
    occ_set = set(bar_cells)

    for chain_idx in range(num_chains):
        chain_len = chain_lens[chain_idx]
        strip_width = rng.randint(4, 5)

        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next(
                (s for s in sides if (space_before if s == "top" else space_after) >= chain_len * 2 + 3),
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
                (s for s in sides if (space_before if s == "left" else space_after) >= chain_len * 2 + 3),
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
            n_avail = {
                (int(r), int(c))
                for r in range(nr0, nr1 + 1)
                for c in range(sc0, sc1 + 1)
                if (r, c) in bar_cells
            }
        else:
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            n_avail = {
                (int(r), int(c))
                for r in range(sA, sB + 1)
                for c in range(nc0, nc1 + 1)
                if (r, c) in bar_cells
            }

        all_holes = set().union(*notch_holes) if notch_holes else set()
        n_avail -= all_holes | {
            (int(r + dr), int(c + dc))
            for r, c in all_holes
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        }
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
                    (int(r), int(c))
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
                        (int(r), int(c))
                        for r in range(prev_r0, prev_r1 + 1)
                        for c in range(sc0, sc1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                else:
                    cavity = {
                        (int(r), int(c))
                        for r in range(sA, sB + 1)
                        for c in range(prev_c0, prev_c1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                if not cavity:
                    return None

                if orientation == "horizontal":
                    es = (max(0, prev_r0 - chain_dim), prev_r0 - 1) if side == "top" else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    ext = {
                        (int(r), int(c))
                        for r in range(es[0], es[1] + 1)
                        for c in range(sc0, sc1 + 1)
                        if (r, c) not in occ_set
                    } if es[0] <= es[1] else set()
                else:
                    es = (max(0, prev_c0 - chain_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    ext = {
                        (int(r), int(c))
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
                            (int(r), int(c))
                            for r in range(t_r0, t_r1 + 1)
                            for c in range(sc0, sc1 + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    else:
                        remaining = {
                            (int(r), int(c))
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
                    "color": int(piece_color),
                    "coords": set(piece_cells),
                    "is_root": bool(piece_idx == 0),
                    "chain_idx": int(chain_idx),
                }
            )
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

        d_avail = {
            (int(r), int(c))
            for r in range(dr0, dr1 + 1)
            for c in range(dc0, dc1 + 1)
            if (r, c) in bar_cells
        }
        all_h = set().union(*notch_holes) if notch_holes else set()
        d_avail -= all_h | {
            (int(r + dr), int(c + dc))
            for r, c in all_h
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        }
        if len(d_avail) < 1:
            continue
        area = min(area, len(d_avail))
        dn = _grow_connected(rng, d_avail, area, {rng.choice(list(d_avail))})
        if dn is None or not _connected(bar_cells - dn):
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
        rr = pts[:, 0]
        cc = pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return None

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = bar_color
    for piece in all_chain_pieces:
        for r, c in piece["coords"]:
            output[r, c] = int(piece["color"])

    piece_infos = []
    transformable_roots = 0
    for piece in all_chain_pieces:
        shape = _shape_from_cells(piece["coords"])
        variants, base_key = _shape_variants(shape)
        can_transform = any(tuple(v["rel"]) != base_key for v in variants)
        if piece["is_root"] and can_transform:
            transformable_roots += 1
        piece_infos.append(
            {
                **piece,
                "shape": shape,
                "variants": variants,
                "base_key": base_key,
                "can_transform": bool(can_transform),
            }
        )
    if transformable_roots != num_chains:
        return None

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color

    placed_occ = set(bar_cells)
    order = list(range(len(piece_infos)))
    rng.shuffle(order)

    transformed_piece_count = 0
    transformed_root_count = 0
    used_flip_count = 0
    transform_hist: Counter[str] = Counter()

    for idx in order:
        piece = piece_infos[idx]
        color = int(piece["color"])
        non_base = [v for v in piece["variants"] if tuple(v["rel"]) != piece["base_key"]]
        if piece["is_root"]:
            candidate_vars = non_base[:]
            rng.shuffle(candidate_vars)
        else:
            if non_base and rng.random() < 0.7:
                candidate_vars = non_base[:] + [v for v in piece["variants"] if tuple(v["rel"]) == piece["base_key"]]
            else:
                candidate_vars = piece["variants"][:]
            rng.shuffle(candidate_vars)

        placed = False
        for variant in candidate_vars:
            rel = variant["rel"]
            p_h, p_w = int(variant["h"]), int(variant["w"])
            for _ in range(180):
                gap = rng.randint(2, 7)
                if orientation == "horizontal":
                    top = (bar_r0 - gap - p_h) if rng.random() < 0.5 else (bar_r1 + gap + 1)
                    left = rng.randint(0, max(0, W - p_w))
                else:
                    left = (bar_c0 - gap - p_w) if rng.random() < 0.5 else (bar_c1 + gap + 1)
                    top = rng.randint(0, max(0, H - p_h))
                if top < 0 or top + p_h > H or left < 0 or left + p_w > W:
                    continue
                pts = {(int(top + dr), int(left + dc)) for dr, dc in rel}
                if pts & placed_occ:
                    continue
                if any(
                    (int(r + dr), int(c + dc)) in placed_occ
                    for r, c in pts
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                ):
                    continue
                for r, c in pts:
                    input_grid[r, c] = color
                placed_occ |= pts
                changed = tuple(rel) != piece["base_key"]
                if changed:
                    transformed_piece_count += 1
                    if piece["is_root"]:
                        transformed_root_count += 1
                if bool(variant["flip"]):
                    used_flip_count += 1
                transform_hist[_transform_name(int(variant["rot"]), bool(variant["flip"]))] += 1
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if transformed_root_count != num_chains or transformed_piece_count < num_chains:
        return None

    info = {
        "num_chains": int(num_chains),
        "piece_count": int(len(piece_infos)),
        "transformed_piece_count": int(transformed_piece_count),
        "transformed_root_count": int(transformed_root_count),
        "used_flip_count": int(used_flip_count),
        "transform_hist": dict(transform_hist),
        "transformable_root_count": int(transformable_roots),
    }
    return input_grid, output, info


def generate_input():
    global _LAST_GEN_INFO
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out, info = result
        if np.array_equal(main(inp), out):
            _LAST_GEN_INFO = dict(info)
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
