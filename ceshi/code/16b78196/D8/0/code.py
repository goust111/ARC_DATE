# Variant of 16b78196
# Dimension: D8 — 条带宽度约束 (meta)
# Change: 每条链所有精灵共享固定条带宽度（4-5格），条带轴与杆跨越轴一致
#         → 条带宽度逐块递增；链中后续每个精灵都在同一侧额外扩 1 格，形成单侧扇形展开

from __future__ import annotations

from itertools import combinations
import numpy as np

from arc_utils import components_mask, find_components, get_bg


# Design analysis for the D8 variant
# ----------------------------------
# spec_variant:
#   preserved:
#   - 30x30 grid, bg=0, one full-span bar, 1-2 outward chains, 3-6 total colored pieces.
#   - Pieces remain monochrome, 4-connected, and keep their original orientation.
#   - A root piece plugs a boundary notch; later pieces fill the previous piece's bg cavity.
#   changed_core_D8:
#   - The strip is no longer fixed after the root.
#   - Let the root width be 4 or 5. Piece k in a chain has width root_width + k.
#   - The widening is one-sided: each chain picks a lateral fan direction once, then every
#     later piece expands the strip by 1 on that same side.
#   concrete_geometry:
#   - Used notches are depth-1 mouths of width root_width-1. This keeps the widening visible.
#   - Root piece bbox has 3 cells along growth direction; its bar-facing row matches the notch.
#   - Every non-root piece also has 3 cells along growth direction. Inside its bbox the only bg
#     cells are on the outward tip row/column, so the next piece has a unique cavity to fill.
#   - Example (horizontal bar, chain grows upward, expands right):
#       root width 4 occupies:
#         row -2: ...###. becomes ...###? no: front row has only the right anchor cell
#         row -1: ####
#         row  0: ###. but the dot is bar, not bg, because row 0 is the notch mouth row
#       next piece width 5 covers the root's front-row cavity and adds one new column on the right.
#   complexity calibration:
#   - Widths are restricted to a tiny ladder: 4->5 or 5->6 or 4->5->6. This is enough to be
#     visible in 2-3 examples without creating too many match branches.
#   - Length-3 chains always start at width 4; otherwise the third piece would need width 7,
#     which would violate the 8-13 area constraint for these template pieces.
#   hidden-assumption audit of the original solver:
#   - Original root search stored a fixed [A,B] strip and every continuation reused it.
#     That assumption is invalid now; [A,B] must become per-chain mutable state.
#   - Original aligned_positions/placement_pts only needed a single fixed strip range.
#     Now each extension must try exactly one-step widened ranges.
#   - Original search never stored which side expansion should use. In this variant, once the
#     second piece is placed, that lateral direction becomes a chain invariant.
#   - The original generator's tail-zero-cavity condition is not solver-enforced. Under one-sided
#     widening it conflicts with non-overlap and the 8-13 area cap, so the variant keeps the
#     fill-previous-cavity rule but does not require the last piece to end cavity-free.
#   legality / cleanliness:
#   - Generator constructs output first, checks in-bounds/non-overlap at every placed piece,
#     then scatters the exact piece masks into the input without touching the bar or each other.
#   - Chains are templated rather than grown randomly, so widths and fan direction are visually
#     stable and the widening pattern is easy to read from a few examples.


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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
    """Grow a 4-connected shape of exactly *target* cells within *available*."""
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


def _piece_bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _shape_from_coords(coords: np.ndarray, bbox):
    r0, c0, r1, c1 = map(int, bbox)
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": r1 - r0 + 1, "w": c1 - c0 + 1}


def _span_len(shape, axis):
    return int(shape["w"] if axis == "cols" else shape["h"])


def _span_range(axis, top, left, shape):
    if axis == "cols":
        return int(left), int(left + shape["w"] - 1)
    return int(top), int(top + shape["h"] - 1)


def _pixels(shape, top, left):
    return {(int(top + lr), int(left + lc)) for lr, lc in shape["rel"]}


def _placement_pts(shape, top, left, H, W, occ, must_cover=None, strip=None):
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
        for rN, cN in cover:
            for lr, lc in shape["rel"]:
                if cN == left + lc:
                    tops.add(rN - lr)
        for top in tops:
            yield int(top), int(left)
    else:
        top = A
        lefts = set()
        for rN, cN in cover:
            for lr, lc in shape["rel"]:
                if rN == top + lr:
                    lefts.add(cN - lc)
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
    for rH, cH in hole:
        for lr, lc in shape["rel"]:
            pos.add((int(rH - lr), int(cH - lc)))
    return pos


def _interval_after_step(A, B, fan_dir):
    return (A - 1, B) if fan_dir < 0 else (A, B + 1)


def _build_template_piece(orientation, side, interval, anchor_curr, prev_anchor, back_coord):
    """Create one output piece mask for the widening-chain template."""
    A, B = map(int, interval)
    cells = set()

    if orientation == "horizontal":
        width_cols = list(range(A, B + 1))
        if side == "top":
            front_r, mid_r, back_r = back_coord - 2, back_coord - 1, back_coord
        else:
            back_r, mid_r, front_r = back_coord, back_coord + 1, back_coord + 2

        cells.add((front_r, anchor_curr))
        for c in width_cols:
            cells.add((mid_r, c))
        for c in width_cols:
            if c != prev_anchor:
                cells.add((back_r, c))
    else:
        width_rows = list(range(A, B + 1))
        if side == "left":
            front_c, mid_c, back_c = back_coord - 2, back_coord - 1, back_coord
        else:
            back_c, mid_c, front_c = back_coord, back_coord + 1, back_coord + 2

        cells.add((anchor_curr, front_c))
        for r in width_rows:
            cells.add((r, mid_c))
        for r in width_rows:
            if r != prev_anchor:
                cells.add((r, back_c))

    return cells


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

        side, mouth_n = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )
        axis = "cols" if side in ("top", "bottom") else "rows"
        hole_gl = {(bar_r0 + int(rr), bar_c0 + int(cc)) for rr, cc in pts}
        notches.append({"axis": axis, "side": side, "hole": hole_gl, "mouth_n": mouth_n})
    return notches


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
        shape = _shape_from_coords(p["coords"], p["bbox"])
        pieces.append({"color": int(p["color"]), "shape": shape, "orig": (r0, c0)})

    notches = _bar_notches(bar_mask, bar_r0, bar_c0)

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            shape = piece["shape"]
            root_width = _span_len(shape, axis)
            if root_width not in (4, 5):
                continue
            for top, left in _hole_positions(notch["hole"], shape):
                pts = _placement_pts(shape, top, left, H, W, bar_occ, must_cover=notch["hole"])
                if pts is None:
                    continue
                A, B = _span_range(axis, top, left, shape)
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), int(root_width), -int(notch["mouth_n"])),
                        {
                            "ni": ni,
                            "pi": pi,
                            "top": int(top),
                            "left": int(left),
                            "axis": axis,
                            "A": int(A),
                            "B": int(B),
                            "shape": shape,
                        },
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:30]]

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

        def candidate_strips(chain):
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            dirs = [-1, 1] if chain["fan_dir"] is None else [int(chain["fan_dir"])]
            out_ranges = []
            for d in dirs:
                nA, nB = _interval_after_step(A, B, d)
                if axis == "cols":
                    if 0 <= nA and nB < W:
                        out_ranges.append((d, nA, nB))
                else:
                    if 0 <= nA and nB < H:
                        out_ranges.append((d, nA, nB))
            return out_ranges

        def gen_extend_moves(chain, remaining_set):
            axis = chain["axis"]
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
            for fan_dir, nA, nB in candidate_strips(chain):
                expected_width = int(nB - nA + 1)
                for pi in remaining_set:
                    shape = pieces[pi]["shape"]
                    if _span_len(shape, axis) != expected_width:
                        continue
                    for top, left in _aligned_positions(axis, nA, cavity, shape):
                        pts = _placement_pts(shape, top, left, H, W, occ_set, must_cover=cavity, strip=(axis, nA, nB))
                        if pts is None:
                            continue
                        if not _touches(pts, head_pts):
                            continue
                        mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                        moves.append((mv, pi, int(top), int(left), pts, shape, int(fan_dir), int(nA), int(nB)))
            moves.sort(key=lambda x: x[0])
            return moves[:10]

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for _, pi, top, left, pts, shape, fan_dir, nA, nB in moves:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_state = (ch["A"], ch["B"], ch["fan_dir"], ch["head"])
                ch["A"], ch["B"], ch["fan_dir"] = int(nA), int(nB), int(fan_dir)
                ch["head"] = {"top": int(top), "left": int(left), "shape": shape}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["A"], ch["B"], ch["fan_dir"], ch["head"] = old_state
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
            # Try other chains only if this one truly cannot extend.
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
                        "fan_dir": None,
                        "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                    }
                )
            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


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

    num_chains = rng.randint(1, 2)
    chain_lens = [3] if num_chains == 1 else [rng.randint(2, 3) for _ in range(2)]
    if num_chains == 2:
        while not (3 <= sum(chain_lens) <= 6):
            chain_lens = [rng.randint(2, 3) for _ in range(2)]

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    color_idx = 0

    all_chain_pieces = []
    notch_holes = []
    occ_set = set(bar_cells)

    for chain_len in chain_lens:
        base_width = 4 if chain_len == 3 else rng.choice([4, 5])
        final_width = base_width + chain_len - 1
        fan_dir = rng.choice([-1, 1])

        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            need_space = 2 * chain_len
            side = next(
                (s for s in sides if (space_before if s == "top" else space_after) >= need_space),
                None,
            )
            if side is None:
                return None

            min_start = 1 + (chain_len - 1 if fan_dir < 0 else 0)
            max_start = (W - 1 - final_width) - (0 if fan_dir < 0 else 0)
            if max_start < min_start:
                return None
            start = rng.randint(min_start, max_start)
            A0, B0 = start, start + base_width - 1
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            need_space = 2 * chain_len
            side = next(
                (s for s in sides if (space_before if s == "left" else space_after) >= need_space),
                None,
            )
            if side is None:
                return None

            min_start = 1 + (chain_len - 1 if fan_dir < 0 else 0)
            max_start = (H - 1 - final_width) - (0 if fan_dir < 0 else 0)
            if max_start < min_start:
                return None
            start = rng.randint(min_start, max_start)
            A0, B0 = start, start + base_width - 1

        anchor0 = B0 if fan_dir > 0 else A0

        if orientation == "horizontal":
            notch_row = bar_r0 if side == "top" else bar_r1
            notch = {(notch_row, c) for c in range(A0, B0 + 1) if c != anchor0}
        else:
            notch_col = bar_c0 if side == "left" else bar_c1
            notch = {(r, notch_col) for r in range(A0, B0 + 1) if r != anchor0}

        all_holes = set().union(*notch_holes) if notch_holes else set()
        if notch & (all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}):
            return None
        if not notch.issubset(bar_cells):
            return None

        bar_cells -= notch
        occ_set -= notch
        notch_holes.append(set(notch))

        interval = (A0, B0)
        prev_anchor = anchor0
        back_coord = bar_r0 if orientation == "horizontal" and side == "top" else None
        if orientation == "horizontal" and side == "bottom":
            back_coord = bar_r1
        if orientation == "vertical" and side == "left":
            back_coord = bar_c0
        if orientation == "vertical" and side == "right":
            back_coord = bar_c1

        for piece_idx in range(chain_len):
            if color_idx >= len(avail_colors):
                return None
            piece_color = avail_colors[color_idx]
            color_idx += 1

            A, B = interval
            anchor_curr = B if fan_dir > 0 else A
            piece_cells = _build_template_piece(
                orientation=orientation,
                side=side,
                interval=interval,
                anchor_curr=anchor_curr,
                prev_anchor=prev_anchor,
                back_coord=back_coord,
            )
            if len(piece_cells) < 8 or len(piece_cells) > 13:
                return None
            if any(not (0 <= r < H and 0 <= c < W) for r, c in piece_cells):
                return None
            if piece_cells & occ_set:
                return None
            if not _connected(piece_cells):
                return None

            all_chain_pieces.append({"color": piece_color, "coords": set(piece_cells)})
            occ_set |= piece_cells

            # Step outward: next piece uses the current front row/col as its overlap row/col.
            if orientation == "horizontal":
                front = min(r for r, _ in piece_cells) if side == "top" else max(r for r, _ in piece_cells)
            else:
                front = min(c for _, c in piece_cells) if side == "left" else max(c for _, c in piece_cells)

            back_coord = int(front)
            prev_anchor = int(anchor_curr)
            interval = _interval_after_step(A, B, fan_dir)

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
        all_holes = set().union(*notch_holes) if notch_holes else set()
        d_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
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
        p_r0, p_c0, p_r1, p_c1 = _piece_bbox(coords)
        rel = [(r - p_r0, c - p_c0) for r, c in coords]
        p_h, p_w = p_r1 - p_r0 + 1, p_c1 - p_c0 + 1

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
