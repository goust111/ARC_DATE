# Variant of 16b78196
# Dimension: D5 — 变换：放置后的颜色或形状变化 (transform)
# Change: 无变换——精灵保持原色原形直接放置到目标位置... → 精灵间颜色交换
#
# Design analysis for this variant
# 1. Concrete behavior:
#    - Geometry stays the same as the original task: colored sprites are scattered in the input,
#      then assembled into notch-rooted outward chains on fixed-width strips.
#    - The D5 change happens only after the geometric assembly is known.
#    - We make the swap rule explicit and visually recoverable:
#      there are always exactly two parallel chains, on the same side of the bar, with equal
#      depth 2 or 3. After assembly, depth-0 pieces across the two chains swap colors, depth-1
#      pieces swap colors, and so on.
#    - Example: if the left chain uses colors [2, 4, 6] from root to tail and the right chain
#      uses [3, 5, 7], the assembled geometry remains on the same strips, but the output colors
#      become left=[3, 5, 7], right=[2, 4, 6].
#
# 2. Where the original logic changes:
#    - Original flow: infer placement -> paint each placed sprite with its own input color.
#    - Variant flow: infer placement -> group pieces by chain and depth -> repaint with partner
#      chain's color at the same depth.
#    - Therefore the search should solve geometry first and defer final coloring.
#
# 3. Complexity calibration:
#    - A single chain would make "color swap" invisible, so the generator always creates 2 chains.
#    - Unequal chain lengths would create unmatched pieces and weaken the rule, so both chains have
#      the same depth. Depth is still varied between 2 and 3 to keep samples non-trivial.
#    - The two chains are adjacent parallel strips, which gives a strong human-visible cue for
#      pairing without adding arbitrary mapping freedom.
#
# 4. Implicit-assumption audit of the original solver:
#    - The original painting step `out_grid[r, c] = pieces[pi]["color"]` assumed geometry and color
#      identity were the same object. That is no longer true; the placed shape's final color now
#      depends on another chain.
#    - Cavity detection in the original solver only relied on bg-vs-non-bg occupancy. That remains
#      valid, so the search can still paint temporary colors during backtracking.
#    - Root/extension matching never used color equality, only shape/coverage/strip constraints.
#      Those assumptions still hold unchanged.
#    - The new post-processing assumes the chain order is meaningful. The generator guarantees this
#      by creating exactly two adjacent chains on the same side of the bar and with equal depth.
#
# 5. Placement legality:
#    - The generator remains output-first. It constructs valid chains in assembled space first,
#      then scatters the original-color sprites so they do not touch the bar or each other.
#    - Used notches are non-overlapping, chains stay in-bounds, and extension pieces still fill the
#      previous bbox cavity exactly within the fixed strip.
#
# 6. Visual observability / cleanliness:
#    - The swap is visible because every sample contains two neighboring chains and every depth pair
#      changes color. No hidden numeric ordering on color IDs is used.
#    - Colors are distinct across all sprites, so the swap cannot collapse into a no-op.
#    - Keeping the chains parallel and near each other makes the swapped correspondence legible from
#      a few examples rather than looking like arbitrary recoloring.
#
# 7. Reference task note:
#    - The requested source solver was read from
#      `/home/lixinyu/snap/new_code/solvers/arc2/a25697e4.py`.
#    - The relevant baseline idea used here is the same separation of concerns:
#      first determine the geometric docking/placement, then apply a color transform that depends on
#      the winning placement relation rather than on the input sprite alone.

from __future__ import annotations

import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/arc2_solver_select"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import components_mask, find_components, get_bg


def _shape_from_coords(coords: np.ndarray, bbox):
    r0, c0, r1, c1 = map(int, bbox)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "h": h, "w": w}


def _connected(cells):
    """Check 4-connectivity of a cell set."""
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
    """Grow a 4-connected shape of exactly *target* cells within *available*, starting from *seeds*."""
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


def _paint_swapped_output(H, W, bg, bar_cells, bar_color, chains):
    """Paint bar plus placed chains, swapping colors between adjacent chains at each depth."""
    out = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_cells:
        out[r, c] = int(bar_color)

    ordered = sorted(chains, key=lambda ch: int(ch["A"]))
    max_depth = max((len(ch["placements"]) for ch in ordered), default=0)
    piece_to_color = {}
    for depth in range(max_depth):
        present = []
        colors = []
        for idx, ch in enumerate(ordered):
            if depth < len(ch["placements"]):
                pl = ch["placements"][depth]
                present.append((idx, pl))
                colors.append(int(pl["color"]))
        swapped = list(colors)
        for i in range(0, len(swapped) - 1, 2):
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        for (_, pl), col in zip(present, swapped):
            piece_to_color[(id(pl), depth)] = int(col)

    for ch in ordered:
        for depth, pl in enumerate(ch["placements"]):
            col = int(piece_to_color[(id(pl), depth)])
            for r, c in pl["pts"]:
                out[r, c] = col
    return out


def _paint_plain_output(H, W, bg, bar_cells, bar_color, chains):
    """Paint the assembled geometry without the D5 swap, for generator-side checks."""
    out = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_cells:
        out[r, c] = int(bar_color)
    for ch in chains:
        for pl in ch["placements"]:
            for r, c in pl["pts"]:
                out[r, c] = int(pl["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered sprites into two parallel chains rooted in bar notches, then swap colors
    between adjacent chains depth-by-depth.
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

    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": _shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    def bar_notches(mask, base_r0, base_c0):
        empty = ~mask
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
            hole_gl = {(base_r0 + int(rr), base_c0 + int(cc)) for rr, cc in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "side": side, "hole": hole_gl, "mouth_n": mouth_n})
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

    def hole_positions(hole, v):
        pos = set()
        for rH, cH in hole:
            for lr, lc in v["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    root_cands = []
    for ni, n in enumerate(notches):
        axis = n["axis"]
        for pi, p in enumerate(pieces):
            v = p["shape"]
            for top, left in hole_positions(n["hole"], v):
                pts = placement_pts(v, top, left, bar_occ, must_cover=n["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + v["w"] - 1)
                else:
                    A, B = int(top), int(top + v["h"] - 1)
                mv = abs(int(top) - int(p["orig"][0])) + abs(int(left) - int(p["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), -int(n["mouth_n"])),
                        {
                            "ni": ni,
                            "pi": pi,
                            "top": int(top),
                            "left": int(left),
                            "axis": axis,
                            "side": n["side"],
                            "A": A,
                            "B": B,
                            "shape": v,
                        },
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:24]]

    def snapshot_chains(chains):
        snap = []
        for ch in chains:
            snap.append(
                {
                    "axis": ch["axis"],
                    "side": ch["side"],
                    "A": ch["A"],
                    "B": ch["B"],
                    "placements": [
                        {"pi": pl["pi"], "color": int(pl["color"]), "pts": set(pl["pts"])}
                        for pl in ch["placements"]
                    ],
                }
            )
        return snap

    solved_snapshot = None
    best_snapshot = None
    best_key = None

    def search(out_grid, occ_set, chains, remaining):
        nonlocal solved_snapshot, best_snapshot, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (placed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_snapshot = snapshot_chains(chains)
        if not remaining:
            solved_snapshot = snapshot_chains(chains)
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
                    moves.append((mv, pi, top, left, pts, v))
            moves.sort(key=lambda x: x[0])
            return moves

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for _, pi, top, left, pts, v in moves[:8]:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = ch["head"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": v}
                ch["placements"].append({"pi": pi, "color": int(pieces[pi]["color"]), "pts": set(pts)})
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["placements"].pop()
                ch["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
                for ch2 in chains:
                    for pl in ch2["placements"]:
                        for rr, cc in pl["pts"]:
                            out_grid[rr, cc] = int(pl["color"])
        return False

    from itertools import combinations

    # The variant generator always creates exactly two used notches/chains, so the solver first
    # searches the matching 2-root geometry. The 1-root fallback only exists for degenerate inputs.
    root_sizes = [2] if len(cand_roots) >= 2 else [1]
    for k in root_sizes:
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(bar_occ)
            out0 = np.full((H, W), int(bg), dtype=int)
            for r, c in bar_occ:
                out0[r, c] = bar_color

            chains = []
            ok = True
            root_sides = set()
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
                root_sides.add(root["side"])
                for r, c in pts:
                    out0[r, c] = int(pieces[root["pi"]]["color"])
                occ0 |= pts
                chains.append(
                    {
                        "axis": root["axis"],
                        "side": root["side"],
                        "A": root["A"],
                        "B": root["B"],
                        "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                        "placements": [{"pi": root["pi"], "color": int(pieces[root["pi"]]["color"]), "pts": set(pts)}],
                    }
                )

            if not ok:
                continue
            if len(chains) == 2 and len(root_sides) != 1:
                continue
            remaining = set(range(len(pieces))) - used_p
            if search(out0, occ0, chains, remaining):
                return _paint_swapped_output(H, W, bg, bar_occ, bar_color, solved_snapshot)

    snapshot = solved_snapshot if solved_snapshot is not None else best_snapshot
    if snapshot is not None:
        return _paint_swapped_output(H, W, bg, bar_occ, bar_color, snapshot)
    out = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_occ:
        out[r, c] = bar_color
    return out


def _gen_attempt():
    """Build one (input, output) pair for the D5 color-swap variant."""
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
    chain_len = rng.choice([2, 3, 3])
    chain_lens = [chain_len, chain_len]
    strip_width = rng.randint(4, 5)

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    if len(avail_colors) < num_chains * chain_len:
        return None

    if orientation == "horizontal":
        sides = ["top", "bottom"]
        rng.shuffle(sides)
        side = next((s for s in sides if (space_before if s == "top" else space_after) >= chain_len * 2 + 3), None)
        if side is None:
            return None
        gap = rng.randint(1, 2)
        total_w = 2 * strip_width + gap
        if total_w > W - 2:
            return None
        sc0_a = rng.randint(1, W - total_w - 1)
        sc0_b = sc0_a + strip_width + gap
        strip_ranges = [(sc0_a, sc0_a + strip_width - 1), (sc0_b, sc0_b + strip_width - 1)]
    else:
        sides = ["left", "right"]
        rng.shuffle(sides)
        side = next((s for s in sides if (space_before if s == "left" else space_after) >= chain_len * 2 + 3), None)
        if side is None:
            return None
        gap = rng.randint(1, 2)
        total_h = 2 * strip_width + gap
        if total_h > H - 2:
            return None
        sr0_a = rng.randint(1, H - total_h - 1)
        sr0_b = sr0_a + strip_width + gap
        strip_ranges = [(sr0_a, sr0_a + strip_width - 1), (sr0_b, sr0_b + strip_width - 1)]

    all_chain_pieces = []
    chains_data = []
    notch_holes = []
    occ_set = set(bar_cells)
    color_idx = 0

    for chain_idx in range(num_chains):
        chain_len = chain_lens[chain_idx]
        sA, sB = strip_ranges[chain_idx]

        notch_depth = rng.randint(1, 2)
        notch_area = rng.randint(1, min(4, notch_depth * strip_width))

        if orientation == "horizontal":
            sc0, sc1 = sA, sB
            nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
            nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
            n_avail = {(r, c) for r in range(nr0, nr1 + 1) for c in range(sc0, sc1 + 1) if (r, c) in bar_cells}
        else:
            sr0, sr1 = sA, sB
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            n_avail = {(r, c) for r in range(sr0, sr1 + 1) for c in range(nc0, nc1 + 1) if (r, c) in bar_cells}

        all_holes = set().union(*notch_holes) if notch_holes else set()
        n_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
        if len(n_avail) < notch_area:
            return None

        notch = _grow_connected(rng, n_avail, notch_area, {rng.choice(list(n_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        notch_holes.append(notch)

        chain_pieces = []
        for piece_idx in range(chain_len):
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
                    es = (max(0, prev_r0 - chain_dim), prev_r0 - 1) if side == "top" else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    ext = {
                        (r, c)
                        for r in range(es[0], es[1] + 1)
                        for c in range(sA, sB + 1)
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
            piece = {"color": piece_color, "coords": piece_cells}
            chain_pieces.append(piece)
            occ_set |= piece_cells

        chains_data.append({"A": sA, "B": sB, "placements": [{"color": p["color"], "pts": set(p["coords"])} for p in chain_pieces]})
        all_chain_pieces.extend(chain_pieces)

    if len(all_chain_pieces) not in (4, 6):
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
        d_avail -= all_h | {(r + dr, c + dc) for r, c in all_h for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
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

    plain_output = _paint_plain_output(H, W, bg, bar_cells, bar_color, chains_data)
    output = _paint_swapped_output(H, W, bg, bar_cells, bar_color, chains_data)
    if np.array_equal(plain_output, output):
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
    """Generate a valid input grid for the D5 variant (max 300 retries)."""
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
