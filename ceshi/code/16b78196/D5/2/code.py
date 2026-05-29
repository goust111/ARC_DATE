# Variant of 16b78196
# Dimension: D5 — 变换：放置后的颜色或形状变化 (transform)
# Change: 无变换——精灵保持原色原形直接放置到目标位置... → 根据链中位置渐变着色
#
# Design analysis
# ===============
# Source-task note:
# - The prompt cites 4c416de3 for color_transfer, but within the allowed local files there is no readable
#   solver source for 4c416de3, only its evaluation JSON. So this variant uses the requested color-transfer
#   idea directly, while keeping the base task's geometry/mechanics unchanged.
#
# a) Concrete behavior and trace
# - The bar/notch/chaining geometry is the same as the original task: scattered monochrome sprites are
#   inserted into true bar notches, each root locks a strip, and later sprites fill the cavity inside the
#   previous sprite's bbox while extending farther from the bar.
# - D5 now changes only the color written into the assembled output. Each chain piece still matches and
#   places by its original monochrome shape. The root (depth 0) keeps its own color. A continuation at
#   depth 1 is recolored to the "next" legend color, and a depth-2 tail is recolored to the second-next
#   legend color.
# - Because numeric palette order is invisible to humans, the input/output also contain a small visible
#   legend made of tiny 2-cell dominoes. Their left-to-right or top-to-bottom order defines the color
#   progression used by recoloring. Example: if the legend order is [red, teal, yellow, green, blue, pink],
#   an input sprite colored teal stays teal when used as a root, becomes yellow when used as the second
#   piece of a chain, and becomes green when used as the third piece.
# - Flow difference at the write step:
#   original: place piece P at chain depth d -> output uses P.color
#   variant:  place piece P at chain depth d -> output uses legend_shift(P.color, d)
#
# b) Complexity calibration
# - The variant keeps 2 independent geometric signals: notch-fitting and cavity-filling still determine
#   which shape belongs where, while the legend determines only recoloring.
# - The legend has 6 colors: enough to support 4-5 chain pieces with repeated evidence for "shift by 1"
#   and occasional evidence for "shift by 2", without flooding the board with too many tiny markers.
# - Total chain pieces are restricted to 4-5, typically from chain lengths [2,2], [2,3], or [3,2]. This
#   keeps the color rule visible in nearly every sample and avoids degenerate one-chain cases with too
#   little repetition.
#
# c) Implicit-assumption audit of the original main()
# - `pieces[pi]["color"]` was written directly for roots and extensions. That relied on "placed color ==
#   source color". It is no longer valid; the search state must track chain depth and transform the color
#   at write time.
# - `remaining` pieces are still selected only by shape/placement, so the original matching logic remains
#   valid. The recolor rule does not affect occupancy or cavity geometry.
# - Undo logic originally restored placed cells to background and then restored the bar. With the legend
#   added, guides must also be preserved in the immutable base occupancy, so placements must treat legend
#   cells as blocked from the start.
# - Component parsing originally treated every non-bar component as a candidate piece. That becomes unsafe
#   after adding the legend; solver must ignore tiny guide components and only treat area-8..13 components
#   as movable sprites.
#
# d) Placement legality
# - The generator still builds output first, then scatters input pieces, then validates with `main(inp)`.
# - Legend cells are reserved as occupied both in output assembly and input scattering.
# - If a chain grows on the same side as the legend, the chain strip is forbidden to overlap the legend's
#   projected span, preventing later chain growth from colliding with the guide.
# - Input scattering also blocks a buffer around the legend so the tiny markers remain visually separate.
#
# e) Visual observability
# - The color rule is inferable from 2-4 examples because the legend is visible and unchanged between input
#   and output, while placed chain pieces demonstrably step forward along that legend as their depth grows.
# - No invisible ordering based on raw ARC color ids is used anywhere. The only ordering source is the
#   visible spatial order of the legend markers.
#
# f) Visual cleanliness
# - The legend uses tiny dominoes (area 2), while movable pieces remain area 8-13. This size gap prevents
#   confusion about which objects participate in chaining.
# - The legend is anchored to a corner-side margin with a reserved buffer, so it does not visually merge
#   with the bar or scattered sprites.
#
# g) Color-change inferability
# - Input piece colors are chosen from the legend colors only.
# - The root staying unchanged gives the "shift by 0" case directly.
# - Every sample includes at least two depth-1 pieces, and many include a depth-2 piece, so the rule has
#   repeated support rather than a single isolated transformed object.
#
# 9) New assumptions introduced by the variant and how the generator guarantees them
# - Assumption: a visible legend exists and is composed of tiny area-2 components.
#   Guarantee: generator always inserts 6 guide dominoes and never uses area<8 for real sprites.
# - Assumption: every movable sprite color is present in the legend and can be shifted by its chain depth.
#   Guarantee: generator assigns piece colors only from legend slots with enough room for +depth.
# - Assumption: chain growth never overlaps the legend.
#   Guarantee: generator reserves the legend footprint and rejects strips on the legend side that overlap
#   the legend span.

from __future__ import annotations

import sys
from pathlib import Path
from itertools import combinations

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, find_components, get_bg

D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LEGEND_LEN = 6
PIECE_MIN_AREA = 8
PIECE_MAX_AREA = 13


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


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
    visited = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in D4:
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
            frontier = [
                nb
                for r, c in cells
                for dr, dc in D4
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
                        cand = [nb for nb in frontier if nb[1] < cm]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if cx < B:
                        cand = [nb for nb in frontier if nb[1] > cx]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                else:
                    cm = min((r for r, _ in cells), default=B)
                    cx = max((r for r, _ in cells), default=A)
                    if cm > A:
                        cand = [nb for nb in frontier if nb[0] < cm]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if cx < B:
                        cand = [nb for nb in frontier if nb[0] > cx]
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


def _legend_components(comps):
    guides = [c for c in comps if int(c["area"]) <= 2]
    if len(guides) < 3:
        return guides, []
    coords = []
    for g in guides:
        r0, c0, r1, c1 = map(int, g["bbox"])
        coords.append(((r0 + r1) / 2.0, (c0 + c1) / 2.0))
    row_span = max(r for r, _ in coords) - min(r for r, _ in coords)
    col_span = max(c for _, c in coords) - min(c for _, c in coords)
    if col_span >= row_span:
        guides.sort(key=lambda d: (int(d["bbox"][1]), int(d["bbox"][0])))
    else:
        guides.sort(key=lambda d: (int(d["bbox"][0]), int(d["bbox"][1])))
    palette = [int(g["color"]) for g in guides]
    return guides, palette


def _legend_layout(H, W, orientation):
    if orientation == "horizontal":
        row = 1
        doms = []
        for i in range(LEGEND_LEN):
            c0 = 1 + 3 * i
            doms.append({(row, c0), (row, c0 + 1)})
        r0, c0, r1, c1 = 0, 0, row + 1, 1 + 3 * (LEGEND_LEN - 1) + 2
        side = "top"
        span = (1, 1 + 3 * (LEGEND_LEN - 1) + 1)
    else:
        col = 1
        doms = []
        for i in range(LEGEND_LEN):
            r0d = 1 + 3 * i
            doms.append({(r0d, col), (r0d + 1, col)})
        r0, c0, r1, c1 = 0, 0, 1 + 3 * (LEGEND_LEN - 1) + 2, col + 1
        side = "left"
        span = (1, 1 + 3 * (LEGEND_LEN - 1) + 1)

    occ = set().union(*doms)
    buffer = {
        (r, c)
        for r in range(max(0, r0), min(H - 1, r1) + 1)
        for c in range(max(0, c0), min(W - 1, c1) + 1)
    }
    return {"dominoes": doms, "occ": occ, "buffer": buffer, "side": side, "span": span}


def _intervals_overlap(a0, a1, b0, b1):
    return not (a1 < b0 or b1 < a0)


def _assign_piece_colors(rng, chains, palette):
    slots = []
    for ci, chain in enumerate(chains):
        for pi, piece in enumerate(chain):
            slots.append((ci, pi, int(piece["depth"])))
    order = sorted(range(len(slots)), key=lambda idx: (-slots[idx][2], idx))
    used_in = set()
    used_out = set()
    assign = {}

    def backtrack(k):
        if k == len(order):
            return True
        slot = slots[order[k]]
        ci, pi, depth = slot
        candidates = [idx for idx in range(len(palette) - depth) if idx not in used_in and idx + depth not in used_out]
        rng.shuffle(candidates)
        for idx in candidates:
            used_in.add(idx)
            used_out.add(idx + depth)
            assign[(ci, pi)] = idx
            if backtrack(k + 1):
                return True
            used_in.remove(idx)
            used_out.remove(idx + depth)
            del assign[(ci, pi)]
        return False

    if not backtrack(0):
        return None

    for ci, chain in enumerate(chains):
        for pi, piece in enumerate(chain):
            piece["color"] = int(palette[assign[(ci, pi)]])
    return chains


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -int(d["area"]))
    assert comps

    bar = comps[0]
    guides, palette = _legend_components(comps[1:])
    assert len(palette) >= 3
    palette_index = {color: idx for idx, color in enumerate(palette)}

    pieces_raw = [c for c in comps[1:] if PIECE_MIN_AREA <= int(c["area"]) <= PIECE_MAX_AREA]
    guide_occ = {(int(r), int(c)) for guide in guides for r, c in guide["coords"]}

    out = np.full((H, W), int(bg), dtype=int)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    base_occ = set(bar_occ) | set(guide_occ)
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color
    for guide in guides:
        for r, c in guide["coords"]:
            out[int(r), int(c)] = int(guide["color"])

    def transform_color(color, depth):
        idx = palette_index[int(color)]
        assert idx + depth < len(palette)
        return int(palette[idx + depth])

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": _shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    def bar_notches(bar_mask_local, top0, left0):
        empty = ~bar_mask_local
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
            hole_gl = {(top0 + int(rr), left0 + int(cc)) for rr, cc in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

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

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in D4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, shape):
        pos = set()
        for rH, cH in hole:
            for lr, lc in shape["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            shape = piece["shape"]
            for top, left in hole_positions(notch["hole"], shape):
                pts = placement_pts(shape, top, left, base_occ, must_cover=notch["hole"])
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
                color = int(pieces[pi]["color"])
                if color not in palette_index or palette_index[color] + chain["depth"] + 1 >= len(palette):
                    continue
                for top, left in aligned_positions(axis, A, cavity, shape):
                    pts = placement_pts(shape, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                    if pts is None:
                        continue
                    if not touches(pts, head_pts):
                        continue
                    moves.append((pi, top, left, pts, shape))
            return moves

        for chain in chains:
            moves = gen_extend_moves(chain, remaining)
            if not moves:
                continue
            for pi, top, left, pts, shape in moves[:8]:
                new_depth = int(chain["depth"]) + 1
                color = transform_color(pieces[pi]["color"], new_depth)
                for r, c in pts:
                    out_grid[r, c] = color
                old_head = chain["head"]
                old_depth = chain["depth"]
                chain["head"] = {"top": int(top), "left": int(left), "shape": shape}
                chain["depth"] = new_depth
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                chain["head"] = old_head
                chain["depth"] = old_depth
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
                for guide in guides:
                    for r, c in guide["coords"]:
                        out_grid[int(r), int(c)] = int(guide["color"])
        return False

    for k in range(1, min(3, len(cand_roots)) + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(base_occ)
            out0 = out.copy()
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
                color = transform_color(pieces[root["pi"]]["color"], 0)
                for r, c in pts:
                    out0[r, c] = color
                occ0 |= pts
                chains.append(
                    {
                        "axis": root["axis"],
                        "A": root["A"],
                        "B": root["B"],
                        "depth": 0,
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

    legend = _legend_layout(H, W, orientation)
    legend_colors_all = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(legend_colors_all)
    palette = legend_colors_all[:LEGEND_LEN]

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    num_chains = 2
    chain_lens = rng.choice([[2, 2], [2, 3], [3, 2]])

    all_chain_pieces = []
    all_chains = []
    notch_holes = []
    occ_set = set(bar_cells) | set(legend["occ"])

    for chain_idx in range(num_chains):
        chain_len = chain_lens[chain_idx]
        strip_width = rng.randint(4, 5)

        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next(
                (
                    s
                    for s in sides
                    if (space_before if s == "top" else space_after) >= chain_len * 2 + 3
                ),
                None,
            )
            if side is None:
                return None
            for _ in range(50):
                sc0 = rng.randint(1, W - strip_width - 1)
                sc1 = sc0 + strip_width - 1
                if side == legend["side"] and _intervals_overlap(sc0, sc1, legend["span"][0] - 1, legend["span"][1] + 1):
                    continue
                break
            else:
                return None
            sA, sB = sc0, sc1
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            side = next(
                (
                    s
                    for s in sides
                    if (space_before if s == "left" else space_after) >= chain_len * 2 + 3
                ),
                None,
            )
            if side is None:
                return None
            for _ in range(50):
                sr0 = rng.randint(1, H - strip_width - 1)
                sr1 = sr0 + strip_width - 1
                if side == legend["side"] and _intervals_overlap(sr0, sr1, legend["span"][0] - 1, legend["span"][1] + 1):
                    continue
                break
            else:
                return None
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
        n_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in D4}
        if len(n_avail) < notch_area:
            return None

        notch = _grow_connected(rng, n_avail, notch_area, {rng.choice(list(n_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        notch_holes.append(notch)

        chain_pieces = []
        for piece_idx in range(chain_len):
            piece_area = rng.randint(PIECE_MIN_AREA, PIECE_MAX_AREA)
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
                target = max(max(PIECE_MIN_AREA, int(bbox_area * 0.54)), min(piece_area, int(bbox_area * 0.80)))
                if target > bbox_area or target < PIECE_MIN_AREA:
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
                    ext = {
                        (r, c)
                        for r in range(es[0], es[1] + 1)
                        for c in range(sc0, sc1 + 1)
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
                target = min(PIECE_MAX_AREA, max(len(cavity), min(piece_area, len(all_avail))))
                if len(cavity) > PIECE_MAX_AREA or target < PIECE_MIN_AREA:
                    return None
                is_tail = piece_idx == chain_len - 1
                if is_tail:
                    target = min(PIECE_MAX_AREA, len(all_avail))

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

            if not (PIECE_MIN_AREA <= len(piece_cells) <= PIECE_MAX_AREA):
                return None
            chain_pieces.append({"coords": piece_cells, "depth": piece_idx})
            occ_set |= piece_cells

        all_chains.append(chain_pieces)
        all_chain_pieces.extend(chain_pieces)

    if not (4 <= len(all_chain_pieces) <= 5):
        return None

    if _assign_piece_colors(rng, all_chains, palette) is None:
        return None
    all_chain_pieces = [piece for chain in all_chains for piece in chain]

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
        d_avail -= all_h | {(r + dr, c + dc) for r, c in all_h for dr, dc in D4}
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
    for color, dom in zip(palette, legend["dominoes"]):
        for r, c in dom:
            output[r, c] = int(color)
    palette_index = {color: idx for idx, color in enumerate(palette)}
    for piece in all_chain_pieces:
        out_color = palette[palette_index[piece["color"]] + piece["depth"]]
        for r, c in piece["coords"]:
            output[r, c] = int(out_color)

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color
    for color, dom in zip(palette, legend["dominoes"]):
        for r, c in dom:
            input_grid[r, c] = int(color)

    placed_occ = set(bar_cells) | set(legend["buffer"])
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
            if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in D4):
                continue
            for r, c in pts:
                input_grid[r, c] = int(color)
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
