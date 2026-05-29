# Variant of 16b78196
# Dimension: D13 — 链终止条件 (meta)
# Change: 尾部精灵的bbox在条带内无剩余空腔时链终止... → 链到达网格边界时终止

from __future__ import annotations

import numpy as np

from arc_utils import components_mask, find_components, get_bg


# Design analysis
# ----------------
# spec:
#   unchanged_core:
#     - 30x30 grid, bg=0.
#     - largest monochrome 4-connected object is a full-span bar with boundary notches.
#     - scattered monochrome pieces are reassembled into one outward chain on a fixed-width strip.
#     - pieces keep their original orientation; no rotation/flip.
#   D13_variant:
#     - original solver stops extending a chain when the current head bbox has no bg cavity.
#     - new rule keeps the old "cover the previous cavity when it exists" mechanism, but if the
#       head has zero cavity and still has not reached the grid edge, the next piece may attach
#       flush on the outward side of the same strip and continue the chain.
#     - termination is therefore spatial: stop only when the head piece touches the outer boundary
#       on the growth side.
#   concrete_visualization:
#     - the generated chain always contains a root with a small outer-side cavity, then a solid
#       filler piece that removes that cavity, and then more solid rectangles that keep growing.
#     - at least one non-tail piece has zero cavity and is not on the boundary, so the sample
#       visibly depends on the new termination rule instead of the old one.
#   old_vs_new_trace:
#     - old: root has cavity -> piece 2 fills it; piece 2 is solid -> search stops because cavity
#       is empty, even if there is free strip space above/below/left/right.
#     - new: after piece 2 becomes the head, cavity is empty but boundary is not reached, so piece
#       3 is allowed to attach outward on the same strip; the chain continues until a piece touches
#       the grid border.
#   complexity_calibration:
#     - only one used chain is generated, but 5-7 decoy notches remain; ambiguity still comes from
#       root-notch matching rather than from multiple independent rule families.
#     - strip width is fixed at 4 for used chains so 2-cell and 3-cell outward blocks stay within
#       the original 8-13 area range while letting boundary distance control chain length.
#   implicit-assumption_audit:
#     - original `gen_extend_moves` assumed `if not cavity: return []`; this is no longer valid.
#     - new search must know the growth direction (`dir`) and whether the head already touches the
#       relevant boundary; both are stored on the chain state from the chosen notch side.
#     - candidate extensions now require monotone outward progress in addition to strip alignment
#       and 4-neighbor contact, otherwise an empty-cavity head could accept sideward/backward noise.
#   placement_legality:
#     - generator constructs output first from exact strip depths, so pieces never overlap, always
#       stay in-bounds, and the last piece provably touches the boundary.
#     - used notch cells are derived from the root's inner slice, guaranteeing exact hole coverage.
#     - decoy notches are kept smaller than the used notch mouth and separated by at least one cell.
#   observability_and_cleanliness:
#     - the continued growth after a solid middle piece is visually obvious in 2-3 examples.
#     - no invisible color ordering is used; colors remain arbitrary labels only.
#     - the chain is a clean single strip, and decoy holes stay on the bar rather than inside the
#       same region as the chain.
#   main_new_assumptions_checked_against_generator:
#     - every real chain uses one boundary-opening notch and exactly one outward direction.
#     - every continuation piece either covers the current head cavity or is a flush outward block.
#     - every piece in the chain spans the full strip width in bbox terms, so strip locking remains
#       sufficient for reconstruction.


def main(grid: np.ndarray) -> np.ndarray:
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

    def shape_from_coords(coords: np.ndarray, bbox):
        r0, c0, r1, c1 = map(int, bbox)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    def bar_notches(mask, base_r0, base_c0):
        empty = ~mask
        h, w = empty.shape
        notches = []
        for pts in components_mask(empty, connectivity=4):
            r = pts[:, 0]
            c = pts[:, 1]
            top = int((r == 0).sum())
            bottom = int((r == h - 1).sum())
            left = int((c == 0).sum())
            right = int((c == w - 1).sum())
            sides = [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0]
            if not sides:
                continue
            side, mouth_n = max(sides, key=lambda x: x[1])
            axis = "cols" if side in ("top", "bottom") else "rows"
            grow_dir = -1 if side in ("top", "left") else 1
            hole_gl = {(base_r0 + int(rr), base_c0 + int(cc)) for rr, cc in pts}
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n, "side": side, "dir": grow_dir})
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

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, v):
        pos = set()
        for r_h, c_h in hole:
            for lr, lc in v["rel"]:
                pos.add((r_h - lr, c_h - lc))
        return pos

    def head_bbox(head):
        top = int(head["top"])
        left = int(head["left"])
        return top, left, top + int(head["shape"]["h"]) - 1, left + int(head["shape"]["w"]) - 1

    def head_cavity(head, out_grid):
        r0, c0, r1, c1 = head_bbox(head)
        return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if out_grid[r, c] == bg}

    def touches_growth_boundary(chain, head):
        pts = pixels(head["shape"], head["top"], head["left"])
        if chain["axis"] == "cols":
            row_needed = 0 if chain["dir"] < 0 else H - 1
            return any(r == row_needed for r, _ in pts)
        col_needed = 0 if chain["dir"] < 0 else W - 1
        return any(c == col_needed for _, c in pts)

    def extends_outward(chain, head, pts):
        r0, c0, r1, c1 = head_bbox(head)
        pr = [r for r, _ in pts]
        pc = [c for _, c in pts]
        if chain["axis"] == "cols":
            p0, p1 = min(pr), max(pr)
            if chain["dir"] < 0:
                return p0 < r0 and p1 <= r1
            return p1 > r1 and p0 >= r0
        p0, p1 = min(pc), max(pc)
        if chain["dir"] < 0:
            return p0 < c0 and p1 <= c1
        return p1 > c1 and p0 >= c0

    def all_positions_in_strip(axis, A, v):
        if axis == "cols":
            left = A
            for top in range(H - v["h"] + 1):
                yield top, left
        else:
            top = A
            for left in range(W - v["w"] + 1):
                yield top, left

    root_cands = []
    for ni, n in enumerate(notches):
        for pi, p in enumerate(pieces):
            v = p["shape"]
            for top, left in hole_positions(n["hole"], v):
                pts = placement_pts(v, top, left, bar_occ, must_cover=n["hole"])
                if pts is None:
                    continue
                if n["axis"] == "cols":
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
                            "axis": n["axis"],
                            "A": A,
                            "B": B,
                            "shape": v,
                            "dir": int(n["dir"]),
                        },
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
            head = chain["head"]
            if touches_growth_boundary(chain, head):
                return []
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            head_pts = pixels(head["shape"], head["top"], head["left"])
            cavity = head_cavity(head, out_grid)
            moves = []
            for pi in remaining_set:
                v = pieces[pi]["shape"]
                for top, left in all_positions_in_strip(axis, A, v):
                    pts = placement_pts(v, top, left, occ_set, must_cover=cavity if cavity else None, strip=(axis, A, B))
                    if pts is None:
                        continue
                    if not touches(pts, head_pts):
                        continue
                    if not extends_outward(chain, head, pts):
                        continue
                    progress = min(r for r, _ in pts) if axis == "cols" and chain["dir"] < 0 else None
                    if axis == "cols" and chain["dir"] > 0:
                        progress = -max(r for r, _ in pts)
                    if axis == "rows" and chain["dir"] < 0:
                        progress = min(c for _, c in pts)
                    if axis == "rows" and chain["dir"] > 0:
                        progress = -max(c for _, c in pts)
                    moves.append((progress, pi, top, left, pts, v))
            moves.sort(key=lambda x: x[0])
            return [m[1:] for m in moves]

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
                chains.append(
                    {
                        "axis": root["axis"],
                        "A": root["A"],
                        "B": root["B"],
                        "dir": root["dir"],
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


def _grow_connected(rng, available, target, seeds, retries=3):
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
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue
        if _connected(cells):
            return cells
    return None


def _project_chain_cell(side, boundary_rc, strip_coord, depth):
    boundary, other = boundary_rc
    if side == "top":
        return boundary - depth, strip_coord
    if side == "bottom":
        return boundary + depth, strip_coord
    if side == "left":
        return strip_coord, boundary - depth
    return strip_coord, boundary + depth


def _make_chain_pieces(rng, H, W, orientation, side, strip_start, strip_width, bar_r0, bar_r1, bar_c0, bar_c1):
    if strip_width != 4:
        return None
    if side == "top":
        space = bar_r0
        boundary = bar_r0
        strip_coords = list(range(strip_start, strip_start + strip_width))
    elif side == "bottom":
        space = H - 1 - bar_r1
        boundary = bar_r1
        strip_coords = list(range(strip_start, strip_start + strip_width))
    elif side == "left":
        space = bar_c0
        boundary = bar_c0
        strip_coords = list(range(strip_start, strip_start + strip_width))
    else:
        space = W - 1 - bar_c1
        boundary = bar_c1
        strip_coords = list(range(strip_start, strip_start + strip_width))

    if space < 6:
        return None

    pieces = []
    notch = set()

    connector_len = rng.choice([1, 2])
    connector_start = rng.randint(0, strip_width - connector_len)
    connector_coords = strip_coords[connector_start : connector_start + connector_len]

    root = set()
    for s in strip_coords:
        root.add(_project_chain_cell(side, (boundary, 0), s, 0))
        root.add(_project_chain_cell(side, (boundary, 0), s, 1))
        notch.add(_project_chain_cell(side, (boundary, 0), s, 0))
    for s in connector_coords:
        root.add(_project_chain_cell(side, (boundary, 0), s, -1))
        notch.add(_project_chain_cell(side, (boundary, 0), s, -1))
    for s in connector_coords:
        root.add(_project_chain_cell(side, (boundary, 0), s, 2))
    pieces.append(root)

    filler = {_project_chain_cell(side, (boundary, 0), s, d) for s in strip_coords for d in (3, 4)}
    for s in strip_coords:
        if s not in connector_coords:
            filler.add(_project_chain_cell(side, (boundary, 0), s, 2))
    pieces.append(filler)

    remaining_depth = space - 4
    spine_col = strip_coords[rng.choice([1, 2])]
    final = {_project_chain_cell(side, (boundary, 0), spine_col, d) for d in range(5, space + 1)}
    for s in strip_coords:
        final.add(_project_chain_cell(side, (boundary, 0), s, 5))
    if remaining_depth <= 4:
        for s in strip_coords:
            final.add(_project_chain_cell(side, (boundary, 0), s, space))
    pieces.append(final)

    for i, cells in enumerate(pieces):
        if not (8 <= len(cells) <= 13):
            return None
        if not _connected(cells):
            return None
        if i > 0 and not any(
            (r + dr, c + dc) in pieces[i - 1]
            for r, c in cells
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ):
            return None

    if side in ("top", "bottom"):
        if any(not (0 <= r < H and 0 <= c < W) for cells in pieces for r, c in cells):
            return None
    else:
        if any(not (0 <= r < H and 0 <= c < W) for cells in pieces for r, c in cells):
            return None

    tail = pieces[-1]
    if side == "top" and not any(r == 0 for r, _ in tail):
        return None
    if side == "bottom" and not any(r == H - 1 for r, _ in tail):
        return None
    if side == "left" and not any(c == 0 for _, c in tail):
        return None
    if side == "right" and not any(c == W - 1 for _, c in tail):
        return None

    return {"pieces": pieces, "notch": notch}


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
        valid_sides = [s for s in ["top", "bottom"] if (bar_r0 if s == "top" else H - 1 - bar_r1) >= 6]
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = bar_c0 + thickness - 1
        if bar_c1 >= W:
            bar_c1 = W - 1
            bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        valid_sides = [s for s in ["left", "right"] if (bar_c0 if s == "left" else W - 1 - bar_c1) >= 6]

    if not valid_sides:
        return None

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    strip_width = 4
    side = rng.choice(valid_sides)
    if orientation == "horizontal":
        strip_start = rng.randint(1, W - strip_width - 2)
    else:
        strip_start = rng.randint(1, H - strip_width - 2)

    chain = _make_chain_pieces(rng, H, W, orientation, side, strip_start, strip_width, bar_r0, bar_r1, bar_c0, bar_c1)
    if chain is None:
        return None

    notch_holes = [set(chain["notch"])]
    if chain["notch"] - bar_cells:
        return None
    bar_cells -= chain["notch"]

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    if len(chain["pieces"]) > len(avail_colors):
        return None

    all_chain_pieces = []
    occ_set = set(bar_cells)
    for idx, cells in enumerate(chain["pieces"]):
        if cells & occ_set:
            return None
        occ_set |= cells
        all_chain_pieces.append({"color": avail_colors[idx], "coords": set(cells)})

    if not (3 <= len(all_chain_pieces) <= 6):
        return None

    num_decoys = rng.randint(5, 7)
    decoys = 0
    for _ in range(800):
        if decoys >= num_decoys:
            break
        depth = rng.randint(1, 2)
        span = rng.randint(1, 3)
        if orientation == "horizontal":
            ds = rng.choice(["top", "bottom"])
            dr0 = bar_r0 if ds == "top" else bar_r1 - depth + 1
            dr1 = bar_r0 + depth - 1 if ds == "top" else bar_r1
            dc0 = rng.randint(bar_c0, bar_c1 - span + 1)
            dc1 = dc0 + span - 1
        else:
            ds = rng.choice(["left", "right"])
            dc0 = bar_c0 if ds == "left" else bar_c1 - depth + 1
            dc1 = bar_c0 + depth - 1 if ds == "left" else bar_c1
            dr0 = rng.randint(bar_r0, bar_r1 - span + 1)
            dr1 = dr0 + span - 1

        d_avail = {(r, c) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        all_holes = set().union(*notch_holes)
        halo = {(r + dr, c + dc) for r, c in all_holes for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
        d_avail -= all_holes | halo
        if not d_avail:
            continue
        area = rng.randint(1, min(len(d_avail), depth * span))
        dn = _grow_connected(rng, d_avail, area, {rng.choice(list(d_avail))}, retries=5)
        if dn is None:
            continue
        if not _connected(bar_cells - dn):
            continue
        bar_cells -= dn
        notch_holes.append(dn)
        decoys += 1
    if decoys < 5:
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
        for _ in range(800):
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
