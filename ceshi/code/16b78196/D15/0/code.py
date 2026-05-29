# Variant of 16b78196
# Dimension: D15 — 符号指令：是否包含图例或提示标记 (symbol_instructions)
# Change: 无图例或符号指令——所有信息从几何关系推断... → 颜色图例指示链顺序

from __future__ import annotations

import numpy as np

from arc_utils import components_mask, find_components, get_bg

# Design analysis for the D15 variant
# -----------------------------------
# Reference-task note:
# - The prompt names source task `58490d8a`, but only its evaluation JSON is available
#   in the allowed paths; there is no solver file for that task in this workspace.
# - I therefore use the observable mechanism from that JSON as the baseline idea:
#   small explicit color markers act as a legend that tells the solver how to interpret
#   the larger geometric objects.
#
# Variant spec (concreteized)
# - The original task has 1-2 chains of scattered monochrome pieces that must be plugged
#   into bar notches and then extended outward by cavity filling.
# - In this variant, each chain also has an explicit legend segment on the outer border:
#   horizontal bar:
#     top-chain legend    = a short left-to-right color run on row 0
#     bottom-chain legend = a short left-to-right color run on row H-1
#   vertical bar:
#     left-chain legend   = a short top-to-bottom color run on col 0
#     right-chain legend  = a short top-to-bottom color run on col W-1
# - The legend length equals the chain length (2-3). Its colors are exactly the piece
#   colors of that chain in assembly order: legend[0] = root piece that plugs the notch,
#   legend[1] = next piece, legend[2] = tail piece.
# - Output does NOT contain the legend. The legend is an input-only instruction layer.
#
# Behavior trace on one horizontal example
# - Original logic:
#   1. detect bar holes
#   2. try many candidate root pieces for many holes
#   3. recursively test which remaining piece can fill the current cavity
# - Variant logic:
#   1. detect bar holes
#   2. detect the border legend run, e.g. row 0 has colors [6, 1, 2]
#   3. interpret that as "for the top-side chain, piece color 6 is root, then 1, then 2"
#   4. search only placements of color 6 on top-opening notches whose strip contains the
#      legend segment, then extend only with color 1, then only with color 2
# - Difference:
#   the old solver inferred chain order from geometry and search; the new solver reads
#   order from an explicit symbol sequence and uses geometry only for placement.
#
# Complexity calibration
# - Legend markers are 1-cell monochrome components on the outer border, while real
#   pieces remain area 8-13 and never touch the border. This keeps parsing simple and
#   visually obvious.
# - Each chain has only one legend run of 2-3 colors, so the new symbolic degree of
#   freedom is visible in every sample but not so large that examples become noisy.
# - Piece colors remain globally unique across the puzzle, so the legend unambiguously
#   names one specific piece at each step.
#
# Hidden-assumption audit against the original `main()`
# - `pieces_raw = comps[1:]` originally assumed every non-bar component is a real piece.
#   This is no longer valid because border singleton legend pixels are also components.
#   Fix: classify border singletons as legend markers and exclude them from piece search.
# - Root search originally iterated over every piece for every notch. In the variant that
#   would ignore the explicit order cue. Fix: build root candidates from legend[0] only.
# - Extension search originally tried any remaining piece. In the variant that would allow
#   solutions inconsistent with the legend. Fix: for each chain step use only the next
#   legend color.
# - Original search had no notion of side-specific symbolic attachment. Fix: a top/bottom
#   or left/right legend may only match a notch on the same outward side, and the placed
#   strip must contain the legend segment projected onto the bar span axis.
#
# Placement legality guarantees required from the generator
# - Legends must stay on the outer border and be separated from each other on the same
#   side by at least one background gap, so the solver can split them into runs.
# - Scattered pieces must stay strictly off the border; otherwise a 1-cell border-touching
#   piece fragment could be confused with a legend marker.
# - Legend colors must be exactly the chain colors, in order, and each piece color must
#   appear in exactly one legend run.
# - Root notches and chain strips must remain disjoint enough that legend-side + strip
#   containment identifies the intended chain without visual clutter.
#
# Visual observability / cleanliness
# - The bar and large scattered pieces still dominate the scene, but the border legend
#   adds a clearly visible new signal: a short ordered palette strip on the outer frame.
# - Because legends are on the border and real pieces are not, the instruction layer does
#   not overlap the object layer.
# - The output removing the legend while assembling pieces makes the instruction role
#   visible across examples.
#
# New implicit assumptions introduced by this variant solver
# - Every chain has one contiguous border legend run of length 2-3.
# - Each run belongs to exactly one side and does not share cells with another run.
# - Real pieces have unique colors and area > 1.
# - The generator below enforces all of these assumptions directly.


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    if not comps:
        return g.copy()

    bar = comps[0]
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}

    out = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_occ:
        out[r, c] = bar_color

    def shape_from_coords(coords: np.ndarray, bbox):
        r0, c0, r1, c1 = map(int, bbox)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}

    def on_border(r, c):
        return r == 0 or c == 0 or r == H - 1 or c == W - 1

    legend_cells = {}
    pieces_raw = []
    for comp in comps[1:]:
        area = int(comp["area"])
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        if area == 1 and any(on_border(r, c) for r, c in coords):
            r, c = coords[0]
            legend_cells[(r, c)] = int(comp["color"])
        else:
            pieces_raw.append(comp)

    pieces_by_color = {}
    for p in pieces_raw:
        color = int(p["color"])
        r0, c0 = map(int, p["bbox"][:2])
        pieces_by_color[color] = {
            "color": color,
            "shape": shape_from_coords(p["coords"], p["bbox"]),
            "orig": (r0, c0),
        }

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True

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
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append(
                {
                    "axis": axis,
                    "side": side,
                    "hole": {(r0 + int(r), c0 + int(c)) for r, c in pts},
                    "mouth_n": mouth_n,
                }
            )
        return notches

    def split_runs(vals):
        if not vals:
            return []
        runs = [[vals[0]]]
        for v in vals[1:]:
            if v == runs[-1][-1] + 1:
                runs[-1].append(v)
            else:
                runs.append([v])
        return runs

    def parse_legends():
        legends = []

        top_cols = sorted(c for (r, c) in legend_cells if r == 0)
        for run in split_runs(top_cols):
            legends.append({"side": "top", "span": (run[0], run[-1]), "colors": [legend_cells[(0, c)] for c in run]})

        bottom_cols = sorted(c for (r, c) in legend_cells if r == H - 1)
        for run in split_runs(bottom_cols):
            legends.append({"side": "bottom", "span": (run[0], run[-1]), "colors": [legend_cells[(H - 1, c)] for c in run]})

        left_rows = sorted(r for (r, c) in legend_cells if c == 0)
        for run in split_runs(left_rows):
            legends.append({"side": "left", "span": (run[0], run[-1]), "colors": [legend_cells[(r, 0)] for r in run]})

        right_rows = sorted(r for (r, c) in legend_cells if c == W - 1)
        for run in split_runs(right_rows):
            legends.append({"side": "right", "span": (run[0], run[-1]), "colors": [legend_cells[(r, W - 1)] for r in run]})

        legends = [d for d in legends if 2 <= len(d["colors"]) <= 3]
        legends.sort(key=lambda d: (["top", "bottom", "left", "right"].index(d["side"]), d["span"][0]))
        return legends

    legends = parse_legends()
    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    if not legends:
        return out

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

    def root_candidates(legend):
        colors = legend["colors"]
        if colors[0] not in pieces_by_color:
            return []
        piece = pieces_by_color[colors[0]]
        v = piece["shape"]
        lo, hi = legend["span"]
        opts = []
        for ni, n in enumerate(notches):
            if n["side"] != legend["side"]:
                continue
            for top, left in hole_positions(n["hole"], v):
                pts = placement_pts(v, top, left, bar_occ, must_cover=n["hole"])
                if pts is None:
                    continue
                if n["axis"] == "cols":
                    A, B = int(left), int(left + v["w"] - 1)
                else:
                    A, B = int(top), int(top + v["h"] - 1)
                if not (A <= lo and hi <= B):
                    continue
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                opts.append(
                    (
                        (int(mv), -int(n["mouth_n"])),
                        {
                            "ni": ni,
                            "top": int(top),
                            "left": int(left),
                            "axis": n["axis"],
                            "A": A,
                            "B": B,
                            "shape": v,
                            "root_color": colors[0],
                            "colors": colors,
                        },
                    )
                )
        opts.sort(key=lambda x: x[0])
        return [x[1] for x in opts[:12]]

    legend_specs = []
    seen_legend_colors = []
    for legend in legends:
        seen_legend_colors.extend(legend["colors"])
        legend_specs.append({"legend": legend, "roots": root_candidates(legend)})

    if sorted(seen_legend_colors) != sorted(pieces_by_color):
        return out
    if any(not spec["roots"] for spec in legend_specs):
        return out

    legend_specs.sort(key=lambda d: len(d["roots"]))
    best_out = out.copy()
    best_score = -1

    def maybe_record(out_grid, chains):
        nonlocal best_out, best_score
        placed = sum(chain["next_idx"] for chain in chains)
        if placed > best_score:
            best_score = placed
            best_out = out_grid.copy()

    def gen_moves(chain, out_grid, occ_set):
        if chain["next_idx"] >= len(chain["colors"]):
            return []
        color = chain["colors"][chain["next_idx"]]
        piece = pieces_by_color[color]
        v = piece["shape"]
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
        for top, left in aligned_positions(axis, A, cavity, v):
            pts = placement_pts(v, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
            if pts is None:
                continue
            if not touches(pts, head_pts):
                continue
            moves.append((int(top), int(left), pts, v, color))
        return moves

    def search_extensions(out_grid, occ_set, chains):
        maybe_record(out_grid, chains)
        done = True
        best_ci = None
        best_moves = None
        for ci, chain in enumerate(chains):
            if chain["next_idx"] >= len(chain["colors"]):
                continue
            done = False
            moves = gen_moves(chain, out_grid, occ_set)
            if not moves:
                return False
            if best_moves is None or len(moves) < len(best_moves):
                best_ci = ci
                best_moves = moves
        if done:
            return True

        chain = chains[best_ci]
        for top, left, pts, v, color in best_moves[:8]:
            for r, c in pts:
                out_grid[r, c] = color
            old_head = chain["head"]
            old_idx = chain["next_idx"]
            chain["head"] = {"top": int(top), "left": int(left), "shape": v}
            chain["next_idx"] += 1
            if search_extensions(out_grid, occ_set | set(pts), chains):
                return True
            chain["head"] = old_head
            chain["next_idx"] = old_idx
            for r, c in pts:
                out_grid[r, c] = int(bg)
            for r, c in bar_occ:
                out_grid[r, c] = bar_color
        return False

    def place_roots(idx, used_notches, out_grid, occ_set, chains):
        maybe_record(out_grid, chains)
        if idx == len(legend_specs):
            return search_extensions(out_grid, occ_set, chains)

        for root in legend_specs[idx]["roots"]:
            if root["ni"] in used_notches:
                continue
            pts = pixels(root["shape"], root["top"], root["left"])
            if pts & occ_set:
                continue
            for r, c in pts:
                out_grid[r, c] = root["root_color"]
            chains.append(
                {
                    "axis": root["axis"],
                    "A": root["A"],
                    "B": root["B"],
                    "colors": root["colors"],
                    "next_idx": 1,
                    "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                }
            )
            used_notches.add(root["ni"])
            if place_roots(idx + 1, used_notches, out_grid, occ_set | pts, chains):
                return True
            used_notches.remove(root["ni"])
            chains.pop()
            for r, c in pts:
                out_grid[r, c] = int(bg)
            for r, c in bar_occ:
                out_grid[r, c] = bar_color
        return False

    out0 = out.copy()
    place_roots(0, set(), out0, set(bar_occ), [])
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
                        pref = [nb for nb in frontier if nb[1] < cm]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                    if cx < B:
                        pref = [nb for nb in frontier if nb[1] > cx]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                else:
                    cm = min((r for r, _ in cells), default=B)
                    cx = max((r for r, _ in cells), default=A)
                    if cm > A:
                        pref = [nb for nb in frontier if nb[0] < cm]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                    if cx < B:
                        pref = [nb for nb in frontier if nb[0] > cx]
                        if pref:
                            cells.add(rng.choice(pref))
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


def _interval_overlap(a0, a1, b0, b1, pad=0):
    return not (a1 + pad < b0 or b1 + pad < a0)


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
    chain_meta = []
    notch_holes = []
    occ_set = set(bar_cells)

    for chain_idx in range(num_chains):
        chain_len = chain_lens[chain_idx]
        strip_width = rng.randint(4, 5)

        placement = None
        for _ in range(80):
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
                    continue
                sc0 = rng.randint(1, W - strip_width - 1)
                sc1 = sc0 + strip_width - 1
                legend_span = (sc0, sc0 + chain_len - 1)
                if legend_span[1] > sc1:
                    continue
                conflict = any(
                    _interval_overlap(sc0, sc1, meta["strip_span"][0], meta["strip_span"][1], pad=1)
                    for meta in chain_meta
                )
                if conflict:
                    continue
                same_side_conflict = any(
                    meta["side"] == side
                    and _interval_overlap(legend_span[0], legend_span[1], meta["legend_span"][0], meta["legend_span"][1], pad=1)
                    for meta in chain_meta
                )
                if same_side_conflict:
                    continue
                placement = {"side": side, "sA": sc0, "sB": sc1, "legend_span": legend_span}
                break
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
                    continue
                sr0 = rng.randint(1, H - strip_width - 1)
                sr1 = sr0 + strip_width - 1
                legend_span = (sr0, sr0 + chain_len - 1)
                if legend_span[1] > sr1:
                    continue
                conflict = any(
                    _interval_overlap(sr0, sr1, meta["strip_span"][0], meta["strip_span"][1], pad=1)
                    for meta in chain_meta
                )
                if conflict:
                    continue
                same_side_conflict = any(
                    meta["side"] == side
                    and _interval_overlap(legend_span[0], legend_span[1], meta["legend_span"][0], meta["legend_span"][1], pad=1)
                    for meta in chain_meta
                )
                if same_side_conflict:
                    continue
                placement = {"side": side, "sA": sr0, "sB": sr1, "legend_span": legend_span}
                break

        if placement is None:
            return None

        side = placement["side"]
        sA, sB = placement["sA"], placement["sB"]

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
        chain_colors = []
        for piece_idx in range(chain_len):
            if color_idx >= len(avail_colors):
                return None
            piece_color = avail_colors[color_idx]
            color_idx += 1
            chain_colors.append(piece_color)
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
                    ext_span = (
                        (max(0, prev_r0 - chain_dim), prev_r0 - 1)
                        if side == "top"
                        else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    )
                    ext = (
                        {(r, c) for r in range(ext_span[0], ext_span[1] + 1) for c in range(sA, sB + 1) if (r, c) not in occ_set}
                        if ext_span[0] <= ext_span[1]
                        else set()
                    )
                else:
                    ext_span = (
                        (max(0, prev_c0 - chain_dim), prev_c0 - 1)
                        if side == "left"
                        else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    )
                    ext = (
                        {(r, c) for r in range(sA, sB + 1) for c in range(ext_span[0], ext_span[1] + 1) if (r, c) not in occ_set}
                        if ext_span[0] <= ext_span[1]
                        else set()
                    )

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
            chain_pieces.append({"color": piece_color, "coords": piece_cells})
            occ_set |= piece_cells

        all_chain_pieces.extend(chain_pieces)
        chain_meta.append(
            {
                "side": side,
                "strip_span": (sA, sB),
                "legend_span": placement["legend_span"],
                "colors": chain_colors,
            }
        )

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

    legend_cells = {}
    legend_buffer = set()
    for meta in chain_meta:
        colors = meta["colors"]
        if orientation == "horizontal":
            row = 0 if meta["side"] == "top" else H - 1
            cols = range(meta["legend_span"][0], meta["legend_span"][1] + 1)
            for color, c in zip(colors, cols):
                legend_cells[(row, c)] = color
        else:
            col = 0 if meta["side"] == "left" else W - 1
            rows = range(meta["legend_span"][0], meta["legend_span"][1] + 1)
            for color, r in zip(colors, rows):
                legend_cells[(r, col)] = color

    for r, c in legend_cells:
        legend_buffer.add((r, c))
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                legend_buffer.add((rr, cc))

    placed_occ = set(bar_cells) | set(legend_buffer)
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
                top = bar_r0 - gap - p_h if rng.random() < 0.5 else bar_r1 + gap + 1
                left = rng.randint(1, max(1, W - p_w - 1))
            else:
                left = bar_c0 - gap - p_w if rng.random() < 0.5 else bar_c1 + gap + 1
                top = rng.randint(1, max(1, H - p_h - 1))
            if top < 1 or top + p_h > H - 1 or left < 1 or left + p_w > W - 1:
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

    for (r, c), color in legend_cells.items():
        input_grid[r, c] = color

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
