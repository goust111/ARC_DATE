# Combo variant of 16b78196 (index 11)
# combo_spec: D7/2+D14/0
# Name: 分离面板+多链短链

from __future__ import annotations

from itertools import combinations
import random

import numpy as np

from arc_utils import components_mask, find_components, get_bg

# Design analysis for the combo variant
# -------------------------------------
# Reference baseline:
# - I read the original 16b78196 solver/generator and the existing D7/2 packed-panel
#   variant. The fusion keeps the original "plug notch -> follow cavity chain" rule,
#   then applies panel packing, but now there are 3-4 chains and each chain is short
#   (1-2 pieces). This changes both the solver search regime and the output layout.
#
# a) Concretisation / behavior trace
# - Original behavior:
#   input has one 30x30 scene with a central bar, 1-2 used notches, and 3-6 scattered
#   sprites. Output is the same 30x30 scene after assembling 1-2 longer chains.
# - D7/2 already changed only the presentation:
#   solve the full scene first, then crop each solved chain together with its bar strip
#   and pack the crops into separated panels with divider lines.
# - D14/0 changes the object inventory:
#   instead of 1-2 deeper chains of length 2-3, build 3-4 shallower chains of length
#   1-2. The cognitive load shifts from "follow one deep cavity cascade" to "solve many
#   independent root matches in parallel".
# - Fused behavior:
#   1. Reconstruct the assembled 30x30 scene exactly as before.
#   2. Track each chain separately.
#   3. Pack each chain into its own panel together with only its matching bar strip.
#   Example trace for a horizontal bar:
#   - original: piece A plugs notch 1, piece B fills A's cavity, piece C plugs notch 2,
#     piece D fills C's cavity; return the single whole scene.
#   - combo: pieces A/B, C, D/E, F become four separate solved chains; return four
#     left-to-right panels, each panel containing one chain and the corresponding bar
#     segment, with 1-cell divider columns in bar color.
#
# b) Complexity calibration
# - If the generator allowed only 1-2 chains, the fusion would collapse visually back
#   toward the original task or plain D7. I therefore force 3-4 chains.
# - If every chain had length 1, the cavity-following rule would become too weak. I
#   therefore force at least one 2-piece chain in every sample, while keeping the total
#   piece count within 4-6 to stay near the original density.
# - Strip intervals are kept disjoint and separated by gaps, so panel boundaries are
#   visually obvious and root assignments do not become needlessly ambiguous.
#
# c) Implicit assumptions audit for main()
# - `bar = comps[0]`: still valid; the input remains one dominant monochrome bar.
# - `axis` from notch mouth still works because used notches still open on the bar bbox
#   boundary and chains still grow perpendicular to that side.
# - Original code only searched root subsets up to size 3. That implicit assumption is
#   broken by 3-4 chain samples, so the fusion raises the root subset limit to 4 and
#   prefers solutions with more chains.
# - Original output-size assumption (`return best_out`) is broken by D7/2. The solver
#   now always reconstructs the full scene internally and only then repacks the chains.
# - New solver assumption:
#   each intended chain owns a unique strip interval `[A, B]` and panels should be
#   ordered by that visible strip position. The generator guarantees non-overlapping
#   strip intervals.
#
# d) Placement legality
# - The generator is still output-first: it constructs the solved scene before scattering
#   pieces into the input.
# - Used strips are chosen with explicit gaps, used notches never overlap, chain pieces
#   are grown only inside in-bounds availability sets, and scattered pieces are placed
#   with non-overlap plus 4-neighbor separation checks.
#
# e) Visual observability
# - The fusion is readable from examples because two visible changes co-occur:
#   1. outputs become multi-panel instead of a single global scene,
#   2. the number of panels rises to 3-4 and most panels are short one-step or two-step
#      chains rather than long cascades.
# - Ordering is based only on the visible bar-strip order, not on hidden attributes such
#   as color ids.
#
# f) Visual cleanliness
# - Each panel is a tight crop of one chain plus its bar strip, so unrelated decoy holes
#   are suppressed in the output.
# - Divider lines use the bar color and occupy a full row/column, making the panel split
#   unambiguous without adding a new symbol system.
#
# g) Change inferability
# - No color remapping is introduced. Colors still identify distinct pieces only.
# - The only new global inference is layout: "solve each local chain, then display each
#   solved chain as a separate panel".
#
# h) Post-edit assumption audit
# - `main()` now prefers more-chain explanations before fewer-chain explanations. The
#   generator must therefore never create two intended chains that share a strip interval
#   or reuse the same root piece shape in a way that makes a lower-chain interpretation
#   equally strong. The generator enforces disjoint strips and mostly distinct geometry.
# - Some generated chains are length-1 rectangles. That is a new allowance compared with
#   the original aesthetic, but it is visible, legal for the solver, and useful to make
#   "short-chain breadth" visually obvious.


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _shape_from_coords(coords: np.ndarray, bbox) -> dict[str, object]:
    r0, c0, r1, c1 = map(int, bbox)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}


def _pixels(shape: dict[str, object], top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top) + int(lr), int(left) + int(lc)) for lr, lc in shape["rel"]}


def _snapshot_chains(chains: list[dict]) -> list[dict]:
    snap = []
    for ch in chains:
        snap.append(
            {
                "axis": str(ch["axis"]),
                "A": int(ch["A"]),
                "B": int(ch["B"]),
                "placements": [
                    {
                        "top": int(p["top"]),
                        "left": int(p["left"]),
                        "shape": p["shape"],
                        "color": int(p["color"]),
                    }
                    for p in ch["placements"]
                ],
            }
        )
    return snap


def _pack_chain_panels(
    scene: np.ndarray,
    bar_occ: set[tuple[int, int]],
    bar_color: int,
    bg: int,
    chains: list[dict],
    bar_orientation: str,
) -> np.ndarray:
    if not chains:
        return np.array(scene, copy=True)

    panels = []
    for ch in sorted(chains, key=lambda x: (int(x["A"]), int(x["B"]))):
        chain_pts: set[tuple[int, int]] = set()
        for p in ch["placements"]:
            chain_pts |= _pixels(p["shape"], int(p["top"]), int(p["left"]))
        if ch["axis"] == "cols":
            bar_seg = {(r, c) for r, c in bar_occ if int(ch["A"]) <= c <= int(ch["B"])}
        else:
            bar_seg = {(r, c) for r, c in bar_occ if int(ch["A"]) <= r <= int(ch["B"])}
        pts = chain_pts | bar_seg
        if not pts:
            continue
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        panel = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in pts:
            panel[r - r0, c - c0] = int(scene[r, c])
        bar_local = [(r - r0, c - c0) for r, c in bar_seg]
        brs = [r for r, _ in bar_local]
        bcs = [c for _, c in bar_local]
        panels.append(
            {
                "grid": panel,
                "bar_r0": min(brs),
                "bar_c0": min(bcs),
            }
        )

    if not panels:
        return np.array(scene, copy=True)

    if bar_orientation == "horizontal":
        anchor_r = max(int(p["bar_r0"]) for p in panels)
        out_h = max(anchor_r - int(p["bar_r0"]) + int(p["grid"].shape[0]) for p in panels)
        out_w = sum(int(p["grid"].shape[1]) for p in panels) + max(0, len(panels) - 1)
        out = np.full((out_h, out_w), int(bg), dtype=int)
        x = 0
        for idx, p in enumerate(panels):
            panel = p["grid"]
            y = anchor_r - int(p["bar_r0"])
            h, w = map(int, panel.shape)
            out[y : y + h, x : x + w] = panel
            x += w
            if idx != len(panels) - 1:
                out[:, x] = int(bar_color)
                x += 1
        return out

    anchor_c = max(int(p["bar_c0"]) for p in panels)
    out_w = max(anchor_c - int(p["bar_c0"]) + int(p["grid"].shape[1]) for p in panels)
    out_h = sum(int(p["grid"].shape[0]) for p in panels) + max(0, len(panels) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    y = 0
    for idx, p in enumerate(panels):
        panel = p["grid"]
        x = anchor_c - int(p["bar_c0"])
        h, w = map(int, panel.shape)
        out[y : y + h, x : x + w] = panel
        y += h
        if idx != len(panels) - 1:
            out[y, :] = int(bar_color)
            y += 1
    return out


def _prune_root_candidates(root_cands: list[tuple[tuple[int, int], dict]]) -> list[dict]:
    kept: list[dict] = []
    seen = set()
    per_piece: dict[int, int] = {}
    per_notch: dict[int, int] = {}
    for _, cand in root_cands:
        key = (int(cand["ni"]), int(cand["pi"]), int(cand["top"]), int(cand["left"]))
        if key in seen:
            continue
        if per_piece.get(int(cand["pi"]), 0) >= 4:
            continue
        if per_notch.get(int(cand["ni"]), 0) >= 4:
            continue
        seen.add(key)
        per_piece[int(cand["pi"])] = per_piece.get(int(cand["pi"]), 0) + 1
        per_notch[int(cand["ni"])] = per_notch.get(int(cand["ni"]), 0) + 1
        kept.append(cand)
        if len(kept) >= 20:
            break
    return kept


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the original assembled scene, then repack each completed chain into its
    own panel with the corresponding bar segment.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    bar = comps[0]
    pieces_raw = comps[1:]
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_orientation = "horizontal" if bar_w >= bar_h else "vertical"
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}

    base_scene = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_occ:
        base_scene[r, c] = int(bar_color)

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append(
            {
                "color": int(p["color"]),
                "shape": _shape_from_coords(p["coords"], p["bbox"]),
                "orig": (int(r0), int(c0)),
            }
        )

    def bar_notches(mask: np.ndarray, off_r: int, off_c: int) -> list[dict]:
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
            hole_gl = {(off_r + int(rr), off_c + int(cc)) for rr, cc in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": int(mouth_n)})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    def placement_pts(shape, top: int, left: int, occ: set[tuple[int, int]], must_cover=None, strip=None):
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

    def aligned_positions(axis: str, A: int, cover: set[tuple[int, int]], shape):
        if axis == "cols":
            left = int(A)
            tops = set()
            for r_n, c_n in cover:
                for lr, lc in shape["rel"]:
                    if c_n == left + int(lc):
                        tops.add(int(r_n) - int(lr))
            for top in tops:
                yield int(top), left
        else:
            top = int(A)
            lefts = set()
            for r_n, c_n in cover:
                for lr, lc in shape["rel"]:
                    if r_n == top + int(lr):
                        lefts.add(int(c_n) - int(lc))
            for left in lefts:
                yield top, int(left)

    def touches(pts: set[tuple[int, int]], other: set[tuple[int, int]]) -> bool:
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole: set[tuple[int, int]], shape):
        pos = set()
        for r_h, c_h in hole:
            for lr, lc in shape["rel"]:
                pos.add((int(r_h) - int(lr), int(c_h) - int(lc)))
        return pos

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            shape = piece["shape"]
            for top, left in hole_positions(notch["hole"], shape):
                pts = placement_pts(shape, int(top), int(left), bar_occ, must_cover=notch["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + int(shape["w"]) - 1)
                else:
                    A, B = int(top), int(top + int(shape["h"]) - 1)
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
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
                            "shape": shape,
                        },
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = _prune_root_candidates(root_cands)

    best_scene = np.array(base_scene, copy=True)
    best_chains: list[dict] = []
    best_key = None

    def search(out_grid: np.ndarray, occ_set: set[tuple[int, int]], chains: list[dict], remaining: set[int]) -> bool:
        nonlocal best_scene, best_chains, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (int(len(chains)), int(placed_cnt), -int(len(remaining)), -int(len(occ_set)))
        if best_key is None or key > best_key:
            best_key = key
            best_scene = np.array(out_grid, copy=True)
            best_chains = _snapshot_chains(chains)
        if not remaining:
            return True

        def gen_extend_moves(chain: dict, remaining_set: set[int]):
            axis = str(chain["axis"])
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = _pixels(head["shape"], int(head["top"]), int(head["left"]))
            head_top, head_left = int(head["top"]), int(head["left"])
            head_h, head_w = int(head["shape"]["h"]), int(head["shape"]["w"])

            cavity = {
                (r, c)
                for r in range(head_top, head_top + head_h)
                for c in range(head_left, head_left + head_w)
                if int(out_grid[r, c]) == int(bg)
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
                    if not touches(pts, head_pts):
                        continue
                    moves.append((int(pi), int(top), int(left), pts, shape))
            return moves

        ordered = sorted(chains, key=lambda ch: int(ch["A"]))
        for ch in ordered:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for pi, top, left, pts, shape in moves[:8]:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = ch["head"]
                placement = {"top": int(top), "left": int(left), "shape": shape, "color": int(pieces[pi]["color"])}
                ch["head"] = placement
                ch["placements"].append(placement)
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(int(pi))
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["placements"].pop()
                ch["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = int(bar_color)
        return False

    max_k = min(4, len(cand_roots), len(pieces))
    for k in range(max_k, 0, -1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(bar_occ)
            out0 = np.array(base_scene, copy=True)
            chains = []
            ok = True
            for idx in roots_sel:
                root = cand_roots[idx]
                if root["pi"] in used_p or root["ni"] in used_n:
                    ok = False
                    break
                pts = _pixels(root["shape"], int(root["top"]), int(root["left"]))
                if pts & occ0:
                    ok = False
                    break
                used_p.add(int(root["pi"]))
                used_n.add(int(root["ni"]))
                for r, c in pts:
                    out0[r, c] = int(pieces[root["pi"]]["color"])
                occ0 |= pts
                placement = {
                    "top": int(root["top"]),
                    "left": int(root["left"]),
                    "shape": root["shape"],
                    "color": int(pieces[root["pi"]]["color"]),
                }
                chains.append(
                    {
                        "axis": str(root["axis"]),
                        "A": int(root["A"]),
                        "B": int(root["B"]),
                        "head": placement,
                        "placements": [placement],
                    }
                )
            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            if search(out0, occ0, chains, remaining):
                return _pack_chain_panels(out0, bar_occ, bar_color, bg, _snapshot_chains(chains), bar_orientation)

    return _pack_chain_panels(best_scene, bar_occ, bar_color, bg, best_chains, bar_orientation)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


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


def _grow_connected(
    rng: random.Random,
    available: set[tuple[int, int]],
    target: int,
    seeds: set[tuple[int, int]],
    must_span_axis: str | None = None,
    span_range: tuple[int, int] | None = None,
    retries: int = 6,
) -> set[tuple[int, int]] | None:
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 30):
            if len(cells) >= target:
                break
            frontier = [nb for r, c in cells for dr, dc in DIR4 if (nb := (r + dr, c + dc)) in available and nb not in cells]
            if not frontier:
                break
            if must_span_axis is not None and span_range is not None:
                A, B = span_range
                if must_span_axis == "cols":
                    cmin = min(c for _, c in cells)
                    cmax = max(c for _, c in cells)
                    if cmin > A:
                        picks = [nb for nb in frontier if nb[1] < cmin]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                    if cmax < B:
                        picks = [nb for nb in frontier if nb[1] > cmax]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                else:
                    rmin = min(r for r, _ in cells)
                    rmax = max(r for r, _ in cells)
                    if rmin > A:
                        picks = [nb for nb in frontier if nb[0] < rmin]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                    if rmax < B:
                        picks = [nb for nb in frontier if nb[0] > rmax]
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
                if must_span_axis is not None and span_range is not None:
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
        if must_span_axis is not None and span_range is not None:
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


def _choose_strip_interval(
    rng: random.Random,
    length: int,
    limit: int,
    used: list[tuple[int, int]],
    min_gap: int = 1,
) -> tuple[int, int] | None:
    for _ in range(300):
        start = rng.randint(1, limit - length - 1)
        end = start + length - 1
        if all(end < a - min_gap or start > b + min_gap for a, b in used):
            return start, end
    return None


def _single_piece_cells(
    orientation: str,
    side: str,
    A: int,
    B: int,
    bar_r0: int,
    bar_r1: int,
    bar_c0: int,
    bar_c1: int,
    H: int,
    W: int,
    outside_depth: int,
) -> set[tuple[int, int]] | None:
    if orientation == "horizontal":
        if side == "top":
            r0, r1 = bar_r0 - outside_depth, bar_r0
        else:
            r0, r1 = bar_r1, bar_r1 + outside_depth
        if r0 < 0 or r1 >= H:
            return None
        return {(r, c) for r in range(r0, r1 + 1) for c in range(A, B + 1)}

    if side == "left":
        c0, c1 = bar_c0 - outside_depth, bar_c0
    else:
        c0, c1 = bar_c1, bar_c1 + outside_depth
    if c0 < 0 or c1 >= W:
        return None
    return {(r, c) for r in range(A, B + 1) for c in range(c0, c1 + 1)}


def _two_piece_chain_cells(
    orientation: str,
    side: str,
    A: int,
    B: int,
    bar_r0: int,
    bar_r1: int,
    bar_c0: int,
    bar_c1: int,
    H: int,
    W: int,
    handedness: str,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    width = B - A + 1
    root_rel = {(-2, c) for c in range(width - 1)} | {(-1, 0)} | {(0, c) for c in range(width)}
    tail_rel = {(-3, c) for c in range(width)} | {(-2, width - 1)} | {(-1, c) for c in range(1, width)}

    if handedness == "right":
        root_rel = {(r, width - 1 - c) for r, c in root_rel}
        tail_rel = {(r, width - 1 - c) for r, c in tail_rel}

    if orientation == "horizontal":
        if side == "top":
            root = {(bar_r0 + dr, A + dc) for dr, dc in root_rel}
            tail = {(bar_r0 + dr, A + dc) for dr, dc in tail_rel}
        else:
            root = {(bar_r1 - dr, A + dc) for dr, dc in root_rel}
            tail = {(bar_r1 - dr, A + dc) for dr, dc in tail_rel}
    else:
        if side == "left":
            root = {(A + dc, bar_c0 + dr) for dr, dc in root_rel}
            tail = {(A + dc, bar_c0 + dr) for dr, dc in tail_rel}
        else:
            root = {(A + dc, bar_c1 - dr) for dr, dc in root_rel}
            tail = {(A + dc, bar_c1 - dr) for dr, dc in tail_rel}

    all_pts = root | tail
    if not all(0 <= r < H and 0 <= c < W for r, c in all_pts):
        return None
    return root, tail


def _scatter_piece(
    rng: random.Random,
    input_grid: np.ndarray,
    placed_occ: set[tuple[int, int]],
    piece: dict,
    orientation: str,
    bar_r0: int,
    bar_r1: int,
    bar_c0: int,
    bar_c1: int,
) -> bool:
    H, W = map(int, input_grid.shape)
    coords = piece["coords"]
    color = int(piece["color"])
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    p_r0, p_c0 = min(rs), min(cs)
    rel = [(r - p_r0, c - p_c0) for r, c in coords]
    p_h, p_w = max(rs) - p_r0 + 1, max(cs) - p_c0 + 1

    for _ in range(1200):
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
            input_grid[r, c] = int(color)
        placed_occ |= pts
        return True
    return False


def _panel_count(output: np.ndarray, bar_color: int, orientation: str) -> int:
    if orientation == "horizontal":
        divs = [c for c in range(output.shape[1]) if np.all(output[:, c] == bar_color)]
    else:
        divs = [r for r in range(output.shape[0]) if np.all(output[r, :] == bar_color)]
    return len(divs) + 1


def _spaced_intervals(rng: random.Random, widths: list[int], limit: int) -> list[tuple[int, int]] | None:
    total = sum(widths)
    min_required = total + (len(widths) - 1) + 2
    if min_required > limit:
        return None
    slack = limit - min_required
    gaps = [1] * (len(widths) + 1)
    for _ in range(slack):
        gaps[rng.randrange(len(gaps))] += 1
    pos = gaps[0]
    out = []
    for idx, width in enumerate(widths):
        out.append((pos, pos + width - 1))
        pos += width + gaps[idx + 1]
    return out


def _gen_attempt(seed: int, debug: bool = False):
    rng = random.Random(int(seed))
    H, W = 30, 30
    bg = 0

    def fail(reason: str):
        return (None, reason) if debug else None

    orientation = rng.choice(["horizontal", "vertical"])
    bar_color = rng.randint(1, 9)
    thickness = rng.randint(5, 7)
    center = rng.randint(12, 17)

    if orientation == "horizontal":
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = min(H - 1, bar_r0 + thickness - 1)
        bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        side_space = {"top": bar_r0, "bottom": H - 1 - bar_r1}
    else:
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = min(W - 1, bar_c0 + thickness - 1)
        bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        side_space = {"left": bar_c0, "right": W - 1 - bar_c1}

    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    num_chains = rng.randint(3, 4)
    num_twos = rng.randint(1, min(2, num_chains))
    chain_lens = [1] * num_chains
    for idx in rng.sample(range(num_chains), num_twos):
        chain_lens[idx] = 2
    rng.shuffle(chain_lens)
    piece_total = sum(chain_lens)

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    if piece_total > len(avail_colors):
        return fail("too_many_colors")

    # Give each chain a distinct visible strip width so root-to-notch assignments stay
    # observable instead of relying on hidden scatter proximity. This materially reduces
    # ambiguous "panel swap" solutions between otherwise compatible short chains.
    width_pool = [3, 4, 5, 6]
    widths = rng.sample(width_pool, num_chains)
    rng.shuffle(widths)
    limit = W if orientation == "horizontal" else H
    strips = _spaced_intervals(rng, widths, limit)
    if strips is None:
        return fail("strip_layout")

    side_names = ["top", "bottom"] if orientation == "horizontal" else ["left", "right"]
    base_sides = side_names if rng.random() < 0.5 else list(reversed(side_names))
    sides = [base_sides[i % 2] for i in range(num_chains)]

    min_required_space = 3 if any(x == 2 for x in chain_lens) else 2
    if any(side_space[s] < min_required_space for s in side_names):
        return fail("side_space")

    notch_holes: list[set[tuple[int, int]]] = []
    all_chain_pieces = []
    chain_records = []
    occ_set = set(bar_cells)
    color_idx = 0

    for chain_len, (A, B), side in zip(chain_lens, strips, sides):
        if orientation == "horizontal":
            notch = {(bar_r0 if side == "top" else bar_r1, c) for c in range(A, B + 1)}
        else:
            notch = {(r, bar_c0 if side == "left" else bar_c1) for r in range(A, B + 1)}
        if not notch.issubset(bar_cells):
            return fail("notch_subset")
        bar_cells -= notch
        occ_set -= notch
        notch_holes.append(set(notch))

        chain_axis = "cols" if orientation == "horizontal" else "rows"
        chain_record = {"axis": chain_axis, "A": int(A), "B": int(B), "placements": []}
        chain_pieces = []

        if chain_len == 1:
            piece_cells = _single_piece_cells(
                orientation,
                side,
                A,
                B,
                bar_r0,
                bar_r1,
                bar_c0,
                bar_c1,
                H,
                W,
                1 if rng.random() < 0.5 else 2,
            )
            if piece_cells is None:
                return fail("single_piece_none")
            if piece_cells & occ_set:
                return fail(f"single_piece_overlap:{len(piece_cells & occ_set)}")
            piece_sets = [piece_cells]
        else:
            pair = _two_piece_chain_cells(
                orientation,
                side,
                A,
                B,
                bar_r0,
                bar_r1,
                bar_c0,
                bar_c1,
                H,
                W,
                rng.choice(["left", "right"]),
            )
            if pair is None:
                return fail("pair_bounds")
            root_cells, tail_cells = pair
            root_hit = len(root_cells & occ_set)
            tail_hit = len(tail_cells & (occ_set | root_cells))
            if root_hit or tail_hit:
                return fail(f"pair_overlap:{root_hit}:{tail_hit}")
            piece_sets = [root_cells, tail_cells]

        for piece_cells in piece_sets:
            piece_color = int(avail_colors[color_idx])
            color_idx += 1
            coords_arr = np.array(sorted(piece_cells), dtype=int)
            bbox = (
                int(coords_arr[:, 0].min()),
                int(coords_arr[:, 1].min()),
                int(coords_arr[:, 0].max()),
                int(coords_arr[:, 1].max()),
            )
            placement = {
                "top": int(bbox[0]),
                "left": int(bbox[1]),
                "shape": _shape_from_coords(coords_arr, bbox),
                "color": int(piece_color),
            }
            chain_record["placements"].append(placement)
            chain_pieces.append({"color": int(piece_color), "coords": set(piece_cells)})
            occ_set |= piece_cells

        all_chain_pieces.extend(chain_pieces)
        chain_records.append(chain_record)

    # Minimal boundary decoys keep the bar visibly holey without destabilising generation.
    boundary_cells = []
    for side in side_names:
        if orientation == "horizontal":
            row = bar_r0 if side == "top" else bar_r1
            boundary_cells.extend((row, c) for c in range(W))
        else:
            col = bar_c0 if side == "left" else bar_c1
            boundary_cells.extend((r, col) for r in range(H))
    used_holes = set().union(*notch_holes) if notch_holes else set()
    rng.shuffle(boundary_cells)
    decoys_added = 0
    for cell in boundary_cells:
        if decoys_added >= 2:
            break
        if cell not in bar_cells or cell in used_holes:
            continue
        if any((cell[0] + dr, cell[1] + dc) in used_holes for dr, dc in DIR4):
            continue
        if not _connected(bar_cells - {cell}):
            continue
        bar_cells.remove(cell)
        notch_holes.append({cell})
        used_holes.add(cell)
        decoys_added += 1

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr, cc = pts[:, 0], pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return fail("internal_hole")

    full_output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        full_output[r, c] = int(bar_color)
    for p in all_chain_pieces:
        for r, c in p["coords"]:
            full_output[r, c] = int(p["color"])

    output = _pack_chain_panels(full_output, bar_cells, bar_color, bg, chain_records, orientation)
    if _panel_count(output, bar_color, orientation) != num_chains:
        return fail("panel_count")

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = int(bar_color)

    placed_occ = set(bar_cells)
    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)
    pieces_order.sort(key=lambda p: -len(p["coords"]))
    for p in pieces_order:
        if not _scatter_piece(rng, input_grid, placed_occ, p, orientation, bar_r0, bar_r1, bar_c0, bar_c1):
            return fail("scatter")

    meta = {"orientation": orientation, "num_chains": num_chains, "piece_total": piece_total}
    return input_grid.astype(int), output.astype(int), meta


def generate_input() -> np.ndarray:
    base = int(np.random.randint(0, 1_000_000))
    for attempt in range(700):
        seed = base + attempt * 1009 + 17
        result = _gen_attempt(seed)
        if result is None:
            continue
        inp, out, meta = result
        try:
            calc = main(inp)
        except Exception:
            continue
        if not np.array_equal(calc, out):
            continue
        if meta["num_chains"] < 3:
            continue
        return inp
    raise RuntimeError("generate_input: failed after 700 attempts")
