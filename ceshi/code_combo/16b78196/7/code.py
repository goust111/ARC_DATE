# Combo variant of 16b78196 (index 7)
# combo_spec: D4/0+D13/1
# Name: 双向生长+碰撞终止
#
# Design analysis for the fused variant
# ====================================
# a) Concrete behavior and flow change
#    Original task:
#    - 1-2 independent chains start from unrelated one-sided notches.
#    - Every chain grows only away from the bar.
#    - A chain stops when its own head bbox has no black cavity left.
#
#    Fused variant here:
#    - The used opening is a through-channel hole that touches both opposite bar
#      boundaries (top+bottom for a horizontal bar, left+right for a vertical
#      bar). The channel defines one fixed strip.
#    - Two roots are inserted into the two opposite mouths of that same channel.
#      They are distinct chains: one starts from the low mouth, the other from
#      the high mouth.
#    - Each chain still follows the old local rule: the next sprite must fill
#      the black cavity inside the previous sprite's bbox and keep the same strip
#      width. The difference is that both chains now advance inside one shared
#      strip, so they can traverse the bar and head toward each other.
#    - The termination rule is dynamic. The first time a newly placed sprite
#      touches the other chain, both chains stop even if their own head bboxes
#      still contain black cavities.
#
#    Small horizontal example:
#    - The bar has one vertical corridor in columns 10..14.
#    - Root A covers the top mouth, sticks one row above the bar, and reaches a
#      little way downward inside the corridor.
#    - Root B does the symmetric thing from the bottom mouth.
#    - Piece A2 fills Root A's cavity and extends farther downward.
#    - Piece B2 fills Root B's cavity and extends farther upward.
#    - If A2 and B2 become edge-adjacent inside the corridor, both chains stop.
#      Their remaining cavities are left unfilled because collision, not cavity
#      exhaustion, is now the stopping condition.
#
# b) Complexity calibration
#    - The generator always creates exactly two used roots, one at each mouth of
#      a single through-channel. This keeps the collision rule visually obvious.
#    - The strip width is fixed to 5 in generated samples. Keeping the width
#      stable reduces pairing ambiguity and makes the shared corridor easy to see.
#    - Decoy holes remain one-sided and smaller. They preserve the original
#      visual ambiguity without creating extra collision partners.
#    - Each chain still has only 2-3 pieces, so the state space stays close to
#      the original task and the visual pattern remains human-solvable.
#
# c) Implicit-assumption audit for main()
#    - Original bar detection assumed the bar is one largest connected component.
#      A through-channel can split the bar into multiple same-color components,
#      so the solver now identifies the bar by dominant total color area.
#    - Original root search assumed each used hole is independent. That breaks:
#      the two used roots are coupled because they must occupy opposite mouths of
#      the same through-channel and share the same strip.
#    - Original search treated chains independently. That no longer works:
#      every extension move must be checked for contact with the other chain, and
#      a contact move deactivates both chains immediately.
#    - Original termination assumed "no cavity => stop". That is no longer the
#      only stopping rule. A head can retain cavity after collision, so the
#      solver must allow chains to stop with non-empty cavities.
#    - Original root preference only needed a valid hole cover. Here the root
#      must cover one mouth, remain in the through-channel strip, extend outside
#      on the correct side, and enter the corridor without already reaching the
#      opposite mouth. The generator guarantees those properties.
#
# d) Placement legality
#    - The through-channel is carved before any piece is placed, so all chain
#      cells live either in that channel or outside the bar.
#    - Both chains reuse the same strip, but pieces are generated so they never
#      overlap. The only allowed inter-chain interaction is final 4-neighbor
#      contact.
#    - Scattered input pieces keep their original orientation and are placed away
#      from the bar and from each other.
#    - Every sample is validated with main(input) == output.
#
# e) Visual observability
#    - The bar visibly contains one complete pass-through corridor.
#    - The output visibly contains two differently colored sequences entering that
#      corridor from opposite mouths and meeting inside one strip.
#    - Because the final touching sprites usually still leave black cavity inside
#      their own bboxes, the stopping event is visibly not the original static
#      "tail has no cavity" rule.
#
# f) Visual cleanliness
#    - Only one through-channel is used, so the shared-corridor story is easy to
#      parse.
#    - Decoys stay smaller and never merge with the used corridor.
#    - The two roots are forced to start at opposite mouths and are rejected if
#      they touch too early, preventing cluttered ambiguous outputs.
#
# g) Change inferability
#    - Colors remain arbitrary and do not encode chain order.
#    - The rule is inferred entirely from geometry: one shared corridor, fixed
#      strip width, cavity-filling continuation, and collision-based stopping.
#
# New solver assumptions introduced by this combo
# -----------------------------------------------
# - There is exactly one used through-channel, and the two used roots occupy its
#   opposite mouths.
# - Both used chains share the same strip width and the same strip coordinates.
# - Every generated extension still leaves a cavity; termination happens only
#   because the opposite chain is touched.
# - The generator enforces "no contact before the final collision" so that the
#   first inter-chain touch is the intended stopping event.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, find_components, get_bg


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps

    area_by_color: dict[int, int] = {}
    for comp in comps:
        color = int(comp["color"])
        area_by_color[color] = area_by_color.get(color, 0) + int(comp["area"])
    bar_color = max(area_by_color.items(), key=lambda kv: kv[1])[0]
    bar_parts = [comp for comp in comps if int(comp["color"]) == bar_color]
    pieces_raw = [comp for comp in comps if int(comp["color"]) != bar_color]

    out = np.full((H, W), int(bg), dtype=int)
    bar_occ = {(int(r), int(c)) for part in bar_parts for r, c in part["coords"]}
    bar_rs = [r for r, _ in bar_occ]
    bar_cs = [c for _, c in bar_occ]
    bar_r0, bar_c0, bar_r1, bar_c1 = min(bar_rs), min(bar_cs), max(bar_rs), max(bar_cs)
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    def in_bar_bbox(rc: tuple[int, int]) -> bool:
        r, c = rc
        return bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1

    def growth_coord(rc: tuple[int, int], axis: str) -> int:
        return int(rc[0] if axis == "cols" else rc[1])

    def span_coord(rc: tuple[int, int], axis: str) -> int:
        return int(rc[1] if axis == "cols" else rc[0])

    def shape_from_coords(coords: np.ndarray, bbox) -> dict[str, object]:
        r0, c0, r1, c1 = map(int, bbox)
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        return {"rel": rel, "h": r1 - r0 + 1, "w": c1 - c0 + 1}

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    def pixels(v: dict[str, object], top: int, left: int) -> set[tuple[int, int]]:
        return {(top + int(lr), left + int(lc)) for lr, lc in v["rel"]}

    def placement_pts(
        v: dict[str, object],
        top: int,
        left: int,
        occ: set[tuple[int, int]],
        must_cover: set[tuple[int, int]] | None = None,
        strip: tuple[str, int, int] | None = None,
    ) -> set[tuple[int, int]] | None:
        if top < 0 or left < 0 or top + int(v["h"]) > H or left + int(v["w"]) > W:
            return None
        if strip is not None:
            axis, A, B = strip
            if axis == "cols":
                if left != A or int(v["w"]) != B - A + 1:
                    return None
            else:
                if top != A or int(v["h"]) != B - A + 1:
                    return None
        pts = pixels(v, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        return pts

    def aligned_positions(axis: str, A: int, cover: set[tuple[int, int]], v: dict[str, object]):
        if axis == "cols":
            left = A
            tops: set[int] = set()
            for rN, cN in cover:
                for lr, lc in v["rel"]:
                    if cN == left + int(lc):
                        tops.add(int(rN - int(lr)))
            for top in tops:
                yield top, left
        else:
            top = A
            lefts: set[int] = set()
            for rN, cN in cover:
                for lr, lc in v["rel"]:
                    if rN == top + int(lr):
                        lefts.add(int(cN - int(lc)))
            for left in lefts:
                yield top, left

    def touches(pts: set[tuple[int, int]], other: set[tuple[int, int]]) -> bool:
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def bar_openings(mask: np.ndarray, top0: int, left0: int) -> list[dict[str, object]]:
        empty = ~mask
        h, w = empty.shape
        openings: list[dict[str, object]] = []
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
            through_tb = top > 0 and bottom > 0 and left == 0 and right == 0
            through_lr = left > 0 and right > 0 and top == 0 and bottom == 0
            if through_tb:
                openings.append(
                    {
                        "axis": "cols",
                        "through": True,
                        "hole": hole_gl,
                        "A": int(min(c) + left0),
                        "B": int(max(c) + left0),
                        "mouths": {
                            "low": {(top0 + int(rr), left0 + int(cc)) for rr, cc in pts if int(rr) == 0},
                            "high": {(top0 + int(rr), left0 + int(cc)) for rr, cc in pts if int(rr) == h - 1},
                        },
                    }
                )
                continue
            if through_lr:
                openings.append(
                    {
                        "axis": "rows",
                        "through": True,
                        "hole": hole_gl,
                        "A": int(min(r) + top0),
                        "B": int(max(r) + top0),
                        "mouths": {
                            "low": {(top0 + int(rr), left0 + int(cc)) for rr, cc in pts if int(cc) == 0},
                            "high": {(top0 + int(rr), left0 + int(cc)) for rr, cc in pts if int(cc) == w - 1},
                        },
                    }
                )
        return openings

    openings = [o for o in bar_openings(bar_mask, bar_r0, bar_c0) if o["through"]]

    def is_outside_side(pts: set[tuple[int, int]], axis: str, side: str) -> bool:
        if axis == "cols":
            return min(r for r, _ in pts) < bar_r0 if side == "low" else max(r for r, _ in pts) > bar_r1
        return min(c for _, c in pts) < bar_c0 if side == "low" else max(c for _, c in pts) > bar_c1

    root_cands: list[tuple[tuple[int, int, int, int], dict[str, object]]] = []
    for ni, opening in enumerate(openings):
        axis = str(opening["axis"])
        A, B = int(opening["A"]), int(opening["B"])
        hole = set(opening["hole"])
        mouths = {k: set(v) for k, v in opening["mouths"].items()}
        strip = (axis, A, B)
        for side in ("low", "high"):
            mouth = mouths[side]
            far_mouth = mouths["high" if side == "low" else "low"]
            for pi, p in enumerate(pieces):
                v = p["shape"]
                if axis == "cols" and int(v["w"]) != B - A + 1:
                    continue
                if axis == "rows" and int(v["h"]) != B - A + 1:
                    continue
                for top, left in aligned_positions(axis, A, mouth, v):
                    pts = placement_pts(v, top, left, bar_occ, must_cover=mouth, strip=strip)
                    if pts is None:
                        continue
                    if not is_outside_side(pts, axis, side):
                        continue
                    if not any(rc in hole and rc not in mouth for rc in pts):
                        continue
                    if pts & far_mouth:
                        continue
                    mv = abs(int(top) - int(p["orig"][0])) + abs(int(left) - int(p["orig"][1]))
                    root_cands.append(
                        (
                            (int(mv), -int(len(mouth)), int(pi), 0 if side == "low" else 1),
                            {
                                "ni": ni,
                                "pi": pi,
                                "top": int(top),
                                "left": int(left),
                                "axis": axis,
                                "A": A,
                                "B": B,
                                "shape": v,
                                "side": side,
                                "hole": hole,
                            },
                        )
                    )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:40]]

    best_out = out.copy()
    best_key: tuple[int, int, int] | None = None

    def front_of_head(head: dict[str, object], axis: str, grow: str) -> int:
        pts = pixels(head["shape"], int(head["top"]), int(head["left"]))
        vals = [growth_coord(rc, axis) for rc in pts]
        return max(vals) if grow == "high" else min(vals)

    def extends_forward(chain: dict[str, object], pts: set[tuple[int, int]]) -> bool:
        vals = [growth_coord(rc, str(chain["axis"])) for rc in pts]
        new_front = max(vals) if chain["grow"] == "high" else min(vals)
        old_front = front_of_head(chain["head"], str(chain["axis"]), str(chain["grow"]))
        return new_front > old_front if chain["grow"] == "high" else new_front < old_front

    def gen_extend_moves(chain_idx: int, chains: list[dict[str, object]], out_grid: np.ndarray, occ_set: set[tuple[int, int]], remaining: set[int]):
        chain = chains[chain_idx]
        if not chain["active"]:
            return []
        axis = str(chain["axis"])
        A, B = int(chain["A"]), int(chain["B"])
        head = chain["head"]
        head_pts = pixels(head["shape"], int(head["top"]), int(head["left"]))
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

        other_cells = set()
        for oi, other in enumerate(chains):
            if oi != chain_idx:
                other_cells |= set(other["cells"])

        moves = []
        for pi in remaining:
            v = pieces[pi]["shape"]
            if axis == "cols" and int(v["w"]) != B - A + 1:
                continue
            if axis == "rows" and int(v["h"]) != B - A + 1:
                continue
            for top, left in aligned_positions(axis, A, cavity, v):
                pts = placement_pts(v, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                if pts is None:
                    continue
                if not touches(pts, head_pts):
                    continue
                if not extends_forward(chain, pts):
                    continue
                collision = touches(pts, other_cells)
                adv_vals = [growth_coord(rc, axis) for rc in pts]
                adv = max(adv_vals) if chain["grow"] == "high" else -min(adv_vals)
                moves.append(((0 if collision else 1, -adv, abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))), (pi, top, left, pts, v, collision)))
        moves.sort(key=lambda x: x[0])
        return [mv for _, mv in moves[:12]]

    def search(out_grid: np.ndarray, occ_set: set[tuple[int, int]], chains: list[dict[str, object]], remaining: set[int]) -> bool:
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (placed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        move_options = []
        for ci in range(len(chains)):
            for move in gen_extend_moves(ci, chains, out_grid, occ_set, remaining):
                move_options.append((ci, move))
        if not move_options:
            return False

        for ci, (pi, top, left, pts, v, collision) in move_options:
            changed_active = []
            chain = chains[ci]
            old_head = chain["head"]
            old_cells = set(chain["cells"])

            for r, c in pts:
                out_grid[r, c] = int(pieces[pi]["color"])
            chain["head"] = {"top": int(top), "left": int(left), "shape": v}
            chain["cells"] = old_cells | set(pts)
            occ2 = set(occ_set) | set(pts)
            rem2 = set(remaining)
            rem2.remove(pi)

            if collision:
                for oi, other in enumerate(chains):
                    if oi == ci:
                        continue
                    if touches(pts, set(other["cells"])):
                        changed_active.append((oi, other["active"]))
                        other["active"] = False
                changed_active.append((ci, chain["active"]))
                chain["active"] = False

            if search(out_grid, occ2, chains, rem2):
                return True

            if collision:
                for oi, old_val in changed_active:
                    chains[oi]["active"] = old_val
            chain["head"] = old_head
            chain["cells"] = old_cells
            for r, c in pts:
                out_grid[r, c] = int(bg)
            for r, c in bar_occ:
                out_grid[r, c] = bar_color
        return False

    pair_cands = []
    for i, r1 in enumerate(cand_roots):
        for j, r2 in enumerate(cand_roots):
            if j <= i:
                continue
            if r1["ni"] != r2["ni"]:
                continue
            if r1["pi"] == r2["pi"] or r1["side"] == r2["side"]:
                continue
            if r1["axis"] != r2["axis"] or r1["A"] != r2["A"] or r1["B"] != r2["B"]:
                continue
            pts1 = pixels(r1["shape"], int(r1["top"]), int(r1["left"]))
            pts2 = pixels(r2["shape"], int(r2["top"]), int(r2["left"]))
            if pts1 & pts2:
                continue
            if touches(pts1, pts2):
                continue
            score = (
                abs(int(r1["top"]) - int(r2["top"])) + abs(int(r1["left"]) - int(r2["left"])),
                i + j,
            )
            pair_cands.append((score, (r1, r2, pts1, pts2)))

    pair_cands.sort(key=lambda x: x[0])

    for _, (r1, r2, pts1, pts2) in pair_cands[:24]:
        low_root = r1 if r1["side"] == "low" else r2
        high_root = r2 if r1["side"] == "low" else r1
        low_pts = pts1 if r1["side"] == "low" else pts2
        high_pts = pts2 if r1["side"] == "low" else pts1

        occ0 = set(bar_occ)
        if low_pts & occ0 or high_pts & occ0 or low_pts & high_pts:
            continue
        out0 = np.full((H, W), int(bg), dtype=int)
        for r, c in bar_occ:
            out0[r, c] = bar_color

        for r, c in low_pts:
            out0[r, c] = int(pieces[int(low_root["pi"])]["color"])
        for r, c in high_pts:
            out0[r, c] = int(pieces[int(high_root["pi"])]["color"])
        occ0 |= low_pts | high_pts

        chains = [
            {
                "axis": low_root["axis"],
                "A": low_root["A"],
                "B": low_root["B"],
                "head": {"top": int(low_root["top"]), "left": int(low_root["left"]), "shape": low_root["shape"]},
                "cells": set(low_pts),
                "active": True,
                "grow": "high",
            },
            {
                "axis": high_root["axis"],
                "A": high_root["A"],
                "B": high_root["B"],
                "head": {"top": int(high_root["top"]), "left": int(high_root["left"]), "shape": high_root["shape"]},
                "cells": set(high_pts),
                "active": True,
                "grow": "low",
            },
        ]
        remaining = set(range(len(pieces))) - {int(low_root["pi"]), int(high_root["pi"])}
        if search(out0, occ0, chains, remaining):
            return best_out

    return best_out


def _rot90_cw(piece: np.ndarray) -> np.ndarray:
    return np.rot90(piece, k=-1)


def _rot90_ccw(piece: np.ndarray) -> np.ndarray:
    return np.rot90(piece, k=1)


def _rot180(piece: np.ndarray) -> np.ndarray:
    return np.rot90(piece, k=2)


def _trim_piece(piece: np.ndarray) -> np.ndarray | None:
    rows = np.any(piece, axis=1)
    cols = np.any(piece, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return piece[rmin : rmax + 1, cmin : cmax + 1].copy()


def _connected(cells: set[tuple[int, int]]) -> bool:
    if len(cells) <= 1:
        return True
    todo = [next(iter(cells))]
    seen = {todo[0]}
    while todo:
        r, c = todo.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                todo.append(nb)
    return len(seen) == len(cells)


def _is_rectangular_piece(piece: np.ndarray) -> bool:
    rows = np.any(piece, axis=1)
    cols = np.any(piece, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(np.sum(piece)) == int((rmax - rmin + 1) * (cmax - cmin + 1))


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _bbox_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(cells)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in b:
                return True
    return False


def _growth_coord(cell: tuple[int, int], orientation: str) -> int:
    return cell[0] if orientation == "horizontal" else cell[1]


def _span_coord(cell: tuple[int, int], orientation: str) -> int:
    return cell[1] if orientation == "horizontal" else cell[0]


def _strip_rect(orientation: str, A: int, B: int, g0: int, g1: int) -> set[tuple[int, int]]:
    if g0 > g1:
        return set()
    if orientation == "horizontal":
        return {(r, c) for r in range(g0, g1 + 1) for c in range(A, B + 1)}
    return {(r, c) for r in range(A, B + 1) for c in range(g0, g1 + 1)}


def _growth_bounds(cells: set[tuple[int, int]], orientation: str) -> tuple[int, int]:
    vals = [_growth_coord(rc, orientation) for rc in cells]
    return min(vals), max(vals)


def _outside_on_side(cell: tuple[int, int], orientation: str, bar_low: int, bar_high: int, side: str) -> bool:
    g = _growth_coord(cell, orientation)
    return g < bar_low if side == "low" else g > bar_high


def _connected_mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.int8)
    for r, c in cells:
        arr[r - r0, c - c0] = 1
    return arr


def _piece_quality(cells: set[tuple[int, int]], strip_width: int, orientation: str) -> bool:
    area = len(cells)
    if not (8 <= area <= 13):
        return False
    if not _connected(cells):
        return False
    coords = _connected_mask_from_cells(cells)
    if _is_rectangular_piece(coords):
        return False
    r0, c0, r1, c1 = _bbox(cells)
    strip_dim = (c1 - c0 + 1) if orientation == "horizontal" else (r1 - r0 + 1)
    if strip_dim != strip_width:
        return False
    fill = area / float((r1 - r0 + 1) * (c1 - c0 + 1))
    return 0.38 <= fill <= 0.90


def _grow_connected(
    rng: random.Random,
    available: set[tuple[int, int]],
    target: int,
    seeds: set[tuple[int, int]],
    *,
    orientation: str,
    A: int,
    B: int,
    retries: int = 18,
) -> set[tuple[int, int]] | None:
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
            min_span = min(_span_coord(rc, orientation) for rc in cells)
            max_span = max(_span_coord(rc, orientation) for rc in cells)
            if min_span > A:
                pref = [rc for rc in frontier if _span_coord(rc, orientation) < min_span]
                if pref:
                    cells.add(rng.choice(pref))
                    continue
            if max_span < B:
                pref = [rc for rc in frontier if _span_coord(rc, orientation) > max_span]
                if pref:
                    cells.add(rng.choice(pref))
                    continue
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue
        if min(_span_coord(rc, orientation) for rc in cells) != A or max(_span_coord(rc, orientation) for rc in cells) != B:
            continue
        if _connected(cells):
            return cells
    return None


def _grow_connected_plain(
    rng: random.Random,
    available: set[tuple[int, int]],
    target: int,
    seeds: set[tuple[int, int]],
    *,
    retries: int = 18,
) -> set[tuple[int, int]] | None:
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
            cells.add(rng.choice(frontier))
        if len(cells) == target and _connected(cells):
            return cells
    return None


def _through_hole(
    rng: random.Random,
    orientation: str,
    bar_cells: set[tuple[int, int]],
    bar_r0: int,
    bar_c0: int,
    bar_r1: int,
    bar_c1: int,
    A: int,
    B: int,
) -> set[tuple[int, int]] | None:
    del rng
    if orientation == "horizontal":
        return {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(A, B + 1) if (r, c) in bar_cells}
    return {(r, c) for r in range(A, B + 1) for c in range(bar_c0, bar_c1 + 1) if (r, c) in bar_cells}


def _one_sided_decoy(
    rng: random.Random,
    orientation: str,
    bar_cells: set[tuple[int, int]],
    existing_holes: list[set[tuple[int, int]]],
    bar_r0: int,
    bar_c0: int,
    bar_r1: int,
    bar_c1: int,
) -> set[tuple[int, int]] | None:
    blocked = set().union(*existing_holes) if existing_holes else set()
    blocked |= {(r + dr, c + dc) for r, c in blocked for dr, dc in DIR4}
    for _ in range(120):
        depth = rng.randint(1, 2)
        area = rng.randint(1, 5)
        if orientation == "horizontal":
            side = rng.choice(["top", "bottom"])
            if side == "top":
                r0, r1 = bar_r0, min(bar_r1, bar_r0 + depth - 1)
            else:
                r0, r1 = max(bar_r0, bar_r1 - depth + 1), bar_r1
            c0 = rng.randint(bar_c0, max(bar_c0, bar_c1 - 3))
            c1 = min(bar_c1, c0 + rng.randint(0, 3))
            avail = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) in bar_cells and (r, c) not in blocked}
            seed_row = bar_r0 if side == "top" else bar_r1
            seeds = {(seed_row, c) for c in range(c0, c1 + 1) if (seed_row, c) in avail}
        else:
            side = rng.choice(["left", "right"])
            if side == "left":
                c0, c1 = bar_c0, min(bar_c1, bar_c0 + depth - 1)
            else:
                c0, c1 = max(bar_c0, bar_c1 - depth + 1), bar_c1
            r0 = rng.randint(bar_r0, max(bar_r0, bar_r1 - 3))
            r1 = min(bar_r1, r0 + rng.randint(0, 3))
            avail = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) in bar_cells and (r, c) not in blocked}
            seed_col = bar_c0 if side == "left" else bar_c1
            seeds = {(r, seed_col) for r in range(r0, r1 + 1) if (r, seed_col) in avail}
        if not seeds or len(avail) < 1:
            continue
        target = min(area, len(avail))
        hole = _grow_connected_plain(rng, avail, target, {rng.choice(list(seeds))}, retries=12)
        if hole is None:
            continue
        if any(rc in blocked for rc in hole):
            continue
        return hole
    return None


def _build_root_piece(
    rng: random.Random,
    orientation: str,
    side: str,
    A: int,
    B: int,
    bar_low: int,
    bar_high: int,
    hole: set[tuple[int, int]],
    mouth: set[tuple[int, int]],
    occ_now: set[tuple[int, int]],
    other_cells: set[tuple[int, int]],
    H: int,
    W: int,
) -> set[tuple[int, int]] | None:
    del rng, hole, mouth, H, W
    strip_width = B - A + 1
    base_templates = [
        np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=np.int8),
        np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1]], dtype=np.int8),
    ]
    if side == "high":
        base_templates = [_rot180(t) for t in base_templates]

    for tpl in base_templates:
        if side == "low":
            anchor_g = bar_low - 1
        else:
            anchor_g = bar_high - tpl.shape[0] + 2
        piece: set[tuple[int, int]] = set()
        for lr in range(tpl.shape[0]):
            for lc in range(tpl.shape[1]):
                if tpl[lr, lc] != 1:
                    continue
                if orientation == "horizontal":
                    rc = (anchor_g + lr, A + lc)
                else:
                    rc = (A + lc, anchor_g + lr)
                piece.add(rc)
        if any(rc in occ_now for rc in piece):
            continue
        if any(rc in other_cells for rc in piece) or _touches(piece, other_cells):
            continue
        if not all(0 <= r < 30 and 0 <= c < 30 for r, c in piece):
            continue
        if not _piece_quality(piece, strip_width, orientation):
            continue
        cavity = _bbox_cells(piece) - (occ_now | piece)
        if not (1 <= len(cavity) <= 7 and _connected(cavity)):
            continue
        if not any(_outside_on_side(rc, orientation, bar_low, bar_high, side) for rc in piece):
            continue
        return piece
    return None


def _build_next_piece(
    rng: random.Random,
    orientation: str,
    A: int,
    B: int,
    grow: str,
    prev_piece: set[tuple[int, int]],
    occ_now: set[tuple[int, int]],
    other_cells: set[tuple[int, int]],
    walkway: set[tuple[int, int]],
    require_touch_other: bool,
    must_reach_side: str | None,
    bar_low: int,
    bar_high: int,
    H: int,
    W: int,
) -> set[tuple[int, int]] | None:
    strip_width = B - A + 1
    grid_high = H - 1 if orientation == "horizontal" else W - 1
    cavity = _bbox_cells(prev_piece) - occ_now
    if not (1 <= len(cavity) <= 7) or not _connected(cavity):
        return None
    prev_low, prev_high = _growth_bounds(prev_piece, orientation)
    for _ in range(180):
        max_ext = 6 if must_reach_side is not None else (4 if require_touch_other else 3)
        ext_len = rng.randint(1, max_ext)
        if grow == "high":
            ext = _strip_rect(orientation, A, B, prev_high + 1, min(grid_high, prev_high + ext_len))
        else:
            ext = _strip_rect(orientation, A, B, max(0, prev_low - ext_len), prev_low - 1)
        ext = {rc for rc in ext if rc in walkway and rc not in occ_now}
        avail = cavity | ext
        if len(avail) < 8:
            continue
        target = rng.randint(max(8, len(cavity) + 1), min(13, len(avail)))
        piece = _grow_connected(rng, avail, target, cavity, orientation=orientation, A=A, B=B, retries=20)
        if piece is None:
            continue
        if not cavity.issubset(piece):
            continue
        if not _piece_quality(piece, strip_width, orientation):
            continue
        new_low, new_high = _growth_bounds(piece, orientation)
        if grow == "high" and new_high <= prev_high:
            continue
        if grow == "low" and new_low >= prev_low:
            continue
        touch_other = _touches(piece, other_cells)
        if require_touch_other != touch_other:
            continue
        if must_reach_side is not None:
            if not any(_outside_on_side(rc, orientation, bar_low, bar_high, must_reach_side) for rc in piece):
                continue
        rem = _bbox_cells(piece) - (occ_now | piece)
        if not (1 <= len(rem) <= 7 and _connected(rem)):
            continue
        return piece
    return None


def _coords_to_piece(coords: set[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.int8)
    for r, c in coords:
        arr[r - r0, c - c0] = 1
    return arr, (r0, c0)


def _place_scattered_piece(
    rng: random.Random,
    grid: np.ndarray,
    placed_occ: set[tuple[int, int]],
    piece_mask: np.ndarray,
    color: int,
    orientation: str,
    bar_r0: int,
    bar_c0: int,
    bar_r1: int,
    bar_c1: int,
) -> bool:
    ph, pw = piece_mask.shape
    H, W = grid.shape
    rel = [(r, c) for r in range(ph) for c in range(pw) if piece_mask[r, c] == 1]
    for _ in range(900):
        gap = rng.randint(2, 7)
        if orientation == "horizontal":
            top = bar_r0 - gap - ph if rng.random() < 0.5 else bar_r1 + gap + 1
            left = rng.randint(0, max(0, W - pw))
        else:
            left = bar_c0 - gap - pw if rng.random() < 0.5 else bar_c1 + gap + 1
            top = rng.randint(0, max(0, H - ph))
        if top < 0 or left < 0 or top + ph > H or left + pw > W:
            continue
        pts = {(top + dr, left + dc) for dr, dc in rel}
        if pts & placed_occ:
            continue
        if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in DIR4):
            continue
        for r, c in pts:
            grid[r, c] = int(color)
        placed_occ |= pts
        return True
    return False


def _bar_components_have_only_boundary_holes(bar_cells: set[tuple[int, int]], bar_r0: int, bar_c0: int, bar_r1: int, bar_c1: int) -> bool:
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~mask, connectivity=4):
        rr, cc = pts[:, 0], pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return False
    return True


def _collision_metric(output: np.ndarray) -> tuple[int, int]:
    bg = get_bg(output)
    comps = find_components(output, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    area_by_color: dict[int, int] = {}
    for comp in comps:
        color = int(comp["color"])
        area_by_color[color] = area_by_color.get(color, 0) + int(comp["area"])
    bar_color = max(area_by_color.items(), key=lambda kv: kv[1])[0]
    pieces = [comp for comp in comps if int(comp["color"]) != bar_color]
    if len(pieces) < 2:
        return 0, 0
    piece_sets = [{(int(r), int(c)) for r, c in comp["coords"]} for comp in pieces]
    touch_pairs = 0
    for i in range(len(piece_sets)):
        for j in range(i + 1, len(piece_sets)):
            if _touches(piece_sets[i], piece_sets[j]):
                touch_pairs += 1
    cavity_heads = 0
    for pts in piece_sets:
        cavity = _bbox_cells(pts) - ({(int(r), int(c)) for r, c in np.argwhere(output != bg)})
        if cavity:
            cavity_heads += 1
    return touch_pairs, cavity_heads


def _mirror_span(local_cells: set[tuple[int, int]], strip_width: int) -> set[tuple[int, int]]:
    return {(g, strip_width - 1 - s) for g, s in local_cells}


def _project_local_cells(
    local_cells: set[tuple[int, int]],
    orientation: str,
    bar_low: int,
    A: int,
) -> set[tuple[int, int]]:
    if orientation == "horizontal":
        return {(bar_low + g, A + s) for g, s in local_cells}
    return {(A + s, bar_low + g) for g, s in local_cells}


def _build_chain_templates(strip_width: int, mirror: bool) -> dict[str, set[tuple[int, int]]]:
    templates = {
        "root_low": {(-1, s) for s in range(strip_width)} | {(0, s) for s in range(strip_width)} | {(1, 0)},
        "root_high": {(3, strip_width - 1)} | {(4, s) for s in range(strip_width)} | {(5, s) for s in range(strip_width)},
        "next_low": {(1, s) for s in range(1, strip_width)} | {(2, 0), (2, 1)},
        "next_high": {(2, strip_width - 3), (2, strip_width - 2), (2, strip_width - 1)} | {(3, s) for s in range(strip_width - 1)},
    }
    if not mirror:
        return templates
    return {name: _mirror_span(cells, strip_width) for name, cells in templates.items()}


def _gen_attempt(seed: int):
    rng = random.Random(seed)
    H = W = 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    thickness = 5
    bar_color = rng.choice([c for c in range(1, 10)])

    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * H)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = min(H - 1, bar_r0 + thickness - 1)
        bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        bar_low, bar_high = bar_r0, bar_r1
        free_before, free_after = bar_r0, H - 1 - bar_r1
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = min(W - 1, bar_c0 + thickness - 1)
        bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        bar_low, bar_high = bar_c0, bar_c1
        free_before, free_after = bar_c0, W - 1 - bar_c1

    if min(free_before, free_after) < 3:
        return None

    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}
    strip_width = 5
    if orientation == "horizontal":
        A = rng.randint(2, W - strip_width - 3)
    else:
        A = rng.randint(2, H - strip_width - 3)
    B = A + strip_width - 1

    hole = _through_hole(rng, orientation, bar_cells, bar_r0, bar_c0, bar_r1, bar_c1, A, B)
    if hole is None:
        return None
    bar_cells -= hole

    templates = _build_chain_templates(strip_width, mirror=bool(rng.randint(0, 1)))
    chains = {
        "low": {
            "pieces": [
                _project_local_cells(templates["root_low"], orientation, bar_low, A),
                _project_local_cells(templates["next_low"], orientation, bar_low, A),
            ]
        },
        "high": {
            "pieces": [
                _project_local_cells(templates["root_high"], orientation, bar_low, A),
                _project_local_cells(templates["next_high"], orientation, bar_low, A),
            ]
        },
    }

    root_low = chains["low"]["pieces"][0]
    root_high = chains["high"]["pieces"][0]
    next_low = chains["low"]["pieces"][1]
    next_high = chains["high"]["pieces"][1]
    all_piece_coords = [root_low, root_high, next_low, next_high]

    if any(not pts for pts in all_piece_coords):
        return None
    if any(not all(0 <= r < H and 0 <= c < W for r, c in pts) for pts in all_piece_coords):
        return None
    if any(pts & bar_cells for pts in all_piece_coords):
        return None
    for i in range(len(all_piece_coords)):
        for j in range(i + 1, len(all_piece_coords)):
            if all_piece_coords[i] & all_piece_coords[j]:
                return None

    mouth_low = {rc for rc in hole if _growth_coord(rc, orientation) == bar_low}
    mouth_high = {rc for rc in hole if _growth_coord(rc, orientation) == bar_high}
    if not mouth_low.issubset(root_low) or not mouth_high.issubset(root_high):
        return None
    if not root_low & (hole - mouth_low):
        return None
    if not root_high & (hole - mouth_high):
        return None
    if not any(_outside_on_side(rc, orientation, bar_low, bar_high, "low") for rc in root_low):
        return None
    if not any(_outside_on_side(rc, orientation, bar_low, bar_high, "high") for rc in root_high):
        return None
    if _touches(root_low, root_high):
        return None
    if not _touches(root_low, next_low) or not _touches(root_high, next_high):
        return None
    if _touches(root_low, next_high) or _touches(root_high, next_low):
        return None
    if _touches(next_low, root_high) or _touches(next_high, root_low):
        return None
    if not _touches(next_low, next_high):
        return None

    low_root_cavity = _bbox_cells(root_low) - (set(bar_cells) | root_low)
    high_root_cavity = _bbox_cells(root_high) - (set(bar_cells) | root_high)
    if not low_root_cavity or not high_root_cavity:
        return None
    if not low_root_cavity.issubset(next_low):
        return None
    if not high_root_cavity.issubset(next_high):
        return None

    existing_holes = [hole]
    decoy_target = rng.randint(2, 4)
    decoys = 0
    for _ in range(400):
        if decoys >= decoy_target:
            break
        dh = _one_sided_decoy(rng, orientation, bar_cells, existing_holes, bar_r0, bar_c0, bar_r1, bar_c1)
        if dh is None:
            continue
        if any(rc not in bar_cells for rc in dh):
            continue
        bar_cells -= dh
        existing_holes.append(dh)
        decoys += 1
    if decoys < 2:
        return None
    if not _bar_components_have_only_boundary_holes(bar_cells, bar_r0, bar_c0, bar_r1, bar_c1):
        return None

    if not (4 <= len(all_piece_coords) <= 6):
        return None

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = bar_color

    colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(colors)
    if len(colors) < len(all_piece_coords):
        return None

    pieces_info = []
    for color, coords in zip(colors, all_piece_coords):
        for r, c in coords:
            output[r, c] = int(color)
        pieces_info.append({"color": int(color), "coords": set(coords)})

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color
    placed_occ = set(bar_cells)
    scatter_order = pieces_info[:]
    rng.shuffle(scatter_order)
    for info in scatter_order:
        mask, _ = _coords_to_piece(info["coords"])
        if not _place_scattered_piece(rng, input_grid, placed_occ, mask, int(info["color"]), orientation, bar_r0, bar_c0, bar_r1, bar_c1):
            return None

    touch_pairs, cavity_heads = _collision_metric(output)
    if touch_pairs < 1:
        return None
    return input_grid.astype(int), output.astype(int), {"touch_pairs": touch_pairs, "cavity_heads": cavity_heads}


def generate_input() -> np.ndarray:
    for attempt in range(500):
        seed = attempt * 1009 + int(np.random.randint(1, 100_000))
        built = _gen_attempt(seed)
        if built is None:
            continue
        inp, out, _meta = built
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    raise RuntimeError("generate_input: failed after 500 attempts")
