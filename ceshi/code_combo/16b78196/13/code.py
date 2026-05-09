# Combo variant of 16b78196 (index 13)
# combo_spec: D9/2+D11/1+D14/1
# Name: 短杆+角落散布+干扰精灵
#
# Design analysis
# ---
# concrete_behavior:
#   input:
#     - The largest component is still a single-color bar with boundary-opening
#       notches, but the bar is now short: it spans only 50-63% of the grid
#       along its main axis instead of touching both borders.
#     - Real chain sprites are no longer scattered next to the bar. Every sprite
#       is relocated into one of the four corner rectangles defined by the bar's
#       bbox, so the solver must search far from the notches.
#     - Extra 1-2 decoy sprites are mixed into the corner pool. They are valid
#       monochrome connected pieces, but they never belong to a fully completable
#       chain.
#   output:
#     - The short bar stays in place.
#     - Only the genuine chain pieces are assembled into used notches and grown
#       outward strip-by-strip. Decoys disappear.
# behavior_trace:
#   original:
#     - Find the largest full-span bar.
#     - Use every remaining component as a true piece.
#     - Pieces start near the bar, so movement cost is a useful heuristic.
#   combo:
#     - Find the largest short bar by the same "largest monochrome component"
#       rule; the bbox is no longer the whole grid, so notch detection must rely
#       on the bar's own bbox boundary rather than global borders.
#     - Enumerate root placements only if a candidate piece really plugs a notch
#       and extends outward through that notch's side. This removes many false
#       matches created by the short-bar geometry.
#     - Search scores partial explanations, so unused decoys are allowed.
#     - Because pieces live in corners, movement cost becomes only a weak tie-
#       breaker; the solver therefore keeps a wider root candidate pool per notch.
#   concrete_example:
#     - Suppose a horizontal bar occupies columns 6..22 and rows 12..17. A real
#       top-notch chain may have its root piece scattered in the top-left corner,
#       a continuation in the top-right, and a decoy in the bottom-left.
#     - The solver must first find the notch inside the bar bbox, then notice
#       that only one corner piece can cover that hole while sticking upward, and
#       finally reject the bottom-left decoy because it cannot finish any cavity.
# complexity_calibration:
#   - The bar length is kept in a narrow but visibly short range (15-19 cells on
#     a 30x30 grid). This makes the "short bar" change obvious without starving
#     the corner regions of space.
#   - Real chain count remains 1-2 and chain lengths 2-3, preserving the core
#     rule. Decoys are limited to 1-2, enough to force elimination but not to
#     flood the board.
#   - At least one sample-level cue repeats across examples: the input always has
#     more scattered components than the output, and those components live in the
#     four corners rather than flanking the bar.
# implicit_assumptions_review:
#   - `bar = comps[0]`: still valid because the bar stays much larger than each
#     individual piece. The generator enforces piece areas 8-13 and bar area 75+.
#   - `bar_notches(~bar_mask)` originally worked because the bar bbox equaled the
#     grid span along one axis. In the combo it must be interpreted strictly as
#     holes opening to the bar bbox, not to the grid border; the helper already
#     does this, so the solver keeps that invariant.
#   - Root placement in the original code implicitly tolerated candidates that
#     covered a hole but did not visibly protrude out through the correct side.
#     With a short bar that creates extra ambiguity near bar ends, so the combo
#     solver explicitly requires outward extension on the notch side.
#   - `if not remaining: return True` is invalid once decoys exist. The solver
#     therefore keeps the best partially explained state instead of consuming all
#     components.
# placement_legality:
#   - The generator builds the target output first, then relocates every piece to
#     a corner rectangle using normalized local coordinates.
#   - Corner placement rejects overlap and 4-neighbor touching. Regions are fully
#     outside both the bar's row span and column span, so pieces cannot touch the
#     bar either.
#   - Notches are carved only from boundary bands of the short bar. Decoy notches
#     are added only if the remaining bar stays 4-connected and every empty cell
#     in the bar bbox still reaches the bbox boundary.
# visual_observability:
#   - Across examples, humans can see three stable changes at once:
#       1. the bar stops before the grid edges,
#       2. sprites sit in corner zones instead of left/right or top/bottom bands,
#       3. one or two corner sprites never appear in the assembled output.
#   - No invisible ordering such as color id or generation order is used. The
#     discrimination cue is purely geometric: fit a notch, preserve strip width,
#     and allow the cavity chain to terminate cleanly.
# visual_cleanliness:
#   - Corner rectangles keep scattered pieces visually separated from the central
#     assembly zone.
#   - Shortening the bar also opens whitespace near its endpoints, making the
#     "not full span" property easy to notice rather than hidden by clutter.
# color_reasoning:
#   - Colors stay per-piece and mostly distinct. The rule does not depend on any
#     color mapping; colors only let humans track which sprite moved where.
# generator_checks_against_solver:
#   - `main()` assumes each true root piece is the unique piece that can plug its
#     notch while protruding through that notch side. The generator validates this.
#   - `main()` assumes decoys cannot complete a chain to a better score than the
#     true solution. One active decoy is built from a damaged tail-like piece and
#     passive decoys avoid the real strip widths, then `generate_input()` keeps
#     only instances where `main(input) == output`.
#   - `main()` assumes every corner region exists. The generator keeps bar spans
#     central enough that all four corner rectangles have non-zero area.

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


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=4):
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(max(40, target * 20)):
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


def _root_match_exists(shape, hole, H, W, bar_occ, side=None, bar_bbox=None) -> bool:
    for top, left in _hole_positions(hole, shape):
        pts = _placement_pts(shape, top, left, H, W, bar_occ, must_cover=hole)
        if pts is None:
            continue
        if side is not None and bar_bbox is not None and not _extends_from_side(pts, side, bar_bbox):
            continue
        return True
    return False


def _extends_from_side(pts, side: str, bar_bbox) -> bool:
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar_bbox)
    if side == "top":
        return any(r < bar_r0 for r, _ in pts)
    if side == "bottom":
        return any(r > bar_r1 for r, _ in pts)
    if side == "left":
        return any(c < bar_c0 for _, c in pts)
    return any(c > bar_c1 for _, c in pts)


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


def _coords_bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _chain_sequence_is_solver_valid(chain_info, H: int, W: int, bar_occ, bar_bbox) -> bool:
    axis = chain_info["axis"]
    A = int(chain_info["A"])
    B = int(chain_info["B"])
    side = chain_info["side"]
    notch = set(chain_info["notch"])
    occ = set(bar_occ)
    prev_coords = None

    for piece_idx, piece in enumerate(chain_info["pieces"]):
        coords = set(piece["coords"])
        shape = _shape_from_cells(coords)
        top, left, _, _ = _coords_bbox(coords)

        if piece_idx == 0:
            pts = _placement_pts(shape, top, left, H, W, bar_occ, must_cover=notch)
            if pts != coords or not _extends_from_side(coords, side, bar_bbox):
                return False
        else:
            prev_r0, prev_c0, prev_r1, prev_c1 = _coords_bbox(prev_coords)
            if axis == "cols":
                cavity = {
                    (r, c)
                    for r in range(prev_r0, prev_r1 + 1)
                    for c in range(A, B + 1)
                    if (r, c) not in occ and (r, c) not in prev_coords
                }
            else:
                cavity = {
                    (r, c)
                    for r in range(A, B + 1)
                    for c in range(prev_c0, prev_c1 + 1)
                    if (r, c) not in occ and (r, c) not in prev_coords
                }
            if not cavity:
                return False
            pts = _placement_pts(shape, top, left, H, W, occ, must_cover=cavity, strip=(axis, A, B))
            if pts != coords or not _touches(coords, prev_coords):
                return False

        occ |= coords
        prev_coords = coords

    if prev_coords is None:
        return False

    tail_r0, tail_c0, tail_r1, tail_c1 = _coords_bbox(prev_coords)
    if axis == "cols":
        tail_cavity = {
            (r, c)
            for r in range(tail_r0, tail_r1 + 1)
            for c in range(A, B + 1)
            if (r, c) not in occ and (r, c) not in prev_coords
        }
    else:
        tail_cavity = {
            (r, c)
            for r in range(A, B + 1)
            for c in range(tail_c0, tail_c1 + 1)
            if (r, c) not in occ and (r, c) not in prev_coords
        }
    return not tail_cavity


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
    bar_bbox = (bar_r0, bar_c0, bar_r1, bar_c1)

    root_cands = []
    for ni, notch in enumerate(notches):
        notch_cands = []
        axis = notch["axis"]
        side = notch["side"]
        for pi, piece in enumerate(pieces):
            shape = piece["shape"]
            for top, left in _hole_positions(notch["hole"], shape):
                pts = _placement_pts(shape, top, left, H, W, bar_occ, must_cover=notch["hole"])
                if pts is None:
                    continue
                if not _extends_from_side(pts, side, bar_bbox):
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + shape["w"] - 1)
                else:
                    A, B = int(top), int(top + shape["h"] - 1)
                move_cost = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                notch_cands.append(
                    (
                        (int(move_cost), -int(notch["mouth_n"]), -int(len(pts - bar_occ))),
                        {
                            "ni": ni,
                            "pi": pi,
                            "top": int(top),
                            "left": int(left),
                            "axis": axis,
                            "A": A,
                            "B": B,
                            "shape": shape,
                        },
                    )
                )
        notch_cands.sort(key=lambda x: x[0])
        root_cands.extend(item[1] for item in notch_cands[:8])

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
                moves.append((int(move_cost), -int(len(pts)), pi, int(top), int(left), pts, shape))
        moves.sort(key=lambda x: (x[0], x[1], x[2]))
        return moves[:10]

    def search(out_grid, occ_set, chains, remaining):
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = state_key(out_grid, chains, placed_cnt)
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()

        for chain in chains:
            for _cost, _neg_area, pi, top, left, pts, shape in gen_extend_moves(out_grid, occ_set, chain, remaining):
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

    max_roots = min(2, len(root_cands), len(notches), len(pieces))
    for k in range(1, max_roots + 1):
        for roots_sel in combinations(range(len(root_cands)), k):
            used_p = set()
            used_n = set()
            occ0 = set(bar_occ)
            out0 = np.full((H, W), int(bg), dtype=int)
            for r, c in bar_occ:
                out0[r, c] = bar_color

            chains = []
            ok = True
            for idx in roots_sel:
                root = root_cands[idx]
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


def _choose_bar_bbox(rng: random.Random, H: int, W: int, orientation: str):
    thickness = rng.randint(5, 7)
    span_len = rng.randint(15, 19)
    if orientation == "horizontal":
        top_center = int(rng.uniform(0.38, 0.58) * H)
        bar_r0 = max(2, top_center - thickness // 2)
        bar_r1 = bar_r0 + thickness - 1
        if bar_r1 >= H - 2:
            bar_r1 = H - 3
            bar_r0 = bar_r1 - thickness + 1
        start_min = 5
        start_max = W - span_len - 5
        preferred = [s for s in range(start_min, start_max + 1) if abs((2 * s + span_len - 1) / 2 - (W - 1) / 2) <= 3.5]
        bar_c0 = rng.choice(preferred if preferred else list(range(start_min, start_max + 1)))
        bar_c1 = bar_c0 + span_len - 1
    else:
        left_center = int(rng.uniform(0.38, 0.58) * W)
        bar_c0 = max(2, left_center - thickness // 2)
        bar_c1 = bar_c0 + thickness - 1
        if bar_c1 >= W - 2:
            bar_c1 = W - 3
            bar_c0 = bar_c1 - thickness + 1
        start_min = 5
        start_max = H - span_len - 5
        preferred = [s for s in range(start_min, start_max + 1) if abs((2 * s + span_len - 1) / 2 - (H - 1) / 2) <= 3.5]
        bar_r0 = rng.choice(preferred if preferred else list(range(start_min, start_max + 1)))
        bar_r1 = bar_r0 + span_len - 1
    return bar_r0, bar_c0, bar_r1, bar_c1


def _bar_corner_regions(H: int, W: int, bar_r0: int, bar_c0: int, bar_r1: int, bar_c1: int):
    return {
        "tl": (0, 0, bar_r0 - 1, bar_c0 - 1),
        "tr": (0, bar_c1 + 1, bar_r0 - 1, W - 1),
        "bl": (bar_r1 + 1, 0, H - 1, bar_c0 - 1),
        "br": (bar_r1 + 1, bar_c1 + 1, H - 1, W - 1),
    }


def _scatter_piece_to_region(rng, grid, placed_occ, piece, region):
    y0, x0, y1, x1 = map(int, region)
    if y0 > y1 or x0 > x1:
        return False
    coords = piece["coords"]
    color = int(piece["color"])
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    p_r0, p_c0 = min(rs), min(cs)
    rel = [(r - p_r0, c - p_c0) for r, c in coords]
    p_h = max(rs) - p_r0 + 1
    p_w = max(cs) - p_c0 + 1
    if p_h > y1 - y0 + 1 or p_w > x1 - x0 + 1:
        return False

    top_choices = list(range(y0, y1 - p_h + 2))
    left_choices = list(range(x0, x1 - p_w + 2))
    rng.shuffle(top_choices)
    rng.shuffle(left_choices)
    tries = 0
    for top in top_choices:
        for left in left_choices:
            tries += 1
            if tries > 700:
                return False
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


def _all_bar_holes_touch_boundary(bar_cells, bar_r0, bar_c0, bar_r1, bar_c1):
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr, cc = pts[:, 0], pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return False
    return True


def _input_metrics(inp: np.ndarray):
    H, W = inp.shape
    bg = get_bg(inp)
    comps = find_components(inp, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    bar = comps[0]
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    corner_regions = _bar_corner_regions(H, W, bar_r0, bar_c0, bar_r1, bar_c1)
    used_corners = set()
    for comp in comps[1:]:
        rs = [int(r) for r, _ in comp["coords"]]
        cs = [int(c) for _, c in comp["coords"]]
        rr0, rr1 = min(rs), max(rs)
        cc0, cc1 = min(cs), max(cs)
        for name, (r0, c0, r1, c1) in corner_regions.items():
            if r0 <= rr0 and rr1 <= r1 and c0 <= cc0 and cc1 <= c1:
                used_corners.add(name)
                break
    span_ratio = (bar_c1 - bar_c0 + 1) / W if (bar_r1 - bar_r0 + 1) <= (bar_c1 - bar_c0 + 1) else (bar_r1 - bar_r0 + 1) / H
    return {
        "piece_count": len(comps) - 1,
        "corner_count": len(used_corners),
        "span_ratio": span_ratio,
    }


def _gen_attempt():
    rng = random.Random()
    H, W = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    bar_color = rng.randint(1, 9)
    bar_r0, bar_c0, bar_r1, bar_c1 = _choose_bar_bbox(rng, H, W, orientation)

    if orientation == "horizontal":
        space_before, space_after = bar_r0, H - 1 - bar_r1
    else:
        space_before, space_after = bar_c0, W - 1 - bar_c1

    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    if min(bar_r0, H - 1 - bar_r1, bar_c0, W - 1 - bar_c1) < 5:
        return None

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
    chain_infos = []
    occ_set = set(bar_cells)
    chain_axis = "cols" if orientation == "horizontal" else "rows"

    for chain_len in chain_lens:
        strip_width = rng.randint(4, 5)
        chain_dim_need = chain_len * 2 + 3

        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "top" else space_after) >= chain_dim_need), None)
            if side is None or bar_w < strip_width + 2:
                return None
            sA = rng.randint(bar_c0 + 1, bar_c1 - strip_width)
            sB = sA + strip_width - 1
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "left" else space_after) >= chain_dim_need), None)
            if side is None or bar_h < strip_width + 2:
                return None
            sA = rng.randint(bar_r0 + 1, bar_r1 - strip_width)
            sB = sA + strip_width - 1

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
                    retries=6,
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
                    if side == "top":
                        ext_span = (max(0, prev_r0 - chain_dim), prev_r0 - 1)
                    else:
                        ext_span = (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    ext = {
                        (r, c)
                        for r in range(ext_span[0], ext_span[1] + 1)
                        for c in range(sA, sB + 1)
                        if (r, c) not in occ_set
                    } if ext_span[0] <= ext_span[1] else set()
                else:
                    if side == "left":
                        ext_span = (max(0, prev_c0 - chain_dim), prev_c0 - 1)
                    else:
                        ext_span = (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
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
                    retries=6,
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

    num_decoy_notches = rng.randint(3, 5)
    carved_decoys = 0
    for _ in range(600):
        if carved_decoys >= num_decoy_notches:
            break
        depth = rng.randint(1, 2)
        if orientation == "horizontal":
            ds = rng.choice(["top", "bottom"])
            dr0 = bar_r0 if ds == "top" else bar_r1 - depth + 1
            dr1 = bar_r0 + depth - 1 if ds == "top" else bar_r1
            seg_w = rng.randint(1, min(4, bar_w - 2))
            dc0 = rng.randint(bar_c0 + 1, bar_c1 - seg_w)
            dc1 = dc0 + seg_w - 1
        else:
            ds = rng.choice(["left", "right"])
            dc0 = bar_c0 if ds == "left" else bar_c1 - depth + 1
            dc1 = bar_c0 + depth - 1 if ds == "left" else bar_c1
            seg_h = rng.randint(1, min(4, bar_h - 2))
            dr0 = rng.randint(bar_r0 + 1, bar_r1 - seg_h)
            dr1 = dr0 + seg_h - 1

        d_avail = {(r, c) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        all_holes = set().union(*notch_holes) if notch_holes else set()
        d_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
        if not d_avail:
            continue
        area = min(len(d_avail), rng.randint(1, min(6, len(d_avail))))
        dn = _grow_connected(rng, d_avail, area, {rng.choice(list(d_avail))})
        if dn is None or not _connected(bar_cells - dn):
            continue
        new_bar = bar_cells - dn
        if not _all_bar_holes_touch_boundary(new_bar, bar_r0, bar_c0, bar_r1, bar_c1):
            continue
        bar_cells = new_bar
        notch_holes.append(dn)
        carved_decoys += 1
    if carved_decoys < 3:
        return None

    if not _all_bar_holes_touch_boundary(bar_cells, bar_r0, bar_c0, bar_r1, bar_c1):
        return None

    bar_occ = set(bar_cells)
    bar_bbox = (bar_r0, bar_c0, bar_r1, bar_c1)
    for info in chain_infos:
        for piece_idx, piece in enumerate(info["pieces"]):
            shape = _shape_from_cells(piece["coords"])
            matches = [hole for hole in notch_holes if _root_match_exists(shape, hole, H, W, bar_occ, side=info["side"], bar_bbox=bar_bbox)]
            if piece_idx == 0:
                if len(matches) != 1 or matches[0] != info["notch"]:
                    return None
            elif matches:
                return None
        if not _chain_sequence_is_solver_valid(info, H, W, bar_occ, bar_bbox):
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
        if len(info["pieces"]) < 2:
            continue
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

    corner_regions = _bar_corner_regions(H, W, bar_r0, bar_c0, bar_r1, bar_c1)
    if any(r0 > r1 or c0 > c1 for r0, c0, r1, c1 in corner_regions.values()):
        return None

    placed_occ = set(bar_cells)
    all_pieces = list(all_chain_pieces) + decoy_pieces
    all_pieces.sort(
        key=lambda piece: (
            -max(r for r, _ in piece["coords"]) + min(r for r, _ in piece["coords"]),
            -max(c for _, c in piece["coords"]) + min(c for _, c in piece["coords"]),
            -len(piece["coords"]),
        )
    )
    corner_names = ["tl", "tr", "bl", "br"]
    assignments = {}
    for idx, piece in enumerate(all_pieces):
        if idx < 4:
            assignments[id(piece)] = corner_names[idx]
        else:
            assignments[id(piece)] = rng.choice(corner_names)

    for piece in all_pieces:
        preferred = assignments[id(piece)]
        regions_try = [preferred] + [name for name in corner_names if name != preferred]
        placed = False
        for name in regions_try:
            if _scatter_piece_to_region(rng, input_grid, placed_occ, piece, corner_regions[name]):
                assignments[id(piece)] = name
                placed = True
                break
        if not placed:
            return None

    if len({assignments[id(piece)] for piece in all_pieces}) < 4:
        return None

    metrics = _input_metrics(input_grid)
    if metrics["corner_count"] < 4 or not (0.50 <= metrics["span_ratio"] <= 0.70) or metrics["piece_count"] < len(all_chain_pieces) + 1:
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
