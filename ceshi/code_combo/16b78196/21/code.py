# Combo variant of 16b78196 (index 21)
# combo_spec: D13/2+D15/1
# Name: 终止标记精灵+方向箭头
#
# Design analysis
# ---------------
# a) Concrete behavior and flow trace
#    Original task:
#    - The largest monochrome component is a central full-span bar with several
#      boundary-opening notches.
#    - A scattered root sprite plugs one used notch.
#    - Later sprites fill the background cavity inside the previous sprite bbox,
#      staying locked to the same strip width.
#    - A chain stops only when the tail sprite leaves no cavity in its own bbox.
#
#    Combo task:
#    - Keep the same bar / notch / strip / cavity-filling mechanics.
#    - Add a grey 3-cell L marker near each active notch. Its orientation encodes
#      one of {top, bottom, left, right}, so it explicitly tells the solver which
#      side is active and therefore which direction the chain grows.
#    - Reserve the same visible grey color for the terminal sprite of each chain.
#      This terminal sprite is a normal movable piece, not a marker: it has area
#      8-13 and is scattered like the other sprites.
#    - Once the grey terminal sprite is inserted, the chain closes immediately,
#      even though its strip-aligned bbox still contains 1-6 background cells.
#
#    Small trace:
#    - Suppose the bar is horizontal and has three top/bottom notches, but only
#      one top notch has a grey L above it. The solver must first decode that L
#      as "grow upward from this marked top notch".
#    - It places a non-grey root into that notch, then maybe one more non-grey
#      piece that fills the root cavity, then finally a grey terminal piece.
#    - Unlike the original task, the final grey piece is allowed to leave a black
#      cavity inside its own bbox; that visible residual cavity is the evidence
#      that termination is semantic, not geometric.
#
# b) Complexity calibration
#    - Only one marker vocabulary is introduced: a 3-cell L, repeated 1-2 times.
#      This keeps symbolic load small.
#    - Only one special sprite color is introduced: grey. It is reused across
#      examples for both arrow markers and terminal sprites, but the two roles are
#      visually separable by shape/size: 3-cell 2x2 L components are markers,
#      8-13-cell irregular components are movable terminal pieces.
#    - Decoy notches remain present, so the arrow is not decorative.
#    - Generator enforces positive terminal residual cavity, so the terminal rule
#      cannot collapse back into the original "stop when cavity vanishes".
#
# c) Implicit-assumption review for main()
#    - Original assumption: every non-bar component is a movable sprite.
#      Broken: the combo input also contains small grey L markers, so main() must
#      classify components into markers vs real pieces before search starts.
#    - Original assumption: valid root notches are found purely from geometry.
#      Broken: only marker-selected notches may start chains.
#    - Original assumption: any piece color may serve as a root or internal node.
#      Broken: grey movable pieces are terminal-only and therefore excluded from
#      root generation and from non-terminal extension states.
#    - Original assumption: a chain ends only when the head cavity is empty.
#      Broken: in the combo task, a chain may be closed while cavity remains, so
#      the search state must carry a `closed` flag per chain.
#    - Original assumption: `if not remaining: return True` is enough.
#      Broken: all chains must additionally have consumed a grey terminal.
#    - New interaction check: marker side and notch side must agree. The solver
#      maps each marker only to notches on the same side of the bar.
#
# d) Placement legality
#    - Markers sit one cell away from the bar and never overlap or touch each
#      other, so they remain isolated components.
#    - Scattered pieces are placed away from the bar and markers.
#    - Used notches and decoys are kept sufficiently separated along each side to
#      make marker-to-notch mapping stable and visually clean.
#    - Terminal pieces obey the same strip-width and connectivity rules as the
#      other chain pieces.
#
# e) Visual observability
#    - From 2-4 examples, a human sees that only marked notches produce chains in
#      the output, while unmarked notches stay empty.
#    - The final piece of every chain is grey and still contains black holes in
#      its bbox in the output. That repeated residual-cavity pattern is the key
#      cue that grey means "stop here".
#    - No hidden use of numeric color order is needed; everything is visible from
#      shape, placement, and repeated grey semantics.
#
# f) Visual cleanliness
#    - The board still has the same coarse layout as the source task: a single
#      bar in the middle, scattered sprites around it.
#    - Only 1-2 small arrow markers are added, and each is placed adjacent to one
#      active notch rather than deep inside cluttered regions.
#    - Decoy notches are kept away from active marker spans, reducing ambiguity.
#
# g) Change inferability
#    - Grey has only one semantic family: instruction/termination. The marker role
#      is tied to a tiny fixed L-shape, while the terminal role is tied to a large
#      movable irregular sprite at the chain tail.
#    - Because both signals repeat across chains/examples, the solver can infer:
#      grey L near a notch chooses the chain direction/start, and grey irregular
#      sprite ends the chain even if cavity remains.
#
# new_main_assumptions_checked_against_generator:
#    - Marker components are always grey, area 3, and fit in a 2x2 bbox.
#    - Real terminal sprites are also grey but always area 8-13, so they never
#      collide with the marker classifier.
#    - Every active chain has exactly one marker and exactly one grey terminal.
#    - Every non-terminal placed piece leaves a positive residual cavity, and every
#      terminal piece also leaves a positive residual cavity.

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg


MARKER_COLOR = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MARKER_SHAPES = {
    "top": {(0, 0), (1, 0), (1, 1)},
    "bottom": {(0, 0), (0, 1), (1, 1)},
    "left": {(0, 1), (1, 0), (1, 1)},
    "right": {(0, 0), (0, 1), (1, 0)},
}
REL_TO_SIDE = {frozenset(v): k for k, v in MARKER_SHAPES.items()}


def _shape_from_coords(coords: np.ndarray, bbox):
    r0, c0, r1, c1 = map(int, bbox)
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": int(r1 - r0 + 1), "w": int(c1 - c0 + 1)}


def _pixels(shape, top, left):
    return {(top + dr, left + dc) for dr, dc in shape["rel"]}


def _touches_any(pts, other):
    for r, c in pts:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in other:
                return True
    return False


def _connected(cells):
    if len(cells) <= 1:
        return True
    cells = set(cells)
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _classify_marker(comp, shape):
    if int(comp["color"]) != MARKER_COLOR:
        return None
    if int(comp["area"]) != 3:
        return None
    if int(shape["h"]) > 2 or int(shape["w"]) > 2:
        return None
    return REL_TO_SIDE.get(frozenset(shape["rel_set"]))


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

        hole_gl = {(bar_r0 + int(rr), bar_c0 + int(cc)) for rr, cc in pts}
        side, mouth_n = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )
        axis = "cols" if side in ("top", "bottom") else "rows"
        if side == "top":
            mouth = {(bar_r0, bar_c0 + int(cc)) for cc in c[r == 0]}
        elif side == "bottom":
            mouth = {(bar_r0 + h - 1, bar_c0 + int(cc)) for cc in c[r == h - 1]}
        elif side == "left":
            mouth = {(bar_r0 + int(rr), bar_c0) for rr in r[c == 0]}
        else:
            mouth = {(bar_r0 + int(rr), bar_c0 + w - 1) for rr in r[c == w - 1]}
        notches.append({"axis": axis, "side": side, "hole": hole_gl, "mouth": mouth, "mouth_n": mouth_n})
    return notches


def _marker_outside_bar(marker_bbox, bar_bbox, side):
    mr0, mc0, mr1, mc1 = map(int, marker_bbox)
    br0, bc0, br1, bc1 = map(int, bar_bbox)
    if side == "top":
        return mr1 < br0
    if side == "bottom":
        return mr0 > br1
    if side == "left":
        return mc1 < bc0
    return mc0 > bc1


def _marker_notch_overlap(marker_bbox, notch):
    mr0, mc0, mr1, mc1 = map(int, marker_bbox)
    if notch["side"] in ("top", "bottom"):
        cols = [c for _, c in notch["mouth"]]
        return max(0, min(mc1, max(cols)) - max(mc0, min(cols)) + 1)
    rows = [r for r, _ in notch["mouth"]]
    return max(0, min(mr1, max(rows)) - max(mr0, min(rows)) + 1)


def _map_markers_to_notches(markers, notches, bar_bbox):
    # This combo requires explicit direction markers. If they are missing or
    # cannot be matched to notches, do not fall back to the original geometric
    # rule where any notch may start a chain.
    if not markers:
        return []

    picked = []
    used = set()
    for marker in markers:
        side = marker["side"]
        if not _marker_outside_bar(marker["bbox"], bar_bbox, side):
            continue
        marker_pts = {(int(r), int(c)) for r, c in marker["coords"]}
        scored = []
        for ni, notch in enumerate(notches):
            if ni in used or notch["side"] != side:
                continue
            overlap = _marker_notch_overlap(marker["bbox"], notch)
            if overlap <= 0:
                continue
            dist = min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in marker_pts for r2, c2 in notch["mouth"])
            scored.append(((overlap, -dist, int(notch["mouth_n"])), ni))
        if not scored:
            continue
        scored.sort(reverse=True)
        chosen = scored[0][1]
        used.add(chosen)
        picked.append(chosen)
    return picked


def _piece_cavity(piece_cells, occ_before, orientation, A, B):
    prs = [r for r, _ in piece_cells]
    pcs = [c for _, c in piece_cells]
    r0, r1 = min(prs), max(prs)
    c0, c1 = min(pcs), max(pcs)
    if orientation == "horizontal":
        bbox_strip = {(r, c) for r in range(r0, r1 + 1) for c in range(A, B + 1)}
    else:
        bbox_strip = {(r, c) for r in range(A, B + 1) for c in range(c0, c1 + 1)}
    return {cell for cell in bbox_strip if cell not in occ_before and cell not in piece_cells}


def _interval_too_close(start_a: int, size_a: int, start_b: int, size_b: int, min_gap: int) -> bool:
    end_a = start_a + size_a - 1
    end_b = start_b + size_b - 1
    if end_a < start_b:
        gap = start_b - end_a - 1
    elif end_b < start_a:
        gap = start_a - end_b - 1
    else:
        gap = -1
    return gap < min_gap


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    bar = comps[0]
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
    markers = []
    for comp in comps[1:]:
        shape = _shape_from_coords(comp["coords"], comp["bbox"])
        marker_side = _classify_marker(comp, shape)
        if marker_side is not None:
            markers.append(
                {
                    "bbox": tuple(map(int, comp["bbox"])),
                    "coords": [(int(r), int(c)) for r, c in comp["coords"]],
                    "side": marker_side,
                }
            )
            continue
        r0, c0 = map(int, comp["bbox"][:2])
        color = int(comp["color"])
        pieces.append(
            {
                "color": color,
                "is_terminal": color == MARKER_COLOR,
                "shape": shape,
                "orig": (r0, c0),
            }
        )

    notches = _bar_notches(bar_mask, bar_r0, bar_c0)
    active_notch_ids = _map_markers_to_notches(markers, notches, tuple(map(int, bar["bbox"])))
    terminal_count = sum(int(p["is_terminal"]) for p in pieces)
    expected_chains = len(active_notch_ids)

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

    def hole_positions(hole, shape):
        pos = set()
        for rH, cH in hole:
            for lr, lc in shape["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    def head_cavity(shape, top, left, occ_after):
        return {
            (r, c)
            for r in range(top, top + shape["h"])
            for c in range(left, left + shape["w"])
            if (r, c) not in occ_after
        }

    root_cands = []
    seen_roots = set()
    for ni in active_notch_ids:
        notch = notches[ni]
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            if piece["is_terminal"]:
                continue
            shape = piece["shape"]
            for top, left in hole_positions(notch["hole"], shape):
                pts = placement_pts(shape, top, left, bar_occ, must_cover=notch["hole"])
                if pts is None:
                    continue
                if not head_cavity(shape, int(top), int(left), bar_occ | pts):
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + shape["w"] - 1)
                else:
                    A, B = int(top), int(top + shape["h"] - 1)
                key = (ni, pi, int(top), int(left))
                if key in seen_roots:
                    continue
                seen_roots.add(key)
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), -int(notch["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": shape},
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands]

    best_out = out.copy()
    best_key = None

    def search(out_grid, occ_set, chains, remaining):
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        closed_cnt = sum(1 for ch in chains if ch["closed"])
        key = (placed_cnt, closed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return all(ch["closed"] for ch in chains)

        def gen_extend_moves(chain, remaining_set):
            if chain["closed"]:
                return []
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = _pixels(head["shape"], head["top"], head["left"])
            cavity = head_cavity(head["shape"], int(head["top"]), int(head["left"]), occ_set)
            if not cavity:
                return []

            moves = []
            seen_moves = set()
            for pi in remaining_set:
                piece = pieces[pi]
                shape = piece["shape"]
                for top, left in aligned_positions(axis, A, cavity, shape):
                    pts = placement_pts(shape, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                    if pts is None:
                        continue
                    if not _touches_any(pts, head_pts):
                        continue
                    new_cavity = head_cavity(shape, int(top), int(left), occ_set | pts)
                    if (not piece["is_terminal"]) and not new_cavity:
                        continue
                    move_key = (pi, int(top), int(left))
                    if move_key in seen_moves:
                        continue
                    seen_moves.add(move_key)
                    mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                    moves.append(((int(piece["is_terminal"]), int(mv)), pi, int(top), int(left), pts, shape))
            moves.sort(key=lambda x: x[0])
            return [(pi, top, left, pts, shape) for _, pi, top, left, pts, shape in moves]

        for chain in chains:
            moves = gen_extend_moves(chain, remaining)
            if not moves:
                continue
            for pi, top, left, pts, shape in moves:
                piece = pieces[pi]
                for r, c in pts:
                    out_grid[r, c] = int(piece["color"])
                old_head = chain["head"]
                old_closed = chain["closed"]
                chain["head"] = {"top": int(top), "left": int(left), "shape": shape}
                chain["closed"] = bool(piece["is_terminal"])
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                chain["head"] = old_head
                chain["closed"] = old_closed
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
        return False

    if not cand_roots or expected_chains <= 0 or terminal_count != expected_chains:
        return best_out

    root_count = expected_chains
    for roots_sel in combinations(range(len(cand_roots)), root_count):
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
                    "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                    "closed": False,
                }
            )

        if not ok or len(used_n) != root_count:
            continue
        remaining = set(range(len(pieces))) - used_p
        if search(out0, occ0, chains, remaining):
            return best_out

    return best_out


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=3):
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


def _make_marker_cells(side, bar_r0, bar_r1, bar_c0, bar_c1, A, B, rng):
    rel = MARKER_SHAPES[side]
    if side in ("top", "bottom"):
        start = A if rng.random() < 0.5 else B - 1
        row0 = bar_r0 - 3 if side == "top" else bar_r1 + 2
        return {(row0 + dr, start + dc) for dr, dc in rel}
    start = A if rng.random() < 0.5 else B - 1
    col0 = bar_c0 - 3 if side == "left" else bar_c1 + 2
    return {(start + dr, col0 + dc) for dr, dc in rel}


def _gen_attempt(return_meta=False):
    import random as _random

    rng = _random.Random()
    H, W = 30, 30
    bg = 0
    residual_lo, residual_hi = 1, 6
    marker_gap = 4

    orientation = rng.choice(["horizontal", "vertical"])
    bar_color = rng.choice([c for c in range(1, 10) if c != MARKER_COLOR])
    thickness = rng.randint(5, 7)

    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * H)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = min(H - 1, bar_r0 + thickness - 1)
        bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        space_before, space_after = bar_r0, H - 1 - bar_r1
        side_names = ["top", "bottom"]
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = min(W - 1, bar_c0 + thickness - 1)
        bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        space_before, space_after = bar_c0, W - 1 - bar_c1
        side_names = ["left", "right"]

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    num_chains = rng.randint(1, 2)
    chain_lens = [3] if num_chains == 1 else [rng.randint(2, 3) for _ in range(2)]
    if num_chains == 2:
        while not (3 <= sum(chain_lens) <= 6):
            chain_lens = [rng.randint(2, 3) for _ in range(2)]

    avail_colors = [c for c in range(1, 10) if c not in (bar_color, MARKER_COLOR)]
    rng.shuffle(avail_colors)
    color_idx = 0

    all_chain_pieces = []
    notch_holes = []
    marker_cells_all = set()
    occ_set = set(bar_cells)
    chain_metas = []
    protected_intervals = {side: [] for side in side_names}

    for chain_len in chain_lens:
        strip_width = rng.randint(4, 5)
        side = None
        sA = sB = None
        side_options = side_names[:]
        rng.shuffle(side_options)
        for _ in range(80):
            candidate_side = rng.choice(side_options)
            free_space = space_before if candidate_side in ("top", "left") else space_after
            if free_space < chain_len * 2 + 4:
                continue
            start_hi = (W if orientation == "horizontal" else H) - strip_width - 1
            if start_hi < 1:
                continue
            candidate_A = rng.randint(1, start_hi)
            if any(
                _interval_too_close(candidate_A, strip_width, prev_start, prev_size, marker_gap)
                for prev_start, prev_size in protected_intervals[candidate_side]
            ):
                continue
            side = candidate_side
            sA = candidate_A
            sB = candidate_A + strip_width - 1
            protected_intervals[side].append((sA, strip_width))
            break
        if side is None:
            return None

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
        chain_residuals = []
        for piece_idx in range(chain_len):
            is_terminal = piece_idx == chain_len - 1
            if is_terminal:
                piece_color = MARKER_COLOR
            else:
                if color_idx >= len(avail_colors):
                    return None
                piece_color = avail_colors[color_idx]
                color_idx += 1

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
                lo = max(8, int(bbox_area * 0.54))
                hi = min(13, int(bbox_area * 0.78))
                if lo > hi:
                    return None
                piece_cells = None
                for _ in range(20):
                    target = rng.randint(lo, hi)
                    avail = {
                        (r, c)
                        for r in range(pr0, pr1 + 1)
                        for c in range(pc0, pc1 + 1)
                        if (r, c) in notch or (r, c) not in occ_set
                    }
                    cand = _grow_connected(
                        rng,
                        avail,
                        target,
                        notch,
                        must_span_axis="cols" if orientation == "horizontal" else "rows",
                        span_range=(sA, sB),
                        retries=5,
                    )
                    if cand is None:
                        continue
                    if not any(not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1) for r, c in cand):
                        continue
                    residual = _piece_cavity(cand, occ_set, orientation, sA, sB)
                    if not (residual_lo <= len(residual) <= residual_hi):
                        continue
                    piece_cells = cand
                    chain_residuals.append(len(residual))
                    break
                if piece_cells is None:
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
                    es = (max(0, prev_r0 - chain_dim), prev_r0 - 1) if side == "top" else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    ext = {(r, c) for r in range(es[0], es[1] + 1) for c in range(sA, sB + 1) if (r, c) not in occ_set} if es[0] <= es[1] else set()
                else:
                    cavity = {
                        (r, c)
                        for r in range(sA, sB + 1)
                        for c in range(prev_c0, prev_c1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                    es = (max(0, prev_c0 - chain_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    ext = {(r, c) for r in range(sA, sB + 1) for c in range(es[0], es[1] + 1) if (r, c) not in occ_set} if es[0] <= es[1] else set()

                if not cavity or not ext:
                    return None
                all_avail = cavity | ext
                lo = max(8, len(cavity) + 1)
                hi = min(13, len(all_avail))
                if lo > hi:
                    return None
                piece_cells = None
                for _ in range(24):
                    target = rng.randint(lo, hi)
                    cand = _grow_connected(
                        rng,
                        all_avail,
                        target,
                        cavity,
                        must_span_axis="cols" if orientation == "horizontal" else "rows",
                        span_range=(sA, sB),
                        retries=5,
                    )
                    if cand is None or not cavity.issubset(cand):
                        continue
                    if not (cand & ext):
                        continue
                    residual = _piece_cavity(cand, occ_set, orientation, sA, sB)
                    if not (residual_lo <= len(residual) <= residual_hi):
                        continue
                    piece_cells = cand
                    chain_residuals.append(len(residual))
                    break
                if piece_cells is None:
                    return None

            if not (8 <= len(piece_cells) <= 13):
                return None
            chain_pieces.append({"color": piece_color, "coords": piece_cells, "terminal": is_terminal})
            occ_set |= piece_cells

        marker_cells = _make_marker_cells(side, bar_r0, bar_r1, bar_c0, bar_c1, sA, sB, rng)
        if any(r < 0 or r >= H or c < 0 or c >= W for r, c in marker_cells):
            return None
        if marker_cells & marker_cells_all:
            return None
        if _touches_any(marker_cells, marker_cells_all):
            return None
        marker_cells_all |= marker_cells

        all_chain_pieces.extend(chain_pieces)
        chain_metas.append(
            {
                "len": chain_len,
                "side": side,
                "strip": (sA, sB),
                "residuals": list(chain_residuals),
                "terminal_residual": chain_residuals[-1],
            }
        )

    if not (3 <= len(all_chain_pieces) <= 6):
        return None

    num_decoys = rng.randint(4, 7)
    decoys = 0
    for _ in range(700):
        if decoys >= num_decoys:
            break
        depth = rng.randint(1, 2)
        area = rng.randint(1, min(6, depth * 5))
        if orientation == "horizontal":
            ds = rng.choice(["top", "bottom"])
            dc0 = rng.randint(bar_c0, bar_c1)
            dc1 = min(bar_c1, dc0 + rng.randint(0, 4))
            size = dc1 - dc0 + 1
            if any(_interval_too_close(dc0, size, start, width, marker_gap) for start, width in protected_intervals[ds]):
                continue
            dr0 = bar_r0 if ds == "top" else bar_r1 - depth + 1
            dr1 = bar_r0 + depth - 1 if ds == "top" else bar_r1
        else:
            ds = rng.choice(["left", "right"])
            dr0 = rng.randint(bar_r0, bar_r1)
            dr1 = min(bar_r1, dr0 + rng.randint(0, 4))
            size = dr1 - dr0 + 1
            if any(_interval_too_close(dr0, size, start, width, marker_gap) for start, width in protected_intervals[ds]):
                continue
            dc0 = bar_c0 if ds == "left" else bar_c1 - depth + 1
            dc1 = bar_c0 + depth - 1 if ds == "left" else bar_c1

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
        protected_intervals[ds].append((dc0 if orientation == "horizontal" else dr0, size))
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
    for piece in all_chain_pieces:
        for r, c in piece["coords"]:
            output[r, c] = piece["color"]

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color
    for r, c in marker_cells_all:
        input_grid[r, c] = MARKER_COLOR

    placed_occ = set(bar_cells) | set(marker_cells_all)
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
            gap = rng.randint(3, 7)
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
            if _touches_any(pts, placed_occ):
                continue
            for r, c in pts:
                input_grid[r, c] = color
            placed_occ |= pts
            placed = True
            break
        if not placed:
            return None

    meta = {
        "orientation": orientation,
        "num_chains": num_chains,
        "chain_lens": list(chain_lens),
        "marker_count": len(chain_metas),
        "decoy_count": decoys,
        "marker_sides": [m["side"] for m in chain_metas],
        "terminal_residuals": [m["terminal_residual"] for m in chain_metas],
        "chain_metas": chain_metas,
    }
    return (input_grid, output, meta) if return_meta else (input_grid, output)


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
