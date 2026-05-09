# Combo variant of 16b78196 (index 19)
# combo_spec: D1/0+D5/1+D16/0
# Name: 颜色匹配+颜色交换+颜色重复

from __future__ import annotations

import random
import sys
from itertools import combinations

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/arc2_solver_select"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import components_mask, find_components, get_bg

"""
Design analysis for combo D1/0 + D5/1 + D16/0
==============================================

a) Concrete behavior and trace
  Original task:
    - try every scattered piece against every open notch of the central bar
    - geometry decides which root plugs which notch
    - after rooting, later pieces fill the previous bbox cavity inside that strip
    - placed pieces keep their input colors

  Combo task:
    - each used notch has a 1-cell outer cue; cue color says which color may root there
    - both used notches share the same cue color, and both root sprites share that same color
    - color alone therefore yields a 2 pieces x 2 notches ambiguity
    - shape and cavity geometry resolve which same-colored root belongs to which notch
    - after both chains are assembled, the two parallel chains swap colors depth-by-depth

  Small trace:
    - notch A cue = red, notch B cue = red
    - scattered roots R1 and R2 are both red but have different silhouettes
    - only R1 can legally cover notch A's hole while still leaving the right cavity pattern
    - only R2 can do the same for notch B
    - then depth-1 colors across the two chains are exchanged in the output

b) Complexity calibration
  - The generator always makes exactly two adjacent parallel chains on the same side of the bar.
  - Both roots reuse one color; deeper pieces use distinct colors so the post-assembly swap is visible.
  - Chain length is 2 or 3, keeping the puzzle legible while still forcing multi-step reasoning.
  - Root shapes are required to differ, so repeated color does not collapse into duplicate objects.

c) Implicit assumption audit against the source main()
  - `pieces_raw = comps[1:]` is no longer valid because the input also contains cue components.
    The solver now separates pieces (area >= 8) from cues (area <= 2).
  - Root enumeration can no longer assume every notch is eligible. Only cue-marked notches are.
  - Matching by color is no longer one-to-one: one cue color may correspond to multiple pieces.
    The solver must keep all same-color root candidates and let geometry disambiguate.
  - Painting with a piece's own color during search is only a temporary occupancy encoding.
    Final colors are produced afterward by chain-wise depth swap.
  - The swap assumes two comparable chains. The generator guarantees exactly two chains, same side,
    same length, and adjacent strips.

d) Placement legality
  - The generator is output-first: build the solved geometry, choose hint cells from true root cells,
    then scatter pieces away from the bar while treating both bar and cues as occupied.
  - Used notches do not overlap; chains stay in-bounds; scattered pieces do not overlap or 4-touch.
  - The hint cell is guaranteed to be absorbed by the correct root placement.

e) Visual observability
  - Across examples, the human sees two cue cells of the same color and two scattered pieces of that
    same color, but with different shapes. This makes "color chooses the candidate set, shape chooses
    the exact notch" visible from pixels alone.
  - The output recoloring is also visible because non-root depth pairs swap distinct colors.

f) Visual cleanliness
  - Cues are only 1 cell each and lie just outside the bar mouth.
  - Repeated color is restricted to the two roots, avoiding a scene with too many indistinguishable
    same-colored objects.
  - Parallel strips and a small gap make the chain pairing legible without clutter.

g) Color-change inferability
  - Root color repetition introduces the ambiguity deliberately, but only once.
  - Deeper colors remain distinct, so the swap rule can be inferred from corresponding depths across
    the two neighboring chains rather than from hidden numeric color ids.

Additional audit of new solver assumptions
  - The solver assumes cue components belong to exactly one notch outer ring; generator enforces this.
  - The solver assumes all used notches are marked; generator always emits two cues.
  - The solver assumes the bar is the largest monochrome component and that pieces are area 8-13;
    generator stays inside the original task's size regime.
  - The solver assumes swapping is between adjacent chains ordered by strip coordinate; generator
    always creates exactly two such chains with identical depth.
"""


def _shape_from_coords(coords: np.ndarray, bbox):
    r0, c0, r1, c1 = map(int, bbox)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}


def _connected(cells) -> bool:
    if len(cells) <= 1:
        return True
    cells = set(cells)
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=4):
    d4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    available = set(available)
    seeds = set(seeds)
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 24):
            if len(cells) >= target:
                break
            frontier = [
                nb
                for r, c in cells
                for dr, dc in d4
                if (nb := (r + dr, c + dc)) in available and nb not in cells
            ]
            if not frontier:
                break
            if must_span_axis is not None and span_range is not None:
                a0, a1 = span_range
                if must_span_axis == "cols":
                    cur0 = min((c for _, c in cells), default=a1)
                    cur1 = max((c for _, c in cells), default=a0)
                    if cur0 > a0:
                        leftward = [nb for nb in frontier if nb[1] < cur0]
                        if leftward:
                            cells.add(rng.choice(leftward))
                            continue
                    if cur1 < a1:
                        rightward = [nb for nb in frontier if nb[1] > cur1]
                        if rightward:
                            cells.add(rng.choice(rightward))
                            continue
                else:
                    cur0 = min((r for r, _ in cells), default=a1)
                    cur1 = max((r for r, _ in cells), default=a0)
                    if cur0 > a0:
                        upward = [nb for nb in frontier if nb[0] < cur0]
                        if upward:
                            cells.add(rng.choice(upward))
                            continue
                    if cur1 < a1:
                        downward = [nb for nb in frontier if nb[0] > cur1]
                        if downward:
                            cells.add(rng.choice(downward))
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
                    a0, a1 = span_range
                    if must_span_axis == "cols":
                        if rc[1] == a0 and sum(1 for _, c in cells if c == a0) <= 1:
                            continue
                        if rc[1] == a1 and sum(1 for _, c in cells if c == a1) <= 1:
                            continue
                    else:
                        if rc[0] == a0 and sum(1 for r, _ in cells if r == a0) <= 1:
                            continue
                        if rc[0] == a1 and sum(1 for r, _ in cells if r == a1) <= 1:
                            continue
                if _connected(cells - {rc}):
                    removable.append(rc)
            if not removable:
                break
            cells.remove(rng.choice(removable))
        if len(cells) != target:
            continue
        if must_span_axis is not None and span_range is not None:
            a0, a1 = span_range
            if must_span_axis == "cols":
                if min(c for _, c in cells) != a0 or max(c for _, c in cells) != a1:
                    continue
            else:
                if min(r for r, _ in cells) != a0 or max(r for r, _ in cells) != a1:
                    continue
        if _connected(cells):
            return cells
    return None


def _pixels(shape, top: int, left: int):
    return {(top + dr, left + dc) for dr, dc in shape["rel"]}


def _paint_swapped_output(h: int, w: int, bg: int, bar_cells, bar_color: int, chains):
    out = np.full((h, w), int(bg), dtype=int)
    for r, c in bar_cells:
        out[r, c] = int(bar_color)
    ordered = sorted(chains, key=lambda ch: int(ch["A"]))
    depth_n = max((len(ch["placements"]) for ch in ordered), default=0)
    for depth in range(depth_n):
        if len(ordered) != 2:
            present = [(idx, ch["placements"][depth]) for idx, ch in enumerate(ordered) if depth < len(ch["placements"])]
            colors = [pl["color"] for _, pl in present]
            swapped = list(colors)
            for i in range(0, len(swapped) - 1, 2):
                swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            for (_, pl), color in zip(present, swapped):
                for r, c in pl["pts"]:
                    out[r, c] = int(color)
            continue
        if depth >= len(ordered[0]["placements"]) or depth >= len(ordered[1]["placements"]):
            continue
        left_pl = ordered[0]["placements"][depth]
        right_pl = ordered[1]["placements"][depth]
        for r, c in left_pl["pts"]:
            out[r, c] = int(right_pl["color"])
        for r, c in right_pl["pts"]:
            out[r, c] = int(left_pl["color"])
    return out


def _paint_plain_output(h: int, w: int, bg: int, bar_cells, bar_color: int, chains):
    out = np.full((h, w), int(bg), dtype=int)
    for r, c in bar_cells:
        out[r, c] = int(bar_color)
    for ch in chains:
        for pl in ch["placements"]:
            for r, c in pl["pts"]:
                out[r, c] = int(pl["color"])
    return out


def _choose_hint_cell(root_cells, orientation: str, side: str, bar_r0: int, bar_r1: int, bar_c0: int, bar_c1: int, a0: int, a1: int):
    if orientation == "horizontal":
        target_r = bar_r0 - 1 if side == "top" else bar_r1 + 1
        candidates = sorted((r, c) for r, c in root_cells if r == target_r and a0 <= c <= a1)
        if not candidates:
            return None
        center = (a0 + a1) / 2.0
        return min(candidates, key=lambda rc: (abs(rc[1] - center), rc[1]))
    target_c = bar_c0 - 1 if side == "left" else bar_c1 + 1
    candidates = sorted((r, c) for r, c in root_cells if c == target_c and a0 <= r <= a1)
    if not candidates:
        return None
    center = (a0 + a1) / 2.0
    return min(candidates, key=lambda rc: (abs(rc[0] - center), rc[0]))


def _scene_from_grid(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    if not comps:
        return None
    bar = comps[0]
    misc = comps[1:]
    pieces_raw = [c for c in misc if int(c["area"]) >= 8]
    cue_raw = [c for c in misc if int(c["area"]) <= 2]

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

    notches = []
    empty = ~bar_mask
    for pts in components_mask(empty, connectivity=4):
        r = pts[:, 0]
        c = pts[:, 1]
        top = int((r == 0).sum())
        bottom = int((r == bar_h - 1).sum())
        left = int((c == 0).sum())
        right = int((c == bar_w - 1).sum())
        if top + bottom + left + right == 0:
            continue
        side, mouth_n = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )
        axis = "cols" if side in ("top", "bottom") else "rows"
        hole_gl = {(bar_r0 + int(rr), bar_c0 + int(cc)) for rr, cc in pts}
        if side == "top":
            outer = {(bar_r0 - 1, bar_c0 + int(cc)) for cc in c[r == 0] if bar_r0 - 1 >= 0}
        elif side == "bottom":
            outer = {(bar_r1 + 1, bar_c0 + int(cc)) for cc in c[r == bar_h - 1] if bar_r1 + 1 < h}
        elif side == "left":
            outer = {(bar_r0 + int(rr), bar_c0 - 1) for rr in r[c == 0] if bar_c0 - 1 >= 0}
        else:
            outer = {(bar_r0 + int(rr), bar_c1 + 1) for rr in r[c == bar_w - 1] if bar_c1 + 1 < w}
        notches.append(
            {
                "axis": axis,
                "side": side,
                "hole": hole_gl,
                "mouth_n": mouth_n,
                "outer": outer,
                "hint_color": None,
                "hint_coords": set(),
            }
        )

    for cue in cue_raw:
        cue_coords = {(int(r), int(c)) for r, c in cue["coords"]}
        matches = [n for n in notches if cue_coords <= n["outer"]]
        if len(matches) == 1:
            matches[0]["hint_color"] = int(cue["color"])
            matches[0]["hint_coords"] |= cue_coords

    return {
        "grid": g,
        "H": h,
        "W": w,
        "bg": int(bg),
        "bar_occ": bar_occ,
        "bar_color": bar_color,
        "bar_bbox": (bar_r0, bar_c0, bar_r1, bar_c1),
        "pieces": pieces,
        "notches": notches,
    }


def _solve_snapshot(scene):
    h = scene["H"]
    w = scene["W"]
    bg = scene["bg"]
    bar_occ = scene["bar_occ"]
    bar_color = scene["bar_color"]
    pieces = scene["pieces"]
    notches = scene["notches"]
    d4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    marked_notches = [n for n in notches if n["hint_color"] is not None]
    required_roots = len(marked_notches)
    if required_roots == 0:
        return None, None

    def placement_pts(shape, top, left, occ, must_cover=None, strip=None, must_include=None):
        if top < 0 or left < 0 or top + shape["h"] > h or left + shape["w"] > w:
            return None
        if strip is not None:
            axis, a0, a1 = strip
            if axis == "cols":
                if left != a0 or shape["w"] != a1 - a0 + 1:
                    return None
            else:
                if top != a0 or shape["h"] != a1 - a0 + 1:
                    return None
        pts = _pixels(shape, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        if must_include is not None and not (must_include <= pts):
            return None
        return pts

    def aligned_positions(axis, a0, cover, shape):
        if axis == "cols":
            left = a0
            tops = set()
            for r_n, c_n in cover:
                for lr, lc in shape["rel"]:
                    if c_n == left + lc:
                        tops.add(r_n - lr)
            for top in tops:
                yield top, left
        else:
            top = a0
            lefts = set()
            for r_n, c_n in cover:
                for lr, lc in shape["rel"]:
                    if r_n == top + lr:
                        lefts.add(c_n - lc)
            for left in lefts:
                yield top, left

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in d4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, shape):
        pos = set()
        for r_h, c_h in hole:
            for lr, lc in shape["rel"]:
                pos.add((r_h - lr, c_h - lc))
        return pos

    root_cands = []
    for ni, notch in enumerate(notches):
        if notch["hint_color"] is None:
            continue
        for pi, piece in enumerate(pieces):
            if int(piece["color"]) != int(notch["hint_color"]):
                continue
            shape = piece["shape"]
            for top, left in hole_positions(notch["hole"], shape):
                pts = placement_pts(shape, top, left, bar_occ, must_cover=notch["hole"], must_include=notch["hint_coords"])
                if pts is None:
                    continue
                if notch["axis"] == "cols":
                    a0, a1 = int(left), int(left + shape["w"] - 1)
                else:
                    a0, a1 = int(top), int(top + shape["h"] - 1)
                move = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
                    (
                        (move, -int(notch["mouth_n"])),
                        {
                            "ni": ni,
                            "pi": pi,
                            "top": int(top),
                            "left": int(left),
                            "axis": notch["axis"],
                            "side": notch["side"],
                            "A": a0,
                            "B": a1,
                            "shape": shape,
                        },
                    )
                )
    root_cands.sort(key=lambda x: x[0])
    cand_roots = [item[1] for item in root_cands]
    if len(cand_roots) < required_roots:
        return None, None

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
            a0, a1 = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = _pixels(head["shape"], head["top"], head["left"])
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
                shape = pieces[pi]["shape"]
                for top, left in aligned_positions(axis, a0, cavity, shape):
                    pts = placement_pts(shape, top, left, occ_set, must_cover=cavity, strip=(axis, a0, a1))
                    if pts is None:
                        continue
                    if not touches(pts, head_pts):
                        continue
                    move = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                    moves.append((move, pi, top, left, pts, shape))
            moves.sort(key=lambda x: x[0])
            return moves

        chain_order = sorted(range(len(chains)), key=lambda i: (len(chains[i]["placements"]), chains[i]["A"]))
        for ch_idx in chain_order:
            chain = chains[ch_idx]
            moves = gen_extend_moves(chain, remaining)
            if not moves:
                continue
            for _, pi, top, left, pts, shape in moves[:10]:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = chain["head"]
                chain["head"] = {"top": int(top), "left": int(left), "shape": shape}
                chain["placements"].append({"pi": pi, "color": int(pieces[pi]["color"]), "pts": set(pts)})
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                chain["placements"].pop()
                chain["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = int(bar_color)
                for ch2 in chains:
                    for pl in ch2["placements"]:
                        for rr, cc in pl["pts"]:
                            out_grid[rr, cc] = int(pl["color"])
        return False

    root_sizes = [required_roots]
    if required_roots != 2 and len(cand_roots) >= 2:
        root_sizes.append(2)

    for k in root_sizes:
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            root_sides = set()
            occ0 = set(bar_occ)
            out0 = np.full((h, w), int(bg), dtype=int)
            for r, c in bar_occ:
                out0[r, c] = int(bar_color)
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
            if not ok or len(used_n) != k:
                continue
            if len(chains) == 2 and len(root_sides) != 1:
                continue
            remaining = set(range(len(pieces))) - used_p
            if search(out0, occ0, chains, remaining):
                return solved_snapshot, best_snapshot
    return solved_snapshot, best_snapshot


def main(grid: np.ndarray) -> np.ndarray:
    scene = _scene_from_grid(np.array(grid, dtype=int))
    if scene is None:
        return np.array(grid, dtype=int)
    snapshot, best_snapshot = _solve_snapshot(scene)
    chosen = snapshot if snapshot is not None else best_snapshot
    out = np.full((scene["H"], scene["W"]), int(scene["bg"]), dtype=int)
    for r, c in scene["bar_occ"]:
        out[r, c] = int(scene["bar_color"])
    if chosen is None:
        return out
    return _paint_swapped_output(scene["H"], scene["W"], scene["bg"], scene["bar_occ"], scene["bar_color"], chosen)


def _combo_observable(scene, snapshot) -> bool:
    if scene is None or snapshot is None:
        return False

    marked_notches = [n for n in scene["notches"] if n["hint_color"] is not None]
    if len(marked_notches) != 2:
        return False
    if len({int(n["hint_color"]) for n in marked_notches}) != 1:
        return False

    repeated_color = int(marked_notches[0]["hint_color"])
    repeated_pieces = [p for p in scene["pieces"] if int(p["color"]) == repeated_color]
    if len(repeated_pieces) != 2:
        return False

    if len(snapshot) != 2:
        return False
    if len(snapshot[0]["placements"]) != len(snapshot[1]["placements"]):
        return False

    plain = _paint_plain_output(scene["H"], scene["W"], scene["bg"], scene["bar_occ"], scene["bar_color"], snapshot)
    swapped = _paint_swapped_output(scene["H"], scene["W"], scene["bg"], scene["bar_occ"], scene["bar_color"], snapshot)
    return not np.array_equal(plain, swapped)


def _scatter_piece(rng, input_grid, placed_occ, coords, color, orientation, bar_r0, bar_r1, bar_c0, bar_c1, h, w):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    p_r0, p_c0 = min(rs), min(cs)
    rel = [(r - p_r0, c - p_c0) for r, c in coords]
    p_h = max(rs) - p_r0 + 1
    p_w = max(cs) - p_c0 + 1
    for _ in range(600):
        gap = rng.randint(2, 7)
        if orientation == "horizontal":
            top = bar_r0 - gap - p_h if rng.random() < 0.5 else bar_r1 + gap + 1
            left = rng.randint(0, max(0, w - p_w))
        else:
            left = bar_c0 - gap - p_w if rng.random() < 0.5 else bar_c1 + gap + 1
            top = rng.randint(0, max(0, h - p_h))
        if top < 0 or left < 0 or top + p_h > h or left + p_w > w:
            continue
        pts = {(top + dr, left + dc) for dr, dc in rel}
        if pts & placed_occ:
            continue
        if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            continue
        for r, c in pts:
            input_grid[r, c] = int(color)
        placed_occ |= pts
        return True
    return False


def _gen_attempt():
    rng = random.Random()
    h, w = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    bar_color = rng.randint(1, 9)
    thickness = rng.randint(5, 7)

    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * h)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = min(h - 1, bar_r0 + thickness - 1)
        bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, w - 1
        space_before, space_after = bar_r0, h - 1 - bar_r1
    else:
        center = int(rng.uniform(0.35, 0.53) * w)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = min(w - 1, bar_c0 + thickness - 1)
        bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, h - 1
        space_before, space_after = bar_c0, w - 1 - bar_c1

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    chain_len = rng.choice([2, 3, 3])
    strip_width = rng.randint(4, 5)
    num_chains = 2

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    repeated_root_color = avail_colors[0]
    other_colors = avail_colors[1:]
    needed_other = num_chains * (chain_len - 1)
    if len(other_colors) < needed_other:
        return None

    if orientation == "horizontal":
        sides = ["top", "bottom"]
        rng.shuffle(sides)
        side = next((s for s in sides if (space_before if s == "top" else space_after) >= chain_len * 2 + 3), None)
        if side is None:
            return None
        gap = rng.randint(1, 2)
        total_w = 2 * strip_width + gap
        if total_w > w - 2:
            return None
        sc0_a = rng.randint(1, w - total_w - 1)
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
        if total_h > h - 2:
            return None
        sr0_a = rng.randint(1, h - total_h - 1)
        sr0_b = sr0_a + strip_width + gap
        strip_ranges = [(sr0_a, sr0_a + strip_width - 1), (sr0_b, sr0_b + strip_width - 1)]

    chains_data = []
    all_chain_pieces = []
    chain_infos = []
    notch_holes = []
    occ_set = set(bar_cells)
    other_idx = 0

    for chain_idx in range(num_chains):
        a0, a1 = strip_ranges[chain_idx]
        notch_depth = rng.randint(1, 2)
        notch_area = rng.randint(1, min(4, notch_depth * strip_width))

        if orientation == "horizontal":
            nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
            nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
            notch_avail = {(r, c) for r in range(nr0, nr1 + 1) for c in range(a0, a1 + 1) if (r, c) in bar_cells}
        else:
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            notch_avail = {(r, c) for r in range(a0, a1 + 1) for c in range(nc0, nc1 + 1) if (r, c) in bar_cells}

        all_holes = set().union(*notch_holes) if notch_holes else set()
        notch_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
        if len(notch_avail) < notch_area:
            return None
        notch = _grow_connected(rng, notch_avail, notch_area, {rng.choice(list(notch_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        notch_holes.append(notch)

        chain_pieces = []
        piece_colors = [repeated_root_color] + other_colors[other_idx : other_idx + chain_len - 1]
        other_idx += chain_len - 1

        for depth, piece_color in enumerate(piece_colors):
            piece_area = rng.randint(8, 13)
            chain_dim = rng.randint(2, 3)
            if depth == 0:
                if orientation == "horizontal":
                    if side == "top":
                        pr0, pr1 = max(0, bar_r0 - chain_dim), max(r for r, _ in notch)
                    else:
                        pr0, pr1 = min(r for r, _ in notch), min(h - 1, bar_r1 + chain_dim)
                    pc0, pc1 = a0, a1
                else:
                    if side == "left":
                        pc0, pc1 = max(0, bar_c0 - chain_dim), max(c for _, c in notch)
                    else:
                        pc0, pc1 = min(c for _, c in notch), min(w - 1, bar_c1 + chain_dim)
                    pr0, pr1 = a0, a1
                bbox_area = (pr1 - pr0 + 1) * (pc1 - pc0 + 1)
                target = max(max(8, int(bbox_area * 0.54)), min(piece_area, int(bbox_area * 0.80)))
                if target > bbox_area or target < 8:
                    return None
                avail = {(r, c) for r in range(pr0, pr1 + 1) for c in range(pc0, pc1 + 1) if (r, c) in notch or (r, c) not in occ_set}
                piece_cells = _grow_connected(
                    rng,
                    avail,
                    target,
                    notch,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(a0, a1),
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
                        for c in range(a0, a1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                    ext_range = (max(0, prev_r0 - chain_dim), prev_r0 - 1) if side == "top" else (prev_r1 + 1, min(h - 1, prev_r1 + chain_dim))
                    ext = {(r, c) for r in range(ext_range[0], ext_range[1] + 1) for c in range(a0, a1 + 1) if (r, c) not in occ_set} if ext_range[0] <= ext_range[1] else set()
                else:
                    cavity = {
                        (r, c)
                        for r in range(a0, a1 + 1)
                        for c in range(prev_c0, prev_c1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                    ext_range = (max(0, prev_c0 - chain_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(w - 1, prev_c1 + chain_dim))
                    ext = {(r, c) for r in range(a0, a1 + 1) for c in range(ext_range[0], ext_range[1] + 1) if (r, c) not in occ_set} if ext_range[0] <= ext_range[1] else set()
                if not cavity:
                    return None
                all_avail = cavity | ext
                target = min(13, max(len(cavity), min(piece_area, len(all_avail))))
                if len(cavity) > 13 or target < 8:
                    return None
                is_tail = depth == chain_len - 1
                if is_tail:
                    target = min(13, len(all_avail))
                piece_cells = _grow_connected(
                    rng,
                    all_avail,
                    target,
                    cavity,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(a0, a1),
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
                            for c in range(a0, a1 + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    else:
                        remaining = {
                            (r, c)
                            for r in range(a0, a1 + 1)
                            for c in range(t_c0, t_c1 + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    if remaining:
                        return None
            if not (8 <= len(piece_cells) <= 13):
                return None
            chain_pieces.append({"color": int(piece_color), "coords": set(piece_cells)})
            occ_set |= piece_cells

        chains_data.append({"A": a0, "B": a1, "placements": [{"color": p["color"], "pts": set(p["coords"])} for p in chain_pieces]})
        all_chain_pieces.extend(chain_pieces)

        hint_cell = _choose_hint_cell(chain_pieces[0]["coords"], orientation, side, bar_r0, bar_r1, bar_c0, bar_c1, a0, a1)
        if hint_cell is None:
            return None
        chain_infos.append({"hint": hint_cell, "root_color": repeated_root_color})

    root_shape_signatures = []
    for chain in chains_data:
        pts = chain["placements"][0]["pts"]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, c0 = min(rs), min(cs)
        root_shape_signatures.append(frozenset((r - r0, c - c0) for r, c in pts))
    if root_shape_signatures[0] == root_shape_signatures[1]:
        return None

    num_decoys = rng.randint(4, 7)
    decoys = 0
    for _ in range(600):
        if decoys >= num_decoys:
            break
        depth = rng.randint(1, 2)
        area = rng.randint(1, min(6, depth * 5))
        if orientation == "horizontal":
            decoy_side = rng.choice(["top", "bottom"])
            dr0 = bar_r0 if decoy_side == "top" else bar_r1 - depth + 1
            dr1 = bar_r0 + depth - 1 if decoy_side == "top" else bar_r1
            dc0 = rng.randint(bar_c0, bar_c1)
            dc1 = min(bar_c1, dc0 + rng.randint(0, 4))
        else:
            decoy_side = rng.choice(["left", "right"])
            dc0 = bar_c0 if decoy_side == "left" else bar_c1 - depth + 1
            dc1 = bar_c0 + depth - 1 if decoy_side == "left" else bar_c1
            dr0 = rng.randint(bar_r0, bar_r1)
            dr1 = min(bar_r1, dr0 + rng.randint(0, 4))
        decoy_avail = {(r, c) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        all_holes = set().union(*notch_holes) if notch_holes else set()
        decoy_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
        if not decoy_avail:
            continue
        area = min(area, len(decoy_avail))
        dn = _grow_connected(rng, decoy_avail, area, {rng.choice(list(decoy_avail))})
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

    plain_output = _paint_plain_output(h, w, bg, bar_cells, bar_color, chains_data)
    output = _paint_swapped_output(h, w, bg, bar_cells, bar_color, chains_data)
    if np.array_equal(plain_output, output):
        return None

    input_grid = np.full((h, w), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = int(bar_color)
    placed_occ = set(bar_cells)
    for info in chain_infos:
        hr, hc = info["hint"]
        if (hr, hc) in placed_occ:
            return None
        input_grid[hr, hc] = int(info["root_color"])
        placed_occ.add((hr, hc))

    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)
    for piece in pieces_order:
        if not _scatter_piece(rng, input_grid, placed_occ, piece["coords"], piece["color"], orientation, bar_r0, bar_r1, bar_c0, bar_c1, h, w):
            return None

    repeated_count = sum(1 for p in all_chain_pieces if p["color"] == repeated_root_color)
    if repeated_count != 2:
        return None
    if len({info["root_color"] for info in chain_infos}) != 1:
        return None

    return input_grid, output


def generate_input() -> np.ndarray:
    for _ in range(1600):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        scene = _scene_from_grid(inp)
        snapshot, _ = _solve_snapshot(scene) if scene is not None else (None, None)
        if not _combo_observable(scene, snapshot):
            continue
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    raise RuntimeError("generate_input: failed after 1600 attempts")
