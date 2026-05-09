# Variant of 16b78196
# Dimension: D7 — 输出网格形状 (grid_shape)
# Change: 输出与输入同尺寸（30x30）... → 输出按链分离排列

from __future__ import annotations

import numpy as np

from arc_utils import components_mask, find_components, get_bg

# Design analysis for the D7 variant
# ----------------------------------
# Reference baseline:
# - I read the original 16b78196 solver/generator and the packed-grid source task
#   solver at `/home/lixinyu/snap/new_code/solvers/arc2/58f5dbd5.py`.
# - The relevant mechanism from 58f5dbd5 is not the carving logic but the output
#   layout logic: extract already-solved local objects, preserve their order on the
#   source axis, and repack them into a compact multi-panel grid separated by 1-cell
#   dividers.
#
# a) Concretisation / behavior trace
# - Original output: a single 30x30 scene containing the full bar and every chain in
#   its in-situ location.
# - Variant output: after reconstructing the same assembled scene internally, split it
#   into one panel per chain. Each panel contains:
#   1. all cells of that chain, and
#   2. only the corresponding strip of the bar (same A..B strip that governs the chain).
# - Then pack the panels while preserving strip order along the bar:
#   horizontal bar -> left-to-right panels with a 1-cell vertical divider;
#   vertical bar -> top-to-bottom panels with a 1-cell horizontal divider.
# - Small trace:
#   original: fill notch -> extend chain -> paint into 30x30 output.
#   variant: fill notch -> extend chain -> still paint full 30x30 scene internally ->
#   crop chain+bar-strip panel -> packed multi-panel output.
#   The chain-matching logic is unchanged; only the final presentation is changed.
#
# b) Complexity calibration
# - If the generator still allowed a single chain too often, D7 could visually collapse
#   into a mere crop. To keep the grid-shape change obvious, the generator is adapted
#   to always create 2 chains and to keep their strip intervals separated.
# - I keep the original chain geometry, notch ambiguity, and no-rotation rule so the
#   puzzle is not simplified into trivial object grouping.
#
# c) Implicit assumptions audit for main()
# - `bar = comps[0]`: still valid because the bar remains the largest monochrome
#   component in the input; packed output does not affect input parsing.
# - `axis = "cols"/"rows"` from notch mouth: still valid because chains are still
#   attached to top/bottom or left/right bar sides exactly as before.
# - Search code assumes all chain pieces live in one shared 30x30 scene while cavities
#   are computed from the current head bbox. That remains true; panel separation happens
#   only after search finishes.
# - Original `return best_out` assumed the final answer had input size. This is the key
#   line that no longer holds. It is replaced by packing `best_out` / solved chains into
#   a compact multi-panel grid.
# - New assumption introduced here: every intended chain corresponds to one unique strip
#   interval `[A, B]`, and different chains should not share the same strip. The updated
#   generator guarantees this by choosing non-overlapping strip intervals.
#
# d) Placement legality
# - Generator still builds the full assembled 30x30 scene first, so all original checks
#   for overlap, connectivity, cavity filling, and in-bounds placement remain applicable.
# - For the new packed output, panel placement is deterministic and non-overlapping:
#   panels are placed on a fresh background canvas with exactly 1 divider cell between
#   neighbors.
#
# e) Visual observability
# - Human-visible change: outputs are no longer global scenes. They become 2 small
#   panels, each showing one chain and its matching bar strip. The divider is explicit.
# - No invisible attribute is used for ordering; panel order is the visible strip order
#   along the bar.
#
# f) Visual cleanliness
# - Each panel is a tight crop of one chain plus its bar segment, so unrelated decoy
#   notches from far-away bar regions disappear instead of cluttering the output.
# - Dividers use the bar color, which is already semantically meaningful and easy to see.
#
# g) Color-change inference
# - No new color mapping is introduced. Colors keep their original role; only spatial
#   layout changes.
#
# h) Post-edit assumption audit
# - Packing aligns the bar-band rows/columns across panels. This assumes each panel does
#   contain a non-empty bar segment. That is guaranteed because each chain strip always
#   intersects the bar.
# - The generator now always emits 2 chains and separated strips, so the packed order and
#   divider are always meaningful and stable across examples.


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
                "order": int(ch["A"]),
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
    cand_roots = [pair[1] for pair in root_cands[:20]]

    best_scene = np.array(base_scene, copy=True)
    best_chains: list[dict] = []
    best_key = None

    def search(out_grid: np.ndarray, occ_set: set[tuple[int, int]], chains: list[dict], remaining: set[int]) -> bool:
        nonlocal best_scene, best_chains, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (int(placed_cnt), -int(len(remaining)), -int(len(occ_set)))
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

        for ch in chains:
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

    from itertools import combinations

    for k in range(1, min(3, len(cand_roots)) + 1):
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
                return _pack_chain_panels(
                    out0,
                    bar_occ,
                    bar_color,
                    bg,
                    _snapshot_chains(chains),
                    bar_orientation,
                )

    return _pack_chain_panels(best_scene, bar_occ, bar_color, bg, best_chains, bar_orientation)


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
            frontier = [nb for r, c in cells for dr, dc in DIR4 if (nb := (r + dr, c + dc)) in available and nb not in cells]
            if not frontier:
                break
            if must_span_axis and span_range:
                A, B = span_range
                if must_span_axis == "cols":
                    cm = min((c for _, c in cells), default=B)
                    cx = max((c for _, c in cells), default=A)
                    if cm > A:
                        picks = [nb for nb in frontier if nb[1] < cm]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                    if cx < B:
                        picks = [nb for nb in frontier if nb[1] > cx]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                else:
                    cm = min((r for r, _ in cells), default=B)
                    cx = max((r for r, _ in cells), default=A)
                    if cm > A:
                        picks = [nb for nb in frontier if nb[0] < cm]
                        if picks:
                            cells.add(rng.choice(picks))
                            continue
                    if cx < B:
                        picks = [nb for nb in frontier if nb[0] > cx]
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


def _choose_strip_interval(rng, length: int, limit: int, used: list[tuple[int, int]]):
    for _ in range(200):
        start = rng.randint(1, limit - length - 1)
        end = start + length - 1
        if all(end < a - 1 or start > b + 1 for a, b in used):
            return start, end
    return None


def _gen_attempt():
    import random as _random

    rng = _random.Random()
    H, W = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    bar_color = rng.randint(1, 9)
    thickness = rng.randint(5, 7)

    if orientation == "horizontal":
        center = rng.randint(12, 17)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = bar_r0 + thickness - 1
        if bar_r1 >= H:
            bar_r1 = H - 1
            bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        space_before, space_after = bar_r0, H - 1 - bar_r1
    else:
        center = rng.randint(12, 17)
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
    chain_lens = [rng.randint(2, 3), rng.randint(2, 3)]
    while not (4 <= sum(chain_lens) <= 6):
        chain_lens = [rng.randint(2, 3), rng.randint(2, 3)]

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    color_idx = 0

    all_chain_pieces = []
    notch_holes: list[set] = []
    occ_set = set(bar_cells)
    chain_records = []
    used_strips: list[tuple[int, int]] = []
    chosen_sides: list[str] = []

    for chain_idx in range(num_chains):
        chain_len = chain_lens[chain_idx]
        strip_width = rng.randint(4, 5)

        if orientation == "horizontal":
            base_sides = ["top", "bottom"]
            feasible_sides = [s for s in base_sides if (space_before if s == "top" else space_after) >= chain_len * 2 + 3]
            if not feasible_sides:
                return None
            preferred = "bottom" if chosen_sides and chosen_sides[0] == "top" else "top"
            if preferred in feasible_sides and rng.random() < 0.8:
                side = preferred
            else:
                side = rng.choice(feasible_sides)
            strip = _choose_strip_interval(rng, strip_width, W, used_strips)
            if strip is None:
                return None
            sc0, sc1 = strip
            sA, sB = sc0, sc1
        else:
            base_sides = ["left", "right"]
            feasible_sides = [s for s in base_sides if (space_before if s == "left" else space_after) >= chain_len * 2 + 3]
            if not feasible_sides:
                return None
            preferred = "right" if chosen_sides and chosen_sides[0] == "left" else "left"
            if preferred in feasible_sides and rng.random() < 0.8:
                side = preferred
            else:
                side = rng.choice(feasible_sides)
            strip = _choose_strip_interval(rng, strip_width, H, used_strips)
            if strip is None:
                return None
            sr0, sr1 = strip
            sA, sB = sr0, sr1

        chosen_sides.append(side)
        used_strips.append((sA, sB))

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
        n_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
        if len(n_avail) < notch_area:
            return None

        notch = _grow_connected(rng, n_avail, notch_area, {rng.choice(list(n_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        notch_holes.append(notch)

        chain_axis = "cols" if orientation == "horizontal" else "rows"
        chain_record = {"axis": chain_axis, "A": int(sA), "B": int(sB), "placements": []}
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
                    pc0, pc1 = sc0, sc1
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

                avail = {(r, c) for r in range(pr0, pr1 + 1) for c in range(pc0, pc1 + 1) if (r, c) in notch or (r, c) not in occ_set}
                piece_cells = _grow_connected(
                    rng,
                    avail,
                    target,
                    notch,
                    must_span_axis=chain_axis,
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
                    ext = (
                        {(r, c) for r in range(es[0], es[1] + 1) for c in range(sc0, sc1 + 1) if (r, c) not in occ_set}
                        if es[0] <= es[1]
                        else set()
                    )
                else:
                    es = (max(0, prev_c0 - chain_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    ext = (
                        {(r, c) for r in range(sA, sB + 1) for c in range(es[0], es[1] + 1) if (r, c) not in occ_set}
                        if es[0] <= es[1]
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
                    must_span_axis=chain_axis,
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

            if not (8 <= len(piece_cells) <= 13):
                return None

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

    if not (4 <= len(all_chain_pieces) <= 6):
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

    full_output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        full_output[r, c] = int(bar_color)
    for p in all_chain_pieces:
        for r, c in p["coords"]:
            full_output[r, c] = int(p["color"])

    output = _pack_chain_panels(full_output, bar_cells, bar_color, bg, chain_records, orientation)

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = int(bar_color)

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
            if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in DIR4):
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
    for _ in range(600):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 600 attempts")
