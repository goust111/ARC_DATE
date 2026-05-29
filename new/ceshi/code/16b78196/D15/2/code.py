# Variant of 16b78196
# Dimension: D15 — 符号指令：是否包含图例或提示标记 (symbol_instructions)
# Change: 无图例或符号指令——所有信息从几何关系推断... → 精灵上的数字状标记指示拼装顺序

from __future__ import annotations

from itertools import combinations

import numpy as np

from arc_utils import components_mask, find_components, get_bg


# Design analysis for the D15 variant
# a) Concretization and behavior trace
#    - The original task has a large monochrome bar plus several scattered monochrome
#      sprites. The solver inferred the whole assembly order only from hole-coverage
#      geometry.
#    - This variant keeps the same bar/notch/chain mechanics, but each scattered sprite
#      now contains 1, 2, or 3 single-cell dots of one shared marker color. Dot count
#      explicitly tells the sprite's role in its chain: 1=root, 2=middle, 3=tail.
#    - The dots remain part of the sprite in both input and output, so the visible change
#      is real rather than metadata hidden from the grid.
#    - Example trace:
#      Original: a piece fits a bar notch, another piece fits the root cavity, and a
#      third piece fits the next cavity; solver must search these possibilities.
#      Variant: the same three pieces also carry 1/2/3 dots, so the root search may use
#      only 1-dot pieces, extension from a root may use only 2-dot pieces, and the final
#      extension may use only 3-dot pieces.
# b) Complexity calibration
#    - Chain geometry is left intact so the puzzle is still the same family.
#    - To make the numeric symbols consistently meaningful and observable, every chain is
#      fixed to length 3 in the generator. This guarantees that 1/2/3 all appear and are
#      reused across examples, instead of making "2 = middle" disappear in many cases.
#    - The marker color is shared across all sprites and used only for isolated 1-cell
#      dots. That makes the instruction channel easy to spot without creating color-map
#      ambiguity.
# c) Old assumptions that no longer hold
#    - Original `pieces_raw = comps[1:]` assumed every sprite was a monochrome component.
#      That fails here because marker dots split each sprite into multiple monochrome
#      components. The new solver still finds the bar monochromatically, but finds sprites
#      as multicolor foreground components.
#    - Original painting logic wrote one solid color per piece. That would erase the
#      marker dots, so the new solver stores per-cell colors and paints whole multicolor
#      sprites.
#    - Original root/extension search allowed any remaining piece at each step. That is
#      now too permissive and can bypass the intended symbol decoding, so the new search
#      enforces 1 -> 2 -> 3 order.
# d) Placement legality
#    - The generator still constructs the final assembled layout first, then scatters
#      intact sprites away from the bar without overlap or edge-contact with other
#      objects.
#    - Marker dots are chosen from cells already belonging to a sprite, so they never
#      change occupancy, connectivity, strip width, or cavity coverage.
# e) Visual observability
#    - A human now sees a repeated special color inside every sprite and can count 1/2/3
#      dots directly. Because the chains remain otherwise similar to the original task,
#      the new information is visibly the main new cue.
#    - The rule does not rely on numeric color ids; only visible dot counts matter.
# f) Visual cleanliness
#    - Marker dots are isolated and limited to at most three per sprite.
#    - They are placed on non-fringe cells when possible and kept non-adjacent, so they
#      read as separate dots rather than reshaping the silhouette.
# g) Color inference
#    - Only one extra semantic color is introduced: the shared marker color.
#    - The solver infers it structurally: it is the only non-bar color whose monochrome
#      components are all isolated 1-cell dots. Piece body colors remain otherwise
#      ordinary and mostly distinct.
# Extra note about the source task
#    - The prompt cites 8698868d as the inspiration for count-based symbolic cues. In the
#      provided workspace only the JSON for that task is present; no solver source is
#      available to inspect, so the count-decoding mechanism is implemented directly here.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble marked sprites into bar notches. Dot count in the shared marker color gives
    chain order: 1-dot root, 2-dot middle, 3-dot tail.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    bg = get_bg(g)

    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    mono.sort(key=lambda d: -d["area"])
    assert mono

    bar = mono[0]
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1

    out = np.full((H, W), int(bg), dtype=int)
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    def infer_marker_color() -> int:
        stats: dict[int, list[int]] = {}
        for comp in mono[1:]:
            color = int(comp["color"])
            if color == bar_color:
                continue
            stats.setdefault(color, []).append(int(comp["area"]))

        candidates = []
        for color, areas in stats.items():
            if areas and max(areas) == 1:
                candidates.append((len(areas), -sum(areas), -color, color))
        if candidates:
            candidates.sort(reverse=True)
            return int(candidates[0][3])

        # Fallback: choose the color with the smallest largest monochrome component.
        fallback = []
        for color, areas in stats.items():
            fallback.append((-max(areas), len(areas), -sum(areas), color))
        fallback.sort(reverse=True)
        return int(fallback[0][3])

    marker_color = infer_marker_color()

    def shape_from_component(coords: np.ndarray, bbox):
        r0, c0, r1, c1 = map(int, bbox)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        rel = []
        rel_set = set()
        rel_colors = []
        marker_count = 0
        body_colors: dict[int, int] = {}
        for r, c in coords:
            rr, cc = int(r), int(c)
            lr, lc = rr - r0, cc - c0
            rel.append((lr, lc))
            rel_set.add((lr, lc))
            color = int(g[rr, cc])
            rel_colors.append((lr, lc, color))
            if color == marker_color:
                marker_count += 1
            else:
                body_colors[color] = body_colors.get(color, 0) + 1
        body_color = max(body_colors.items(), key=lambda kv: kv[1])[0]
        return {
            "rel": rel,
            "rel_set": rel_set,
            "rel_colors": rel_colors,
            "h": h,
            "w": w,
            "order": int(marker_count),
            "body_color": int(body_color),
        }

    multicolor = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    multicolor.sort(key=lambda d: -d["area"])
    assert multicolor
    pieces_raw = multicolor[1:]

    pieces = []
    for comp in pieces_raw:
        r0, c0 = map(int, comp["bbox"][:2])
        shape = shape_from_component(comp["coords"], comp["bbox"])
        pieces.append(
            {
                "shape": shape,
                "orig": (r0, c0),
                "order": int(shape["order"]),
                "body_color": int(shape["body_color"]),
            }
        )

    by_order = {1: [], 2: [], 3: []}
    for i, p in enumerate(pieces):
        if p["order"] in by_order:
            by_order[p["order"]].append(i)
    num_chains = len(by_order[1])
    if num_chains == 0:
        return out

    def bar_notches(mask, base_r, base_c):
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
            hole_gl = {(base_r + int(rr), base_c + int(cc)) for rr, cc in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    def pixels(shape, top, left):
        return {(top + lr, left + lc) for lr, lc in shape["rel"]}

    def paint_shape(out_grid, shape, top, left):
        for lr, lc, color in shape["rel_colors"]:
            out_grid[top + lr, left + lc] = int(color)

    def clear_shape(out_grid, shape, top, left):
        for lr, lc, _ in shape["rel_colors"]:
            out_grid[top + lr, left + lc] = int(bg)

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
        pts = pixels(shape, top, left)
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
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, shape):
        pos = set()
        for rH, cH in hole:
            for lr, lc in shape["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    def cavity_for_head(out_grid, head):
        top, left = int(head["top"]), int(head["left"])
        h, w = int(head["shape"]["h"]), int(head["shape"]["w"])
        return {
            (r, c)
            for r in range(top, top + h)
            for c in range(left, left + w)
            if out_grid[r, c] == bg
        }

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = notch["axis"]
        for pi in by_order[1]:
            shape = pieces[pi]["shape"]
            for top, left in hole_positions(notch["hole"], shape):
                pts = placement_pts(shape, top, left, bar_occ, must_cover=notch["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + shape["w"] - 1)
                else:
                    A, B = int(top), int(top + shape["h"] - 1)
                mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), -int(notch["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": shape},
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:40]]

    best_out = out.copy()
    best_key = None

    def search(out_grid, occ_set, chains, remaining):
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        completed = sum(1 for ch in chains if ch["next_order"] is None)
        key = (placed_cnt, completed, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return all(ch["next_order"] is None for ch in chains)

        choice = None
        for ci, ch in enumerate(chains):
            desired = ch["next_order"]
            if desired is None:
                continue
            axis = ch["axis"]
            A, B = int(ch["A"]), int(ch["B"])
            head = ch["head"]
            head_pts = pixels(head["shape"], head["top"], head["left"])
            cavity = cavity_for_head(out_grid, head)
            if not cavity:
                return False

            moves = []
            for pi in remaining:
                if pieces[pi]["order"] != desired:
                    continue
                shape = pieces[pi]["shape"]
                for top, left in aligned_positions(axis, A, cavity, shape):
                    pts = placement_pts(shape, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                    if pts is None:
                        continue
                    if not touches(pts, head_pts):
                        continue
                    moves.append((pi, top, left, pts, shape))
            if not moves:
                return False
            moves.sort(key=lambda m: abs(m[1] - pieces[m[0]]["orig"][0]) + abs(m[2] - pieces[m[0]]["orig"][1]))
            if choice is None or len(moves) < len(choice[2]):
                choice = (ci, ch, moves)

        if choice is None:
            return False

        ci, ch, moves = choice
        for pi, top, left, pts, shape in moves[:10]:
            old_head = ch["head"]
            old_next = ch["next_order"]
            paint_shape(out_grid, shape, top, left)
            ch["head"] = {"top": int(top), "left": int(left), "shape": shape}
            new_cavity = cavity_for_head(out_grid, ch["head"])
            if old_next == 2 and not new_cavity:
                clear_shape(out_grid, shape, top, left)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
                ch["head"] = old_head
                continue
            if old_next == 3 and new_cavity:
                clear_shape(out_grid, shape, top, left)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
                ch["head"] = old_head
                continue

            ch["next_order"] = 3 if old_next == 2 else None
            occ2 = set(occ_set) | set(pts)
            rem2 = set(remaining)
            rem2.remove(pi)
            if search(out_grid, occ2, chains, rem2):
                return True

            ch["next_order"] = old_next
            ch["head"] = old_head
            clear_shape(out_grid, shape, top, left)
            for r, c in bar_occ:
                out_grid[r, c] = bar_color

        return False

    if len(by_order[2]) != num_chains or len(by_order[3]) != num_chains:
        return best_out

    for roots_sel in combinations(range(len(cand_roots)), num_chains):
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
            paint_shape(out0, root["shape"], root["top"], root["left"])
            occ0 |= pts
            head = {"top": root["top"], "left": root["left"], "shape": root["shape"]}
            if not cavity_for_head(out0, head):
                ok = False
                break
            chains.append({"axis": root["axis"], "A": root["A"], "B": root["B"], "head": head, "next_order": 2})

        if not ok:
            continue

        remaining = set(range(len(pieces))) - used_p
        if search(out0, occ0, chains, remaining):
            return out0

    return best_out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

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
            frontier = [nb for r, c in cells for dr, dc in _D4 if (nb := (r + dr, c + dc)) in available and nb not in cells]
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


def _choose_marker_cells(cells, count):
    """Pick isolated marker positions inside a sprite without changing its occupancy."""
    cell_set = set(cells)
    cells = sorted(cell_set)
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0

    def degree(rc):
        r, c = rc
        return sum((r + dr, c + dc) in cell_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])

    def edge_margin(rc):
        r, c = rc
        return min(r - r0, r1 - r, c - c0, c1 - c)

    candidates = [rc for rc in cells if degree(rc) >= 2]
    if len(candidates) < count:
        candidates = [rc for rc in cells if degree(rc) >= 1]
    if len(candidates) < count:
        return None

    best = None
    best_score = None
    for combo in combinations(candidates, count):
        if any(abs(r1_ - r2_) + abs(c1_ - c2_) == 1 for (r1_, c1_), (r2_, c2_) in combinations(combo, 2)):
            continue
        score = (
            sum(degree(rc) for rc in combo),
            sum(edge_margin(rc) for rc in combo),
            -sum((rc[0] - cr) ** 2 + (rc[1] - cc) ** 2 for rc in combo),
        )
        if best_score is None or score > best_score:
            best_score = score
            best = combo
    return set(best) if best is not None else None


def _gen_attempt():
    """Build one (input, output) pair; returns (input, output) or None."""
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

    num_chains = 2 if rng.random() < 0.65 else 1
    chain_lens = [3] * num_chains

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    marker_color = avail_colors.pop()
    color_idx = 0

    all_chain_pieces = []
    notch_holes: list[set] = []
    occ_set = set(bar_cells)

    for chain_idx in range(num_chains):
        chain_len = chain_lens[chain_idx]
        strip_width = rng.randint(4, 5)

        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "top" else space_after) >= chain_len * 2 + 3), None)
            if side is None:
                return None
            sc0 = rng.randint(1, W - strip_width - 1)
            sc1 = sc0 + strip_width - 1
            sA, sB = sc0, sc1
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "left" else space_after) >= chain_len * 2 + 3), None)
            if side is None:
                return None
            sr0 = rng.randint(1, H - strip_width - 1)
            sr1 = sr0 + strip_width - 1
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
                    ext = {(r, c) for r in range(es[0], es[1] + 1) for c in range(sc0, sc1 + 1) if (r, c) not in occ_set} if es[0] <= es[1] else set()
                else:
                    es = (max(0, prev_c0 - chain_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    ext = {(r, c) for r in range(sA, sB + 1) for c in range(es[0], es[1] + 1) if (r, c) not in occ_set} if es[0] <= es[1] else set()

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

                prs2 = [r for r, _ in piece_cells]
                pcs2 = [c for _, c in piece_cells]
                t_r0, t_r1 = min(prs2), max(prs2)
                t_c0, t_c1 = min(pcs2), max(pcs2)
                remaining = {
                    (r, c)
                    for r in range(t_r0, t_r1 + 1)
                    for c in range(sc0, sc1 + 1)
                    if (r, c) not in occ_set and (r, c) not in piece_cells
                } if orientation == "horizontal" else {
                    (r, c)
                    for r in range(sA, sB + 1)
                    for c in range(t_c0, t_c1 + 1)
                    if (r, c) not in occ_set and (r, c) not in piece_cells
                }

                if is_tail and remaining:
                    return None
                if not is_tail and not remaining:
                    return None

            if not (8 <= len(piece_cells) <= 13):
                return None

            marker_cells = _choose_marker_cells(piece_cells, piece_idx + 1)
            if marker_cells is None:
                return None

            chain_pieces.append(
                {
                    "body_color": piece_color,
                    "marker_color": marker_color,
                    "order": piece_idx + 1,
                    "coords": piece_cells,
                    "marker_cells": marker_cells,
                }
            )
            occ_set |= piece_cells

        all_chain_pieces.extend(chain_pieces)

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

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = bar_color
    for p in all_chain_pieces:
        for r, c in p["coords"]:
            output[r, c] = p["body_color"]
        for r, c in p["marker_cells"]:
            output[r, c] = p["marker_color"]

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color

    placed_occ = set(bar_cells)
    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)

    for p in pieces_order:
        coords = p["coords"]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        p_r0, p_c0 = min(rs), min(cs)
        rel = [(r - p_r0, c - p_c0) for r, c in coords]
        marker_rel = {(r - p_r0, c - p_c0) for r, c in p["marker_cells"]}
        p_h = max(rs) - p_r0 + 1
        p_w = max(cs) - p_c0 + 1

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
            for dr, dc in rel:
                input_grid[top + dr, left + dc] = p["marker_color"] if (dr, dc) in marker_rel else p["body_color"]
            placed_occ |= pts
            placed = True
            break
        if not placed:
            return None

    return input_grid, output


def generate_input():
    """Generate a valid input grid for the D15 variant of 16b78196."""
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
