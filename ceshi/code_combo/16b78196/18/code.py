# Combo variant of 16b78196 (index 18)
# combo_spec: D3/0+D16/2
# Name: 十字框架+条纹杆

from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from arc_utils import components_mask, find_components, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    Original task:
#    - A single straight monochrome bar is the dominant object.
#    - Every useful notch lies on one of the two long exposed sides of that bar.
#    - A root sprite plugs a notch, then later sprites grow along the same strip.
#    Combo behavior:
#    - The bar becomes a central plus/cross made from one horizontal band and one
#      vertical band of the same thickness.
#    - The whole cross uses exactly two alternating stripe colors, but the
#      alternation is global by row or by column. Because the cross has two arm
#      orientations, the same stripe axis looks parallel on one pair of arms and
#      perpendicular on the other pair.
#    - There is one notch at each arm tip (top / bottom / left / right). Three of
#      them are active and host chains; the fourth is a decoy.
#    Example sketch:
#      Suppose the cross is striped by columns. Then the top notch mouth visibly
#      alternates A/B/A/B across its width, while the left notch mouth is mostly a
#      single stripe segment repeated down the arm. The solver cannot rely on one
#      biggest monochrome blob any more; it must first recover the two-color cross,
#      then inspect the four arm tips separately, then grow chains in the correct
#      outward direction for each side.
# b) Complexity calibration
#    - The bar still uses only two colors, not an arbitrary palette, so the visual
#      rule is repeated and learnable from a few examples.
#    - Active-chain count is fixed at three. This forces multi-direction reasoning
#      but avoids a large search tree.
#    - Every chain has length two. That preserves the "root creates cavity; tail
#      fills it" logic while keeping generation reliable.
# c) Hidden-assumption audit against the original solver
#    - "largest monochrome component = bar": broken, because each stripe color is
#      only part of the bar.
#    - "all bar holes can be read from one rectangular bbox": broken, because a
#      plus shape has large empty corner quadrants inside its outer bbox that are
#      not notches.
#    - "one scalar bar_color is enough for restore": broken, because undo must
#      restore per-cell stripe colors.
#    - "notch orientation is inferred only from which side of the bbox it touches":
#      broken, because the global bbox has four huge background corners. We instead
#      inspect four dedicated arm-tip windows.
# d) Placement legality
#    - Generator builds the output first from a cross occupancy set, carves one tip
#      notch per arm, then grows three non-overlapping chains outward.
#    - Only after the output is valid are the pieces scattered into empty space.
#      Scatter placement forbids overlap and 4-neighbor touching with the bar or
#      other pieces.
# e) Visual observability
#    - Humans can see both changes: the bar is no longer straight, and it is no
#      longer monochrome.
#    - The stripe axis is repeated across the full cross, so "stripe alignment"
#      around the arm-tip notches is visible instead of being a one-off cue.
# f) Cleanliness
#    - Each arm has exactly one notch, so the cross silhouette remains readable.
#    - Only three chains are active, preventing the center from becoming cluttered.
# g) New solver assumptions introduced here
#    - The two most frequent non-background colors belong to the cross.
#    - The cross is the union of one horizontal and one vertical rectangle with a
#      shared thickness.
#    - There is exactly one notch in each arm-tip window and exactly three active
#      chains, each of length two.
#    Generator construction enforces each of these assumptions.


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = list(range(1, 10))


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


def _stripe_color(axis: str, color0: int, color1: int, r: int, c: int, r0: int, c0: int) -> int:
    idx = (r - r0) if axis == "rows" else (c - c0)
    return int(color0 if idx % 2 == 0 else color1)


def _grow_connected(
    rng: random.Random,
    available: set[tuple[int, int]],
    target: int,
    seeds: set[tuple[int, int]],
    must_span_axis: str | None = None,
    span_range: tuple[int, int] | None = None,
    retries: int = 8,
) -> set[tuple[int, int]] | None:
    for _ in range(retries):
        cells = set(seeds)
        if not cells or not cells.issubset(available):
            return None
        for __ in range(max(30, target * 25)):
            if len(cells) >= target:
                break
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nb = (r + dr, c + dc)
                    if nb in available and nb not in cells:
                        frontier.append(nb)
            if not frontier:
                break
            if must_span_axis is not None and span_range is not None:
                lo, hi = span_range
                if must_span_axis == "cols":
                    cur_lo = min(c for _, c in cells)
                    cur_hi = max(c for _, c in cells)
                    if cur_lo > lo:
                        cand = [nb for nb in frontier if nb[1] < cur_lo]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if cur_hi < hi:
                        cand = [nb for nb in frontier if nb[1] > cur_hi]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                else:
                    cur_lo = min(r for r, _ in cells)
                    cur_hi = max(r for r, _ in cells)
                    if cur_lo > lo:
                        cand = [nb for nb in frontier if nb[0] < cur_lo]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if cur_hi < hi:
                        cand = [nb for nb in frontier if nb[0] > cur_hi]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue
        if must_span_axis is not None and span_range is not None:
            lo, hi = span_range
            if must_span_axis == "cols":
                if min(c for _, c in cells) != lo or max(c for _, c in cells) != hi:
                    continue
            else:
                if min(r for r, _ in cells) != lo or max(r for r, _ in cells) != hi:
                    continue
        if _connected(cells):
            return cells
    return None


def _shape_from_coords(coords: np.ndarray, bbox) -> dict:
    r0, c0, r1, c1 = map(int, bbox)
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "h": int(r1 - r0 + 1), "w": int(c1 - c0 + 1)}


def _pixels(shape: dict, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + dr, left + dc) for dr, dc in shape["rel"]}


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in b:
                return True
    return False


def _hole_positions(hole: set[tuple[int, int]], shape: dict) -> set[tuple[int, int]]:
    pos = set()
    for r_h, c_h in hole:
        for dr, dc in shape["rel"]:
            pos.add((r_h - dr, c_h - dc))
    return pos


def _placement_pts(
    shape: dict,
    top: int,
    left: int,
    occ: set[tuple[int, int]],
    H: int,
    W: int,
    must_cover: set[tuple[int, int]] | None = None,
    strip: tuple[str, int, int] | None = None,
) -> set[tuple[int, int]] | None:
    if top < 0 or left < 0 or top + shape["h"] > H or left + shape["w"] > W:
        return None
    if strip is not None:
        axis, lo, hi = strip
        if axis == "cols":
            if left != lo or shape["w"] != hi - lo + 1:
                return None
        else:
            if top != lo or shape["h"] != hi - lo + 1:
                return None
    pts = _pixels(shape, top, left)
    if pts & occ:
        return None
    if must_cover is not None and not (must_cover <= pts):
        return None
    return pts


def _aligned_positions(axis: str, lo: int, cover: set[tuple[int, int]], shape: dict):
    if axis == "cols":
        left = lo
        tops = set()
        for r_n, c_n in cover:
            for dr, dc in shape["rel"]:
                if c_n == left + dc:
                    tops.add(r_n - dr)
        for top in tops:
            yield top, left
    else:
        top = lo
        lefts = set()
        for r_n, c_n in cover:
            for dr, dc in shape["rel"]:
                if r_n == top + dr:
                    lefts.add(c_n - dc)
        for left in lefts:
            yield top, left


def _detect_cross_bar(g: np.ndarray, bg: int) -> dict | None:
    colors, counts = np.unique(g, return_counts=True)
    color_counts = sorted(
        [(int(cnt), int(col)) for col, cnt in zip(colors, counts) if int(col) != bg],
        reverse=True,
    )
    if len(color_counts) < 2:
        return None
    bar_colors = [color_counts[0][1], color_counts[1][1]]
    mask = np.isin(g, bar_colors)
    occ_arr = np.argwhere(mask)
    if occ_arr.size == 0:
        return None

    occ = {(int(r), int(c)) for r, c in occ_arr}
    best = None
    H, W = g.shape

    for thickness in [5, 6]:
        for center_r in [14, 15]:
            for center_c in [14, 15]:
                for horiz_len in [19, 21]:
                    for vert_len in [19, 21]:
                        hr0 = center_r - thickness // 2
                        hr1 = hr0 + thickness - 1
                        vc0 = center_c - thickness // 2
                        vc1 = vc0 + thickness - 1
                        left_c0 = center_c - horiz_len // 2
                        right_c1 = left_c0 + horiz_len - 1
                        top_r0 = center_r - vert_len // 2
                        bottom_r1 = top_r0 + vert_len - 1
                        if min(hr0, vc0, left_c0, top_r0) < 0 or max(hr1, vc1, right_c1, bottom_r1) >= H:
                            continue

                        full_occ = set()
                        for r in range(hr0, hr1 + 1):
                            for c in range(left_c0, right_c1 + 1):
                                full_occ.add((r, c))
                        for r in range(top_r0, bottom_r1 + 1):
                            for c in range(vc0, vc1 + 1):
                                full_occ.add((r, c))
                        if not occ <= full_occ:
                            continue

                        top_win = {(r, c) for r in range(top_r0, min(top_r0 + 2, hr0)) for c in range(vc0, vc1 + 1)}
                        bottom_win = {(r, c) for r in range(max(hr1 + 1, bottom_r1 - 1), bottom_r1 + 1) for c in range(vc0, vc1 + 1)}
                        left_win = {(r, c) for r in range(hr0, hr1 + 1) for c in range(left_c0, min(left_c0 + 2, vc0))}
                        right_win = {(r, c) for r in range(hr0, hr1 + 1) for c in range(max(vc1 + 1, right_c1 - 1), right_c1 + 1)}
                        missing = full_occ - occ
                        if missing - (top_win | bottom_win | left_win | right_win):
                            continue

                        axis = None
                        color0 = color1 = None
                        row_fit = True
                        row_even = set()
                        row_odd = set()
                        for r in range(top_r0, bottom_r1 + 1):
                            vals = {int(g[r, c]) for c in range(W) if (r, c) in occ}
                            if not vals:
                                continue
                            if len(vals) != 1:
                                row_fit = False
                                break
                            (val,) = tuple(vals)
                            if (r - top_r0) % 2 == 0:
                                row_even.add(val)
                            else:
                                row_odd.add(val)

                        col_fit = True
                        col_even = set()
                        col_odd = set()
                        for c in range(left_c0, right_c1 + 1):
                            vals = {int(g[r, c]) for r in range(H) if (r, c) in occ}
                            if not vals:
                                continue
                            if len(vals) != 1:
                                col_fit = False
                                break
                            (val,) = tuple(vals)
                            if (c - left_c0) % 2 == 0:
                                col_even.add(val)
                            else:
                                col_odd.add(val)

                        if row_fit and len(row_even) == 1 and len(row_odd) == 1 and row_even != row_odd:
                            axis = "rows"
                            color0 = next(iter(row_even))
                            color1 = next(iter(row_odd))
                        elif col_fit and len(col_even) == 1 and len(col_odd) == 1 and col_even != col_odd:
                            axis = "cols"
                            color0 = next(iter(col_even))
                            color1 = next(iter(col_odd))
                        if axis is None:
                            continue

                        cand = {
                            "occ": occ,
                            "bbox": (top_r0, left_c0, bottom_r1, right_c1),
                            "top_r0": top_r0,
                            "bottom_r1": bottom_r1,
                            "left_c0": left_c0,
                            "right_c1": right_c1,
                            "hr0": hr0,
                            "hr1": hr1,
                            "vc0": vc0,
                            "vc1": vc1,
                            "thickness": thickness,
                            "stripe_axis": axis,
                            "color0": int(color0),
                            "color1": int(color1),
                            "missing": len(missing),
                        }
                        if best is None or cand["missing"] < best["missing"]:
                            best = cand
    return best


def _bar_color_map(bar: dict) -> dict[tuple[int, int], int]:
    top_r0, left_c0 = int(bar["top_r0"]), int(bar["left_c0"])
    return {
        (r, c): _stripe_color(
            bar["stripe_axis"],
            int(bar["color0"]),
            int(bar["color1"]),
            r,
            c,
            top_r0,
            left_c0,
        )
        for r, c in bar["occ"]
    }


def _tip_notches(g: np.ndarray, bar: dict, bg: int) -> list[dict]:
    def extract(side: str, r0: int, r1: int, c0: int, c1: int, axis: str) -> list[dict]:
        sub = (g[r0 : r1 + 1, c0 : c1 + 1] == bg)
        out = []
        if not np.any(sub):
            return out
        for pts in components_mask(sub, connectivity=4):
            hole = {(r0 + int(rr), c0 + int(cc)) for rr, cc in pts}
            if side == "top":
                mouth_n = sum(1 for rr, _ in hole if rr == r0)
            elif side == "bottom":
                mouth_n = sum(1 for rr, _ in hole if rr == r1)
            elif side == "left":
                mouth_n = sum(1 for _, cc in hole if cc == c0)
            else:
                mouth_n = sum(1 for _, cc in hole if cc == c1)
            if mouth_n <= 0:
                continue
            out.append({"side": side, "axis": axis, "hole": hole, "mouth_n": int(mouth_n)})
        return out

    top_depth = min(2, bar["hr0"] - bar["top_r0"])
    bottom_depth = min(2, bar["bottom_r1"] - bar["hr1"])
    left_depth = min(2, bar["vc0"] - bar["left_c0"])
    right_depth = min(2, bar["right_c1"] - bar["vc1"])

    notches = []
    if top_depth >= 1:
        notches.extend(
            extract("top", bar["top_r0"], bar["top_r0"] + top_depth - 1, bar["vc0"], bar["vc1"], "cols")
        )
    if bottom_depth >= 1:
        notches.extend(
            extract(
                "bottom",
                bar["bottom_r1"] - bottom_depth + 1,
                bar["bottom_r1"],
                bar["vc0"],
                bar["vc1"],
                "cols",
            )
        )
    if left_depth >= 1:
        notches.extend(
            extract("left", bar["hr0"], bar["hr1"], bar["left_c0"], bar["left_c0"] + left_depth - 1, "rows")
        )
    if right_depth >= 1:
        notches.extend(
            extract(
                "right",
                bar["hr0"],
                bar["hr1"],
                bar["right_c1"] - right_depth + 1,
                bar["right_c1"],
                "rows",
            )
        )
    side_order = {"top": 0, "bottom": 1, "left": 2, "right": 3}
    notches.sort(key=lambda d: (side_order[d["side"]], -d["mouth_n"], len(d["hole"])))
    return notches


def _extends_outward(pts: set[tuple[int, int]], side: str, bar: dict) -> bool:
    if side == "top":
        return any(r < bar["top_r0"] for r, _ in pts)
    if side == "bottom":
        return any(r > bar["bottom_r1"] for r, _ in pts)
    if side == "left":
        return any(c < bar["left_c0"] for _, c in pts)
    return any(c > bar["right_c1"] for _, c in pts)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    bar = _detect_cross_bar(g, bg)
    assert bar is not None

    bar_occ = set(bar["occ"])
    bar_colors = _bar_color_map(bar)
    out = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_occ:
        out[r, c] = bar_colors[(r, c)]

    g_pieces = g.copy()
    for r, c in bar_occ:
        g_pieces[r, c] = int(bg)
    pieces_raw = find_components(g_pieces, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": _shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    notches = _tip_notches(g, bar, bg)
    assert len({n["side"] for n in notches}) == 4

    root_budget = max(1, min(len(notches), len(pieces) // 2))
    per_notch: list[list[dict]] = []
    for ni, notch in enumerate(notches):
        cands = []
        for pi, piece in enumerate(pieces):
            shape = piece["shape"]
            for top, left in _hole_positions(notch["hole"], shape):
                pts = _placement_pts(shape, top, left, bar_occ, H, W, must_cover=notch["hole"])
                if pts is None:
                    continue
                if not _extends_outward(pts, notch["side"], bar):
                    continue
                if notch["axis"] == "cols":
                    lo, hi = int(left), int(left + shape["w"] - 1)
                else:
                    lo, hi = int(top), int(top + shape["h"] - 1)
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                cands.append(
                    (
                        (int(mv), -int(notch["mouth_n"]), len(pts)),
                        {
                            "ni": ni,
                            "pi": pi,
                            "top": int(top),
                            "left": int(left),
                            "axis": notch["axis"],
                            "lo": lo,
                            "hi": hi,
                            "side": notch["side"],
                            "shape": shape,
                        },
                    )
                )
        cands.sort(key=lambda x: x[0])
        per_notch.append([entry for _, entry in cands[:4]])

    cand_roots = [entry for group in per_notch for entry in group]

    best_out = out.copy()
    best_key = None

    def restore_bar(out_grid: np.ndarray) -> None:
        for r, c in bar_occ:
            out_grid[r, c] = bar_colors[(r, c)]

    def search(out_grid: np.ndarray, occ_set: set[tuple[int, int]], chains: list[dict], remaining: set[int]) -> bool:
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (placed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        for ch in chains:
            axis, lo, hi = ch["axis"], int(ch["lo"]), int(ch["hi"])
            head = ch["head"]
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
                continue

            moves = []
            for pi in remaining:
                shape = pieces[pi]["shape"]
                for top, left in _aligned_positions(axis, lo, cavity, shape):
                    pts = _placement_pts(shape, top, left, occ_set, H, W, must_cover=cavity, strip=(axis, lo, hi))
                    if pts is None:
                        continue
                    if not _touches(pts, head_pts):
                        continue
                    moves.append((pi, top, left, pts, shape))
            moves.sort(key=lambda m: (len(m[3]), m[0]))
            for pi, top, left, pts, shape in moves[:6]:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = ch["head"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": shape}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                restore_bar(out_grid)
        return False

    if not cand_roots:
        return out

    for roots_sel in combinations(range(len(cand_roots)), root_budget):
        used_p = set()
        used_n = set()
        occ0 = set(bar_occ)
        out0 = np.full((H, W), int(bg), dtype=int)
        restore_bar(out0)
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
                    "lo": root["lo"],
                    "hi": root["hi"],
                    "side": root["side"],
                    "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                }
            )
        if not ok:
            continue
        remaining = set(range(len(pieces))) - used_p
        if search(out0, occ0, chains, remaining):
            return best_out

    return best_out


def _make_cross_params(rng: random.Random) -> dict:
    H = W = 30
    thickness = rng.choice([5, 6])
    center_r = rng.choice([14, 15])
    center_c = rng.choice([14, 15])
    horiz_len = rng.choice([19, 21])
    vert_len = rng.choice([19, 21])

    hr0 = center_r - thickness // 2
    hr1 = hr0 + thickness - 1
    vc0 = center_c - thickness // 2
    vc1 = vc0 + thickness - 1
    hc0 = center_c - horiz_len // 2
    hc1 = hc0 + horiz_len - 1
    vr0 = center_r - vert_len // 2
    vr1 = vr0 + vert_len - 1
    if min(hr0, vc0, hc0, vr0) < 0 or max(hr1, vc1, hc1, vr1) >= H:
        raise ValueError("cross params out of bounds")
    bar_cells = set()
    for r in range(hr0, hr1 + 1):
        for c in range(hc0, hc1 + 1):
            bar_cells.add((r, c))
    for r in range(vr0, vr1 + 1):
        for c in range(vc0, vc1 + 1):
            bar_cells.add((r, c))
    return {
        "H": H,
        "W": W,
        "thickness": thickness,
        "hr0": hr0,
        "hr1": hr1,
        "hc0": hc0,
        "hc1": hc1,
        "vr0": vr0,
        "vr1": vr1,
        "vc0": vc0,
        "vc1": vc1,
        "top_r0": vr0,
        "bottom_r1": vr1,
        "left_c0": hc0,
        "right_c1": hc1,
        "bar_cells": bar_cells,
    }


def _side_specs(params: dict) -> dict[str, dict]:
    return {
        "top": {"axis": "cols", "span_min": params["vc0"], "span_max": params["vc1"]},
        "bottom": {"axis": "cols", "span_min": params["vc0"], "span_max": params["vc1"]},
        "left": {"axis": "rows", "span_min": params["hr0"], "span_max": params["hr1"]},
        "right": {"axis": "rows", "span_min": params["hr0"], "span_max": params["hr1"]},
    }


def _make_notch(
    rng: random.Random,
    side: str,
    params: dict,
    bar_cells: set[tuple[int, int]],
    used: bool,
) -> dict | None:
    spec = _side_specs(params)[side]
    thickness = params["thickness"]
    for _ in range(20):
        strip_width = rng.choice([4, 5]) if used else rng.choice([2, 3])
        if strip_width > thickness:
            strip_width = thickness
        if strip_width < 2:
            return None
        span0 = rng.randint(spec["span_min"], spec["span_max"] - strip_width + 1)
        span1 = span0 + strip_width - 1
        depth = rng.choice([1, 2])
        if side == "top":
            avail = {(r, c) for r in range(params["top_r0"], params["top_r0"] + depth) for c in range(span0, span1 + 1)}
        elif side == "bottom":
            avail = {(r, c) for r in range(params["bottom_r1"] - depth + 1, params["bottom_r1"] + 1) for c in range(span0, span1 + 1)}
        elif side == "left":
            avail = {(r, c) for r in range(span0, span1 + 1) for c in range(params["left_c0"], params["left_c0"] + depth)}
        else:
            avail = {(r, c) for r in range(span0, span1 + 1) for c in range(params["right_c1"] - depth + 1, params["right_c1"] + 1)}
        avail &= bar_cells
        max_area = min(len(avail), strip_width * depth)
        min_area = strip_width if used else min(strip_width, max_area)
        if max_area < min_area:
            continue
        area = rng.randint(min_area, max_area)
        seeds = {rng.choice(list(avail))}
        notch = _grow_connected(rng, avail, area, seeds, spec["axis"], (span0, span1), retries=12)
        if notch is None:
            continue
        return {"side": side, "axis": spec["axis"], "span0": span0, "span1": span1, "cells": notch}
    return None


def _chain_cavity(prev: set[tuple[int, int]], axis: str, lo: int, hi: int, occ_before: set[tuple[int, int]]) -> set[tuple[int, int]]:
    rows = [r for r, _ in prev]
    cols = [c for _, c in prev]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if axis == "cols":
        return {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(lo, hi + 1)
            if (r, c) not in occ_before and (r, c) not in prev
        }
    return {
        (r, c)
        for r in range(lo, hi + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in occ_before and (r, c) not in prev
    }


def _tail_has_zero_cavity(piece: set[tuple[int, int]], axis: str, lo: int, hi: int, occ_before: set[tuple[int, int]]) -> bool:
    rows = [r for r, _ in piece]
    cols = [c for _, c in piece]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if axis == "cols":
        remaining = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(lo, hi + 1)
            if (r, c) not in occ_before and (r, c) not in piece
        }
    else:
        remaining = {
            (r, c)
            for r in range(lo, hi + 1)
            for c in range(c0, c1 + 1)
            if (r, c) not in occ_before and (r, c) not in piece
        }
    return not remaining


def _build_two_piece_chain(
    rng: random.Random,
    params: dict,
    bar_cells: set[tuple[int, int]],
    occ_before: set[tuple[int, int]],
    notch: dict,
) -> list[set[tuple[int, int]]] | None:
    axis = notch["axis"]
    lo, hi = int(notch["span0"]), int(notch["span1"])
    notch_cells = set(notch["cells"])
    width = hi - lo + 1

    for _ in range(25):
        root_depth = rng.choice([2, 3])
        if notch["side"] == "top":
            outer_lo = max(0, params["top_r0"] - root_depth)
            outer_hi = max(r for r, _ in notch_cells)
            avail = {(r, c) for r in range(outer_lo, outer_hi + 1) for c in range(lo, hi + 1) if (r, c) not in occ_before or (r, c) in notch_cells}
        elif notch["side"] == "bottom":
            outer_lo = min(r for r, _ in notch_cells)
            outer_hi = min(params["H"] - 1, params["bottom_r1"] + root_depth)
            avail = {(r, c) for r in range(outer_lo, outer_hi + 1) for c in range(lo, hi + 1) if (r, c) not in occ_before or (r, c) in notch_cells}
        elif notch["side"] == "left":
            outer_lo = max(0, params["left_c0"] - root_depth)
            outer_hi = max(c for _, c in notch_cells)
            avail = {(r, c) for r in range(lo, hi + 1) for c in range(outer_lo, outer_hi + 1) if (r, c) not in occ_before or (r, c) in notch_cells}
        else:
            outer_lo = min(c for _, c in notch_cells)
            outer_hi = min(params["W"] - 1, params["right_c1"] + root_depth)
            avail = {(r, c) for r in range(lo, hi + 1) for c in range(outer_lo, outer_hi + 1) if (r, c) not in occ_before or (r, c) in notch_cells}

        root_targets = list(range(min(11, len(avail) - 1), max(8, len(notch_cells) + 2) - 1, -1))
        rng.shuffle(root_targets)
        for root_target in root_targets:
            root = _grow_connected(rng, avail, root_target, notch_cells, axis, (lo, hi), retries=12)
            if root is None:
                continue
            if notch["side"] == "top" and not any(r < params["top_r0"] for r, _ in root):
                continue
            if notch["side"] == "bottom" and not any(r > params["bottom_r1"] for r, _ in root):
                continue
            if notch["side"] == "left" and not any(c < params["left_c0"] for _, c in root):
                continue
            if notch["side"] == "right" and not any(c > params["right_c1"] for _, c in root):
                continue

            cavity = _chain_cavity(root, axis, lo, hi, occ_before)
            if not cavity:
                continue
            tail_depth = rng.choice([2, 3])
            root_rows = [r for r, _ in root]
            root_cols = [c for _, c in root]
            if notch["side"] == "top":
                ext = {
                    (r, c)
                    for r in range(max(0, min(root_rows) - tail_depth), min(root_rows))
                    for c in range(lo, hi + 1)
                    if (r, c) not in occ_before and (r, c) not in root
                }
            elif notch["side"] == "bottom":
                ext = {
                    (r, c)
                    for r in range(max(root_rows) + 1, min(params["H"], max(root_rows) + tail_depth + 1))
                    for c in range(lo, hi + 1)
                    if (r, c) not in occ_before and (r, c) not in root
                }
            elif notch["side"] == "left":
                ext = {
                    (r, c)
                    for r in range(lo, hi + 1)
                    for c in range(max(0, min(root_cols) - tail_depth), min(root_cols))
                    if (r, c) not in occ_before and (r, c) not in root
                }
            else:
                ext = {
                    (r, c)
                    for r in range(lo, hi + 1)
                    for c in range(max(root_cols) + 1, min(params["W"], max(root_cols) + tail_depth + 1))
                    if (r, c) not in occ_before and (r, c) not in root
                }

            avail2 = cavity | ext
            if len(avail2) < 8:
                continue
            tail_targets = list(range(min(13, len(avail2)), max(8, len(cavity)) - 1, -1))
            rng.shuffle(tail_targets)
            for tail_target in tail_targets:
                tail = _grow_connected(rng, avail2, tail_target, cavity, axis, (lo, hi), retries=12)
                if tail is None:
                    continue
                if not _tail_has_zero_cavity(tail, axis, lo, hi, occ_before | root):
                    continue
                if not _touches(root, tail):
                    continue
                return [root, tail]
    return None


def _coords_to_rel(coords: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    return [(r - r0, c - c0) for r, c in coords], r0, c0


def _place_scattered(
    rng: random.Random,
    input_grid: np.ndarray,
    placed_occ: set[tuple[int, int]],
    coords: set[tuple[int, int]],
    color: int,
) -> bool:
    H, W = input_grid.shape
    rel, _, _ = _coords_to_rel(coords)
    h = max(r for r, _ in rel) + 1
    w = max(c for _, c in rel) + 1
    for _ in range(700):
        top = rng.randint(0, H - h)
        left = rng.randint(0, W - w)
        pts = {(top + dr, left + dc) for dr, dc in rel}
        if pts & placed_occ:
            continue
        bad = False
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in placed_occ:
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue
        for r, c in pts:
            input_grid[r, c] = int(color)
        placed_occ |= pts
        return True
    return False


def _fusion_signature(grid: np.ndarray) -> dict | None:
    g = np.array(grid, dtype=int)
    bg = get_bg(g)
    bar = _detect_cross_bar(g, bg)
    if bar is None:
        return None
    notches = _tip_notches(g, bar, bg)
    return {
        "stripe_axis": bar["stripe_axis"],
        "notch_sides": sorted({n["side"] for n in notches}),
        "notch_count": len(notches),
        "bar_colors": sorted({bar["color0"], bar["color1"]}),
    }


def _gen_attempt(seed: int):
    rng = random.Random(seed)
    params = _make_cross_params(rng)
    H, W = params["H"], params["W"]
    bg = 0

    bar_colors = rng.sample(ALL_COLORS, 2)
    stripe_axis = rng.choice(["rows", "cols"])
    active_sides = set(rng.sample(["top", "bottom", "left", "right"], 3))
    side_order = ["top", "bottom", "left", "right"]

    bar_cells = set(params["bar_cells"])
    notch_infos = {}
    for side in side_order:
        notch = _make_notch(rng, side, params, bar_cells, used=(side in active_sides))
        if notch is None:
            return None
        notch_infos[side] = notch
        bar_cells -= notch["cells"]

    occ_set = set(bar_cells)
    piece_colors = [c for c in ALL_COLORS if c not in bar_colors]
    rng.shuffle(piece_colors)
    if len(piece_colors) < 6:
        return None

    pieces = []
    color_idx = 0
    for side in side_order:
        if side not in active_sides:
            continue
        chain = _build_two_piece_chain(rng, params, bar_cells, occ_set, notch_infos[side])
        if chain is None:
            return None
        for piece_cells in chain:
            pieces.append({"color": piece_colors[color_idx], "coords": piece_cells, "side": side})
            color_idx += 1
            occ_set |= piece_cells

    if len(pieces) != 6:
        return None

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = _stripe_color(stripe_axis, bar_colors[0], bar_colors[1], r, c, params["top_r0"], params["left_c0"])
    for piece in pieces:
        for r, c in piece["coords"]:
            output[r, c] = int(piece["color"])

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = _stripe_color(stripe_axis, bar_colors[0], bar_colors[1], r, c, params["top_r0"], params["left_c0"])

    placed_occ = set(bar_cells)
    scatter_order = list(pieces)
    rng.shuffle(scatter_order)
    for piece in scatter_order:
        if not _place_scattered(rng, input_grid, placed_occ, piece["coords"], int(piece["color"])):
            return None

    sig = _fusion_signature(input_grid)
    if sig is None or sig["notch_count"] < 4 or len(sig["notch_sides"]) != 4:
        return None
    # Reject ambiguous layouts here rather than relying on generate_input() to
    # retry after the fact. This keeps the raw generator aligned with main().
    if not np.array_equal(main(input_grid), output):
        return None
    return input_grid, output


def generate_input() -> np.ndarray:
    for attempt in range(400):
        result = _gen_attempt(1000 + attempt * 17 + random.randint(0, 999_999))
        if result is None:
            continue
        inp, _ = result
        return inp
    raise RuntimeError("generate_input: failed after 400 attempts")
