# Variant of 16b78196
# Dimension: D1 — 匹配条件：形状嵌合 vs 其他匹配逻辑 (match_cond)
# Change: s.shape fits into d.notch.shape — 精灵形状恰好填充杆上缺口的形状... → 连接器颜色+形状双重匹配
#
# Design analysis
# ---------------
# spec_like:
#   kept_from_original:
#     grid: 30x30
#     bg: 0
#     bar:
#       largest_monochrome_component: true
#       orientation: horizontal or vertical
#       thickness: 5-7
#       span: full width or full height
#     chains:
#       count: 1-2
#       length_each: 2-3
#       continuation_rule: later piece fills previous bbox cavity inside the same strip
#       no_rotation: true
#     piece_area:
#       continuation: 8-13
#       root: 10-13 in practice because the socket itself already occupies 8-10 cells
#   changed_for_D1:
#     sockets:
#       all bar notches are standardized 2-cell-deep sockets of one shared strip width (4 or 5)
#       geometry alone is intentionally weakened, so several sockets on the same side are body-compatible
#     connector:
#       every socket contains a tiny colored marker (2-3 cells) chosen from a small prototype family
#       every root piece contains the same tiny marker embedded at its bar-facing side
#       a root may dock only when both connector color and connector shape match exactly
#     decoys:
#       non-used sockets also carry connector markers
#       at least one extra socket exists on each used side so ignoring connectors leaves multiple geometric fits
#
# a) concrete behavior trace
#   Original root logic:
#     choose a scattered root piece whose full shape can be translated so that it covers a notch hole.
#   Variant root logic:
#     first align a root piece to a socket by its connector anchor.
#     then verify that the translated full piece covers the entire socket region and only overlaps the bar
#     at the matching connector cells.
#   Example:
#     two top-side sockets have the same 2x4 geometry. A root piece can geometrically cover either socket.
#     one socket has an orange L-connector, the other has an orange domino. The root has an orange L-connector.
#     original shape-only logic would treat both placements as plausible; the variant must pick only the L socket.
#
# b) complexity calibration
#   Shapes/colors are not exploded into a large codebook. Connector shapes are limited to 3 prototypes and
#   connector colors are limited to the 1-2 colors used by actual roots, re-used by decoys. This keeps the
#   matching rule learnable while still requiring a conjunction of attributes.
#
# c) implicit assumption audit against the original main()
#   - Original piece parsing assumed every piece is monochrome. That is no longer safe: roots are now
#     two-color components, so the solver must parse multicolor components and identify a dominant body color
#     plus one small connector component.
#   - Original root enumeration searched all translations that cover a hole. That depended on "hole shape
#     determines the root". In the variant, translation is instead fixed by connector alignment.
#   - Original undo logic restored only bar pixels after backtracking. That was harmless because the bar was
#     the only fixed foreground. Here notch markers are also fixed foreground, so undo must restore both bar
#     cells and fixed connector cells.
#   - Original extension search could consider any remaining piece. Here root pieces have connectors and are
#     guaranteed by the generator to be used only as roots, so extension search explicitly skips connector-bearing
#     pieces.
#
# d) placement legality
#   Generator uses an output-first construction:
#   - carve disjoint sockets from the bar;
#   - plant fixed connector markers inside them;
#   - build root pieces that fully occupy their used socket and extend outward in the correct strip;
#   - grow later pieces from cavity + further extension exactly as in the original task;
#   - scatter the finished pieces away from the bar without touching it or one another.
#   This guarantees no overlap, no out-of-bounds placement, and compatibility with the solver's assumptions.
#
# e) visual observability
#   The new rule is visible in 2-4 examples because:
#   - sockets on the same side share the same geometry;
#   - roots are the only scattered pieces that carry tiny colored connector marks;
#   - decoy sockets share either connector color or connector shape with a true socket, so both attributes
#     must be noticed rather than only one.
#
# f) visual cleanliness
#   Connectors are tiny 2-3 cell glyphs placed wholly inside a 2-cell-deep socket. They do not overlap later
#   chain pieces except at the root docking zone, and scattered pieces remain separated from the bar.
#
# g) color inferability
#   Connector colors are few and repeated. The solver never relies on numeric color order. Colors only act as
#   visible equality labels paired with a small visible shape family.

from __future__ import annotations

from collections import Counter
import random as _random

import numpy as np

from arc_utils import components_mask, find_components, get_bg


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CANONICAL_CONNECTORS = [
    {(0, 1), (1, 1)},
    {(0, 1), (0, 2)},
    {(0, 1), (1, 1), (1, 2)},
]


def _bbox_of(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(cells) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    cells = list(cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells), (r0, c0)


def _shape_from_coords(coords: np.ndarray, bbox) -> dict:
    r0, c0, r1, c1 = map(int, bbox)
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": int(r1 - r0 + 1), "w": int(c1 - c0 + 1)}


def _shift_rel(rel_cells, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in rel_cells}


def _piece_from_component(grid: np.ndarray, coords: np.ndarray) -> dict:
    bbox = tuple(map(int, (coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max())))
    shape = _shape_from_coords(coords, bbox)
    r0, c0, _r1, _c1 = bbox
    colors_rel = {}
    counts = Counter()
    for r, c in coords:
        rr, cc = int(r - r0), int(c - c0)
        col = int(grid[int(r), int(c)])
        colors_rel[(rr, cc)] = col
        counts[col] += 1
    body_color = counts.most_common(1)[0][0]
    conn_rel = {rc for rc, col in colors_rel.items() if col != body_color}
    conn_color = None
    conn_norm = None
    conn_anchor = None
    if conn_rel:
        conn_colors = {colors_rel[rc] for rc in conn_rel}
        assert len(conn_colors) == 1
        conn_color = next(iter(conn_colors))
        conn_norm, conn_anchor = _normalize(conn_rel)
    return {
        "shape": shape,
        "colors_rel": colors_rel,
        "body_color": int(body_color),
        "conn_rel": conn_rel,
        "conn_color": None if conn_color is None else int(conn_color),
        "conn_norm": conn_norm,
        "conn_anchor": conn_anchor,
        "orig": (int(r0), int(c0)),
    }


def _pixels(shape: dict, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in shape["rel"]}


def _placement_pts(
    shape: dict,
    top: int,
    left: int,
    occ: set[tuple[int, int]],
    H: int,
    W: int,
    *,
    must_cover: set[tuple[int, int]] | None = None,
    strip: tuple[str, int, int] | None = None,
    allowed_overlap: set[tuple[int, int]] | None = None,
):
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
    allowed_overlap = allowed_overlap or set()
    if (pts & occ) - allowed_overlap:
        return None
    if must_cover is not None and not (must_cover <= pts):
        return None
    return pts


def _aligned_positions(axis: str, A: int, cover: set[tuple[int, int]], shape: dict):
    if axis == "cols":
        left = int(A)
        tops = set()
        for r_cov, c_cov in cover:
            for lr, lc in shape["rel"]:
                if c_cov == left + lc:
                    tops.add(int(r_cov - lr))
        for top in tops:
            yield top, left
    else:
        top = int(A)
        lefts = set()
        for r_cov, c_cov in cover:
            for lr, lc in shape["rel"]:
                if r_cov == top + lr:
                    lefts.add(int(c_cov - lc))
        for left in lefts:
            yield top, left


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in b:
                return True
    return False


def _paint_piece(out: np.ndarray, piece: dict, top: int, left: int) -> None:
    for (rr, cc), col in piece["colors_rel"].items():
        out[int(top + rr), int(left + cc)] = int(col)


def _clear_piece(out: np.ndarray, piece: dict, top: int, left: int, fixed_colors: dict, bg: int) -> None:
    for rr, cc in piece["shape"]["rel"]:
        rc = (int(top + rr), int(left + cc))
        out[rc] = int(fixed_colors.get(rc, bg))


def _component_inside_bbox(cells: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells)


def _socket_connector_local(side: str, proto_id: int) -> set[tuple[int, int]]:
    canon = CANONICAL_CONNECTORS[int(proto_id)]
    if side == "top":
        return set(canon)
    if side == "bottom":
        return {(1 - r, c) for r, c in canon}
    if side == "left":
        return {(c, r) for r, c in canon}
    if side == "right":
        return {(c, 1 - r) for r, c in canon}
    raise ValueError(side)


def _socket_window(side: str, start: int, width: int, bar_bounds: tuple[int, int, int, int]) -> tuple[set[tuple[int, int]], callable]:
    bar_r0, bar_c0, bar_r1, bar_c1 = bar_bounds
    if side == "top":
        return (
            {(bar_r0 + r, start + c) for r in range(2) for c in range(width)},
            lambda rr, cc: (bar_r0 + rr, start + cc),
        )
    if side == "bottom":
        return (
            {(bar_r1 - 1 + r, start + c) for r in range(2) for c in range(width)},
            lambda rr, cc: (bar_r1 - 1 + rr, start + cc),
        )
    if side == "left":
        return (
            {(start + r, bar_c0 + c) for r in range(width) for c in range(2)},
            lambda rr, cc: (start + rr, bar_c0 + cc),
        )
    if side == "right":
        return (
            {(start + r, bar_c1 - 1 + c) for r in range(width) for c in range(2)},
            lambda rr, cc: (start + rr, bar_c1 - 1 + cc),
        )
    raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    mono.sort(key=lambda d: -d["area"])
    assert mono
    bar = mono[0]

    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_bbox = (bar_r0, bar_c0, bar_r1, bar_c1)
    bar_h, bar_w = int(bar_r1 - bar_r0 + 1), int(bar_c1 - bar_c0 + 1)
    bar_color = int(bar["color"])

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[int(r - bar_r0), int(c - bar_c0)] = True

    def parse_notches():
        notches = []
        empty = ~bar_mask
        for pts in components_mask(empty, connectivity=4):
            rr = pts[:, 0]
            cc = pts[:, 1]
            top = int((rr == 0).sum())
            bottom = int((rr == bar_h - 1).sum())
            left = int((cc == 0).sum())
            right = int((cc == bar_w - 1).sum())
            if top + bottom + left + right == 0:
                continue
            region = {(bar_r0 + int(r), bar_c0 + int(c)) for r, c in pts}
            hole_bg = {rc for rc in region if int(g[rc]) == int(bg)}
            marker_colors = {rc: int(g[rc]) for rc in region if int(g[rc]) not in (int(bg), int(bar_color))}
            conn_cells = set(marker_colors)
            conn_color = None
            conn_norm = None
            conn_anchor = None
            if conn_cells:
                colors = {marker_colors[rc] for rc in conn_cells}
                assert len(colors) == 1
                conn_color = next(iter(colors))
                conn_norm, conn_anchor = _normalize(conn_cells)
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append(
                {
                    "axis": axis,
                    "side": side,
                    "region": region,
                    "hole_bg": hole_bg,
                    "marker_colors": marker_colors,
                    "conn_cells": conn_cells,
                    "conn_color": None if conn_color is None else int(conn_color),
                    "conn_norm": conn_norm,
                    "conn_anchor": conn_anchor,
                    "mouth_n": int(mouth_n),
                }
            )
        return notches

    notches = parse_notches()

    fixed = np.full((H, W), int(bg), dtype=int)
    fixed_colors = {}
    for r, c in bar_occ:
        fixed[r, c] = int(bar_color)
        fixed_colors[(r, c)] = int(bar_color)
    for n in notches:
        for rc, col in n["marker_colors"].items():
            fixed[rc] = int(col)
            fixed_colors[rc] = int(col)

    work = g.copy()
    for r, c in bar_occ:
        work[r, c] = int(bg)
    other = find_components(work, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    pieces = []
    for comp in other:
        cells = {(int(r), int(c)) for r, c in comp["coords"]}
        if len(cells) < 8:
            continue
        pieces.append(_piece_from_component(g, comp["coords"]))

    occ_fixed = set(fixed_colors)
    root_cands = []
    for ni, n in enumerate(notches):
        if not n["conn_cells"]:
            continue
        for pi, p in enumerate(pieces):
            if not p["conn_rel"]:
                continue
            if p["conn_color"] != n["conn_color"] or p["conn_norm"] != n["conn_norm"]:
                continue
            top = int(n["conn_anchor"][0] - p["conn_anchor"][0])
            left = int(n["conn_anchor"][1] - p["conn_anchor"][1])
            if _shift_rel(p["conn_rel"], top, left) != n["conn_cells"]:
                continue
            pts = _placement_pts(
                p["shape"],
                top,
                left,
                occ_fixed,
                H,
                W,
                must_cover=n["region"],
                allowed_overlap=n["conn_cells"],
            )
            if pts is None:
                continue
            if n["axis"] == "cols":
                A, B = int(left), int(left + p["shape"]["w"] - 1)
            else:
                A, B = int(top), int(top + p["shape"]["h"] - 1)
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
                        "A": int(A),
                        "B": int(B),
                        "pts": pts,
                        "allow_overlap": set(n["conn_cells"]),
                    },
                )
            )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [x[1] for x in root_cands[:20]]
    best_out = fixed.copy()
    best_key = None

    def search(out_grid: np.ndarray, occ_set: set[tuple[int, int]], chains: list[dict], remaining: set[int]) -> bool:
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (placed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        def gen_extend_moves(chain: dict):
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = _pixels(head["shape"], head["top"], head["left"])
            cavity = {
                (r, c)
                for r in range(int(head["top"]), int(head["top"] + head["shape"]["h"]))
                for c in range(int(head["left"]), int(head["left"] + head["shape"]["w"]))
                if int(out_grid[r, c]) == int(bg)
            }
            if not cavity:
                return []
            moves = []
            for pi in remaining:
                piece = pieces[pi]
                if piece["conn_rel"]:
                    continue
                for top, left in _aligned_positions(axis, A, cavity, piece["shape"]):
                    pts = _placement_pts(piece["shape"], top, left, occ_set, H, W, must_cover=cavity, strip=(axis, A, B))
                    if pts is None or not _touches(pts, head_pts):
                        continue
                    mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                    moves.append((int(mv), pi, int(top), int(left), pts))
            moves.sort()
            return moves

        for ch in chains:
            moves = gen_extend_moves(ch)
            if not moves:
                continue
            for _score, pi, top, left, pts in moves[:8]:
                _paint_piece(out_grid, pieces[pi], top, left)
                old_head = ch["head"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": pieces[pi]["shape"]}
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ_set | pts, chains, rem2):
                    return True
                ch["head"] = old_head
                _clear_piece(out_grid, pieces[pi], top, left, fixed_colors, bg)
        return False

    from itertools import combinations

    max_roots = min(3, len(cand_roots))
    for k in range(1, max_roots + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(occ_fixed)
            out0 = fixed.copy()
            chains = []
            ok = True
            for idx in roots_sel:
                root = cand_roots[idx]
                if root["pi"] in used_p or root["ni"] in used_n:
                    ok = False
                    break
                if (root["pts"] & occ0) - root["allow_overlap"]:
                    ok = False
                    break
                used_p.add(root["pi"])
                used_n.add(root["ni"])
                _paint_piece(out0, pieces[root["pi"]], root["top"], root["left"])
                occ0 |= root["pts"]
                chains.append(
                    {
                        "axis": root["axis"],
                        "A": root["A"],
                        "B": root["B"],
                        "head": {"top": root["top"], "left": root["left"], "shape": pieces[root["pi"]]["shape"]},
                    }
                )
            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


def _connected(cells: set[tuple[int, int]]) -> bool:
    if len(cells) <= 1:
        return True
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=4):
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 24):
            if len(cells) >= target:
                break
            frontier = []
            for r, c in list(cells):
                for dr, dc in DIR4:
                    nb = (r + dr, c + dc)
                    if nb in available and nb not in cells:
                        frontier.append(nb)
            if not frontier:
                break
            if must_span_axis and span_range:
                A, B = span_range
                if must_span_axis == "cols":
                    cur0 = min(c for _, c in cells)
                    cur1 = max(c for _, c in cells)
                    if cur0 > A:
                        pref = [nb for nb in frontier if nb[1] < cur0]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                    if cur1 < B:
                        pref = [nb for nb in frontier if nb[1] > cur1]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                else:
                    cur0 = min(r for r, _ in cells)
                    cur1 = max(r for r, _ in cells)
                    if cur0 > A:
                        pref = [nb for nb in frontier if nb[0] < cur0]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                    if cur1 < B:
                        pref = [nb for nb in frontier if nb[0] > cur1]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
            cells.add(rng.choice(frontier))
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


def _bar_setup(rng, H: int, W: int):
    orientation = rng.choice(["horizontal", "vertical"])
    thickness = rng.randint(5, 7)
    bar_color = rng.randint(1, 9)
    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * H)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = min(H - 1, bar_r0 + thickness - 1)
        bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        room = {"top": bar_r0, "bottom": H - 1 - bar_r1}
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = min(W - 1, bar_c0 + thickness - 1)
        bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        room = {"left": bar_c0, "right": W - 1 - bar_c1}
    return orientation, thickness, bar_color, (bar_r0, bar_c0, bar_r1, bar_c1), room


def _slot_starts(span_len: int, strip_width: int) -> list[int]:
    return list(range(1, span_len - strip_width + 1, strip_width + 1))


def _pick_used_windows(rng, orientation, room, chain_lens, strip_width, span_len):
    sides = ["top", "bottom"] if orientation == "horizontal" else ["left", "right"]
    slots = {side: _slot_starts(span_len, strip_width) for side in sides}
    used = []
    for chain_len in chain_lens:
        valid_sides = [side for side in sides if room[side] >= chain_len * 2 + 3 and slots[side]]
        if not valid_sides:
            return None
        side = rng.choice(valid_sides)
        start = rng.choice(slots[side])
        slots[side].remove(start)
        used.append({"side": side, "start": start})
    return used, slots


def _piece_grid(piece: dict) -> tuple[dict, int, int]:
    rs = [r for r, _ in piece["colors"]]
    cs = [c for _, c in piece["colors"]]
    r0, c0 = min(rs), min(cs)
    return {(r - r0, c - c0): col for (r, c), col in piece["colors"].items()}, max(rs) - r0 + 1, max(cs) - c0 + 1


def _scatter_piece(rng, piece: dict, input_grid: np.ndarray, placed_occ: set[tuple[int, int]], orientation: str, bar_bounds):
    H, W = input_grid.shape
    rel_colors, p_h, p_w = _piece_grid(piece)
    bar_r0, bar_c0, bar_r1, bar_c1 = bar_bounds
    for _ in range(600):
        gap = rng.randint(2, 7)
        if orientation == "horizontal":
            top = (bar_r0 - gap - p_h) if rng.random() < 0.5 else (bar_r1 + gap + 1)
            left = rng.randint(0, max(0, W - p_w))
        else:
            left = (bar_c0 - gap - p_w) if rng.random() < 0.5 else (bar_c1 + gap + 1)
            top = rng.randint(0, max(0, H - p_h))
        if top < 0 or left < 0 or top + p_h > H or left + p_w > W:
            continue
        pts = {(top + r, left + c) for r, c in rel_colors}
        if pts & placed_occ:
            continue
        if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in DIR4):
            continue
        for (rr, cc), col in rel_colors.items():
            input_grid[int(top + rr), int(left + cc)] = int(col)
        placed_occ |= pts
        return True
    return False


def _gen_attempt():
    rng = _random.Random()
    H = W = 30
    bg = 0

    orientation, _thickness, bar_color, bar_bounds, room = _bar_setup(rng, H, W)
    bar_r0, bar_c0, bar_r1, bar_c1 = bar_bounds
    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    num_chains = rng.randint(1, 2)
    chain_lens = [3] if num_chains == 1 else [rng.randint(2, 3) for _ in range(2)]
    if num_chains == 2:
        while not (3 <= sum(chain_lens) <= 6):
            chain_lens = [rng.randint(2, 3) for _ in range(2)]
    total_pieces = sum(chain_lens)

    strip_width = rng.randint(4, 5)
    span_len = W if orientation == "horizontal" else H
    picked = _pick_used_windows(rng, orientation, room, chain_lens, strip_width, span_len)
    if picked is None:
        return None
    used_windows, free_slots = picked

    palette = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(palette)
    if len(palette) < total_pieces + num_chains:
        return None
    body_colors = palette[:total_pieces]
    connector_colors = palette[total_pieces : total_pieces + num_chains]

    proto_ids = list(range(len(CANONICAL_CONNECTORS)))
    used_specs = []
    seen_pairs = set()
    for i in range(num_chains):
        pairs = [(connector_colors[i], pid) for pid in proto_ids if (connector_colors[i], pid) not in seen_pairs]
        if not pairs:
            return None
        pair = rng.choice(pairs)
        used_specs.append({"color": pair[0], "proto": pair[1]})
        seen_pairs.add(pair)

    side_counts = Counter(w["side"] for w in used_windows)
    reserved_decoys = []
    for idx, win in enumerate(used_windows):
        if side_counts[win["side"]] == 1:
            side = win["side"]
            if not free_slots[side]:
                return None
            start = rng.choice(free_slots[side])
            free_slots[side].remove(start)
            reserved_decoys.append({"side": side, "start": start, "ref_chain": idx})

    total_slots = sum(len(v) for v in free_slots.values()) + len(reserved_decoys) + len(used_windows)
    target_notches = rng.randint(max(6, num_chains + 4), min(8, total_slots))
    need_more = target_notches - len(used_windows) - len(reserved_decoys)
    if need_more < 0:
        return None

    decoy_windows = list(reserved_decoys)
    flat_slots = [(side, start) for side, vals in free_slots.items() for start in vals]
    rng.shuffle(flat_slots)
    for side, start in flat_slots[:need_more]:
        free_slots[side].remove(start)
        decoy_windows.append({"side": side, "start": start, "ref_chain": None})
    if len(decoy_windows) != target_notches - len(used_windows):
        return None

    def make_decoy_spec(ref_chain):
        if ref_chain is not None:
            base = used_specs[ref_chain]
            share_modes = ["color"]
            if num_chains >= 2:
                share_modes.append("shape")
            mode = rng.choice(share_modes)
            if mode == "color":
                alt = [pid for pid in proto_ids if pid != base["proto"]]
                return {"color": base["color"], "proto": rng.choice(alt)}
            other_colors = [spec["color"] for spec in used_specs if spec["color"] != base["color"]]
            if not other_colors:
                alt = [pid for pid in proto_ids if pid != base["proto"]]
                return {"color": base["color"], "proto": rng.choice(alt)}
            return {"color": rng.choice(other_colors), "proto": base["proto"]}
        base = rng.choice(used_specs)
        if rng.random() < 0.6:
            alt = [pid for pid in proto_ids if pid != base["proto"]]
            return {"color": base["color"], "proto": rng.choice(alt)}
        other = rng.choice(used_specs)
        pair = {"color": other["color"], "proto": base["proto"]}
        if (pair["color"], pair["proto"]) not in seen_pairs:
            return pair
        alt = [pid for pid in proto_ids if pid != base["proto"]]
        return {"color": base["color"], "proto": rng.choice(alt)}

    notches = []
    for idx, win in enumerate(used_windows):
        notches.append({**win, "used": True, "spec": used_specs[idx], "chain_idx": idx})
    for win in decoy_windows:
        spec = make_decoy_spec(win["ref_chain"])
        if (spec["color"], spec["proto"]) in seen_pairs:
            if spec["color"] == used_specs[win["ref_chain"]]["color"] if win["ref_chain"] is not None else False:
                return None
        notches.append({**win, "used": False, "spec": spec, "chain_idx": None})

    connector_markers = {}
    notch_regions = []
    for n in notches:
        socket_cells, to_global = _socket_window(n["side"], n["start"], strip_width, bar_bounds)
        conn_local = _socket_connector_local(n["side"], n["spec"]["proto"])
        conn_cells = {to_global(r, c) for r, c in conn_local}
        for rc in conn_cells:
            connector_markers[rc] = int(n["spec"]["color"])
        notch_regions.append({**n, "socket_cells": socket_cells, "conn_cells": conn_cells})
    for n in notch_regions:
        bar_cells -= n["socket_cells"]

    occ_set = set(bar_cells) | set(connector_markers)
    all_chain_pieces = []
    body_idx = 0

    for chain_idx in range(num_chains):
        n = next(x for x in notch_regions if x["used"] and x["chain_idx"] == chain_idx)
        A = n["start"]
        B = n["start"] + strip_width - 1
        side = n["side"]
        chain_pieces = []
        for piece_idx in range(chain_lens[chain_idx]):
            piece_color = body_colors[body_idx]
            body_idx += 1
            chain_dim = rng.randint(2, 3)
            if piece_idx == 0:
                socket = set(n["socket_cells"])
                if orientation == "horizontal":
                    if side == "top":
                        ext_rows = range(max(0, bar_r0 - chain_dim), bar_r0)
                    else:
                        ext_rows = range(bar_r1 + 1, min(H, bar_r1 + 1 + chain_dim))
                    ext = {(r, c) for r in ext_rows for c in range(A, B + 1) if (r, c) not in occ_set}
                    span_axis = "cols"
                else:
                    if side == "left":
                        ext_cols = range(max(0, bar_c0 - chain_dim), bar_c0)
                    else:
                        ext_cols = range(bar_c1 + 1, min(W, bar_c1 + 1 + chain_dim))
                    ext = {(r, c) for r in range(A, B + 1) for c in ext_cols if (r, c) not in occ_set}
                    span_axis = "rows"
                all_avail = socket | ext
                if len(ext) < 2:
                    return None
                low = max(len(socket) + 2, 10)
                high = min(13, len(all_avail))
                if low > high:
                    return None
                target = rng.randint(low, high)
                piece_cells = _grow_connected(rng, all_avail, target, socket, must_span_axis=span_axis, span_range=(A, B), retries=6)
                if piece_cells is None or len(piece_cells - socket) < 2:
                    return None
                colors = {rc: int(n["spec"]["color"]) if rc in n["conn_cells"] else int(piece_color) for rc in piece_cells}
            else:
                prev = chain_pieces[-1]["coords"]
                pr0, pc0, pr1, pc1 = _bbox_of(prev)
                if orientation == "horizontal":
                    cavity = {
                        (r, c)
                        for r in range(pr0, pr1 + 1)
                        for c in range(A, B + 1)
                        if (r, c) not in occ_set and (r, c) not in prev
                    }
                    if side == "top":
                        ext_rows = range(max(0, pr0 - chain_dim), pr0)
                    else:
                        ext_rows = range(pr1 + 1, min(H, pr1 + 1 + chain_dim))
                    ext = {(r, c) for r in ext_rows for c in range(A, B + 1) if (r, c) not in occ_set}
                    span_axis = "cols"
                else:
                    cavity = {
                        (r, c)
                        for r in range(A, B + 1)
                        for c in range(pc0, pc1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev
                    }
                    if side == "left":
                        ext_cols = range(max(0, pc0 - chain_dim), pc0)
                    else:
                        ext_cols = range(pc1 + 1, min(W, pc1 + 1 + chain_dim))
                    ext = {(r, c) for r in range(A, B + 1) for c in ext_cols if (r, c) not in occ_set}
                    span_axis = "rows"
                if not cavity:
                    return None
                all_avail = cavity | ext
                low = max(8, len(cavity))
                high = min(13, len(all_avail))
                if piece_idx == chain_lens[chain_idx] - 1:
                    low = high
                if low > high:
                    return None
                target = rng.randint(low, high)
                piece_cells = _grow_connected(rng, all_avail, target, cavity, must_span_axis=span_axis, span_range=(A, B), retries=6)
                if piece_cells is None or not cavity.issubset(piece_cells):
                    return None
                if piece_idx == chain_lens[chain_idx] - 1:
                    t_r0, t_c0, t_r1, t_c1 = _bbox_of(piece_cells)
                    if orientation == "horizontal":
                        remaining = {
                            (r, c)
                            for r in range(t_r0, t_r1 + 1)
                            for c in range(A, B + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    else:
                        remaining = {
                            (r, c)
                            for r in range(A, B + 1)
                            for c in range(t_c0, t_c1 + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    if remaining:
                        return None
                colors = {rc: int(piece_color) for rc in piece_cells}

            chain_pieces.append({"coords": piece_cells, "colors": colors})
            occ_set |= piece_cells
        all_chain_pieces.extend(chain_pieces)

    if len(all_chain_pieces) != total_pieces:
        return None

    output = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_cells:
        output[r, c] = int(bar_color)
    for rc, col in connector_markers.items():
        output[rc] = int(col)
    for piece in all_chain_pieces:
        for rc, col in piece["colors"].items():
            output[rc] = int(col)

    input_grid = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = int(bar_color)
    for rc, col in connector_markers.items():
        input_grid[rc] = int(col)

    placed_occ = set(bar_cells) | set(connector_markers)
    order = list(all_chain_pieces)
    rng.shuffle(order)
    for piece in order:
        if not _scatter_piece(rng, piece, input_grid, placed_occ, orientation, bar_bounds):
            return None

    return input_grid, output


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 400 attempts")
