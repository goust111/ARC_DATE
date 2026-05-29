# Combo variant of 16b78196 (index 4)
# combo_spec: D1/3+D2/0
# Name: 连接器双匹配+多色精灵
#
# Design analysis
# ---------------
# a) Concrete behavior trace
#   Original task:
#   - A monochrome root piece is searched by pure silhouette fit: if its occupied cells can translate to cover a
#     bar notch, it can become the chain root.
#   - Later monochrome pieces fill the previous piece's bbox cavity within the same strip width.
#   - Piece color is incidental after identification.
#
#   Fused variant:
#   - Each used socket in the bar contains a tiny connector marker: a 2-3 cell glyph with its own color.
#   - Each root sprite is now three-layered:
#       1) connector cells: tiny boundary mark, used only for root docking,
#       2) outline color: the visible body boundary,
#       3) fill color: strict interior cells that must survive placement.
#   - Each non-root sprite is two-layered:
#       1) outline color on the silhouette boundary,
#       2) fill color on strict interior cells.
#   - Root matching is no longer "any translation covering the notch". The connector anchor fixes the translation,
#     then the full silhouette must still cover the socket region. Later chain growth still follows cavity filling,
#     but painting must preserve outline/fill colors per cell.
#
#   Small example:
#   - Suppose two top sockets have the same 2x4 geometry. Socket A has an orange L-connector, socket B has an
#     orange domino connector. A scattered root sprite has an orange L-connector, blue outline, red interior.
#   - Original logic: the same silhouette could dock into either socket.
#   - Fused logic: only socket A is legal because both connector color and connector shape match. After docking,
#     the piece still renders as blue boundary + red interior + orange connector.
#
# b) Complexity calibration
#   - Connector shapes are limited to three tiny prototypes, and connector colors are reused across used/decoy
#     sockets. This keeps the joint rule learnable rather than turning into a large arbitrary codebook.
#   - Body colors are also controlled: pieces reuse a small outline pool and a small fill pool. The visual change is
#     obvious, but the number of colors remains bounded and repeated across examples.
#   - Geometry ambiguity is intentionally preserved by standardized sockets of one strip width, so connector logic
#     does real work instead of being redundant.
#
# c) Implicit assumption audit against the original solver
#   - "Every piece is monochrome" is broken. Parsing must use multicolor connected components and identify outline,
#     fill, and possibly connector subsets separately.
#   - "Root translation can be enumerated from hole coverage" is broken. In the fused rule, connector anchor
#     alignment determines root placement; pure geometry is only used as a legality check afterward.
#   - "Undo only needs to restore bar pixels" is broken. Socket connector markers are fixed foreground too, so
#     backtracking must restore both the bar and marker cells.
#   - "Any remaining piece may extend a chain" is broken. Connector-bearing pieces are roots only; continuation
#     search must skip them.
#   - New fused assumption: fill cells are strict interior, connector cells are tiny boundary marks. The generator
#     explicitly constructs pieces to satisfy this so the solver's structural parsing is valid.
#
# d) Placement legality
#   - The generator is output-first. It first builds the final bar, sockets, socket markers, and chain silhouettes.
#   - Each piece is colored only after its silhouette is fixed, ensuring connector cells are inside the root,
#     fill cells are strict interior, outline is the remaining body, and nothing overlaps or goes out of bounds.
#   - Input is produced by scattering finished pieces away from the bar while preventing overlap and adjacency.
#
# e) Visual observability
#   - A human can see the fusion because:
#       1) roots are the only scattered sprites with tiny boundary connectors,
#       2) multiple sockets on the same side share the same body geometry,
#       3) pieces are visibly multicolor, and their interior colors persist after assembly.
#   - Decoy sockets reuse either connector color or connector shape from true sockets, so both connector attributes
#     must be noticed rather than only one.
#
# f) Visual cleanliness
#   - Connector marks are tiny and confined to socket zones / root boundaries.
#   - Fill colors are only used on strict interior cells, so outline vs fill remains legible.
#   - Scattered pieces are separated from one another and from the bar, avoiding ambiguous merges.
#
# g) Inferability of color changes
#   - There is no hidden numeric-color logic. Colors act as visible equality labels:
#       connector color pairs root-to-socket,
#       outline/fill color layering is local to each piece and preserved verbatim in the output.
#   - Because outline/fill pools are reused, examples show repeated evidence of the same two-layer representation.

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
    pts = list(cells)
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return frozenset((int(r - r0), int(c - c0)) for r, c in pts), (int(r0), int(c0))


def _shape_from_coords(coords: np.ndarray, bbox) -> dict:
    r0, c0, r1, c1 = map(int, bbox)
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": int(r1 - r0 + 1), "w": int(c1 - c0 + 1)}


def _strict_interior_rel(rel_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {rc for rc in rel_set if all((rc[0] + dr, rc[1] + dc) in rel_set for dr, dc in DIR4)}


def _connected(cells: set[tuple[int, int]]) -> bool:
    if len(cells) <= 1:
        return True
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (int(r + dr), int(c + dc))
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _shift_rel(rel_cells, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in rel_cells}


def _piece_from_component(grid: np.ndarray, coords: np.ndarray) -> dict:
    bbox = tuple(map(int, (coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max())))
    shape = _shape_from_coords(coords, bbox)
    strict_interior = _strict_interior_rel(shape["rel_set"])
    r0, c0, _r1, _c1 = bbox

    colors_rel: dict[tuple[int, int], int] = {}
    colors_to_rel: dict[int, set[tuple[int, int]]] = {}
    counts = Counter()
    for r, c in coords:
        rr, cc = int(r - r0), int(c - c0)
        col = int(grid[int(r), int(c)])
        colors_rel[(rr, cc)] = col
        colors_to_rel.setdefault(col, set()).add((rr, cc))
        counts[col] += 1

    conn_rel: set[tuple[int, int]] = set()
    conn_color = None
    conn_norm = None
    conn_anchor = None
    boundary_touch = lambda cell_set: any(
        rr == 0 or rr == int(shape["h"]) - 1 or cc == 0 or cc == int(shape["w"]) - 1 for rr, cc in cell_set
    )
    candidates = []
    for col, cell_set in colors_to_rel.items():
        if len(cell_set) not in (2, 3):
            continue
        if cell_set <= strict_interior:
            continue
        if not boundary_touch(cell_set):
            continue
        if not _connected(cell_set):
            continue
        norm, anchor = _normalize(cell_set)
        edge_counts = (
            sum(rr == 0 for rr, _ in cell_set),
            sum(rr == int(shape["h"]) - 1 for rr, _ in cell_set),
            sum(cc == 0 for _, cc in cell_set),
            sum(cc == int(shape["w"]) - 1 for _, cc in cell_set),
        )
        candidates.append((max(edge_counts), -len(cell_set), col, cell_set, norm, anchor))
    if candidates:
        candidates.sort(reverse=True)
        _score, _neg_n, col, cell_set, norm, anchor = candidates[0]
        conn_rel = set(cell_set)
        conn_color = int(col)
        conn_norm = norm
        conn_anchor = anchor

    boundary_non_conn = [rc for rc in shape["rel_set"] if rc not in strict_interior and rc not in conn_rel]
    boundary_counts = Counter(colors_rel[rc] for rc in boundary_non_conn)
    if boundary_counts:
        outline_color = int(boundary_counts.most_common(1)[0][0])
    else:
        outline_color = int(next(col for col, _n in counts.most_common() if col != conn_color))

    fill_rel = {rc for rc in strict_interior if rc not in conn_rel and colors_rel[rc] != outline_color}
    fill_color = None
    if fill_rel:
        fill_colors = {colors_rel[rc] for rc in fill_rel}
        if len(fill_colors) == 1:
            fill_color = int(next(iter(fill_colors)))
    outline_rel = {rc for rc in shape["rel_set"] if rc not in conn_rel and rc not in fill_rel}

    return {
        "shape": shape,
        "colors_rel": colors_rel,
        "outline_rel": outline_rel,
        "outline_color": outline_color,
        "fill_rel": fill_rel,
        "fill_color": fill_color,
        "strict_interior": strict_interior,
        "conn_rel": conn_rel,
        "conn_color": conn_color,
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
    if top < 0 or left < 0 or top + int(shape["h"]) > H or left + int(shape["w"]) > W:
        return None
    if strip is not None:
        axis, A, B = strip
        if axis == "cols":
            if left != int(A) or int(shape["w"]) != int(B - A + 1):
                return None
        else:
            if top != int(A) or int(shape["h"]) != int(B - A + 1):
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
            for rr, cc in shape["rel"]:
                if c_cov == left + int(cc):
                    tops.add(int(r_cov - rr))
        for top in tops:
            yield int(top), int(left)
    else:
        top = int(A)
        lefts = set()
        for r_cov, c_cov in cover:
            for rr, cc in shape["rel"]:
                if r_cov == top + int(rr):
                    lefts.add(int(c_cov - cc))
        for left in lefts:
            yield int(top), int(left)


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIR4:
            if (int(r + dr), int(c + dc)) in b:
                return True
    return False


def _hole_positions(hole: set[tuple[int, int]], shape: dict):
    pos = set()
    for r_h, c_h in hole:
        for rr, cc in shape["rel"]:
            pos.add((int(r_h - rr), int(c_h - cc)))
    return pos


def _paint_piece(out: np.ndarray, piece: dict, top: int, left: int) -> None:
    for (rr, cc), col in piece["colors_rel"].items():
        out[int(top + rr), int(left + cc)] = int(col)


def _clear_piece(out: np.ndarray, piece: dict, top: int, left: int, fixed_colors: dict, bg: int) -> None:
    for rr, cc in piece["shape"]["rel"]:
        rc = (int(top + rr), int(left + cc))
        out[rc] = int(fixed_colors.get(rc, bg))


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


def _socket_window(side: str, start: int, width: int, bar_bounds: tuple[int, int, int, int]):
    bar_r0, bar_c0, bar_r1, bar_c1 = bar_bounds
    if side == "top":
        return (
            {(int(bar_r0 + r), int(start + c)) for r in range(2) for c in range(width)},
            lambda rr, cc: (int(bar_r0 + rr), int(start + cc)),
        )
    if side == "bottom":
        return (
            {(int(bar_r1 - 1 + r), int(start + c)) for r in range(2) for c in range(width)},
            lambda rr, cc: (int(bar_r1 - 1 + rr), int(start + cc)),
        )
    if side == "left":
        return (
            {(int(start + r), int(bar_c0 + c)) for r in range(width) for c in range(2)},
            lambda rr, cc: (int(start + rr), int(bar_c0 + cc)),
        )
    if side == "right":
        return (
            {(int(start + r), int(bar_c1 - 1 + c)) for r in range(width) for c in range(2)},
            lambda rr, cc: (int(start + rr), int(bar_c1 - 1 + cc)),
        )
    raise ValueError(side)


def _parse_bar_and_notches(g: np.ndarray, bg: int):
    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    mono.sort(key=lambda d: -d["area"])
    assert mono
    bar = mono[0]

    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = int(bar_r1 - bar_r0 + 1), int(bar_c1 - bar_c0 + 1)
    bar_color = int(bar["color"])
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[int(r - bar_r0), int(c - bar_c0)] = True

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

        region = {(int(bar_r0 + r), int(bar_c0 + c)) for r, c in pts}
        hole_bg = {rc for rc in region if int(g[rc]) == int(bg)}
        marker_colors = {rc: int(g[rc]) for rc in region if int(g[rc]) not in (int(bg), int(bar_color))}
        conn_cells = set(marker_colors)
        conn_color = None
        conn_norm = None
        conn_anchor = None
        if conn_cells:
            colors = {marker_colors[rc] for rc in conn_cells}
            assert len(colors) == 1
            conn_color = int(next(iter(colors)))
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
                "conn_color": conn_color,
                "conn_norm": conn_norm,
                "conn_anchor": conn_anchor,
                "mouth_n": int(mouth_n),
            }
        )

    fixed = np.full(g.shape, int(bg), dtype=int)
    fixed_colors = {}
    for r, c in bar_occ:
        fixed[r, c] = int(bar_color)
        fixed_colors[(r, c)] = int(bar_color)
    for n in notches:
        for rc, col in n["marker_colors"].items():
            fixed[rc] = int(col)
            fixed_colors[rc] = int(col)

    return {
        "bar_occ": bar_occ,
        "bar_color": int(bar_color),
        "bar_bounds": (int(bar_r0), int(bar_c0), int(bar_r1), int(bar_c1)),
        "notches": notches,
        "fixed": fixed,
        "fixed_colors": fixed_colors,
    }


def _extract_pieces(g: np.ndarray, bg: int, bar_occ: set[tuple[int, int]]) -> list[dict]:
    work = np.array(g, copy=True)
    for r, c in bar_occ:
        work[r, c] = int(bg)
    comps = find_components(work, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    pieces = []
    for comp in comps:
        cells = {(int(r), int(c)) for r, c in comp["coords"]}
        if len(cells) < 8:
            continue
        pieces.append(_piece_from_component(g, comp["coords"]))
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    parsed = _parse_bar_and_notches(g, bg)
    bar_occ = parsed["bar_occ"]
    notches = parsed["notches"]
    fixed = parsed["fixed"]
    fixed_colors = parsed["fixed_colors"]
    pieces = _extract_pieces(g, bg, bar_occ)

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
                if not piece["fill_rel"]:
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


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=6):
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
                    nb = (int(r + dr), int(c + dc))
                    if nb in available and nb not in cells:
                        frontier.append(nb)
            if not frontier:
                break
            if must_span_axis and span_range:
                A, B = span_range
                if must_span_axis == "cols":
                    cur0 = min(c for _, c in cells)
                    cur1 = max(c for _, c in cells)
                    if cur0 > int(A):
                        pref = [nb for nb in frontier if nb[1] < cur0]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                    if cur1 < int(B):
                        pref = [nb for nb in frontier if nb[1] > cur1]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                else:
                    cur0 = min(r for r, _ in cells)
                    cur1 = max(r for r, _ in cells)
                    if cur0 > int(A):
                        pref = [nb for nb in frontier if nb[0] < cur0]
                        if pref:
                            cells.add(rng.choice(pref))
                            continue
                    if cur1 < int(B):
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
                if min(c for _, c in cells) != int(A) or max(c for _, c in cells) != int(B):
                    continue
            else:
                if min(r for r, _ in cells) != int(A) or max(r for r, _ in cells) != int(B):
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
    if len(chain_lens) == 2:
        side_orders = [list(sides), list(reversed(sides))]
        rng.shuffle(side_orders)
        for side_order in side_orders:
            tentative = []
            ok = True
            for chain_len, side in zip(chain_lens, side_order):
                if room[side] < chain_len * 2 + 3 or len(slots[side]) < 3:
                    ok = False
                    break
                start = rng.choice(slots[side])
                tentative.append({"side": side, "start": start})
            if not ok:
                continue
            for item in tentative:
                slots[item["side"]].remove(item["start"])
            used = tentative
            break
        if not used:
            return None
        return used, slots

    for chain_len in chain_lens:
        valid_sides = [side for side in sides if room[side] >= chain_len * 2 + 3 and len(slots[side]) >= 3]
        if not valid_sides:
            return None
        side = rng.choice(valid_sides)
        start = rng.choice(slots[side])
        slots[side].remove(start)
        used.append({"side": side, "start": start})
    return used, slots


def _strict_interior_global(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {rc for rc in cells if all((rc[0] + dr, rc[1] + dc) in cells for dr, dc in DIR4)}


def _pick_body_pair(
    rng,
    outline_pool: list[int],
    fill_pool: list[int],
    piece_idx: int,
    chain_idx: int,
    forbidden: set[int],
) -> tuple[int, int] | None:
    options = []
    for oi, outline in enumerate(outline_pool):
        for fi, fill in enumerate(fill_pool):
            if outline == fill or outline in forbidden or fill in forbidden:
                continue
            score = abs((piece_idx + chain_idx) - (oi * 3 + fi))
            options.append((score, oi, fi, int(outline), int(fill)))
    if not options:
        return None
    options.sort()
    best_score = options[0][0]
    tied = [item for item in options if item[0] <= best_score + 1]
    _score, _oi, _fi, outline, fill = rng.choice(tied)
    return int(outline), int(fill)


def _colorize_piece(
    piece_cells: set[tuple[int, int]],
    connector_cells: set[tuple[int, int]],
    outline_color: int,
    fill_color: int,
    connector_color: int | None,
) -> dict[tuple[int, int], int] | None:
    strict = _strict_interior_global(piece_cells)
    fill_cells = strict - connector_cells
    if not fill_cells:
        return None
    colors = {}
    for rc in piece_cells:
        if rc in connector_cells:
            if connector_color is None:
                return None
            colors[rc] = int(connector_color)
        elif rc in fill_cells:
            colors[rc] = int(fill_color)
        else:
            colors[rc] = int(outline_color)
    return colors


def _piece_grid(piece: dict) -> tuple[dict[tuple[int, int], int], int, int]:
    rs = [r for r, _ in piece["colors"]]
    cs = [c for _, c in piece["colors"]]
    r0, c0 = min(rs), min(cs)
    return (
        {(int(r - r0), int(c - c0)): int(col) for (r, c), col in piece["colors"].items()},
        int(max(rs) - r0 + 1),
        int(max(cs) - c0 + 1),
    )


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
        pts = {(int(top + r), int(left + c)) for r, c in rel_colors}
        if pts & placed_occ:
            continue
        if any((int(r + dr), int(c + dc)) in placed_occ for r, c in pts for dr, dc in DIR4):
            continue
        for (rr, cc), col in rel_colors.items():
            input_grid[int(top + rr), int(left + cc)] = int(col)
        placed_occ |= pts
        return True
    return False


def _fusion_signature(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)
    parsed = _parse_bar_and_notches(g, bg)
    pieces = _extract_pieces(g, bg, parsed["bar_occ"])
    occ_fixed = set(parsed["fixed_colors"])

    extra_geometric_matches = 0
    exact_root_matches = 0
    geometric_root_matches = 0
    pieces_with_color_only_decoy = 0
    pieces_with_shape_only_decoy = 0

    for p in pieces:
        if not p["conn_rel"]:
            continue
        geom_for_piece = 0
        exact_for_piece = 0
        has_color_only_decoy = False
        has_shape_only_decoy = False
        for n in parsed["notches"]:
            found_geom = False
            for top, left in _hole_positions(n["region"], p["shape"]):
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
                if pts is not None:
                    found_geom = True
                    break
            if found_geom:
                geom_for_piece += 1
                if n["conn_cells"] and n["conn_color"] == p["conn_color"] and n["conn_norm"] != p["conn_norm"]:
                    has_color_only_decoy = True
                if n["conn_cells"] and n["conn_norm"] == p["conn_norm"] and n["conn_color"] != p["conn_color"]:
                    has_shape_only_decoy = True
            if n["conn_cells"] and p["conn_color"] == n["conn_color"] and p["conn_norm"] == n["conn_norm"]:
                top = int(n["conn_anchor"][0] - p["conn_anchor"][0])
                left = int(n["conn_anchor"][1] - p["conn_anchor"][1])
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
                if pts is not None and _shift_rel(p["conn_rel"], top, left) == n["conn_cells"]:
                    exact_for_piece += 1
        geometric_root_matches += geom_for_piece
        exact_root_matches += exact_for_piece
        extra_geometric_matches += max(0, geom_for_piece - exact_for_piece)
        pieces_with_color_only_decoy += int(has_color_only_decoy)
        pieces_with_shape_only_decoy += int(has_shape_only_decoy)

    return {
        "pieces": int(len(pieces)),
        "pieces_with_fill": int(sum(1 for p in pieces if p["fill_rel"])),
        "pieces_with_connectors": int(sum(1 for p in pieces if p["conn_rel"])),
        "geometric_root_matches": int(geometric_root_matches),
        "exact_root_matches": int(exact_root_matches),
        "extra_geometric_matches": int(extra_geometric_matches),
        "pieces_with_color_only_decoy": int(pieces_with_color_only_decoy),
        "pieces_with_shape_only_decoy": int(pieces_with_shape_only_decoy),
    }


def _gen_attempt():
    rng = _random.Random()
    H = W = 30
    bg = 0

    orientation, _thickness, bar_color, bar_bounds, room = _bar_setup(rng, H, W)
    bar_r0, bar_c0, bar_r1, bar_c1 = bar_bounds
    bar_cells = {(int(r), int(c)) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    num_chains = 2
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
    if len(palette) < num_chains + 6:
        return None
    connector_colors = [int(c) for c in palette[:num_chains]]
    outline_pool = [int(c) for c in palette[num_chains : num_chains + 3]]
    fill_pool = [int(c) for c in palette[num_chains + 3 : num_chains + 6]]
    if len(outline_pool) < 3 or len(fill_pool) < 3:
        return None

    proto_ids = list(range(len(CANONICAL_CONNECTORS)))
    proto_pair = rng.sample(proto_ids, num_chains)
    used_specs = [{"color": int(connector_colors[i]), "proto": int(proto_pair[i])} for i in range(num_chains)]
    seen_pairs = {(spec["color"], spec["proto"]) for spec in used_specs}

    reserved_decoys = []
    for idx, win in enumerate(used_windows):
        side = win["side"]
        if len(free_slots[side]) < 2:
            return None
        starts = rng.sample(free_slots[side], 2)
        for start in starts:
            free_slots[side].remove(start)
        reserved_decoys.append({"side": side, "start": starts[0], "ref_chain": idx, "mode": "color"})
        reserved_decoys.append({"side": side, "start": starts[1], "ref_chain": idx, "mode": "shape"})

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

    def make_decoy_spec(ref_chain, mode=None):
        if ref_chain is not None:
            base = used_specs[ref_chain]
            if mode == "color":
                alt = [pid for pid in proto_ids if pid != base["proto"]]
                return {"color": int(base["color"]), "proto": int(rng.choice(alt))}
            other_colors = [spec["color"] for i, spec in enumerate(used_specs) if i != ref_chain and spec["color"] != base["color"]]
            if other_colors:
                return {"color": int(rng.choice(other_colors)), "proto": int(base["proto"])}
            alt = [pid for pid in proto_ids if pid != base["proto"]]
            return {"color": int(base["color"]), "proto": int(rng.choice(alt))}
        base = rng.choice(used_specs)
        if rng.random() < 0.6:
            alt = [pid for pid in proto_ids if pid != base["proto"]]
            return {"color": int(base["color"]), "proto": int(rng.choice(alt))}
        other = rng.choice(used_specs)
        pair = {"color": int(other["color"]), "proto": int(base["proto"])}
        if (pair["color"], pair["proto"]) not in seen_pairs:
            return pair
        alt = [pid for pid in proto_ids if pid != base["proto"]]
        return {"color": int(base["color"]), "proto": int(rng.choice(alt))}

    notches = []
    for idx, win in enumerate(used_windows):
        notches.append({**win, "used": True, "spec": used_specs[idx], "chain_idx": idx})
    for win in decoy_windows:
        notches.append(
            {**win, "used": False, "spec": make_decoy_spec(win["ref_chain"], win.get("mode")), "chain_idx": None}
        )

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
    piece_serial = 0

    for chain_idx in range(num_chains):
        n = next(x for x in notch_regions if x["used"] and x["chain_idx"] == chain_idx)
        A = int(n["start"])
        B = int(n["start"] + strip_width - 1)
        side = n["side"]
        chain_pieces = []
        for piece_idx in range(chain_lens[chain_idx]):
            pair = _pick_body_pair(
                rng,
                outline_pool,
                fill_pool,
                piece_serial,
                chain_idx,
                {int(n["spec"]["color"])} if piece_idx == 0 else set(),
            )
            if pair is None:
                return None
            outline_color, fill_color = pair
            piece_serial += 1
            chain_dim = rng.randint(2, 3)

            if piece_idx == 0:
                socket = set(n["socket_cells"])
                if orientation == "horizontal":
                    if side == "top":
                        ext_rows = range(max(0, bar_r0 - chain_dim), bar_r0)
                    else:
                        ext_rows = range(bar_r1 + 1, min(H, bar_r1 + 1 + chain_dim))
                    ext = {(int(r), int(c)) for r in ext_rows for c in range(A, B + 1) if (int(r), int(c)) not in occ_set}
                    span_axis = "cols"
                else:
                    if side == "left":
                        ext_cols = range(max(0, bar_c0 - chain_dim), bar_c0)
                    else:
                        ext_cols = range(bar_c1 + 1, min(W, bar_c1 + 1 + chain_dim))
                    ext = {(int(r), int(c)) for r in range(A, B + 1) for c in ext_cols if (int(r), int(c)) not in occ_set}
                    span_axis = "rows"
                all_avail = socket | ext
                if len(ext) < 2:
                    return None
                low = max(len(socket) + 2, 10)
                high = min(13, len(all_avail))
                if low > high:
                    return None
                target = rng.randint(low, high)
                piece_cells = _grow_connected(rng, all_avail, target, socket, must_span_axis=span_axis, span_range=(A, B))
                if piece_cells is None or len(piece_cells - socket) < 2:
                    return None
                colors = _colorize_piece(piece_cells, set(n["conn_cells"]), outline_color, fill_color, int(n["spec"]["color"]))
            else:
                prev = chain_pieces[-1]["coords"]
                pr0, pc0, pr1, pc1 = _bbox_of(prev)
                if orientation == "horizontal":
                    cavity = {
                        (int(r), int(c))
                        for r in range(pr0, pr1 + 1)
                        for c in range(A, B + 1)
                        if (int(r), int(c)) not in occ_set and (int(r), int(c)) not in prev
                    }
                    if side == "top":
                        ext_rows = range(max(0, pr0 - chain_dim), pr0)
                    else:
                        ext_rows = range(pr1 + 1, min(H, pr1 + 1 + chain_dim))
                    ext = {(int(r), int(c)) for r in ext_rows for c in range(A, B + 1) if (int(r), int(c)) not in occ_set}
                    span_axis = "cols"
                else:
                    cavity = {
                        (int(r), int(c))
                        for r in range(A, B + 1)
                        for c in range(pc0, pc1 + 1)
                        if (int(r), int(c)) not in occ_set and (int(r), int(c)) not in prev
                    }
                    if side == "left":
                        ext_cols = range(max(0, pc0 - chain_dim), pc0)
                    else:
                        ext_cols = range(pc1 + 1, min(W, pc1 + 1 + chain_dim))
                    ext = {(int(r), int(c)) for r in range(A, B + 1) for c in ext_cols if (int(r), int(c)) not in occ_set}
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
                piece_cells = _grow_connected(rng, all_avail, target, cavity, must_span_axis=span_axis, span_range=(A, B))
                if piece_cells is None or not cavity.issubset(piece_cells):
                    return None
                if piece_idx == chain_lens[chain_idx] - 1:
                    t_r0, t_c0, t_r1, t_c1 = _bbox_of(piece_cells)
                    if orientation == "horizontal":
                        remaining = {
                            (int(r), int(c))
                            for r in range(t_r0, t_r1 + 1)
                            for c in range(A, B + 1)
                            if (int(r), int(c)) not in occ_set and (int(r), int(c)) not in piece_cells
                        }
                    else:
                        remaining = {
                            (int(r), int(c))
                            for r in range(A, B + 1)
                            for c in range(t_c0, t_c1 + 1)
                            if (int(r), int(c)) not in occ_set and (int(r), int(c)) not in piece_cells
                        }
                    if remaining:
                        return None
                colors = _colorize_piece(piece_cells, set(), outline_color, fill_color, None)

            if colors is None:
                return None
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


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        sig = _fusion_signature(inp)
        if sig["pieces"] < 4:
            continue
        if sig["pieces_with_fill"] != sig["pieces"]:
            continue
        if sig["pieces_with_connectors"] != 2:
            continue
        if sig["exact_root_matches"] != sig["pieces_with_connectors"]:
            continue
        if sig["extra_geometric_matches"] < sig["pieces_with_connectors"]:
            continue
        if sig["pieces_with_color_only_decoy"] != sig["pieces_with_connectors"]:
            continue
        if sig["pieces_with_shape_only_decoy"] != sig["pieces_with_connectors"]:
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    raise RuntimeError("generate_input: failed after 800 attempts")
