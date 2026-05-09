# Combo variant of 16b78196 (index 20)
# combo_spec: D1/3+D2/0+D5/2
# Name: 连接器匹配+多色精灵+渐变着色
#
# Design analysis
# ---------------
# a) Concrete behavior trace
# - Base geometry stays close to 16b78196: a central full-span bar has several boundary sockets, 2 true
#   chains grow outward along fixed-width strips, and later pieces fill the previous piece's cavity.
# - D1/3 changes root matching: sockets carry tiny connector glyphs (2-3 cells) and only a root whose
#   connector color+shape matches that socket may dock there. Geometry alone is intentionally ambiguous.
# - D2/0 changes piece representation: every movable piece is a multicolor 4-connected sprite with an
#   outline layer and an interior fill layer. Roots add a third tiny connector layer on the bar-facing side.
# - D5/2 changes output coloring: a visible domino legend defines a color order. When a piece is assembled
#   at chain depth d, only its interior fill color shifts forward by d slots in that legend; outline and
#   connector colors stay unchanged. Example: an input fill colored with legend slot 1 stays slot 1 for a
#   root, becomes slot 2 for the next piece, and slot 3 for a tail piece.
# - So the fused reasoning path is:
#   1. recognize the bar, sockets, legend, and scattered multicolor sprites;
#   2. parse each root candidate into silhouette + connector marker;
#   3. dock roots by connector color+shape equality, not by notch geometry alone;
#   4. continue each chain by cavity-filling geometry;
#   5. recolor only the fill cells according to visible chain depth.
#
# b) Complexity calibration
# - Connector shapes are restricted to 3 tiny prototypes and the legend length is fixed at 5. This keeps the
#   attribute spaces small enough to learn from 2-4 examples, while still forcing conjunction over shape and color.
# - Each sample uses 1-2 chains with lengths in {2,3}, totaling 3-5 pieces. Depth-1 always appears and
#   depth-2 appears frequently, which is enough to expose the fill-gradient rule while keeping generation stable.
# - Strip width is fixed at 4 in this fused variant. That reduces accidental thin pieces, keeps visible
#   interiors for the D2 layer, and still preserves the original strip-locking insight.
#
# c) Implicit assumption audit
# - Original code assumed every piece was monochrome. Here that breaks immediately: parsing must use
#   non-monochromatic components, preserve per-cell colors, and separately identify connector cells.
# - Original root search assumed a notch was determined only by the occupied silhouette. Here translation is
#   fixed by connector alignment first, then validated against the whole socket region.
# - Original code wrote piece.color directly. Here only the fill layer is transformed; outline and connector
#   must be restored exactly during backtracking.
# - Original fixed foreground was just the bar. Here it is bar + socket markers + legend. All placement and
#   undo logic must preserve those cells.
# - Original tiny components could be ignored. Here tiny area-2 legend dominoes are semantically essential,
#   while tiny socket markers must not be mistaken for movable objects. The solver therefore filters movable
#   components by area 8-13 and detects the legend from aligned area-2 monochrome components outside the bar.
#
# d) Placement legality
# - The generator is output-first: build bar, legend, sockets, connector markers, then synthesize the full
#   assembled chains in place, then scatter the movable pieces to make the input.
# - Every placement checks bounds and occupied cells. Legend spans are reserved so chains on the same side
#   cannot grow into the legend, and scattered pieces also avoid a buffer around the legend.
# - Tail pieces are validated to leave no cavity in their own bbox, preserving the original termination rule.
#
# e) Visual observability
# - The legend is the only source of color order. No numeric ARC color id ordering is used anywhere.
# - Used and decoy sockets share strip geometry; at least one decoy per used side shares connector color or
#   connector shape but not both. That makes the matching layer visibly necessary.
# - Multicolor pieces expose a stable outline while only the interior changes in the output, making the
#   transform layer easy to isolate from the matching and shape layers.
#
# f) Visual cleanliness
# - The legend uses tiny dominoes in a corner margin; movable pieces remain area 8-13, so guides do not
#   visually compete with sprites.
# - Connector markers stay inside 2-deep sockets and root connectors overlap exactly on docking, so the bar
#   region remains readable instead of accumulating extra clutter.
# - Outline/fill separation is geometric: fill cells are strict interior cells, outline cells are the rest.
#
# g) Change inferability
# - Fill colors are always chosen from the visible legend and are shifted only by depth. Roots show shift 0,
#   both chains show shift 1, and many samples show shift 2. This repeated structure makes the color rule
#   inferable from a few examples.
# - Outline colors are not transformed, so humans can track piece identity through movement while still
#   noticing that the interior layer follows the legend progression.

from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
from itertools import combinations
from pathlib import Path
import random as _random

import numpy as np

from arc_utils import components_mask, find_components, get_bg


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PIECE_MIN_AREA = 8
PIECE_MAX_AREA = 13
LEGEND_LEN = 5
STRIP_WIDTH = 4
CANONICAL_CONNECTORS = [
    {(0, 1), (1, 1)},
    {(0, 1), (0, 2)},
    {(0, 1), (1, 1), (1, 2)},
]


def _normalize(cells) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    cells = list(cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((int(r - r0), int(c - c0)) for r, c in cells), (int(r0), int(c0))


def _all_connector_shapes() -> set[frozenset[tuple[int, int]]]:
    out = set()
    for side in ("top", "bottom", "left", "right"):
        for proto in range(len(CANONICAL_CONNECTORS)):
            out.add(_normalize(_socket_connector_local(side, proto))[0])
    return out


def _bbox_of(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _shape_from_coords(coords: np.ndarray, bbox) -> dict[str, object]:
    r0, c0, r1, c1 = map(int, bbox)
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {
        "rel": rel,
        "rel_set": set(rel),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
    }


def _pixels(shape: dict[str, object], top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in shape["rel"]}


def _strict_interior(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {rc for rc in cells if all((rc[0] + dr, rc[1] + dc) in cells for dr, dc in DIR4)}


def _inner_fill_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if not cells:
        return set()
    r0, c0, r1, c1 = _bbox_of(cells)
    fill = {rc for rc in cells if r0 < rc[0] < r1 and c0 < rc[1] < c1}
    return fill if fill else _strict_interior(cells)


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


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=5):
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
                for dr, dc in DIR4
                if (nb := (r + dr, c + dc)) in available and nb not in cells
            ]
            if not frontier:
                break
            if must_span_axis is not None and span_range is not None:
                A, B = span_range
                if must_span_axis == "cols":
                    cur0 = min(c for _, c in cells)
                    cur1 = max(c for _, c in cells)
                    if cur0 > A:
                        cand = [nb for nb in frontier if nb[1] < cur0]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if cur1 < B:
                        cand = [nb for nb in frontier if nb[1] > cur1]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                else:
                    cur0 = min(r for r, _ in cells)
                    cur1 = max(r for r, _ in cells)
                    if cur0 > A:
                        cand = [nb for nb in frontier if nb[0] < cur0]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if cur1 < B:
                        cand = [nb for nb in frontier if nb[0] > cur1]
                        if cand:
                            cells.add(rng.choice(cand))
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
            {(bar_r0 + r, start + c) for r in range(2) for c in range(width)},
            lambda rr, cc: (int(bar_r0 + rr), int(start + cc)),
        )
    if side == "bottom":
        return (
            {(bar_r1 - 1 + r, start + c) for r in range(2) for c in range(width)},
            lambda rr, cc: (int(bar_r1 - 1 + rr), int(start + cc)),
        )
    if side == "left":
        return (
            {(start + r, bar_c0 + c) for r in range(width) for c in range(2)},
            lambda rr, cc: (int(start + rr), int(bar_c0 + cc)),
        )
    if side == "right":
        return (
            {(start + r, bar_c1 - 1 + c) for r in range(width) for c in range(2)},
            lambda rr, cc: (int(start + rr), int(bar_c1 - 1 + cc)),
        )
    raise ValueError(side)


ALL_CONNECTOR_SHAPES = _all_connector_shapes()
_ROOT = Path(__file__).resolve().parents[4]


def _load_base_variant():
    spec = importlib.util.spec_from_file_location("base_d13_16b78196", _ROOT / "ceshi" / "code" / "16b78196" / "D1" / "3" / "code.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_BASE_VARIANT = _load_base_variant()


def _legend_layout(H: int, W: int, orientation: str):
    if orientation == "horizontal":
        row = 1
        dominoes = []
        for i in range(LEGEND_LEN):
            c0 = 1 + 3 * i
            dominoes.append({(row, c0), (row, c0 + 1)})
        r0, c0, r1, c1 = 0, 0, row + 1, 1 + 3 * (LEGEND_LEN - 1) + 2
        side = "top"
        span = (1, 1 + 3 * (LEGEND_LEN - 1) + 1)
    else:
        col = 1
        dominoes = []
        for i in range(LEGEND_LEN):
            r0d = 1 + 3 * i
            dominoes.append({(r0d, col), (r0d + 1, col)})
        r0, c0, r1, c1 = 0, 0, 1 + 3 * (LEGEND_LEN - 1) + 2, col + 1
        side = "left"
        span = (1, 1 + 3 * (LEGEND_LEN - 1) + 1)
    occ = set().union(*dominoes)
    buffer = {
        (r, c)
        for r in range(max(0, r0), min(H - 1, r1) + 1)
        for c in range(max(0, c0), min(W - 1, c1) + 1)
    }
    return {"dominoes": dominoes, "occ": occ, "buffer": buffer, "side": side, "span": span}


def _legend_components(comps: list[dict], bar_bbox: tuple[int, int, int, int]):
    bar_r0, bar_c0, bar_r1, bar_c1 = bar_bbox
    horiz = []
    vert = []
    for comp in comps:
        if int(comp["area"]) != 2:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        inside_bar = bar_r0 <= r0 <= bar_r1 and bar_r0 <= r1 <= bar_r1 and bar_c0 <= c0 <= bar_c1 and bar_c0 <= c1 <= bar_c1
        if inside_bar:
            continue
        coords = {(int(r), int(c)) for r, c in comp["coords"]}
        norm, _ = _normalize(coords)
        if norm == frozenset({(0, 0), (0, 1)}) and r0 <= 2 and r1 <= 2:
            horiz.append(comp)
        if norm == frozenset({(0, 0), (1, 0)}) and c0 <= 2 and c1 <= 2:
            vert.append(comp)
    guides = horiz if len(horiz) >= len(vert) else vert
    if len(guides) < 4:
        return [], []
    if guides is horiz:
        guides.sort(key=lambda d: (int(d["bbox"][1]), int(d["bbox"][0])))
    else:
        guides.sort(key=lambda d: (int(d["bbox"][0]), int(d["bbox"][1])))
    guides = guides[:LEGEND_LEN]
    palette = [int(g["color"]) for g in guides]
    return guides, palette


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _shift_rel(rel_cells, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in rel_cells}


def _piece_from_component(grid: np.ndarray, coords: np.ndarray) -> dict[str, object]:
    bbox = tuple(map(int, (coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max())))
    shape = _shape_from_coords(coords, bbox)
    rel_set = shape["rel_set"]
    r0, c0, _r1, _c1 = bbox
    colors_rel = {}
    by_color = defaultdict(set)
    for r, c in coords:
        rr, cc = int(r - r0), int(c - c0)
        col = int(grid[int(r), int(c)])
        colors_rel[(rr, cc)] = col
        by_color[col].add((rr, cc))

    fill_rel = set(_inner_fill_cells(set(rel_set)))
    conn_rel = set()
    conn_color = None
    conn_norm = None
    conn_anchor = None
    conn_candidates = []
    if len(by_color) >= 3:
        for color, cells in by_color.items():
            if len(cells) not in (2, 3):
                continue
            norm, anchor = _normalize(cells)
            if norm in ALL_CONNECTOR_SHAPES:
                conn_candidates.append((color, set(cells), norm, anchor))
    if len(conn_candidates) == 1:
        conn_color, conn_rel, conn_norm, conn_anchor = conn_candidates[0]
        fill_rel -= set(conn_rel)

    fill_colors = {colors_rel[rc] for rc in fill_rel}
    fill_color = next(iter(fill_colors)) if len(fill_colors) == 1 else None
    return {
        "shape": shape,
        "colors_rel": colors_rel,
        "fill_rel": set(fill_rel),
        "fill_color": None if fill_color is None else int(fill_color),
        "conn_rel": set(conn_rel),
        "conn_color": None if conn_color is None else int(conn_color),
        "conn_norm": conn_norm,
        "conn_anchor": conn_anchor,
        "orig": (int(r0), int(c0)),
    }


def _make_piece_dict(grid: np.ndarray, cells: set[tuple[int, int]], body_color: int, conn_cells: set[tuple[int, int]] | None = None, conn_color: int | None = None):
    coords = np.array(sorted(cells), dtype=int)
    bbox = tuple(map(int, (coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max())))
    shape = _shape_from_coords(coords, bbox)
    r0, c0, _r1, _c1 = bbox
    colors_rel = {(int(r - r0), int(c - c0)): int(grid[int(r), int(c)]) for r, c in coords}
    conn_rel = set()
    conn_norm = None
    conn_anchor = None
    if conn_cells:
        conn_rel = {(int(r - r0), int(c - c0)) for r, c in conn_cells}
        conn_norm, conn_anchor = _normalize(conn_rel)
    return {
        "shape": shape,
        "colors_rel": colors_rel,
        "fill_rel": set(),
        "fill_color": int(body_color),
        "conn_rel": conn_rel,
        "conn_color": None if conn_color is None else int(conn_color),
        "conn_norm": conn_norm,
        "conn_anchor": conn_anchor,
        "orig": (int(r0), int(c0)),
        "body_color": int(body_color),
    }


def _placement_pts(
    shape: dict[str, object],
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
            if left != A or int(shape["w"]) != B - A + 1:
                return None
        else:
            if top != A or int(shape["h"]) != B - A + 1:
                return None
    pts = _pixels(shape, top, left)
    allowed_overlap = allowed_overlap or set()
    if (pts & occ) - allowed_overlap:
        return None
    if must_cover is not None and not must_cover.issubset(pts):
        return None
    return pts


def _aligned_positions(axis: str, A: int, cover: set[tuple[int, int]], shape: dict[str, object]):
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


def _paint_piece(
    out: np.ndarray,
    piece: dict[str, object],
    top: int,
    left: int,
    depth: int,
    palette: list[int],
    palette_index: dict[int, int],
) -> None:
    fill_color = piece["fill_color"]
    shifted_fill = None
    if fill_color is not None:
        shifted_fill = int(palette[int(palette_index[int(fill_color)]) + depth])
    for (rr, cc), col in piece["colors_rel"].items():
        val = shifted_fill if shifted_fill is not None and (rr, cc) in piece["fill_rel"] else int(col)
        out[int(top + rr), int(left + cc)] = int(val)


def _clear_piece(out: np.ndarray, piece: dict[str, object], top: int, left: int, fixed_colors: dict[tuple[int, int], int], bg: int) -> None:
    for rr, cc in piece["shape"]["rel"]:
        rc = (int(top + rr), int(left + cc))
        out[rc] = int(fixed_colors.get(rc, bg))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    mono.sort(key=lambda d: -int(d["area"]))
    assert mono
    bar = mono[0]

    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_bbox = (bar_r0, bar_c0, bar_r1, bar_c1)
    bar_color = int(bar["color"])
    bar_h, bar_w = int(bar_r1 - bar_r0 + 1), int(bar_c1 - bar_c0 + 1)

    guides, palette = _legend_components(mono[1:], bar_bbox)
    assert len(palette) >= 4
    palette_index = {int(color): idx for idx, color in enumerate(palette)}
    guide_occ = {(int(r), int(c)) for guide in guides for r, c in guide["coords"]}

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
            marker_colors = {rc: int(g[rc]) for rc in region if int(g[rc]) not in (int(bg), int(bar_color))}
            conn_cells = set(marker_colors)
            conn_color = None
            conn_norm = None
            conn_anchor = None
            if conn_cells:
                colors = {marker_colors[rc] for rc in conn_cells}
                if len(colors) == 1:
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
    fixed_colors: dict[tuple[int, int], int] = {}
    for r, c in bar_occ:
        fixed[r, c] = int(bar_color)
        fixed_colors[(r, c)] = int(bar_color)
    for guide in guides:
        for r, c in guide["coords"]:
            fixed[int(r), int(c)] = int(guide["color"])
            fixed_colors[(int(r), int(c))] = int(guide["color"])
    for notch in notches:
        for rc in notch["conn_cells"]:
            fixed[rc] = int(g[rc])
            fixed_colors[rc] = int(g[rc])

    work = g.copy()
    for r, c in bar_occ | guide_occ:
        work[r, c] = int(bg)
    other = find_components(work, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    pieces = []
    for comp in other:
        area = int(comp["area"])
        if area < PIECE_MIN_AREA or area > PIECE_MAX_AREA:
            continue
        piece = _piece_from_component(g, comp["coords"])
        if piece["fill_color"] is None or len(piece["fill_rel"]) == 0:
            continue
        pieces.append(piece)

    base_occ = set(fixed_colors)
    root_cands = []
    for ni, notch in enumerate(notches):
        if not notch["conn_cells"] or notch["conn_color"] is None or notch["conn_norm"] is None:
            continue
        for pi, piece in enumerate(pieces):
            if not piece["conn_rel"]:
                continue
            if piece["conn_color"] != notch["conn_color"] or piece["conn_norm"] != notch["conn_norm"]:
                continue
            if piece["fill_color"] not in palette_index:
                continue
            top = int(notch["conn_anchor"][0] - piece["conn_anchor"][0])
            left = int(notch["conn_anchor"][1] - piece["conn_anchor"][1])
            if _shift_rel(piece["conn_rel"], top, left) != notch["conn_cells"]:
                continue
            pts = _placement_pts(
                piece["shape"],
                top,
                left,
                base_occ,
                H,
                W,
                must_cover=notch["region"],
                allowed_overlap=notch["conn_cells"],
            )
            if pts is None:
                continue
            if notch["axis"] == "cols":
                A, B = int(left), int(left + int(piece["shape"]["w"]) - 1)
            else:
                A, B = int(top), int(top + int(piece["shape"]["h"]) - 1)
            mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
            root_cands.append(
                (
                    (int(mv), -int(notch["mouth_n"])),
                    {
                        "ni": ni,
                        "pi": pi,
                        "top": int(top),
                        "left": int(left),
                        "axis": notch["axis"],
                        "A": int(A),
                        "B": int(B),
                        "pts": pts,
                        "allow_overlap": set(notch["conn_cells"]),
                    },
                )
            )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [x[1] for x in root_cands[:20]]
    best_out = fixed.copy()
    best_key = None

    def search(out_grid: np.ndarray, occ_set: set[tuple[int, int]], chains: list[dict[str, object]], remaining: set[int]) -> bool:
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (placed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        def gen_extend_moves(chain: dict[str, object]):
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = _pixels(head["shape"], head["top"], head["left"])
            cavity = {
                (r, c)
                for r in range(int(head["top"]), int(head["top"] + int(head["shape"]["h"])))
                for c in range(int(head["left"]), int(head["left"] + int(head["shape"]["w"])))
                if int(out_grid[r, c]) == int(bg)
            }
            if not cavity:
                return []
            new_depth = int(chain["depth"]) + 1
            moves = []
            for pi in remaining:
                piece = pieces[pi]
                if piece["conn_rel"]:
                    continue
                fill_color = piece["fill_color"]
                if fill_color not in palette_index or int(palette_index[int(fill_color)]) + new_depth >= len(palette):
                    continue
                for top, left in _aligned_positions(axis, A, cavity, piece["shape"]):
                    pts = _placement_pts(piece["shape"], top, left, occ_set, H, W, must_cover=cavity, strip=(axis, A, B))
                    if pts is None or not _touches(pts, head_pts):
                        continue
                    mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                    moves.append((int(mv), pi, int(top), int(left), pts))
            moves.sort()
            return moves

        for chain in chains:
            moves = gen_extend_moves(chain)
            if not moves:
                continue
            for _score, pi, top, left, pts in moves[:8]:
                new_depth = int(chain["depth"]) + 1
                _paint_piece(out_grid, pieces[pi], top, left, new_depth, palette, palette_index)
                old_head = chain["head"]
                old_depth = chain["depth"]
                chain["head"] = {"top": int(top), "left": int(left), "shape": pieces[pi]["shape"]}
                chain["depth"] = int(new_depth)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ_set | pts, chains, rem2):
                    return True
                chain["head"] = old_head
                chain["depth"] = old_depth
                _clear_piece(out_grid, pieces[pi], top, left, fixed_colors, int(bg))
        return False

    max_roots = min(3, len(cand_roots))
    for k in range(1, max_roots + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(base_occ)
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
                _paint_piece(out0, pieces[root["pi"]], root["top"], root["left"], 0, palette, palette_index)
                occ0 |= root["pts"]
                chains.append(
                    {
                        "axis": root["axis"],
                        "A": root["A"],
                        "B": root["B"],
                        "depth": 0,
                        "head": {"top": root["top"], "left": root["left"], "shape": pieces[root["pi"]]["shape"]},
                    }
                )
            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


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
    return orientation, bar_color, (bar_r0, bar_c0, bar_r1, bar_c1), room


def _slot_starts(span_len: int, strip_width: int) -> list[int]:
    return list(range(1, span_len - strip_width + 1, strip_width + 1))


def _pick_used_windows(rng, orientation, room, chain_lens, strip_width, span_len, legend):
    sides = ["top", "bottom"] if orientation == "horizontal" else ["left", "right"]
    slots = {side: [] for side in sides}
    for side in sides:
        for start in _slot_starts(span_len, strip_width):
            end = start + strip_width - 1
            if side == legend["side"] and _intervals_overlap(start, end, legend["span"][0] - 1, legend["span"][1] + 1):
                continue
            slots[side].append(start)
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


def _assign_fill_colors(rng, chains, palette: list[int]):
    used_bases = set()
    for chain in chains:
        max_depth = max(int(piece["depth"]) for piece in chain)
        root_forbidden = set(chain[0].get("forbidden_fill_colors", set()))
        candidates = [idx for idx in range(len(palette) - max_depth) if int(palette[idx]) not in root_forbidden]
        if not candidates:
            return None
        fresh = [idx for idx in candidates if idx not in used_bases]
        base_idx = rng.choice(fresh or candidates)
        used_bases.add(int(base_idx))
        base_color = int(palette[base_idx])
        for piece in chain:
            piece["fill_color"] = int(base_color)
    return chains


def _rel_colors(piece: dict[str, object]) -> tuple[dict[tuple[int, int], int], int, int]:
    rs = [r for r, _ in piece["colors"]]
    cs = [c for _, c in piece["colors"]]
    r0, c0 = min(rs), min(cs)
    rel = {(int(r - r0), int(c - c0)): int(col) for (r, c), col in piece["colors"].items()}
    return rel, int(max(rs) - r0 + 1), int(max(cs) - c0 + 1)


def _scatter_piece(rng, piece: dict[str, object], input_grid: np.ndarray, placed_occ: set[tuple[int, int]], orientation: str, bar_bounds, legend_buffer) -> bool:
    H, W = input_grid.shape
    rel_colors, p_h, p_w = _rel_colors(piece)
    bar_r0, bar_c0, bar_r1, bar_c1 = bar_bounds
    blocked = set(placed_occ) | set(legend_buffer)
    for _ in range(700):
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
        if pts & blocked:
            continue
        if any((r + dr, c + dc) in blocked for r, c in pts for dr, dc in DIR4):
            continue
        for (rr, cc), col in rel_colors.items():
            input_grid[int(top + rr), int(left + cc)] = int(col)
        placed_occ |= pts
        return True
    return False


def _piece_layers(coords: set[tuple[int, int]], conn_cells: set[tuple[int, int]]):
    fill = set(_inner_fill_cells(coords)) - set(conn_cells)
    outline = set(coords) - set(fill) - set(conn_cells)
    return outline, fill


def _color_piece(piece: dict[str, object], outline_color: int, fill_color: int, conn_color: int | None = None) -> dict[str, object]:
    colors = {}
    for rc in piece["outline"]:
        colors[rc] = int(outline_color)
    for rc in piece["fill"]:
        colors[rc] = int(fill_color)
    for rc in piece["conn_cells"]:
        colors[rc] = int(conn_color)
    piece = dict(piece)
    piece["colors"] = colors
    piece["outline_color"] = int(outline_color)
    piece["fill_color"] = int(fill_color)
    piece["conn_color"] = None if conn_color is None else int(conn_color)
    return piece


def _make_piece_with_layers(base_piece: dict[str, object], outline_color: int, fill_color: int, conn_color: int | None = None):
    outline, fill = _piece_layers(base_piece["coords"], base_piece["conn_cells"])
    if len(fill) < 1:
        return None
    piece = dict(base_piece)
    piece["outline"] = outline
    piece["fill"] = fill
    return _color_piece(piece, outline_color, fill_color, conn_color)


def _partial_match_count(notches):
    used = [n for n in notches if n["used"]]
    decoys = [n for n in notches if not n["used"]]
    count = 0
    for dec in decoys:
        for tru in used:
            same_color = dec["spec"]["color"] == tru["spec"]["color"]
            same_proto = dec["spec"]["proto"] == tru["spec"]["proto"]
            if same_color ^ same_proto:
                count += 1
                break
    return count


def _parse_base_structure(grid: np.ndarray, *, arranged: bool) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = get_bg(g)
    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    mono.sort(key=lambda d: -int(d["area"]))
    assert mono
    bar = mono[0]
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = int(bar_r1 - bar_r0 + 1), int(bar_c1 - bar_c0 + 1)
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
        region = {(bar_r0 + int(r), bar_c0 + int(c)) for r, c in pts}
        marker_colors = {rc: int(g[rc]) for rc in region if int(g[rc]) not in (int(bg), int(bar["color"]))}
        conn_cells = set(marker_colors)
        conn_color = None
        conn_norm = None
        conn_anchor = None
        if conn_cells:
            colors = {marker_colors[rc] for rc in conn_cells}
            if len(colors) == 1:
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
                "conn_cells": conn_cells,
                "conn_color": None if conn_color is None else int(conn_color),
                "conn_norm": conn_norm,
                "conn_anchor": conn_anchor,
                "mouth_n": int(mouth_n),
            }
        )

    work = g.copy()
    for r, c in bar_occ:
        work[r, c] = int(bg)
    other_mono = find_components(work, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    body_comps = [comp for comp in other_mono if PIECE_MIN_AREA <= int(comp["area"]) <= PIECE_MAX_AREA]
    connector_comps = []
    if not arranged:
        for comp in other_mono:
            area = int(comp["area"])
            if area not in (2, 3):
                continue
            cells = {(int(r), int(c)) for r, c in comp["coords"]}
            norm, _ = _normalize(cells)
            if norm in ALL_CONNECTOR_SHAPES:
                connector_comps.append(comp)

    pieces = []
    for bi, body in enumerate(body_comps):
        body_cells = {(int(r), int(c)) for r, c in body["coords"]}
        merged = set(body_cells)
        attached_conn = set()
        attached_color = None
        if arranged:
            for notch in notches:
                conn_cells = set(notch["conn_cells"])
                if conn_cells and any((r + dr, c + dc) in body_cells for r, c in conn_cells for dr, dc in DIR4):
                    merged |= conn_cells
                    attached_conn = conn_cells
                    attached_color = int(notch["conn_color"])
                    break
        else:
            for comp in connector_comps:
                conn_cells = {(int(r), int(c)) for r, c in comp["coords"]}
                if any((r + dr, c + dc) in body_cells for r, c in conn_cells for dr, dc in DIR4):
                    merged |= conn_cells
                    attached_conn = conn_cells
                    attached_color = int(comp["color"])
                    break
        pieces.append(_make_piece_dict(g, merged, int(body["color"]), attached_conn, attached_color))

    orientation = "horizontal" if (bar_c1 - bar_c0) > (bar_r1 - bar_r0) else "vertical"
    return {
        "grid": g,
        "bg": int(bg),
        "bar": bar,
        "bar_occ": bar_occ,
        "bar_bbox": (bar_r0, bar_c0, bar_r1, bar_c1),
        "orientation": orientation,
        "notches": notches,
        "pieces": pieces,
    }


def _body_color(piece: dict[str, object]) -> int:
    cols = [int(col) for rc, col in piece["colors_rel"].items() if rc not in piece["conn_rel"]]
    return int(Counter(cols).most_common(1)[0][0])


def _piece_global_parts(piece: dict[str, object]) -> dict[str, set[tuple[int, int]]]:
    top, left = map(int, piece["orig"])
    coords = _pixels(piece["shape"], top, left)
    conn = _shift_rel(piece["conn_rel"], top, left) if piece["conn_rel"] else set()
    fill = set(_inner_fill_cells(coords)) - conn
    outline = coords - fill - conn
    return {"coords": coords, "conn": conn, "fill": fill, "outline": outline}


def _reconstruct_base_chains(base_input: np.ndarray, base_output: np.ndarray):
    info = _parse_base_structure(base_input, arranged=False)
    g = np.array(base_output, dtype=int)
    bg = int(info["bg"])
    work = g.copy()
    for r, c in info["bar_occ"]:
        work[r, c] = int(bg)
    other_mono = find_components(work, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    pieces = []
    for body in other_mono:
        area = int(body["area"])
        if area < PIECE_MIN_AREA or area > PIECE_MAX_AREA:
            continue
        body_cells = {(int(r), int(c)) for r, c in body["coords"]}
        merged = set(body_cells)
        attached_conn = set()
        attached_color = None
        for notch in info["notches"]:
            conn_cells = set(notch["conn_cells"])
            if not conn_cells:
                continue
            candidate = body_cells | conn_cells
            if not any((r + dr, c + dc) in body_cells for r, c in conn_cells for dr, dc in DIR4):
                continue
            if not notch["region"].issubset(candidate):
                continue
            merged = candidate
            attached_conn = conn_cells
            attached_color = int(notch["conn_color"])
            break
        pieces.append(_make_piece_dict(g, merged, int(body["color"]), attached_conn, attached_color))

    fixed_occ = set(info["bar_occ"]) | set().union(*(n["conn_cells"] for n in info["notches"]))

    root_infos = []
    for pi, piece in enumerate(pieces):
        if not piece["conn_rel"]:
            continue
        top, left = map(int, piece["orig"])
        for notch in info["notches"]:
            if piece["conn_color"] != notch["conn_color"] or piece["conn_norm"] != notch["conn_norm"]:
                continue
            if _shift_rel(piece["conn_rel"], top, left) != notch["conn_cells"]:
                continue
            pts = _pixels(piece["shape"], top, left)
            if not notch["region"].issubset(pts):
                continue
            if notch["axis"] == "cols":
                A, B = int(left), int(left + int(piece["shape"]["w"]) - 1)
            else:
                A, B = int(top), int(top + int(piece["shape"]["h"]) - 1)
            root_infos.append({"pi": pi, "axis": notch["axis"], "A": A, "B": B, "side": notch["side"]})
            break

    if not (1 <= len(root_infos) <= 2):
        return None

    unused = set(range(len(pieces)))
    chains = []
    for root in root_infos:
        axis = root["axis"]
        A, B = int(root["A"]), int(root["B"])
        side = root["side"]
        members = []
        for pj, piece in enumerate(pieces):
            if axis == "cols":
                if int(piece["orig"][1]) != A or int(piece["shape"]["w"]) != B - A + 1:
                    continue
            else:
                if int(piece["orig"][0]) != A or int(piece["shape"]["h"]) != B - A + 1:
                    continue
            members.append(int(pj))
        if axis == "cols":
            if side == "top":
                members.sort(key=lambda idx: -int(pieces[idx]["orig"][0]))
            else:
                members.sort(key=lambda idx: int(pieces[idx]["orig"][0]))
        else:
            if side == "left":
                members.sort(key=lambda idx: -int(pieces[idx]["orig"][1]))
            else:
                members.sort(key=lambda idx: int(pieces[idx]["orig"][1]))
        if not members or members[0] != int(root["pi"]):
            return None
        chain = []
        for depth, current_idx in enumerate(members):
            parts = _piece_global_parts(pieces[current_idx])
            chain.append({"piece_idx": current_idx, "depth": depth, "axis": axis, "A": A, "B": B, "parts": parts})
            unused.discard(current_idx)
        chains.append(chain)

    if unused:
        return None
    info = dict(info)
    info["pieces"] = pieces
    return info, chains


def _piece_key(piece: dict[str, object]) -> tuple[int, int | None]:
    rel = frozenset((int(r), int(c)) for r, c in piece["shape"]["rel_set"])
    return (int(piece.get("body_color", _body_color(piece))), (rel, piece["conn_norm"]))


def _gen_attempt():
    rng = _random.Random()
    for _ in range(30):
        try:
            base_input = _BASE_VARIANT.generate_input()
            base_output = _BASE_VARIANT.main(base_input)
        except Exception:
            continue

        recon = _reconstruct_base_chains(base_input, base_output)
        if recon is None:
            continue
        out_info, chains = recon
        in_info = _parse_base_structure(base_input, arranged=False)

        orientation = out_info["orientation"]
        legend = _legend_layout(*base_input.shape, orientation)
        bg = int(out_info["bg"])
        if any(int(base_input[rc]) != bg or int(base_output[rc]) != bg for rc in legend["occ"]):
            continue

        bar_color = int(out_info["bar"]["color"])
        connector_colors = {int(piece["conn_color"]) for piece in out_info["pieces"] if piece["conn_color"] is not None}
        palette_choices = [c for c in range(1, 10) if c != bar_color and c not in connector_colors]
        if len(palette_choices) < LEGEND_LEN:
            continue
        rng.shuffle(palette_choices)
        palette = palette_choices[:LEGEND_LEN]
        outline_pool = [c for c in range(1, 10) if c != bar_color and c not in connector_colors and c not in palette]
        if not outline_pool:
            continue

        color_plan = []
        for chain in chains:
            chain_plan = []
            for step in chain:
                out_piece = out_info["pieces"][int(step["piece_idx"])]
                forbidden = set()
                if out_piece["conn_color"] is not None:
                    forbidden.add(int(out_piece["conn_color"]))
                chain_plan.append({"depth": int(step["depth"]), "forbidden_fill_colors": forbidden})
            color_plan.append(chain_plan)
        if _assign_fill_colors(rng, color_plan, palette) is None:
            continue

        input_pieces_by_key = {}
        bad_map = False
        for piece in in_info["pieces"]:
            key = _piece_key(piece)
            if key in input_pieces_by_key:
                bad_map = True
                break
            input_pieces_by_key[key] = piece
        if bad_map:
            continue
        out_keys = [_piece_key(out_info["pieces"][int(step["piece_idx"])]) for chain in chains for step in chain]
        if len(out_keys) != len(set(out_keys)):
            continue

        input_grid = np.array(base_input, copy=True)
        output_grid = np.array(base_output, copy=True)
        transformed_pieces = 0
        valid = True
        outline_color = int(outline_pool[0])
        for chain_idx, chain in enumerate(chains):
            for pos, step in enumerate(chain):
                out_piece = out_info["pieces"][int(step["piece_idx"])]
                key = _piece_key(out_piece)
                if key not in input_pieces_by_key:
                    valid = False
                    break
                in_piece = input_pieces_by_key[key]
                in_parts = _piece_global_parts(in_piece)
                out_parts = step["parts"]
                if len(in_parts["fill"]) < 1 or len(out_parts["fill"]) < 1:
                    valid = False
                    break
                fill_color = int(color_plan[chain_idx][pos]["fill_color"])
                out_fill = int(palette[int(palette.index(fill_color)) + int(step["depth"])])
                if out_fill != fill_color:
                    transformed_pieces += 1
                in_conn_color = None if not in_piece["conn_rel"] else int(in_piece["conn_color"])
                out_conn_color = None if not out_piece["conn_rel"] else int(out_piece["conn_color"])

                for rc in in_parts["outline"]:
                    input_grid[rc] = int(outline_color)
                for rc in in_parts["fill"]:
                    input_grid[rc] = int(fill_color)
                for rc in in_parts["conn"]:
                    input_grid[rc] = int(in_conn_color)

                for rc in out_parts["outline"]:
                    output_grid[rc] = int(outline_color)
                for rc in out_parts["fill"]:
                    output_grid[rc] = int(out_fill)
                for rc in out_parts["conn"]:
                    output_grid[rc] = int(out_conn_color)
            if not valid:
                break
        if not valid:
            continue

        for color, domino in zip(palette, legend["dominoes"]):
            for r, c in domino:
                input_grid[int(r), int(c)] = int(color)
                output_grid[int(r), int(c)] = int(color)

        meta = {
            "chain_lens": tuple(len(chain) for chain in chains),
            "palette_len": len(palette),
            "num_chains": len(chains),
            "num_pieces": sum(len(chain) for chain in chains),
            "root_connectors": sum(1 for chain in chains for step in chain if out_info["pieces"][int(step["piece_idx"])]["conn_rel"]),
            "transformed_pieces": transformed_pieces,
            "partial_match_decoys": 0,
        }
        if not np.array_equal(main(input_grid), output_grid):
            continue
        return input_grid, output_grid, meta
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out, _meta = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 500 attempts")
