# Variant of 16b78196
# Dimension: D2 — 源对象：散落精灵的类型与属性 (src)
# Change: 单色连通精灵，面积8-13，外接矩形填充率0.54-0.80，不旋转不翻转，散布在杆两侧... → 多色复合精灵
#
# Design analysis
# - The bar/notch/chain geometry stays the same as the original task.
# - The D2 change is isolated to the piece representation: each scattered piece is now one
#   4-connected multicolor component with one outline color plus at least one strict interior
#   fill cell of a second color.
# - Solver matching still uses the occupied silhouette only; once a piece is placed, its
#   per-cell colors are preserved instead of being flattened to one monochrome value.
# - The generator reuses the original valid input construction and recolors each scattered
#   piece into a two-layer version, so the old spatial constraints remain intact.

from __future__ import annotations

import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import components_mask, find_components, get_bg

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _load_original():
    spec = importlib.util.spec_from_file_location("orig_16b78196", _ROOT / "arc2_solver_select" / "16b78196.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_ORIG = _load_original()


def _shape_from_component(grid: np.ndarray, comp: dict) -> dict[str, object]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    rel = sorted((int(r - r0), int(c - c0)) for r, c in comp["coords"])
    rel_colors = {(int(r - r0), int(c - c0)): int(grid[int(r), int(c)]) for r, c in comp["coords"]}
    return {
        "rel": rel,
        "rel_set": set(rel),
        "rel_colors": rel_colors,
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
    }


def _pixels(shape: dict[str, object], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + lr, left + lc) for lr, lc in shape["rel"]}


def _paint_shape(out: np.ndarray, shape: dict[str, object], top: int, left: int) -> None:
    for (lr, lc), color in shape["rel_colors"].items():
        out[top + lr, left + lc] = int(color)


def _clear_shape(out: np.ndarray, pts: set[tuple[int, int]], bg: int, bar_occ: set[tuple[int, int]], bar_color: int) -> None:
    for r, c in pts:
        out[r, c] = int(bg)
    for r, c in bar_occ:
        out[r, c] = int(bar_color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    bar = comps[0]
    pieces_raw = comps[1:]

    out = np.full((H, W), int(bg), dtype=int)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    br, bc = next(iter(bar_occ))
    bar_color = int(g[br, bc])
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"shape": _shape_from_component(g, p), "orig": (r0, c0)})

    def bar_notches(local_bar_mask: np.ndarray, r0: int, c0: int) -> list[dict[str, object]]:
        empty = ~local_bar_mask
        h, w = map(int, empty.shape)
        notches: list[dict[str, object]] = []
        for pts in components_mask(empty, connectivity=4):
            r = pts[:, 0]
            c = pts[:, 1]
            top = int((r == 0).sum())
            bottom = int((r == h - 1).sum())
            left = int((c == 0).sum())
            right = int((c == w - 1).sum())
            if top + bottom + left + right == 0:
                continue
            hole_gl = {(r0 + int(rr), c0 + int(cc)) for rr, cc in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    def placement_pts(shape: dict[str, object], top: int, left: int, occ: set[tuple[int, int]], must_cover=None, strip=None):
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

    def aligned_positions(axis: str, A: int, cover: set[tuple[int, int]], shape: dict[str, object]):
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

    def touches(pts: set[tuple[int, int]], other: set[tuple[int, int]]) -> bool:
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole: set[tuple[int, int]], shape: dict[str, object]) -> set[tuple[int, int]]:
        pos = set()
        for rH, cH in hole:
            for lr, lc in shape["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            shape = piece["shape"]
            for top, left in hole_positions(notch["hole"], shape):
                pts = placement_pts(shape, top, left, bar_occ, must_cover=notch["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + shape["w"] - 1)
                else:
                    A, B = int(top), int(top + shape["h"] - 1)
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(((int(mv), -int(notch["mouth_n"])), {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": shape}))

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:20]]

    best_out = out.copy()
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

        def gen_extend_moves(chain: dict[str, object], remaining_set: set[int]):
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = _pixels(head["shape"], head["top"], head["left"])
            head_top, head_left = int(head["top"]), int(head["left"])
            head_h, head_w = int(head["shape"]["h"]), int(head["shape"]["w"])
            cavity = {(r, c) for r in range(head_top, head_top + head_h) for c in range(head_left, head_left + head_w) if out_grid[r, c] == bg}
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
                    moves.append((pi, top, left, pts, shape))
            return moves

        for chain in chains:
            moves = gen_extend_moves(chain, remaining)
            if not moves:
                continue
            for pi, top, left, pts, shape in moves[:8]:
                _paint_shape(out_grid, shape, top, left)
                old_head = chain["head"]
                chain["head"] = {"top": int(top), "left": int(left), "shape": shape}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                chain["head"] = old_head
                _clear_shape(out_grid, pts, int(bg), bar_occ, int(bar_color))
        return False

    for k in range(1, min(3, len(cand_roots)) + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
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
                _paint_shape(out0, pieces[root["pi"]]["shape"], root["top"], root["left"])
                occ0 |= pts
                chains.append({"axis": root["axis"], "A": root["A"], "B": root["B"], "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]}})
            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


def _strict_interior(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {rc for rc in coords if all((rc[0] + dr, rc[1] + dc) in coords for dr, dc in DIR4)}


def _recolor_multicolor_input(grid: np.ndarray) -> np.ndarray | None:
    g = np.array(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    if len(comps) < 2:
        return None
    bar = comps[0]
    pieces = comps[1:]
    bar_color = int(bar["color"])
    piece_colors = [int(p["color"]) for p in pieces]
    if len(piece_colors) != len(set(piece_colors)):
        return None

    palette = [c for c in range(1, 10) if c != bar_color]
    if len(palette) < len(pieces) + 1:
        return None
    outline_colors = palette[: len(pieces)]
    fill_pool = palette[len(pieces) :]
    if not fill_pool:
        return None

    out = np.array(g, copy=True)
    for piece, outline in zip(pieces, outline_colors):
        coords = {(int(r), int(c)) for r, c in piece["coords"]}
        interior = _strict_interior(coords)
        if not interior:
            return None
        fill = fill_pool[(outline + len(interior)) % len(fill_pool)]
        if fill == outline:
            fill = fill_pool[(fill_pool.index(fill) + 1) % len(fill_pool)]
        for r, c in coords:
            out[r, c] = int(fill if (r, c) in interior else outline)
    return out


def generate_input() -> np.ndarray:
    for _ in range(800):
        base = _ORIG.generate_input()
        recolored = _recolor_multicolor_input(base)
        if recolored is None:
            continue
        _ = main(recolored)
        return recolored
    raise RuntimeError("generate_input: failed after 800 attempts")
