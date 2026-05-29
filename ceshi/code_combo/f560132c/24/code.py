# Combo variant of f560132c (index 24)
# combo_spec: D6/0+D12/2+D15/1
# Name: 原始网格输出+连接线轨迹+翻转输出

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


# Design analysis for the fused variant:
# - Concrete behavior:
#   Original f560132c assembled the four scattered sprites into a tight square and returned
#   only that cropped square. Here the same exact-cover assembly still happens first, but the
#   rendered result is projected back onto the original canvas at a fixed anchor `(1, 1)`.
#   Example: if the original logic would output an 8x8 colored square, this variant instead
#   paints that 8x8 square at the upper-left of the old 20x20/26x26 canvas, mirrors the whole
#   square left-right, and draws one thin target-colored line from each source-piece centroid to
#   the centroid of its mirrored destination piece.
# - Complexity calibration:
#   The output changes are visible but controlled: only one fixed placement rule, one fixed
#   reflection axis, and exactly four trails. The generator scores source placements so the
#   trails stay long and readable instead of collapsing into tiny local edits.
# - Broken assumptions from the original main():
#   The original solver assumed the final answer was the assembled square itself. That no longer
#   holds, so the assembly stage is separated from the rendering stage. Another old assumption
#   was that only the palette colors matter in the output; after adding trails we must retain
#   each piece's destination color as metadata so the trails stay tied to the recolored target
#   rather than introducing a second, unrelated color system. The generator guarantees the legend
#   body color remains the dominant color inside the multicolor legend component.
# - Placement legality:
#   The generator reserves the future output square in the input canvas, then searches bbox-safe
#   placements for all four source pieces with a one-cell padding buffer, so nothing overlaps or
#   clips. Lines are allowed to run into the destination square; the mirrored assembly is painted
#   last so the final object stays legible.
# - Visual observability and cleanliness:
#   The upper-left anchored mirrored square is obvious across examples, while the background
#   stays otherwise unchanged. Top-row destination pieces are sourced from the right side of the
#   input and bottom-row destination pieces from the lower side, which keeps the four trails
#   mostly radial and limits line-line collisions.
# - New solver assumptions and generator support:
#   `main()` now assumes the assembled square fits at `(1, 1)` and that every piece has a
#   meaningful centroid-to-centroid displacement so the trail is visible. The generator enforces
#   input sizes large enough for that anchor and rejects samples whose shortest trail or total
#   visible trail area is too small.


OUTPUT_R0 = 1
OUTPUT_C0 = 1

SIDE_CHOICES = [8, 10, 12]
K = 2
NUM_PIECES = K * K
NUM_COLORS = 8
AREA_FRAC_LO = 0.08
AREA_FRAC_HI = 0.43
AREA_ABS_LO = 11
AREA_ABS_HI = 62
LEGEND_THRESHOLD = {8: 4, 10: 5, 12: 9}
LEGEND_CORE_SIZE = 4
LEGEND_ARM_RANGE = range(1, 4)
PARTITION_GROWTH_EXP = 1.45
PARTITION_FRONTIER_EXP = 0.2
PARTITION_SEED_JITTER = 0.2
INPUT_SIZE_LO = 20
INPUT_SIZE_HI = 26
DENSITY_LO = 0.15
DENSITY_HI = 0.25
GEN_RETRIES = 400
SOURCE_PADDING = 1
TARGET_MOAT = 1
CANDIDATE_LIMIT = 160
MIN_TRAIL_MANHATTAN = 6
MIN_TRAIL_PIXELS = 12


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, int(n**0.5) + 1):
        if n % h == 0:
            w = n // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    return out


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, w = _normalize_coords(coords)
    rot = [(c, h - 1 - r) for r, c in norm]
    return _normalize_coords(rot)


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, w = _normalize_coords(coords)
    flp = [(r, w - 1 - c) for r, c in norm]
    return _normalize_coords(flp)


def _dihedral_orientations(
    coords: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]] = []
    cur = coords
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), False))
        cur = _rot90(cur)[0]
    cur = _flip_h(coords)[0]
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), True))
        cur = _rot90(cur)[0]
    return out


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, k: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert k > 0 and hh >= k and ww >= k, "legend component must fit k×k map"
    best = None
    best_rc = None
    for rr in range(hh - k + 1):
        for cc in range(ww - k + 1):
            window = sub[rr : rr + k, cc : cc + k]
            distinct = int(len(set(map(int, window.ravel()))))
            key = (distinct, -rr, -cc)
            if best is None or key > best:
                best = key
                best_rc = (rr, cc)
    assert best_rc is not None, "legend component must contain a valid k×k color map window"
    rr, cc = best_rc
    return np.asarray(sub[rr : rr + k, cc : cc + k], dtype=int).copy()


def _fill_holes_in_component(g: np.ndarray, pts: np.ndarray, bg: int) -> tuple[tuple[tuple[int, int], ...], int]:
    r0, c0, r1, c1 = bbox_coords(pts)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    occ = np.zeros((hh, ww), dtype=bool)
    for r, c in pts:
        occ[r - r0, c - c0] = True

    bgmask = ~occ
    vis = np.zeros_like(bgmask, dtype=bool)
    st: list[tuple[int, int]] = []

    for rr in range(hh):
        for cc in range(ww):
            if rr in (0, hh - 1) or cc in (0, ww - 1):
                if bgmask[rr, cc] and not vis[rr, cc]:
                    vis[rr, cc] = True
                    st.append((rr, cc))
                    while st:
                        r, c = st.pop()
                        for r2, c2 in neighbors4(r, c):
                            if 0 <= r2 < hh and 0 <= c2 < ww and bgmask[r2, c2] and not vis[r2, c2]:
                                vis[r2, c2] = True
                                st.append((r2, c2))

    holes = bgmask & ~vis
    hole_coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(holes), strict=False)]
    filled = tuple(sorted(set(map(tuple, pts)).union(hole_coords)))
    return filled, int(holes.sum())


def _component_body_color(g: np.ndarray, pts: np.ndarray, bg: int) -> int:
    vals = [int(g[r, c]) for r, c in pts if int(g[r, c]) != int(bg)]
    return int(Counter(vals).most_common(1)[0][0])


def _centroid(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[float, float]:
    rs = [float(r) for r, _ in coords]
    cs = [float(c) for _, c in coords]
    return float(sum(rs) / len(rs)), float(sum(cs) / len(cs))


def _round_point(pt: tuple[float, float]) -> tuple[int, int]:
    return int(round(pt[0])), int(round(pt[1]))


def _line_cells(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, p0)
    r1, c1 = map(int, p1)
    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((int(r), int(c)))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def _assemble_solution(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> dict:
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = _factor_pairs(total)
    dims = [(H, W) for (H, W) in dims if H >= max_h and W >= max_w]
    assert len(dims) == 1, "piece areas and bounds must imply a unique output rectangle"
    H, W = dims[0]
    nbits = H * W
    all_mask = (1 << nbits) - 1

    other_pis = [i for i in range(len(pieces)) if i != legend_idx]
    placements_other: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
    for pi in other_pis:
        opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
        for coords, ph, pw, _, _ in pieces[pi]["oris"]:
            for r0 in range(0, H - ph + 1):
                for c0 in range(0, W - pw + 1):
                    m = 0
                    for dr, dc in coords:
                        idx = (int(r0) + int(dr)) * int(W) + (int(c0) + int(dc))
                        m |= 1 << idx
                    opts.append((m, int(r0), int(c0), coords))
        placements_other.append(opts)

    def solve_with_fixed_legend(
        legend_coords: tuple[tuple[int, int], ...],
    ) -> dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] | None:
        legend_mask = 0
        for dr, dc in legend_coords:
            idx = int(dr) * int(W) + int(dc)
            legend_mask |= 1 << idx

        filtered: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
        for opts in placements_other:
            filtered.append([pl for pl in opts if (pl[0] & legend_mask) == 0])

        remaining = set(range(len(other_pis)))
        chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] = {
            legend_idx: (legend_mask, 0, 0, legend_coords)
        }

        def search(used: int) -> bool:
            if not remaining:
                return used == all_mask
            empty = (~used) & all_mask
            if empty == 0:
                return False
            cell = (empty & -empty).bit_length() - 1
            candidates = []
            for rem_idx in sorted(remaining):
                opts_k = []
                for pl in filtered[rem_idx]:
                    m = pl[0]
                    if (m >> cell) & 1 and (m & used) == 0:
                        opts_k.append(pl)
                if opts_k:
                    candidates.append((len(opts_k), rem_idx, opts_k))
            if not candidates:
                return False
            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, rem_idx, opts_k in candidates:
                remaining.remove(rem_idx)
                pi = other_pis[rem_idx]
                for pl in opts_k:
                    chosen[pi] = pl
                    if search(used | pl[0]):
                        return True
                chosen.pop(pi, None)
                remaining.add(rem_idx)
            return False

        if not search(legend_mask):
            return None
        return chosen

    chosen = None
    for coords, ph, pw, _, _ in pieces[legend_idx]["oris"]:
        assert ph <= H and pw <= W, "legend orientation must fit the unique output rectangle"
        sol = solve_with_fixed_legend(coords)
        if sol is not None:
            chosen = sol
            break
    assert chosen is not None, "exact-cover assembly failed: no valid legend orientation"

    piece_cell: dict[int, tuple[int, int]] = {}
    piece_colors: dict[int, int] = {}
    piece_coords_local: dict[int, tuple[tuple[int, int], ...]] = {}
    square = np.zeros((H, W), dtype=int)

    for pi, pl in chosen.items():
        _, r0, c0, coords = pl
        placed = tuple(sorted((int(r0 + dr), int(c0 + dc)) for dr, dc in coords))
        piece_coords_local[pi] = placed
        cr, cc = _centroid(list(placed))
        rr = max(0, min(int(k) - 1, int(cr * float(k) / float(H))))
        cc2 = max(0, min(int(k) - 1, int(cc * float(k) / float(W))))
        piece_cell[pi] = (rr, cc2)
        piece_colors[pi] = int(legend_map[rr, cc2])
        for r, c in placed:
            square[r, c] = int(piece_colors[pi])

    return {
        "side_h": int(H),
        "side_w": int(W),
        "chosen": chosen,
        "piece_cell": piece_cell,
        "piece_colors": piece_colors,
        "piece_coords_local": piece_coords_local,
        "square_unflipped": square,
        "square_flipped": np.fliplr(square),
    }


def _build_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = components_mask(g != bg, connectivity=4)
    assert comps, "grid must contain at least one non-background component"

    k = int(round(len(comps) ** 0.5))
    assert k * k == len(comps), "piece count must be a perfect square to determine k"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend component must be uniquely identified by distinct colors"
    legend_idx = legend_candidates[0]
    legend_map = _extract_legend_map(g, comps[legend_idx], k)

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        pieces.append(
            {
                "oris": oris,
                "area": len(filled_abs),
                "filled_abs": filled_abs,
                "body_color": _component_body_color(g, pts, bg),
                "source_center": _centroid(list(filled_abs)),
            }
        )

    solution = _assemble_solution(pieces, legend_idx, legend_map, k)
    side = int(solution["side_h"])
    assert side == int(solution["side_w"]), "generator only produces square assemblies"
    H, W = g.shape
    assert OUTPUT_R0 + side <= H and OUTPUT_C0 + side <= W, "fixed output anchor must fit inside the input canvas"

    target_centers_unflipped: dict[int, tuple[float, float]] = {}
    target_centers_flipped: dict[int, tuple[float, float]] = {}
    target_centers_unflipped_round: dict[int, tuple[int, int]] = {}
    target_centers_flipped_round: dict[int, tuple[int, int]] = {}
    for pi, local_coords in solution["piece_coords_local"].items():
        unflip_abs = [(OUTPUT_R0 + r, OUTPUT_C0 + c) for r, c in local_coords]
        flip_abs = [(OUTPUT_R0 + r, OUTPUT_C0 + (side - 1 - c)) for r, c in local_coords]
        cu = _centroid(unflip_abs)
        cf = _centroid(flip_abs)
        target_centers_unflipped[pi] = cu
        target_centers_flipped[pi] = cf
        target_centers_unflipped_round[pi] = _round_point(cu)
        target_centers_flipped_round[pi] = _round_point(cf)

    return {
        "grid": g,
        "bg": bg,
        "legend_idx": legend_idx,
        "legend_map": legend_map,
        "pieces": pieces,
        "solution": solution,
        "side": side,
        "target_centers_unflipped": target_centers_unflipped,
        "target_centers_flipped": target_centers_flipped,
        "target_centers_unflipped_round": target_centers_unflipped_round,
        "target_centers_flipped_round": target_centers_flipped_round,
    }


def _render_scene(scene: dict, *, flipped: bool = True, with_lines: bool = True) -> np.ndarray:
    g = scene["grid"]
    bg = int(scene["bg"])
    side = int(scene["side"])
    out = np.full_like(g, bg)

    if with_lines:
        for pi, piece in enumerate(scene["pieces"]):
            src = _round_point(piece["source_center"])
            dst = scene["target_centers_flipped_round"][pi] if flipped else scene["target_centers_unflipped_round"][pi]
            trail_color = int(scene["solution"]["piece_colors"][pi])
            for r, c in _line_cells(src, dst):
                if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                    out[r, c] = trail_color

    square = scene["solution"]["square_flipped"] if flipped else scene["solution"]["square_unflipped"]
    paste(out, square, OUTPUT_R0, OUTPUT_C0)
    return out


def _scene_metrics(scene: dict) -> dict[str, int]:
    out = _render_scene(scene, flipped=True, with_lines=True)
    side = int(scene["side"])
    mask = np.ones(out.shape, dtype=bool)
    mask[OUTPUT_R0 : OUTPUT_R0 + side, OUTPUT_C0 : OUTPUT_C0 + side] = False
    trail_pixels = int(np.count_nonzero(out[mask] != int(scene["bg"])))
    dists = []
    for pi, piece in enumerate(scene["pieces"]):
        sr, sc = _round_point(piece["source_center"])
        tr, tc = scene["target_centers_flipped_round"][pi]
        dists.append(abs(int(sr) - int(tr)) + abs(int(sc) - int(tc)))
    flip_diff = int(
        np.count_nonzero(
            scene["solution"]["square_flipped"] != scene["solution"]["square_unflipped"]
        )
    )
    return {
        "trail_pixels": int(trail_pixels),
        "min_trail_manhattan": int(min(dists)),
        "max_trail_manhattan": int(max(dists)),
        "flip_diff_cells": int(flip_diff),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered sprites exactly as in the base task, then reproject the result onto the original canvas with motion trails and a horizontal mirror.

    Input structure:
    - bg = most common color.
    - Four disconnected jigsaw sprites are scattered on a larger canvas; one is a multicolor legend piece with an embedded 2x2 palette.
    - The other three pieces are monochrome, and all four body colors remain distinct from the legend palette colors.

    Rule:
    - Recover the same exact-cover square used by the base task: tile the four pieces into the unique full square and recolor each piece from the legend palette according to its assembled centroid cell.
    - Instead of cropping that square, place it at the fixed anchor `(1, 1)` in the original-size output, then mirror the whole assembled square left-right.
    - Remove the source sprites, keep the original background elsewhere, and draw one thin line in each piece's destination color from its source centroid to the centroid of its mirrored destination piece.

    Key insight:
    - The legend still defines the recoloring, but the final answer is no longer just the assembled square: the solver must combine exact-cover assembly, a global horizontal reflection of that assembly, and centroid-to-centroid motion traces on the unchanged canvas.
    """
    scene = _build_scene(np.asarray(grid, dtype=int))
    return _render_scene(scene, flipped=True, with_lines=True)


def _area_bounds(side: int) -> tuple[int, int]:
    total = side * side
    return max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total)), min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))


def _has_hole(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return True
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    empty = ~sub
    if not empty.any():
        return False
    for comp in components_mask(empty, connectivity=4):
        if not (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == sub.shape[0] - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == sub.shape[1] - 1)
        ):
            return True
    return False


def _forces_unique_square(masks: list[np.ndarray], side: int) -> bool:
    bbs = [bbox_coords(np.argwhere(m)) for m in masks]
    max_h = max(r1 - r0 + 1 for r0, _, r1, _ in bbs)
    max_w = max(c1 - c0 + 1 for _, c0, _, c1 in bbs)
    dims = [(h, w) for h, w in _factor_pairs(side * side) if h >= max_h and w >= max_w]
    return dims == [(side, side)]


def _make_legend_mask(side: int, min_area: int, max_area: int) -> np.ndarray:
    threshold = LEGEND_THRESHOLD[side]
    opts = []
    for h0 in range(threshold + 1, side + 1):
        for w0 in range(threshold + 1, side + 1):
            for arm_w in LEGEND_ARM_RANGE:
                for arm_h in LEGEND_ARM_RANGE:
                    mask = np.zeros((side, side), dtype=bool)
                    mask[:LEGEND_CORE_SIZE, :LEGEND_CORE_SIZE] = True
                    mask[:arm_h, :w0] = True
                    mask[:h0, :arm_w] = True
                    area = int(mask.sum())
                    if min_area <= area <= max_area:
                        opts.append(mask)
    return random.choice(opts).copy()


def _partition_rest(rest: np.ndarray) -> np.ndarray | None:
    side = rest.shape[0]
    labels = np.full((side, side), -1, dtype=int)
    available = set(map(tuple, np.argwhere(rest)))
    anchors = [
        (random.randint(0, side // 3), side - 1),
        (side - 1, random.randint(0, side // 3)),
        (side - 1, side - 1),
    ]

    seeds = []
    for ar, ac in anchors:
        seed = min(available, key=lambda p: abs(p[0] - ar) + abs(p[1] - ac) + random.random() * PARTITION_SEED_JITTER)
        seeds.append(seed)
        available.remove(seed)

    areas = [0, 0, 0]
    frontiers = [set() for _ in range(3)]

    def add_frontier(i: int, r: int, c: int) -> None:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < side and 0 <= cc < side and rest[rr, cc] and labels[rr, cc] < 0:
                frontiers[i].add((rr, cc))

    for i, seed in enumerate(seeds):
        labels[seed] = i
        areas[i] = 1
        add_frontier(i, seed[0], seed[1])

    unassigned = int(rest.sum()) - len(seeds)
    while unassigned:
        candidates = []
        for i in range(3):
            frontiers[i] = {p for p in frontiers[i] if labels[p] < 0}
            if frontiers[i]:
                candidates.append(i)
        if not candidates:
            return None

        weights = [
            (1.0 / (areas[i] ** PARTITION_GROWTH_EXP)) * (len(frontiers[i]) ** PARTITION_FRONTIER_EXP)
            for i in candidates
        ]
        i = random.choices(candidates, weights=weights, k=1)[0]
        cell = random.choice(tuple(frontiers[i]))
        frontiers[i].discard(cell)
        labels[cell] = i
        areas[i] += 1
        unassigned -= 1
        add_frontier(i, cell[0], cell[1])

    return labels


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _unique_oriented_masks(mask: np.ndarray) -> list[np.ndarray]:
    arr = mask.astype(int)
    out = []
    seen = set()
    for k in range(4):
        rot = np.rot90(arr, k)
        for oriented in (rot, np.fliplr(rot)):
            pts = np.argwhere(oriented > 0)
            r0, c0, r1, c1 = bbox_coords(pts)
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                out.append(crop)
    return out


def _palette_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(1, h - K):
        for c in range(1, w - K):
            if mask[r : r + K, c : c + K].all() and mask[r - 1 : r + K + 1, c - 1 : c + K + 1].all():
                spots.append((r, c))
    return spots


def _centroid_from_mask(mask: np.ndarray, r0: int = 0, c0: int = 0) -> tuple[float, float]:
    pts = np.argwhere(mask)
    coords = [(int(r0 + r), int(c0 + c)) for r, c in pts]
    return _centroid(coords)


def _ideal_source_point(local_center: tuple[float, float], side: int, grid_size: int) -> tuple[int, int]:
    lr, lc = local_center
    if lr < side / 2.0:
        extra = 0 if lc >= side / 2.0 else side // 2
        r = OUTPUT_R0 + int(round(lr)) + extra
        c = grid_size - 3
    else:
        extra = 0 if lc < side / 2.0 else side // 2
        r = grid_size - 3
        c = OUTPUT_C0 + int(round(lc)) + extra
    return max(1, min(grid_size - 2, r)), max(1, min(grid_size - 2, c))


def _candidate_placements(
    patch_variants: list[np.ndarray],
    local_center: tuple[float, float],
    grid_size: int,
    reserved: np.ndarray,
    side: int,
) -> list[dict]:
    candidates = []
    ideal_r, ideal_c = _ideal_source_point(local_center, side, grid_size)
    boundary_c = OUTPUT_C0 + side + TARGET_MOAT
    boundary_r = OUTPUT_R0 + side + TARGET_MOAT
    for patch in patch_variants:
        h, w = patch.shape
        for r0 in range(1, grid_size - h):
            for c0 in range(1, grid_size - w):
                if reserved[r0 : r0 + h, c0 : c0 + w].any():
                    continue
                src_r, src_c = _centroid_from_mask(patch > 0, r0, c0)
                score = -(
                    abs(int(round(src_r)) - int(ideal_r)) + abs(int(round(src_c)) - int(ideal_c))
                )
                if local_center[0] < side / 2.0 and int(round(src_c)) <= boundary_c:
                    score -= 200
                if local_center[0] >= side / 2.0 and int(round(src_r)) <= boundary_r:
                    score -= 200
                score += random.random() * 0.25
                candidates.append(
                    {
                        "score": float(score),
                        "r0": int(r0),
                        "c0": int(c0),
                        "h": int(h),
                        "w": int(w),
                        "patch": patch,
                    }
                )
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:CANDIDATE_LIMIT]


def _search_scatter(cands_by_piece: dict[int, list[dict]], grid_size: int, reserved: np.ndarray) -> dict[int, dict] | None:
    order = sorted(cands_by_piece, key=lambda pi: len(cands_by_piece[pi]))
    occ = reserved.copy()
    chosen: dict[int, dict] = {}

    def mark(cand: dict, val: bool) -> None:
        r0 = max(0, int(cand["r0"]) - SOURCE_PADDING)
        c0 = max(0, int(cand["c0"]) - SOURCE_PADDING)
        r1 = min(grid_size, int(cand["r0"]) + int(cand["h"]) + SOURCE_PADDING)
        c1 = min(grid_size, int(cand["c0"]) + int(cand["w"]) + SOURCE_PADDING)
        occ[r0:r1, c0:c1] = val

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        pi = order[idx]
        for cand in cands_by_piece[pi]:
            r0 = max(0, int(cand["r0"]) - SOURCE_PADDING)
            c0 = max(0, int(cand["c0"]) - SOURCE_PADDING)
            r1 = min(grid_size, int(cand["r0"]) + int(cand["h"]) + SOURCE_PADDING)
            c1 = min(grid_size, int(cand["c0"]) + int(cand["w"]) + SOURCE_PADDING)
            if occ[r0:r1, c0:c1].any():
                continue
            chosen[pi] = cand
            mark(cand, True)
            if dfs(idx + 1):
                return True
            mark(cand, False)
            chosen.pop(pi, None)
        return False

    if dfs(0):
        return chosen
    return None


def _build_input_grid(
    masks: list[np.ndarray],
    body_colors: list[int],
    legend_map: np.ndarray,
    side: int,
    target_centers_local_flipped: dict[int, tuple[float, float]],
) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    nlo = max(nlo, side + 12)
    if nlo > nhi:
        return None
    grid_size = random.randint(nlo, nhi)

    reserved = np.zeros((grid_size, grid_size), dtype=bool)
    rr0 = max(0, OUTPUT_R0 - TARGET_MOAT)
    cc0 = max(0, OUTPUT_C0 - TARGET_MOAT)
    rr1 = min(grid_size, OUTPUT_R0 + side + TARGET_MOAT)
    cc1 = min(grid_size, OUTPUT_C0 + side + TARGET_MOAT)
    reserved[rr0:rr1, cc0:cc1] = True

    cands_by_piece: dict[int, list[dict]] = {}
    for pi, mask in enumerate(masks):
        variants = []
        for oriented in _unique_oriented_masks(_crop_mask(mask)):
            patch = np.zeros(oriented.shape, dtype=int)
            patch[oriented] = int(body_colors[pi])
            if pi == 0:
                spots = _palette_spots(oriented)
                if not spots:
                    continue
                spot = random.choice(spots)
                patch = patch.copy()
                r, c = spot
                patch[r : r + K, c : c + K] = legend_map
            variants.append(patch)
        if not variants:
            return None
        cands = _candidate_placements(variants, target_centers_local_flipped[pi], grid_size, reserved, side)
        if not cands:
            return None
        cands_by_piece[pi] = cands

    placement = _search_scatter(cands_by_piece, grid_size, reserved)
    if placement is None:
        return None

    grid = np.zeros((grid_size, grid_size), dtype=int)
    for pi in range(NUM_PIECES):
        cand = placement[pi]
        paste(grid, cand["patch"], cand["r0"], cand["c0"], transparent=0)
    return grid


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SIDE_CHOICES)
    min_area, max_area = _area_bounds(side)
    legend = _make_legend_mask(side, min_area, max_area)
    labels = _partition_rest(~legend)
    if labels is None:
        return None

    masks = [legend] + [labels == i for i in range(3)]
    areas = [int(m.sum()) for m in masks]
    if any(a < min_area or a > max_area for a in areas):
        return None
    if any(len(components_mask(m, connectivity=4)) != 1 or _has_hole(m) for m in masks):
        return None
    if not _forces_unique_square(masks, side):
        return None

    qcells = []
    for mask in masks:
        pts = np.argwhere(mask)
        qcells.append((min(1, int(pts[:, 0].mean() * K / side)), min(1, int(pts[:, 1].mean() * K / side))))
    if len(set(qcells)) != NUM_PIECES:
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_PIECES]
    palette_colors = colors[NUM_PIECES:]
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)

    target_centers_local_flipped: dict[int, tuple[float, float]] = {}
    for pi, mask in enumerate(masks):
        pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        flipped = [(r, side - 1 - c) for r, c in pts]
        target_centers_local_flipped[pi] = _centroid(flipped)

    grid = _build_input_grid(masks, body_colors, legend_map, side, target_centers_local_flipped)
    if grid is None:
        return None

    scene = _build_scene(grid)
    metrics = _scene_metrics(scene)
    if metrics["min_trail_manhattan"] < MIN_TRAIL_MANHATTAN:
        return None
    if metrics["trail_pixels"] < MIN_TRAIL_PIXELS:
        return None
    if metrics["flip_diff_cells"] <= 0:
        return None

    expected = _render_scene(scene, flipped=True, with_lines=True)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate fused f560132c input")
