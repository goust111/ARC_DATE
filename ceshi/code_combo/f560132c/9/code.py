# Combo variant of f560132c (index 9)
# combo_spec: D5/0+D6/2
# Name: 线性色带图例+象限分离输出

from __future__ import annotations

# Design analysis for the combo variant D5/0 + D6/2
# a) Concretization and behavior trace:
#    - Input still shows 4 scattered monochrome jigsaw pieces; exactly 1 piece is
#      the legend because it contains extra colors.
#    - Original task: the legend embeds a 2x2 palette block; solver reads that
#      block directly as [TL TR; BL BR], assembles the full square, and outputs
#      the whole recolored square.
#    - Fused task: the legend instead embeds a 1x4 horizontal strip. Reading
#      order is strictly left-to-right and means [TL, TR, BL, BR]. After exact
#      cover assembly, we still recolor by centroid quadrant, but the final grid
#      is no longer the full square: we cut out the four geometric quadrants of
#      the assembled result, crop each quadrant tightly, and place those four
#      cropped panels in a 2x2 layout with one-pixel separator gaps.
#    - Example sketch:
#         strip [8 2 4 6] -> legend_map [[8,2],[4,6]]
#         assembled square -> split into TL/TR/BL/BR windows -> crop each
#         window independently -> output = [[crop(TL) | crop(TR)],
#                                          [crop(BL) | crop(BR)]]
#
# b) Complexity calibration:
#    - The strip orientation is fixed to horizontal. Allowing vertical strips
#      here would add a second decoding branch that the fusion description does
#      not ask for, and it would dilute the visibility of the new output-format
#      change.
#    - The output layout uses exactly one separator width and always a 2x2 panel
#      arrangement. This keeps the display rule small and repeatable across 2-4
#      examples.
#    - Each sample still uses only 4 body colors + 4 legend colors, so the
#      visual vocabulary stays close to the source task.
#
# c) Implicit assumption audit:
#    - Original `_extract_legend_map()` assumed a kxk color window exists. That
#      is false now; solver must instead detect one horizontal length-4 strip.
#    - The original legend decoder did not need a notion of read direction. The
#      fused solver now assumes the strip is shown in canonical left-to-right
#      order. The generator keeps the legend piece unrotated to satisfy this.
#    - Original output code assumed the final answer is the assembled rectangle
#      itself. That assumption is replaced by "assemble first, then reformat
#      into four cropped quadrant panels".
#    - The new panel formatter assumes every geometric quadrant contains at least
#      one foreground cell. The generator enforces distinct centroid quadrants
#      for the four pieces, so each quadrant is occupied.
#
# d) Placement legality:
#    - The legend strip is only embedded where a full 3x6 body region exists, so
#      the four colored strip cells remain inside the same connected component
#      and have a 1-cell monochrome ring around them.
#    - Scattering still uses `random_free_location`, keeping piece bounding boxes
#      non-overlapping, in-bounds, and separated by padding.
#    - Output-side panel placement is deterministic and non-overlapping because
#      each cropped panel is pasted into its own row/column slot with a fixed
#      one-pixel gap between slots.
#
# e) Visual observability:
#    - The legend visibly changes from a square palette to a horizontal 4-cell
#      ribbon. Humans can read the only plausible order as left-to-right.
#    - The output visibly changes from a single square mosaic into four separate
#      cropped panels. This is strong enough that a viewer can notice the grid-
#      shape rule even before understanding the exact recoloring logic.
#
# f) Visual cleanliness:
#    - The input keeps the original sparse layout: four disconnected sprites on a
#      black background, with only the legend containing a compact interior mark.
#    - The output has only one extra structural device: 1-pixel background gaps
#      between panels. No extra border color or textural overlay is added.
#
# g) Change inferability:
#    - Color remapping is still explained structurally by the legend strip and
#      the 2x2 quadrant concept. The strip order is consistent across all
#      samples, and the separated output preserves the same TL/TR/BL/BR ordering
#      in panel layout.
#    - Degeneracy guard: generated samples always use the horizontal strip, and
#      the output is always panel-separated rather than the original full square.

import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, int(n**0.5) + 1):
        if n % h == 0:
            w = n // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    return out


def _normalize_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, _ = _normalize_coords(coords)
    rot = [(c, h - 1 - r) for r, c in norm]
    return _normalize_coords(rot)


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, _, w = _normalize_coords(coords)
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


def _extract_legend_sequence(g: np.ndarray, pts: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Find the unique interior 1xN horizontal color strip inside the legend bbox.
    The strip is surrounded by a 1-cell ring of the legend body color.
    """
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert hh >= 3 and ww >= seq_len + 2, "legend component must fit an interior horizontal strip"

    body_color = Counter(int(g[r, c]) for r, c in pts).most_common(1)[0][0]
    candidates: list[np.ndarray] = []

    for rr in range(1, hh - 1):
        for cc in range(1, ww - seq_len):
            frame = sub[rr - 1 : rr + 2, cc - 1 : cc + seq_len + 1]
            strip = np.asarray(sub[rr, cc : cc + seq_len], dtype=int)
            if len(set(map(int, strip))) != seq_len:
                continue
            if any(int(v) == body_color for v in strip):
                continue
            ring_ok = True
            for fr in range(3):
                for fc in range(seq_len + 2):
                    if fr == 1 and 1 <= fc <= seq_len:
                        continue
                    if int(frame[fr, fc]) != body_color:
                        ring_ok = False
                        break
                if not ring_ok:
                    break
            if ring_ok:
                candidates.append(strip.copy())

    assert len(candidates) == 1, f"expected a unique horizontal legend strip, got {len(candidates)}"
    return candidates[0]


def _sequence_to_legend_map(seq: np.ndarray, k: int) -> np.ndarray:
    assert int(seq.size) == k * k, "linear legend sequence must have k*k entries"
    flat = [int(x) for x in np.asarray(seq, dtype=int).ravel()]
    return np.asarray([[flat[0], flat[1]], [flat[2], flat[3]]], dtype=int)


def _fill_holes_in_component(
    g: np.ndarray, pts: np.ndarray, bg: int
) -> tuple[tuple[tuple[int, int], ...], int]:
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
    hole_coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(holes))]
    filled = tuple(sorted(set(map(tuple, pts)).union(hole_coords)))
    return filled, int(holes.sum())


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray:
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = _factor_pairs(total)
    dims = [(h, w) for h, w in dims if h >= max_h and w >= max_w]
    assert len(dims) == 1, "piece areas and bounds must imply a unique output rectangle"

    for H, W in dims:
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
                for rel_idx in sorted(remaining):
                    opts_k = []
                    for pl in filtered[rel_idx]:
                        m = pl[0]
                        if (m >> cell) & 1 and (m & used) == 0:
                            opts_k.append(pl)
                    if opts_k:
                        candidates.append((len(opts_k), rel_idx, opts_k))
                if not candidates:
                    return False

                candidates.sort(key=lambda x: (x[0], x[1]))
                for _, rel_idx, opts_k in candidates:
                    remaining.remove(rel_idx)
                    pi = other_pis[rel_idx]
                    for pl in opts_k:
                        chosen[pi] = pl
                        if search(used | pl[0]):
                            return True
                    chosen.pop(pi, None)
                    remaining.add(rel_idx)
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
        assert chosen is not None, "exact-cover assembly failed"

        piece_cell: dict[int, tuple[int, int]] = {}
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            rs = [r0 + dr for dr, _ in coords]
            cs = [c0 + dc for _, dc in coords]
            cr = float(sum(rs)) / float(len(rs))
            cc = float(sum(cs)) / float(len(cs))
            rr = max(0, min(k - 1, int(cr * float(k) / float(H))))
            cc2 = max(0, min(k - 1, int(cc * float(k) / float(W))))
            piece_cell[pi] = (rr, cc2)

        out = np.zeros((H, W), dtype=int)
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            rr, cc2 = piece_cell[pi]
            col = int(legend_map[rr, cc2])
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        return out

    raise AssertionError("unreachable assembly branch")


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(arr != bg)
    assert len(pts) > 0, "cropped panel must contain foreground"
    r0, c0, r1, c1 = bbox_coords(pts)
    return np.asarray(arr[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _separate_quadrants(full: np.ndarray, k: int, bg: int) -> np.ndarray:
    H, W = full.shape
    assert k == 2, "this fused output formatter is defined for 2x2 quadrants"
    assert H % k == 0 and W % k == 0, "assembled rectangle must split cleanly into 2x2 quadrants"

    qh = H // k
    qw = W // k
    tiles: list[list[np.ndarray]] = []
    for rr in range(k):
        row_tiles = []
        for cc in range(k):
            sub = full[rr * qh : (rr + 1) * qh, cc * qw : (cc + 1) * qw]
            row_tiles.append(_crop_non_bg(sub, bg))
        tiles.append(row_tiles)

    row_heights = [max(tile.shape[0] for tile in row) for row in tiles]
    col_widths = [max(tiles[r][c].shape[1] for r in range(k)) for c in range(k)]
    out_h = sum(row_heights) + (k - 1)
    out_w = sum(col_widths) + (k - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    r_base = 0
    for rr in range(k):
        c_base = 0
        for cc in range(k):
            tile = tiles[rr][cc]
            paste(out, tile, r_base, c_base, transparent=bg)
            c_base += col_widths[cc] + 1
        r_base += row_heights[rr] + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces, decode a horizontal legend strip, and output separated quadrant panels.

    Input structure:
    - Background is the most common color.
    - Four scattered non-background sprites act as jigsaw pieces; one is the legend because it contains multiple colors.
    - The legend no longer carries a 2x2 palette block. Instead it embeds one interior 1x4 horizontal strip of four distinct colors inside its monochrome body.

    Rule:
    - Reassemble all four pieces into the unique tight rectangle, allowing rotations and reflections for the tiling step.
    - Read the legend strip from left to right as [top-left, top-right, bottom-left, bottom-right], convert that into a 2x2 quadrant color map, and recolor each assembled piece by the quadrant containing its centroid.
    - Do not return the full recolored rectangle. Split that rectangle into its four geometric quadrants, tightly crop each quadrant's occupied cells, and place the four cropped panels in a 2x2 arrangement with 1-pixel separator gaps.

    Key insight:
    - The puzzle now combines two abstractions: a 1D legend must be unfolded back into the hidden 2D quadrant map, and the final answer must be understood as a separated presentation of those four quadrants rather than the intact assembled square.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, "grid must contain at least one non-background component"

    k = int(round(len(comps) ** 0.5))
    assert k * k == len(comps), "piece count must be a perfect square"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend component must be uniquely identifiable"
    legend_idx = legend_candidates[0]

    legend_seq = _extract_legend_sequence(g, comps[legend_idx], k * k)
    legend_map = _sequence_to_legend_map(legend_seq, k)

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        pieces.append({"oris": _dihedral_orientations(rel_norm), "area": len(filled_abs)})

    full = _assemble_exact_cover(pieces, legend_idx, legend_map, k)
    return _separate_quadrants(full, k, bg=0)


# --------------- generator (combo D5/0 + D6/2) ---------------

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
LEGEND_STRIP_LEN = NUM_PIECES
PARTITION_GROWTH_EXP = 1.45
PARTITION_FRONTIER_EXP = 0.2
PARTITION_SEED_JITTER = 0.2
INPUT_SIZE_LO = 20
INPUT_SIZE_HI = 26
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 400


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


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _legend_strip_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots: list[tuple[int, int]] = []
    h, w = mask.shape
    if h < 3 or w < LEGEND_STRIP_LEN + 2:
        return spots
    for r in range(1, h - 1):
        for c in range(1, w - LEGEND_STRIP_LEN):
            if mask[r - 1 : r + 2, c - 1 : c + LEGEND_STRIP_LEN + 1].all():
                spots.append((r, c))
    return spots


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
                    if not (min_area <= area <= max_area):
                        continue
                    if not _legend_strip_spots(_crop_mask(mask)):
                        continue
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


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(int)
    opts = []
    seen = set()
    for k2 in range(4):
        rot = np.rot90(arr, k2)
        for oriented in (rot, np.fliplr(rot)):
            r0, c0, r1, c1 = bbox_coords(np.argwhere(oriented > 0))
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop)
    return random.choice(opts)


def _scatter_pieces(
    masks: list[np.ndarray], body_colors: list[int], legend_seq: np.ndarray, side: int
) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _crop_mask(masks[pi]) if pi == 0 else _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        if pi == 0:
            spots = _legend_strip_spots(oriented)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r, c : c + LEGEND_STRIP_LEN] = legend_seq
        try:
            box = np.ones(oriented.shape, dtype=int)
            r0, c0 = random_free_location(
                occupied,
                box,
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        occupied[r0 : r0 + oriented.shape[0], c0 : c0 + oriented.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=0)
    return grid


def _assembled_expected(masks: list[np.ndarray], legend_seq: np.ndarray, side: int) -> np.ndarray:
    legend_map = _sequence_to_legend_map(legend_seq, K)
    full = np.zeros((side, side), dtype=int)
    qcells = []
    for mask in masks:
        pts = np.argwhere(mask)
        qcells.append((min(1, int(pts[:, 0].mean() * K / side)), min(1, int(pts[:, 1].mean() * K / side))))
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        full[mask] = legend_map[rr, cc]
    return full


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
    legend_seq = np.asarray(palette_colors, dtype=int)

    expected_full = _assembled_expected(masks, legend_seq, side)
    expected_sep = _separate_quadrants(expected_full, K, bg=0)
    if expected_sep.shape == expected_full.shape and np.array_equal(expected_sep, expected_full):
        return None

    grid = _scatter_pieces(masks, body_colors, legend_seq, side)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected_sep):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c combo input")
