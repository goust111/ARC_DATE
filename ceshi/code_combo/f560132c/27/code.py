# Combo variant of f560132c (index 27)
# combo_spec: D1/0+D5/2+D8/1+D9/3
# Name: 旋转匹配+分散边缘图例+干扰精灵

import math
import random
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    - Original task:
#      4 components are exactly the 4 pieces; one of them is also a multicolor legend
#      component. `main()` can infer k from component count, find the unique multicolor
#      piece, then assemble all 4 pieces directly.
#    - Combo behavior:
#      the visible border now holds the legend, split into 4 singleton pixels:
#        top edge    -> assembled top-left color
#        right edge  -> assembled top-right color
#        left edge   -> assembled bottom-left color
#        bottom edge -> assembled bottom-right color
#      After stripping those 4 border markers, the remaining interior has 5-6 monochrome
#      sprites: 4 true jigsaw pieces plus 1-2 distractors. The true anchor is the unique
#      largest sprite, but it is scattered in a rotated/flipped orientation relative to
#      its assembled orientation.
#    - Small trace:
#      suppose the border markers decode to `[[2, 7], [4, 8]]`, and the remaining sprites
#      are `A, B, C, D, X` where `A` is the largest L-shape and `X` is a distractor.
#      Old logic would fail immediately because there is no multicolor component and the
#      post-marker sprite count is 5, not 4. The new logic does:
#        1. read the 4 border pixels into a 2x2 legend map;
#        2. remove them from the mask;
#        3. pick the unique largest sprite as anchor;
#        4. try every 4-piece subset containing that anchor;
#        5. for each subset, search D4 orientations and placements;
#        6. keep the unique subset whose rotated/flipped pieces tile one square exactly.
#
# b) Complexity calibration
#    - The border legend uses only one fixed side-to-quadrant mapping, so the edge rule is
#      learnable from a few examples instead of exploding into many cases.
#    - Distractors are limited to 1-2 sprites. That is enough to force subset selection
#      without making the search or the visual field chaotic.
#    - The true anchor is uniquely largest and uses an asymmetric L-shape. That prevents
#      the detached legend from making the assembly underconstrained and also makes the
#      required reorientation visually noticeable.
#    - k stays fixed at 2. The fusion adds three reasoning layers, but each layer has small,
#      repeated structure: 4 markers, 4 true pieces, and at most 2 distractors.
#
# c) Hidden assumptions from the original code that break here
#    - `len(comps) == 4` no longer holds because there are 4 extra border markers and 1-2
#      extra distractor sprites.
#    - "legend = component with most distinct colors" is invalid because every sprite is now
#      monochrome; the legend must be decoded from the border instead.
#    - "assemble every remaining component" is invalid because some sprites are distractors.
#    - "the legend piece is the anchor" is invalid because the legend is not a piece anymore.
#    - "input orientation is irrelevant in generation" is too weak for D1; the generator must
#      explicitly produce cases where at least one true piece, specifically the anchor, is not
#      already in its assembled orientation.
#
# d) Placement legality
#    - All sprites are placed with border >= 2, so row 0 / col 0 / last row / last col remain
#      free for detached markers and markers never merge with sprites under 4-connectivity.
#    - Markers are single non-corner pixels, one per side, with an interior background neighbor.
#    - Distractors and true pieces are scattered by `random_free_location`, so they stay
#      disjoint and in-bounds.
#
# e) Visual observability
#    - Humans can see the legend moved from the center into four edge cues.
#    - Humans can also see there are too many sprites, because one or two shapes disappear in
#      the output.
#    - The anchor L-shape often appears visibly rotated/flipped versus the output, so the
#      orientation search is supported by the examples rather than hidden in solver internals.
#
# f) Visual cleanliness
#    - Border markers occupy empty border cells only; they do not overlap piece bodies.
#    - Palette colors are disjoint from body colors, so the legend dots read as instructions
#      rather than as more candidate pieces.
#    - Distractors are small connected polyominoes, not speckle noise.
#
# g) Color-rule inferability
#    - Only 4 palette colors exist, matching the 2x2 assembled quadrants exactly.
#    - The side-to-quadrant mapping is stable across examples:
#        top -> TL, right -> TR, left -> BL, bottom -> BR.
#      No color-id ordering is used.
#
# h) Generator-side anti-degeneration checks
#    - `main(grid) == expected` must hold.
#    - There must be exactly one valid 4-piece subset containing the anchor.
#    - The anchor must be scattered in a non-assembled orientation.
#    - The generated scene must still contain 5-6 non-marker sprites after removing border dots.
#
# i) New solver assumptions and matching generator guarantees
#    - Assumption: exactly 4 singleton edge markers exist, one per side.
#      Guarantee: generator stamps exactly one isolated marker on each side.
#    - Assumption: the unique largest remaining sprite is a true anchor piece.
#      Guarantee: generator rejects samples unless the anchor beats every other sprite by area.
#    - Assumption: exactly one anchor-containing 4-piece subset exact-covers the output square.
#      Guarantee: generator counts valid subsets and rejects ambiguous cases.
#    - Assumption: D4 search is genuinely needed.
#      Guarantee: generator rejects samples unless the anchor is not scattered in its assembled
#      orientation, and tracks how many true pieces were reoriented.
# ---------------------------------------------------------------------------


K = 2
NUM_TRUE_PIECES = K * K


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


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = np.asarray(mask, dtype=int)
    return tuple(map(int, arr.shape)), tuple(map(int, arr.ravel()))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _unique_oriented_masks(mask: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(mask, dtype=int)
    out: list[np.ndarray] = []
    seen = set()
    for rot_k in range(4):
        rot = np.rot90(arr, rot_k)
        for oriented in (rot, np.fliplr(rot)):
            pts = np.argwhere(oriented > 0)
            r0, c0, r1, c1 = bbox_coords(pts)
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = _mask_key(crop)
            if key not in seen:
                seen.add(key)
                out.append(crop)
    return out


def _shape_keys_from_mask(mask: np.ndarray) -> set[tuple[tuple[int, int], ...]]:
    coords = tuple(map(tuple, np.argwhere(mask)))
    rel_norm, _, _ = _normalize_coords(coords)
    return {ori[0] for ori in _dihedral_orientations(rel_norm)}


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_shape_keys_from_mask(mask))


def _mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    norm, h, w = _normalize_coords(coords)
    out = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        out[r, c] = True
    return out


def _edge_side(r: int, c: int, h: int, w: int) -> str | None:
    if r == 0 and 0 < c < w - 1:
        return "top"
    if r == h - 1 and 0 < c < w - 1:
        return "bottom"
    if c == 0 and 0 < r < h - 1:
        return "left"
    if c == w - 1 and 0 < r < h - 1:
        return "right"
    return None


def _extract_edge_legend_map(g: np.ndarray, bg: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    h, w = g.shape
    by_side: dict[str, tuple[int, int, int]] = {}
    for pts in components_mask(g != bg, connectivity=4):
        if int(len(pts)) != 1:
            continue
        r, c = map(int, pts[0])
        side = _edge_side(r, c, h, w)
        if side is None:
            continue
        assert side not in by_side, "expected one singleton legend marker per border side"
        by_side[side] = (r, c, int(g[r, c]))

    need = {"top", "right", "left", "bottom"}
    assert set(by_side) == need, "expected exactly four edge legend markers"
    legend_map = np.array(
        [
            [by_side["top"][2], by_side["right"][2]],
            [by_side["left"][2], by_side["bottom"][2]],
        ],
        dtype=int,
    )
    marker_cells = {(r, c) for r, c, _ in by_side.values()}
    return legend_map, marker_cells


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


def _build_piece(pts: np.ndarray, bg: int, g: np.ndarray) -> dict:
    filled_abs, _ = _fill_holes_in_component(g, pts, bg)
    r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
    rel = tuple((r - r0, c - c0) for r, c in filled_abs)
    rel_norm, _, _ = _normalize_coords(rel)
    return {"oris": _dihedral_orientations(rel_norm), "area": len(filled_abs)}


def _try_assemble_exact_cover(
    pieces: list[dict],
    anchor_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray | None:
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = [(h, w) for h, w in _factor_pairs(total) if h >= max_h and w >= max_w]
    if len(dims) != 1:
        return None

    H, W = dims[0]
    nbits = H * W
    all_mask = (1 << nbits) - 1

    other_pis = [i for i in range(len(pieces)) if i != anchor_idx]
    placements_other: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
    for pi in other_pis:
        opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
        for coords, ph, pw, _, _ in pieces[pi]["oris"]:
            for r0 in range(0, H - ph + 1):
                for c0 in range(0, W - pw + 1):
                    mask = 0
                    for dr, dc in coords:
                        idx = (int(r0) + int(dr)) * int(W) + (int(c0) + int(dc))
                        mask |= 1 << idx
                    opts.append((mask, int(r0), int(c0), coords))
        placements_other.append(opts)

    def solve_with_fixed_anchor(
        anchor_coords: tuple[tuple[int, int], ...],
    ) -> dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] | None:
        anchor_mask = 0
        for dr, dc in anchor_coords:
            idx = int(dr) * int(W) + int(dc)
            anchor_mask |= 1 << idx

        filtered = [[pl for pl in opts if (pl[0] & anchor_mask) == 0] for opts in placements_other]
        remaining = set(range(len(other_pis)))
        chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] = {
            anchor_idx: (anchor_mask, 0, 0, anchor_coords)
        }

        def search(used: int) -> bool:
            if not remaining:
                return used == all_mask
            empty = (~used) & all_mask
            if empty == 0:
                return False
            cell = (empty & -empty).bit_length() - 1
            candidates = []
            for local_idx in sorted(remaining):
                opts_here = []
                for pl in filtered[local_idx]:
                    mask = pl[0]
                    if (mask >> cell) & 1 and (mask & used) == 0:
                        opts_here.append(pl)
                if opts_here:
                    candidates.append((len(opts_here), local_idx, opts_here))
            if not candidates:
                return False

            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, local_idx, opts_here in candidates:
                remaining.remove(local_idx)
                pi = other_pis[local_idx]
                for pl in opts_here:
                    chosen[pi] = pl
                    if search(used | pl[0]):
                        return True
                chosen.pop(pi, None)
                remaining.add(local_idx)
            return False

        if not search(anchor_mask):
            return None
        return chosen

    chosen = None
    for coords, ph, pw, _, _ in pieces[anchor_idx]["oris"]:
        if ph > H or pw > W:
            continue
        sol = solve_with_fixed_anchor(coords)
        if sol is not None:
            chosen = sol
            break
    if chosen is None:
        return None

    piece_cell: dict[int, tuple[int, int]] = {}
    for pi, pl in chosen.items():
        _, r0, c0, coords = pl
        rs = [r0 + dr for dr, _ in coords]
        cs = [c0 + dc for _, dc in coords]
        cr = float(sum(rs)) / float(len(rs))
        cc = float(sum(cs)) / float(len(cs))
        rr = int(cr * float(k) / float(H))
        cc2 = int(cc * float(k) / float(W))
        rr = max(0, min(int(k) - 1, rr))
        cc2 = max(0, min(int(k) - 1, cc2))
        piece_cell[pi] = (rr, cc2)

    out = np.zeros((H, W), dtype=int)
    for pi, pl in chosen.items():
        _, r0, c0, coords = pl
        rr, cc2 = piece_cell[pi]
        col = int(legend_map[rr, cc2])
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = col
    return out


def _count_valid_anchor_subsets(pieces: list[dict], anchor_idx: int, legend_map: np.ndarray, k: int) -> int:
    need = k * k - 1
    others = [i for i in range(len(pieces)) if i != anchor_idx]
    valid = 0
    for chosen_others in combinations(others, need):
        subset_indices = [anchor_idx, *chosen_others]
        subset_pieces = [pieces[i] for i in subset_indices]
        if _try_assemble_exact_cover(subset_pieces, 0, legend_map, k) is not None:
            valid += 1
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the true monochrome sprites into one square using border markers as a detached legend.

    Input structure:
    - `bg` is the most common color.
    - Four singleton pixels sit on the outer border, one on each side, and together encode the 2x2 recoloring legend.
    - Inside the border are 5-6 monochrome sprites: 4 true jigsaw pieces and 1-2 distractors that never appear in the output.
    - The true anchor piece is the unique largest sprite, but sprites may be rotated or mirrored relative to their assembled form.

    Rule:
    - Decode the four edge markers into a 2x2 legend map, then remove those marker cells from the piece mask.
    - Among the remaining sprites, keep the unique largest one as the fixed top-left anchor and search over subsets of three others.
    - For each candidate subset, try rotations/reflections and placements until one 4-piece subset tiles a tight square exactly.
    - Recolor the assembled pieces by the legend color selected from the piece centroid's quadrant and output the cropped assembly.

    Key insight:
    - Solving requires three linked inferences: reconstruct the legend from detached border cues, ignore extra sprites that do not
      participate in the tiling, and rotate/flip the surviving pieces so the unique valid exact cover can be formed.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    legend_map, marker_cells = _extract_edge_legend_map(g, bg)
    piece_mask = g != bg
    for r, c in marker_cells:
        piece_mask[r, c] = False

    comps = components_mask(piece_mask, connectivity=4)
    assert len(comps) in (5, 6), "expected 5-6 interior sprites after removing edge markers"

    pieces = [_build_piece(pts, bg, g) for pts in comps]
    areas = [int(piece["area"]) for piece in pieces]
    max_area = max(areas)
    anchor_candidates = [i for i, area in enumerate(areas) if area == max_area]
    assert len(anchor_candidates) == 1, "expected a unique largest anchor sprite"
    anchor_idx = anchor_candidates[0]

    need = K * K - 1
    others = [i for i in range(len(pieces)) if i != anchor_idx]
    solutions = []
    for chosen_others in combinations(others, need):
        subset_indices = [anchor_idx, *chosen_others]
        subset_pieces = [pieces[i] for i in subset_indices]
        out = _try_assemble_exact_cover(subset_pieces, 0, legend_map, K)
        if out is not None:
            solutions.append(out)

    assert len(solutions) == 1, "expected exactly one valid anchor-containing exact-cover subset"
    return solutions[0]


# --------------- generator ---------------

SIDE_CHOICES = [8, 10, 12]
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
SCATTER_BORDER = 2
SCATTER_PADDING = 1
ANCHOR_AREA_GAP = 4
GEN_RETRIES = 500
DISTRACTOR_COUNT_CHOICES = [1, 2]
DISTRACTOR_TEMPLATES = [
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2), (3, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0), (3, 1), (3, 2)),
]


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


def _make_anchor_mask(side: int, min_area: int, max_area: int) -> np.ndarray:
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


def _quadrant_cells(masks: list[np.ndarray], side: int) -> list[tuple[int, int]]:
    qcells = []
    for mask in masks:
        pts = np.argwhere(mask)
        qcells.append((min(1, int(pts[:, 0].mean() * K / side)), min(1, int(pts[:, 1].mean() * K / side))))
    return qcells


def _edge_marker_positions(grid_size: int) -> dict[str, tuple[int, int]]:
    mid = grid_size // 2
    top_cols = list(range(2, mid))
    right_rows = list(range(2, mid))
    left_rows = list(range(mid, grid_size - 2))
    bottom_cols = list(range(mid, grid_size - 2))
    assert top_cols and right_rows and left_rows and bottom_cols, "grid too small for border markers"
    return {
        "top": (0, random.choice(top_cols)),
        "right": (random.choice(right_rows), grid_size - 1),
        "left": (random.choice(left_rows), 0),
        "bottom": (grid_size - 1, random.choice(bottom_cols)),
    }


def _stamp_edge_markers(grid: np.ndarray, legend_map: np.ndarray, bg: int) -> None:
    coords = _edge_marker_positions(grid.shape[0])
    side_to_color = {
        "top": int(legend_map[0, 0]),
        "right": int(legend_map[0, 1]),
        "left": int(legend_map[1, 0]),
        "bottom": int(legend_map[1, 1]),
    }
    for side, (r, c) in coords.items():
        assert int(grid[r, c]) == bg, "marker cell must be empty"
        if side == "top":
            assert int(grid[r + 1, c]) == bg
        elif side == "bottom":
            assert int(grid[r - 1, c]) == bg
        elif side == "left":
            assert int(grid[r, c + 1]) == bg
        else:
            assert int(grid[r, c - 1]) == bg
        grid[r, c] = side_to_color[side]


def _make_distractors(true_masks: list[np.ndarray]) -> list[np.ndarray] | None:
    true_keys = set()
    for mask in true_masks:
        true_keys |= _shape_keys_from_mask(_crop_mask(mask))

    distractors = []
    used_keys = set()
    count = random.choice(DISTRACTOR_COUNT_CHOICES)
    for _ in range(count):
        placed = False
        for _attempt in range(80):
            base = _mask_from_coords(random.choice(DISTRACTOR_TEMPLATES))
            oriented = random.choice(_unique_oriented_masks(base))
            key = _canonical_shape_key(oriented)
            if key in true_keys or key in used_keys:
                continue
            if _has_hole(oriented):
                continue
            distractors.append(oriented)
            used_keys.add(key)
            placed = True
            break
        if not placed:
            return None
    return distractors


def _choose_true_piece_orientations(
    masks: list[np.ndarray],
) -> tuple[list[np.ndarray], int, bool] | None:
    oriented_masks: list[np.ndarray] = []
    rotated_count = 0
    anchor_rotated = False
    for pi, mask in enumerate(masks):
        base = _crop_mask(mask)
        base_key = _mask_key(base)
        opts = _unique_oriented_masks(base)
        diff_opts = [opt for opt in opts if _mask_key(opt) != base_key]
        if pi == 0:
            if not diff_opts:
                return None
            chosen = random.choice(diff_opts)
            anchor_rotated = True
        else:
            if diff_opts and random.random() < 0.75:
                chosen = random.choice(diff_opts)
            else:
                chosen = random.choice(opts)
        if _mask_key(chosen) != base_key:
            rotated_count += 1
        oriented_masks.append(chosen)
    return oriented_masks, rotated_count, anchor_rotated


def _scatter_scene(
    masks: list[np.ndarray],
    body_colors: list[int],
    legend_map: np.ndarray,
    distractor_masks: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, int | bool]] | None:
    total = sum(int(m.sum()) for m in masks) + sum(int(m.sum()) for m in distractor_masks)
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None

    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    chosen_true = _choose_true_piece_orientations(masks)
    if chosen_true is None:
        return None
    true_oriented, rotated_true_pieces, anchor_rotated = chosen_true

    sprites: list[np.ndarray] = []
    for pi, oriented in enumerate(true_oriented):
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = int(body_colors[pi])
        sprites.append(patch)

    distractor_colors = [random.choice(body_colors) for _ in distractor_masks]
    for mask, color in zip(distractor_masks, distractor_colors):
        oriented = random.choice(_unique_oriented_masks(mask))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = int(color)
        sprites.append(patch)

    random.shuffle(sprites)
    for patch in sprites:
        try:
            box = np.ones(patch.shape, dtype=int)
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
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=0)

    _stamp_edge_markers(grid, legend_map, bg=0)
    meta = {
        "grid_size": int(grid_size),
        "rotated_true_pieces": int(rotated_true_pieces),
        "anchor_rotated": bool(anchor_rotated),
        "distractor_count": int(len(distractor_masks)),
    }
    return grid, meta


def _gen_attempt() -> dict | None:
    side = random.choice(SIDE_CHOICES)
    min_area, max_area = _area_bounds(side)
    anchor = _make_anchor_mask(side, min_area, max_area)
    labels = _partition_rest(~anchor)
    if labels is None:
        return None

    masks = [anchor] + [labels == i for i in range(3)]
    areas = [int(m.sum()) for m in masks]
    if any(a < min_area or a > max_area for a in areas):
        return None
    if areas[0] < max(areas[1:]) + ANCHOR_AREA_GAP:
        return None
    if any(len(components_mask(m, connectivity=4)) != 1 or _has_hole(m) for m in masks):
        return None
    if not _forces_unique_square(masks, side):
        return None

    qcells = _quadrant_cells(masks, side)
    if len(set(qcells)) != NUM_TRUE_PIECES:
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_TRUE_PIECES]
    palette_colors = colors[NUM_TRUE_PIECES:]
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)

    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        expected[mask] = legend_map[rr, cc]

    distractors = _make_distractors(masks)
    if distractors is None:
        return None

    scattered = _scatter_scene(masks, body_colors, legend_map, distractors)
    if scattered is None:
        return None
    grid, meta = scattered

    if meta["rotated_true_pieces"] < 1 or not meta["anchor_rotated"]:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    bg = get_bg(grid)
    decoded_map, marker_cells = _extract_edge_legend_map(grid, bg)
    if not np.array_equal(decoded_map, legend_map):
        return None
    piece_mask = grid != bg
    for r, c in marker_cells:
        piece_mask[r, c] = False
    comps = components_mask(piece_mask, connectivity=4)
    if len(comps) != NUM_TRUE_PIECES + meta["distractor_count"]:
        return None

    pieces = [_build_piece(pts, bg, grid) for pts in comps]
    piece_areas = [int(piece["area"]) for piece in pieces]
    anchor_candidates = [i for i, area in enumerate(piece_areas) if area == max(piece_areas)]
    if len(anchor_candidates) != 1:
        return None
    valid_subset_count = _count_valid_anchor_subsets(pieces, anchor_candidates[0], legend_map, K)
    if valid_subset_count != 1:
        return None

    meta["valid_subset_count"] = int(valid_subset_count)
    meta["piece_components"] = int(len(comps))
    meta["output_side"] = int(side)
    return {"grid": grid, "expected": expected, "meta": meta}


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _gen_attempt()
        if case is not None:
            return case["grid"]
    raise RuntimeError("failed to generate f560132c combo variant input")
