# Variant of f560132c
# Dimension: D9 — 精灵数量与象限划分 (meta)
# Change: 4个精灵对应2x2=4个象限，每个象限恰好一个精灵... → 含干扰精灵（多余精灵不参与拼装）

import math
import random
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D9 variant
#
# a) Concrete behavior and flow impact
#    - The assembled output is unchanged: exactly four regions still tile one SxS
#      square and are recolored by the 2x2 palette embedded in the legend piece.
#    - The input now contains 5-6 connected sprites. Exactly one sprite is still the
#      legend, three other sprites are real jigsaw pieces, and 1-2 additional
#      monochrome sprites are distractors.
#    - Original flow at the critical step:
#         components -> len(comps)=4 -> infer k=2 from sqrt(4) -> assemble all 4.
#      Variant flow:
#         components -> len(comps)=5 or 6 -> identify the unique multicolor legend
#         -> keep k fixed at 2 -> try every subset of 3 non-legend sprites -> accept
#         the unique subset whose 4 pieces can tile a rectangle exactly.
#      Small example: with components [legend, A, B, C, X], the old logic crashes at
#      sqrt(5). The new logic tests {A,B,C}, {A,B,X}, {A,C,X}, {B,C,X}; only
#      {A,B,C} exact-covers the square, so X is excluded.
#
# b) Complexity calibration
#    - Extra sprites are connected polyominoes rather than single pixels or obvious
#      markers, so the visual change is real and the solver must reason over shape.
#    - Distractors are kept to 1-2 only. That is enough to force exclusion without
#      exploding the combination count or making examples unreadable.
#    - Distractors are drawn from a small template family and then D4-randomized.
#      This gives repeated visual support across examples while keeping variety.
#
# c) Hidden assumptions in the original main()
#    - `k = round(sqrt(len(comps)))`: invalid once extras exist. Replaced by fixed
#      task constant `K=2`, which matches the visible 2x2 legend palette.
#    - `assert k*k == len(comps)`: invalid. Removed.
#    - `pieces = comps` and `assemble all pieces`: invalid. Replaced by subset search.
#    - `legend is unique by max distinct colors`: still valid because generator keeps
#      all distractors monochrome and the legend remains the only multicolor sprite.
#    - `_assemble_exact_cover` used assertions for invalid candidate sets. In the
#      variant, invalid subsets are expected during search, so a non-throwing
#      `_try_assemble_exact_cover` wrapper is required.
#
# d) Placement legality
#    - Every sprite, including distractors, is scattered by `random_free_location`
#      with the same border/padding rules. This guarantees no overlap and keeps each
#      component separable under 4-connectivity.
#    - The legend palette is still embedded strictly inside the legend body so it is
#      observed as one connected component rather than four loose cells.
#
# e) Visual observability
#    - Across examples, humans still see one multicolor legend and a group of
#      monochrome shapes. The difference from the base task is visible because there
#      are now extra disconnected shapes that never appear in the output.
#    - The core clue remains spatial and shape-based; no hidden ordering by color id
#      or index is introduced.
#
# f) Visual cleanliness
#    - Distractors stay monochrome, smaller than the main pieces, and spatially
#      separated. They do not overlap the palette or other sprites, so the grid does
#      not become visually noisy.
#
# g) Color inference
#    - Output recoloring is unchanged and still fully explained by the 2x2 palette in
#      the legend. Distractors do not introduce new recoloring rules; they only add
#      a shape-selection step before assembly.
#
# h) Generator-side anti-degeneration checks
#    - For every candidate input, generator verifies `main(input) == expected`.
#    - It also counts how many 4-piece subsets (legend + 3 others) can tile exactly.
#      Only instances with exactly one valid subset are accepted. This prevents
#      distractors from being silently interchangeable with real pieces.
#
# i) New solver assumptions and generator guarantees
#    - Solver assumes exactly one legend candidate has the maximum distinct-color
#      count. Generator enforces that.
#    - Solver assumes exactly one subset of size 4 assembles. Generator rejects any
#      sample with zero or multiple valid subsets.
#    - Solver assumes the visible legend palette is 2x2. Generator always embeds a
#      2x2 palette and keeps the surrounding body intact.
# ---------------------------------------------------------------------------


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
    """
    Unique orientations under rotations + mirror (D4).
    Returned as (coords, h, w, rot_k, flipped), where the transform is:
      - if flipped: horizontal flip (mirror left-right)
      - then rot_k times 90° clockwise rotation
    coords are normalized to start at (0,0).
    """
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
    """
    Inside the legend component's bounding box, find a k×k color map block (may include 0).
    We pick the k×k window with the maximum number of distinct colors; ties go to top-left.
    """
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


def _fill_holes_in_component(
    g: np.ndarray, pts: np.ndarray, bg: int
) -> tuple[tuple[tuple[int, int], ...], int]:
    """
    Treat background 'holes' fully enclosed inside the component bbox as part of the piece.
    Returns (filled_pts, hole_count), where filled_pts is a sorted tuple of absolute coords.
    """
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
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray | None:
    """
    Try to tile all pieces into one full rectangle (no holes), allowing D4 transforms.
    Return the recolored output grid on success, or None for invalid subsets.
    """
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = [(h, w) for h, w in _factor_pairs(total) if h >= max_h and w >= max_w]
    if len(dims) != 1:
        return None

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

        filtered = [[pl for pl in opts if (pl[0] & legend_mask) == 0] for opts in placements_other]
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
            for local_idx in sorted(remaining):
                opts_k = []
                for pl in filtered[local_idx]:
                    m = pl[0]
                    if (m >> cell) & 1 and (m & used) == 0:
                        opts_k.append(pl)
                if opts_k:
                    candidates.append((len(opts_k), local_idx, opts_k))
            if not candidates:
                return False

            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, local_idx, opts_k in candidates:
                remaining.remove(local_idx)
                pi = other_pis[local_idx]
                for pl in opts_k:
                    chosen[pi] = pl
                    if search(used | pl[0]):
                        return True
                chosen.pop(pi, None)
                remaining.add(local_idx)
            return False

        if not search(legend_mask):
            return None
        return chosen

    chosen = None
    for coords, ph, pw, _, _ in pieces[legend_idx]["oris"]:
        if ph > H or pw > W:
            continue
        sol = solve_with_fixed_legend(coords)
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


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray:
    out = _try_assemble_exact_cover(pieces, legend_idx, legend_map, k)
    assert out is not None, "exact-cover assembly failed"
    return out


def _count_valid_piece_subsets(pieces: list[dict], legend_idx: int, legend_map: np.ndarray, k: int) -> int:
    need = k * k - 1
    others = [i for i in range(len(pieces)) if i != legend_idx]
    valid = 0
    for chosen_others in combinations(others, need):
        subset_indices = [legend_idx, *chosen_others]
        subset_pieces = [pieces[i] for i in subset_indices]
        if _try_assemble_exact_cover(subset_pieces, 0, legend_map, k) is not None:
            valid += 1
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the true jigsaw sprites into one rectangle, ignoring extra distractor sprites.

    Input structure:
    - bg is the most common color.
    - One connected component is a multicolor legend piece containing a 2×2 palette block.
    - The remaining non-bg components are monochrome sprites: three are real puzzle pieces and
      1-2 more are distractors that do not belong in the final tiling.

    Rule:
    - Keep the legend piece and choose exactly three additional sprites so that those four pieces
      can tile one tight rectangle under rotations/reflections.
    - Discard any extra sprites that cannot participate in that exact cover.
    - Recolor each selected piece using the legend's 2×2 palette according to the piece centroid's
      quadrant in the assembled rectangle, then output the cropped assembly.

    Key insight:
    - The legend is still both a piece and a color key, but component counting is no longer enough:
      the solver must identify the unique 4-piece exact-cover subset and treat the leftover sprites
      as visual decoys.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    k = K

    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) >= k * k, "grid must contain at least four non-background components"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend component must be uniquely identified by distinct colors"
    legend_idx = legend_candidates[0]
    legend_map = _extract_legend_map(g, comps[legend_idx], k)

    pieces = [_build_piece(pts, bg, g) for pts in comps]

    need = k * k - 1
    others = [i for i in range(len(pieces)) if i != legend_idx]
    solutions = []
    for chosen_others in combinations(others, need):
        subset_indices = [legend_idx, *chosen_others]
        subset_pieces = [pieces[i] for i in subset_indices]
        out = _try_assemble_exact_cover(subset_pieces, 0, legend_map, k)
        if out is not None:
            solutions.append(out)

    assert len(solutions) == 1, "there must be exactly one valid 4-piece assembly subset"
    return solutions[0]


# --------------- generator (o5x) ---------------

# === CFG ===
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
SCATTER_BORDER = 1
SCATTER_PADDING = 1
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


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(int)
    opts = []
    seen = set()
    for rot_k in range(4):
        rot = np.rot90(arr, rot_k)
        for oriented in (rot, np.fliplr(rot)):
            r0, c0, r1, c1 = bbox_coords(np.argwhere(oriented > 0))
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop)
    return random.choice(opts)


def _palette_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(1, h - K):
        for c in range(1, w - K):
            if mask[r : r + K, c : c + K].all() and mask[r - 1 : r + K + 1, c - 1 : c + K + 1].all():
                spots.append((r, c))
    return spots


def _mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    norm, h, w = _normalize_coords(coords)
    out = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        out[r, c] = True
    return out


def _shape_keys_from_mask(mask: np.ndarray) -> set[tuple[tuple[int, int], ...]]:
    coords = tuple(map(tuple, np.argwhere(mask)))
    rel_norm, _, _ = _normalize_coords(coords)
    return {ori[0] for ori in _dihedral_orientations(rel_norm)}


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_shape_keys_from_mask(mask))


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
            oriented = _random_orientation(base)
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


def _scatter_pieces(
    masks: list[np.ndarray],
    body_colors: list[int],
    legend_map: np.ndarray,
    distractor_masks: list[np.ndarray],
) -> np.ndarray | None:
    total = sum(int(m.sum()) for m in masks) + sum(int(m.sum()) for m in distractor_masks)
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None

    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    sprites: list[np.ndarray] = []
    for pi in range(NUM_PIECES):
        oriented = _crop_mask(masks[pi]) if pi == 0 else _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        if pi == 0:
            spots = _palette_spots(oriented)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r : r + K, c : c + K] = legend_map
        sprites.append(patch)

    distractor_colors = [random.choice(body_colors) for _ in distractor_masks]
    for mask, color in zip(distractor_masks, distractor_colors):
        oriented = _random_orientation(mask)
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

    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        expected[mask] = legend_map[rr, cc]

    distractors = _make_distractors(masks)
    if distractors is None:
        return None

    grid = _scatter_pieces(masks, body_colors, legend_map, distractors)
    if grid is None:
        return None

    if not np.array_equal(main(grid), expected):
        return None

    bg = get_bg(grid)
    comps = components_mask(grid != bg, connectivity=4)
    pieces = [_build_piece(pts, bg, grid) for pts in comps]
    color_counts = [len(set(int(grid[r, c]) for r, c in pts if int(grid[r, c]) != bg)) for pts in comps]
    legend_candidates = [i for i, cnt in enumerate(color_counts) if cnt == max(color_counts)]
    if len(legend_candidates) != 1:
        return None
    if _count_valid_piece_subsets(pieces, legend_candidates[0], legend_map, K) != 1:
        return None
    if len(comps) == NUM_PIECES:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c D9 variant input")
