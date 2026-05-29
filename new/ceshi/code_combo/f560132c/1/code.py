# Combo variant of f560132c (index 1)
# combo_spec: D1/0+D9/3
# Name: 旋转匹配+干扰精灵

import math
import random
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for combo D1/0 + D9/3
#
# a) Concrete behavior and flow impact
#    - Base task: four sprites are exactly the four true pieces. The multicolor one is
#      the legend; the other three are monochrome. Solver can assemble all four at
#      once, and rotations/reflections are merely allowed.
#    - Combo task: input has 5-6 connected sprites. Exactly four belong to the final
#      square; 1-2 are distractors that never appear in the output. Among the four
#      true sprites, every visible input orientation is deliberately different from
#      every target quadrant orientation, so shape matching now requires both
#      subset selection and non-trivial D4 reorientation.
#    - Small trace:
#         base input: [legend, A, B, C] -> use all four -> place them.
#         combo input: [legend', A', B', C', X, maybe Y]
#           1. identify the unique multicolor legend
#           2. test each 3-of-(others) subset with the legend
#           3. only one subset exact-covers a square
#           4. within that subset, the chosen placement for each real piece uses an
#              orientation different from the sprite's visible input orientation
#           5. recolor by the 2x2 palette and crop.
#      Difference: the old "all components are pieces" assumption and the old "input
#      orientation may already fit a quadrant" convenience both disappear.
#
# b) Complexity calibration
#    - Distractors are limited to 1-2. That keeps subset search small enough for ARC
#      while still making the extra sprites visually obvious.
#    - Real pieces are still only four, so the task remains readable from examples.
#      The added difficulty comes from coupling two moderate decisions: which sprites
#      are real, and how each selected sprite must be reoriented.
#    - Generator rejects degenerate samples where a real sprite's visible shape already
#      equals any target quadrant shape. This makes the D1 change observable instead
#      of being a silent "solver can rotate if needed" allowance.
#
# c) Hidden assumptions in the original main()
#    - `k = round(sqrt(len(comps)))` is invalid once extras exist. The combo keeps
#      `K=2`, inferred from the visible 2x2 legend palette rather than component count.
#    - `assert k*k == len(comps)` is invalid because 5-6 components are now expected.
#    - `assemble all pieces` is invalid. Main must enumerate legend + 3-piece subsets.
#    - `_assemble_exact_cover` in the base code assumes the provided piece set is
#      valid and uses assertions. Here many candidate subsets are intentionally wrong,
#      so solver needs a non-throwing exact-cover attempt helper.
#    - Rotational search is no longer optional generator noise. The generator now
#      guarantees that the accepted subset found by exact cover reorients every real
#      sprite relative to its visible input shape.
#
# d) Placement legality
#    - True pieces and distractors are all placed through `random_free_location` with
#      the same border/padding constraints, so they stay disconnected under 4-connectivity.
#    - The legend palette remains embedded strictly inside legend body cells before any
#      rotation/flip. After the whole patch is transformed, it is still part of a
#      single connected component and still visible as a 2x2 color key.
#
# e) Visual observability
#    - A human sees one multicolor sprite, several monochrome sprites, and at least one
#      leftover shape that never participates in the output. That already advertises
#      the D9 portion.
#    - Across examples, the output always contains four recolored regions, while the
#      input true pieces visibly appear "turned" relative to their final shapes. That
#      makes the D1 portion visible without relying on hidden metadata like piece order.
#
# f) Visual cleanliness
#    - Distractors are small monochrome polyominoes and stay spatially separated from
#      the true pieces. They do not overlap or touch the legend palette.
#    - Colors are still limited to the original eight non-background colors, so the
#      extra complexity comes from geometry, not from palette clutter.
#
# g) Color inference
#    - Recoloring still comes only from the visible 2x2 palette inside the legend.
#      Distractors never introduce extra output colors or alternate recoloring rules.
#    - Because every final region takes one palette cell by assembled centroid, color
#      logic remains structured and learnable from a few examples.
#
# h) Generator-side anti-degeneration checks
#    - Every candidate sample is verified by `main(input) == expected`.
#    - Generator also parses the finished input back into components and requires:
#         * exactly one valid 4-piece subset (legend + 3 others)
#         * that valid subset's recovered assembly reorients all four real sprites
#           relative to their visible input shape
#         * at least one distractor exists
#      This prevents regression back to either the base task or the D9-only variant.
#
# i) New solver assumptions and matching generator guarantees
#    - Solver assumes the legend is the unique component with the most distinct colors.
#      Generator enforces that by keeping all non-legend sprites monochrome.
#    - Solver assumes there is exactly one valid subset. Generator rejects samples with
#      zero or multiple exact-cover subsets.
#    - Solver assumes the legend palette is 2x2. Generator always embeds a 2x2 block.
#    - Solver does not explicitly assert "all true pieces need rotation"; instead the
#      generator guarantees that property and the post-generation metric checks it.
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


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, k: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert k > 0 and hh >= k and ww >= k, "legend component must fit kxk map"
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
    assert best_rc is not None, "legend component must contain a valid kxk color map window"
    rr, cc = best_rc
    return np.asarray(sub[rr : rr + k, cc : cc + k], dtype=int).copy()


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
    base, _, _ = _normalize_coords(rel)
    return {"base": base, "oris": _dihedral_orientations(base), "area": len(filled_abs)}


def _solve_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> dict | None:
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

    transformed_count = sum(1 for pi, pl in chosen.items() if pl[3] != pieces[pi]["base"])
    return {
        "out": out,
        "chosen": chosen,
        "dims": (int(H), int(W)),
        "transformed_count": int(transformed_count),
    }


def _valid_subset_solutions(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> list[tuple[list[int], dict]]:
    need = k * k - 1
    others = [i for i in range(len(pieces)) if i != legend_idx]
    out = []
    for chosen_others in combinations(others, need):
        subset_indices = [legend_idx, *chosen_others]
        subset_pieces = [pieces[i] for i in subset_indices]
        sol = _solve_exact_cover(subset_pieces, 0, legend_map, k)
        if sol is not None:
            out.append((subset_indices, sol))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the true jigsaw sprites into one rectangle, ignoring decoys and undoing their visible reorientation.

    Input structure:
    - bg is the most common color.
    - One connected component is the multicolor legend piece containing a 2x2 palette block.
    - The other non-bg components are monochrome sprites: exactly three are real puzzle pieces and
      1-2 more are distractors that never belong to the final assembly.

    Rule:
    - Keep the legend piece and choose exactly three additional sprites so that those four pieces can
      exact-cover one tight rectangle under rotations/reflections.
    - Discard the extra sprites, then reorient the selected pieces as needed to complete the tiling.
    - Recolor each selected piece by the legend's 2x2 palette according to the piece centroid's cell
      in the assembled rectangle, and output the cropped assembly.

    Key insight:
    - The task is now a coupled search: first isolate the unique 4-piece subset that can really tile,
      then use D4 shape matching to undo the visible turning of those true sprites before applying the palette.
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
    solutions = _valid_subset_solutions(pieces, legend_idx, legend_map, k)
    assert len(solutions) == 1, "there must be exactly one valid 4-piece assembly subset"
    return solutions[0][1]["out"]


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
GEN_RETRIES = 700
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


def _exact_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    crop = _crop_mask(mask.astype(bool))
    return crop.shape, tuple(int(x) for x in crop.astype(int).ravel())


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


def _unique_patch_variants(patch: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    opts = []
    seen = set()
    for rot_k in range(4):
        rot = np.rot90(patch, rot_k)
        for oriented in (rot, np.fliplr(rot)):
            mask = oriented != 0
            r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
            crop_patch = oriented[r0 : r1 + 1, c0 : c1 + 1].copy()
            crop_mask = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
            key = (_exact_shape_key(crop_mask), tuple(int(x) for x in crop_patch[crop_mask].ravel()))
            if key not in seen:
                seen.add(key)
                opts.append((crop_patch, crop_mask))
    return opts


def _extract_patch_legend_map(patch: np.ndarray, k: int) -> np.ndarray:
    hh, ww = patch.shape
    assert hh >= k and ww >= k
    best = None
    best_rc = None
    for rr in range(hh - k + 1):
        for cc in range(ww - k + 1):
            window = patch[rr : rr + k, cc : cc + k]
            distinct = int(len(set(map(int, window.ravel()))))
            key = (distinct, -rr, -cc)
            if best is None or key > best:
                best = key
                best_rc = (rr, cc)
    assert best_rc is not None
    rr, cc = best_rc
    return np.asarray(patch[rr : rr + k, cc : cc + k], dtype=int).copy()


def _palette_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(1, h - K):
        for c in range(1, w - K):
            if mask[r : r + K, c : c + K].all() and mask[r - 1 : r + K + 1, c - 1 : c + K + 1].all():
                spots.append((r, c))
    return spots


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
            key = _canonical_shape_key(base)
            if key in true_keys or key in used_keys:
                continue
            if _has_hole(base):
                continue
            distractors.append(base)
            used_keys.add(key)
            placed = True
            break
        if not placed:
            return None
    return distractors


def _choose_true_sprite_patches(
    masks: list[np.ndarray],
    body_colors: list[int],
    legend_map: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    assembled_exact_keys = {_exact_shape_key(_crop_mask(mask)) for mask in masks}

    base_patches = []
    for pi in range(NUM_PIECES):
        crop = _crop_mask(masks[pi])
        patch = np.zeros(crop.shape, dtype=int)
        patch[crop] = body_colors[pi]
        if pi == 0:
            spots = _palette_spots(crop)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r : r + K, c : c + K] = legend_map
        base_patches.append(patch)

    chosen = []
    for patch in base_patches:
        candidates = []
        for oriented_patch, oriented_mask in _unique_patch_variants(patch):
            if _exact_shape_key(oriented_mask) in assembled_exact_keys:
                continue
            candidates.append(oriented_patch)
        if not candidates:
            return None
        chosen.append(random.choice(candidates))
    return chosen, _extract_patch_legend_map(chosen[0], K)


def _scatter_patches(patches: list[np.ndarray], distractor_masks: list[np.ndarray]) -> np.ndarray | None:
    total = sum(int((patch != 0).sum()) for patch in patches) + sum(int(mask.sum()) for mask in distractor_masks)
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None

    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    sprites = list(patches)
    true_body_colors = [int(np.bincount(patch[patch != 0]).argmax()) for patch in patches]
    distractor_colors = [random.choice(true_body_colors) for _ in distractor_masks]
    for mask, color in zip(distractor_masks, distractor_colors):
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(color)
        variants = _unique_patch_variants(patch)
        sprites.append(random.choice(variants)[0])

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

    true_info = _choose_true_sprite_patches(masks, body_colors, legend_map)
    if true_info is None:
        return None
    true_patches, visible_legend_map = true_info

    distractors = _make_distractors(masks)
    if distractors is None:
        return None

    grid = _scatter_patches(true_patches, distractors)
    if grid is None:
        return None

    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        expected[mask] = visible_legend_map[rr, cc]
    if not np.array_equal(main(grid), expected):
        return None

    bg = get_bg(grid)
    comps = components_mask(grid != bg, connectivity=4)
    if len(comps) == NUM_PIECES:
        return None

    comp_colors = []
    for pts in comps:
        cols = set(int(grid[r, c]) for r, c in pts if int(grid[r, c]) != bg)
        comp_colors.append(cols)
    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    if len(legend_candidates) != 1:
        return None

    legend_idx = legend_candidates[0]
    pieces = [_build_piece(pts, bg, grid) for pts in comps]
    sols = _valid_subset_solutions(pieces, legend_idx, visible_legend_map, K)
    if len(sols) != 1:
        return None
    if sols[0][1]["transformed_count"] != NUM_PIECES:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c combo variant input")
