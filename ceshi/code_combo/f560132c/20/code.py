# Combo variant of f560132c (index 20)
# combo_spec: D1/0+D2/3+D9/3
# Name: 旋转匹配+同色精灵+干扰精灵

import math
import random
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and behavior trace
#    - Visible input: 5-6 disconnected non-bg sprites on a black field. Exactly one is
#      a multicolor legend sprite containing a 2x2 palette block. Among the remaining
#      monochrome sprites, exactly three are true puzzle pieces and 1-2 are distractors.
#      Of the four true pieces total, two non-legend pieces share the same body color
#      while having different D4 shape signatures.
#    - Canonical assembled state still has four pieces tiling one SxS square. The new
#      difference is that every true sprite is scattered in a non-identity D4 pose
#      relative to that canonical assembly, including the legend.
#    - Small example flow:
#         Original task:
#           components=4 -> legend found by multicolor patch -> assemble all 4 pieces.
#         This combo:
#           components=6 -> legend found by multicolor patch -> choose 3 of the 5
#           remaining sprites -> for each subset, search D4 placements -> keep the
#           visible 2x2 legend palette read in its displayed orientation -> accept the
#           unique subset+orientation combination that exact-covers the square.
#      Difference summary:
#        1. component count no longer identifies the real piece set,
#        2. body color no longer identifies every piece,
#        3. the legend body may rotate as a piece, but the visible 2x2 palette still
#           serves as the fixed orientation cue for output recoloring.
#
# b) Complexity calibration
#    - Distractors are limited to 1-2 so subset search stays human-comprehensible.
#    - The duplicated color is restricted to exactly one pair among the three monochrome
#      true non-legend pieces; this is enough to break color-based bookkeeping without
#      turning the grid into a same-color blob.
#    - Every true piece must have at least one visibly distinct non-identity D4 pose.
#      This avoids fake "rotation required" cases caused by symmetry.
#    - Distractors come from a small template family and are D4-randomized. They look
#      like plausible sprites rather than markers, but their shape keys never match any
#      true piece under D4.
#
# c) Hidden assumption audit of the original main()
#    - `k = round(sqrt(len(comps)))` breaks because len(comps) is now 5 or 6. Replaced
#      with the task constant `K=2`, which matches the visible legend palette size.
#    - `assert k*k == len(comps)` is invalid once distractors exist and is removed.
#    - "Assemble every component" is invalid. The solver must enumerate size-4 subsets
#      containing the legend and require exactly one valid assembly solution overall.
#    - Original exact-cover logic let the legend shape rotate while reading the palette
#      exactly as displayed in the input. That behavior is still appropriate here: the
#      palette is a visible orientation key, while the surrounding legend body is just
#      one more piece that may need reorientation to fit.
#    - Body-color uniqueness is no longer a valid invariant. The solver therefore keeps
#      relying only on connectivity, shape, and the legend palette.
#
# d) Placement legality
#    - All scattered patches are placed with `random_free_location` plus a border and
#      one-cell padding, so components remain disjoint under 4-connectivity.
#    - The legend palette is embedded strictly inside the oriented legend body, keeping
#      the legend as one connected component.
#    - Distractors are monochrome and separately placed, so they cannot merge with real
#      pieces or alter the legend connectivity.
#
# e) Visual observability
#    - Across examples, a human can notice three consistent cues: there are extra loose
#      sprites, two real loose sprites share a color, and the final pieces do not match
#      the input poses directly because the pieces must be reoriented to interlock.
#    - No rule depends on numeric color ids, component ordering, or hidden coordinate
#      conventions. The key evidence is visible in shape geometry and the 2x2 palette.
#
# f) Visual cleanliness
#    - Only one multicolor component exists. Everything else stays monochrome.
#    - Padding prevents touching bounding boxes, so repeated body colors remain easy to
#      separate by connectivity.
#    - Distractors are kept smaller than main pieces and are not overlaid with markers,
#      which preserves readability.
#
# g) Color-rule inferability
#    - Output recoloring is still controlled by the spatial 2x2 palette in the legend
#      exactly as it appears in the input. The new twist is purely in shape matching:
#      the legend body can rotate away from that displayed palette orientation.
#    - Palette colors remain four distinct non-zero colors, so each quadrant cue is
#      explicit and repeated across the four output regions.
#
# h) Anti-degeneration checks used by the generator
#    - `main(input) == expected` is necessary but insufficient, so the generator also
#      requires:
#        1. component count is 5 or 6,
#        2. exactly one pair of true non-legend pieces shares a body color,
#        3. every true piece is scattered in a visibly non-base D4 pose,
#        4. exactly one subset/orientation assembly solution exists overall once the
#           visible legend palette orientation is treated as fixed.
#    - A later evaluation script also compares this combo solver to the original solver;
#      rotated legends should usually make the original logic disagree.
#
# i) New solver assumptions and matching generator guarantees
#    - Solver assumes exactly one multicolor legend candidate has the highest distinct-
#      color count. Generator enforces that.
#    - Solver assumes the legend contains a visible 2x2 palette block. Generator always
#      embeds one and keeps it away from the legend bbox edge.
#    - Solver assumes exactly one full assembly solution exists across all size-4
#      subsets and D4 placements. Generator enumerates solutions under the fixed
#      displayed legend palette plus the "every true piece reorients" filter and
#      rejects ambiguous instances.
#    - Solver assumes every filled component can be treated as a piece shape after hole
#      filling. Generator keeps all shapes simply connected without interior holes.
# ---------------------------------------------------------------------------


# === CFG ===
K = 2
NUM_TRUE_PIECES = K * K
SIDE_CHOICES = [8, 10, 12]
BODY_COLOR_COUNT = 3
PALETTE_COLOR_COUNT = 4
TOTAL_NON_BG_COLORS = BODY_COLOR_COUNT + PALETTE_COLOR_COUNT
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


def _transform_array_d4(arr: np.ndarray, rot_k: int, flipped: bool) -> np.ndarray:
    out = np.asarray(arr)
    if flipped:
        out = np.fliplr(out)
    if rot_k:
        out = np.rot90(out, -int(rot_k))
    return np.asarray(out).copy()


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, k: int) -> np.ndarray:
    """
    Inside the legend component's bounding box, find a k×k color map block.
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
    Treat background holes fully enclosed inside the component bbox as part of the piece.
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


def _enumerate_exact_cover_solutions(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
    limit: int = 2,
) -> list[dict]:
    """
    Enumerate up to `limit` exact-cover assembly solutions.
    Each solution records the chosen placements plus the legend D4 transform.
    """
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)
    dims = [(h, w) for h, w in _factor_pairs(total) if h >= max_h and w >= max_w]
    if len(dims) != 1:
        return []

    H, W = dims[0]
    nbits = H * W
    all_mask = (1 << nbits) - 1
    other_pis = [i for i in range(len(pieces)) if i != legend_idx]

    placements_other: list[list[tuple[int, int, int, tuple[tuple[int, int], ...], int, bool]]] = []
    for pi in other_pis:
        opts: list[tuple[int, int, int, tuple[tuple[int, int], ...], int, bool]] = []
        for coords, ph, pw, ori_rot_k, ori_flipped in pieces[pi]["oris"]:
            for r0 in range(0, H - ph + 1):
                for c0 in range(0, W - pw + 1):
                    m = 0
                    for dr, dc in coords:
                        idx = (int(r0) + int(dr)) * int(W) + (int(c0) + int(dc))
                        m |= 1 << idx
                    opts.append((m, int(r0), int(c0), coords, int(ori_rot_k), bool(ori_flipped)))
        placements_other.append(opts)

    results: list[dict] = []

    def record_solution(
        chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...], int, bool]],
        rot_k: int,
        flipped: bool,
    ) -> None:
        if rot_k == 0 and not flipped:
            return
        for pi, pl in chosen.items():
            if pi == legend_idx:
                continue
            if pl[4] == 0 and not pl[5]:
                return
        piece_cell: dict[int, tuple[int, int]] = {}
        out = np.zeros((H, W), dtype=int)
        for pi, pl in chosen.items():
            _, r0, c0, coords, _, _ = pl
            rs = [r0 + dr for dr, _ in coords]
            cs = [c0 + dc for _, dc in coords]
            cr = float(sum(rs)) / float(len(rs))
            cc = float(sum(cs)) / float(len(cs))
            rr = int(cr * float(k) / float(H))
            cc2 = int(cc * float(k) / float(W))
            rr = max(0, min(int(k) - 1, rr))
            cc2 = max(0, min(int(k) - 1, cc2))
            piece_cell[pi] = (rr, cc2)
            col = int(legend_map[rr, cc2])
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        results.append(
            {
                "out": out,
                "chosen": {pi: chosen[pi] for pi in chosen},
                "legend_rot_k": int(rot_k),
                "legend_flipped": bool(flipped),
                "legend_map": np.asarray(legend_map, dtype=int).copy(),
                "piece_cell": piece_cell,
                "shape": (int(H), int(W)),
            }
        )

    def solve_with_fixed_legend(
        legend_coords: tuple[tuple[int, int], ...],
        rot_k: int,
        flipped: bool,
    ) -> None:
        legend_mask = 0
        for dr, dc in legend_coords:
            idx = int(dr) * int(W) + int(dc)
            legend_mask |= 1 << idx

        filtered = [[pl for pl in opts if (pl[0] & legend_mask) == 0] for opts in placements_other]
        remaining = set(range(len(other_pis)))
        chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...], int, bool]] = {
            legend_idx: (legend_mask, 0, 0, legend_coords, int(rot_k), bool(flipped))
        }

        def search(used: int) -> None:
            if len(results) >= limit:
                return
            if not remaining:
                if used == all_mask:
                    record_solution(chosen, rot_k, flipped)
                return
            empty = (~used) & all_mask
            if empty == 0:
                return
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
                return

            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, local_idx, opts_k in candidates:
                remaining.remove(local_idx)
                pi = other_pis[local_idx]
                for pl in opts_k:
                    chosen[pi] = pl
                    search(used | pl[0])
                    if len(results) >= limit:
                        break
                chosen.pop(pi, None)
                remaining.add(local_idx)
                if len(results) >= limit:
                    return

        search(legend_mask)

    for coords, ph, pw, rot_k, flipped in pieces[legend_idx]["oris"]:
        if ph > H or pw > W:
            continue
        solve_with_fixed_legend(coords, rot_k, flipped)
        if len(results) >= limit:
            break
    return results


def _try_assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray | None:
    sols = _enumerate_exact_cover_solutions(pieces, legend_idx, legend_map, k, limit=1)
    return None if not sols else sols[0]["out"]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _orientation_variants(mask: np.ndarray) -> list[tuple[np.ndarray, int, bool]]:
    base = _crop_mask(mask).astype(int)
    opts = []
    seen = set()
    for rot_k in range(4):
        for flipped in (False, True):
            oriented = _transform_array_d4(base, rot_k, flipped)
            pts = np.argwhere(oriented > 0)
            r0, c0, r1, c1 = bbox_coords(pts)
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append((crop, int(rot_k), bool(flipped)))
    return opts


def _orientation_key(mask: np.ndarray) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    crop = _crop_mask(mask.astype(bool))
    return tuple(map(int, crop.shape)), tuple(bool(x) for x in crop.ravel())


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = tuple(map(tuple, np.argwhere(_crop_mask(mask))))
    rel_norm, _, _ = _normalize_coords(pts)
    return min(coords for coords, _, _, _, _ in _dihedral_orientations(rel_norm))


def _pick_shared_color_pair(masks: list[np.ndarray]) -> tuple[int, int] | None:
    nonlegend = [1, 2, 3]
    sigs = {pi: _shape_signature(masks[pi]) for pi in nonlegend}
    pairs = [(i, j) for i in nonlegend for j in nonlegend if i < j and sigs[i] != sigs[j]]
    if not pairs:
        return None
    return random.choice(pairs)


def _count_bundle_solutions(pieces: list[dict], legend_idx: int, legend_map: np.ndarray, k: int, limit: int = 3) -> int:
    need = k * k - 1
    others = [i for i in range(len(pieces)) if i != legend_idx]
    total = 0
    for chosen_others in combinations(others, need):
        subset_indices = [legend_idx, *chosen_others]
        subset_pieces = [pieces[i] for i in subset_indices]
        total += len(_enumerate_exact_cover_solutions(subset_pieces, 0, legend_map, k, limit=max(1, limit - total)))
        if total >= limit:
            return total
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the true sprites into one rectangle, discarding extras and correcting every piece pose before recoloring.

    Input structure:
    - bg is the most common color.
    - One connected component is a multicolor legend piece containing a 2×2 palette block.
    - The other non-bg components are monochrome sprites: exactly three belong to the puzzle, while 1-2 are distractors.
    - Among the four true pieces, two non-legend sprites may share the same body color, and all true sprites can arrive rotated or mirrored.

    Rule:
    - Keep the legend and choose exactly three additional sprites whose shapes can exact-cover one tight rectangle under D4 transforms.
    - Reject candidate assemblies that would leave any selected true sprite in the same visible pose it had in the input; every real piece must be reoriented.
    - The legend body may rotate or mirror as part of the assembly, but its visible 2×2 palette is read exactly as displayed in the input.
    - Recolor the assembled pieces from that displayed palette according to the piece centroid's quadrant in the final 2×2 layout, and ignore leftover distractors.

    Key insight:
    - Color is not enough and component count is not enough. The solver must find the unique exact-cover subset by shape, and the "every true piece reorients" condition removes many symmetric false assemblies that would otherwise survive on a square board.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    k = K

    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) >= NUM_TRUE_PIECES, "grid must contain at least four non-background components"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    assert max_distinct > 1, "legend component must be multicolor"
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend component must be uniquely identified by color diversity"
    legend_idx = legend_candidates[0]
    legend_map = _extract_legend_map(g, comps[legend_idx], k)

    pieces = [_build_piece(pts, bg, g) for pts in comps]

    need = k * k - 1
    others = [i for i in range(len(pieces)) if i != legend_idx]
    solutions = []
    for chosen_others in combinations(others, need):
        subset_indices = [legend_idx, *chosen_others]
        subset_pieces = [pieces[i] for i in subset_indices]
        subset_solutions = _enumerate_exact_cover_solutions(subset_pieces, 0, legend_map, k, limit=2)
        for sol in subset_solutions:
            solutions.append(sol["out"])
            if len(solutions) > 1:
                break
        if len(solutions) > 1:
            break

    assert len(solutions) == 1, "there must be exactly one valid subset/orientation assembly"
    return solutions[0]


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
    return min(_shape_keys_from_mask(_crop_mask(mask)))


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
            oriented = random.choice(_orientation_variants(base))[0]
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


def _build_real_sprite_patches(
    masks: list[np.ndarray],
    body_colors: list[int],
    legend_map: np.ndarray,
) -> tuple[list[np.ndarray], list[dict]] | None:
    sprites = []
    meta = []
    for pi in range(NUM_TRUE_PIECES):
        base_mask = _crop_mask(masks[pi])
        base_key = _orientation_key(base_mask)
        variants = []
        for oriented, rot_k, flipped in _orientation_variants(base_mask):
            if _orientation_key(oriented) == base_key:
                continue
            variants.append((oriented, rot_k, flipped))
        if not variants:
            return None

        oriented, rot_k, flipped = random.choice(variants)
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        if pi == 0:
            spots = _palette_spots(oriented)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r : r + K, c : c + K] = _transform_array_d4(legend_map, rot_k, flipped)
        sprites.append(patch)
        meta.append(
            {
                "piece_index": pi,
                "rot_k": int(rot_k),
                "flipped": bool(flipped),
                "observed_key": _orientation_key(oriented),
                "base_key": base_key,
            }
        )
    return sprites, meta


def _scatter_all_patches(
    real_patches: list[np.ndarray],
    distractor_masks: list[np.ndarray],
    distractor_colors: list[int],
) -> np.ndarray | None:
    total = sum(int((patch != 0).sum()) for patch in real_patches)
    total += sum(int(mask.sum()) for mask in distractor_masks)
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None

    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    sprites = list(real_patches)
    for mask, color in zip(distractor_masks, distractor_colors):
        oriented = random.choice(_orientation_variants(mask))[0]
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
    if any(len(_orientation_variants(mask)) < 2 for mask in masks):
        return None

    qcells = []
    for mask in masks:
        pts = np.argwhere(mask)
        qcells.append((min(1, int(pts[:, 0].mean() * K / side)), min(1, int(pts[:, 1].mean() * K / side))))
    if len(set(qcells)) != NUM_TRUE_PIECES:
        return None

    shared_pair = _pick_shared_color_pair(masks)
    if shared_pair is None:
        return None

    colors = random.sample(range(1, 10), TOTAL_NON_BG_COLORS)
    body_pool = colors[:BODY_COLOR_COUNT]
    palette_colors = colors[BODY_COLOR_COUNT:]
    random.shuffle(body_pool)
    legend_color, shared_color, lone_color = body_pool
    body_colors = [legend_color, lone_color, lone_color, lone_color]
    for pi in shared_pair:
        body_colors[pi] = shared_color

    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)

    distractors = _make_distractors(masks)
    if distractors is None:
        return None

    built = _build_real_sprite_patches(masks, body_colors, legend_map)
    if built is None:
        return None
    real_patches, real_meta = built
    if not all(m["observed_key"] != m["base_key"] for m in real_meta):
        return None

    displayed_legend_map = _transform_array_d4(legend_map, real_meta[0]["rot_k"], real_meta[0]["flipped"])
    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        expected[mask] = displayed_legend_map[rr, cc]

    distractor_colors = [random.choice(body_colors[1:]) for _ in distractors]
    grid = _scatter_all_patches(real_patches, distractors, distractor_colors)
    if grid is None:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    bg = get_bg(grid)
    comps = components_mask(grid != bg, connectivity=4)
    if len(comps) not in (5, 6):
        return None
    pieces = [_build_piece(pts, bg, grid) for pts in comps]
    color_counts = [len(set(int(grid[r, c]) for r, c in pts if int(grid[r, c]) != bg)) for pts in comps]
    legend_candidates = [i for i, cnt in enumerate(color_counts) if cnt == max(color_counts)]
    if len(legend_candidates) != 1:
        return None
    extracted_map = _extract_legend_map(grid, comps[legend_candidates[0]], K)
    if _count_bundle_solutions(pieces, legend_candidates[0], extracted_map, K, limit=2) != 1:
        return None

    nonlegend_true_colors = [body_colors[i] for i in range(1, NUM_TRUE_PIECES)]
    counts = sorted(nonlegend_true_colors.count(c) for c in set(nonlegend_true_colors))
    if counts != [1, 2]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c combo variant input")
