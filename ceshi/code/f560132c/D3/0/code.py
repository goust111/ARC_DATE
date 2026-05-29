# Variant of f560132c
# Dimension: D3 — 目标结构：拼装目标的形状与组织 (dst)
# Change: 4个精灵拼装为一个矩形复合体，图例精灵的轮廓定义了该矩形的大小和各象限的分割方式... → 拼装为非矩形轮廓

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the D3 variant
# a) Concrete behavior:
#    The original task assembled 4 pieces into a solid square. Here the output bbox is still a tight square,
#    but some cells inside that bbox remain background, so the assembled object is visibly non-rectangular.
#    The generator uses one of four silhouette families:
#      - C: top bar + bottom bar + left bar
#      - H: left bar + right bar + middle bar
#      - U: left bar + right bar + bottom bar
#      - T: top bar + centered stem
#    The full silhouette is first built on an SxS canvas, then split by the 2x2 bbox quadrants into 4 connected
#    pieces. The legend is always the northwest quadrant piece, so it still doubles as a jigsaw piece and a color key.
#    Example trace:
#      - Old: total area implied a unique full rectangle; all 4 pieces had to cover every cell of that rectangle.
#      - New: total area matches one irregular target mask; the solver must cover only the occupied cells of that mask,
#        leaving the silhouette's missing cells as background in the final output.
#      - Difference: exact-cover search no longer ranges over full rectangles; it ranges over template masks with holes/
#        notches visible inside the cropped bbox.
# b) Complexity calibration:
#    Using only one irregular family would be too easy and visually repetitive. Using too many arbitrary silhouettes
#    would make inference noisy. The chosen family set keeps the freedom small and human-readable while still ensuring
#    that multiple training examples can show clearly different non-rectangular outlines.
# c) Original assumptions that break:
#      - `_assemble_exact_cover` assumed the destination is a full rectangle whose area equals the sum of piece areas.
#        This is false now because the bbox contains background cells.
#      - The original rectangle factorization check relied on `total = H*W`. That no longer holds.
#      - The old generator validated only square tilings; it must now validate candidate irregular masks instead.
#    The revised solver therefore enumerates a small catalog of target silhouettes and exact-covers only their occupied cells.
# d) Placement legality:
#    The generator still scatters cropped piece sprites with `random_free_location`, using bbox padding and borders so
#    components remain disjoint. The legend palette is embedded only in an interior 2x2 spot surrounded by body cells.
# e) Visual observability:
#    The D3 change is directly visible: outputs have zero-valued cutouts/notches inside the tight square bbox. The
#    recoloring rule remains spatial and visible because the four pieces still correspond to the 2x2 bbox quadrants.
# f) Visual cleanliness:
#    The quadrant split avoids overlaid markers or secondary annotations. Each piece stays monochrome except for the
#    legend's palette patch, and output background holes are large enough to read at a glance.
# g) Color inferability:
#    Colors still come from a visible 2x2 palette. No color-id ordering is used; piece colors in the output depend only
#    on which bbox quadrant the assembled piece occupies.
# h) Variant-specific solver assumptions introduced here:
#      - One component is uniquely multicolored and is the legend.
#      - The assembled silhouette belongs to the small family catalog below.
#      - The legend piece is the northwest sector of the target, so one valid placement anchors it at (0,0).
#    The generator enforces all three properties.


K = 2
NUM_PIECES = K * K
NUM_COLORS = 8
SIDE_CHOICES = [10, 12]
THICKNESS = 4
TARGET_FAMILIES = ("C", "H", "U", "T")
TARGET_SIDES = {
    "C": (10, 12),
    "H": (10, 12),
    "U": (10, 12),
    "T": (12,),
}
AREA_FRAC_LO = 0.08
AREA_FRAC_HI = 0.43
AREA_ABS_LO = 11
AREA_ABS_HI = 62
INPUT_SIZE_LO = 20
INPUT_SIZE_HI = 28
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
PARTITION_GROWTH_EXP = 1.35
PARTITION_FRONTIER_EXP = 0.25
PARTITION_SEED_JITTER = 0.25
PARTITION_CELL_NOISE = 0.1
GEN_RETRIES = 400


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
            out.append((c, int(h), int(w), int(rot_k), False))
        cur = _rot90(cur)[0]
    cur = _flip_h(coords)[0]
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, int(h), int(w), int(rot_k), True))
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
    hole_coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(holes))]
    filled = tuple(sorted(set(map(tuple, pts)).union(hole_coords)))
    return filled, int(holes.sum())


def _target_mask(family: str, side: int, thickness: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    if family == "C":
        mask[:thickness, :] = True
        mask[-thickness:, :] = True
        mask[:, :thickness] = True
    elif family == "H":
        mask[:, :thickness] = True
        mask[:, -thickness:] = True
        r0 = (side - thickness) // 2
        mask[r0 : r0 + thickness, :] = True
    elif family == "U":
        mask[:, :thickness] = True
        mask[:, -thickness:] = True
        mask[-thickness:, :] = True
    elif family == "T":
        mask[:thickness, :] = True
        c0 = (side - thickness) // 2
        mask[:, c0 : c0 + thickness] = True
    else:
        raise ValueError(f"unknown target family: {family}")
    return mask


def _candidate_target_infos(total: int, max_h: int, max_w: int) -> list[dict]:
    out = []
    for family in TARGET_FAMILIES:
        for side in TARGET_SIDES[family]:
            if side < max_h or side < max_w:
                continue
            mask = _target_mask(family, side, THICKNESS)
            if int(mask.sum()) != total:
                continue
            out.append({"family": family, "side": side, "mask": mask})
    return out


def _piece_quadrant(r0: int, c0: int, coords: tuple[tuple[int, int], ...], side: int) -> tuple[int, int]:
    rs = [r0 + dr for dr, _ in coords]
    cs = [c0 + dc for _, dc in coords]
    rr = min(K - 1, int((float(sum(rs)) / float(len(rs))) * K / side))
    cc = min(K - 1, int((float(sum(cs)) / float(len(cs))) * K / side))
    return int(rr), int(cc)


def _solve_on_target(
    pieces: list[dict],
    legend_idx: int,
    target_mask: np.ndarray,
    solution_limit: int = 1,
) -> list[dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]]]:
    side = int(target_mask.shape[0])
    target_cells = [tuple(map(int, rc)) for rc in np.argwhere(target_mask)]
    pos_to_bit = {rc: i for i, rc in enumerate(target_cells)}
    all_mask = (1 << len(target_cells)) - 1
    other_pis = [i for i in range(len(pieces)) if i != legend_idx]

    placements_other: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
    for pi in other_pis:
        opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
        for coords, ph, pw, _, _ in pieces[pi]["oris"]:
            for r0 in range(0, side - ph + 1):
                for c0 in range(0, side - pw + 1):
                    m = 0
                    ok = True
                    for dr, dc in coords:
                        abs_rc = (int(r0 + dr), int(c0 + dc))
                        bit = pos_to_bit.get(abs_rc)
                        if bit is None:
                            ok = False
                            break
                        m |= 1 << bit
                    if ok:
                        opts.append((m, int(r0), int(c0), coords))
        placements_other.append(opts)

    solutions: list[dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []

    for coords, _, _, _, _ in pieces[legend_idx]["oris"]:
        legend_mask = 0
        ok = True
        for dr, dc in coords:
            bit = pos_to_bit.get((int(dr), int(dc)))
            if bit is None:
                ok = False
                break
            legend_mask |= 1 << bit
        if not ok:
            continue

        filtered: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
        for opts in placements_other:
            filtered.append([pl for pl in opts if (pl[0] & legend_mask) == 0])

        remaining = set(range(len(other_pis)))
        chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] = {
            legend_idx: (legend_mask, 0, 0, coords)
        }

        def search(used: int) -> None:
            if len(solutions) >= solution_limit:
                return
            if not remaining:
                if used == all_mask:
                    solutions.append(chosen.copy())
                return

            empty = all_mask & ~used
            if empty == 0:
                return
            cell = (empty & -empty).bit_length() - 1

            candidates = []
            for local_idx in sorted(remaining):
                opts_k = []
                for pl in filtered[local_idx]:
                    m = pl[0]
                    if ((m >> cell) & 1) and (m & used) == 0:
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
                    if len(solutions) >= solution_limit:
                        return
                chosen.pop(pi, None)
                remaining.add(local_idx)

        search(legend_mask)
        if len(solutions) >= solution_limit:
            break

    return solutions


def _find_assemblies(
    pieces: list[dict],
    legend_idx: int,
    solution_limit: int = 1,
) -> list[tuple[dict, dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]]]]:
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)
    targets = _candidate_target_infos(total, max_h, max_w)
    assert targets, "no candidate non-rectangular target matches the pieces"

    out = []
    for info in targets:
        sols = _solve_on_target(pieces, legend_idx, info["mask"], solution_limit=max(1, solution_limit - len(out)))
        for sol in sols:
            out.append((info, sol))
            if len(out) >= solution_limit:
                return out
    return out


def _assemble_irregular_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray:
    assemblies = _find_assemblies(pieces, legend_idx, solution_limit=1)
    assert assemblies, "exact-cover assembly failed for irregular target"
    info, chosen = assemblies[0]
    side = int(info["side"])
    out = np.zeros((side, side), dtype=int)
    for pi, pl in chosen.items():
        _, r0, c0, coords = pl
        rr, cc = _piece_quadrant(r0, c0, coords, side)
        col = int(legend_map[rr, cc])
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into an irregular silhouette, recoloring each piece by a legend sprite's 2x2 map.

    Input structure:
    - bg = the most common color.
    - Four disconnected non-bg sprites appear on the canvas as separate monochrome pieces.
    - Exactly one piece is the legend: it contains a multicolor 2x2 patch embedded inside its body.

    Rule:
    - Reassemble the four pieces under D4 transformations into a tight non-rectangular target silhouette, not a full rectangle.
    - The valid destination shape is inferred from the piece set and must match one irregular outline family.
    - Once assembled, color each piece by the legend-map entry of the bbox quadrant where that piece lands; cells outside the silhouette stay background.

    Key insight: The legend still acts as both a jigsaw piece and a color key, but the search target is now an irregular mask, so the solver must match pieces to a silhouette with visible cutouts rather than simply filling a rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

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
        pieces.append({"oris": _dihedral_orientations(rel_norm), "area": len(filled_abs)})

    return _assemble_irregular_cover(pieces, legend_idx, legend_map, k)


def _area_bounds(side: int) -> tuple[int, int]:
    total = side * side
    lo = max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total))
    hi = min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))
    return int(lo), int(hi)


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


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(int)
    opts = []
    seen = set()
    for k_rot in range(4):
        rot = np.rot90(arr, k_rot)
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
                spots.append((int(r), int(c)))
    return spots


def _partition_target(target_mask: np.ndarray) -> list[np.ndarray] | None:
    side = int(target_mask.shape[0])
    mid = side // 2
    labels = np.full(target_mask.shape, -1, dtype=int)
    occupied = set(map(tuple, np.argwhere(target_mask)))

    legend_core = [(r, c) for r in range(THICKNESS) for c in range(THICKNESS) if target_mask[r, c]]
    if len(legend_core) != THICKNESS * THICKNESS:
        return None

    for r, c in legend_core:
        labels[r, c] = 0
    available = occupied - set(legend_core)

    quadrant_windows = {
        1: (0, mid, mid, side),
        2: (mid, side, 0, mid),
        3: (mid, side, mid, side),
    }
    anchors = {
        0: (0, 0),
        1: (0, side - 1),
        2: (side - 1, 0),
        3: (side - 1, side - 1),
    }

    def pick_seed(idx: int) -> tuple[int, int] | None:
        r0, r1, c0, c1 = quadrant_windows[idx]
        pool = [p for p in available if r0 <= p[0] < r1 and c0 <= p[1] < c1]
        if not pool:
            pool = list(available)
        if not pool:
            return None
        ar, ac = anchors[idx]
        return min(pool, key=lambda p: abs(p[0] - ar) + abs(p[1] - ac) + random.random() * PARTITION_SEED_JITTER)

    for idx in (1, 2, 3):
        seed = pick_seed(idx)
        if seed is None:
            return None
        labels[seed] = idx
        available.remove(seed)

    areas = [int(np.sum(labels == i)) for i in range(NUM_PIECES)]
    frontiers = [set() for _ in range(NUM_PIECES)]

    def add_frontier(i: int, r: int, c: int) -> None:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < side and 0 <= cc < side and target_mask[rr, cc] and labels[rr, cc] < 0:
                frontiers[i].add((int(rr), int(cc)))

    for i in range(NUM_PIECES):
        for r, c in zip(*np.where(labels == i)):
            add_frontier(i, int(r), int(c))

    unassigned = int(target_mask.sum()) - sum(areas)
    while unassigned > 0:
        candidates = []
        for i in range(NUM_PIECES):
            frontiers[i] = {p for p in frontiers[i] if labels[p] < 0}
            if frontiers[i]:
                candidates.append(i)
        if not candidates:
            return None

        weights = [
            (1.0 / (areas[i] ** PARTITION_GROWTH_EXP)) * (len(frontiers[i]) ** PARTITION_FRONTIER_EXP)
            for i in candidates
        ]
        idx = random.choices(candidates, weights=weights, k=1)[0]
        ar, ac = anchors[idx]
        cell = min(
            frontiers[idx],
            key=lambda p: abs(p[0] - ar) + abs(p[1] - ac) + random.random() * PARTITION_CELL_NOISE,
        )
        frontiers[idx].discard(cell)
        labels[cell] = idx
        areas[idx] += 1
        unassigned -= 1
        add_frontier(idx, cell[0], cell[1])

    return [(labels == i) for i in range(NUM_PIECES)]


def _scatter_pieces(masks: list[np.ndarray], body_colors: list[int], legend_map: np.ndarray, side: int) -> np.ndarray | None:
    total = sum(int(m.sum()) for m in masks)
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None

    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    order = random.sample(range(NUM_PIECES), NUM_PIECES)
    for pi in order:
        oriented = _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        if pi == 0:
            spots = _palette_spots(oriented)
            if not spots:
                return None
            rr, cc = random.choice(spots)
            patch[rr : rr + K, cc : cc + K] = legend_map
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


def _expected_output(masks: list[np.ndarray], legend_map: np.ndarray) -> np.ndarray:
    side = int(masks[0].shape[0])
    out = np.zeros((side, side), dtype=int)
    for pi in range(NUM_PIECES):
        pts = np.argwhere(masks[pi])
        rr = min(K - 1, int(pts[:, 0].mean() * K / side))
        cc = min(K - 1, int(pts[:, 1].mean() * K / side))
        out[masks[pi]] = int(legend_map[rr, cc])
    return out


def _candidate_masks_for_generation() -> list[tuple[str, int, np.ndarray]]:
    out = []
    for family in TARGET_FAMILIES:
        for side in TARGET_SIDES[family]:
            out.append((family, side, _target_mask(family, side, THICKNESS)))
    return out


GEN_TARGETS = _candidate_masks_for_generation()


def _gen_attempt() -> np.ndarray | None:
    family, side, target = random.choice(GEN_TARGETS)
    masks = _partition_target(target)
    if masks is None:
        return None
    min_area, max_area = _area_bounds(side)

    qcells = []
    for mask in masks:
        area = int(mask.sum())
        if area < min_area or area > max_area:
            return None
        if len(components_mask(mask, connectivity=4)) != 1:
            return None
        if _has_hole(mask):
            return None
        pts = np.argwhere(mask)
        qcells.append((min(K - 1, int(pts[:, 0].mean() * K / side)), min(K - 1, int(pts[:, 1].mean() * K / side))))

    if qcells != [(0, 0), (0, 1), (1, 0), (1, 1)]:
        return None

    if not _palette_spots(_crop_mask(masks[0])):
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_PIECES]
    palette_colors = colors[NUM_PIECES:]
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)
    expected = _expected_output(masks, legend_map)

    grid = _scatter_pieces(masks, body_colors, legend_map, side)
    if grid is None:
        return None

    if not np.array_equal(main(grid), expected):
        return None

    pieces = []
    bg = get_bg(grid)
    comps = components_mask(grid != bg, connectivity=4)
    comp_colors = [set(int(grid[r, c]) for r, c in pts if int(grid[r, c]) != bg) for pts in comps]
    legend_idx = max(range(len(comps)), key=lambda i: len(comp_colors[i]))
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(grid, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        pieces.append({"oris": _dihedral_orientations(rel_norm), "area": len(filled_abs)})
    assemblies = _find_assemblies(pieces, legend_idx, solution_limit=2)
    if len(assemblies) != 1:
        return None

    return grid


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c D3 input")
