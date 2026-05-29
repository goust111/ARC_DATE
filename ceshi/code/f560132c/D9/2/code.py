# Variant of f560132c
# Dimension: D9 — 精灵数量与象限划分 (meta)
# Change: 4个精灵对应2x2=4个象限，每个象限恰好一个精灵... → 同象限多精灵合并

"""
Design analysis for the D9 variant.

1. Concrete behavior and trace
   - Original behavior:
     the input has exactly 4 connected sprites, one of them is the multicolor legend,
     and every connected sprite becomes one assembled region. `main()` can therefore
     infer `k=2` from `sqrt(len(comps))` and color each placed sprite directly by its
     own centroid cell.
   - Variant behavior:
     the assembled square still contains exactly 4 final regions, still recolored by a
     2x2 legend map, and the legend sprite is still one of those final regions. The new
     difference is that 1 or 2 non-legend final regions are cut into two separate input
     sprites before scattering. The input therefore has 5 or 6 sprites total, but the
     output still has only 4 centroid-defined color regions after those split sprites are
     merged back into their parent region.
   - Small trace:
     suppose the bottom-left final region is split into two disconnected input sprites A
     and B. Old logic would treat A and B as two independent pieces, so it would either
     fail immediately on `sqrt(5)` / `sqrt(6)`, or if forced to continue, it would color
     A and B by their own centroids. The new logic instead enumerates groupings of the
     4 or 5 non-legend sprites into exactly 3 merged regions, exact-covers the square,
     and colors A∪B by the centroid of the merged union.

2. Complexity calibration
   - The visual change is immediate: there are now 5-6 scattered sprites instead of 4,
     while the output remains a 4-color assembled square.
   - To make the merge behavior matter, each split is accepted only when at least one
     child fragment lands in a different 2x2 legend cell than its parent union. This
     prevents the variant from degenerating into "just more pieces with the same color".
   - The grouping freedom stays controlled: only 1 or 2 of the 3 non-legend regions may
     split, so the solver only needs to recover 3 merged non-legend groups of size 1 or 2.

3. Old assumptions that break and the required fixes
   - `k = round(sqrt(len(comps)))` is no longer valid. With 5 or 6 components, the count
     is not a perfect square even though the legend map is still 2x2. The solver now uses
     the fixed 2x2 palette size for this variant.
   - "one connected component = one final recolored region" is false. The new solver must
     partition the non-legend components into 3 groups and evaluate the centroid of each
     merged union, not of each raw component.
   - The original exact-cover leaf colored pieces one-by-one. That was harmless before
     because every piece represented one final region. In the variant, that would wrongly
     assign different colors to fragments of the same merged region. The new leaf stage
     first validates group connectivity and distinct merged centroid cells, then paints
     every fragment in a group with the same group color.

4. Placement legality
   - Generation is still output-first: first build 4 connected final regions that tile
     one square exactly, then split 1-2 non-legend regions into connected subregions, and
     finally scatter the resulting 5-6 sprites with non-overlap and padding.
   - The legend region is never split. This keeps the legend uniquely identifiable as the
     only multi-color component and avoids introducing a second coupled mechanism unrelated
     to D9.
   - The solver still anchors the legend region at the assembled square's top-left corner.
     The generator preserves that original invariant.

5. Visual observability
   - Across samples, a human sees that some output color regions appear to have been built
     from two differently colored input sprites, because the input contains more sprites
     than the 4 legend cells.
   - The repeated 2x2 legend structure remains the same, so the only new concept to infer
     is "some same-output regions come from merging two fragments".

6. Visual cleanliness and color inferability
   - The palette still uses exactly 4 colors, one per legend cell. No color-id ordering is
     used anywhere; only spatial role in the assembled square matters.
   - With 6 total sprites there are not enough ARC nonzero colors to keep 6 body colors and
     4 palette colors all distinct. The generator therefore keeps the 4 palette colors
     distinct and disjoint from body colors, while allowing one body color repeat among
     spatially separated non-legend sprites when needed. This does not encode hidden data.

7. New solver assumptions and generator guarantees
   - Assumption: there are 5 or 6 components total, with exactly one multi-color legend.
     Guarantee: generation always produces 1 legend + 4 or 5 monochrome body sprites.
   - Assumption: the correct grouping has exactly 4 final merged regions, the legend plus
     3 non-legend groups of size 1 or 2.
     Guarantee: generation splits only 1 or 2 non-legend regions.
   - Assumption: every valid merged region is 4-connected and the 4 merged centroids map
     to the 4 distinct cells of the 2x2 legend.
     Guarantee: generation starts from 4 connected parent regions with distinct cells and
     only cuts them into connected fragments whose union is exactly the parent region.
"""

import math
import random
import sys

import numpy as np

SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

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
    hole_coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(holes))]
    filled = tuple(sorted(set(map(tuple, pts)).union(hole_coords)))
    return filled, int(holes.sum())


def _coords_connected(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    coords_set = set(coords)
    st = [coords[0]]
    seen = {coords[0]}
    while st:
        r, c = st.pop()
        for rr, cc in neighbors4(r, c):
            nxt = (rr, cc)
            if nxt in coords_set and nxt not in seen:
                seen.add(nxt)
                st.append(nxt)
    return len(seen) == len(coords_set)


def _cell_from_coords(coords: list[tuple[int, int]], H: int, W: int, k: int) -> tuple[int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    cr = float(sum(rs)) / float(len(rs))
    cc = float(sum(cs)) / float(len(cs))
    rr = int(cr * float(k) / float(H))
    cc2 = int(cc * float(k) / float(W))
    rr = max(0, min(int(k) - 1, rr))
    cc2 = max(0, min(int(k) - 1, cc2))
    return rr, cc2


def _enumerate_nonlegend_groupings(items: list[int], groups_left: int) -> list[tuple[tuple[int, ...], ...]]:
    if groups_left == 0:
        return [tuple()] if not items else []
    if not items:
        return []
    if len(items) < groups_left or len(items) > groups_left * 2:
        return []

    first = items[0]
    out: list[tuple[tuple[int, ...], ...]] = []

    for tail in _enumerate_nonlegend_groupings(items[1:], groups_left - 1):
        out.append(((first,),) + tail)

    for j in range(1, len(items)):
        rest = items[1:j] + items[j + 1 :]
        for tail in _enumerate_nonlegend_groupings(rest, groups_left - 1):
            out.append(((first, items[j]),) + tail)
    return out


def _enumerate_groupings(n_pieces: int, legend_idx: int) -> list[tuple[tuple[int, ...], ...]]:
    others = [i for i in range(n_pieces) if i != legend_idx]
    raw = _enumerate_nonlegend_groupings(others, 3)
    out = []
    for grouping in raw:
        out.append(((legend_idx,),) + tuple(sorted(tuple(sorted(g)) for g in grouping)))
    return sorted(set(out))


def _extract_scene(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict], int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) in (5, 6), "variant expects 5 or 6 input components"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1 and max_distinct > 1, "legend component must be uniquely multicolor"
    legend_idx = legend_candidates[0]

    k = 2
    legend_map = _extract_legend_map(g, comps[legend_idx], k)

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        pieces.append({"oris": _dihedral_orientations(rel_norm), "area": len(filled_abs)})
    return g, legend_idx, pieces, k, legend_map


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
    *,
    return_meta: bool = False,
):
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
    placements_other: dict[int, list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = {}
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
        placements_other[pi] = opts

    groupings = _enumerate_groupings(len(pieces), legend_idx)

    for legend_coords, ph, pw, _, _ in pieces[legend_idx]["oris"]:
        assert ph <= H and pw <= W, "legend orientation must fit the unique output rectangle"
        legend_mask = 0
        for dr, dc in legend_coords:
            idx = int(dr) * int(W) + int(dc)
            legend_mask |= 1 << idx

        filtered = {
            pi: [pl for pl in placements_other[pi] if (pl[0] & legend_mask) == 0]
            for pi in other_pis
        }

        for grouping in groupings:
            chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] = {
                legend_idx: (legend_mask, 0, 0, legend_coords)
            }
            remaining = set(other_pis)

            def finalize_solution() -> tuple[np.ndarray, dict] | None:
                group_cells: dict[int, tuple[int, int]] = {}
                seen_cells: set[tuple[int, int]] = set()
                for gi, group in enumerate(grouping):
                    merged_coords: list[tuple[int, int]] = []
                    for pi in group:
                        _, r0, c0, coords = chosen[pi]
                        merged_coords.extend((r0 + dr, c0 + dc) for dr, dc in coords)
                    if not _coords_connected(merged_coords):
                        return None
                    cell = _cell_from_coords(merged_coords, H, W, k)
                    if cell in seen_cells:
                        return None
                    seen_cells.add(cell)
                    group_cells[gi] = cell

                out = np.zeros((H, W), dtype=int)
                for gi, group in enumerate(grouping):
                    rr, cc = group_cells[gi]
                    col = int(legend_map[rr, cc])
                    for pi in group:
                        _, r0, c0, coords = chosen[pi]
                        for dr, dc in coords:
                            out[r0 + dr, c0 + dc] = col

                meta = {
                    "shape": (H, W),
                    "grouping": grouping,
                    "group_cells": group_cells,
                    "chosen": chosen.copy(),
                }
                return out, meta

            def search(used: int):
                if not remaining:
                    if used != all_mask:
                        return None
                    return finalize_solution()

                empty = (~used) & all_mask
                if empty == 0:
                    return None
                cell = (empty & -empty).bit_length() - 1

                candidates = []
                for pi in sorted(remaining):
                    opts_pi = []
                    for pl in filtered[pi]:
                        m = pl[0]
                        if ((m >> cell) & 1) and (m & used) == 0:
                            opts_pi.append(pl)
                    if opts_pi:
                        candidates.append((len(opts_pi), pi, opts_pi))
                if not candidates:
                    return None

                candidates.sort(key=lambda x: (x[0], x[1]))
                for _, pi, opts_pi in candidates:
                    remaining.remove(pi)
                    for pl in opts_pi:
                        chosen[pi] = pl
                        solved = search(used | pl[0])
                        if solved is not None:
                            return solved
                    chosen.pop(pi, None)
                    remaining.add(pi)
                return None

            solved = search(legend_mask)
            if solved is not None:
                out, meta = solved
                return (out, meta) if return_meta else out

    raise AssertionError("exact-cover assembly failed: no valid grouped tiling")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into one rectangle, merging some same-output regions before recoloring.

    Input structure:
    - bg = most common color.
    - There are 5 or 6 non-bg components: one multicolor legend sprite plus 4 or 5
      monochrome body sprites.
    - The legend still contains a 2x2 palette patch, but 1 or 2 non-legend final regions
      have been split into two separate input sprites.

    Rule:
    - Tile all input sprites into the unique full rectangle using rotations/reflections.
    - Among the non-legend sprites, infer which ones should merge so that the assembled
      rectangle represents exactly 4 connected final regions in the 2x2 legend layout.
    - Recolor each merged region by the legend color at the merged region's centroid cell.

    Key insight:
    - The extra sprites are not extra output regions. They are fragments of existing
      legend-cell regions, so color must be assigned after grouping by merged unions, not
      directly per connected input component.
    """
    _, legend_idx, pieces, k, legend_map = _extract_scene(grid)
    return _assemble_exact_cover(pieces, legend_idx, legend_map, k)


# --------------- generator (o5x) ---------------

K = 2
SIDE_CHOICES = [8, 10]
AREA_FRAC_LO = 0.08
AREA_FRAC_HI = 0.43
AREA_ABS_LO = 11
AREA_ABS_HI = 62
LEGEND_THRESHOLD = {8: 4, 10: 5}
LEGEND_CORE_SIZE = 4
LEGEND_ARM_RANGE = range(1, 4)
PARTITION_GROWTH_EXP = 1.45
PARTITION_FRONTIER_EXP = 0.2
PARTITION_SEED_JITTER = 0.2
SPLIT_GROWTH_EXP = 1.25
SPLIT_FRONTIER_EXP = 0.25
FRAGMENT_AREA_LO = 5
SPLIT_ATTEMPTS = 120
INPUT_SIZE_LO = 22
INPUT_SIZE_HI = 28
DENSITY_LO = 0.12
DENSITY_HI = 0.22
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


def _partition_rest(rest: np.ndarray):
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
    for k in range(4):
        rot = np.rot90(arr, k)
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


def _mask_cell(mask: np.ndarray, side: int) -> tuple[int, int]:
    pts = [tuple(p) for p in np.argwhere(mask)]
    return _cell_from_coords(pts, side, side, K)


def _split_mask(mask: np.ndarray, side: int, parent_cell: tuple[int, int]):
    pts = [tuple(p) for p in np.argwhere(mask)]
    coords_set = set(pts)
    if len(pts) < 2 * FRAGMENT_AREA_LO:
        return None

    for _ in range(SPLIT_ATTEMPTS):
        seed0 = random.choice(pts)
        seed1 = max(pts, key=lambda p: abs(p[0] - seed0[0]) + abs(p[1] - seed0[1]) + random.random() * 0.2)
        if seed0 == seed1:
            continue

        labels = {seed0: 0, seed1: 1}
        parts = [[seed0], [seed1]]
        frontiers = [set(), set()]
        areas = [1, 1]

        def add_frontier(i: int, cell: tuple[int, int]) -> None:
            r, c = cell
            for rr, cc in neighbors4(r, c):
                nxt = (rr, cc)
                if nxt in coords_set and nxt not in labels:
                    frontiers[i].add(nxt)

        add_frontier(0, seed0)
        add_frontier(1, seed1)
        left = len(pts) - 2

        while left:
            candidates = []
            for i in range(2):
                frontiers[i] = {p for p in frontiers[i] if p not in labels}
                if frontiers[i]:
                    candidates.append(i)
            if not candidates:
                break
            weights = [
                (1.0 / (areas[i] ** SPLIT_GROWTH_EXP)) * (len(frontiers[i]) ** SPLIT_FRONTIER_EXP)
                for i in candidates
            ]
            i = random.choices(candidates, weights=weights, k=1)[0]
            cell = random.choice(tuple(frontiers[i]))
            frontiers[i].discard(cell)
            if cell in labels:
                continue
            labels[cell] = i
            parts[i].append(cell)
            areas[i] += 1
            left -= 1
            add_frontier(i, cell)

        if left:
            continue
        if min(len(parts[0]), len(parts[1])) < FRAGMENT_AREA_LO:
            continue
        if not _coords_connected(parts[0]) or not _coords_connected(parts[1]):
            continue

        masks = []
        child_cells = []
        for coords in parts:
            part = np.zeros_like(mask, dtype=bool)
            for r, c in coords:
                part[r, c] = True
            if _has_hole(part):
                break
            masks.append(part)
            child_cells.append(_mask_cell(part, side))
        if len(masks) != 2:
            continue
        if not any(cell != parent_cell for cell in child_cells):
            continue
        return tuple(masks)
    return None


def _scatter_pieces(masks: list[np.ndarray], body_colors: list[int], legend_map: np.ndarray, side: int):
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for pi in random.sample(range(len(masks)), len(masks)):
        oriented = _crop_mask(masks[pi]) if pi == 0 else _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        if pi == 0:
            spots = _palette_spots(oriented)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r : r + K, c : c + K] = legend_map
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


def _body_colors(num_pieces: int, palette_colors: list[int]) -> list[int]:
    pool = [c for c in range(1, 10) if c not in palette_colors]
    random.shuffle(pool)
    legend_body = pool[0]
    others = pool[1:].copy()
    fallback_pool = pool[1:] if len(pool) > 1 else pool
    while len(others) < num_pieces - 1:
        others.append(random.choice(fallback_pool))
    random.shuffle(others)
    return [legend_body] + others[: num_pieces - 1]


def _gen_attempt():
    side = random.choice(SIDE_CHOICES)
    min_area, max_area = _area_bounds(side)
    legend = _make_legend_mask(side, min_area, max_area)
    labels = _partition_rest(~legend)
    if labels is None:
        return None

    parent_masks = [legend] + [labels == i for i in range(3)]
    if any(a < min_area or a > max_area for a in (int(m.sum()) for m in parent_masks)):
        return None
    if any(len(components_mask(m, connectivity=4)) != 1 or _has_hole(m) for m in parent_masks):
        return None

    parent_cells = [_mask_cell(mask, side) for mask in parent_masks]
    if len(set(parent_cells)) != 4:
        return None

    split_candidates = {}
    for pi in range(1, 4):
        split = _split_mask(parent_masks[pi], side, parent_cells[pi])
        if split is not None:
            split_candidates[pi] = split
    split_count = random.choice([1, 2])
    if len(split_candidates) < split_count:
        return None
    chosen_split = set(random.sample(sorted(split_candidates), split_count))

    piece_masks = [parent_masks[0]]
    for pi in range(1, 4):
        if pi in chosen_split:
            piece_masks.extend(split_candidates[pi])
        else:
            piece_masks.append(parent_masks[pi])

    if len(piece_masks) not in (5, 6):
        return None
    if any(int(mask.sum()) < FRAGMENT_AREA_LO for mask in piece_masks[1:]):
        return None
    if any(len(components_mask(mask, connectivity=4)) != 1 or _has_hole(mask) for mask in piece_masks):
        return None
    if not _forces_unique_square(piece_masks, side):
        return None

    palette_colors = random.sample(range(1, 10), 4)
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)
    body_colors = _body_colors(len(piece_masks), palette_colors)

    expected = np.zeros((side, side), dtype=int)
    for mask, cell in zip(parent_masks, parent_cells):
        rr, cc = cell
        expected[mask] = legend_map[rr, cc]

    grid = _scatter_pieces(piece_masks, body_colors, legend_map, side)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c D9 input")


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    _, legend_idx, pieces, k, legend_map = _extract_scene(grid)
    _, meta = _assemble_exact_cover(pieces, legend_idx, legend_map, k, return_meta=True)
    grouping = meta["grouping"]
    chosen = meta["chosen"]
    H, W = meta["shape"]

    split_group_count = 0
    off_cell_fragment_count = 0
    for gi, group in enumerate(grouping):
        merged_coords: list[tuple[int, int]] = []
        for pi in group:
            _, r0, c0, coords = chosen[pi]
            merged_coords.extend((r0 + dr, c0 + dc) for dr, dc in coords)
        merged_cell = _cell_from_coords(merged_coords, H, W, k)
        if len(group) == 2:
            split_group_count += 1
        for pi in group:
            _, r0, c0, coords = chosen[pi]
            part_coords = [(r0 + dr, c0 + dc) for dr, dc in coords]
            if _cell_from_coords(part_coords, H, W, k) != merged_cell:
                off_cell_fragment_count += 1

    return {
        "component_count": len(pieces),
        "split_group_count": split_group_count,
        "off_cell_fragment_count": off_cell_fragment_count,
    }
