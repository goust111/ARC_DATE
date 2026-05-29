# Variant of f560132c
# Dimension: D14 — 图例颜色与精灵颜色的映射方式 (meta)
# Change: 图例2x2中的4种颜色与精灵的4种颜色完全不同——图例颜色是目标色，精灵颜色是源色，形成完全的重映射... → 图例颜色与精灵颜色一一对应（颜色映射表）

import math
import random
import itertools
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location

# Design analysis for this D14 variant
# -----------------------------------
# a) Concrete visual change / behavior trace
#    - Original task: the legend piece contained a 2x2 palette; each assembled output quadrant directly used the
#      color stored in the matching legend quadrant.
#    - Variant: the legend is now a 4x4 mapping block split into four 2x2 slots. Inside each 2x2 slot, the left
#      column repeats a source sprite color and the right column repeats its target color. The solver must first
#      decode source->target pairs, then recolor each assembled piece by matching the piece's original body color.
#    - Small trace: if a red piece appears in input and the legend has a slot [[red, yellow], [red, yellow]],
#      then the assembled red piece becomes yellow, regardless of which output quadrant that slot sits in.
#      This differs from the original logic, where the slot's position itself determined the output color.
# b) Complexity calibration
#    - The geometry stays unchanged: four pieces still tile one unique square under D4 transforms.
#    - The new symbolic step is limited to four repeated pair-entries, so it is visible but not overloaded.
#    - To avoid accidental collapse back to the original spatial rule, generator places mapping entries using a
#      derangement: every piece's mapping entry is stored in a different legend quadrant from the piece's output
#      quadrant.
# c) Hidden assumptions in the original main()
#    - `_extract_legend_map(...)` assumed one pixel per logical legend cell; that fails once each logical cell
#      becomes a 2x2 source/target pair, so extraction must be replaced.
#    - `_assemble_exact_cover(...)` assumed output color came from assembled centroid position. In this variant
#      that assumption is invalid; recoloring must use the decoded source color of each piece instead.
#    - Legend detection by "most distinct colors" still holds because the legend component contains all source
#      colors plus four distinct target colors, while non-legend pieces remain monochrome.
#    - Piece-mask extraction, hole filling, orientation enumeration, and exact-cover search do not depend on the
#      old palette semantics, so they are kept intact.
# d) Placement legality
#    - Pieces are still scattered on a larger background grid with padding and border, so components remain
#      separated and solver component detection remains valid.
#    - The 4x4 mapping block is embedded only in a solid 4x4 region of the legend piece, so it never leaks
#      outside the piece or creates disconnected fragments.
# e) Visual observability
#    - The repeated 2x2 source/target slots make the legend visibly different from the original 2x2 palette.
#    - Each source color is repeated twice inside its slot and also appears on exactly one sprite, giving enough
#      support to infer the mapping from a few examples.
# f) Visual cleanliness
#    - The legend uses a regular 2x2 grid of slots; there is no overlapping annotation layer besides the mapping
#      block itself, so the piece remains readable.
# g) Color-change inferability
#    - There are exactly four source colors and four target colors.
#    - Targets stay pairwise distinct and disjoint from sources, so the mapping is easy to see and does not rely
#      on hidden color ids or numeric ordering.
# h) New solver assumptions introduced here
#    - The legend block is always a 4x4 window whose four 2x2 quadrants follow the repeated-column pattern
#      [[src, tgt], [src, tgt]].
#    - Each piece has one dominant body color; generator guarantees this by making the legend piece area large
#      enough that its body color remains strictly more frequent than any embedded table color.
#    - The decoded mapping is bijective across the four piece colors; generator enforces four unique sources and
#      four unique targets.


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


def _extract_legend_mapping(g: np.ndarray, pts: np.ndarray, k: int) -> dict[int, int]:
    """
    Inside the legend component's bounding box, find a (2k)x(2k) mapping table.
    Each logical quadrant is 2x2 and must look like [[src, tgt], [src, tgt]].
    We pick the window with the most valid quadrants; ties go to the one with more
    distinct colors, then top-left.
    """
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    span = 2 * k
    hh, ww = sub.shape
    assert k > 0 and hh >= span and ww >= span, 'legend component must fit the mapping table'

    best_key = None
    best_rc = None
    for rr in range(hh - span + 1):
        for cc in range(ww - span + 1):
            window = sub[rr : rr + span, cc : cc + span]
            valid = 0
            for qr in range(k):
                for qc in range(k):
                    block = window[2 * qr : 2 * qr + 2, 2 * qc : 2 * qc + 2]
                    left = set(map(int, block[:, 0]))
                    right = set(map(int, block[:, 1]))
                    if len(left) == 1 and len(right) == 1 and next(iter(left)) != next(iter(right)):
                        valid += 1
            key = (valid, len(set(map(int, window.ravel()))), -rr, -cc)
            if best_key is None or key > best_key:
                best_key = key
                best_rc = (rr, cc)

    assert best_rc is not None and best_key is not None, 'legend component must contain a mapping table'
    assert best_key[0] == k * k, 'legend mapping table must decode all quadrants'
    rr, cc = best_rc
    window = sub[rr : rr + span, cc : cc + span]

    mapping: dict[int, int] = {}
    for qr in range(k):
        for qc in range(k):
            block = window[2 * qr : 2 * qr + 2, 2 * qc : 2 * qc + 2]
            src = int(block[0, 0])
            tgt = int(block[0, 1])
            mapping[src] = tgt

    assert len(mapping) == k * k, 'legend mapping must contain one entry per piece color'
    assert len(set(mapping.values())) == k * k, 'legend targets must be distinct'
    return mapping


def _fill_holes_in_component(g: np.ndarray, pts: np.ndarray, bg: int) -> tuple[tuple[tuple[int, int], ...], int]:
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


def _piece_body_color(g: np.ndarray, pts: np.ndarray, bg: int) -> int:
    counts = Counter(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
    assert counts, 'piece must contain at least one non-background color'
    return int(counts.most_common(1)[0][0])


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_mapping: dict[int, int],
) -> np.ndarray:
    """
    Tile all pieces into a full rectangle (no holes), allowing rotations + reflections.
    Recolor each assembled piece by looking up its original body color in the legend table.
    """
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = _factor_pairs(total)
    dims = [(H, W) for (H, W) in dims if H >= max_h and W >= max_w]
    assert len(dims) == 1, 'piece areas and bounds must imply a unique output rectangle'

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
                for k2 in sorted(remaining):
                    opts_k = []
                    for pl in filtered[k2]:
                        m = pl[0]
                        if (m >> cell) & 1 and (m & used) == 0:
                            opts_k.append(pl)
                    if opts_k:
                        candidates.append((len(opts_k), k2, opts_k))
                if not candidates:
                    return False

                candidates.sort(key=lambda x: (x[0], x[1]))
                for _, k2, opts_k in candidates:
                    remaining.remove(k2)
                    pi = other_pis[k2]
                    for pl in opts_k:
                        chosen[pi] = pl
                        if search(used | pl[0]):
                            return True
                    chosen.pop(pi, None)
                    remaining.add(k2)
                return False

            if not search(legend_mask):
                return None
            return chosen

        chosen = None
        for coords, ph, pw, _, _ in pieces[legend_idx]["oris"]:
            assert ph <= H and pw <= W, 'legend orientation must fit the unique output rectangle'
            sol = solve_with_fixed_legend(coords)
            if sol is None:
                continue
            chosen = sol
            break
        assert chosen is not None, 'exact-cover assembly failed: no valid legend orientation'

        out = np.zeros((H, W), dtype=int)
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            src = int(pieces[pi]["body_color"])
            assert src in legend_mapping, 'every piece color must appear in the legend mapping'
            col = int(legend_mapping[src])
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        return out

    raise AssertionError('unreachable rectangle assembly state')


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a rectangle, then recolor them by a legend piece's source->target table.

    Input structure:
    - bg = most common color.
    - Four separated non-bg sprites are scattered on the canvas; three are monochrome and one is the legend piece.
    - The legend piece contains a 4x4 table split into four 2x2 slots; each slot shows a source sprite color next
      to its target color as repeated vertical pairs.

    Rule:
    - Ignore the legend colors when matching shapes: all four pieces still tile one tight rectangle under D4 transforms.
    - Read the legend's four source->target pairs.
    - For each assembled piece, identify its original body color in the input and recolor the whole piece to the
      mapped target color.

    Key insight: The legend no longer tells you "which output quadrant gets which color"; it tells you how to translate
    each sprite's input color into its output color after the pieces are assembled.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, 'grid must contain at least one non-background component'

    k = int(round(len(comps) ** 0.5))
    assert k * k == len(comps), 'piece count must be a perfect square to determine k'

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, 'legend component must be uniquely identified by distinct colors'
    legend_idx = legend_candidates[0]
    legend_mapping = _extract_legend_mapping(g, comps[legend_idx], k)

    pieces: list[dict] = []
    body_colors = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        body = _piece_body_color(g, pts, bg)
        body_colors.append(body)
        pieces.append({"oris": oris, "area": len(filled_abs), "body_color": body})

    assert set(body_colors) == set(legend_mapping.keys()), 'legend sources must match the four piece colors'
    return _assemble_exact_cover(pieces, legend_idx, legend_mapping)


# --------------- generator (D14) ---------------

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
LEGEND_REQUIRED_AREA = 20
PARTITION_GROWTH_EXP = 1.45
PARTITION_FRONTIER_EXP = 0.2
PARTITION_SEED_JITTER = 0.2
INPUT_SIZE_LO = 20
INPUT_SIZE_HI = 26
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 300


def _area_bounds(side):
    total = side * side
    return max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total)), min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))


def _has_hole(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return True
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    empty = ~sub
    if not empty.any():
        return False
    for comp in components_mask(empty, connectivity=4):
        if not (np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == sub.shape[0] - 1)
                or np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == sub.shape[1] - 1)):
            return True
    return False


def _forces_unique_square(masks, side):
    bbs = [bbox_coords(np.argwhere(m)) for m in masks]
    max_h = max(r1 - r0 + 1 for r0, _, r1, _ in bbs)
    max_w = max(c1 - c0 + 1 for _, c0, _, c1 in bbs)
    dims = [(h, w) for h, w in _factor_pairs(side * side) if h >= max_h and w >= max_w]
    return dims == [(side, side)]


def _make_legend_mask(side, min_area, max_area):
    threshold = LEGEND_THRESHOLD[side]
    min_area = max(min_area, LEGEND_REQUIRED_AREA)
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
    return random.choice(opts).copy() if opts else None


def _partition_rest(rest):
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

    def add_frontier(i, r, c):
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

        weights = [(1.0 / (areas[i] ** PARTITION_GROWTH_EXP)) * (len(frontiers[i]) ** PARTITION_FRONTIER_EXP)
                    for i in candidates]
        i = random.choices(candidates, weights=weights, k=1)[0]
        cell = random.choice(tuple(frontiers[i]))
        frontiers[i].discard(cell)
        labels[cell] = i
        areas[i] += 1
        unassigned -= 1
        add_frontier(i, cell[0], cell[1])

    return labels


def _crop_mask(mask):
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_orientation(mask):
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


def _mapping_spots(mask):
    spots = []
    h, w = mask.shape
    span = 2 * K
    for r in range(0, h - span + 1):
        for c in range(0, w - span + 1):
            if mask[r : r + span, c : c + span].all():
                spots.append((r, c))
    return spots


def _slot_index(rc: tuple[int, int]) -> int:
    rr, cc = rc
    return int(rr) * K + int(cc)


def _derangement(items: list[int]) -> list[int] | None:
    perms = []
    for perm in itertools.permutations(items):
        if all(a != b for a, b in zip(items, perm)):
            perms.append(list(perm))
    return random.choice(perms) if perms else None


def _build_mapping_patch(slot_entries: list[tuple[int, int]]) -> np.ndarray:
    patch = np.zeros((2 * K, 2 * K), dtype=int)
    for slot, (src, tgt) in enumerate(slot_entries):
        rr, cc = divmod(slot, K)
        patch[2 * rr : 2 * rr + 2, 2 * cc] = src
        patch[2 * rr : 2 * rr + 2, 2 * cc + 1] = tgt
    return patch


def _scatter_pieces(masks, body_colors, mapping_patch, side):
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
            spots = _mapping_spots(oriented)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r : r + 2 * K, c : c + 2 * K] = mapping_patch
        try:
            box = np.ones(oriented.shape, dtype=int)
            r0, c0 = random_free_location(occupied, box, transparent=0,
                                           border=SCATTER_BORDER, padding=SCATTER_PADDING, padding_connectivity=8)
        except ValueError:
            return None
        occupied[r0 : r0 + oriented.shape[0], c0 : c0 + oriented.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=0)
    return grid


def _variant_metrics(qcells, legend_slot_for_piece):
    displaced = sum(int(legend_slot_for_piece[pi] != _slot_index(qcells[pi])) for pi in range(NUM_PIECES))
    return {"displaced_entries": displaced}


def _gen_attempt():
    side = random.choice(SIDE_CHOICES)
    min_area, max_area = _area_bounds(side)
    legend = _make_legend_mask(side, min_area, max_area)
    if legend is None:
        return None
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
    target_colors = colors[NUM_PIECES:]
    random.shuffle(target_colors)

    piece_slots = [_slot_index(rc) for rc in qcells]
    slots = list(range(NUM_PIECES))
    der = _derangement(slots)
    if der is None:
        return None

    legend_slot_for_piece = {}
    used_slots = set()
    for pi in range(NUM_PIECES):
        legend_slot_for_piece[pi] = der[piece_slots[pi]]
        used_slots.add(legend_slot_for_piece[pi])
    if len(used_slots) != NUM_PIECES:
        return None

    slot_entries: list[tuple[int, int]] = [(-1, -1)] * NUM_PIECES
    target_by_source = {}
    for pi in range(NUM_PIECES):
        src = body_colors[pi]
        tgt = target_colors[pi]
        target_by_source[src] = tgt
        slot_entries[legend_slot_for_piece[pi]] = (src, tgt)

    if any(src < 0 for src, _ in slot_entries):
        return None

    mapping_patch = _build_mapping_patch(slot_entries)

    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        expected[mask] = target_by_source[body_colors[pi]]

    grid = _scatter_pieces(masks, body_colors, mapping_patch, side)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    metrics = _variant_metrics(qcells, legend_slot_for_piece)
    if metrics["displaced_entries"] != NUM_PIECES:
        return None
    return grid


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate f560132c D14 input')
