# Combo variant of f560132c (index 13)
# combo_spec: D6/1+D11/0
# Name: 输出含图例副本+象限分割线

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the combo:
# - Concrete behavior trace:
#   Original task assembles the four scattered pieces into a tight SxS square and recolors each piece
#   by the legend's 2x2 palette cell selected from its assembled centroid. Here we keep that core step,
#   then transform the compact square into a structured output in two visible ways:
#   1) insert one full-height vertical separator and one full-width horizontal separator at the 2x2
#      quadrant split, so an 8x8 body becomes 9x9 with row/col 4 occupied by separator color;
#   2) append a tight copy of the original legend sprite on the right with a 1-cell background gap.
#   Example: if the compact assembly is 10x10 and the legend bbox is 7x6, the final output is
#   max(11,7) by (11 + 1 + 6), with the left 11x11 region holding the separated assembly.
# - Complexity calibration:
#   The separator rule stays low-entropy: always one row + one column because k=2, always straight,
#   always colored with the legend body's dominant color. The legend copy is always top-aligned on the
#   right, so the extra formatting is obvious but does not create packing ambiguity.
# - Implicit assumptions audited against main():
#   The original exact-cover logic assumes piece count is a perfect square and the multicolor legend is
#   unique; those still hold. New assumptions are: the assembled square side is divisible by k so the
#   separator location is well-defined, and the legend body color is the dominant non-bg color in the
#   legend crop. The generator enforces even side lengths and legend area >= 11, so the body strictly
#   dominates the four 1-cell palette colors.
# - Placement legality:
#   The generator still scatters non-overlapping pieces with padding. The output formatter only writes
#   into a freshly allocated canvas sized from the separated assembly and legend bbox, so there is no
#   risk of overlap or out-of-bounds placement.
# - Visual observability / cleanliness:
#   The copied legend directly explains the separator color source, and the separators are full-span
#   straight lines rather than subtle local marks. Across 2-3 examples, a human can see that outputs
#   now contain both a reference sprite and a partitioned result instead of only the compact rectangle.


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
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...],
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
    g: np.ndarray,
    pts: np.ndarray,
    bg: int,
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


def _dominant_non_bg_color(arr: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in arr.ravel() if int(v) != bg]
    assert vals, "legend crop must contain non-background pixels"
    return int(Counter(vals).most_common(1)[0][0])


def _insert_quadrant_separators(compact: np.ndarray, line_color: int, k: int) -> np.ndarray:
    h, w = compact.shape
    assert h % k == 0 and w % k == 0, "assembled rectangle must split evenly into kxk regions"
    out = np.full((h + k - 1, w + k - 1), int(line_color), dtype=int)
    row_bounds = [i * h // k for i in range(k + 1)]
    col_bounds = [i * w // k for i in range(k + 1)]
    for rr in range(k):
        sr0, sr1 = row_bounds[rr], row_bounds[rr + 1]
        dr0 = sr0 + rr
        for cc in range(k):
            sc0, sc1 = col_bounds[cc], col_bounds[cc + 1]
            dc0 = sc0 + cc
            out[dr0 : dr0 + (sr1 - sr0), dc0 : dc0 + (sc1 - sc0)] = compact[sr0:sr1, sc0:sc1]
    return out


def _format_variant_output(
    compact: np.ndarray,
    legend_crop: np.ndarray,
    line_color: int,
    k: int,
    bg: int,
) -> np.ndarray:
    structured = _insert_quadrant_separators(compact, line_color, k)
    sh, sw = structured.shape
    lh, lw = legend_crop.shape
    out = np.full((max(sh, lh), sw + 1 + lw), int(bg), dtype=int)
    out[:sh, :sw] = structured
    out[:lh, sw + 1 : sw + 1 + lw] = legend_crop
    return out


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
    dims = [(h, w) for (h, w) in dims if h >= max_h and w >= max_w]
    assert len(dims) == 1, "piece areas and bounds must imply a unique output rectangle"

    for h, w in dims:
        nbits = h * w
        all_mask = (1 << nbits) - 1
        other_pis = [i for i in range(len(pieces)) if i != legend_idx]
        placements_other: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
        for pi in other_pis:
            opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
            for coords, ph, pw, _, _ in pieces[pi]["oris"]:
                for r0 in range(0, h - ph + 1):
                    for c0 in range(0, w - pw + 1):
                        mask = 0
                        for dr, dc in coords:
                            idx = (int(r0) + int(dr)) * int(w) + (int(c0) + int(dc))
                            mask |= 1 << idx
                        opts.append((mask, int(r0), int(c0), coords))
            placements_other.append(opts)

        def solve_with_fixed_legend(
            legend_coords: tuple[tuple[int, int], ...],
        ) -> dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] | None:
            legend_mask = 0
            for dr, dc in legend_coords:
                idx = int(dr) * int(w) + int(dc)
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
                for key in sorted(remaining):
                    opts_key = []
                    for pl in filtered[key]:
                        mask = pl[0]
                        if (mask >> cell) & 1 and (mask & used) == 0:
                            opts_key.append(pl)
                    if opts_key:
                        candidates.append((len(opts_key), key, opts_key))
                if not candidates:
                    return False

                candidates.sort(key=lambda x: (x[0], x[1]))
                for _, key, opts_key in candidates:
                    remaining.remove(key)
                    pi = other_pis[key]
                    for pl in opts_key:
                        chosen[pi] = pl
                        if search(used | pl[0]):
                            return True
                    chosen.pop(pi, None)
                    remaining.add(key)
                return False

            if not search(legend_mask):
                return None
            return chosen

        chosen = None
        for coords, ph, pw, _, _ in pieces[legend_idx]["oris"]:
            assert ph <= h and pw <= w, "legend orientation must fit the unique output rectangle"
            sol = solve_with_fixed_legend(coords)
            if sol is not None:
                chosen = sol
                break
        assert chosen is not None, "exact-cover assembly failed: no valid legend orientation"

        piece_cell: dict[int, tuple[int, int]] = {}
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            rs = [r0 + dr for dr, _ in coords]
            cs = [c0 + dc for _, dc in coords]
            cr = float(sum(rs)) / float(len(rs))
            cc = float(sum(cs)) / float(len(cs))
            rr = int(cr * float(k) / float(h))
            cc2 = int(cc * float(k) / float(w))
            rr = max(0, min(int(k) - 1, rr))
            cc2 = max(0, min(int(k) - 1, cc2))
            piece_cell[pi] = (rr, cc2)

        out = np.zeros((h, w), dtype=int)
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            rr, cc2 = piece_cell[pi]
            col = int(legend_map[rr, cc2])
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        return out

    raise AssertionError("unique output rectangle was not realized")


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int]:
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
        pieces.append({"oris": _dihedral_orientations(rel_norm), "area": len(filled_abs)})

    compact = _assemble_exact_cover(pieces, legend_idx, legend_map, k)
    lr0, lc0, lr1, lc1 = bbox_coords(comps[legend_idx])
    legend_crop = g[lr0 : lr1 + 1, lc0 : lc1 + 1].copy()
    line_color = _dominant_non_bg_color(legend_crop, bg)
    return compact, legend_crop, line_color, k, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered pieces into the color-coded square, then emit a structured output
    that also preserves the legend sprite as a reference.

    Input structure:
    - bg = most common color.
    - Four 4-connected scattered polyomino pieces form an exact-cover square under D4 transforms.
    - Exactly one piece is the legend: its body is mostly one color and it contains an interior 2x2 palette.

    Rule:
    - Reconstruct the compact square exactly as in the base task: anchor the legend piece at the top-left,
      tile the remaining pieces, and recolor each assembled piece from the legend palette using centroid quadrant.
    - Expand that square by inserting a 1-pixel separator row and a 1-pixel separator column between the
      2x2 quadrants, colored with the legend body's dominant color.
    - Copy the original legend sprite's tight bounding-box crop to the right of the separated assembly,
      leaving a 1-pixel background gap between the two parts.

    Key insight:
    - The legend has two jobs now: its palette still determines the recoloring of the assembled square,
      and its body color also becomes the separator-line color, while the original legend sprite is preserved
      in the output as an explicit reference copy.
    """
    compact, legend_crop, line_color, k, bg = _solve_core(grid)
    return _format_variant_output(compact, legend_crop, line_color, k, bg)


# --------------- generator (o5x) ---------------

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
GEN_RETRIES = 300


def _area_bounds(side: int) -> tuple[int, int]:
    total = side * side
    lo = max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total))
    hi = min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))
    return lo, hi


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


def _scatter_pieces(
    masks: list[np.ndarray],
    body_colors: list[int],
    legend_map: np.ndarray,
    side: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)
    legend_patch = None

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _crop_mask(masks[pi]) if pi == 0 else _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = int(body_colors[pi])
        if pi == 0:
            spots = _palette_spots(oriented)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r : r + K, c : c + K] = legend_map
            legend_patch = patch.copy()
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

    assert legend_patch is not None
    return grid, legend_patch


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

    compact_expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        compact_expected[mask] = int(legend_map[rr, cc])

    scattered = _scatter_pieces(masks, body_colors, legend_map, side)
    if scattered is None:
        return None
    grid, legend_patch = scattered
    expected = _format_variant_output(compact_expected, legend_patch, body_colors[0], K, 0)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c combo input")
