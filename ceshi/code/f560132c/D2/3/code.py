# Variant of f560132c
# Dimension: D2 — 源对象：散落精灵的类型与结构 (src)
# Change: 4个单色不规则精灵散布在网格中，每个精灵使用不同的单一颜色，形状各异但共同组装后恰好形成矩形... → 精灵间共享颜色（同色不同形）

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location
import math
import random

# Design analysis for the D2 variant
# - Concretization:
#   The input still has 4 disconnected pieces and one legend piece with an embedded 2x2 palette.
#   The visible change is that exactly two non-legend pieces now share the same body color while
#   remaining different under D4. A solver therefore cannot use body-color uniqueness to track
#   pieces and must rely on connectivity plus shape assembly.
# - Behavior trace:
#   In the original task, if loose pieces had body colors 2/3/7/8, a human could casually refer to
#   "the orange piece" or "the teal piece" while matching the legend quadrants. In this variant,
#   two loose pieces may both be orange; only their geometry distinguishes which one lands in which
#   quadrant after exact-cover assembly. The output rule itself is unchanged.
# - Complexity calibration:
#   The duplicated color is restricted to one pair among the three monochrome non-legend pieces.
#   This keeps the ambiguity obvious without flooding the grid with repeated colors or extra cases.
# - Implicit assumptions from `main()`:
#   `components_mask(g != bg)` remains valid because disconnected same-color pieces are still
#   separate components. Legend detection by "most distinct colors" remains valid because only the
#   legend carries the 2x2 palette. Exact-cover assembly already ignores body-color identity.
# - Placement legality:
#   The generator must ensure the duplicate-color pair do not touch, remain monochrome, and are not
#   D4-equivalent shapes. `random_free_location` plus a one-cell padding keeps components separate.
# - Visual observability and tidiness:
#   Across a few examples, one repeated color appears on two clearly different loose shapes, while
#   the output still resolves into four palette-colored regions. No rule depends on numeric color ids.
# - New assumptions introduced here:
#   `main()` assumes only one multicolor component exists and the 4 pieces still tile a unique
#   rectangle. `generate_input()` explicitly preserves those conditions and verifies them with
#   `np.array_equal(main(grid), expected)` before accepting a sample.


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


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, k: int) -> np.ndarray:
    """
    Inside the legend component's bounding box, find a k×k color map block (may include 0).
    We pick the k×k window with the maximum number of distinct colors; ties go to top-left.
    """
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert k > 0 and hh >= k and ww >= k, 'legend component must fit k×k map'
    best = None  # (distinct, -rr, -cc)
    best_rc = None
    for rr in range(hh - k + 1):
        for cc in range(ww - k + 1):
            window = sub[rr : rr + k, cc : cc + k]
            distinct = int(len(set(map(int, window.ravel()))))
            key = (distinct, -rr, -cc)
            if best is None or key > best:
                best = key
                best_rc = (rr, cc)
    assert best_rc is not None, 'legend component must contain a valid k×k color map window'
    rr, cc = best_rc
    return np.asarray(sub[rr : rr + k, cc : cc + k], dtype=int).copy()


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


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Tile all pieces into a full rectangle (no holes), allowing rotations + reflections.
    Recolor each piece by quantizing its centroid into a k×k grid, using legend_map.
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

        # Precompute placements for non-legend pieces (legend is fixed at (0,0), but orientation varies).
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
                # Branch over *any* remaining piece that can cover this cell.
                # (Picking only one "best" piece can miss valid tilings.)
                candidates = []
                for k in sorted(remaining):
                    opts_k = []
                    for pl in filtered[k]:
                        m = pl[0]
                        if (m >> cell) & 1 and (m & used) == 0:
                            opts_k.append(pl)
                    if opts_k:
                        candidates.append((len(opts_k), k, opts_k))
                if not candidates:
                    return False

                candidates.sort(key=lambda x: (x[0], x[1]))
                for _, k, opts_k in candidates:
                    remaining.remove(k)
                    pi = other_pis[k]
                    for pl in opts_k:
                        chosen[pi] = pl
                        if search(used | pl[0]):
                            return True
                    chosen.pop(pi, None)
                    remaining.add(k)
                return False

            if not search(legend_mask):
                return None
            return chosen

        # Try each possible legend orientation at (0,0).
        chosen = None
        for coords, ph, pw, _, _ in pieces[legend_idx]["oris"]:
            assert ph <= H and pw <= W, 'legend orientation must fit the unique output rectangle'
            sol = solve_with_fixed_legend(coords)
            if sol is None:
                continue
            chosen = sol
            break
        assert chosen is not None, 'exact-cover assembly failed: no valid legend orientation'

        # Determine legend-map cell for each piece by centroid.
        piece_cell: dict[int, tuple[int, int]] = {}
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            # centroid: average over occupied cells
            rs = [r0 + dr for dr, _ in coords]
            cs = [c0 + dc for _, dc in coords]
            cr = float(sum(rs)) / float(len(rs))
            cc = float(sum(cs)) / float(len(cs))
            rr = int(cr * float(k) / float(H))
            cc2 = int(cc * float(k) / float(W))
            rr = max(0, min(int(k) - 1, rr))
            cc2 = max(0, min(int(k) - 1, cc2))
            piece_cell[pi] = (rr, cc2)

        # Build final output.
        out = np.zeros((H, W), dtype=int)
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            rr, cc2 = piece_cell[pi]
            col = int(legend_map[rr, cc2])
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four scattered sprites into one rectangle and repaint them from the legend sprite's palette.

    Input structure:
    - bg = most common color.
    - Four disconnected non-bg sprites are scattered across the grid.
    - One sprite is the legend because it contains a multicolor 2×2 patch that acts as a color key.
    - The other sprites are monochrome, but two of them may share the same body color despite having different shapes.

    Rule:
    - Treat each connected sprite as a jigsaw piece and tile the four pieces into the unique tight rectangle.
    - Rotations and reflections are allowed when fitting the pieces together.
    - Recolor each assembled piece by the legend patch cell reached by that piece's centroid in the final 2×2 layout.

    Key insight:
    - Duplicate body colors are only distractors. Shape determines where each piece belongs, and the legend's internal palette determines the output colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, 'grid must contain at least one non-background component'

    # Repeated body colors are allowed; the legend is still the only multicolor component.
    k = int(round(len(comps) ** 0.5))
    assert k * k == len(comps), 'piece count must be a perfect square to determine k'

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert max_distinct > 1, 'legend component must be multicolor'
    assert len(legend_candidates) == 1, 'legend component must be uniquely identified by distinct colors'
    legend_idx = legend_candidates[0]
    legend_map = _extract_legend_map(g, comps[legend_idx], k)

    # Build piece masks (coords in bbox space), including filled holes.
    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        pieces.append({"oris": oris, "area": len(filled_abs)})

    return _assemble_exact_cover(pieces, legend_idx, legend_map, k)

# --------------- generator (o5x) ---------------

# === CFG ===
SIDE_CHOICES = [8, 10, 12]                     # output square side lengths
K = 2                                          # k*k quadrant grid (2*2 = 4 pieces)
NUM_PIECES = K * K                             # total jigsaw pieces
BODY_COLOR_COUNT = 3                           # legend unique + one duplicated pair + one remaining unique body color
PALETTE_COLOR_COUNT = 4                        # 2x2 legend palette colors
TOTAL_NON_BG_COLORS = BODY_COLOR_COUNT + PALETTE_COLOR_COUNT
AREA_FRAC_LO = 0.08                           # min piece area as fraction of total
AREA_FRAC_HI = 0.43                           # max piece area as fraction of total
AREA_ABS_LO = 11                              # absolute min piece area (cells)
AREA_ABS_HI = 62                              # absolute max piece area (cells)
LEGEND_THRESHOLD = {8: 4, 10: 5, 12: 9}       # min extent threshold per side for legend L-shape
LEGEND_CORE_SIZE = 4                           # legend always has a 4x4 filled corner block
LEGEND_ARM_RANGE = range(1, 4)                 # arm width/height options for L-shape construction
PARTITION_GROWTH_EXP = 1.45                    # exponent penalizing large regions during partition
PARTITION_FRONTIER_EXP = 0.2                   # exponent boosting regions with wide frontiers
PARTITION_SEED_JITTER = 0.2                    # random jitter when selecting seed cells
INPUT_SIZE_LO = 20                             # min input grid side
INPUT_SIZE_HI = 26                             # max input grid side
DENSITY_LO = 0.15                              # min non-bg density target
DENSITY_HI = 0.25                              # max non-bg density target
SCATTER_BORDER = 1                             # border margin when placing pieces
SCATTER_PADDING = 1                            # gap between piece bounding boxes
GEN_RETRIES = 300                              # max attempts in generate_input()


def _area_bounds(side):
    """Min/max piece area for a given output side length."""
    total = side * side
    return max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total)), min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))


def _has_hole(mask):
    """Return True if the mask contains an interior hole (bg region not touching the bbox edge)."""
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
    """Check that piece bbox dims force a unique S*S factorization of total area."""
    bbs = [bbox_coords(np.argwhere(m)) for m in masks]
    max_h = max(r1 - r0 + 1 for r0, _, r1, _ in bbs)
    max_w = max(c1 - c0 + 1 for _, c0, _, c1 in bbs)
    dims = [(h, w) for h, w in _factor_pairs(side * side) if h >= max_h and w >= max_w]
    return dims == [(side, side)]


def _make_legend_mask(side, min_area, max_area):
    """Build an L-shaped legend piece mask anchored at (0,0) of the side*side grid."""
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


def _partition_rest(rest):
    """Flood-fill partition the complement of the legend into 3 connected regions."""
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
    """Crop a boolean mask to its bounding box."""
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_orientation(mask):
    """Return a random D4 orientation of the cropped mask (unique shapes only)."""
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


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Canonical D4-invariant signature of a piece mask."""
    pts = tuple(map(tuple, np.argwhere(_crop_mask(mask))))
    norm, _, _ = _normalize_coords(pts)
    return min(coords for coords, _, _, _, _ in _dihedral_orientations(norm))


def _pick_shared_color_pair(masks: list[np.ndarray]) -> tuple[int, int] | None:
    """Choose two non-legend pieces that are visibly different under D4."""
    nonlegend = [1, 2, 3]
    sigs = {pi: _shape_signature(masks[pi]) for pi in nonlegend}
    pairs = [(i, j) for i in nonlegend for j in nonlegend if i < j and sigs[i] != sigs[j]]
    if not pairs:
        return None
    return random.choice(pairs)


def _palette_spots(mask):
    """Find valid 2x2 palette embedding positions (fully surrounded by piece cells)."""
    spots = []
    h, w = mask.shape
    for r in range(1, h - K):
        for c in range(1, w - K):
            if mask[r : r + K, c : c + K].all() and mask[r - 1 : r + K + 1, c - 1 : c + K + 1].all():
                spots.append((r, c))
    return spots


def _scatter_pieces(masks, body_colors, legend_map, side):
    """Place all pieces on a random-sized input grid with separation."""
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
            spots = _palette_spots(oriented)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r : r + K, c : c + K] = legend_map
        try:
            box = np.ones(oriented.shape, dtype=int)
            r0, c0 = random_free_location(occupied, box, transparent=0,
                                           border=SCATTER_BORDER, padding=SCATTER_PADDING, padding_connectivity=8)
        except ValueError:
            return None
        occupied[r0 : r0 + oriented.shape[0], c0 : c0 + oriented.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=0)
    return grid


def _gen_attempt():
    """Single generation attempt: build tiling, validate, scatter pieces."""
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

    # Compute centroid quadrant for each piece; all 4 must be distinct.
    qcells = []
    for mask in masks:
        pts = np.argwhere(mask)
        qcells.append((min(1, int(pts[:, 0].mean() * K / side)), min(1, int(pts[:, 1].mean() * K / side))))
    if len(set(qcells)) != NUM_PIECES:
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

    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        expected[mask] = legend_map[rr, cc]

    grid = _scatter_pieces(masks, body_colors, legend_map, side)
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
    raise RuntimeError('failed to generate f560132c input')
