# Variant of f560132c
# Dimension: D8 — 图例与精灵的空间关系 (meta)
# Change: 图例精灵位于网格中固定区域（通常靠近左上角），与其他精灵在空间上分离，通过包含多色像素来区分... → 图例位于网格边缘一行/列

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the D8 variant
# a) Concrete behavior:
#    - The assembled output remains a 2x2 quadrant recoloring of four jigsaw pieces.
#    - The legend no longer lives inside a multicolor piece body. Instead, the palette is a
#      single edge strip on the outermost bottom row or right column, split into four contiguous
#      color runs. Horizontal strips are read row-major; vertical strips are read column-major.
#    - A single monochrome piece is still the "legend piece" for assembly anchoring, but now it is
#      identified spatially: it is the unique piece closest to the edge strip.
#    - Flow delta on a small example:
#      old: find the only multicolor component -> crop its internal 2x2 palette -> anchor that
#      same component at output (0,0).
#      new: find the only edge strip with four runs -> decode its 2x2 palette -> remove the strip
#      from the piece mask -> pick the piece nearest that strip -> anchor that piece at output (0,0).
# b) Complexity calibration:
#    - Only four legend runs are used, matching the original k*k = 4 palette cells.
#    - Generator varies the strip side between bottom and right so the edge behavior is visible,
#      but avoids all four sides to keep the training examples easy to infer.
#    - The nearest-piece cue is unique by construction; other pieces are pushed away from the strip.
# c) Old implicit assumptions that break here:
#    - `components_mask(g != bg)` used to contain only pieces. Now it would also include the strip,
#      so solver must delete legend-strip cells first.
#    - `k = sqrt(len(comps))` used to work because legend lived inside one piece. Now `k` comes from
#      the run count on the edge strip, and the piece count is checked afterwards.
#    - `max distinct colors` used to identify the legend component. That is invalid because all four
#      pieces are monochrome in this variant.
#    - `_extract_legend_map()` used to scan a piece bbox. The new legend map is decoded from one edge line.
# d) Placement legality:
#    - The strip is kept away from corners so only one border line is non-background.
#    - The legend piece is placed one blank cell away from the strip, so the strip stays a separate
#      component and cannot merge with any piece.
#    - Other pieces are scattered with a reserved exclusion band near the legend side, guaranteeing a
#      unique nearest piece and keeping the strip visually isolated.
# e) Visual observability:
#    - The edge strip is large enough to be obvious and is the only non-background content touching
#      the border, so a human must attend to the border rather than to an internal multicolor patch.
#    - The piece nearest the strip is visually distinguished by proximity alone; no hidden ordering
#      on color ids is used.
# f) Visual cleanliness:
#    - Strip, legend piece, and the remaining pieces are separated by blank rows/cols or padding.
#    - No overlays are used; the border legend and the piece field occupy different regions.
# g) Color-rule inferability:
#    - Only four palette colors are used, exactly once each in the edge strip, preserving the original
#      2x2 color-map semantics.
#    - The horizontal/vertical unpacking rule follows the strip orientation, which is visible in the grid.


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


def _parse_edge_runs(vals: np.ndarray, bg: int) -> tuple[list[int], int, int] | None:
    idxs = [i for i, v in enumerate(vals) if int(v) != bg]
    if not idxs:
        return None
    lo = min(idxs)
    hi = max(idxs)
    segment = [int(v) for v in vals[lo : hi + 1]]
    if any(v == bg for v in segment):
        return None
    colors = [segment[0]]
    for v in segment[1:]:
        if v != colors[-1]:
            colors.append(v)
    if len(set(colors)) != len(colors):
        return None
    return colors, int(lo), int(hi)


def _extract_edge_legend(g: np.ndarray, bg: int) -> tuple[str, int, int, int, np.ndarray]:
    """
    Find the unique legend strip on the outermost border.

    Returns (side, lo, hi, k, legend_map).
    - Horizontal strips (top/bottom) are unpacked row-major.
    - Vertical strips (left/right) are unpacked column-major.
    """
    h, w = g.shape
    edge_specs = [
        ("top", g[0, :]),
        ("bottom", g[h - 1, :]),
        ("left", g[:, 0]),
        ("right", g[:, w - 1]),
    ]
    candidates: list[tuple[str, int, int, int, np.ndarray]] = []
    for side, vals in edge_specs:
        parsed = _parse_edge_runs(np.asarray(vals, dtype=int), bg)
        if parsed is None:
            continue
        colors, lo, hi = parsed
        k = int(round(len(colors) ** 0.5))
        if k * k != len(colors):
            continue
        arr = np.asarray(colors, dtype=int)
        if side in ("top", "bottom"):
            legend_map = arr.reshape(k, k)
        else:
            legend_map = arr.reshape(k, k, order="F")
        candidates.append((side, lo, hi, int(k), legend_map))
    assert len(candidates) == 1, "grid must contain exactly one valid edge legend strip"
    return candidates[0]


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
            assert ph <= H and pw <= W, "legend orientation must fit the unique output rectangle"
            sol = solve_with_fixed_legend(coords)
            if sol is None:
                continue
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


def _distance_to_side(pts: np.ndarray, side: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox_coords(pts)
    if side == "top":
        return int(r0)
    if side == "bottom":
        return int(h - 1 - r1)
    if side == "left":
        return int(c0)
    if side == "right":
        return int(w - 1 - c1)
    raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a rectangle, recoloring each by a border legend strip.

    Input structure:
    - bg = most common color.
    - Several monochrome non-bg sprites are scattered in the interior of the grid.
    - A separate legend appears on the outermost bottom row or right column as four contiguous color runs.
    - The piece closest to that strip is the assembly anchor; the strip itself is not one of the pieces.

    Rule:
    - Decode the edge strip into a 2x2 color map.
    - Remove the strip, then tile the four pieces into a tight rectangle with rotations/reflections allowed.
    - Anchor the piece nearest the strip at the assembled top-left, then recolor every assembled piece by the
      legend-map cell reached by its centroid in the final rectangle.

    Key insight: The legend information has moved out of the sprite field and onto the border, so solving now
    depends on reading an edge run-pattern first and only then using proximity to recover which piece plays the old legend role.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    side, lo, hi, k, legend_map = _extract_edge_legend(g, bg)

    piece_mask = g != bg
    if side == "top":
        piece_mask[0, lo : hi + 1] = False
    elif side == "bottom":
        piece_mask[h - 1, lo : hi + 1] = False
    elif side == "left":
        piece_mask[lo : hi + 1, 0] = False
    else:
        piece_mask[lo : hi + 1, w - 1] = False

    comps = components_mask(piece_mask, connectivity=4)
    assert comps, "grid must contain at least one non-background component after removing the edge legend"
    assert len(comps) == k * k, "piece count must match the decoded edge legend"

    dists = [_distance_to_side(pts, side, g.shape) for pts in comps]
    min_dist = min(dists)
    legend_candidates = [i for i, d in enumerate(dists) if d == min_dist]
    assert len(legend_candidates) == 1, "legend piece must be uniquely closest to the edge legend"
    legend_idx = legend_candidates[0]

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
SCATTER_BORDER = 2
SCATTER_PADDING = 1
EDGE_SIDES = ("bottom", "right")
EDGE_SEG_LEN_CHOICES = [2, 3]
EDGE_MARGIN = 1
LEGEND_GAP = 1
LEGEND_EXCLUSION_BAND = 5
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
        if not (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == sub.shape[0] - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == sub.shape[1] - 1)
        ):
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


def _paint_edge_strip(grid: np.ndarray, side: str, start: int, seg_len: int, legend_map: np.ndarray):
    colors = legend_map.ravel().tolist() if side == "bottom" else legend_map.T.ravel().tolist()
    if side == "bottom":
        c = start
        for col in colors:
            grid[-1, c : c + seg_len] = int(col)
            c += seg_len
    elif side == "right":
        r = start
        for col in colors:
            grid[r : r + seg_len, -1] = int(col)
            r += seg_len
    else:
        raise ValueError(f"unsupported edge side: {side}")


def _legend_piece_position(grid_size: int, edge_side: str, shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = shape
    if edge_side == "bottom":
        r0 = grid_size - h - (LEGEND_GAP + 1)
        c_lo = 1
        c_hi = grid_size - w - 2
        if c_lo > c_hi or r0 < 1:
            return None
        return int(r0), random.randint(c_lo, c_hi)
    if edge_side == "right":
        c0 = grid_size - w - (LEGEND_GAP + 1)
        r_lo = 1
        r_hi = grid_size - h - 2
        if r_lo > r_hi or c0 < 1:
            return None
        return random.randint(r_lo, r_hi), int(c0)
    return None


def _strip_start_candidates(grid_size: int, edge_side: str, span: int, bbox_lo: int, bbox_hi: int) -> list[int]:
    lo = EDGE_MARGIN
    hi = grid_size - span - EDGE_MARGIN
    if lo > hi:
        return []
    starts = []
    for start in range(lo, hi + 1):
        end = start + span - 1
        if start <= bbox_hi and end >= bbox_lo:
            starts.append(int(start))
    return starts


def _reserve_band(occupied: np.ndarray, edge_side: str):
    if edge_side == "bottom":
        occupied[-LEGEND_EXCLUSION_BAND:, :] = 1
    elif edge_side == "right":
        occupied[:, -LEGEND_EXCLUSION_BAND:] = 1
    else:
        raise ValueError(f"unsupported edge side: {edge_side}")


def _scatter_pieces(masks, body_colors, legend_map, out_side):
    total = out_side * out_side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    edge_side = random.choice(EDGE_SIDES)
    seg_len = random.choice(EDGE_SEG_LEN_CHOICES)
    span = K * K * seg_len

    legend_oriented = _random_orientation(_crop_mask(masks[0]))
    legend_pos = _legend_piece_position(grid_size, edge_side, legend_oriented.shape)
    if legend_pos is None:
        return None
    lr0, lc0 = legend_pos
    lh, lw = legend_oriented.shape

    if edge_side == "bottom":
        starts = _strip_start_candidates(grid_size, edge_side, span, lc0, lc0 + lw - 1)
    else:
        starts = _strip_start_candidates(grid_size, edge_side, span, lr0, lr0 + lh - 1)
    if not starts:
        return None
    strip_start = random.choice(starts)

    _paint_edge_strip(grid, edge_side, strip_start, seg_len, legend_map)
    if edge_side == "bottom":
        occupied[-1, strip_start : strip_start + span] = 1
    else:
        occupied[strip_start : strip_start + span, -1] = 1

    legend_patch = np.zeros(legend_oriented.shape, dtype=int)
    legend_patch[legend_oriented] = body_colors[0]
    paste(grid, legend_patch, lr0, lc0, transparent=0)
    occupied[lr0 : lr0 + lh, lc0 : lc0 + lw] = 1

    _reserve_band(occupied, edge_side)
    occupied[lr0 : lr0 + lh, lc0 : lc0 + lw] = 1

    for pi in random.sample(range(1, NUM_PIECES), NUM_PIECES - 1):
        oriented = _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
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


def _gen_attempt():
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
    raise RuntimeError("failed to generate f560132c D8 input")
