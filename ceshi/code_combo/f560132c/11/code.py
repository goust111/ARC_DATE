# Combo variant of f560132c (index 11)
# combo_spec: D5/2+D8/0
# Name: 分散边缘图例+嵌入精灵内部

"""
Design analysis for the combo variant.

1. Concrete behavior and trace
   - Original behavior:
     one special jigsaw piece already contains the 2x2 color map, and that piece is
     scattered without reorientation. `main()` can therefore read the palette exactly
     as seen and use it directly after reassembling the pieces.
   - Combo behavior:
     one jigsaw piece still contains the full 2x2 palette, but that multicolor piece is
     now scattered under a random non-identity D4 transform. Separately, four detached
     singleton pixels are placed on the four canvas edges:
       top edge    -> target top-left output color
       right edge  -> target top-right output color
       left edge   -> target bottom-left output color
       bottom edge -> target bottom-right output color
     The embedded 2x2 block now gives the complete set and local arrangement of colors,
     while the edge markers tell the solver which oriented version of that 2x2 block is
     the true output legend.
   - Small trace:
     old solver step: "extract legend block from multicolor piece" -> valid only because
     the piece keeps canonical orientation.
     combo solver step:
       (a) strip the four border markers and decode the target edge legend,
       (b) find the unique multicolor piece and read its embedded 2x2 block,
       (c) try D4 orientations of that piece; only the orientation whose transformed
           2x2 block matches the edge legend may serve as the assembly anchor,
       (d) finish the exact-cover tiling and recolor by quadrant.

2. Complexity calibration
   - The base exact-cover task stays intact: 4 simply connected polyominoes still tile a
     unique square under rotations/reflections.
   - The visible difference is strong: the input now contains both a multicolor embedded
     block inside one sprite and four detached colored edge dots. This is clearly not the
     original single-source legend layout.
   - To avoid collapsing back to the D5-only shortcut, the generator does not require the
     embedded piece to be the unique largest piece. The solver must identify it by color
     structure, not by area.
   - To avoid collapsing back to the original task, the embedded legend piece is always
     scattered in a non-identity orientation, so reading its 2x2 block without edge-based
     reorientation would give the wrong quadrant assignment.

3. Broken assumptions and fixes
   - `len(comps)` no longer equals the number of pieces on the raw grid because four edge
     markers are extra components. Fix: detect and remove border singletons first.
   - "legend map can be used exactly as read from the multicolor piece" is false because
     the multicolor piece is reoriented in the input. Fix: compare transformed palette
     candidates against the edge legend before accepting an anchor orientation.
   - "anchor can be inferred from largest area" is not reliable in the combo. Fix: the
     anchor is the unique multicolor component among the four pieces.
   - Orientation deduplication by shape alone is insufficient for the anchor because two
     transforms may share the same mask but induce different palette orientations. Fix:
     the anchor is tested with all 8 D4 transforms, not only unique mask orientations.

4. Placement legality
   - Pieces are scattered with border margin 2 so border markers stay isolated under
     4-connectivity and never merge with a nearby piece.
   - The 2x2 embedded palette is placed strictly inside the anchor piece, with a full
     one-cell body ring around it, so it remains visibly part of the sprite after any D4
     transform.
   - Edge markers use non-corner border cells only, exactly one per side.

5. Visual observability and cleanliness
   - Across examples, the user sees two redundant but spatially separated information
     sources: a multicolor 2x2 block inside one sprite and four matching edge dots.
   - The edge dots sit alone on the border, while the embedded palette is fully enclosed
     by body-colored cells, so the two roles are visually distinct rather than cluttered.
   - Only four palette colors are involved, repeated between the two sources and then in
     the output quadrants. The mapping remains low-arity and learnable from few examples.

6. Fusion metric to check after implementation
   - Core metric A: in generated inputs, the embedded 2x2 block as read directly from the
     multicolor piece should differ from the edge-decoded legend in essentially all cases.
     This confirms the anchor was reoriented and the edge markers matter.
   - Core metric B: the multicolor anchor should not systematically be the unique largest
     piece, confirming the combo does not reduce to the D5-only largest-anchor heuristic.

7. New solver assumptions and generator guarantees
   - Assumption: exactly 4 singleton border markers exist, one per side.
     Guarantee: generator stamps exactly one top/right/left/bottom marker.
   - Assumption: after removing those markers, exactly 4 piece components remain and
     exactly one of them is multicolor.
     Guarantee: generator creates 4 pieces, with only the anchor containing the palette.
   - Assumption: some D4 transform of the anchor's embedded palette matches the edge legend.
     Guarantee: generator derives the input anchor by applying one non-identity D4 transform
     to a canonical anchor patch that already contains the correct palette.
"""

import math
import random

import numpy as np

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


def _dihedral_transforms_full(
    coords: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]]:
    out: list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]] = []
    cur = coords
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        out.append((c, h, w, int(rot_k), False))
        cur = _rot90(cur)[0]
    cur = _flip_h(coords)[0]
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
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
    assert best_rc is not None, "legend component must contain a valid k×k map window"
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
        assert side not in by_side, "expected exactly one border marker per side"
        by_side[side] = (r, c, int(g[r, c]))

    need = {"top", "right", "left", "bottom"}
    assert set(by_side) == need, "expected four singleton edge markers"
    legend_map = np.array(
        [
            [by_side["top"][2], by_side["right"][2]],
            [by_side["left"][2], by_side["bottom"][2]],
        ],
        dtype=int,
    )
    marker_cells = {(r, c) for r, c, _ in by_side.values()}
    return legend_map, marker_cells


def _orient_map(arr: np.ndarray, rot_k: int, flipped: bool) -> np.ndarray:
    out = np.asarray(arr, dtype=int)
    if flipped:
        out = np.fliplr(out)
    if rot_k:
        out = np.rot90(out, -int(rot_k))
    return np.asarray(out, dtype=int).copy()


def _assemble_exact_cover(
    pieces: list[dict],
    anchor_idx: int,
    edge_legend_map: np.ndarray,
    anchor_palette_input: np.ndarray,
    k: int,
) -> np.ndarray:
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = _factor_pairs(total)
    dims = [(H, W) for (H, W) in dims if H >= max_h and W >= max_w]
    assert len(dims) == 1, "piece areas and bounds must imply a unique output rectangle"

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
                    m = 0
                    for dr, dc in coords:
                        idx = (int(r0) + int(dr)) * int(W) + (int(c0) + int(dc))
                        m |= 1 << idx
                    opts.append((m, int(r0), int(c0), coords))
        placements_other.append(opts)

    def solve_with_fixed_anchor(
        anchor_coords: tuple[tuple[int, int], ...],
    ) -> dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] | None:
        anchor_mask = 0
        for dr, dc in anchor_coords:
            idx = int(dr) * int(W) + int(dc)
            anchor_mask |= 1 << idx

        filtered: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
        for opts in placements_other:
            filtered.append([pl for pl in opts if (pl[0] & anchor_mask) == 0])

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
            for key in sorted(remaining):
                opts_k = []
                for pl in filtered[key]:
                    m = pl[0]
                    if (m >> cell) & 1 and (m & used) == 0:
                        opts_k.append(pl)
                if opts_k:
                    candidates.append((len(opts_k), key, opts_k))
            if not candidates:
                return False

            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, key, opts_k in candidates:
                remaining.remove(key)
                pi = other_pis[key]
                for pl in opts_k:
                    chosen[pi] = pl
                    if search(used | pl[0]):
                        return True
                chosen.pop(pi, None)
                remaining.add(key)
            return False

        if not search(anchor_mask):
            return None
        return chosen

    chosen = None
    for coords, ph, pw, rot_k, flipped in pieces[anchor_idx]["all_oris"]:
        assert ph <= H and pw <= W, "anchor orientation must fit the unique output rectangle"
        oriented_palette = _orient_map(anchor_palette_input, rot_k, flipped)
        if not np.array_equal(oriented_palette, edge_legend_map):
            continue
        sol = solve_with_fixed_anchor(coords)
        if sol is None:
            continue
        chosen = sol
        break
    assert chosen is not None, "exact-cover assembly failed for every edge-consistent anchor orientation"

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
        col = int(edge_legend_map[rr, cc2])
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a square, recoloring them by a 2x2 legend reconstructed from both edge dots and an embedded sprite palette.

    Input structure:
    - bg is the most common color.
    - Four scattered non-bg sprites are the actual jigsaw pieces; exactly one of them is multicolor because it contains an interior 2x2 palette.
    - Four extra non-bg singleton pixels sit on the outer border, one on each side, and their colors indicate the target quadrant colors.

    Rule:
    - Remove the four border markers from the piece set and decode them as a 2x2 target legend: top/right/left/bottom map to top-left/top-right/bottom-left/bottom-right.
    - Find the unique multicolor piece, extract its interior 2x2 palette, and use the border legend to determine which D4 orientation of that piece is the true assembly anchor.
    - Tile the four pieces into the unique tight square, then recolor each assembled piece by the target legend color at its centroid quadrant.

    Key insight: The embedded palette contains the complete color arrangement but appears in the input under a scrambled piece orientation; the detached edge markers provide the canonical output orientation, so the solver must fuse the two cues before the usual shape assembly.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    edge_legend_map, marker_cells = _extract_edge_legend_map(g, bg)
    piece_mask = g != bg
    for r, c in marker_cells:
        piece_mask[r, c] = False

    comps = components_mask(piece_mask, connectivity=4)
    assert len(comps) == 4, "expected 4 piece components after stripping edge markers"

    k = int(round(len(comps) ** 0.5))
    assert k == 2 and k * k == len(comps), "piece count must remain 4"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)
    anchor_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) > 1]
    assert len(anchor_candidates) == 1, "expected exactly one multicolor anchor piece"
    anchor_idx = anchor_candidates[0]

    anchor_palette_input = _extract_legend_map(g, comps[anchor_idx], k)
    assert len(set(map(int, anchor_palette_input.ravel()))) == k * k, "embedded legend must use four distinct colors"

    pieces: list[dict] = []
    for i, pts in enumerate(comps):
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        piece = {
            "oris": _dihedral_orientations(rel_norm),
            "area": len(filled_abs),
        }
        if i == anchor_idx:
            piece["all_oris"] = _dihedral_transforms_full(rel_norm)
        pieces.append(piece)

    return _assemble_exact_cover(pieces, anchor_idx, edge_legend_map, anchor_palette_input, k)


# --------------- generator ---------------

SIDE_CHOICES = [8, 10, 12]
K = 2
NUM_PIECES = K * K
NUM_COLORS = 8
AREA_FRAC_LO = 0.08
AREA_FRAC_HI = 0.43
AREA_ABS_LO = 11
AREA_ABS_HI = 62
ANCHOR_THRESHOLD = {8: 4, 10: 5, 12: 9}
ANCHOR_CORE_SIZE = 4
ANCHOR_ARM_RANGE = range(1, 4)
PARTITION_GROWTH_EXP = 1.45
PARTITION_FRONTIER_EXP = 0.2
PARTITION_SEED_JITTER = 0.2
INPUT_SIZE_LO = 20
INPUT_SIZE_HI = 26
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 2
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


def _make_anchor_mask(side: int, min_area: int, max_area: int) -> np.ndarray:
    threshold = ANCHOR_THRESHOLD[side]
    opts = []
    for h0 in range(threshold + 1, side + 1):
        for w0 in range(threshold + 1, side + 1):
            for arm_w in ANCHOR_ARM_RANGE:
                for arm_h in ANCHOR_ARM_RANGE:
                    mask = np.zeros((side, side), dtype=bool)
                    mask[:ANCHOR_CORE_SIZE, :ANCHOR_CORE_SIZE] = True
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


def _crop_nonzero_patch(patch: np.ndarray) -> np.ndarray:
    pts = np.argwhere(patch != 0)
    r0, c0, r1, c1 = bbox_coords(pts)
    return np.asarray(patch[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _transform_patch(patch: np.ndarray, rot_k: int, flipped: bool) -> np.ndarray:
    out = np.asarray(patch, dtype=int)
    if flipped:
        out = np.fliplr(out)
    if rot_k:
        out = np.rot90(out, -int(rot_k))
    return _crop_nonzero_patch(out)


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(int)
    opts = []
    seen = set()
    for rot_k in range(4):
        rot = np.rot90(arr, rot_k)
        for oriented in (rot, np.fliplr(rot)):
            crop = _crop_nonzero_patch(oriented)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop.astype(bool))
    return random.choice(opts)


def _palette_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(1, h - K):
        for c in range(1, w - K):
            if mask[r : r + K, c : c + K].all() and mask[r - 1 : r + K + 1, c - 1 : c + K + 1].all():
                spots.append((r, c))
    return spots


def _embed_palette_in_anchor(mask: np.ndarray, body_color: int, legend_map: np.ndarray) -> np.ndarray | None:
    crop = _crop_mask(mask)
    patch = np.zeros(crop.shape, dtype=int)
    patch[crop] = body_color
    spots = _palette_spots(crop)
    if not spots:
        return None
    r, c = random.choice(spots)
    patch[r : r + K, c : c + K] = legend_map
    return patch


def _edge_marker_positions(grid_size: int) -> dict[str, tuple[int, int]]:
    mid = grid_size // 2
    top_cols = list(range(2, mid))
    right_rows = list(range(2, mid))
    left_rows = list(range(mid, grid_size - 2))
    bottom_cols = list(range(mid, grid_size - 2))
    assert top_cols and right_rows and left_rows and bottom_cols, "grid too small for edge markers"
    return {
        "top": (0, random.choice(top_cols)),
        "right": (random.choice(right_rows), grid_size - 1),
        "left": (random.choice(left_rows), 0),
        "bottom": (grid_size - 1, random.choice(bottom_cols)),
    }


def _stamp_edge_markers(grid: np.ndarray, legend_map: np.ndarray, bg: int) -> None:
    h, w = grid.shape
    assert h == w, "generated inputs are square"
    coords = _edge_marker_positions(h)
    side_to_color = {
        "top": int(legend_map[0, 0]),
        "right": int(legend_map[0, 1]),
        "left": int(legend_map[1, 0]),
        "bottom": int(legend_map[1, 1]),
    }
    for side, (r, c) in coords.items():
        assert int(grid[r, c]) == bg, "marker cell must be empty"
        if side == "top":
            assert int(grid[r + 1, c]) == bg, "top marker must stay isolated"
        elif side == "bottom":
            assert int(grid[r - 1, c]) == bg, "bottom marker must stay isolated"
        elif side == "left":
            assert int(grid[r, c + 1]) == bg, "left marker must stay isolated"
        else:
            assert int(grid[r, c - 1]) == bg, "right marker must stay isolated"
        grid[r, c] = side_to_color[side]


def _make_anchor_input_patch(anchor_patch: np.ndarray, legend_map: np.ndarray) -> np.ndarray | None:
    candidates = []
    for flipped in (False, True):
        for rot_k in range(4):
            if not flipped and rot_k == 0:
                continue
            oriented = _transform_patch(anchor_patch, rot_k, flipped)
            embedded = _extract_legend_map(oriented, np.argwhere(oriented != 0), K)
            if np.array_equal(embedded, legend_map):
                continue
            key = (oriented.shape, tuple(oriented.ravel()))
            candidates.append((key, oriented))
    if not candidates:
        return None
    uniq: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    for key, oriented in candidates:
        uniq[key] = oriented
    return random.choice(list(uniq.values())).copy()


def _scatter_pieces(masks: list[np.ndarray], body_colors: list[int], legend_map: np.ndarray, side: int) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    anchor_patch = _embed_palette_in_anchor(masks[0], body_colors[0], legend_map)
    if anchor_patch is None:
        return None
    anchor_input_patch = _make_anchor_input_patch(anchor_patch, legend_map)
    if anchor_input_patch is None:
        return None

    patches: list[np.ndarray] = [anchor_input_patch]
    for pi in range(1, NUM_PIECES):
        oriented = _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        patches.append(patch)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        patch = patches[pi]
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
    return grid


def _gen_attempt() -> np.ndarray | None:
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


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c combo variant input")
