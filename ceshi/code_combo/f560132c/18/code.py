# Combo variant of f560132c (index 18)
# combo_spec: D11/1+D13/2
# Name: 锯齿边界+对称精灵

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the fused variant.
# a) Concrete behavior trace:
#    Original task: four scattered monochrome pieces are tiled into a square; the legend piece is only
#    special because it embeds a 2x2 palette, and piece colors are assigned by assembled position.
#    Fused task: the assembled square is still a 2x2 partition, but both inner seams are jagged and share
#    the same notch profile. Example for side=10: the vertical split is [5,5,4,5,5,...] by row instead of
#    the straight line c=5, and the horizontal split repeats the same [5,5,4,5,5,...] pattern by column.
#    - original TL/TR and TL/BL boundaries were straight; fused boundaries show the same one-cell inward bite.
#    - the horizontal seam reuses the vertical seam's profile after swapping row/col, so the notch is a
#      shared structural cue rather than an unrelated irregularity.
#    - TL keeps diagonal symmetry, TR is horizontally symmetric, BL is vertically symmetric, and BR is a
#      symmetric block, so piece symmetry and seam shape reinforce each other.
# b) Complexity calibration:
#    The staircase profile only has two levels (half and half+1) and is monotone on the top half.
#    This keeps the seam visibly non-straight but not noisy, and the repeated profile across both axes
#    gives humans a stable cue across 2-4 examples.
# c) Broken implicit assumptions from the base code:
#    - base solver used centroid quantization to assign colors; here quadrant membership is better read
#      from which piece reaches each outer corner because jagged seams stay near the center.
#    - base generator assumed arbitrary connected polyominoes; here every piece must satisfy a reflective
#      symmetry constraint, so both solver and generator validate symmetry axes.
#    - base exact-cover search accepted any rectangle tiling; here we reject tilings unless rows/cols form
#      the same non-constant left/right and top/bottom split profile.
# d) Placement legality:
#    Generation is output-first. The square partition is created before scattering, then each cropped piece
#    is placed with `random_free_location`, so pieces never overlap or leave the canvas.
# e) Visual observability:
#    The fusion is human-visible because the output seams are not straight and the piece silhouettes show
#    mirrored stair patterns. The same jagged profile appears in both the vertical and horizontal joins.
# f) Visual cleanliness:
#    Only one multicolor component exists (the legend), and the palette block is embedded in a solid
#    interior patch of the legend piece. Scattering keeps at least one-cell padding between bounding boxes.
# g) Inferable change rule:
#    Colors still come from the 2x2 legend, but quadrant assignment now aligns with the four corner-reaching
#    symmetric pieces. The seam shape is structural rather than color-dependent, so nothing relies on raw
#    color ids or hidden ordering.


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
    norm, h, _ = _normalize_coords(coords)
    rot = [(c, h - 1 - r) for r, c in norm]
    return _normalize_coords(rot)


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, _, w = _normalize_coords(coords)
    flp = [(r, w - 1 - c) for r, c in norm]
    return _normalize_coords(flp)


def _local_symmetry_axes(coords: tuple[tuple[int, int], ...]) -> frozenset[str]:
    norm, h, w = _normalize_coords(coords)
    pts = set(norm)
    axes = set()
    if all((r, w - 1 - c) in pts for r, c in norm):
        axes.add("vertical")
    if all((h - 1 - r, c) in pts for r, c in norm):
        axes.add("horizontal")
    if h == w and all((c, r) in pts for r, c in norm):
        axes.add("main_diag")
    if h == w and all((w - 1 - c, h - 1 - r) in pts for r, c in norm):
        axes.add("anti_diag")
    return frozenset(axes)


def _dihedral_orientations(
    coords: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[tuple[int, int], ...], int, int, int, bool, frozenset[str]]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[tuple[int, int], ...], int, int, int, bool, frozenset[str]]] = []
    cur = coords
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), False, _local_symmetry_axes(c)))
        cur = _rot90(cur)[0]
    cur = _flip_h(coords)[0]
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), True, _local_symmetry_axes(c)))
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
    assert best_rc is not None, "legend component must contain a valid kxk palette"
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


def _owner_grid(
    chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...], frozenset[str]]],
    H: int,
    W: int,
) -> np.ndarray:
    owner = np.full((H, W), -1, dtype=int)
    for pi, (_, r0, c0, coords, _) in chosen.items():
        for dr, dc in coords:
            rr = r0 + dr
            cc = c0 + dc
            assert owner[rr, cc] < 0, "piece overlap in assembled rectangle"
            owner[rr, cc] = pi
    assert np.all(owner >= 0), "assembled rectangle must be fully tiled"
    return owner


def _extract_seam_profile(owner: np.ndarray, left_family: set[int], right_family: set[int]) -> list[int] | None:
    H, W = owner.shape
    profile = []
    for r in range(H):
        row = owner[r]
        split = 0
        while split < W and int(row[split]) in left_family:
            split += 1
        if split == 0 or split == W:
            return None
        if not np.all(np.isin(row[:split], list(left_family))):
            return None
        if not np.all(np.isin(row[split:], list(right_family))):
            return None
        profile.append(int(split))
    return profile


def _extract_horizontal_profile(owner: np.ndarray, top_family: set[int], bottom_family: set[int]) -> list[int] | None:
    H, W = owner.shape
    profile = []
    for c in range(W):
        col = owner[:, c]
        split = 0
        while split < H and int(col[split]) in top_family:
            split += 1
        if split == 0 or split == H:
            return None
        if not np.all(np.isin(col[:split], list(top_family))):
            return None
        if not np.all(np.isin(col[split:], list(bottom_family))):
            return None
        profile.append(int(split))
    return profile


def _validate_fused_layout(
    chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...], frozenset[str]]],
    legend_idx: int,
    H: int,
    W: int,
) -> dict[int, tuple[int, int]] | None:
    if H != W:
        return None
    owner = _owner_grid(chosen, H, W)
    tl = int(owner[0, 0])
    tr = int(owner[0, W - 1])
    bl = int(owner[H - 1, 0])
    br = int(owner[H - 1, W - 1])
    if len({tl, tr, bl, br}) != 4 or tl != legend_idx:
        return None

    piece_axes = {pi: axes for pi, (_, _, _, _, axes) in chosen.items()}
    if "main_diag" not in piece_axes[tl]:
        return None
    if "horizontal" not in piece_axes[tr]:
        return None
    if "vertical" not in piece_axes[bl]:
        return None
    if not piece_axes[br]:
        return None

    vprof = _extract_seam_profile(owner, {tl, bl}, {tr, br})
    hprof = _extract_horizontal_profile(owner, {tl, tr}, {bl, br})
    if vprof is None or hprof is None:
        return None
    if vprof != hprof:
        return None
    if len(set(vprof)) < 2:
        return None

    return {tl: (0, 0), tr: (0, 1), bl: (1, 0), br: (1, 1)}


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray:
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _, _ in p["oris"]) for p in pieces)
    dims = [(H, W) for (H, W) in _factor_pairs(total) if H >= max_h and W >= max_w]
    assert len(dims) == 1, "piece areas and bounds must imply a unique output rectangle"
    H, W = dims[0]
    nbits = H * W
    all_mask = (1 << nbits) - 1

    other_pis = [i for i in range(len(pieces)) if i != legend_idx]
    placements_other: list[list[tuple[int, int, int, tuple[tuple[int, int], ...], frozenset[str]]]] = []
    for pi in other_pis:
        opts: list[tuple[int, int, int, tuple[tuple[int, int], ...], frozenset[str]]] = []
        for coords, ph, pw, _, _, axes in pieces[pi]["oris"]:
            for r0 in range(0, H - ph + 1):
                for c0 in range(0, W - pw + 1):
                    m = 0
                    for dr, dc in coords:
                        idx = (int(r0) + int(dr)) * int(W) + (int(c0) + int(dc))
                        m |= 1 << int(idx)
                    opts.append((m, int(r0), int(c0), coords, axes))
        placements_other.append(opts)

    def solve_with_fixed_legend(
        legend_coords: tuple[tuple[int, int], ...],
        legend_axes: frozenset[str],
    ) -> tuple[dict[int, tuple[int, int, int, tuple[tuple[int, int], ...], frozenset[str]]], dict[int, tuple[int, int]]] | None:
        legend_mask = 0
        for dr, dc in legend_coords:
            idx = int(dr) * int(W) + int(dc)
            legend_mask |= 1 << int(idx)

        filtered: list[list[tuple[int, int, int, tuple[tuple[int, int], ...], frozenset[str]]]] = []
        for opts in placements_other:
            filtered.append([pl for pl in opts if (pl[0] & legend_mask) == 0])

        remaining = set(range(len(other_pis)))
        chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...], frozenset[str]]] = {
            legend_idx: (legend_mask, 0, 0, legend_coords, legend_axes)
        }

        def search(used: int) -> tuple[dict[int, tuple[int, int, int, tuple[tuple[int, int], ...], frozenset[str]]], dict[int, tuple[int, int]]] | None:
            if not remaining:
                if used != all_mask:
                    return None
                cell_map = _validate_fused_layout(chosen, legend_idx, H, W)
                if cell_map is None:
                    return None
                return dict(chosen), cell_map

            empty = (~used) & all_mask
            if empty == 0:
                return None
            cell = (empty & -empty).bit_length() - 1
            candidates = []
            for idx in sorted(remaining):
                opts_idx = []
                for pl in filtered[idx]:
                    m = pl[0]
                    if ((m >> cell) & 1) and (m & used) == 0:
                        opts_idx.append(pl)
                if opts_idx:
                    candidates.append((len(opts_idx), idx, opts_idx))
            if not candidates:
                return None

            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, idx, opts_idx in candidates:
                remaining.remove(idx)
                pi = other_pis[idx]
                for pl in opts_idx:
                    chosen[pi] = pl
                    solved = search(used | pl[0])
                    if solved is not None:
                        return solved
                chosen.pop(pi, None)
                remaining.add(idx)
            return None

        return search(legend_mask)

    solved = None
    for coords, ph, pw, _, _, axes in pieces[legend_idx]["oris"]:
        assert ph <= H and pw <= W, "legend orientation must fit the output rectangle"
        candidate = solve_with_fixed_legend(coords, axes)
        if candidate is not None:
            solved = candidate
            break
    assert solved is not None, "no fused exact-cover assembly found"
    chosen, cell_map = solved

    out = np.zeros((H, W), dtype=int)
    for pi, (_, r0, c0, coords, _) in chosen.items():
        rr, cc = cell_map[pi]
        col = int(legend_map[rr, cc])
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four scattered symmetric sprites into a square whose inner joins share one jagged seam profile.

    Input structure:
    - Background is the most frequent color.
    - There are exactly four disconnected sprites; three are monochrome body pieces and one legend piece
      is multicolored because it contains an embedded 2x2 palette block.
    - Every sprite is reflective-symmetric under some axis, and the intended assembly uses those symmetric
      silhouettes as clues for where the zigzag seams must interlock.

    Rule:
    - Reassemble the four sprites into their unique square exact cover, allowing rotations and reflections.
    - Keep only assemblies whose left/right and top/bottom splits form the same non-straight seam profile,
      and whose corner pieces preserve the expected symmetry axes.
    - Color each assembled piece by the legend's 2x2 palette cell matching the corner quadrant it occupies.

    Key insight:
    - The legend is both a normal piece and the color key, but the fused cue is structural: the four pieces
      are symmetric and the two internal seams reuse one staircase-like notch profile. Once the corner
      ownership of that shared profile is recovered, the palette colors attach directly to TL/TR/BL/BR.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) == 4, "fused variant expects exactly four pieces"
    k = int(round(len(comps) ** 0.5))
    assert k == 2, "this variant uses a 2x2 palette and four pieces"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)
    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend component must be uniquely multicolored"
    legend_idx = legend_candidates[0]
    legend_map = _extract_legend_map(g, comps[legend_idx], k)

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        assert any(axes for _, _, _, _, _, axes in oris), "each sprite must have at least one mirror axis"
        pieces.append({"oris": oris, "area": len(filled_abs)})

    return _assemble_exact_cover(pieces, legend_idx, legend_map, k)


# --------------- generator (fused symmetric seam construction) ---------------

SIDE_CHOICES = [10]
K = 2
NUM_PIECES = 4
NUM_COLORS = 8
AREA_FRAC_LO = 0.08
AREA_FRAC_HI = 0.43
AREA_ABS_LO = 11
AREA_ABS_HI = 62
INPUT_SIZE_LO = 20
INPUT_SIZE_HI = 26
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 300


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
            pts = np.argwhere(oriented > 0)
            r0, c0, r1, c1 = bbox_coords(pts)
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


def _build_piece_masks(side: int) -> list[np.ndarray]:
    half = side // 2
    labels = np.zeros((side, side), dtype=int)
    labels[:half, :half] = 0
    labels[:half, half:] = 1
    labels[half:, :half] = 2
    labels[half:, half:] = 3

    notch_rows = [half - 3]
    for r in notch_rows:
        labels[r, half - 1] = 1
        labels[half - 1, r] = 2
        labels[side - 1 - r, half] = 3
        labels[half, side - 1 - r] = 3
    return [(labels == i) for i in range(NUM_PIECES)]


def _symmetry_signature(mask: np.ndarray) -> frozenset[str]:
    pts = np.argwhere(mask)
    rel = tuple(map(tuple, pts))
    norm, _, _ = _normalize_coords(rel)
    return _local_symmetry_axes(norm)


def _expected_from_masks(masks: list[np.ndarray], legend_map: np.ndarray) -> np.ndarray:
    side = masks[0].shape[0]
    out = np.zeros((side, side), dtype=int)
    quadrant_cells = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    for pi, mask in enumerate(masks):
        rr, cc = quadrant_cells[pi]
        out[mask] = int(legend_map[rr, cc])
    return out


def _scatter_pieces(masks: list[np.ndarray], body_colors: list[int], legend_map: np.ndarray, side: int) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _random_orientation(_crop_mask(masks[pi]))
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


def _fusion_metrics_from_masks(masks: list[np.ndarray]) -> dict[str, int]:
    side = masks[0].shape[0]
    owner = np.full((side, side), -1, dtype=int)
    for pi, mask in enumerate(masks):
        owner[mask] = pi
    vprof = _extract_seam_profile(owner, {0, 2}, {1, 3})
    hprof = _extract_horizontal_profile(owner, {0, 1}, {2, 3})
    assert vprof is not None and hprof is not None
    half = side // 2
    return {
        "v_levels": len(set(vprof)),
        "h_levels": len(set(hprof)),
        "indented_rows": sum(int(v < half) for v in vprof[:half]),
        "indented_cols": sum(int(v < half) for v in hprof[:half]),
        "piece_sym_axes": sum(len(_symmetry_signature(_crop_mask(mask))) for mask in masks),
    }


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    side = random.choice(SIDE_CHOICES)
    min_area, max_area = _area_bounds(side)
    masks = _build_piece_masks(side)

    areas = [int(m.sum()) for m in masks]
    if any(a < min_area or a > max_area for a in areas):
        return None
    if any(len(components_mask(m, connectivity=4)) != 1 or _has_hole(m) for m in masks):
        return None
    if not _forces_unique_square(masks, side):
        return None
    if "main_diag" not in _symmetry_signature(_crop_mask(masks[0])):
        return None
    if "horizontal" not in _symmetry_signature(_crop_mask(masks[1])):
        return None
    if "vertical" not in _symmetry_signature(_crop_mask(masks[2])):
        return None
    if not _symmetry_signature(_crop_mask(masks[3])):
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_PIECES]
    palette_colors = colors[NUM_PIECES:]
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)
    expected = _expected_from_masks(masks, legend_map)

    grid = _scatter_pieces(masks, body_colors, legend_map, side)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, _fusion_metrics_from_masks(masks)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        made = _gen_attempt()
        if made is not None:
            return made[0]
    raise RuntimeError("failed to generate fused f560132c input")
