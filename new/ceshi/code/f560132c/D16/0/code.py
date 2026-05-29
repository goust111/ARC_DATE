# Variant of f560132c
# Dimension: D16 — 象限分割线的形状（分割拓扑） (meta)
# Change: 矩形被分为4个象限，分割线为一条水平线和一条垂直线交叉（十字分割），但交叉点可以偏离中心... → 对角线分割

import itertools
import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the D16 variant
# -----------------------------------
# a) Concrete behavior:
#    The assembled output is still one tight square, but the 4 pieces are now staircase triangles
#    produced by a diagonal X-like partition. They correspond to the four side sectors:
#    top / right / bottom / left, not the old axis-aligned rectangular quadrants.
#    The legend is still a 2x2 palette block, but its colors are read clockwise:
#    top <- TL, right <- TR, bottom <- BR, left <- BL.
#    Example of the changed step:
#      - Old rule after assembly: piece color = legend cell from centroid quantized into a 2x2 box.
#      - New rule after assembly: piece color = legend cell from which border sector the piece owns
#        in the diagonal partition.
#      - Difference: the top triangle can have a centroid in the old top-left or top-right box, so
#        axis-aligned quantization is no longer valid.
# b) Complexity calibration:
#    The split still has only 4 regions and a fixed 2x2 legend, so the rule is inferable from a few
#    examples. The generator varies the diagonal intersection by shifting the sector boundaries, which
#    makes the triangles visibly different across samples without adding extra symbolic cases.
# c) Implicit assumptions audited from the original solver:
#      - Exact-cover assembly under D4 is still valid and is kept unchanged.
#      - `_extract_legend_map` still works because the legend remains an interior 2x2 color block.
#      - The old centroid->quadrant code depended on rectangular quadrants; that assumption is removed.
#      - The new solver instead assumes each assembled piece has one dominant border among
#        top/right/bottom/left. The generator guarantees that by constructing the pieces directly from
#        the four diagonal sectors and rejecting any ambiguous role assignment.
# d) Placement legality:
#    Input pieces are scattered with the same non-overlap helper and padding as before. The legend
#    piece is chosen as the top sector because its assembled bbox starts at (0,0), matching the exact-
#    cover solver's fixed-origin legend placement.
# e) Visual observability:
#    The output now shows four large triangles meeting around diagonal boundaries; this is visibly
#    different from the old rectangular blocks. The generator also rejects cases where the old
#    axis-aligned centroid rule would produce the same coloring.
# f) Visual cleanliness:
#    Only one object layer is present per cell. The 2x2 palette stays fully inside the legend body,
#    and scattering uses padding so pieces remain easy to isolate.
# g) Color inference:
#    Colors remain only the 4 body colors plus the 4 palette colors. The palette-to-output mapping is
#    structural: the four legend colors appear around the output in clockwise order under the diagonal
#    split, so no hidden color-id ordering is used.


ROLE_TOP = 0
ROLE_RIGHT = 1
ROLE_BOTTOM = 2
ROLE_LEFT = 3
ROLE_ORDER = (ROLE_TOP, ROLE_RIGHT, ROLE_BOTTOM, ROLE_LEFT)
ROLE_TO_LEGEND_CELL = {
    ROLE_TOP: (0, 0),
    ROLE_RIGHT: (0, 1),
    ROLE_BOTTOM: (1, 1),
    ROLE_LEFT: (1, 0),
}


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
    assert k > 0 and hh >= k and ww >= k, 'legend component must fit k×k map'
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
    assert best_rc is not None, 'legend component must contain a valid k×k color map window'
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


def _piece_border_scores(r0: int, c0: int, coords: tuple[tuple[int, int], ...], H: int, W: int) -> tuple[int, int, int, int]:
    top = sum(1 for dr, _ in coords if r0 + dr == 0)
    right = sum(1 for _, dc in coords if c0 + dc == W - 1)
    bottom = sum(1 for dr, _ in coords if r0 + dr == H - 1)
    left = sum(1 for _, dc in coords if c0 + dc == 0)
    return int(top), int(right), int(bottom), int(left)


def _assign_roles(
    chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]],
    H: int,
    W: int,
) -> dict[int, int]:
    piece_ids = sorted(chosen)
    scores = {pi: _piece_border_scores(chosen[pi][1], chosen[pi][2], chosen[pi][3], H, W) for pi in piece_ids}
    best = None
    best_perm = None
    ties = 0
    for perm in itertools.permutations(ROLE_ORDER):
        total = sum(scores[pi][perm[idx]] for idx, pi in enumerate(piece_ids))
        if best is None or total > best:
            best = total
            best_perm = perm
            ties = 1
        elif total == best:
            ties += 1
    assert best_perm is not None and ties == 1, 'assembled diagonal roles must be uniquely determined'
    role_of = {pi: best_perm[idx] for idx, pi in enumerate(piece_ids)}
    return role_of


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

        role_of = _assign_roles(chosen, H, W)

        out = np.zeros((H, W), dtype=int)
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            lr, lc = ROLE_TO_LEGEND_CELL[role_of[pi]]
            col = int(legend_map[lr, lc])
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered triangular puzzle pieces into a square and recolor them from a legend sprite.

    Input structure:
    - bg is the most common color.
    - There are four 4-connected staircase-triangle pieces scattered on the canvas.
    - One piece is the legend: it is the only multi-color component and contains an interior 2×2 palette.

    Rule:
    - Tile the four pieces into the unique full square, allowing rotations and reflections.
    - Treat the assembled square as being split by diagonal sector boundaries into four triangular regions:
      top, right, bottom, and left.
    - Recolor each assembled piece using the legend palette in clockwise order:
      top gets the palette's top-left color, right gets top-right, bottom gets bottom-right,
      and left gets bottom-left.

    Key insight:
    The legend still provides four colors, but their meaning is no longer "which rectangular quadrant
    contains the centroid". Instead, each color belongs to one diagonal side-sector of the final square.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, 'grid must contain at least one non-background component'

    k = int(round(len(comps) ** 0.5))
    assert k * k == len(comps), 'piece count must be a perfect square to determine k'
    assert k == 2, 'this variant is defined for four pieces'

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, 'legend component must be uniquely identified by distinct colors'
    legend_idx = legend_candidates[0]
    legend_map = _extract_legend_map(g, comps[legend_idx], k)

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

SIDE_CHOICES = [10, 12]
K = 2
NUM_PIECES = K * K
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


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(int)
    opts = []
    seen = set()
    for k2 in range(4):
        rot = np.rot90(arr, k2)
        for oriented in (rot, np.fliplr(rot)):
            crop = _crop_mask(oriented > 0).astype(bool)
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


def _build_sector_masks(side: int, row_shift: int, col_shift: int) -> list[np.ndarray]:
    masks = [np.zeros((side, side), dtype=bool) for _ in range(NUM_PIECES)]
    for r in range(side):
        for c in range(side):
            vr = 2 * r + 1 - side - row_shift
            vc = 2 * c + 1 - side - col_shift
            if abs(vr) > abs(vc):
                idx = ROLE_TOP if vr < 0 else ROLE_BOTTOM
            else:
                idx = ROLE_LEFT if vc < 0 else ROLE_RIGHT
            masks[idx][r, c] = True
    return masks


def _old_cross_cell(mask: np.ndarray, side: int) -> tuple[int, int]:
    pts = np.argwhere(mask)
    cr = float(pts[:, 0].mean())
    cc = float(pts[:, 1].mean())
    rr = max(0, min(1, int(cr * K / side)))
    cc2 = max(0, min(1, int(cc * K / side)))
    return rr, cc2


def _expected_from_masks(masks: list[np.ndarray], legend_map: np.ndarray, use_diagonal: bool) -> np.ndarray:
    side = masks[0].shape[0]
    out = np.zeros((side, side), dtype=int)
    for role, mask in enumerate(masks):
        if use_diagonal:
            rr, cc = ROLE_TO_LEGEND_CELL[role]
        else:
            rr, cc = _old_cross_cell(mask, side)
        out[mask] = int(legend_map[rr, cc])
    return out


def _forces_unique_square(masks: list[np.ndarray], side: int) -> bool:
    bbs = [bbox_coords(np.argwhere(m)) for m in masks]
    max_h = max(r1 - r0 + 1 for r0, _, r1, _ in bbs)
    max_w = max(c1 - c0 + 1 for _, c0, _, c1 in bbs)
    dims = [(h, w) for h, w in _factor_pairs(side * side) if h >= max_h and w >= max_w]
    return dims == [(side, side)]


def _candidate_sector_configs(side: int) -> list[tuple[int, int]]:
    min_area, max_area = _area_bounds(side)
    candidates = []
    for row_shift in range(-(side // 3), side // 3 + 1):
        for col_shift in range(-(side // 3), side // 3 + 1):
            if (row_shift - col_shift) % 2 == 0:
                continue
            masks = _build_sector_masks(side, row_shift, col_shift)
            areas = [int(m.sum()) for m in masks]
            if any(a < min_area or a > max_area for a in areas):
                continue
            if len(_palette_spots(masks[ROLE_TOP])) == 0:
                continue
            if bbox_coords(np.argwhere(masks[ROLE_TOP]))[:2] != (0, 0):
                continue
            if not _forces_unique_square(masks, side):
                continue
            chosen = {i: (0, 0, 0, tuple(map(tuple, np.argwhere(masks[i])))) for i in range(NUM_PIECES)}
            try:
                _assign_roles(chosen, side, side)
            except AssertionError:
                continue
            test_map = np.array([[1, 2], [3, 4]], dtype=int)
            if np.array_equal(_expected_from_masks(masks, test_map, True), _expected_from_masks(masks, test_map, False)):
                continue
            candidates.append((row_shift, col_shift))
    assert candidates, f'no valid diagonal sector configurations for side={side}'
    return candidates


def _scatter_pieces(masks: list[np.ndarray], body_colors: list[int], legend_map: np.ndarray, side: int) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _crop_mask(masks[pi]) if pi == ROLE_TOP else _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        if pi == ROLE_TOP:
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


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SIDE_CHOICES)
    row_shift, col_shift = random.choice(_candidate_sector_configs(side))
    masks = _build_sector_masks(side, row_shift, col_shift)

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_PIECES]
    palette_colors = colors[NUM_PIECES:]
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)

    expected = _expected_from_masks(masks, legend_map, True)
    old_like = _expected_from_masks(masks, legend_map, False)
    if np.array_equal(expected, old_like):
        return None

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
    raise RuntimeError('failed to generate f560132c D16 input')
