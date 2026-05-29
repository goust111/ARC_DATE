# Combo variant of f560132c (index 15)
# combo_spec: D7/1+D8/2
# Name: 先拼装再查图例+隐式颜色编码

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location

# Design analysis for the combo variant
# - Concrete behavior and trace:
#   The source task exposes one multicolor legend component first, so the solver can read a visible 2x2 palette
#   before doing any assembly. Here every input component is monochrome and the four colors themselves are the
#   hidden code. A concrete 2x2 trace is:
#     source:   read visible palette [[a,b],[c,d]] from one sprite -> assemble square -> color quadrants TL/TR/BL/BR
#               with a/b/c/d.
#     combo:    assemble four monochrome sprites first -> observe which input color lands in each assembled quadrant
#               -> build a hidden legend by swapping both axes, so hidden_map[TL]=color(BR), hidden_map[TR]=color(BL),
#               hidden_map[BL]=color(TR), hidden_map[BR]=color(TL) -> recolor each assembled piece from that map.
#   The visible difference is decisive: there is no multicolor sprite at all, and the "legend lookup" only becomes
#   available after the geometry has been solved.
# - Complexity calibration:
#   Removing the explicit legend introduces a D4 ambiguity: many tilings would be equivalent up to global rotation
#   or reflection, which would make the post-assembly hidden legend ambiguous too. To keep the task inferable rather
#   than arbitrary, generation preserves one asymmetric anchor piece in canonical orientation while every other piece
#   is scattered in a different D4 orientation. This keeps the required reasoning order as "shape assembly first",
#   but still yields a unique assembled square and a unique hidden 2x2 legend.
# - Old assumptions that break in the original main():
#   The source solver assumes one component has the most distinct colors, extracts a local kxk palette window from it,
#   and uses that component as a fixed legend piece at the top-left of the output. None of those premises survive:
#   every piece has exactly one non-background color; there is no embedded kxk palette window; and color assignment is
#   now a global relation among all four pieces after assembly. The replacement solver therefore must:
#   (1) read one body color per component,
#   (2) solve the exact cover with an orientation anchor,
#   (3) derive the hidden legend from assembled quadrant occupancy,
#   (4) recolor from that derived legend.
# - Placement legality:
#   The generator still works output-first, partitions an SxS square into four connected hole-free regions, keeps the
#   anchor canonical, reorients the other three pieces, and scatters all cropped sprites with border and padding so
#   components never overlap or touch in the input grid.
# - Visual observability and inferability:
#   No step depends on numeric color ids, component ordering, or hidden metadata. The color law is a structural
#   diagonal swap over the assembled 2x2 quadrant layout, so 2-3 examples are enough to see that output colors come
#   from opposite input quadrants, not from a visible legend patch.
# - Visual cleanliness:
#   Inputs contain only four separated monochrome sprites, so there is no palette-over-body overlay. Outputs remain
#   a tight recolored square with one clean color per piece.
# - New implicit assumptions and how generation guarantees them:
#   Each piece must have a distinct color; the anchor must be asymmetric and stay canonical; the other three pieces
#   must each admit at least one non-canonical D4 orientation; centroid quantization must send the four assembled
#   pieces to four different 2x2 cells. The generator checks all of these conditions before accepting a sample.


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


def _component_color(g: np.ndarray, pts: np.ndarray, bg: int) -> int:
    cols = {int(g[r, c]) for r, c in pts if int(g[r, c]) != bg}
    assert len(cols) == 1, 'each piece must be monochrome in the combo variant'
    return next(iter(cols))


def _placement_mask(coords: tuple[tuple[int, int], ...], r0: int, c0: int, width: int) -> int:
    m = 0
    for dr, dc in coords:
        idx = (int(r0) + int(dr)) * int(width) + (int(c0) + int(dc))
        m |= 1 << idx
    return m


def _piece_cell_from_coords(
    coords: tuple[tuple[int, int], ...],
    r0: int,
    c0: int,
    height: int,
    width: int,
    k: int,
) -> tuple[int, int]:
    rs = [r0 + dr for dr, _ in coords]
    cs = [c0 + dc for _, dc in coords]
    cr = float(sum(rs)) / float(len(rs))
    cc = float(sum(cs)) / float(len(cs))
    rr = int(cr * float(k) / float(height))
    cc2 = int(cc * float(k) / float(width))
    rr = max(0, min(int(k) - 1, rr))
    cc2 = max(0, min(int(k) - 1, cc2))
    return rr, cc2


def _rectangle_dims(pieces: list[dict]) -> tuple[int, int]:
    total = sum(int(p['area']) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p['oris']) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p['oris']) for p in pieces)
    dims = [(h, w) for h, w in _factor_pairs(total) if h >= max_h and w >= max_w]
    assert len(dims) == 1, 'piece areas and bounds must imply a unique output rectangle'
    return dims[0]


def _find_unique_assembly(pieces: list[dict], k: int) -> tuple[int, int, dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]]]:
    height, width = _rectangle_dims(pieces)
    all_mask = (1 << (height * width)) - 1

    placements: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
    for piece in pieces:
        opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
        for coords, ph, pw, _, _ in piece['oris']:
            for r0 in range(0, height - ph + 1):
                for c0 in range(0, width - pw + 1):
                    opts.append((_placement_mask(coords, r0, c0, width), int(r0), int(c0), coords))
        placements.append(opts)

    solutions: list[dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
    for anchor_idx, piece in enumerate(pieces):
        anchor_coords = piece['input_coords']
        _, ah, aw = _normalize_coords(anchor_coords)
        if ah > height or aw > width:
            continue

        anchor_mask = _placement_mask(anchor_coords, 0, 0, width)
        other_ids = [i for i in range(len(pieces)) if i != anchor_idx]
        filtered = [[pl for pl in placements[pi] if (pl[0] & anchor_mask) == 0] for pi in other_ids]
        remaining = set(range(len(other_ids)))
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
            for local_idx in sorted(remaining):
                opts_k = []
                for pl in filtered[local_idx]:
                    m = pl[0]
                    if (m >> cell) & 1 and (m & used) == 0:
                        opts_k.append(pl)
                if opts_k:
                    candidates.append((len(opts_k), local_idx, opts_k))
            if not candidates:
                return False

            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, local_idx, opts_k in candidates:
                remaining.remove(local_idx)
                pi = other_ids[local_idx]
                for pl in opts_k:
                    chosen[pi] = pl
                    if search(used | pl[0]):
                        return True
                chosen.pop(pi, None)
                remaining.add(local_idx)
            return False

        if search(anchor_mask):
            solutions.append(chosen.copy())

    assert len(solutions) == 1, 'assembled square must be uniquely anchored by observed orientation'
    return height, width, solutions[0]


def _implicit_legend_from_assembly(
    pieces: list[dict],
    chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]],
    height: int,
    width: int,
    k: int,
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    piece_cell: dict[int, tuple[int, int]] = {}
    cell_piece: dict[tuple[int, int], int] = {}
    for pi, pl in chosen.items():
        _, r0, c0, coords = pl
        cell = _piece_cell_from_coords(coords, r0, c0, height, width, k)
        assert cell not in cell_piece, 'each piece must map to a distinct assembled cell'
        piece_cell[pi] = cell
        cell_piece[cell] = pi

    assert len(cell_piece) == len(pieces) == k * k, 'assembled cells must form a complete kxk legend grid'

    legend_map = np.zeros((k, k), dtype=int)
    for rr in range(k):
        for cc in range(k):
            src_pi = cell_piece[(k - 1 - rr, k - 1 - cc)]
            legend_map[rr, cc] = int(pieces[src_pi]['color'])
    return legend_map, piece_cell


def _assemble_by_hidden_legend(pieces: list[dict], k: int) -> np.ndarray:
    height, width, chosen = _find_unique_assembly(pieces, k)
    legend_map, piece_cell = _implicit_legend_from_assembly(pieces, chosen, height, width, k)

    out = np.zeros((height, width), dtype=int)
    for pi, pl in chosen.items():
        _, r0, c0, coords = pl
        rr, cc = piece_cell[pi]
        col = int(legend_map[rr, cc])
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a rectangle, then derive a hidden 2x2 legend from the assembled quadrant colors.

    Input structure:
    - bg = most common color.
    - Four disconnected non-bg sprites, each monochrome and each using a different color.
    - There is no explicit multicolor legend sprite; one asymmetric piece stays in canonical orientation while the other three are scattered in different D4 orientations.

    Rule:
    - Tile all pieces into the unique tight square with no gaps, using shape matching before any color interpretation.
    - After assembly, quantize the four piece centroids into the 2x2 quadrant grid and treat the input piece colors as an implicit legend whose entries come from the diagonally opposite assembled quadrants.
    - Recolor each assembled piece with the hidden-legend color assigned to its own quadrant cell.

    Key insight:
    - The legend is not visible anywhere in the input: it only appears once the square has been reconstructed, because the four piece colors themselves become the 2x2 color key after a diagonal position swap.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, 'grid must contain at least one non-background component'

    k = int(round(len(comps) ** 0.5))
    assert k * k == len(comps), 'piece count must be a perfect square to determine k'
    assert k == 2, 'this combo variant is defined for 4 pieces'

    pieces: list[dict] = []
    seen_colors = set()
    for pts in comps:
        color = _component_color(g, pts, bg)
        assert color not in seen_colors, 'piece colors must be distinct in the combo variant'
        seen_colors.add(color)

        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        pieces.append(
            {
                'color': color,
                'area': len(filled_abs),
                'input_coords': rel_norm,
                'oris': _dihedral_orientations(rel_norm),
            }
        )

    return _assemble_by_hidden_legend(pieces, k)


# --------------- generator (o5x) ---------------

# === CFG ===
SIDE_CHOICES = [8, 10, 12]
K = 2
NUM_PIECES = K * K
NUM_COLORS = NUM_PIECES
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


def _mask_orientations(mask: np.ndarray) -> list[np.ndarray]:
    arr = mask.astype(int)
    opts: list[np.ndarray] = []
    seen = set()
    for rot_k in range(4):
        rot = np.rot90(arr, rot_k)
        for oriented in (rot, np.fliplr(rot)):
            pts = np.argwhere(oriented > 0)
            r0, c0, r1, c1 = bbox_coords(pts)
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop)
    return opts


def _random_noncanonical_orientation(mask: np.ndarray) -> np.ndarray | None:
    canonical = _crop_mask(mask)
    canonical_key = (canonical.shape, tuple(canonical.ravel()))
    variants = []
    for oriented in _mask_orientations(mask):
        key = (oriented.shape, tuple(oriented.ravel()))
        if key != canonical_key:
            variants.append(oriented)
    if not variants:
        return None
    return random.choice(variants)


def _scatter_pieces(masks: list[np.ndarray], body_colors: list[int], side: int) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    orientations: dict[int, np.ndarray] = {0: _crop_mask(masks[0])}
    for pi in range(1, NUM_PIECES):
        oriented = _random_noncanonical_orientation(masks[pi])
        if oriented is None:
            return None
        orientations[pi] = oriented

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = orientations[pi]
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = int(body_colors[pi])
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


def _assembled_output(masks: list[np.ndarray], body_colors: list[int], side: int) -> np.ndarray:
    qpiece = {}
    for pi, mask in enumerate(masks):
        pts = np.argwhere(mask)
        cell = (min(1, int(pts[:, 0].mean() * K / side)), min(1, int(pts[:, 1].mean() * K / side)))
        assert cell not in qpiece, 'assembled pieces must occupy distinct 2x2 cells'
        qpiece[cell] = pi

    legend_map = np.zeros((K, K), dtype=int)
    for rr in range(K):
        for cc in range(K):
            legend_map[rr, cc] = int(body_colors[qpiece[(K - 1 - rr, K - 1 - cc)]])

    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        pts = np.argwhere(mask)
        rr = min(1, int(pts[:, 0].mean() * K / side))
        cc = min(1, int(pts[:, 1].mean() * K / side))
        expected[mask] = int(legend_map[rr, cc])
    return expected


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

    body_colors = random.sample(range(1, 10), NUM_COLORS)
    expected = _assembled_output(masks, body_colors, side)
    grid = _scatter_pieces(masks, body_colors, side)
    if grid is None:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate f560132c combo input')
