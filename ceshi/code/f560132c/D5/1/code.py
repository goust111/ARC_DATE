# Variant of f560132c
# Dimension: D5 — 符号指令：图例的编码方式与结构 (symbol_instructions)
# Change: 图例精灵包含2x2彩色像素嵌入在单色框架中，每个像素的位置（左上、右上、左下、右下）映射到复合体中对应象限的重着色颜色... → 图例为嵌套框架编码

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for this D5 variant
# a) Concrete behavior:
#    The input still contains four scattered polyomino pieces that exactly tile one square output.
#    One piece is the legend because it is the only multicolor component. The old legend had a 2x2
#    color block, so the solver could read colors by position. The new legend instead contains a
#    7x7 nested-square motif with four visible layers: outer ring, second ring, third ring, center.
#    After assembly, the four pieces are no longer recolored by quadrant position. They are sorted
#    by filled area from largest to smallest, then mapped to legend colors from outermost layer to
#    innermost layer. Example: if assembled piece areas are [54, 33, 31, 26], the 54-cell piece
#    gets the outer ring color, 33 gets the next ring color, 31 gets the third ring color, and 26
#    gets the core color. This is the exact point where the original and variant behaviors diverge.
# b) Complexity calibration:
#    The variant must be visually different from the original, so the generator always paints a
#    nested 7x7 motif instead of any 2x2 palette. The four piece areas are forced distinct; without
#    that, area-to-layer matching would be ambiguous and the new rule would collapse.
# c) Hidden assumptions in the original main():
#    `_extract_legend_map()` assumed a kxk spatial palette existed and that color meaning depended
#    on its row/col location. That assumption is invalid here, so legend parsing is replaced by a
#    search for a 7x7 window whose concentric rings are monochrome. `_assemble_exact_cover()` also
#    assumed piece colors came from centroid quantization into a 2x2 map; that is replaced with an
#    area ranking. The exact-cover search itself remains valid because the piece geometry is still
#    the same kind of tiling puzzle.
# d) Placement legality:
#    The generator still builds the output first, partitions it into four connected pieces, then
#    scatters them with `random_free_location`, so pieces never overlap or leave the input bounds.
#    The legend shape is enlarged just enough to guarantee one solid 7x7 region where the nested
#    motif can be painted.
# e) Visual observability:
#    The nested-frame encoding is directly visible as concentric colored squares, which is a strong
#    human cue. The recoloring rule is also supported by repeated examples because the three
#    non-legend pieces change color according to relative size rather than fixed position.
# f) Visual cleanliness:
#    Only the legend carries multiple colors; the other pieces remain monochrome in the input.
#    The nested block is contiguous and compact, so the legend stays readable instead of looking
#    like scattered markers.
# g) Color inferability:
#    The number of functional colors stays at four output colors, matching the four pieces. The
#    mapping signal is geometric and visible: outer/inner legend layers correspond to large/small
#    assembled pieces. No color-id ordering is used anywhere.
# h) Variant verification metric:
#    After implementation we can inspect generated samples by checking that the legend exposes four
#    nested layers and that the area-ranked recoloring usually differs from the original quadrant
#    recoloring. If those outputs frequently coincide, the generator would be degenerating.
# i) New solver assumptions and generator guarantees:
#    The solver now assumes the legend has exactly one valid 7x7 nested-frame window and that the
#    four piece areas are distinct. The generator enforces both by construction.


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


def _ring_mask(size: int, depth: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    lo = int(depth)
    hi = int(size - depth - 1)
    mask[lo, lo : hi + 1] = True
    mask[hi, lo : hi + 1] = True
    mask[lo : hi + 1, lo] = True
    mask[lo : hi + 1, hi] = True
    return mask


def _nested_window_colors(window: np.ndarray) -> list[int] | None:
    size = int(window.shape[0])
    if window.shape != (size, size) or size != NESTED_SIZE:
        return None
    layers = []
    for depth in range(NESTED_LAYERS):
        ring = window[_ring_mask(size, depth)]
        cols = set(map(int, ring.ravel()))
        if len(cols) != 1:
            return None
        layers.append(int(next(iter(cols))))
    if len(set(layers)) != NESTED_LAYERS:
        return None
    return layers


def _extract_nested_legend_colors(g: np.ndarray, pts: np.ndarray) -> list[int]:
    """
    Inside the legend component's bounding box, find the 7x7 nested square whose concentric
    layers are monochrome. Return colors from outer layer to inner core.
    """
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert hh >= NESTED_SIZE and ww >= NESTED_SIZE, 'legend component must fit the nested-frame motif'

    best = None
    best_colors = None
    for rr in range(hh - NESTED_SIZE + 1):
        for cc in range(ww - NESTED_SIZE + 1):
            window = sub[rr : rr + NESTED_SIZE, cc : cc + NESTED_SIZE]
            colors = _nested_window_colors(window)
            if colors is None:
                continue
            key = (len(set(colors)), -rr, -cc)
            if best is None or key > best:
                best = key
                best_colors = colors
    assert best_colors is not None, 'legend component must contain a valid nested-frame code'
    return list(best_colors)


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
    legend_layers: list[int],
) -> np.ndarray:
    """
    Tile all pieces into a full rectangle (no holes), allowing rotations + reflections.
    Recolor pieces by descending area, using the legend's nested-frame colors from outer layer
    to inner core.
    """
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = _factor_pairs(total)
    dims = [(H, W) for (H, W) in dims if H >= max_h and W >= max_w]
    assert len(dims) == 1, 'piece areas and bounds must imply a unique output rectangle'
    assert len(legend_layers) == len(pieces), 'nested-frame color count must match the number of pieces'

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

        chosen = None
        for coords, ph, pw, _, _ in pieces[legend_idx]["oris"]:
            assert ph <= H and pw <= W, 'legend orientation must fit the unique output rectangle'
            sol = solve_with_fixed_legend(coords)
            if sol is None:
                continue
            chosen = sol
            break
        assert chosen is not None, 'exact-cover assembly failed: no valid legend orientation'

        areas = [int(p["area"]) for p in pieces]
        assert len(set(areas)) == len(areas), 'piece areas must be distinct for area-ranked recoloring'
        ranked = sorted(range(len(pieces)), key=lambda pi: (-areas[pi], pi))
        piece_color = {pi: int(legend_layers[rank]) for rank, pi in enumerate(ranked)}

        out = np.zeros((H, W), dtype=int)
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            col = piece_color[pi]
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a square, recoloring them by a legend sprite's nested frames.

    Input structure:
    - bg = most common color.
    - Four scattered non-bg sprites are the puzzle pieces.
    - One sprite is the legend: it has the same piece geometry style as the others, but it also
      contains a compact 7x7 nested-square code with four concentric color layers.

    Rule:
    - Tile all pieces into the unique tight square, allowing rotations and reflections.
    - Read the legend's nested-square colors from outermost ring to innermost core.
    - Sort the assembled pieces by filled area from largest to smallest, and recolor them with
      those legend colors in that same order.

    Key insight: The legend no longer encodes colors by 2x2 position. Its concentric layer order
    encodes an area ranking, so the solver must infer "large piece -> outer color ... small piece
    -> center color" after reconstructing the square.
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
    legend_layers = _extract_nested_legend_colors(g, comps[legend_idx])

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        pieces.append({"oris": oris, "area": len(filled_abs)})

    return _assemble_exact_cover(pieces, legend_idx, legend_layers)


# --------------- generator ---------------

# === CFG ===
SIDE_CHOICES = [10, 12]
K = 2
NUM_PIECES = K * K
NUM_COLORS = 8
AREA_FRAC_LO = 0.08
AREA_FRAC_HI = 0.62
AREA_ABS_LO = 11
AREA_ABS_HI = 80
LEGEND_THRESHOLD = {10: 6, 12: 6}
LEGEND_CORE_SIZE = 7
LEGEND_ARM_RANGE = range(1, 3)
NESTED_SIZE = 7
NESTED_LAYERS = 4
PARTITION_GROWTH_EXP = 1.45
PARTITION_FRONTIER_EXP = 0.2
PARTITION_SEED_JITTER = 0.2
INPUT_SIZE_LO = 22
INPUT_SIZE_HI = 28
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
                    remaining = side * side - area
                    if min_area <= area <= max_area and remaining >= 3 * min_area:
                        opts.append(mask)
    assert opts, 'legend mask options must exist'
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


def _nested_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(0, h - NESTED_SIZE + 1):
        for c in range(0, w - NESTED_SIZE + 1):
            if mask[r : r + NESTED_SIZE, c : c + NESTED_SIZE].all():
                spots.append((r, c))
    return spots


def _paint_nested_frames(patch: np.ndarray, r0: int, c0: int, colors: list[int]) -> None:
    for depth, col in enumerate(colors):
        lo = depth
        hi = NESTED_SIZE - depth
        patch[r0 + lo : r0 + hi, c0 + lo : c0 + hi] = int(col)


def _scatter_pieces(masks: list[np.ndarray], body_colors: list[int], legend_layers: list[int], side: int) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        if pi == 0:
            spots = _nested_spots(oriented)
            if not spots:
                return None
            rr, cc = random.choice(spots)
            _paint_nested_frames(patch, rr, cc, legend_layers)
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


def _quadrant_cells(masks: list[np.ndarray], side: int) -> list[tuple[int, int]]:
    qcells = []
    for mask in masks:
        pts = np.argwhere(mask)
        qcells.append((min(1, int(pts[:, 0].mean() * K / side)), min(1, int(pts[:, 1].mean() * K / side))))
    return qcells


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
    if len(set(areas)) != NUM_PIECES:
        return None
    if any(len(components_mask(m, connectivity=4)) != 1 or _has_hole(m) for m in masks):
        return None
    if not _forces_unique_square(masks, side):
        return None

    qcells = _quadrant_cells(masks, side)
    if len(set(qcells)) != NUM_PIECES:
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_PIECES]
    legend_layers = colors[NUM_PIECES:]
    random.shuffle(legend_layers)

    ranked = sorted(range(NUM_PIECES), key=lambda pi: (-areas[pi], pi))
    expected = np.zeros((side, side), dtype=int)
    for rank, pi in enumerate(ranked):
        expected[masks[pi]] = legend_layers[rank]

    grid = _scatter_pieces(masks, body_colors, legend_layers, side)
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
    raise RuntimeError('failed to generate f560132c D5 nested-frame input')
