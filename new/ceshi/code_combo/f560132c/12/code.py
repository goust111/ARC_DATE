# Combo variant of f560132c (index 12)
# combo_spec: D6/0+D12/0
# Name: 原始网格输出+原位保留轮廓

# Design analysis for the combo variant
# a) Concrete behavior trace:
#    Original task: detect the 4 scattered components, infer the multicolor legend component,
#    exactly tile the 4 shapes into the unique square, recolor by the 2x2 legend map, and
#    return only that assembled square. In this combo variant the same reconstruction still
#    happens first, but the assembled square is pasted into a fixed left-top region of an
#    output grid that keeps the original input size. Each source piece also leaves a gray
#    one-cell outline at its original location, so one sample visibly shows "before" (gray
#    contour in place) and "after" (recolored assembled body in the corner) at once.
# b) Complexity calibration:
#    The assembled square stays fully colored and compact, while the source traces are only
#    outlines. That keeps the two layers visually distinct instead of making the output look
#    like two filled copies of the same shapes. The assembled placement is fixed across all
#    samples, which removes extra freedom and makes the new D6 change inferable from 2-3
#    examples.
# c) Hidden assumptions broken by the combo:
#    The original `main()` assumed the output shape equals the assembled rectangle and that
#    all non-background pixels in the output belong to the assembled tiling. That is no
#    longer true, so the new code must separate "piece extraction from input" from "output
#    rendering". The exact-cover search and legend decoding assumptions still hold because
#    the input pieces themselves are unchanged.
# d) Placement legality:
#    The generator reserves the left-top SxS block, so scattered input pieces never overlap
#    the future assembled output. This also guarantees that gray source outlines never need
#    to overwrite the assembled result.
# e) Visual observability:
#    A human can see that the colored result always appears in the same corner at full input
#    scale, and the original positions are still marked by gray shells. The relation between
#    the two is shape-based, not color-id-based.
# f) Visual cleanliness:
#    Source traces use a single unused gray color and only the exterior contour, avoiding
#    dense double-painting. The reserved corner plus one-cell moat keeps the "before" and
#    "after" regions separated.
# g) Color inference:
#    Recoloring still comes only from the legend's 2x2 embedded map, as in the original
#    task. Gray is reserved exclusively for traces, so it never competes with the palette.
# New solver assumptions introduced here:
# - The assembled square is always pasted at the fixed anchor (0,0).
# - Gray outline color 5 is absent from the input; the generator enforces that.
# - Input pieces are scattered outside the reserved assembled-output zone; the generator
#   explicitly marks that zone as occupied before placement.

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


ASSEMBLY_ANCHOR = (0, 0)
OUTLINE_COLOR = 5
OUTPUT_MOAT = 1


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


def _outline_from_cells(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    cell_set = set((int(r), int(c)) for r, c in cells)
    outline = []
    for r, c in sorted(cell_set):
        if any((int(r2), int(c2)) not in cell_set for r2, c2 in neighbors4(int(r), int(c))):
            outline.append((int(r), int(c)))
    return outline


def _choose_outline_color(g: np.ndarray, bg: int) -> int:
    used = set(map(int, g.ravel()))
    if OUTLINE_COLOR not in used and OUTLINE_COLOR != bg:
        return int(OUTLINE_COLOR)
    for col in range(1, 10):
        if col != bg and col not in used:
            return int(col)
    raise AssertionError("variant needs an unused non-background color for outlines")


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
    bg: int,
) -> np.ndarray:
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
                for rem_idx in sorted(remaining):
                    opts_k = []
                    for pl in filtered[rem_idx]:
                        m = pl[0]
                        if (m >> cell) & 1 and (m & used) == 0:
                            opts_k.append(pl)
                    if opts_k:
                        candidates.append((len(opts_k), rem_idx, opts_k))
                if not candidates:
                    return False

                candidates.sort(key=lambda x: (x[0], x[1]))
                for _, rem_idx, opts_k in candidates:
                    remaining.remove(rem_idx)
                    pi = other_pis[rem_idx]
                    for pl in opts_k:
                        chosen[pi] = pl
                        if search(used | pl[0]):
                            return True
                    chosen.pop(pi, None)
                    remaining.add(rem_idx)
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

        out = np.full((H, W), int(bg), dtype=int)
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            rr, cc2 = piece_cell[pi]
            col = int(legend_map[rr, cc2])
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        return out

    raise AssertionError("rectangle assembly loop must return")


def _extract_piece_data(g: np.ndarray, bg: int) -> tuple[list[dict], int, np.ndarray, int]:
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
        oris = _dihedral_orientations(rel_norm)
        pieces.append(
            {
                "oris": oris,
                "area": len(filled_abs),
                "outline_abs": _outline_from_cells(filled_abs),
            }
        )
    return pieces, int(legend_idx), legend_map, int(k)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered pieces as before, but keep the full input-sized canvas and show source traces.

    Input structure:
    - bg is the dominant color on a larger square canvas.
    - Four separated non-bg sprites appear on that canvas; one sprite is a multicolor legend piece with an interior 2x2 palette.
    - The left-top corner area is empty in the input and serves as the fixed destination zone for the assembled result.

    Rule:
    - Recover the same unique square tiling as in the base task, allowing rotations/reflections and recoloring pieces from the legend palette by assembled position.
    - Paste that completed square into the fixed left-top destination while keeping the output grid the same size as the input.
    - At every original piece location, replace the filled sprite with a one-cell gray outline of its shape.

    Key insight: The puzzle-solving part is unchanged; the new challenge is to render both states at once by combining the reconstructed square with gray "where it used to be" contours on the original canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    outline_color = _choose_outline_color(g, bg)

    pieces, legend_idx, legend_map, k = _extract_piece_data(g, bg)
    assembled = _assemble_exact_cover(pieces, legend_idx, legend_map, k, bg)

    out = np.full_like(g, bg)
    ar, ac = ASSEMBLY_ANCHOR
    ah, aw = assembled.shape
    assert ar + ah <= g.shape[0] and ac + aw <= g.shape[1], "assembled rectangle must fit inside the original canvas"
    paste(out, assembled, ar, ac, transparent=bg)

    for piece in pieces:
        for r, c in piece["outline_abs"]:
            if out[int(r), int(c)] == bg:
                out[int(r), int(c)] = outline_color
    return out


# --------------- generator ---------------

SIDE_CHOICES = [8, 10, 12]
K = 2
NUM_PIECES = K * K
BODY_AND_PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
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


def _palette_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(1, h - K):
        for c in range(1, w - K):
            if mask[r : r + K, c : c + K].all() and mask[r - 1 : r + K + 1, c - 1 : c + K + 1].all():
                spots.append((r, c))
    return spots


def _build_assembled_square(masks: list[np.ndarray], legend_map: np.ndarray, side: int, bg: int) -> np.ndarray:
    qcells = []
    for mask in masks:
        pts = np.argwhere(mask)
        qcells.append((min(1, int(pts[:, 0].mean() * K / side)), min(1, int(pts[:, 1].mean() * K / side))))
    assert len(set(qcells)) == NUM_PIECES, "each quadrant must receive exactly one piece"

    assembled = np.full((side, side), bg, dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        assembled[mask] = int(legend_map[rr, cc])
    return assembled


def _scatter_pieces(
    masks: list[np.ndarray],
    body_colors: list[int],
    legend_map: np.ndarray,
    side: int,
    bg: int,
) -> tuple[np.ndarray, list[dict]] | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None
    grid_size = random.randint(nlo, nhi)
    grid = np.full((grid_size, grid_size), bg, dtype=int)
    occupied = np.zeros_like(grid)

    reserved_h = min(grid_size, side + OUTPUT_MOAT)
    reserved_w = min(grid_size, side + OUTPUT_MOAT)
    occupied[:reserved_h, :reserved_w] = 1

    placements = []
    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _crop_mask(masks[pi]) if pi == 0 else _random_orientation(_crop_mask(masks[pi]))
        patch = np.full(oriented.shape, bg, dtype=int)
        patch[oriented] = int(body_colors[pi])
        if pi == 0:
            spots = _palette_spots(oriented)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r : r + K, c : c + K] = legend_map
        try:
            r0, c0 = random_free_location(
                occupied,
                np.ones(oriented.shape, dtype=int),
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        occupied[r0 : r0 + oriented.shape[0], c0 : c0 + oriented.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=bg)
        placements.append({"pi": int(pi), "r0": int(r0), "c0": int(c0), "mask": oriented.copy()})
    return grid, placements


def _render_variant_output(
    grid_shape: tuple[int, int],
    assembled: np.ndarray,
    placements: list[dict],
    bg: int,
    outline_color: int,
) -> np.ndarray:
    out = np.full(grid_shape, bg, dtype=int)
    paste(out, assembled, ASSEMBLY_ANCHOR[0], ASSEMBLY_ANCHOR[1], transparent=bg)
    for pl in placements:
        mask = pl["mask"]
        local_cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        for rr, cc in _outline_from_cells(local_cells):
            r = int(pl["r0"]) + int(rr)
            c = int(pl["c0"]) + int(cc)
            if out[r, c] == bg:
                out[r, c] = outline_color
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = 0
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

    colors = random.sample(BODY_AND_PALETTE_COLORS, len(BODY_AND_PALETTE_COLORS))
    body_colors = colors[:NUM_PIECES]
    palette_colors = colors[NUM_PIECES:]
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)
    assembled = _build_assembled_square(masks, legend_map, side, bg)

    scattered = _scatter_pieces(masks, body_colors, legend_map, side, bg)
    if scattered is None:
        return None
    grid, placements = scattered

    expected = _render_variant_output(grid.shape, assembled, placements, bg, OUTLINE_COLOR)
    try:
        pred = main(grid)
    except Exception:
        return None
    if pred.shape != expected.shape or not np.array_equal(pred, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c combo input")
