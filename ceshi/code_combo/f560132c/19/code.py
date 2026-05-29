# Combo variant of f560132c (index 19)
# combo_spec: D12/1+D15/0
# Name: 原位着目标色+复合体旋转90度

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for combo D12/1 + D15/0
# 1. Concrete behavior trace:
#    - Original task: assemble the four scattered polyominoes into one tight square A, recolor each piece from the
#      legend's 2x2 palette by assembled centroid quadrant, and output only A.
#    - D15/0 contribution: after building A, rotate the whole square by 90 degrees. The direction is read from the
#      legend silhouette itself: longer top arm => clockwise, longer left arm => counterclockwise.
#    - D12/1 contribution in this task family: the source sprites do not "disappear" conceptually. Their original
#      footprints remain visible in the output, but each footprint is repainted to the same target color that the
#      corresponding piece receives in the assembled square. The legend's internal palette cells are overwritten too,
#      so each source footprint becomes a solid target-color copy.
#    - Fused output layout: showing both structures on one canvas without overlap is cleaner than mixing them into one
#      field. Therefore the output is a two-panel image: left = rotated assembled square, right = tightly cropped
#      recolored source scene, separated by one background column.
# 2. Complexity calibration:
#    - Only two linked signals are added: a binary rotation direction and a duplicated color-transfer view. The same
#      four target colors reappear in both panels, which makes the fusion visible in 2-3 examples.
#    - The legend arm-length gap is forced to be >= 2 cells, avoiding ambiguous near-symmetric cues.
#    - The panel arrangement is fixed, so the task remains about reasoning over shape/color correspondences rather than
#      inferring a new free-form placement rule.
# 3. Implicit assumptions audited against the original main():
#    - Unique multicolor legend: still required; generator preserves one palette-bearing component.
#    - Perfect-square piece count: unchanged; still exactly 4 pieces.
#    - Exact-cover uniqueness from area/bbox constraints: unchanged and still enforced.
#    - New assumption from D15: the legend must be shown in canonical orientation because the solver reads "top arm"
#      versus "left arm" directly from the input. The generator keeps the legend unrotated while other pieces may use
#      any D4 orientation.
#    - New assumption from D12: after exact-cover search the solver must know each piece's target color, not just the
#      final assembled bitmap. `_assemble_exact_cover(...)` is therefore extended to return a per-piece color map.
#    - Original code assumed the assembled square was the entire output. In the combo that final step is replaced by a
#      deterministic panel compositor.
# 4. Placement legality:
#    - Input scattering still uses non-overlapping bounding boxes, border margin, and an interior 2x2 palette inside
#      the legend body.
#    - Output legality is trivial because the two panels are generated on a fresh canvas with a fixed one-column gap.
# 5. Visual observability / cleanliness:
#    - Side-by-side panels avoid clutter that would result from pasting the assembled square onto the same canvas as
#      the scattered traces.
#    - The right panel is tightly cropped to the recolored source scene, so the preserved "original positions" remain
#      visually obvious without wasting large blank margins.
#    - Because body colors and palette colors are disjoint, every source piece visibly changes color in the trace panel.
# 6. Degeneracy control:
#    - Generator rejects cases where the 90-degree rotation leaves the assembled square unchanged.
#    - Post-generation metrics compare the fused output against a rotation-only baseline and a trace-only baseline; if
#      either difference vanished systematically, the combo would have collapsed back toward a single-dimension task.


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
    norm, h, _w = _normalize_coords(coords)
    rot = [(c, h - 1 - r) for r, c in norm]
    return _normalize_coords(rot)


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, _h, w = _normalize_coords(coords)
    flp = [(r, w - 1 - c) for r, c in norm]
    return _normalize_coords(flp)


def _dihedral_orientations(
    coords: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]]:
    """
    Unique orientations under rotations + mirror (D4).
    Returned as (coords, h, w, rot_k, flipped), where the transform is:
      - if flipped: horizontal flip
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
    Inside the legend component's bounding box, find a k×k color map block.
    Pick the k×k window with the maximum number of distinct colors; ties go to top-left.
    """
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
    assert best_rc is not None, "legend component must contain a valid k×k color map window"
    rr, cc = best_rc
    return np.asarray(sub[rr : rr + k, cc : cc + k], dtype=int).copy()


def _fill_holes_in_component(g: np.ndarray, pts: np.ndarray, bg: int) -> tuple[tuple[tuple[int, int], ...], int]:
    """
    Treat background holes fully enclosed inside the component bbox as part of the piece.
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


def _legend_rotation_k_from_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...] | np.ndarray) -> int:
    """
    Infer the final whole-square rotation from the legend silhouette.
    Longer top arm => clockwise (`k=-1` in `np.rot90`), longer left arm => counterclockwise (`k=1`).
    """
    if isinstance(coords, np.ndarray):
        coords_in = [tuple(map(int, rc)) for rc in coords]
    else:
        coords_in = list(coords)
    norm, _, _ = _normalize_coords(coords_in)
    top_len = sum(1 for r, _ in norm if r == 0)
    left_len = sum(1 for _, c in norm if c == 0)
    assert top_len != left_len, "legend arm lengths must disambiguate the 90-degree rotation direction"
    return -1 if top_len > left_len else 1


def _crop_non_bg(arr: np.ndarray, bg: int = 0) -> np.ndarray:
    pts = np.argwhere(arr != bg)
    assert len(pts) > 0, "panel to crop must contain non-background cells"
    r0, c0, r1, c1 = bbox_coords(pts)
    return np.asarray(arr[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _compose_panels(left: np.ndarray, right: np.ndarray, bg: int = 0, gap: int = 1) -> np.ndarray:
    h = max(int(left.shape[0]), int(right.shape[0]))
    w = int(left.shape[1]) + int(gap) + int(right.shape[1])
    out = np.full((h, w), int(bg), dtype=int)
    out[: left.shape[0], : left.shape[1]] = left
    c0 = int(left.shape[1]) + int(gap)
    out[: right.shape[0], c0 : c0 + right.shape[1]] = right
    return out


def _build_trace_scene(shape: tuple[int, int], piece_footprints: list[tuple[tuple[int, int], ...]], piece_colors: dict[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for pi, pts in enumerate(piece_footprints):
        col = int(piece_colors[pi])
        for r, c in pts:
            out[r, c] = col
    return out


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Tile all pieces into a full rectangle, allowing rotations + reflections.
    Recolor each piece by quantizing its centroid into a k×k grid.
    Returns (assembled_grid, piece_color_by_index).
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

        piece_color: dict[int, int] = {}
        out = np.zeros((H, W), dtype=int)
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
            col = int(legend_map[rr, cc2])
            piece_color[pi] = col
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        return out, piece_color

    raise AssertionError("exact-cover assembly failed")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the scattered pieces into a square, rotate that square by 90°, and also keep recolored source footprints.

    Input structure:
    - The background is the most frequent color.
    - Four disconnected sprites are scattered on the canvas; one sprite is a multicolor legend piece carrying a 2x2 palette.
    - The legend keeps an asymmetric L-like silhouette whose top-vs-left arm lengths indicate the final 90° rotation direction.

    Rule:
    - Detect the unique multicolor legend, read its 2x2 palette, and assemble all four piece shapes into the unique tight square using D4 matching.
    - Recolor each assembled piece from the palette cell chosen by its centroid position, then rotate the completed square by 90° according to the legend-arm cue.
    - Also repaint each source sprite footprint in its original scattered location to that same target color, crop that recolored source scene tightly, and place it to the right of the rotated square with a one-cell background gap.

    Key insight:
    - The legend now controls two linked outputs at once: its internal 2x2 patch still defines the color transfer, while its outer arm asymmetry determines the global rotation applied only to the assembled panel; the same target colors then reappear a second time on the preserved source footprints.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

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
    rot_k = _legend_rotation_k_from_coords(comps[legend_idx])

    pieces: list[dict] = []
    piece_footprints: list[tuple[tuple[int, int], ...]] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        piece_footprints.append(filled_abs)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        pieces.append({"oris": oris, "area": len(filled_abs)})

    assembled, piece_colors = _assemble_exact_cover(pieces, legend_idx, legend_map, k)
    rotated = np.rot90(assembled, rot_k)
    trace_scene = _build_trace_scene(g.shape, piece_footprints, piece_colors)
    trace_panel = _crop_non_bg(trace_scene, bg)
    return _compose_panels(rotated, trace_panel, bg=bg, gap=1)


# --------------- generator ---------------

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
LEGEND_ARM_DELTA_MIN = 2
ROTATION_CHOICES = [-1, 1]
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


def _make_legend_mask(side: int, min_area: int, max_area: int, rot_k: int) -> np.ndarray | None:
    threshold = LEGEND_THRESHOLD[side]
    opts = []
    for h0 in range(threshold + 1, side + 1):
        for w0 in range(threshold + 1, side + 1):
            if rot_k == -1 and w0 - h0 < LEGEND_ARM_DELTA_MIN:
                continue
            if rot_k == 1 and h0 - w0 < LEGEND_ARM_DELTA_MIN:
                continue
            for arm_w in LEGEND_ARM_RANGE:
                for arm_h in LEGEND_ARM_RANGE:
                    mask = np.zeros((side, side), dtype=bool)
                    mask[:LEGEND_CORE_SIZE, :LEGEND_CORE_SIZE] = True
                    mask[:arm_h, :w0] = True
                    mask[:h0, :arm_w] = True
                    area = int(mask.sum())
                    if min_area <= area <= max_area:
                        opts.append(mask)
    if not opts:
        return None
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
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool).copy()


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(int)
    opts = []
    seen = set()
    for k2 in range(4):
        rot = np.rot90(arr, k2)
        for oriented in (rot, np.fliplr(rot)):
            pts = np.argwhere(oriented > 0)
            r0, c0, r1, c1 = bbox_coords(pts)
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(map(int, crop.ravel())))
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
) -> tuple[np.ndarray, dict[int, tuple[int, int, np.ndarray]]] | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)
    placements: dict[int, tuple[int, int, np.ndarray]] = {}

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
        placements[pi] = (int(r0), int(c0), oriented.copy())
    return grid, placements


def _build_trace_scene_from_placements(
    grid_shape: tuple[int, int],
    placements: dict[int, tuple[int, int, np.ndarray]],
    piece_colors: dict[int, int],
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for pi, (r0, c0, mask) in placements.items():
        col = int(piece_colors[pi])
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = col
        paste(out, patch, r0, c0, transparent=0)
    return out


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SIDE_CHOICES)
    min_area, max_area = _area_bounds(side)
    rot_k = random.choice(ROTATION_CHOICES)
    legend = _make_legend_mask(side, min_area, max_area, rot_k)
    if legend is None:
        return None
    if _legend_rotation_k_from_coords(np.argwhere(_crop_mask(legend))) != rot_k:
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
    palette_colors = colors[NUM_COLORS - (K * K) :]
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)

    assembled = np.zeros((side, side), dtype=int)
    piece_colors: dict[int, int] = {}
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        col = int(legend_map[rr, cc])
        piece_colors[pi] = col
        assembled[mask] = col

    rotated = np.rot90(assembled, rot_k)
    if np.array_equal(rotated, assembled):
        return None

    scattered = _scatter_pieces(masks, body_colors, legend_map, side)
    if scattered is None:
        return None
    grid, placements = scattered

    trace_scene = _build_trace_scene_from_placements(grid.shape, placements, piece_colors)
    trace_panel = _crop_non_bg(trace_scene)
    expected = _compose_panels(rotated, trace_panel, gap=1)

    if not np.array_equal(main(grid), expected):
        return None

    if trace_panel.shape == rotated.shape and np.array_equal(trace_panel, rotated):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c combo input")
