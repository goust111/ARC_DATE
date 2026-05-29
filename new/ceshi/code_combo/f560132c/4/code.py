# Combo variant of f560132c (index 4)
# combo_spec: D2/0+D10/1
# Name: 空心框架精灵+全部带框架

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the fused variant
# - Concrete behavior trace:
#   Original inputs expose 4 scattered solid polyominoes, with the legend being the
#   only multi-color connected component. Here every sprite becomes a nested object:
#   a shared-color rectangular wrapper frame encloses one hollow body-outline; the
#   legend wrapper additionally encloses a floating 2x2 color core. Solver flow must
#   therefore become wrapper-detection -> inner-outline extraction -> fill the outline's
#   enclosed cells to recover the true tiling footprint -> exact-cover assembly ->
#   redraw only the recolored outlines in the compact output.
# - Complexity calibration:
#   The wrapper color is shared across all 4 sprites, so "find the odd multi-color
#   component" no longer works. At the same time, the design stays human-readable:
#   there is only one wrapper color, the legend still has exactly one 2x2 core, and
#   every piece keeps a single body color. That gives repeated evidence across 4 boxes
#   without exploding the number of free conventions.
# - Implicit assumptions broken from the original solver:
#   "all non-bg cells of one sprite form one component", "the legend is the unique
#   most-colorful component", and "the visible occupied cells are the exact cover
#   footprint" all fail. The new solver explicitly restores those invariants by parsing
#   common-color outer frames, isolating each box's body color, and filling the hollow
#   outline before tiling.
# - Placement legality:
#   Generator scatters the wrapper boxes as non-overlapping rectangles with border and
#   padding, so nested contents can never leak outside their own sprite box.
# - Visual observability:
#   Every training sample shows 4 repeated outer wrappers plus clearly hollow interior
#   shapes. The legend is the only wrapper containing a 2x2 core of 4 distinct colors,
#   so humans can infer both the stripping step and the recoloring cue from 2-3 samples.
# - Cleanliness:
#   Layers are separated spatially: wrapper on the outer perimeter, a one-cell moat,
#   then the hollow body, and for the legend the color core sits in the body's cavity.
#   That avoids ambiguous overlaps between the wrapper and the piece itself.


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


def _fill_holes_in_coords(
    coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]
) -> tuple[tuple[tuple[int, int], ...], int]:
    norm, h, w = _normalize_coords(coords)
    occ = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        occ[r, c] = True

    bgmask = ~occ
    vis = np.zeros_like(bgmask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for rr in range(h):
        for cc in range(w):
            if rr in (0, h - 1) or cc in (0, w - 1):
                if bgmask[rr, cc] and not vis[rr, cc]:
                    vis[rr, cc] = True
                    stack.append((rr, cc))
                    while stack:
                        r, c = stack.pop()
                        for r2, c2 in neighbors4(r, c):
                            if 0 <= r2 < h and 0 <= c2 < w and bgmask[r2, c2] and not vis[r2, c2]:
                                vis[r2, c2] = True
                                stack.append((r2, c2))

    holes = bgmask & ~vis
    hole_coords = [(rr, cc) for rr, cc in zip(*np.where(holes))]
    filled = tuple(sorted(set(norm).union(hole_coords)))
    return filled, int(holes.sum())


def _boundary_coords_from_solid(
    coords: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    occ = set(coords)
    out = []
    for r, c in coords:
        for r2, c2 in neighbors4(r, c):
            if (r2, c2) not in occ:
                out.append((r, c))
                break
    return tuple(sorted(out))


def _solid_mask_to_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for r2, c2 in neighbors4(r, c):
                if not (0 <= r2 < h and 0 <= c2 < w) or not mask[r2, c2]:
                    out[r, c] = True
                    break
    return out


def _extract_legend_map(sub: np.ndarray, k: int, bg: int, frame_color: int) -> np.ndarray | None:
    hh, ww = sub.shape
    found = []
    for rr in range(hh - k + 1):
        for cc in range(ww - k + 1):
            window = sub[rr : rr + k, cc : cc + k]
            vals = [int(v) for v in window.ravel()]
            cols = set(vals)
            if len(cols) != k * k:
                continue
            if bg in cols or frame_color in cols:
                continue
            found.append((rr, cc, window.copy()))
    if not found:
        return None
    found.sort(key=lambda x: (x[0], x[1]))
    return np.asarray(found[0][2], dtype=int)


def _is_rect_frame(pts: np.ndarray) -> bool:
    if len(pts) < 8:
        return False
    r0, c0, r1, c1 = bbox_coords(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if int(len(pts)) != expected:
        return False
    actual = set(map(tuple, pts))
    border = set()
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    return actual == border


def _find_wrappers(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    # New assumption: one color is reused only for the 4 outer wrappers. The generator
    # enforces this by reserving a dedicated frame color and using it nowhere else.
    candidates = []
    for col in sorted(set(map(int, g.ravel())) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        boxes = []
        for pts in comps:
            if not _is_rect_frame(pts):
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            if np.any(g[r0 + 1 : r1, c0 + 1 : c1] != bg):
                boxes.append((r0, c0, r1, c1))
        if len(boxes) == 4:
            candidates.append((int(col), sorted(boxes)))
    assert len(candidates) == 1, "must find exactly one shared wrapper color with 4 framed sprites"
    return candidates[0]


def _parse_sprite(
    g: np.ndarray,
    box: tuple[int, int, int, int],
    frame_color: int,
    bg: int,
    k: int,
) -> tuple[dict, np.ndarray | None]:
    r0, c0, r1, c1 = box
    inner = np.asarray(g[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    colors = sorted(set(map(int, inner.ravel())) - {bg, frame_color})
    assert colors, "wrapper interior must contain a hollow body"

    # New assumption: within each wrapper, the body color has far more support than any
    # legend-core color. The generator guarantees this because the core is exactly 2x2
    # with 4 singleton colors, while the body outline is the dominant repeated color.
    counts = {col: int(np.sum(inner == col)) for col in colors}
    body_color = max(colors, key=lambda col: (counts[col], -col))
    body_mask = inner == body_color
    body_pts = tuple(sorted(map(tuple, np.argwhere(body_mask))))
    assert body_pts, "body outline cannot be empty"
    outline_norm, _, _ = _normalize_coords(body_pts)
    filled_norm, hole_count = _fill_holes_in_coords(outline_norm)
    assert hole_count > 0, "every inner sprite must be visibly hollow"

    legend_map = _extract_legend_map(inner, k, bg, frame_color)
    piece = {
        "outline": outline_norm,
        "solid": filled_norm,
        "oris": _dihedral_orientations(filled_norm),
        "area": len(filled_norm),
        "body_color": int(body_color),
    }
    return piece, legend_map


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray:
    # New assumption: after filling the hollow outlines, the recovered solid footprints
    # still determine a unique rectangle, and the legend footprint is the top-left tile.
    # The generator constructs the full output square first, then wraps/scatters pieces.
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = _factor_pairs(total)
    dims = [(h, w) for (h, w) in dims if h >= max_h and w >= max_w]
    assert len(dims) == 1, "piece areas and bounds must imply a unique output rectangle"
    h_out, w_out = dims[0]
    nbits = h_out * w_out
    all_mask = (1 << nbits) - 1

    other_pis = [i for i in range(len(pieces)) if i != legend_idx]
    placements_other: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
    for pi in other_pis:
        opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
        for coords, ph, pw, _, _ in pieces[pi]["oris"]:
            for r0 in range(0, h_out - ph + 1):
                for c0 in range(0, w_out - pw + 1):
                    mask = 0
                    for dr, dc in coords:
                        idx = (int(r0) + int(dr)) * int(w_out) + (int(c0) + int(dc))
                        mask |= 1 << idx
                    opts.append((mask, int(r0), int(c0), coords))
        placements_other.append(opts)

    def solve_with_fixed_legend(
        legend_coords: tuple[tuple[int, int], ...]
    ) -> dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] | None:
        legend_mask = 0
        for dr, dc in legend_coords:
            idx = int(dr) * int(w_out) + int(dc)
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
        assert ph <= h_out and pw <= w_out, "legend orientation must fit the output rectangle"
        sol = solve_with_fixed_legend(coords)
        if sol is not None:
            chosen = sol
            break
    assert chosen is not None, "exact-cover assembly failed"

    piece_cell: dict[int, tuple[int, int]] = {}
    for pi, (_, r0, c0, coords) in chosen.items():
        rs = [r0 + dr for dr, _ in coords]
        cs = [c0 + dc for _, dc in coords]
        cr = float(sum(rs)) / float(len(rs))
        cc = float(sum(cs)) / float(len(cs))
        rr = int(cr * float(k) / float(h_out))
        cc2 = int(cc * float(k) / float(w_out))
        rr = max(0, min(k - 1, rr))
        cc2 = max(0, min(k - 1, cc2))
        piece_cell[pi] = (rr, cc2)

    out = np.zeros((h_out, w_out), dtype=int)
    for pi, (_, r0, c0, coords) in chosen.items():
        rr, cc2 = piece_cell[pi]
        col = int(legend_map[rr, cc2])
        for dr, dc in _boundary_coords_from_solid(coords):
            out[r0 + dr, c0 + dc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble wrapped hollow sprites into a rectangle, then recolor their outlines by a legend core.

    Input structure:
    - bg = most common color.
    - There are exactly 4 scattered sprite boxes, all surrounded by the same-color rectangular wrapper.
    - Inside each wrapper sits one monochrome hollow outline; inside exactly one wrapper there is also
      a 2x2 multi-color core that serves as the legend.

    Rule:
    - Ignore the shared outer wrappers completely.
    - For each sprite, use the inner hollow outline as the clue for a hidden filled polyomino footprint:
      fill its enclosed cavity, assemble all 4 recovered footprints into the unique rectangle, and anchor
      the legend footprint at the rectangle's top-left corner.
    - Recolor each assembled piece by the legend's 2x2 core according to the centroid quadrant, but draw
      only the hollow outline of each piece in the final tightly cropped output.

    Key insight:
    - The visible inner outline is not itself the tiling area; it is the boundary of the area. Once the
      common wrappers are stripped, solving requires one more abstraction step: infer the filled shape from
      the hollow boundary, tile those filled shapes, then project the answer back to hollow colored frames.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    k = 2

    frame_color, boxes = _find_wrappers(g, bg)
    pieces = []
    legend_maps = []
    for box in boxes:
        piece, legend_map = _parse_sprite(g, box, frame_color, bg, k)
        pieces.append(piece)
        legend_maps.append(legend_map)

    legend_candidates = [i for i, lm in enumerate(legend_maps) if lm is not None]
    assert len(legend_candidates) == 1, "must have exactly one legend wrapper with a 2x2 color core"
    legend_idx = legend_candidates[0]
    legend_map = np.asarray(legend_maps[legend_idx], dtype=int)
    return _assemble_exact_cover(pieces, legend_idx, legend_map, k)


# --------------- generator (output-first wrapped hollow-frame variant) ---------------

SIDE_CHOICES = [8, 10, 12]
K = 2
NUM_PIECES = K * K
BODY_COLORS = NUM_PIECES
PALETTE_COLORS = K * K
NUM_COLORS = BODY_COLORS + PALETTE_COLORS + 1

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

PANEL_PAD = 2
INPUT_SIZE_LO = 30
INPUT_SIZE_HI = 40
DENSITY_LO = 0.10
DENSITY_HI = 0.20
SCATTER_BORDER = 1
SCATTER_PADDING = 2
MIN_INTERIOR_CELLS = 4
GEN_RETRIES = 400


def _area_bounds(side: int) -> tuple[int, int]:
    total = side * side
    return (
        max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total)),
        min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total)),
    )


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
        seed = min(
            available,
            key=lambda p: abs(p[0] - ar) + abs(p[1] - ac) + random.random() * PARTITION_SEED_JITTER,
        )
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


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    boundary = _solid_mask_to_boundary(mask)
    return mask & ~boundary


def _palette_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    interior = _interior_mask(mask)
    spots = []
    h, w = mask.shape
    for r in range(h - K + 1):
        for c in range(w - K + 1):
            if interior[r : r + K, c : c + K].all():
                spots.append((r, c))
    return spots


def _build_sprite_patch(
    solid_mask: np.ndarray,
    body_color: int,
    frame_color: int,
    legend_map: np.ndarray | None,
) -> np.ndarray | None:
    solid_crop = _crop_mask(solid_mask)
    body = _solid_mask_to_boundary(solid_crop)

    h, w = body.shape
    patch = np.zeros((h + 2 * PANEL_PAD, w + 2 * PANEL_PAD), dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[PANEL_PAD : PANEL_PAD + h, PANEL_PAD : PANEL_PAD + w][body] = body_color

    if legend_map is not None:
        spots = _palette_spots(solid_crop)
        if not spots:
            return None
        rr, cc = random.choice(spots)
        patch[PANEL_PAD + rr : PANEL_PAD + rr + K, PANEL_PAD + cc : PANEL_PAD + cc + K] = legend_map
    return patch


def _scatter_pieces(
    masks: list[np.ndarray],
    body_colors: list[int],
    frame_color: int,
    legend_map: np.ndarray,
) -> np.ndarray | None:
    patches = []
    total_non_bg = 0
    max_dim = 0
    for pi in range(NUM_PIECES):
        oriented = _crop_mask(masks[pi]) if pi == 0 else _random_orientation(_crop_mask(masks[pi]))
        patch = _build_sprite_patch(oriented, body_colors[pi], frame_color, legend_map if pi == 0 else None)
        if patch is None:
            return None
        patches.append(patch)
        total_non_bg += int(np.count_nonzero(patch))
        max_dim = max(max_dim, patch.shape[0], patch.shape[1])

    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total_non_bg / DENSITY_HI)), 2 * max_dim + 2)
    nhi = min(INPUT_SIZE_HI, max(nlo, math.floor(math.sqrt(total_non_bg / DENSITY_LO))))
    if nlo > nhi:
        return None
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

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
    return grid


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
    if any(len(components_mask(m, connectivity=4)) != 1 for m in masks):
        return None
    if not _forces_unique_square(masks, side):
        return None

    boundaries = [_solid_mask_to_boundary(m) for m in masks]
    interiors = [m & ~b for m, b in zip(masks, boundaries)]
    if any(int(interior.sum()) < MIN_INTERIOR_CELLS for interior in interiors):
        return None
    if not _palette_spots(masks[0]):
        return None

    qcells = []
    for mask in masks:
        pts = np.argwhere(mask)
        qcells.append((min(1, int(pts[:, 0].mean() * K / side)), min(1, int(pts[:, 1].mean() * K / side))))
    if len(set(qcells)) != NUM_PIECES:
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    frame_color = colors[0]
    body_colors = colors[1 : 1 + BODY_COLORS]
    palette_colors = colors[1 + BODY_COLORS :]
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)

    expected = np.zeros((side, side), dtype=int)
    for pi, boundary in enumerate(boundaries):
        rr, cc = qcells[pi]
        expected[boundary] = legend_map[rr, cc]

    grid = _scatter_pieces(masks, body_colors, frame_color, legend_map)
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
    raise RuntimeError("failed to generate fused f560132c input")
