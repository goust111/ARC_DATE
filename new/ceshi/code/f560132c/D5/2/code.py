# Variant of f560132c
# Dimension: D5 — 符号指令：图例的编码方式与结构 (symbol_instructions)
# Change: 图例精灵包含2x2彩色像素嵌入在单色框架中，每个像素的位置（左上、右上、左下、右下）映射到复合体中对应象限的重着色颜色... → 图例分散在网格边缘

"""
Design analysis for the D5 variant.

1. Concrete behavior and trace
   - Original behavior:
     one jigsaw piece is multicolored because it contains an internal 2x2 palette.
     `main()` locates that unique multicolor component, extracts the 2x2 block, then
     assembles all 4 pieces into a square and recolors by quadrant.
   - Variant behavior:
     all 4 jigsaw pieces are now plain monochrome body sprites. The legend is rebuilt
     from 4 isolated one-cell markers placed on the outer border of the input grid:
       top edge    -> assembled top-left quadrant color
       right edge  -> assembled top-right quadrant color
       left edge   -> assembled bottom-left quadrant color
       bottom edge -> assembled bottom-right quadrant color
     The markers are also placed near the matching corners/halves of the canvas
     (top marker on the left half, right marker on the upper half, left marker on
     the lower half, bottom marker on the right half) so the cue is visually legible.
   - Small trace:
     old solver step: "find component with most distinct colors" -> succeeds because the
     legend piece contains 4 embedded colors.
     new solver step: that assumption is false because every piece is monochrome; instead
     it first removes the 4 singleton border markers, builds `legend_map=[[top,right],
     [left,bottom]]`, and only then runs the rectangle assembly.

2. Complexity calibration
   - The exact-cover tiling problem stays unchanged: 4 simply connected polyominoes still
     assemble into one square under D4 transforms.
   - The variant is visually noticeable because there is no multicolor piece anymore; the
     4 palette colors now appear as detached edge pixels.
   - To avoid over-simplification and global-orientation ambiguity after removing the old
     legend piece, generation keeps a geometrically distinctive L-shaped anchor piece and
     additionally enforces that it is the unique largest piece by area. The solver uses
     that as the assembly anchor.

3. Original assumptions that break and the fixes
   - `len(comps)` used to equal the number of pieces. Now there are 4 extra marker
     components, so the solver must detect and remove them before inferring `k`.
   - "legend component = component with most distinct colors" is no longer valid because
     no piece is multicolored. Replaced with border-marker extraction.
   - The old anchor was the legend piece itself. With detached markers there is no colored
     anchor, so the solver now assumes the unique largest piece is the top-left anchor.
     The generator explicitly guarantees this.

4. Placement legality
   - Pieces are scattered with a border margin of 2, not 1. This keeps row/col 0 and the
     adjacent inner ring empty, so edge markers stay isolated and cannot accidentally merge
     with a piece under 4-connectivity.
   - Markers are single pixels on non-corner border cells only, one per side.

5. Visual observability
   - Across examples, the user sees four detached colored dots around the canvas instead
     of one embedded 2x2 palette. That difference is immediate.
   - The side-to-quadrant mapping is low-arity and repeated in every example, so it can be
     inferred from 2-3 training pairs.

6. Color inferability
   - The number of palette colors remains 4, exactly matching the 4 assembled quadrants.
   - Colors are not manipulated by id ordering; only spatial roles matter.

7. New solver assumptions and generator guarantees
   - Assumption: exactly 4 singleton border markers exist, one on each side.
     Guarantee: generator stamps one such pixel on top/right/left/bottom only.
   - Assumption: after removing markers, exactly 4 components remain and the unique largest
     one is the anchor.
     Guarantee: generator rejects attempts unless the anchor piece has strictly largest area.
   - Assumption: border markers are not part of any piece.
     Guarantee: pieces are never placed within distance 1 of the outer border.
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
    """
    Detect 4 singleton marker pixels on the canvas border and decode them into a 2x2 legend map.

    Mapping:
      top edge    -> legend_map[0, 0]
      right edge  -> legend_map[0, 1]
      left edge   -> legend_map[1, 0]
      bottom edge -> legend_map[1, 1]
    """
    h, w = g.shape
    by_side: dict[str, tuple[int, int, int]] = {}
    for pts in components_mask(g != bg, connectivity=4):
        if int(len(pts)) != 1:
            continue
        r, c = map(int, pts[0])
        side = _edge_side(r, c, h, w)
        if side is None:
            continue
        assert side not in by_side, "expected a single marker on each border side"
        by_side[side] = (r, c, int(g[r, c]))

    need = {"top", "right", "left", "bottom"}
    assert set(by_side) == need, "expected exactly 4 singleton border markers"
    legend_map = np.array(
        [
            [by_side["top"][2], by_side["right"][2]],
            [by_side["left"][2], by_side["bottom"][2]],
        ],
        dtype=int,
    )
    marker_cells = {(r, c) for r, c, _ in by_side.values()}
    return legend_map, marker_cells


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
    anchor_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Tile all pieces into a full rectangle (no holes), allowing rotations + reflections.
    The unique largest piece is treated as the fixed anchor at (0,0).
    Recolor each piece by quantizing its centroid into a k×k grid, using legend_map.
    """
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
    for coords, ph, pw, _, _ in pieces[anchor_idx]["oris"]:
        assert ph <= H and pw <= W, "anchor orientation must fit the unique output rectangle"
        sol = solve_with_fixed_anchor(coords)
        if sol is None:
            continue
        chosen = sol
        break
    assert chosen is not None, "exact-cover assembly failed: no valid anchor orientation"

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a rectangle, recoloring each by edge markers that encode a 2x2 legend.

    Input structure:
    - bg = most common color.
    - Four monochrome non-bg sprites are the actual jigsaw pieces, scattered across the grid.
    - Four additional non-bg singleton pixels sit on the outer border: one on the top edge, one on the right edge,
      one on the left edge, and one on the bottom edge. Together they encode the four quadrant colors.

    Rule:
    - Remove the four border markers from consideration as pieces and rebuild a 2x2 legend from their edge roles.
    - Assemble the remaining four pieces into the unique tight rectangle, allowing rotations and reflections.
    - Recolor each assembled piece by the legend color corresponding to the piece centroid's quadrant.

    Key insight: The color key is no longer embedded inside a special piece; it is split across four edge markers, so
    the solver must first reconstruct the quadrant palette and only then perform the same shape-assembly logic.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    legend_map, marker_cells = _extract_edge_legend_map(g, bg)
    piece_mask = g != bg
    for r, c in marker_cells:
        piece_mask[r, c] = False

    comps = components_mask(piece_mask, connectivity=4)
    assert comps, "grid must contain piece components after removing edge markers"

    k = int(round(len(comps) ** 0.5))
    assert k * k == len(comps) == 4, "piece count must remain 4 after stripping border markers"

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        pieces.append({"oris": oris, "area": len(filled_abs)})

    areas = [int(p["area"]) for p in pieces]
    max_area = max(areas)
    anchor_candidates = [i for i, area in enumerate(areas) if area == max_area]
    assert len(anchor_candidates) == 1, "expected a unique largest anchor piece"
    anchor_idx = anchor_candidates[0]

    return _assemble_exact_cover(pieces, anchor_idx, legend_map, k)


# --------------- generator (o5x) ---------------

# === CFG ===
SIDE_CHOICES = [8, 10, 12]                     # output square side lengths
K = 2                                          # k*k quadrant grid (2*2 = 4 pieces)
NUM_PIECES = K * K                             # total jigsaw pieces
NUM_COLORS = 8                                 # 4 body + 4 palette, all distinct non-bg
AREA_FRAC_LO = 0.08                            # min piece area as fraction of total
AREA_FRAC_HI = 0.43                            # max piece area as fraction of total
AREA_ABS_LO = 11                               # absolute min piece area (cells)
AREA_ABS_HI = 62                               # absolute max piece area (cells)
LEGEND_THRESHOLD = {8: 4, 10: 5, 12: 9}        # min extent threshold per side for anchor L-shape
LEGEND_CORE_SIZE = 4                           # anchor always has a 4x4 filled corner block
LEGEND_ARM_RANGE = range(1, 4)                 # arm width/height options for L-shape construction
PARTITION_GROWTH_EXP = 1.45                    # exponent penalizing large regions during partition
PARTITION_FRONTIER_EXP = 0.2                   # exponent boosting regions with wide frontiers
PARTITION_SEED_JITTER = 0.2                    # random jitter when selecting seed cells
INPUT_SIZE_LO = 20                             # min input grid side
INPUT_SIZE_HI = 26                             # max input grid side
DENSITY_LO = 0.15                              # min non-bg density target
DENSITY_HI = 0.25                              # max non-bg density target
SCATTER_BORDER = 2                             # keep pieces away from edge markers
SCATTER_PADDING = 1                            # gap between piece bounding boxes
ANCHOR_AREA_GAP = 4                            # anchor must be uniquely and visibly largest
GEN_RETRIES = 300                              # max attempts in generate_input()


def _area_bounds(side: int) -> tuple[int, int]:
    """Min/max piece area for a given output side length."""
    total = side * side
    return max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total)), min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))


def _has_hole(mask: np.ndarray) -> bool:
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
        if not (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == sub.shape[0] - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == sub.shape[1] - 1)
        ):
            return True
    return False


def _forces_unique_square(masks: list[np.ndarray], side: int) -> bool:
    """Check that piece bbox dims force a unique S*S factorization of total area."""
    bbs = [bbox_coords(np.argwhere(m)) for m in masks]
    max_h = max(r1 - r0 + 1 for r0, _, r1, _ in bbs)
    max_w = max(c1 - c0 + 1 for _, c0, _, c1 in bbs)
    dims = [(h, w) for h, w in _factor_pairs(side * side) if h >= max_h and w >= max_w]
    return dims == [(side, side)]


def _make_anchor_mask(side: int, min_area: int, max_area: int) -> np.ndarray:
    """Build an L-shaped anchor piece mask anchored at (0,0) of the side*side grid."""
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
    """Flood-fill partition the complement of the anchor into 3 connected regions."""
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
    """Crop a boolean mask to its bounding box."""
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    """Return a random D4 orientation of the cropped mask (unique shapes only)."""
    arr = mask.astype(int)
    opts = []
    seen = set()
    for rot_k in range(4):
        rot = np.rot90(arr, rot_k)
        for oriented in (rot, np.fliplr(rot)):
            r0, c0, r1, c1 = bbox_coords(np.argwhere(oriented > 0))
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop)
    return random.choice(opts)


def _edge_marker_positions(grid_size: int) -> dict[str, tuple[int, int]]:
    """Choose one non-corner position on each border, biased toward the corresponding quadrant corner."""
    mid = grid_size // 2
    top_cols = list(range(2, mid))
    right_rows = list(range(2, mid))
    left_rows = list(range(mid, grid_size - 2))
    bottom_cols = list(range(mid, grid_size - 2))
    assert top_cols and right_rows and left_rows and bottom_cols, "grid too small for edge-marker layout"
    return {
        "top": (0, random.choice(top_cols)),
        "right": (random.choice(right_rows), grid_size - 1),
        "left": (random.choice(left_rows), 0),
        "bottom": (grid_size - 1, random.choice(bottom_cols)),
    }


def _stamp_edge_markers(grid: np.ndarray, legend_map: np.ndarray, bg: int) -> None:
    """Place the four detached legend pixels on the canvas border."""
    h, w = grid.shape
    assert h == w, "input grid is generated square in this task family"
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


def _scatter_pieces(masks: list[np.ndarray], body_colors: list[int], legend_map: np.ndarray, side: int) -> np.ndarray | None:
    """Place all monochrome pieces on a random-sized input grid, then add edge markers."""
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _crop_mask(masks[pi]) if pi == 0 else _random_orientation(_crop_mask(masks[pi]))
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

    _stamp_edge_markers(grid, legend_map, bg=0)
    return grid


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build tiling, validate, scatter pieces."""
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
    if areas[0] < max(areas[1:]) + ANCHOR_AREA_GAP:
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
    raise RuntimeError("failed to generate f560132c D5 marker-edge variant input")
