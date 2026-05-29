# Variant of f560132c
# Dimension: D4 — 变换：拼装后的颜色重映射规则 (transform)
# Change: s.color = legend[s.position_in_composite].color — 每个精灵在拼装后被重映射为图例对应位置的单色
#         → 基于图例行列分别着色：行色决定主体，列色决定边缘

# Design analysis
# a) Concrete behavior and trace:
#    - The 4 scattered pieces still assemble into one tight square, and one piece is still a legend.
#    - The legend no longer contains a 2x2 "answer palette". Instead it embeds a k x (2k) code block.
#      For k=2 that block is 2x4:
#          [row0 row0 col0 col1]
#          [row1 row1 col0 col1]
#      The left half is decoded by rows; the right half is decoded by columns.
#    - After assembly, a piece is still assigned to one (row, col) quadrant by its centroid.
#      Original logic: piece at (0,1) became one flat color = legend[0,1].
#      New logic: piece at (0,1) gets body=row0 and its 1-cell piece boundary gets edge=col1.
#      The behavior change is localized to the recoloring stage, but the legend encoding must also change so
#      the two independent color roles are visible in the input.
# b) Complexity calibration:
#    - Keep k=2 and four pieces, so the assembly difficulty stays close to the original task.
#    - Use only four legend cue colors total: two repeated row colors and two repeated column colors.
#      This creates repeated evidence across pieces without exploding the color vocabulary.
#    - Reject pieces with no interior cells; otherwise a thin snake would become all "edge color" and hide
#      the main body color, making the new rule visually ambiguous.
# c) Hidden assumptions from the original main():
#    - `_extract_legend_map(...)` assumed the legend cue was a k x k block scored only by distinct colors.
#      That no longer matches the input, so it must be replaced with a structured stripe decoder.
#    - `_assemble_exact_cover(...)` used one color per piece. In the variant, a single fill loop would destroy
#      the body/edge distinction, so the implementation must paint body first and overwrite boundary cells second.
#    - The old generator allowed simply connected but possibly very thin pieces. That was harmless for flat fills,
#      but it is unsafe now because boundary-only pieces would erase the row-color signal.
# d) Placement legality:
#    - Pieces are still scattered with the same non-overlap and padding logic.
#    - The legend code block is written only inside cells belonging to the legend mask, so the legend remains one
#      connected component and the solver still sees the same four-piece object layout.
# e) Visual observability:
#    - In the output, top-row pieces share one body color and bottom-row pieces share another.
#    - Left-column pieces share one border color and right-column pieces share another.
#    - This repeated cross-cutting pattern is visible from a few examples and differs clearly from the old
#      "each quadrant is flat and independent" behavior.
# f) Visual cleanliness:
#    - The edge color is limited to a 1-cell outline of each assembled piece, so the composite remains readable.
#    - The legend code is a compact rectangular stripe block instead of many scattered markers.
# g) Color-rule inferability:
#    - Only four cue colors participate in the output transform.
#    - Each row color appears twice as a body color, and each column color appears twice as an edge color.
#      That repetition gives enough structure for humans to infer the decomposition.
# Main/generator contract review:
#    - `main()` now assumes each piece has at least one non-boundary cell so the body color can appear.
#      `generate_input()` enforces that by rejecting masks whose assembled shape has zero interior cells.
#    - `main()` assumes the legend contains exactly one valid k x (2k) stripe block. The generator guarantees this
#      by inserting one block with 2k distinct cue colors into an otherwise monochrome legend piece.

import math
import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

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


def _extract_legend_codes(g: np.ndarray, pts: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Inside the legend bbox, find a k x (2k) stripe code block.
    Left half: each row is uniform and yields the row colors.
    Right half: each column is uniform and yields the column colors.
    """
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert k > 0 and hh >= k and ww >= 2 * k, "legend component must fit the row/column code block"

    best_rc = None
    for rr in range(hh - k + 1):
        for cc in range(ww - (2 * k) + 1):
            window = sub[rr : rr + k, cc : cc + (2 * k)]
            left = window[:, :k]
            right = window[:, k:]
            row_colors = [int(left[r, 0]) for r in range(k)]
            col_colors = [int(right[0, c]) for c in range(k)]
            if len(set(row_colors)) != k or len(set(col_colors)) != k:
                continue
            if set(row_colors) & set(col_colors):
                continue
            ok_rows = all(np.all(left[r, :] == left[r, 0]) for r in range(k))
            ok_cols = all(np.all(right[:, c] == right[0, c]) for c in range(k))
            if not (ok_rows and ok_cols):
                continue
            if len(set(map(int, window.ravel()))) != 2 * k:
                continue
            best_rc = (rr, cc)
            break
        if best_rc is not None:
            break

    assert best_rc is not None, "legend component must contain a valid row/column stripe code block"
    rr, cc = best_rc
    window = sub[rr : rr + k, cc : cc + (2 * k)]
    row_colors = np.asarray([int(window[r, 0]) for r in range(k)], dtype=int)
    col_colors = np.asarray([int(window[0, k + c]) for c in range(k)], dtype=int)
    return row_colors, col_colors


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


def _boundary_cells(coords: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    coord_set = set(coords)
    edge = set()
    for r, c in coords:
        if any((r2, c2) not in coord_set for r2, c2 in neighbors4(r, c)):
            edge.add((r, c))
    return edge


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    row_colors: np.ndarray,
    col_colors: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Tile all pieces into a full rectangle (no holes), allowing rotations + reflections.
    Recolor each piece by quantizing its centroid into a k x k grid.
    The row code yields the body color and the column code yields the boundary color.
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
            body = int(row_colors[rr])
            edge = int(col_colors[cc2])
            assert body != edge, "row and column colors must stay distinct"
            boundary = _boundary_cells(coords)
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = body
            for dr, dc in boundary:
                out[r0 + dr, c0 + dc] = edge
        return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a rectangle, then recolor each piece with separate body and edge cues.

    Input structure:
    - bg is the most common color.
    - Several scattered non-bg sprites are jigsaw pieces; all but one are monochrome.
    - The legend piece contains a structured k x (2k) color code: left half is row stripes, right half is column stripes.

    Rule:
    - Tile all pieces into one tight rectangle, allowing rotations and reflections.
    - Use each piece's centroid to decide which row and column of the assembled k x k layout it belongs to.
    - Repaint that piece so its interior/body uses the legend row color, while its 1-cell boundary uses the legend column color.

    Key insight:
    - The legend no longer gives one color per quadrant. Instead it splits vertical and horizontal roles, so the output color
      of a piece is decoded by combining a row cue with a column cue.
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
    row_colors, col_colors = _extract_legend_codes(g, comps[legend_idx], k)

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        pieces.append({"oris": oris, "area": len(filled_abs)})

    return _assemble_exact_cover(pieces, legend_idx, row_colors, col_colors, k)


# --------------- generator (o5x) ---------------

# === CFG ===
SIDE_CHOICES = [8, 10, 12]                    # output square side lengths
K = 2                                         # k*k quadrant grid (2*2 = 4 pieces)
NUM_PIECES = K * K                            # total jigsaw pieces
NUM_COLORS = 8                                # 4 piece-body colors + 4 legend cue colors
AREA_FRAC_LO = 0.08                           # min piece area as fraction of total
AREA_FRAC_HI = 0.43                           # max piece area as fraction of total
AREA_ABS_LO = 11                              # absolute min piece area (cells)
AREA_ABS_HI = 62                              # absolute max piece area (cells)
LEGEND_THRESHOLD = {8: 4, 10: 5, 12: 9}      # min extent threshold per side for legend L-shape
LEGEND_CORE_SIZE = 4                          # legend always has a 4x4 filled corner block
LEGEND_ARM_RANGE = range(1, 4)                # arm width/height options for L-shape construction
PARTITION_GROWTH_EXP = 1.45                   # exponent penalizing large regions during partition
PARTITION_FRONTIER_EXP = 0.2                  # exponent boosting regions with wide frontiers
PARTITION_SEED_JITTER = 0.2                   # random jitter when selecting seed cells
INPUT_SIZE_LO = 20                            # min input grid side
INPUT_SIZE_HI = 26                            # max input grid side
DENSITY_LO = 0.15                             # min non-bg density target
DENSITY_HI = 0.25                             # max non-bg density target
SCATTER_BORDER = 1                            # border margin when placing pieces
SCATTER_PADDING = 1                           # gap between piece bounding boxes
GEN_RETRIES = 300                             # max attempts in generate_input()


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


def _interior_count(mask: np.ndarray) -> int:
    pts = set(map(tuple, np.argwhere(mask)))
    count = 0
    for r, c in pts:
        if all((r2, c2) in pts for r2, c2 in neighbors4(r, c)):
            count += 1
    return count


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
    for kturn in range(4):
        rot = np.rot90(arr, kturn)
        for oriented in (rot, np.fliplr(rot)):
            r0, c0, r1, c1 = bbox_coords(np.argwhere(oriented > 0))
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop)
    return random.choice(opts)


def _make_legend_code_block(row_colors: np.ndarray, col_colors: np.ndarray) -> np.ndarray:
    k = len(row_colors)
    block = np.zeros((k, 2 * k), dtype=int)
    for r in range(k):
        block[r, :k] = int(row_colors[r])
    for c in range(k):
        block[:, k + c] = int(col_colors[c])
    return block


def _legend_code_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    k = K
    spots = []
    h, w = mask.shape
    for r in range(0, h - k + 1):
        for c in range(0, w - (2 * k) + 1):
            if mask[r : r + k, c : c + (2 * k)].all():
                spots.append((r, c))
    return spots


def _paint_dual_piece(mask: np.ndarray, body_color: int, edge_color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    coords = tuple(map(tuple, np.argwhere(mask)))
    boundary = _boundary_cells(coords)
    patch[mask] = int(body_color)
    for r, c in boundary:
        patch[r, c] = int(edge_color)
    return patch


def _scatter_pieces(
    masks: list[np.ndarray],
    body_colors: list[int],
    row_colors: np.ndarray,
    col_colors: np.ndarray,
    side: int,
) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)
    legend_block = _make_legend_code_block(row_colors, col_colors)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _crop_mask(masks[pi]) if pi == 0 else _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        if pi == 0:
            spots = _legend_code_spots(oriented)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r : r + K, c : c + (2 * K)] = legend_block
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
    if any(_interior_count(m) == 0 for m in masks):
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
    cue_colors = colors[NUM_PIECES:]
    random.shuffle(cue_colors)
    row_colors = np.asarray(cue_colors[:K], dtype=int)
    col_colors = np.asarray(cue_colors[K:], dtype=int)

    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        piece = _paint_dual_piece(mask, int(row_colors[rr]), int(col_colors[cc]))
        expected = np.where(mask, piece, expected)

    grid = _scatter_pieces(masks, body_colors, row_colors, col_colors, side)
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
    raise RuntimeError("failed to generate f560132c input")
