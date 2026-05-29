# Combo variant of f560132c (index 26)
# combo_spec: D4/3+D5/1+D14/2
# Name: 行列分别着色+嵌套框架图例+颜色循环移位
#
# Design analysis
# a) Concrete behavior and trace:
#    - The input still has four scattered polyomino pieces that tile one square after D4 assembly.
#      One piece is the legend because it is the only multicolor component.
#    - Original task:
#      1. Reassemble the four pieces into one square.
#      2. Read a 2x2 legend palette by position.
#      3. Paint each piece one flat color from its assembled quadrant.
#    - Fused variant:
#      1. Reassemble the four pieces into one square.
#      2. Read a 7x7 nested-frame legend; outer ring, second ring, third ring, and center define
#         four cue colors in that exact outer-to-inner order.
#      3. Rank the assembled pieces by area, largest to smallest, and assign those four cue colors
#         to the four pieces in that rank order.
#      4. Do not use those cue colors directly. Shift them along the legend cycle
#         outer -> second -> third -> center -> outer.
#      5. For each assembled row, the larger piece in that row supplies the body color.
#         For each assembled column, the smaller piece in that column supplies the edge color.
#         Every piece is then repainted as: body = row color, boundary = column color.
#    - Small concrete trace:
#      Suppose the nested legend decodes to [2, 6, 8, 3] outer->inner.
#      After shifting, the usable colors become [6, 8, 3, 2].
#      If the assembled piece areas by quadrant are TL=54, TR=31, BL=27, BR=43, then the rank map is
#      TL->2, BR->6, TR->8, BL->3 before shift, and TL->6, BR->8, TR->3, BL->2 after shift.
#      Top row larger piece is TL, so both top-row pieces get body 6.
#      Bottom row larger piece is BR, so both bottom-row pieces get body 8.
#      Left column smaller piece is BL, so left-column edges become 2.
#      Right column smaller piece is TR, so right-column edges become 3.
#      The original task would have produced four flat quadrant colors; this variant instead reuses
#      the four shifted legend colors across rows and columns in a crossed body/edge pattern.
# b) Complexity calibration:
#    - Keep k=2 and four pieces so the geometric assembly remains close to the source task.
#    - Keep exactly four cue colors in the nested legend, matching the four assembled pieces.
#    - Use only two row colors and two column colors in the output. This creates repeated evidence:
#      each body color appears on two pieces, each edge color appears on two pieces.
#    - Reject assemblies whose row representatives and column representatives are not four distinct
#      pieces. Without that, some shifted legend colors would become unused and the fusion would
#      visually collapse toward a simpler rule.
#    - Reject pieces with no interior cells so the body/edge split remains visible.
# c) Hidden assumptions audited against the original main():
#    - `_extract_legend_map(...)` assumed a 2x2 positional palette. That assumption is invalid; the
#      legend parser must instead search for a monochrome 7x7 nested-frame window.
#    - `_assemble_exact_cover(...)` assumed one output color per piece. That fails here because each
#      piece needs separate body and boundary painting, so coloring must happen after layout recovery.
#    - The original recoloring logic depended only on centroid quadrant. The fused rule also depends
#      on area order and on within-row / within-column size comparisons, so the solver must retain
#      per-piece identity, area, and assembled cell.
#    - A literal splice of D14/2 would cycle raw input body colors. That conflicts with D5/1 because
#      the nested-frame cue colors must stay clearly visible and distinct from the legend body's fill.
#      In this fusion, the cycle is therefore applied to the four decoded nested-frame colors after
#      the area-layer assignment step and before the row/column paint step.
# d) Placement legality:
#    - The generator still uses output-first construction: build a valid square tiling, then scatter
#      the cropped pieces onto a larger black canvas with `random_free_location`.
#    - The legend's 7x7 nested code is painted only inside legend cells, so the legend stays one
#      connected component. Pieces never overlap and always stay within bounds.
# e) Visual observability:
#    - The nested legend is a strong visible cue for the area ordering.
#    - The shifted colors reappear in the output, but not in the same outer-to-inner order, which
#      gives the cycle step observable evidence.
#    - The row/column reuse is visible because two pieces share each body color and two pieces share
#      each edge color. Across examples, the specific pieces supplying those shared colors move with
#      the larger-in-row / smaller-in-column comparisons, so the rule is not reducible to a fixed
#      quadrant template.
# f) Visual cleanliness:
#    - Only the legend is multicolor in the input; the other pieces remain monochrome.
#    - The nested marker is compact and contiguous.
#    - The output uses a 1-cell boundary overlay instead of arbitrary speckles, so body and edge are
#      easy to distinguish.
# g) Change inferability:
#    - There are only four functional cue colors, and all four are reused in the output.
#    - The generator filters out degenerate layouts where a simpler "flat area color" or "same rule
#      without cycle shift" explanation would match too closely.
# Main/generator contract review:
#    - `main()` assumes exactly one valid nested 7x7 legend window. `generate_input()` guarantees it
#      by reserving a solid 7x7 patch inside the legend piece.
#    - `main()` assumes piece areas are pairwise distinct and that each row/column comparison has a
#      unique winner. `generate_input()` enforces distinct areas.
#    - `main()` assumes the four role pieces (two row winners and two column losers) are all
#      different so every shifted legend color is used. `generate_input()` rejects samples that do
#      not satisfy that full-role condition.

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
    if window.shape != (NESTED_SIZE, NESTED_SIZE):
        return None
    layers = []
    for depth in range(NESTED_LAYERS):
        ring = window[_ring_mask(NESTED_SIZE, depth)]
        cols = set(map(int, ring.ravel()))
        if len(cols) != 1:
            return None
        layers.append(int(next(iter(cols))))
    if len(set(layers)) != NESTED_LAYERS:
        return None
    return layers


def _extract_nested_legend_colors(g: np.ndarray, pts: np.ndarray) -> list[int]:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert hh >= NESTED_SIZE and ww >= NESTED_SIZE, "legend component must fit the nested-frame motif"

    best = None
    best_colors = None
    for rr in range(hh - NESTED_SIZE + 1):
        for cc in range(ww - NESTED_SIZE + 1):
            window = sub[rr : rr + NESTED_SIZE, cc : cc + NESTED_SIZE]
            colors = _nested_window_colors(window)
            if colors is None:
                continue
            key = (-rr, -cc)
            if best is None or key > best:
                best = key
                best_colors = colors
    assert best_colors is not None, "legend component must contain a valid nested-frame code"
    return list(best_colors)


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


def _boundary_cells(coords: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    coord_set = set(coords)
    edge = set()
    for r, c in coords:
        if any((r2, c2) not in coord_set for r2, c2 in neighbors4(r, c)):
            edge.add((r, c))
    return edge


def _solve_layout(
    pieces: list[dict],
    legend_idx: int,
) -> tuple[int, int, dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]], dict[int, tuple[int, int]]]:
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
                    pi = other_pis[local_idx]
                    for pl in opts_k:
                        chosen[pi] = pl
                        if search(used | pl[0]):
                            return True
                    chosen.pop(pi, None)
                    remaining.add(local_idx)
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
            rr = int(cr * 2.0 / float(H))
            cc2 = int(cc * 2.0 / float(W))
            rr = max(0, min(1, rr))
            cc2 = max(0, min(1, cc2))
            piece_cell[pi] = (rr, cc2)
        return H, W, chosen, piece_cell

    raise AssertionError("no valid rectangle dimensions")


def _cycle_successor_map(legend_layers: list[int]) -> dict[int, int]:
    assert len(legend_layers) == NESTED_LAYERS, "expected four nested legend colors"
    assert len(set(legend_layers)) == NESTED_LAYERS, "nested legend colors must be distinct"
    return {int(legend_layers[i]): int(legend_layers[(i + 1) % NESTED_LAYERS]) for i in range(NESTED_LAYERS)}


def _derive_fused_colors(
    areas: list[int],
    piece_cell: dict[int, tuple[int, int]],
    legend_layers: list[int],
) -> tuple[dict[int, int], np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    assert len(areas) == NUM_PIECES, "this fusion is defined for four pieces"
    assert len(set(areas)) == NUM_PIECES, "piece areas must be distinct for nested-frame ranking"

    ranked = sorted(range(len(areas)), key=lambda pi: (-areas[pi], pi))
    successor = _cycle_successor_map(legend_layers)
    shifted_piece_color: dict[int, int] = {}
    for rank, pi in enumerate(ranked):
        shifted_piece_color[pi] = int(successor[int(legend_layers[rank])])

    row_rep: dict[int, int] = {}
    col_rep: dict[int, int] = {}
    for rr in range(K):
        row_pis = [pi for pi, cell in piece_cell.items() if cell[0] == rr]
        assert len(row_pis) == K, "each assembled row must contain two pieces"
        row_rep[rr] = max(row_pis, key=lambda pi: (areas[pi], -pi))
    for cc in range(K):
        col_pis = [pi for pi, cell in piece_cell.items() if cell[1] == cc]
        assert len(col_pis) == K, "each assembled column must contain two pieces"
        col_rep[cc] = min(col_pis, key=lambda pi: (areas[pi], pi))

    role_pis = set(row_rep.values()) | set(col_rep.values())
    assert len(role_pis) == NUM_PIECES, "fusion requires four distinct row/column role pieces"

    row_colors = np.asarray([shifted_piece_color[row_rep[rr]] for rr in range(K)], dtype=int)
    col_colors = np.asarray([shifted_piece_color[col_rep[cc]] for cc in range(K)], dtype=int)
    assert len(set(map(int, row_colors.tolist() + col_colors.tolist()))) == NUM_PIECES, "all shifted legend colors must be used"
    return shifted_piece_color, row_colors, col_colors, row_rep, col_rep


def _paint_layout(
    H: int,
    W: int,
    chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]],
    piece_cell: dict[int, tuple[int, int]],
    row_colors: np.ndarray,
    col_colors: np.ndarray,
) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    for pi, pl in chosen.items():
        _, r0, c0, coords = pl
        rr, cc = piece_cell[pi]
        body = int(row_colors[rr])
        edge = int(col_colors[cc])
        boundary = _boundary_cells(coords)
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = body
        for dr, dc in boundary:
            out[r0 + dr, c0 + dc] = edge
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered pieces into one square, then repaint them with row-based bodies and column-based edges derived from a shifted nested-frame legend.

    Input structure:
    - `bg` is the most common color.
    - Four scattered non-background components are the puzzle pieces.
    - Exactly one component is multicolor: the legend piece contains a compact 7x7 nested-frame motif whose four concentric layers encode four cue colors from outermost to innermost.

    Rule:
    - Reassemble the four pieces into the unique tight square, allowing rotations and reflections.
    - Order the assembled pieces by area from largest to smallest and match them to the legend layers from outer ring to center.
    - Shift those four legend colors one step along the nested order.
    - In each assembled row, the larger piece supplies that row's body color; in each assembled column, the smaller piece supplies that column's edge color.
    - Repaint every piece with the body color of its row and the boundary color of its column.

    Key insight:
    - The legend is no longer a direct 2x2 lookup. You must decode concentric layers, convert layer order into an area ranking, apply the color cycle, and only then split the resulting four colors into row and column roles.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, "grid must contain at least one non-background component"

    k = int(round(len(comps) ** 0.5))
    assert k == K and k * k == len(comps), "piece count must be exactly four"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend component must be uniquely identified by distinct colors"
    legend_idx = legend_candidates[0]
    legend_layers = _extract_nested_legend_colors(g, comps[legend_idx])

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        pieces.append({"oris": _dihedral_orientations(rel_norm), "area": len(filled_abs)})

    H, W, chosen, piece_cell = _solve_layout(pieces, legend_idx)
    areas = [int(p["area"]) for p in pieces]
    _, row_colors, col_colors, _, _ = _derive_fused_colors(areas, piece_cell, legend_layers)
    return _paint_layout(H, W, chosen, piece_cell, row_colors, col_colors)


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
GEN_RETRIES = 500
MIN_SHIFTED_CELL_MISMATCH = 20
MIN_FLAT_CELL_MISMATCH = 20


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
                    remaining = side * side - area
                    if min_area <= area <= max_area and remaining >= 3 * min_area:
                        opts.append(mask)
    assert opts, "legend mask options must exist"
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


def _qcells_from_masks(masks: list[np.ndarray], side: int) -> dict[int, tuple[int, int]]:
    piece_cell = {}
    for pi, mask in enumerate(masks):
        pts = np.argwhere(mask)
        rr = min(1, int(pts[:, 0].mean() * K / side))
        cc = min(1, int(pts[:, 1].mean() * K / side))
        piece_cell[pi] = (rr, cc)
    return piece_cell


def _expected_from_masks(masks: list[np.ndarray], legend_layers: list[int], side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    piece_cell = _qcells_from_masks(masks, side)
    areas = [int(mask.sum()) for mask in masks]
    shifted_piece_color, row_colors, col_colors, row_rep, col_rep = _derive_fused_colors(areas, piece_cell, legend_layers)
    out = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = piece_cell[pi]
        coords = tuple(map(tuple, np.argwhere(mask)))
        boundary = _boundary_cells(coords)
        body = int(row_colors[rr])
        edge = int(col_colors[cc])
        out[mask] = body
        for r, c in boundary:
            out[r, c] = edge
    return out, row_colors, col_colors, row_rep, col_rep


def _unshifted_decomp_output(masks: list[np.ndarray], legend_layers: list[int], side: int) -> np.ndarray:
    piece_cell = _qcells_from_masks(masks, side)
    areas = [int(mask.sum()) for mask in masks]
    ranked = sorted(range(NUM_PIECES), key=lambda pi: (-areas[pi], pi))
    base_color = {pi: int(legend_layers[rank]) for rank, pi in enumerate(ranked)}

    row_rep = {}
    col_rep = {}
    for rr in range(K):
        row_pis = [pi for pi, cell in piece_cell.items() if cell[0] == rr]
        row_rep[rr] = max(row_pis, key=lambda pi: (areas[pi], -pi))
    for cc in range(K):
        col_pis = [pi for pi, cell in piece_cell.items() if cell[1] == cc]
        col_rep[cc] = min(col_pis, key=lambda pi: (areas[pi], pi))

    row_colors = np.asarray([base_color[row_rep[rr]] for rr in range(K)], dtype=int)
    col_colors = np.asarray([base_color[col_rep[cc]] for cc in range(K)], dtype=int)
    out = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = piece_cell[pi]
        coords = tuple(map(tuple, np.argwhere(mask)))
        boundary = _boundary_cells(coords)
        out[mask] = int(row_colors[rr])
        for r, c in boundary:
            out[r, c] = int(col_colors[cc])
    return out


def _flat_area_output(masks: list[np.ndarray], legend_layers: list[int]) -> np.ndarray:
    areas = [int(mask.sum()) for mask in masks]
    ranked = sorted(range(NUM_PIECES), key=lambda pi: (-areas[pi], pi))
    piece_color = {pi: int(legend_layers[rank]) for rank, pi in enumerate(ranked)}
    side = masks[0].shape[0]
    out = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        out[mask] = piece_color[pi]
    return out


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
        patch[oriented] = int(body_colors[pi])
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
    if any(_interior_count(m) == 0 for m in masks):
        return None
    if not _forces_unique_square(masks, side):
        return None

    piece_cell = _qcells_from_masks(masks, side)
    if len(set(piece_cell.values())) != NUM_PIECES:
        return None

    body_colors = random.sample(range(1, 10), NUM_COLORS)[:NUM_PIECES]
    cue_colors = random.sample([c for c in range(1, 10) if c not in body_colors], NESTED_LAYERS)
    random.shuffle(cue_colors)
    legend_layers = list(map(int, cue_colors))

    try:
        expected, row_colors, col_colors, row_rep, col_rep = _expected_from_masks(masks, legend_layers, side)
    except AssertionError:
        return None

    if len(set(row_rep.values()) | set(col_rep.values())) != NUM_PIECES:
        return None
    if set(map(int, row_colors.tolist() + col_colors.tolist())) != set(_cycle_successor_map(legend_layers).values()):
        return None

    unshifted = _unshifted_decomp_output(masks, legend_layers, side)
    flat_area = _flat_area_output(masks, legend_layers)
    shift_mismatch = int(np.count_nonzero(expected != unshifted))
    flat_mismatch = int(np.count_nonzero(expected != flat_area))
    if shift_mismatch < MIN_SHIFTED_CELL_MISMATCH or flat_mismatch < MIN_FLAT_CELL_MISMATCH:
        return None

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
    raise RuntimeError("failed to generate fused f560132c combo input")
