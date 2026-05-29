# Combo variant of f560132c (index 16)
# combo_spec: D3/2+D9/1
# Name: 6个精灵2x3拼装+多个独立矩形

import math
import random
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior:
#    - Original: 4 scattered pieces, one multicolor legend, all 4 reassemble into
#      one tight rectangle, and the legend is a 2x2 color key.
#    - Fused: 6 scattered pieces, one legend now contains a 2x3 or 3x2 palette.
#      The 6 pieces either:
#        * tile one rectangle and are recolored by 2x3 / 3x2 centroid cells, or
#        * split into two independent 3-piece rectangles, with a 1-cell black gap
#          between the two finished blocks; the first block uses legend row/column 0
#          and the second block uses row/column 1.
#      Example: with a 2x3 legend, the top block uses palette cells (0,0..2) and
#      the bottom block uses (1,0..2). With a 3x2 legend, the left block uses
#      (0..2,0) and the right block uses (0..2,1).
#
# b) Complexity calibration:
#    - The puzzle is visibly different from the source task because the palette is
#      rectangular (6 cells, not 4) and some outputs contain two separated finished
#      rectangles instead of one solid rectangle.
#    - To avoid unobservable over-complexity, the generator only uses two modes:
#      `single` and `split`, and only two palette shapes: 2x3 and 3x2.
#    - The legend always belongs to the first finished block. This keeps the split
#      layout inferable instead of introducing arbitrary row/column permutations.
#
# c) Hidden-assumption audit from the original `main()`:
#    - `len(comps)` is no longer a perfect square. We now require exactly 6 pieces.
#    - The output is not always a single rectangle. Solver now searches both one
#      full-rectangle cover and two separate 3-piece rectangle covers.
#    - Filtering rectangle dimensions by `max_h/max_w over all orientations` is too
#      strong for non-square cases. We now only require that each piece has at least
#      one orientation that fits a candidate rectangle.
#    - ARC only has colors 1..9, so 6 body colors plus 6 palette colors cannot all
#      be distinct. The generator therefore uses 6 distinct palette colors and the
#      remaining 3 colors as repeating body colors. The legend is still uniquely
#      identifiable because it is the only multicolor component.
#
# d) Placement legality:
#    - Generator constructs assembled masks first, verifies connectivity / no-hole
#      constraints, then scatters the cropped piece sprites with separation by using
#      `random_free_location`, so pieces never overlap or leave the canvas.
#
# e/f/g) Observability, cleanliness, and inferability:
#    - The palette shape itself is visible to humans (2x3 or 3x2).
#    - Split outputs have exactly two clean rectangular blocks separated by one black
#      row or column, so the “multiple independent rectangles” behavior is obvious.
#    - Color changes are structured by position inside those blocks, not by hidden
#      numeric properties of color ids.
# -----------------------------------------------------------------------------


PALETTE_SHAPES = [(2, 3), (3, 2)]
NUM_PIECES = 6
BG = 0

SINGLE_DIMS = {
    (2, 3): [(8, 12), (9, 12), (8, 13), (9, 13), (10, 14)],
    (3, 2): [(12, 8), (12, 9), (13, 8), (13, 9), (14, 10)],
}
SPLIT_STACK_HEIGHTS = [5, 6, 7]
SPLIT_STACK_WIDTHS = [10, 11, 12, 13]
SPLIT_SIDE_HEIGHTS = [10, 11, 12, 13]
SPLIT_SIDE_WIDTHS = [5, 6, 7]

MIN_PIECE_AREA = 8
MAX_PIECE_FRAC = 0.42
INPUT_SIZE_LO = 22
INPUT_SIZE_HI = 30
DENSITY_LO = 0.14
DENSITY_HI = 0.24
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 400
PARTITION_GROWTH_EXP = 1.45
PARTITION_FRONTIER_EXP = 0.2
PARTITION_SEED_JITTER = 0.2


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


def _extract_legend_candidates(g: np.ndarray, pts: np.ndarray) -> list[dict]:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    out = []
    for ph, pw in PALETTE_SHAPES:
        if hh < ph or ww < pw:
            continue
        best = None
        best_rc = None
        best_window = None
        for rr in range(hh - ph + 1):
            for cc in range(ww - pw + 1):
                window = sub[rr : rr + ph, cc : cc + pw]
                distinct = int(len(set(map(int, window.ravel()))))
                key = (distinct, -rr, -cc)
                if best is None or key > best:
                    best = key
                    best_rc = (rr, cc)
                    best_window = np.asarray(window, dtype=int).copy()
        if best_window is not None:
            out.append({"shape": (ph, pw), "score": int(best[0]), "rc": best_rc, "map": best_window})
    out.sort(key=lambda item: (item["score"], item["shape"][0] * item["shape"][1], -item["shape"][0]), reverse=True)
    return out


def _coords_to_mask(coords: tuple[tuple[int, int], ...], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _build_piece_records_from_components(g: np.ndarray, comps: list[np.ndarray], bg: int) -> list[dict]:
    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        arr = np.asarray(list(filled_abs), dtype=int)
        r0, c0, _r1, _c1 = bbox_coords(arr)
        rel = tuple((int(r - r0), int(c - c0)) for r, c in filled_abs)
        rel_norm, h, w = _normalize_coords(rel)
        pieces.append(
            {
                "coords": rel_norm,
                "mask": _coords_to_mask(rel_norm, h, w),
                "oris": _dihedral_orientations(rel_norm),
                "area": len(filled_abs),
            }
        )
    return pieces


def _candidate_dims_for_subset(pieces: list[dict], subset: tuple[int, ...]) -> list[tuple[int, int]]:
    total = sum(int(pieces[pi]["area"]) for pi in subset)
    dims = []
    for h, w in _factor_pairs(total):
        ok = True
        for pi in subset:
            if not any(oh <= h and ow <= w for _coords, oh, ow, _rk, _fl in pieces[pi]["oris"]):
                ok = False
                break
        if ok:
            dims.append((int(h), int(w)))
    dims.sort()
    return dims


def _make_placement(coords: tuple[tuple[int, int], ...], h: int, w: int, r0: int, c0: int) -> tuple[int, int, int, tuple[tuple[int, int], ...]]:
    m = 0
    for dr, dc in coords:
        idx = (int(r0) + int(dr)) * int(w) + (int(c0) + int(dc))
        m |= 1 << idx
    return m, int(r0), int(c0), coords


def _exact_cover_subset(
    pieces: list[dict],
    subset: tuple[int, ...],
    h: int,
    w: int,
    anchor_pi: int | None = None,
) -> dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] | None:
    nbits = int(h) * int(w)
    all_mask = (1 << nbits) - 1
    placements: dict[int, list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = {}
    for pi in subset:
        opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
        for coords, ph, pw, _rk, _fl in pieces[pi]["oris"]:
            if ph > h or pw > w:
                continue
            if anchor_pi is not None and pi == anchor_pi:
                opts.append(_make_placement(coords, h, w, 0, 0))
                continue
            for r0 in range(0, h - ph + 1):
                for c0 in range(0, w - pw + 1):
                    opts.append(_make_placement(coords, h, w, r0, c0))
        if not opts:
            return None
        placements[pi] = opts

    def search(
        used: int,
        chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]],
        remaining: set[int],
    ) -> bool:
        if not remaining:
            return used == all_mask
        empty = (~used) & all_mask
        if empty == 0:
            return False
        cell = (empty & -empty).bit_length() - 1
        branch = []
        for pi in sorted(remaining):
            opts = []
            for pl in placements[pi]:
                m = pl[0]
                if ((m >> cell) & 1) and (m & used) == 0:
                    opts.append(pl)
            if opts:
                branch.append((len(opts), pi, opts))
        if not branch:
            return False
        branch.sort(key=lambda item: (item[0], item[1]))
        for _count, pi, opts in branch:
            remaining.remove(pi)
            for pl in opts:
                chosen[pi] = pl
                if search(used | pl[0], chosen, remaining):
                    return True
            chosen.pop(pi, None)
            remaining.add(pi)
        return False

    if anchor_pi is None:
        chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] = {}
        remaining = set(subset)
        if not search(0, chosen, remaining):
            return None
        return chosen

    if anchor_pi not in placements:
        return None
    for anchor_pl in placements[anchor_pi]:
        chosen = {anchor_pi: anchor_pl}
        remaining = set(subset)
        remaining.remove(anchor_pi)
        if search(anchor_pl[0], chosen, remaining):
            return chosen
    return None


def _piece_centroid_cell(
    placement: tuple[int, int, int, tuple[tuple[int, int], ...]],
    h: int,
    w: int,
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    _mask, r0, c0, coords = placement
    rs = [r0 + dr for dr, _dc in coords]
    cs = [c0 + dc for _dr, dc in coords]
    cr = float(sum(rs)) / float(len(rs))
    cc = float(sum(cs)) / float(len(cs))
    rows, cols = grid_shape
    rr = int(cr * float(rows) / float(h))
    cc2 = int(cc * float(cols) / float(w))
    rr = max(0, min(int(rows) - 1, rr))
    cc2 = max(0, min(int(cols) - 1, cc2))
    return rr, cc2


def _render_single_output(
    chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]],
    legend_map: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray | None:
    rows, cols = map(int, legend_map.shape)
    piece_cells = {pi: _piece_centroid_cell(pl, h, w, (rows, cols)) for pi, pl in chosen.items()}
    if len(set(piece_cells.values())) != len(chosen):
        return None
    want = {(r, c) for r in range(rows) for c in range(cols)}
    if set(piece_cells.values()) != want:
        return None
    out = np.zeros((h, w), dtype=int)
    for pi, pl in chosen.items():
        _mask, r0, c0, coords = pl
        rr, cc = piece_cells[pi]
        col = int(legend_map[rr, cc])
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = col
    return out


def _render_group_output(
    chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]],
    rect_h: int,
    rect_w: int,
    legend_map: np.ndarray,
    major_idx: int,
    palette_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[int, tuple[int, int]]] | None:
    rows, cols = palette_shape
    out = np.zeros((rect_h, rect_w), dtype=int)
    piece_cells: dict[int, tuple[int, int]] = {}
    if palette_shape == (2, 3):
        qcols = {}
        for pi, pl in chosen.items():
            rr, cc = _piece_centroid_cell(pl, rect_h, rect_w, (1, 3))
            qcols[pi] = cc
        if sorted(qcols.values()) != [0, 1, 2]:
            return None
        for pi, pl in chosen.items():
            cell = (major_idx, qcols[pi])
            piece_cells[pi] = cell
            color = int(legend_map[cell])
            _mask, r0, c0, coords = pl
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = color
    else:
        qrows = {}
        for pi, pl in chosen.items():
            rr, cc = _piece_centroid_cell(pl, rect_h, rect_w, (3, 1))
            qrows[pi] = rr
        if sorted(qrows.values()) != [0, 1, 2]:
            return None
        for pi, pl in chosen.items():
            cell = (qrows[pi], major_idx)
            piece_cells[pi] = cell
            color = int(legend_map[cell])
            _mask, r0, c0, coords = pl
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = color
    return out, piece_cells


def _collect_unique(outputs: list[np.ndarray], arr: np.ndarray | None) -> None:
    if arr is None:
        return
    key = (tuple(arr.shape), arr.tobytes())
    for old in outputs:
        if key == (tuple(old.shape), old.tobytes()):
            return
    outputs.append(arr)


def _solve_single_mode(pieces: list[dict], legend_idx: int, legend_map: np.ndarray) -> list[np.ndarray]:
    subset = tuple(range(len(pieces)))
    out = []
    for h, w in _candidate_dims_for_subset(pieces, subset):
        chosen = _exact_cover_subset(pieces, subset, h, w, anchor_pi=legend_idx)
        if chosen is None:
            continue
        _collect_unique(out, _render_single_output(chosen, legend_map, h, w))
    return out


def _solve_split_mode(pieces: list[dict], legend_idx: int, legend_map: np.ndarray) -> list[np.ndarray]:
    palette_shape = tuple(map(int, legend_map.shape))
    assert palette_shape in PALETTE_SHAPES
    others = [pi for pi in range(len(pieces)) if pi != legend_idx]
    out = []
    for combo in combinations(others, 2):
        group0 = tuple(sorted((legend_idx,) + combo))
        group1 = tuple(sorted(pi for pi in range(len(pieces)) if pi not in group0))
        for h0, w0 in _candidate_dims_for_subset(pieces, group0):
            chosen0 = _exact_cover_subset(pieces, group0, h0, w0, anchor_pi=legend_idx)
            if chosen0 is None:
                continue
            rendered0 = _render_group_output(chosen0, h0, w0, legend_map, 0, palette_shape)
            if rendered0 is None:
                continue
            rect0, _cells0 = rendered0
            for h1, w1 in _candidate_dims_for_subset(pieces, group1):
                chosen1 = _exact_cover_subset(pieces, group1, h1, w1, anchor_pi=None)
                if chosen1 is None:
                    continue
                rendered1 = _render_group_output(chosen1, h1, w1, legend_map, 1, palette_shape)
                if rendered1 is None:
                    continue
                rect1, _cells1 = rendered1
                if palette_shape == (2, 3):
                    canvas = np.zeros((h0 + 1 + h1, max(w0, w1)), dtype=int)
                    paste(canvas, rect0, 0, 0, transparent=0)
                    paste(canvas, rect1, h0 + 1, 0, transparent=0)
                else:
                    canvas = np.zeros((max(h0, h1), w0 + 1 + w1), dtype=int)
                    paste(canvas, rect0, 0, 0, transparent=0)
                    paste(canvas, rect1, 0, w0 + 1, transparent=0)
                _collect_unique(out, canvas)
    return out


def _all_candidate_outputs(pieces: list[dict], legend_idx: int, legend_candidates: list[dict]) -> tuple[list[np.ndarray], dict[str, int]]:
    outputs: list[np.ndarray] = []
    stats = {"single": 0, "split": 0}
    for cand in legend_candidates:
        if cand["score"] < 6:
            continue
        single = _solve_single_mode(pieces, legend_idx, cand["map"])
        split = _solve_split_mode(pieces, legend_idx, cand["map"])
        stats["single"] += len(single)
        stats["split"] += len(split)
        for arr in single + split:
            _collect_unique(outputs, arr)
    return outputs, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble six scattered sprites using a 2x3-or-3x2 legend, then recolor them by spatial slot.

    Input structure:
    - Background is the most common color.
    - There are exactly six 4-connected non-background sprites.
    - One sprite is the legend: inside its body sits a 2x3 or 3x2 multicolor palette block.
    - The other five sprites are monochrome body pieces.

    Rule:
    - Treat every sprite as a polyomino piece with D4 symmetry allowed.
    - Either all six pieces tile one rectangle, or they split into two separate
      three-piece rectangles.
    - The legend palette supplies six output colors. In the single-rectangle case,
      a piece gets the color of the palette cell matching its centroid bucket in the
      assembled 2x3 / 3x2 grid. In the split case, the legend-containing block is
      the first row/column of that palette and the other finished block is the second.

    Key insight:
    - The legend is no longer just “four quadrants in one square”. Its visible 2x3
      or 3x2 palette tells you both the number of target slots and whether the final
      arrangement is one big rectangle or two independent rectangular subassemblies.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) == NUM_PIECES, 'grid must contain exactly six non-background components'

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)
    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates_idx = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates_idx) == 1 and max_distinct >= 6, 'legend must be the unique multicolor component'
    legend_idx = legend_candidates_idx[0]

    legend_candidates = _extract_legend_candidates(g, comps[legend_idx])
    assert legend_candidates, 'legend must contain a 2x3 or 3x2 palette candidate'
    pieces = _build_piece_records_from_components(g, comps, bg)
    outputs, _stats = _all_candidate_outputs(pieces, legend_idx, legend_candidates)
    assert len(outputs) == 1, 'assembly must yield exactly one fused-layout output'
    return outputs[0]


def _macro_intervals(n: int, k: int) -> list[tuple[int, int]]:
    base = n // k
    rem = n % k
    out = []
    cur = 0
    for i in range(k):
        span = base + (1 if i < rem else 0)
        out.append((cur, cur + span))
        cur += span
    return out


def _legend_mask_options(host_h: int, host_w: int, palette_h: int, palette_w: int, total_area: int) -> list[np.ndarray]:
    core_h = palette_h + 2
    core_w = palette_w + 2
    min_area = max(core_h * core_w, MIN_PIECE_AREA + 6)
    max_area = max(min_area, int(total_area * MAX_PIECE_FRAC))
    opts = []
    arm_h_max = min(3, core_h)
    arm_w_max = min(3, core_w)
    for h0 in range(core_h + 1, host_h + 1):
        for w0 in range(core_w + 1, host_w + 1):
            for arm_h in range(1, arm_h_max + 1):
                for arm_w in range(1, arm_w_max + 1):
                    mask = np.zeros((host_h, host_w), dtype=bool)
                    mask[:core_h, :core_w] = True
                    mask[:arm_h, :w0] = True
                    mask[:h0, :arm_w] = True
                    area = int(mask.sum())
                    if min_area <= area <= max_area:
                        opts.append(mask)
    return opts


def _find_best_seed(rest: np.ndarray, r_lo: int, r_hi: int, c_lo: int, c_hi: int) -> tuple[int, int] | None:
    cells = np.argwhere(rest)
    if len(cells) == 0:
        return None
    tr = 0.5 * (r_lo + r_hi - 1)
    tc = 0.5 * (c_lo + c_hi - 1)
    best = None
    best_cell = None
    for r, c in cells:
        penalty = 0.0
        if not (r_lo <= r < r_hi):
            penalty += 6.0 + min(abs(r - r_lo), abs(r - (r_hi - 1)))
        if not (c_lo <= c < c_hi):
            penalty += 6.0 + min(abs(c - c_lo), abs(c - (c_hi - 1)))
        dist = abs(float(r) - tr) + abs(float(c) - tc) + random.random() * PARTITION_SEED_JITTER
        key = penalty + dist
        if best is None or key < best:
            best = key
            best_cell = (int(r), int(c))
    return best_cell


def _partition_mask(mask: np.ndarray, seed_cells: list[tuple[int, int]]) -> np.ndarray | None:
    labels = np.full(mask.shape, -1, dtype=int)
    available = set(map(tuple, np.argwhere(mask)))
    if len(available) < len(seed_cells):
        return None
    areas = [0 for _ in seed_cells]
    frontiers = [set() for _ in seed_cells]

    def add_frontier(i: int, r: int, c: int) -> None:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and labels[rr, cc] < 0:
                frontiers[i].add((rr, cc))

    for i, seed in enumerate(seed_cells):
        if seed not in available:
            return None
        labels[seed] = i
        available.remove(seed)
        areas[i] = 1
        add_frontier(i, seed[0], seed[1])

    while available:
        candidates = []
        for i in range(len(seed_cells)):
            frontiers[i] = {p for p in frontiers[i] if labels[p] < 0}
            if frontiers[i]:
                candidates.append(i)
        if not candidates:
            return None
        weights = [
            (1.0 / (areas[i] ** PARTITION_GROWTH_EXP)) * (len(frontiers[i]) ** PARTITION_FRONTIER_EXP)
            for i in candidates
        ]
        idx = random.choices(candidates, weights=weights, k=1)[0]
        cell = random.choice(tuple(frontiers[idx]))
        frontiers[idx].discard(cell)
        labels[cell] = idx
        available.discard(cell)
        areas[idx] += 1
        add_frontier(idx, cell[0], cell[1])

    return labels


def _piece_masks_valid(masks: list[np.ndarray]) -> bool:
    total = int(sum(int(m.sum()) for m in masks))
    for mask in masks:
        area = int(mask.sum())
        if area < MIN_PIECE_AREA or area > int(total * MAX_PIECE_FRAC):
            return False
        if len(components_mask(mask, connectivity=4)) != 1:
            return False
        if _has_hole(mask):
            return False
    return True


def _build_piece_records_from_masks(masks: list[np.ndarray]) -> list[dict]:
    pieces = []
    for mask in masks:
        pts = np.argwhere(mask)
        r0, c0, r1, c1 = bbox_coords(pts)
        crop = mask[r0 : r1 + 1, c0 : c1 + 1]
        coords = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(crop)))
        norm, h, w = _normalize_coords(coords)
        pieces.append({"coords": norm, "mask": _coords_to_mask(norm, h, w), "oris": _dihedral_orientations(norm), "area": len(norm)})
    return pieces


def _palette_spots(mask: np.ndarray, palette_h: int, palette_w: int) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(1, h - palette_h):
        for c in range(1, w - palette_w):
            if mask[r : r + palette_h, c : c + palette_w].all() and mask[r - 1 : r + palette_h + 1, c - 1 : c + palette_w + 1].all():
                spots.append((r, c))
    return spots


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
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


def _render_single_expected(masks: list[np.ndarray], legend_map: np.ndarray, host_shape: tuple[int, int]) -> np.ndarray | None:
    h, w = host_shape
    out = np.zeros((h, w), dtype=int)
    cells = []
    for mask in masks:
        pts = np.argwhere(mask)
        cr = float(pts[:, 0].mean())
        cc = float(pts[:, 1].mean())
        rr = min(legend_map.shape[0] - 1, int(cr * legend_map.shape[0] / h))
        cc2 = min(legend_map.shape[1] - 1, int(cc * legend_map.shape[1] / w))
        cells.append((rr, cc2))
    if set(cells) != {(r, c) for r in range(legend_map.shape[0]) for c in range(legend_map.shape[1])}:
        return None
    for mask, cell in zip(masks, cells):
        out[mask] = int(legend_map[cell])
    return out


def _render_split_expected(
    group0: list[np.ndarray],
    group1: list[np.ndarray],
    legend_map: np.ndarray,
    palette_shape: tuple[int, int],
) -> np.ndarray | None:
    if palette_shape == (2, 3):
        h0, w0 = group0[0].shape
        h1, w1 = group1[0].shape
        for mask in group0 + group1:
            assert mask.ndim == 2
        cols0 = []
        for mask in group0:
            pts = np.argwhere(mask)
            cols0.append(min(2, int(float(pts[:, 1].mean()) * 3 / w0)))
        cols1 = []
        for mask in group1:
            pts = np.argwhere(mask)
            cols1.append(min(2, int(float(pts[:, 1].mean()) * 3 / w1)))
        if sorted(cols0) != [0, 1, 2] or sorted(cols1) != [0, 1, 2]:
            return None
        canvas = np.zeros((h0 + 1 + h1, max(w0, w1)), dtype=int)
        for mask, col in zip(group0, cols0):
            canvas[:h0, :w0][mask] = int(legend_map[0, col])
        for mask, col in zip(group1, cols1):
            canvas[h0 + 1 : h0 + 1 + h1, :w1][mask] = int(legend_map[1, col])
        return canvas

    h0, w0 = group0[0].shape
    h1, w1 = group1[0].shape
    rows0 = []
    for mask in group0:
        pts = np.argwhere(mask)
        rows0.append(min(2, int(float(pts[:, 0].mean()) * 3 / h0)))
    rows1 = []
    for mask in group1:
        pts = np.argwhere(mask)
        rows1.append(min(2, int(float(pts[:, 0].mean()) * 3 / h1)))
    if sorted(rows0) != [0, 1, 2] or sorted(rows1) != [0, 1, 2]:
        return None
    canvas = np.zeros((max(h0, h1), w0 + 1 + w1), dtype=int)
    for mask, row in zip(group0, rows0):
        canvas[:h0, :w0][mask] = int(legend_map[row, 0])
    for mask, row in zip(group1, rows1):
        canvas[:h1, w0 + 1 : w0 + 1 + w1][mask] = int(legend_map[row, 1])
    return canvas


def _scatter_pieces(piece_masks: list[np.ndarray], body_colors: list[int], legend_map: np.ndarray, legend_idx: int) -> np.ndarray | None:
    total = sum(int(mask.sum()) for mask in piece_masks)
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None
    side = random.randint(nlo, nhi)
    grid = np.zeros((side, side), dtype=int)
    occupied = np.zeros_like(grid)
    palette_h, palette_w = legend_map.shape

    order = list(range(len(piece_masks)))
    random.shuffle(order)
    if legend_idx in order:
        order.remove(legend_idx)
    order = [legend_idx] + order

    for pi in order:
        base = _crop_mask(piece_masks[pi])
        oriented = base if pi == legend_idx else _random_orientation(base)
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = int(body_colors[pi])
        if pi == legend_idx:
            spots = _palette_spots(oriented, palette_h, palette_w)
            if not spots:
                return None
            rr, cc = random.choice(spots)
            patch[rr : rr + palette_h, cc : cc + palette_w] = legend_map
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


def _choose_colors() -> tuple[list[int], np.ndarray]:
    palette_colors = random.sample(range(1, 10), 6)
    body_pool = [c for c in range(1, 10) if c not in palette_colors]
    assert len(body_pool) == 3
    body_colors = [body_pool[i % len(body_pool)] for i in range(NUM_PIECES)]
    random.shuffle(body_colors)
    legend_map = np.asarray(palette_colors, dtype=int)
    return body_colors, legend_map


def _vertical_triplet_masks(h: int, w: int, min_left: int) -> list[np.ndarray] | None:
    min_mid = 2
    min_right = 2
    if w < min_left + min_mid + min_right:
        return None
    left = max(min_left, int(round(w / 3.0)))
    left = min(left, w - min_mid - min_right)
    remain = w - left
    mid = max(min_mid, int(round(remain / 2.0)))
    mid = min(mid, remain - min_right)
    right = w - left - mid
    if right < min_right:
        return None
    b1 = left
    b2 = left + mid
    masks = [np.zeros((h, w), dtype=bool) for _ in range(3)]
    for r in range(h):
        if r > 0:
            lo1 = min_left
            hi1 = b2 - min_mid
            b1 = max(lo1, min(hi1, b1 + random.choice([-1, 0, 1])))
            lo2 = b1 + min_mid
            hi2 = w - min_right
            b2 = max(lo2, min(hi2, b2 + random.choice([-1, 0, 1])))
        masks[0][r, :b1] = True
        masks[1][r, b1:b2] = True
        masks[2][r, b2:] = True
    return masks


def _horizontal_triplet_masks(h: int, w: int, min_top: int) -> list[np.ndarray] | None:
    min_mid = 2
    min_bot = 2
    if h < min_top + min_mid + min_bot:
        return None
    top = max(min_top, int(round(h / 3.0)))
    top = min(top, h - min_mid - min_bot)
    remain = h - top
    mid = max(min_mid, int(round(remain / 2.0)))
    mid = min(mid, remain - min_bot)
    bot = h - top - mid
    if bot < min_bot:
        return None
    b1 = top
    b2 = top + mid
    masks = [np.zeros((h, w), dtype=bool) for _ in range(3)]
    for c in range(w):
        if c > 0:
            lo1 = min_top
            hi1 = b2 - min_mid
            b1 = max(lo1, min(hi1, b1 + random.choice([-1, 0, 1])))
            lo2 = b1 + min_mid
            hi2 = h - min_bot
            b2 = max(lo2, min(hi2, b2 + random.choice([-1, 0, 1])))
        masks[0][:b1, c] = True
        masks[1][b1:b2, c] = True
        masks[2][b2:, c] = True
    return masks


def _single_mode_masks(palette_shape: tuple[int, int]) -> list[np.ndarray] | None:
    host_h, host_w = random.choice(SINGLE_DIMS[palette_shape])
    legend_opts = _legend_mask_options(host_h, host_w, palette_shape[0], palette_shape[1], host_h * host_w)
    if not legend_opts:
        return None
    legend = random.choice(legend_opts).copy()
    rest = ~legend
    rows, cols = palette_shape
    row_intervals = _macro_intervals(host_h, rows)
    col_intervals = _macro_intervals(host_w, cols)
    targets = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != (0, 0)]
    seeds = []
    for rr, cc in targets:
        seed = _find_best_seed(rest, row_intervals[rr][0], row_intervals[rr][1], col_intervals[cc][0], col_intervals[cc][1])
        if seed is None or seed in seeds:
            return None
        seeds.append(seed)
        rest[seed] = False
        rest[seed] = True
    labels = _partition_mask(rest, seeds)
    if labels is None:
        return None
    masks = [legend] + [labels == i for i in range(len(seeds))]
    if not _piece_masks_valid(masks):
        return None
    expected = _render_single_expected(masks, np.zeros(palette_shape, dtype=int), (host_h, host_w))
    if expected is None:
        return None
    return masks


def _single_mode_instance(palette_shape: tuple[int, int]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray] | None:
    built = _single_mode_masks(palette_shape)
    if built is None:
        return None
    masks = built
    body_colors, palette_flat = _choose_colors()
    legend_map = palette_flat.reshape(palette_shape)
    expected = _render_single_expected(masks, legend_map, masks[0].shape)
    if expected is None:
        return None
    pieces = _build_piece_records_from_masks(masks)
    single = _solve_single_mode(pieces, 0, legend_map)
    split = _solve_split_mode(pieces, 0, legend_map)
    if len(single) != 1 or split:
        return None
    if not np.array_equal(single[0], expected):
        return None
    return masks, legend_map, np.asarray(body_colors, dtype=int), expected


def _split_mode_instance(palette_shape: tuple[int, int]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray] | None:
    if palette_shape == (2, 3):
        h0 = random.choice(SPLIT_STACK_HEIGHTS)
        h1 = random.choice(SPLIT_STACK_HEIGHTS)
        w0 = random.choice(SPLIT_STACK_WIDTHS)
        w1 = random.choice(SPLIT_STACK_WIDTHS)
        if w0 == w1:
            w1 = SPLIT_STACK_WIDTHS[(SPLIT_STACK_WIDTHS.index(w1) + 1) % len(SPLIT_STACK_WIDTHS)]
        rect0_shape = (h0, w0)
        rect1_shape = (h1, w1)
    else:
        h0 = random.choice(SPLIT_SIDE_HEIGHTS)
        h1 = random.choice(SPLIT_SIDE_HEIGHTS)
        if h0 == h1:
            h1 = SPLIT_SIDE_HEIGHTS[(SPLIT_SIDE_HEIGHTS.index(h1) + 1) % len(SPLIT_SIDE_HEIGHTS)]
        w0 = random.choice(SPLIT_SIDE_WIDTHS)
        w1 = random.choice(SPLIT_SIDE_WIDTHS)
        rect0_shape = (h0, w0)
        rect1_shape = (h1, w1)

    if palette_shape == (2, 3):
        group0 = _vertical_triplet_masks(rect0_shape[0], rect0_shape[1], min_left=palette_shape[1] + 2)
        group1 = _vertical_triplet_masks(rect1_shape[0], rect1_shape[1], min_left=2)
    else:
        group0 = _horizontal_triplet_masks(rect0_shape[0], rect0_shape[1], min_top=palette_shape[0] + 2)
        group1 = _horizontal_triplet_masks(rect1_shape[0], rect1_shape[1], min_top=2)

    if group0 is None or group1 is None:
        return None

    masks_local = group0 + group1
    if not _piece_masks_valid(masks_local):
        return None

    body_colors, palette_flat = _choose_colors()
    legend_map = palette_flat.reshape(palette_shape)
    expected = _render_split_expected(group0, group1, legend_map, palette_shape)
    if expected is None:
        return None

    if palette_shape == (2, 3):
        canvas_h = rect0_shape[0] + 1 + rect1_shape[0]
        canvas_w = max(rect0_shape[1], rect1_shape[1])
        global_masks = []
        for mask in group0:
            big = np.zeros((canvas_h, canvas_w), dtype=bool)
            big[: rect0_shape[0], : rect0_shape[1]] = mask
            global_masks.append(big)
        for mask in group1:
            big = np.zeros((canvas_h, canvas_w), dtype=bool)
            big[rect0_shape[0] + 1 : rect0_shape[0] + 1 + rect1_shape[0], : rect1_shape[1]] = mask
            global_masks.append(big)
    else:
        canvas_h = max(rect0_shape[0], rect1_shape[0])
        canvas_w = rect0_shape[1] + 1 + rect1_shape[1]
        global_masks = []
        for mask in group0:
            big = np.zeros((canvas_h, canvas_w), dtype=bool)
            big[: rect0_shape[0], : rect0_shape[1]] = mask
            global_masks.append(big)
        for mask in group1:
            big = np.zeros((canvas_h, canvas_w), dtype=bool)
            big[: rect1_shape[0], rect0_shape[1] + 1 : rect0_shape[1] + 1 + rect1_shape[1]] = mask
            global_masks.append(big)

    pieces = _build_piece_records_from_masks(global_masks)
    single = _solve_single_mode(pieces, 0, legend_map)
    split = _solve_split_mode(pieces, 0, legend_map)
    if single or len(split) != 1:
        return None
    if not np.array_equal(split[0], expected):
        return None
    return global_masks, legend_map, np.asarray(body_colors, dtype=int), expected


def _gen_attempt() -> np.ndarray | None:
    try:
        mode = random.choice(["single", "split"])
        palette_shape = random.choice(PALETTE_SHAPES)
        if mode == "single":
            built = _single_mode_instance(palette_shape)
        else:
            built = _split_mode_instance(palette_shape)
        if built is None:
            return None
        piece_masks, legend_map, body_colors, expected = built
        grid = _scatter_pieces(piece_masks, list(map(int, body_colors)), legend_map, legend_idx=0)
        if grid is None:
            return None
        out = main(grid)
        if not np.array_equal(out, expected):
            return None
        stats = _analyze_input(grid)
        if stats["unique_outputs"] != 1:
            return None
        if mode == "single" and not (stats["single"] >= 1 and stats["split"] == 0):
            return None
        if mode == "split" and not (stats["split"] >= 1 and stats["single"] == 0):
            return None
        return grid
    except Exception:
        return None


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate fused f560132c input')


def _analyze_input(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)
    legend_idx = max(range(len(comps)), key=lambda i: len(comp_colors[i]))
    pieces = _build_piece_records_from_components(g, comps, bg)
    legend_candidates = _extract_legend_candidates(g, comps[legend_idx])
    outputs, stats = _all_candidate_outputs(pieces, legend_idx, legend_candidates)
    stats["unique_outputs"] = len(outputs)
    return stats
