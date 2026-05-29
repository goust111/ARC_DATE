# Combo variant of f560132c (index 8)
# combo_spec: D4/0+D8/1
# Name: 保留原色+图例在边缘

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Input now contains four monochrome pieces plus one separate border legend strip on the
#      outermost bottom row or right column. The strip has four contiguous runs, read as a 2x2 map.
#    - Output is still a tight rectangle assembled from the four pieces, but pieces keep their own
#      colors. The legend no longer recolors them; instead it says which color belongs in which 2x2 slot.
#    - Small trace:
#      old: find multicolor legend-piece -> assemble -> recolor each assembled piece by the legend cell
#      at its centroid.
#      combo: read the edge strip first -> remove it from the piece mask -> use proximity to pick the
#      anchored piece -> assemble while enforcing "piece color == legend color of its assembled slot" ->
#      paint the output with the original piece colors.
#    - To make this position mapping genuinely matter, the generator builds two congruent rectangles
#      (top-right and bottom-left slots) and swaps their colors in the legend. Geometry alone cannot
#      tell which same-shape rectangle should occupy which slot; the border legend resolves that.
# b) Complexity calibration:
#    - The border legend still has exactly four runs, matching the original 2x2 structure, so the
#      added rule is visible without exploding the hypothesis space.
#    - Only one pair of non-anchor pieces is intentionally congruent. This is enough to force the
#      color-to-slot mapping to matter, while leaving the overall tiling easy to validate.
# c) Implicit assumptions audited against the original solver:
#    - `components_mask(g != bg)` no longer returns only pieces; the border strip must be removed first.
#    - "legend = unique multi-color component" is invalid because all pieces are monochrome here.
#    - The original recoloring step assumed legend colors become output colors. That breaks: output
#      colors must come from the input pieces, so the solver must track each piece's body color.
#    - Exact-cover search can no longer stop at the first geometric tiling. Because two pieces may be
#      shape-swappable, the solver must keep the legend's color-position constraints inside the search.
# d) Placement legality:
#    - The strip is drawn away from corners and one blank cell away from the nearest piece, so it stays
#      a separate object on exactly one border side.
#    - The nearest piece to the strip is always the L-shaped anchor; the other three are pushed outside
#      an exclusion band near that edge.
#    - All four pieces are scattered with non-overlap and padding, and every returned sample is checked
#      by the solver before it is accepted.
# e) Visual observability:
#    - Humans see an obvious border strip and four solid-color pieces. The strip colors are the same
#      four piece colors, so the "which color goes to which slot" cue is directly readable.
#    - Because two scattered pieces can have the same shape after normalization but different colors,
#      the need for the border legend is visible across samples.
# f) Visual cleanliness:
#    - Border legend and piece field are separated spatially; no overlay is used.
#    - Pieces remain large, simply connected, and monochrome, so the scene stays readable.
# g) Color inferability:
#    - Only four non-background colors are used in each sample, and the exact same four colors appear
#      on the border legend. There is no hidden ordering on color ids.
#    - The generator always swaps the two congruent rectangle colors between their natural slots, so the
#      border legend has a measurable effect on the final arrangement instead of redundantly echoing it.
# 9) New solver assumptions and generator guarantees:
#    - Solver assumes piece colors are unique and match the four legend colors exactly. Generator enforces
#      that by using four distinct body colors and reusing them on the border strip.
#    - Solver assumes the closest piece to the strip is unique. Generator places the anchor in a reserved
#      near-strip zone and keeps the others out of that zone.
#    - Solver assumes at least one geometric ambiguity can be broken by color-position mapping. Generator
#      guarantees this by making the top-right and bottom-left output slots congruent and swapping them in
#      the legend map.


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


def _parse_edge_runs(vals: np.ndarray, bg: int) -> tuple[list[int], int, int] | None:
    idxs = [i for i, v in enumerate(vals) if int(v) != bg]
    if not idxs:
        return None
    lo = min(idxs)
    hi = max(idxs)
    segment = [int(v) for v in vals[lo : hi + 1]]
    if any(v == bg for v in segment):
        return None
    colors = [segment[0]]
    for v in segment[1:]:
        if v != colors[-1]:
            colors.append(v)
    if len(set(colors)) != len(colors):
        return None
    return colors, int(lo), int(hi)


def _extract_edge_legend(g: np.ndarray, bg: int) -> tuple[str, int, int, int, np.ndarray]:
    h, w = g.shape
    edge_specs = [
        ("top", g[0, :]),
        ("bottom", g[h - 1, :]),
        ("left", g[:, 0]),
        ("right", g[:, w - 1]),
    ]
    candidates: list[tuple[str, int, int, int, np.ndarray]] = []
    for side, vals in edge_specs:
        parsed = _parse_edge_runs(np.asarray(vals, dtype=int), bg)
        if parsed is None:
            continue
        colors, lo, hi = parsed
        k = int(round(len(colors) ** 0.5))
        if k * k != len(colors):
            continue
        arr = np.asarray(colors, dtype=int)
        legend_map = arr.reshape(k, k) if side in ("top", "bottom") else arr.reshape(k, k, order="F")
        candidates.append((side, lo, hi, int(k), legend_map))
    assert len(candidates) == 1, "grid must contain exactly one valid edge legend strip"
    return candidates[0]


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


def _distance_to_side(pts: np.ndarray, side: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox_coords(pts)
    if side == "top":
        return int(r0)
    if side == "bottom":
        return int(h - 1 - r1)
    if side == "left":
        return int(c0)
    if side == "right":
        return int(w - 1 - c1)
    raise ValueError(f"unknown side: {side}")


def _placement_cell(r0: int, c0: int, coords: tuple[tuple[int, int], ...], H: int, W: int, k: int) -> tuple[int, int]:
    rs = [r0 + dr for dr, _ in coords]
    cs = [c0 + dc for _, dc in coords]
    cr = float(sum(rs)) / float(len(rs))
    cc = float(sum(cs)) / float(len(cs))
    rr = int(cr * float(k) / float(H))
    cc2 = int(cc * float(k) / float(W))
    rr = max(0, min(int(k) - 1, rr))
    cc2 = max(0, min(int(k) - 1, cc2))
    return rr, cc2


def _piece_color(g: np.ndarray, pts: np.ndarray, bg: int) -> int:
    cols = {int(g[r, c]) for r, c in pts if int(g[r, c]) != bg}
    assert len(cols) == 1, "each piece must be monochrome in this combo variant"
    return next(iter(cols))


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
    assert len(dims) == 1, "piece areas and bounds must imply a unique output rectangle"

    legend_colors = [int(x) for x in legend_map.ravel()]
    piece_colors = [int(p["color"]) for p in pieces]
    assert len(set(piece_colors)) == len(piece_colors), "piece colors must be unique"
    assert sorted(piece_colors) == sorted(legend_colors), "legend colors must match the piece colors exactly"

    H, W = dims[0]
    nbits = H * W
    all_mask = (1 << nbits) - 1

    placements: list[list[tuple[int, int, int, tuple[tuple[int, int], ...], tuple[int, int]]]] = []
    for piece in pieces:
        opts: list[tuple[int, int, int, tuple[tuple[int, int], ...], tuple[int, int]]] = []
        for coords, ph, pw, _, _ in piece["oris"]:
            for r0 in range(0, H - ph + 1):
                for c0 in range(0, W - pw + 1):
                    cell = _placement_cell(int(r0), int(c0), coords, H, W, k)
                    if int(piece["color"]) != int(legend_map[cell[0], cell[1]]):
                        continue
                    m = 0
                    for dr, dc in coords:
                        idx = (int(r0) + int(dr)) * int(W) + (int(c0) + int(dc))
                        m |= 1 << idx
                    opts.append((m, int(r0), int(c0), coords, cell))
        assert opts, "each piece must have at least one placement compatible with the legend map"
        placements.append(opts)

    other_pis = [i for i in range(len(pieces)) if i != legend_idx]
    placements_other = [placements[pi] for pi in other_pis]

    def solve_with_fixed_legend(
        legend_pl: tuple[int, int, int, tuple[tuple[int, int], ...], tuple[int, int]],
    ) -> dict[int, tuple[int, int, int, tuple[tuple[int, int], ...], tuple[int, int]]] | None:
        legend_mask = legend_pl[0]
        filtered: list[list[tuple[int, int, int, tuple[tuple[int, int], ...], tuple[int, int]]]] = []
        for opts in placements_other:
            filtered.append([pl for pl in opts if (pl[0] & legend_mask) == 0])

        remaining = set(range(len(other_pis)))
        chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...], tuple[int, int]]] = {
            legend_idx: legend_pl
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
    for legend_pl in placements[legend_idx]:
        if legend_pl[1] != 0 or legend_pl[2] != 0:
            continue
        sol = solve_with_fixed_legend(legend_pl)
        if sol is None:
            continue
        chosen = sol
        break
    assert chosen is not None, "exact-cover assembly failed: no color-consistent tiling"

    out = np.zeros((H, W), dtype=int)
    for pi, pl in chosen.items():
        _, r0, c0, coords, _cell = pl
        col = int(pieces[pi]["color"])
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered pieces into a rectangle, using a border legend to decide positions while preserving each piece's own color.

    Input structure:
    - `bg` is the most common color.
    - Four monochrome non-background sprites are scattered inside the grid.
    - A separate legend sits on the outermost bottom row or right column as four contiguous color runs.
    - The piece nearest that border legend is the assembly anchor; the legend strip itself is not a piece.

    Rule:
    - Decode the border strip as a 2x2 color-position map.
    - Remove the strip, tile the four pieces into a tight rectangle with rotations/reflections allowed, and anchor the nearest piece at assembled top-left.
    - Keep every piece's original color; only accept placements whose assembled 2x2 slot matches that piece color in the decoded legend.

    Key insight:
    - The border strip no longer tells you how to recolor the finished assembly. It tells you which colored piece belongs in each assembled slot, and that extra cue is necessary because some pieces are shape-swappable.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    side, lo, hi, k, legend_map = _extract_edge_legend(g, bg)

    piece_mask = g != bg
    if side == "top":
        piece_mask[0, lo : hi + 1] = False
    elif side == "bottom":
        piece_mask[h - 1, lo : hi + 1] = False
    elif side == "left":
        piece_mask[lo : hi + 1, 0] = False
    else:
        piece_mask[lo : hi + 1, w - 1] = False

    comps = components_mask(piece_mask, connectivity=4)
    assert comps, "grid must contain pieces after removing the edge legend"
    assert len(comps) == k * k, "piece count must match the decoded edge legend"

    dists = [_distance_to_side(pts, side, g.shape) for pts in comps]
    min_dist = min(dists)
    legend_candidates = [i for i, d in enumerate(dists) if d == min_dist]
    assert len(legend_candidates) == 1, "legend anchor piece must be uniquely closest to the edge legend"
    legend_idx = legend_candidates[0]

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        pieces.append(
            {
                "oris": _dihedral_orientations(rel_norm),
                "area": len(filled_abs),
                "color": _piece_color(g, pts, bg),
            }
        )

    return _assemble_exact_cover(pieces, legend_idx, legend_map, k)


# --------------- generator (o5x) ---------------

# === CFG ===
SIDE_CHOICES = [8, 10]
K = 2
NUM_PIECES = K * K
INPUT_SIZE_LO = 20
INPUT_SIZE_HI = 26
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 2
SCATTER_PADDING = 1
EDGE_SIDES = ("bottom", "right")
EDGE_SEG_LEN_CHOICES = [2, 3]
EDGE_MARGIN = 1
LEGEND_GAP = 1
LEGEND_EXCLUSION_BAND = 6
GEN_RETRIES = 300
SHAPE_PARAMS = {
    8: (4, [5]),
    10: (4, [6, 7]),
}


def _build_masks(side: int, arm: int, extent: int) -> list[np.ndarray]:
    legend = np.zeros((side, side), dtype=bool)
    legend[:arm, :extent] = True
    legend[:extent, :arm] = True

    top_right = np.zeros((side, side), dtype=bool)
    top_right[:arm, extent:] = True

    bottom_left = np.zeros((side, side), dtype=bool)
    bottom_left[extent:, :arm] = True

    bottom_right = np.zeros((side, side), dtype=bool)
    bottom_right[arm:, arm:] = True
    return [legend, top_right, bottom_left, bottom_right]


def _forces_unique_square(masks: list[np.ndarray], side: int) -> bool:
    bbs = [bbox_coords(np.argwhere(m)) for m in masks]
    max_h = max(r1 - r0 + 1 for r0, _, r1, _ in bbs)
    max_w = max(c1 - c0 + 1 for _, c0, _, c1 in bbs)
    dims = [(h, w) for h, w in _factor_pairs(side * side) if h >= max_h and w >= max_w]
    return dims == [(side, side)]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_orientation(mask: np.ndarray) -> np.ndarray:
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


def _paint_edge_strip(grid: np.ndarray, side: str, start: int, seg_len: int, legend_map: np.ndarray):
    colors = legend_map.ravel().tolist() if side == "bottom" else legend_map.T.ravel().tolist()
    if side == "bottom":
        c = start
        for col in colors:
            grid[-1, c : c + seg_len] = int(col)
            c += seg_len
    elif side == "right":
        r = start
        for col in colors:
            grid[r : r + seg_len, -1] = int(col)
            r += seg_len
    else:
        raise ValueError(f"unsupported edge side: {side}")


def _legend_piece_position(grid_size: int, edge_side: str, shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = shape
    if edge_side == "bottom":
        r0 = grid_size - h - (LEGEND_GAP + 1)
        c_lo = 1
        c_hi = grid_size - w - 2
        if c_lo > c_hi or r0 < 1:
            return None
        return int(r0), random.randint(c_lo, c_hi)
    if edge_side == "right":
        c0 = grid_size - w - (LEGEND_GAP + 1)
        r_lo = 1
        r_hi = grid_size - h - 2
        if r_lo > r_hi or c0 < 1:
            return None
        return random.randint(r_lo, r_hi), int(c0)
    return None


def _strip_start_candidates(grid_size: int, span: int, bbox_lo: int, bbox_hi: int) -> list[int]:
    lo = EDGE_MARGIN
    hi = grid_size - span - EDGE_MARGIN
    if lo > hi:
        return []
    starts = []
    for start in range(lo, hi + 1):
        end = start + span - 1
        if start <= bbox_hi and end >= bbox_lo:
            starts.append(int(start))
    return starts


def _reserve_band(occupied: np.ndarray, edge_side: str):
    if edge_side == "bottom":
        occupied[-LEGEND_EXCLUSION_BAND:, :] = 1
    elif edge_side == "right":
        occupied[:, -LEGEND_EXCLUSION_BAND:] = 1
    else:
        raise ValueError(f"unsupported edge side: {edge_side}")


def _expected_output(masks: list[np.ndarray], legend_map: np.ndarray) -> np.ndarray:
    side = masks[0].shape[0]
    out = np.zeros((side, side), dtype=int)
    out[masks[0]] = int(legend_map[0, 0])
    out[masks[1]] = int(legend_map[0, 1])
    out[masks[2]] = int(legend_map[1, 0])
    out[masks[3]] = int(legend_map[1, 1])
    return out


def _natural_output(masks: list[np.ndarray], body_colors: list[int]) -> np.ndarray:
    side = masks[0].shape[0]
    out = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        out[mask] = int(body_colors[pi])
    return out


def _scatter_pieces(masks: list[np.ndarray], body_colors: list[int], legend_map: np.ndarray, out_side: int) -> tuple[np.ndarray, dict] | None:
    total = out_side * out_side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    edge_side = random.choice(EDGE_SIDES)
    seg_len = random.choice(EDGE_SEG_LEN_CHOICES)
    span = K * K * seg_len

    legend_oriented = _random_orientation(_crop_mask(masks[0]))
    legend_pos = _legend_piece_position(grid_size, edge_side, legend_oriented.shape)
    if legend_pos is None:
        return None
    lr0, lc0 = legend_pos
    lh, lw = legend_oriented.shape

    if edge_side == "bottom":
        starts = _strip_start_candidates(grid_size, span, lc0, lc0 + lw - 1)
    else:
        starts = _strip_start_candidates(grid_size, span, lr0, lr0 + lh - 1)
    if not starts:
        return None
    strip_start = random.choice(starts)

    _paint_edge_strip(grid, edge_side, strip_start, seg_len, legend_map)
    if edge_side == "bottom":
        occupied[-1, strip_start : strip_start + span] = 1
    else:
        occupied[strip_start : strip_start + span, -1] = 1

    legend_patch = np.zeros(legend_oriented.shape, dtype=int)
    legend_patch[legend_oriented] = int(body_colors[0])
    paste(grid, legend_patch, lr0, lc0, transparent=0)
    occupied[lr0 : lr0 + lh, lc0 : lc0 + lw] = 1

    _reserve_band(occupied, edge_side)
    occupied[lr0 : lr0 + lh, lc0 : lc0 + lw] = 1

    for pi in random.sample(range(1, NUM_PIECES), NUM_PIECES - 1):
        oriented = _random_orientation(_crop_mask(masks[pi]))
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

    meta = {
        "edge_side": edge_side,
        "seg_len": int(seg_len),
        "grid_size": int(grid_size),
    }
    return grid, meta


def _gen_attempt(return_meta: bool = False):
    side = random.choice(SIDE_CHOICES)
    arm, extents = SHAPE_PARAMS[side]
    extent = random.choice(extents)

    masks = _build_masks(side, arm, extent)
    areas = [int(m.sum()) for m in masks]
    if not all(a > 0 for a in areas):
        return None
    if not _forces_unique_square(masks, side):
        return None

    colors = random.sample(range(1, 10), NUM_PIECES)
    natural_map = np.array([[colors[0], colors[1]], [colors[2], colors[3]]], dtype=int)
    legend_map = np.array([[colors[0], colors[2]], [colors[1], colors[3]]], dtype=int)

    expected = _expected_output(masks, legend_map)
    natural = _natural_output(masks, colors)

    scatter_res = _scatter_pieces(masks, colors, legend_map, side)
    if scatter_res is None:
        return None
    grid, scatter_meta = scatter_res
    out = main(grid)
    if not np.array_equal(out, expected):
        return None

    if return_meta:
        meta = {
            "side": int(side),
            "arm": int(arm),
            "extent": int(extent),
            "areas": tuple(int(a) for a in areas),
            "body_colors": tuple(int(c) for c in colors),
            "natural_map": natural_map.copy(),
            "legend_map": legend_map.copy(),
            "swap_changed": bool(legend_map[0, 1] != natural_map[0, 1]),
            "swap_pixel_diff": int(np.count_nonzero(expected != natural)),
            "swap_piece_area": int(areas[1] + areas[2]),
            **scatter_meta,
        }
        return grid, meta
    return grid


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(return_meta=False)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo f560132c input")
