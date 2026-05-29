# Combo variant of f560132c (index 17)
# combo_spec: D10/2+D14/1
# Name: 框架颜色编码位置+颜色映射表图例

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location

# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior trace
#    - Original task: after exact-cover assembly, each piece was recolored only by assembled position; the legend
#      was just a 2x2 target palette. A top-right output piece took the top-right legend color, regardless of the
#      piece's input body color.
#    - D14/1 changed the legend core into a source->target lookup table. That removed positional coloring and made
#      recoloring depend on piece body color instead.
#    - This fusion keeps positional targets but embeds source->target evidence inside the same positional slots.
#      The legend annotation is a 4x6 framed patch. Each quadrant slot contains two horizontal pixels:
#        [source_color, target_color]
#      and the surrounding frame segment touching that slot is also painted with the same target_color.
#      Example: if the assembled bottom-left piece is the green sprite and its output target is yellow, then the
#      bottom-left slot shows [green, yellow] and the adjacent bottom-left frame segment is also yellow.
#      After assembly, the solver can infer yellow either from the frame position or from green->yellow in the slot;
#      both must agree.
# b) Complexity calibration
#    - Geometry stays identical to the base task: four simply-connected pieces tile one unique square under D4.
#    - The legend change is visually strong: the old plain 2x2 palette becomes a framed 4x6 annotation with both
#      body-color references and target-color echoes.
#    - Degrees of freedom are still controlled: exactly four source colors, four distinct target colors, and one
#      slot per output quadrant. This keeps the symbolic burden visible but small.
# c) Hidden assumption review against the original main()
#    - Original `_extract_legend_map(...)` assumed one pixel per logical color cell. That fails here because each
#      logical slot is now a 1x2 pair and border cells also carry target information; extraction must scan for the
#      full framed 4x6 motif.
#    - Original `_assemble_exact_cover(...)` assumed legend position alone determined output color. In this combo,
#      that still holds as one route, but a second route exists via source->target decoding. The solver therefore
#      computes both and asserts equality instead of trusting only one cue.
#    - Original legend body-color recovery relied on the legend body dominating the color histogram. That is unsafe
#      once the frame repeats target colors many times, so this variant extracts the legend body from component cells
#      outside the decoded annotation window.
#    - Piece extraction, hole filling, orientation enumeration, and exact-cover search remain valid because they only
#      depend on occupied cells, not on the internal legend colors.
# d) Placement legality
#    - Generator first builds the assembled square, then scatters cropped pieces with border/padding using
#      `random_free_location`, so components stay separated and in-bounds.
#    - The legend mask is filtered to contain a solid 4x6 rectangle for the annotation and at least a few extra body
#      cells outside it, guaranteeing the framed patch fits without leaking outside the sprite.
# e) Visual observability
#    - Across examples, the same target color appears twice in each clue slot: once as the slot's right pixel and
#      again on the adjacent frame segment. The source color appears inside the same slot and also as one whole input
#      sprite. That repeated support makes the mapping human-visible from a few samples.
#    - Because the slot positions match assembled quadrants, a human can notice both "position gives this target"
#      and "this sprite color maps to this target" at the same time.
# f) Visual cleanliness
#    - The annotation is a single compact framed patch; there are no overlapping meta-markers elsewhere in the grid.
#      Non-legend sprites stay monochrome, so the legend remains the only multicolor object.
# g) Color-change inferability
#    - Source colors are exactly the four piece body colors. Target colors are four distinct colors disjoint from the
#      sources. No numeric ordering or hidden tie-breaks are used.
# h) New solver assumptions and generator guarantees
#    - Assumption: the legend contains one valid 4x6 framed annotation whose four slots follow `[src, tgt]`.
#      Guarantee: generator embeds exactly one such patch in the legend piece.
#    - Assumption: each output quadrant corresponds to exactly one piece centroid bucket.
#      Guarantee: generator rejects tilings whose four centroids are not all in different 2x2 cells.
#    - Assumption: the source color recorded in each legend slot matches the body color of the piece that lands in
#      that assembled quadrant.
#      Guarantee: generator writes slot sources from the assembled pieces' actual centroid cells.

K = 2
NUM_PIECES = K * K
ANNOT_H = K + 2
ANNOT_W = 2 * K + 2
ANNOT_AREA = ANNOT_H * ANNOT_W

INNER_SRC_OFFSETS = {
    (0, 0): (1, 1),
    (0, 1): (1, 3),
    (1, 0): (2, 1),
    (1, 1): (2, 3),
}
INNER_TGT_OFFSETS = {
    (0, 0): (1, 2),
    (0, 1): (1, 4),
    (1, 0): (2, 2),
    (1, 1): (2, 4),
}
FRAME_OFFSETS = {
    (0, 0): ((0, 0), (0, 1), (0, 2), (1, 0)),
    (0, 1): ((0, 3), (0, 4), (0, 5), (1, 5)),
    (1, 0): ((2, 0), (3, 0), (3, 1), (3, 2)),
    (1, 1): ((2, 5), (3, 3), (3, 4), (3, 5)),
}


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


def _extract_legend_annotation(
    g: np.ndarray,
    pts: np.ndarray,
    k: int,
) -> tuple[dict[int, int], np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
    Find a framed annotation window of shape (k+2) x (2k+2).
    Each logical slot stores `[src, tgt]`, and the adjacent frame cells must also equal `tgt`.
    Returns:
      - mapping: source -> target
      - slot_sources: k x k source-color array
      - slot_targets: k x k target-color array
      - abs_rect: absolute bbox (r0, c0, r1, c1) of the chosen annotation window
    """
    assert k == K, 'this combo variant is defined for k=2'

    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert hh >= ANNOT_H and ww >= ANNOT_W, 'legend component must fit the framed annotation'

    best_key = None
    best = None
    for rr in range(hh - ANNOT_H + 1):
        for cc in range(ww - ANNOT_W + 1):
            window = sub[rr : rr + ANNOT_H, cc : cc + ANNOT_W]
            slot_sources = np.zeros((k, k), dtype=int)
            slot_targets = np.zeros((k, k), dtype=int)
            valid_pairs = 0
            valid_frames = 0

            for qr in range(k):
                for qc in range(k):
                    sr, sc = INNER_SRC_OFFSETS[(qr, qc)]
                    tr, tc = INNER_TGT_OFFSETS[(qr, qc)]
                    src = int(window[sr, sc])
                    tgt = int(window[tr, tc])
                    slot_sources[qr, qc] = src
                    slot_targets[qr, qc] = tgt
                    if src != tgt:
                        valid_pairs += 1
                    frame_vals = [int(window[fr, fc]) for fr, fc in FRAME_OFFSETS[(qr, qc)]]
                    if all(v == tgt for v in frame_vals):
                        valid_frames += 1

            sources = set(map(int, slot_sources.ravel()))
            targets = set(map(int, slot_targets.ravel()))
            key = (
                valid_pairs,
                valid_frames,
                int(len(sources)),
                int(len(targets)),
                -rr,
                -cc,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (rr, cc, slot_sources.copy(), slot_targets.copy())

    assert best is not None and best_key is not None, 'legend must contain a framed annotation'
    assert best_key[:4] == (k * k, k * k, k * k, k * k), 'legend annotation must decode four distinct source->target slots'

    rr, cc, slot_sources, slot_targets = best
    mapping: dict[int, int] = {}
    for qr in range(k):
        for qc in range(k):
            src = int(slot_sources[qr, qc])
            tgt = int(slot_targets[qr, qc])
            mapping[src] = tgt

    assert len(mapping) == k * k, 'mapping must contain one source per piece color'
    assert len(set(mapping.values())) == k * k, 'targets must be pairwise distinct'
    abs_rect = (r0 + rr, c0 + cc, r0 + rr + ANNOT_H - 1, c0 + cc + ANNOT_W - 1)
    return mapping, slot_sources, slot_targets, abs_rect


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


def _piece_body_color(
    g: np.ndarray,
    pts: np.ndarray,
    bg: int,
    exclude_rect: tuple[int, int, int, int] | None = None,
) -> int:
    counts: Counter[int] = Counter()
    if exclude_rect is None:
        for r, c in pts:
            col = int(g[r, c])
            if col != bg:
                counts[col] += 1
    else:
        er0, ec0, er1, ec1 = exclude_rect
        for r, c in pts:
            if er0 <= int(r) <= er1 and ec0 <= int(c) <= ec1:
                continue
            col = int(g[r, c])
            if col != bg:
                counts[col] += 1
        assert counts, 'legend must leave body-colored cells outside the annotation'
    assert counts, 'piece must contain at least one non-background color'
    return int(counts.most_common(1)[0][0])


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_mapping: dict[int, int],
    slot_sources: np.ndarray,
    slot_targets: np.ndarray,
    k: int,
) -> np.ndarray:
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = _factor_pairs(total)
    dims = [(H, W) for (H, W) in dims if H >= max_h and W >= max_w]
    assert len(dims) == 1, 'piece areas and bounds must imply a unique output rectangle'

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
                for idx2 in sorted(remaining):
                    opts_k = []
                    for pl in filtered[idx2]:
                        m = pl[0]
                        if (m >> cell) & 1 and (m & used) == 0:
                            opts_k.append(pl)
                    if opts_k:
                        candidates.append((len(opts_k), idx2, opts_k))
                if not candidates:
                    return False

                candidates.sort(key=lambda x: (x[0], x[1]))
                for _, idx2, opts_k in candidates:
                    remaining.remove(idx2)
                    pi = other_pis[idx2]
                    for pl in opts_k:
                        chosen[pi] = pl
                        if search(used | pl[0]):
                            return True
                    chosen.pop(pi, None)
                    remaining.add(idx2)
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
            src_from_piece = int(pieces[pi]["body_color"])
            src_from_slot = int(slot_sources[rr, cc2])
            tgt_from_slot = int(slot_targets[rr, cc2])
            tgt_from_map = int(legend_mapping[src_from_piece])

            assert src_from_piece == src_from_slot, 'slot source must match the piece assigned to that output cell'
            assert tgt_from_map == tgt_from_slot, 'frame/slot target and source->target mapping must agree'

            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = tgt_from_slot
        return out

    raise AssertionError('unreachable rectangle assembly state')


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered pieces into one square, then recolor each piece using a legend that redundantly encodes both position and source->target mapping.

    Input structure:
    - bg is the most common color.
    - Four disconnected non-bg sprites are scattered on the canvas; three are monochrome and one is the legend piece.
    - The legend contains a compact 4x6 framed annotation. In each of the four logical quadrants, the inner slot is
      a horizontal two-pixel pair `[source_color, target_color]`, and the adjacent frame segment repeats that same
      target color around the slot.

    Rule:
    - Ignore internal colors while fitting shapes: all four pieces still tile one tight square under rotations and reflections.
    - Decode the legend in two parallel ways:
      1. slot position / frame segment -> target color for that assembled quadrant
      2. source color in the slot -> target color for the piece whose body has that color
    - Recolor each assembled piece only when those two routes agree for the piece that lands in that quadrant.

    Key insight: The legend is both a positional frame code and a source->target table. The correct output is the
    unique assembly where each piece's body color matches the source recorded in its quadrant slot, and the slot's
    target matches the frame color surrounding that slot.
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
    legend_mapping, slot_sources, slot_targets, legend_rect = _extract_legend_annotation(g, comps[legend_idx], k)

    pieces: list[dict] = []
    body_colors = []
    for i, pts in enumerate(comps):
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        exclude_rect = legend_rect if i == legend_idx else None
        body = _piece_body_color(g, pts, bg, exclude_rect=exclude_rect)
        body_colors.append(body)
        pieces.append({"oris": oris, "area": len(filled_abs), "body_color": body})

    assert set(body_colors) == set(legend_mapping.keys()), 'legend sources must match the four piece colors'
    return _assemble_exact_cover(pieces, legend_idx, legend_mapping, slot_sources, slot_targets, k)


# --------------- generator (combo) ---------------

SIDE_CHOICES = [8, 10, 12]
NUM_COLORS = 8
AREA_FRAC_LO = 0.08
AREA_FRAC_HI = 0.43
AREA_ABS_LO = 11
AREA_ABS_HI = 62
LEGEND_THRESHOLD = {8: 4, 10: 5, 12: 9}
LEGEND_ARM_RANGE = range(1, 5)
LEGEND_REQUIRED_AREA = ANNOT_AREA + 3
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


def _area_bounds(side):
    total = side * side
    return max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total)), min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))


def _has_hole(mask):
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


def _forces_unique_square(masks, side):
    bbs = [bbox_coords(np.argwhere(m)) for m in masks]
    max_h = max(r1 - r0 + 1 for r0, _, r1, _ in bbs)
    max_w = max(c1 - c0 + 1 for _, c0, _, c1 in bbs)
    dims = [(h, w) for h, w in _factor_pairs(side * side) if h >= max_h and w >= max_w]
    return dims == [(side, side)]


def _annotation_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(0, h - ANNOT_H + 1):
        for c in range(0, w - ANNOT_W + 1):
            if not mask[r : r + ANNOT_H, c : c + ANNOT_W].all():
                continue
            if int(mask.sum()) <= ANNOT_AREA:
                continue
            spots.append((r, c))
    return spots


def _make_legend_mask(side, min_area, max_area):
    threshold = LEGEND_THRESHOLD[side]
    min_area = max(min_area, LEGEND_REQUIRED_AREA)
    opts = []
    for h0 in range(threshold + 1, side + 1):
        for w0 in range(threshold + 1, side + 1):
            for arm_w in LEGEND_ARM_RANGE:
                for arm_h in LEGEND_ARM_RANGE:
                    mask = np.zeros((side, side), dtype=bool)
                    mask[:arm_h, :w0] = True
                    mask[:h0, :arm_w] = True
                    area = int(mask.sum())
                    if not (min_area <= area <= max_area):
                        continue
                    if _annotation_spots(mask):
                        opts.append(mask.copy())
    return random.choice(opts).copy() if opts else None


def _partition_rest(rest):
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

    def add_frontier(i, r, c):
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


def _crop_mask(mask):
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_orientation(mask):
    arr = mask.astype(int)
    opts = []
    seen = set()
    for k2 in range(4):
        rot = np.rot90(arr, k2)
        for oriented in (rot, np.fliplr(rot)):
            r0, c0, r1, c1 = bbox_coords(np.argwhere(oriented > 0))
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop)
    return random.choice(opts)


def _build_annotation_patch(slot_sources: np.ndarray, slot_targets: np.ndarray) -> np.ndarray:
    patch = np.zeros((ANNOT_H, ANNOT_W), dtype=int)
    for qr in range(K):
        for qc in range(K):
            src = int(slot_sources[qr, qc])
            tgt = int(slot_targets[qr, qc])
            sr, sc = INNER_SRC_OFFSETS[(qr, qc)]
            tr, tc = INNER_TGT_OFFSETS[(qr, qc)]
            patch[sr, sc] = src
            patch[tr, tc] = tgt
            for fr, fc in FRAME_OFFSETS[(qr, qc)]:
                patch[fr, fc] = tgt
    return patch


def _scatter_pieces(masks, body_colors, annotation_patch, side):
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _crop_mask(masks[pi]) if pi == 0 else _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        if pi == 0:
            spots = _annotation_spots(oriented)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r : r + ANNOT_H, c : c + ANNOT_W] = annotation_patch
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


def _variant_metrics(qcells, slot_sources, body_colors, slot_targets, original_out, combo_out):
    source_matches = 0
    for pi in range(NUM_PIECES):
        rr, cc = qcells[pi]
        source_matches += int(int(slot_sources[rr, cc]) == int(body_colors[pi]))
    target_duplicates = len(set(map(int, slot_targets.ravel())))
    return {
        'source_matches': source_matches,
        'distinct_targets': target_duplicates,
        'differs_from_original_logic': int(not np.array_equal(original_out, combo_out)),
    }


def _gen_attempt():
    side = random.choice(SIDE_CHOICES)
    min_area, max_area = _area_bounds(side)
    legend = _make_legend_mask(side, min_area, max_area)
    if legend is None:
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
    target_colors = colors[NUM_PIECES:]
    random.shuffle(target_colors)
    slot_targets = np.array(target_colors, dtype=int).reshape(K, K)

    slot_sources = np.zeros((K, K), dtype=int)
    for pi in range(NUM_PIECES):
        rr, cc = qcells[pi]
        slot_sources[rr, cc] = int(body_colors[pi])
    if len(set(map(int, slot_sources.ravel()))) != NUM_PIECES:
        return None

    annotation_patch = _build_annotation_patch(slot_sources, slot_targets)

    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        expected[mask] = int(slot_targets[rr, cc])

    grid = _scatter_pieces(masks, body_colors, annotation_patch, side)
    if grid is None:
        return None
    try:
        combo_out = main(grid)
    except Exception:
        return None
    if not np.array_equal(combo_out, expected):
        return None

    # Compare against the original spatial-palette logic to catch accidental degeneration.
    from arc2_solver_select.f560132c import main as original_main

    try:
        original_out = original_main(grid)
    except Exception:
        original_out = np.full_like(expected, -1)
    metrics = _variant_metrics(qcells, slot_sources, body_colors, slot_targets, original_out, combo_out)
    if metrics['source_matches'] != NUM_PIECES:
        return None
    if metrics['distinct_targets'] != NUM_PIECES:
        return None
    if metrics['differs_from_original_logic'] != 1:
        return None
    return grid


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate f560132c combo input')
