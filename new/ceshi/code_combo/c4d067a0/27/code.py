# Combo variant of c4d067a0 (index 27)
# combo_spec: D2/2+D8/0+D10/1
# Name: 多图例+迭代放大+多数已填少数缺失

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The original task has one singleton-pixel legend and one block field. This
#    combo has two singleton legends in the left strip. Each top legend first
#    maps to a medium-size block region where almost every block is already
#    present and only 1-2 are absent. After those few medium blocks are filled,
#    each completed medium region itself becomes a new legend for a second
#    scaling round, which maps to a larger block region where the same local
#    positions are missing again.
#    Small trace:
#      - Round 0 source: sparse singleton legend A / legend B.
#      - Round 1 target: medium blocks A / medium blocks B, with only a small
#        number of holes. Fill those holes.
#      - Round 2 source: the now-complete medium blocks A / B.
#      - Round 2 target: larger blocks A / B, again with only a small number
#        of holes. Fill those holes.
#      - Difference from the original: the block output of one round is not the
#        end state; it is the next round's source pattern.
# b) Complexity calibration:
#    The variant uses exactly two legend chains and exactly two block scales.
#    That is enough to make the fusion visible without introducing too many
#    free variables. Each chain reuses one palette across both rounds, and each
#    target keeps most blocks already present so the intended "spot the missing
#    few" effect is clear to a human viewer.
# c) Hidden-assumption audit against the original solver:
#      - "all singleton legend pixels belong to one sprite" is false now.
#        Fix: normalize the singleton lattice and split it into coarse
#        connected components.
#      - "there is one global block size" is false now.
#        Fix: solve block sizes in ascending order, one round per size.
#      - "filling blocks ends the task" is false now.
#        Fix: after painting one size, regroup those completed blocks as the
#        next round's legends.
#      - "all missing targets can be inferred directly from the input" is false
#        for round 2. The generator removes the same local cells from the medium
#        source and the large target, so round 2 only becomes possible after
#        round 1 reconstructs those source cells.
#      - "only two anchor blocks exist" is false now.
#        Fix: the solver still uses vote-based offset inference, but now expects
#        the winning base to explain all but at most two cells of each pattern.
# d) Placement legality:
#    Legends are stacked on the left on the same step-2 lattice with coarse
#    blank rows between them. Medium regions share one medium lattice residue;
#    large regions share one larger lattice residue. The generator searches
#    stage bases so same-stage regions do not overlap, and output construction
#    rejects any cross-stage overlap as well.
# e) Visual observability:
#    A solver can see two distinct legend sprites, two partially completed
#    medium regions, and two partially completed larger regions. The second
#    round is visible because some holes in the largest regions align with holes
#    in the medium regions and therefore cannot be inferred until the medium
#    regions are completed.
# f) Visual cleanliness:
#    Legends stay in the left strip while block regions stay to the right.
#    Different stages use different block sizes, so the medium and large rounds
#    are visually distinguishable even when they share colors.
# g) Change inferability:
#    Colors are not remapped between rounds. Each chain keeps a small disjoint
#    palette across its legend, medium region, and large region, so humans can
#    track correspondence structurally rather than by hidden ordering. The
#    generator validates every sample with `main(input) == output`.


def _split_objects(
    g: np.ndarray, bg: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
    for pts in components_mask(g != bg, connectivity=4):
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))
    return singles, blocks


def _infer_singleton_legends(
    singles: list[tuple[int, int, int]]
) -> tuple[list[list[tuple[int, int, int]]], int]:
    rs = sorted({r for r, _, _ in singles})
    cs = sorted({c for _, c, _ in singles})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    step = reduce(gcd, drs + dcs)
    assert step > 0, "legend lattice step must be positive"

    rmin = min(r for r, _, _ in singles)
    cmin = min(c for _, c, _ in singles)
    coarse = [((r - rmin) // step, (c - cmin) // step, v) for r, c, v in singles]
    max_r = max(rr for rr, _, _ in coarse)
    max_c = max(cc for _, cc, _ in coarse)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    color_at: dict[tuple[int, int], int] = {}
    for rr, cc, v in coarse:
        mask[rr, cc] = True
        color_at[(rr, cc)] = int(v)

    legends: list[tuple[tuple[int, int], list[tuple[int, int, int]]]] = []
    for comp in components_mask(mask, connectivity=4):
        rr0, cc0, _rr1, _cc1 = bbox_coords(comp)
        pts: list[tuple[int, int, int]] = []
        for rr, cc in comp:
            pts.append((int(rr - rr0), int(cc - cc0), color_at[(int(rr), int(cc))]))
        pts.sort()
        legends.append(((int(rr0), int(cc0)), pts))

    legends.sort(key=lambda item: (item[0][0], item[0][1], len(item[1])))
    return [pts for _origin, pts in legends], step


def _infer_block_step(blocks: list[tuple[int, int, int, int]]) -> int:
    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    assert step_candidates, "block step needs at least two aligned blocks"
    step = reduce(gcd, step_candidates)
    assert step > 0, "block step must be positive"
    return step


def _group_block_regions(
    blocks: list[tuple[int, int, int, int]], step: int
) -> list[list[tuple[int, int, int]]]:
    rmin = min(r0 for r0, _, _, _ in blocks)
    cmin = min(c0 for _, c0, _, _ in blocks)
    coarse_entries = []
    for r0, c0, _size, col in blocks:
        assert (r0 - rmin) % step == 0 and (c0 - cmin) % step == 0, (
            "all same-size blocks must share one lattice residue"
        )
        coarse_entries.append(((r0 - rmin) // step, (c0 - cmin) // step, int(col)))

    max_r = max(rr for rr, _, _ in coarse_entries)
    max_c = max(cc for _, cc, _ in coarse_entries)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    color_at: dict[tuple[int, int], int] = {}
    for rr, cc, col in coarse_entries:
        mask[rr, cc] = True
        color_at[(rr, cc)] = int(col)

    regions: list[tuple[tuple[int, int], list[tuple[int, int, int]]]] = []
    for comp in components_mask(mask, connectivity=4):
        rr0, cc0, _rr1, _cc1 = bbox_coords(comp)
        pts: list[tuple[int, int, int]] = []
        for rr, cc in comp:
            pts.append((int(rr - rr0), int(cc - cc0), color_at[(int(rr), int(cc))]))
        pts.sort()
        regions.append(((int(rr0), int(cc0)), pts))

    regions.sort(key=lambda item: (item[0][0], item[0][1], len(item[1])))
    return [pts for _origin, pts in regions]


def _vote_base(
    legend_pts: list[tuple[int, int, int]],
    by_color: dict[int, list[tuple[int, int]]],
    step: int,
) -> tuple[int, int]:
    bases: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        for br0, bc0 in by_color.get(col, []):
            base = (int(br0 - rr * step), int(bc0 - cc * step))
            bases[base] = bases.get(base, 0) + 1
    assert bases, "existing blocks must determine each round offset"
    max_votes = max(bases.values())
    assert max_votes >= len(legend_pts) - 2, "too many targets are missing for this combo"
    best = [base for base, votes in bases.items() if votes == max_votes]
    assert len(best) == 1, "each legend must map to one unique offset"
    return best[0]


def _paint_pattern(
    out: np.ndarray,
    legend_pts: list[tuple[int, int, int]],
    base: tuple[int, int],
    step: int,
    block_size: int,
) -> None:
    h, w = out.shape
    base_r, base_c = base
    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, (
            "legend block projection must fit in canvas"
        )
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete several sparse-to-block mappings across two iterative scaling rounds.

    Input structure:
    - bg = most common color.
    - Left strip: multiple disconnected singleton-pixel legends on one step-2
      lattice.
    - Main area: two block scales. Medium square blocks are the first-round
      targets and also the second-round sources after completion; larger square
      blocks are the final targets. In both scales, most blocks already exist
      and only a small number are missing.

    Rule:
    - Split the singleton pixels into separate legends and use them to complete
      the medium block regions.
    - Regroup those completed medium regions as new legends on their own block
      lattice.
    - Use each completed medium legend to infer the offset of its larger target
      region and fill the remaining large blocks.

    Key insight: The puzzle must be solved by iteration. The first round does
    not merely add missing blocks; it creates the complete source patterns that
    unlock the second round of scaling.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)

    singles, blocks = _split_objects(out, bg)
    assert singles and blocks, "grid must contain legends and block regions"
    round_sources, _legend_step = _infer_singleton_legends(singles)
    assert len(round_sources) >= 2, "combo variant requires multiple legends"

    sizes = sorted({size for _, _, size, _col in blocks})
    assert len(sizes) >= 2, "combo variant requires iterative block sizes"

    for idx, block_size in enumerate(sizes):
        _singles_now, blocks_now = _split_objects(out, bg)
        blocks_this = [b for b in blocks_now if b[2] == block_size]
        assert blocks_this, "each round must have visible target blocks"
        step = _infer_block_step(blocks_this)

        by_color: dict[int, list[tuple[int, int]]] = {}
        for r0, c0, _s, col in blocks_this:
            by_color.setdefault(int(col), []).append((int(r0), int(c0)))

        for legend_pts in round_sources:
            base = _vote_base(legend_pts, by_color, step)
            _paint_pattern(out, legend_pts, base, step, block_size)

        if idx != len(sizes) - 1:
            _singles_next, blocks_next = _split_objects(out, bg)
            next_sources = [b for b in blocks_next if b[2] == block_size]
            round_sources = _group_block_regions(next_sources, step)
            assert len(round_sources) >= 2, "completed medium regions must become new legends"

    return out


# --------------- generator (combo variant) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 32, 36
NUM_COLORS = 10
LEGEND_COUNT = 2

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_ROWS = (3,)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 5, 6
LEGEND_GAP_COARSE = 2
LEGEND_COL_SHIFT_CHOICES = (0, 1)
PALETTE_SIZE_CHOICES = (3, 4)

# === CFG: medium stage ===
MID_BLOCK_SIZE_RANGE = range(2, 4)
MID_GAP_RANGE = range(1, 3)
MID_ORIGIN = (1, 10)
MID_REGION_GAP = 1

# === CFG: large stage ===
BIG_BLOCK_SIZE_RANGE = range(3, 5)
BIG_GAP_RANGE = range(1, 3)
BIG_REGION_GAP = 1

# === CFG: generation ===
MAX_RETRIES = 500


def _enumerate_staircase_shapes():
    shapes = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix, remaining, r):
                    if r == rows:
                        if remaining == 0 and max(prefix) == cols:
                            widths = tuple(prefix)
                            if any(w >= 2 for w in widths):
                                shapes.append(widths)
                        return
                    left = rows - r - 1
                    lo = max(1, remaining - left * cols)
                    hi = min(cols, remaining - left)
                    for width in range(lo, hi + 1):
                        rec(prefix + [width], remaining - width, r + 1)

                rec([], count, 0)
    return shapes


_MID_PAIRS = [(n, g) for n in MID_BLOCK_SIZE_RANGE for g in MID_GAP_RANGE]
_BIG_PAIRS = [(n, g) for n in BIG_BLOCK_SIZE_RANGE for g in BIG_GAP_RANGE]
_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    return pts


def _assign_legend_colors(
    pts: list[tuple[int, int]], palette: list[int]
) -> dict[tuple[int, int], int]:
    order = pts[:]
    random.shuffle(order)
    colors: dict[tuple[int, int], int] = {}
    for pos, col in zip(order, palette):
        colors[pos] = int(col)
    for pos in order[len(palette) :]:
        colors[pos] = int(random.choice(palette))
    return colors


def _place_legends(
    legend_pts_list: list[list[tuple[int, int]]]
) -> list[list[tuple[int, int]]]:
    placed: list[list[tuple[int, int]]] = []
    row_offset = 0
    for pts in legend_pts_list:
        col_offset = random.choice(LEGEND_COL_SHIFT_CHOICES)
        shifted = [(row_offset + r, col_offset + c) for r, c in pts]
        placed.append(shifted)
        max_r = max(r for r, _ in shifted)
        row_offset = max_r + 1 + LEGEND_GAP_COARSE
    return placed


def _bbox_from_pts(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _shift_pts(pts: list[tuple[int, int]], base: tuple[int, int]) -> set[tuple[int, int]]:
    br, bc = base
    return {(br + r, bc + c) for r, c in pts}


def _bbox_gap_ok(
    base1: tuple[int, int],
    bbox1: tuple[int, int, int, int],
    base2: tuple[int, int],
    bbox2: tuple[int, int, int, int],
    min_gap: int,
) -> bool:
    a_r0, a_c0, a_r1, a_c1 = bbox1
    b_r0, b_c0, b_r1, b_c1 = bbox2
    a_r0 += base1[0]
    a_r1 += base1[0]
    a_c0 += base1[1]
    a_c1 += base1[1]
    b_r0 += base2[0]
    b_r1 += base2[0]
    b_c0 += base2[1]
    b_c1 += base2[1]
    row_gap = max(b_r0 - a_r1 - 1, a_r0 - b_r1 - 1, 0)
    col_gap = max(b_c0 - a_c1 - 1, a_c0 - b_c1 - 1, 0)
    return row_gap >= min_gap or col_gap >= min_gap


def _find_stage_bases(
    size: int,
    origin: tuple[int, int],
    block_size: int,
    step: int,
    shape_pts_list: list[list[tuple[int, int]]],
    min_gap: int,
) -> list[tuple[int, int]] | None:
    origin_r, origin_c = origin
    bboxes = [_bbox_from_pts(pts) for pts in shape_pts_list]
    candidate_lists: list[list[tuple[int, int]]] = []
    for bbox in bboxes:
        _r0, _c0, r1, c1 = bbox
        max_br = (size - block_size - origin_r - r1 * step) // step
        max_bc = (size - block_size - origin_c - c1 * step) // step
        if max_br < 0 or max_bc < 0:
            return None
        cands = [(br, bc) for br in range(max_br + 1) for bc in range(max_bc + 1)]
        random.shuffle(cands)
        candidate_lists.append(cands)

    valid_pairs = []
    for base1 in candidate_lists[0]:
        cells1 = _shift_pts(shape_pts_list[0], base1)
        for base2 in candidate_lists[1]:
            cells2 = _shift_pts(shape_pts_list[1], base2)
            if cells1 & cells2:
                continue
            if not _bbox_gap_ok(base1, bboxes[0], base2, bboxes[1], min_gap):
                continue
            valid_pairs.append((base1, base2))
    if not valid_pairs:
        return None

    chosen = random.choice(valid_pairs)
    return [
        (origin_r + br * step, origin_c + bc * step)
        for br, bc in chosen
    ]


def _paint_block_checked(
    grid: np.ndarray,
    occ: np.ndarray,
    r0: int,
    c0: int,
    block_size: int,
    color: int,
) -> bool:
    if occ[r0 : r0 + block_size, c0 : c0 + block_size].any():
        return False
    grid[r0 : r0 + block_size, c0 : c0 + block_size] = int(color)
    occ[r0 : r0 + block_size, c0 : c0 + block_size] = True
    return True


def _clear_block(grid: np.ndarray, r0: int, c0: int, block_size: int, bg: int) -> None:
    grid[r0 : r0 + block_size, c0 : c0 + block_size] = int(bg)


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    bg = random.randrange(NUM_COLORS)

    mid_size, mid_gap = random.choice(_MID_PAIRS)
    big_choices = [(n, g) for n, g in _BIG_PAIRS if n > mid_size]
    if not big_choices:
        return None
    big_size, big_gap = random.choice(big_choices)
    mid_step = mid_size + mid_gap
    big_step = big_size + big_gap

    legend_shapes = [_pick_staircase() for _ in range(LEGEND_COUNT)]
    placed_legends = _place_legends(legend_shapes)
    max_legend_r = max(r for pts in placed_legends for r, _ in pts)
    max_legend_c = max(c for pts in placed_legends for _, c in pts)
    if LEGEND_ORIGIN[0] + max_legend_r * LEGEND_STEP >= size:
        return None
    if LEGEND_ORIGIN[1] + max_legend_c * LEGEND_STEP >= MID_ORIGIN[1] - 2:
        return None

    available = [c for c in range(NUM_COLORS) if c != bg]
    palette_sizes = [
        random.randint(3, min(max(PALETTE_SIZE_CHOICES), len(pts)))
        for pts in legend_shapes
    ]
    if sum(palette_sizes) > len(available):
        return None
    random.shuffle(available)

    legend_colors_list: list[dict[tuple[int, int], int]] = []
    at = 0
    for pts, k in zip(legend_shapes, palette_sizes):
        palette = available[at : at + k]
        at += k
        legend_colors_list.append(_assign_legend_colors(pts, palette))

    mid_bases = _find_stage_bases(size, MID_ORIGIN, mid_size, mid_step, legend_shapes, MID_REGION_GAP)
    if mid_bases is None:
        return None

    big_origin = (size // 2, 10)
    big_bases = _find_stage_bases(size, big_origin, big_size, big_step, legend_shapes, BIG_REGION_GAP)
    if big_bases is None:
        return None

    out = np.full((size, size), bg, dtype=int)
    occ = np.zeros((size, size), dtype=bool)

    for placed_pts, colors in zip(placed_legends, legend_colors_list):
        local_r0 = min(r for r, _ in placed_pts)
        local_c0 = min(c for _, c in placed_pts)
        for rr, cc in placed_pts:
            pr = LEGEND_ORIGIN[0] + rr * LEGEND_STEP
            pc = LEGEND_ORIGIN[1] + cc * LEGEND_STEP
            if occ[pr, pc]:
                return None
            out[pr, pc] = colors[(rr - local_r0, cc - local_c0)]
            occ[pr, pc] = True

    for pts, colors, base in zip(legend_shapes, legend_colors_list, mid_bases):
        base_r, base_c = base
        for rr, cc in pts:
            col = colors[(rr, cc)]
            br = base_r + rr * mid_step
            bc = base_c + cc * mid_step
            if not _paint_block_checked(out, occ, br, bc, mid_size, col):
                return None

    for pts, colors, base in zip(legend_shapes, legend_colors_list, big_bases):
        base_r, base_c = base
        for rr, cc in pts:
            col = colors[(rr, cc)]
            br = base_r + rr * big_step
            bc = base_c + cc * big_step
            if not _paint_block_checked(out, occ, br, bc, big_size, col):
                return None

    inp = out.copy()
    missing_sets = []
    for pts in legend_shapes:
        max_missing = 2 if len(pts) >= 6 else 1
        miss_n = random.randint(1, max_missing)
        missing_sets.append(set(random.sample(pts, miss_n)))

    for pts, miss, mid_base, big_base in zip(legend_shapes, missing_sets, mid_bases, big_bases):
        mid_r, mid_c = mid_base
        big_r, big_c = big_base
        for rr, cc in pts:
            if (rr, cc) not in miss:
                continue
            _clear_block(inp, mid_r + rr * mid_step, mid_c + cc * mid_step, mid_size, bg)
            _clear_block(inp, big_r + rr * big_step, big_c + cc * big_step, big_size, bg)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    meta = {
        "legend_count": LEGEND_COUNT,
        "size_levels": [mid_size, big_size],
        "missing_per_chain": [len(miss) for miss in missing_sets],
        "mid_step": mid_step,
        "big_step": big_step,
    }
    return inp, out, meta


def generate_input():
    for _ in range(MAX_RETRIES):
        built = _gen_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid c4d067a0 combo variant input")
