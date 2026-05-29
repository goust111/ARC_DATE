# Combo variant of 8b7bacbf (index 12)
# combo_spec: D4/3+D19/0
# Name: 模板种子+颜色映射图例
#
# Design analysis for the fused variant
# a) Concrete behavior and execution trace
#    - Original task: a unique border pixel identifies one line component, and every cavity attached to that
#      component is filled with the seed's own color.
#    - D4 changes the seed object: the seed is now a border-touching 2x2 monochrome block, and the line starts
#      only through diagonal 8-connectivity from one inward-facing corner of that block.
#    - D19 changes the write-out color: the border seed color is no longer the final fill color; instead a
#      top-left legend explicitly maps seed_color -> fill_color.
#    - Fused trace:
#      1. Read the 4x5 legend in the top-left corner. Each usable row is [seed_color, bg, fill_color].
#      2. Outside the legend, find border-touching 2x2 monochrome blocks. Their colors are the legend lookup keys.
#      3. For each block, inspect its 8-neighborhood to recover the diagonally attached line color.
#      4. Trace the seed block's whole non-background 8-connected component.
#      5. Fill every single-color enclosed cavity touched by that traced component, but write the mapped fill color
#         from the legend rather than the block's own color.
#    - Concrete miniature example:
#      Original: border pixel color 7 touches a line of color 3, so cavities on that line become 7.
#      Fused: a 2x2 border block of color 7 sits outside the legend; the legend row [7, bg, 4] says "fill with 4";
#      the first line pixel of color 3 is only diagonal to the block; the same attached cavities now become 4.
#      The differences are therefore visible in both seed detection and final color choice.
# b) Complexity calibration
#    - The seed template remains a simple repeated object: always a monochrome 2x2 block touching one border.
#    - The legend is fixed-size and fixed-position, with exactly two rows, which matches the 1-2 seed regime while
#      still repeating the mapping structure enough for a human to read it as a table instead of noise.
#    - Fill colors are forced to differ from seed colors, so the D19 signal is always observable.
#    - Frame colors are kept to a single pool color to avoid clutter now that the puzzle also contains a legend and
#      duplicated seed colors.
# c) Hidden assumptions audit against the original main()
#    - Original `_seed_pixels()` assumed a seed color appears exactly once globally. That is invalid twice over here:
#      the seed is a 2x2 block, and its color also reappears inside the legend. The fused solver therefore detects
#      2x2 blocks using counts only outside the legend mask.
#    - Original line-color inference used 4-neighbors of the seed pixel. That fails because the line is diagonal-only
#      relative to the 2x2 block. The fused solver infers line color from the block's 8-neighborhood.
#    - Original `out[r, c] = seed_col` assumed seed color equals fill color. That fails because the legend remaps it.
#      The fused solver uses `legend_map[seed_col]`.
#    - Original outside-background detection relied on any background corner. The top-left corner is now occupied by
#      the legend; generator placement therefore keeps at least one of the other corners as background.
# d) Placement legality
#    - The legend occupies the top-left 4x5 box and reserves a one-cell moat on its right and bottom sides so no line,
#      frame, or seed block can visually merge with it.
#    - Seed blocks are sampled on border sides with ranges that avoid the legend footprint and the corners.
#    - Each seed's first line pixel is chosen from diagonal-only options and each seed block imposes a forbidden
#      4-neighbor halo, preventing accidental orthogonal contact later in the grown line.
#    - Different seed components are kept non-overlapping and non-touching at placement time so two seeds never start
#      as one merged 8-connected component.
# e) Visual observability
#    - Humans can see two independent cues: a border 2x2 template instead of a lone border dot, and a legend whose
#      left cell color matches that template while the output cavity color matches the right cell.
#    - Across examples, the seed block location, the line path, and the legend row ordering vary, so the rule cannot
#      be solved by memorizing one corner or one fixed color identity.
# f) Visual cleanliness
#    - The legend is visually isolated in one corner with a moat.
#    - Seed blocks stay on borders, lines remain one-pixel wide, and frames keep simple rounded-rectangle rings.
#      This preserves the original task's readability despite the extra symbolic object.
# g) Change inferability
#    - The mapping relation is explicit and low-arity: two legend rows, each with a left/right color pair separated
#      by background. That gives a human enough repeated structure to infer "left color names the seed block, right
#      color names the fill" from a small number of demonstrations.
# 9) New assumptions introduced by the fused solver and generator guarantees
#    - Assumption: true seeds are the only legend-external colors that form border-touching 2x2 monochrome blocks.
#      Guarantee: generator reserves unique seed colors, keeps frames away from borders, and validates detected blocks.
#    - Assumption: each true seed block has at least one non-background 8-neighbor that belongs to its own line.
#      Guarantee: generator explicitly places the first line pixel on one of the two legal diagonal-only positions.
#    - Assumption: the legend rows are well-formed and use distinct colors.
#      Guarantee: generator draws a fixed border box and samples all legend entries from a unique palette.
#    - Assumption: the mapped fill color affects the actual output rather than coincidentally matching the seed color.
#      Guarantee: generator forces `fill_color != seed_color` and later rejects any sample whose mapped output matches
#      the seed-color baseline.

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


LEGEND_H = 4
LEGEND_W = 5
LEGEND_ROWS = 2
LEGEND_PAD = 1
RESERVED = -2


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _legend_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:LEGEND_H, :LEGEND_W] = True
    return mask


def _legend_border_coords() -> list[tuple[int, int]]:
    coords = []
    for r in range(LEGEND_H):
        for c in range(LEGEND_W):
            if r in (0, LEGEND_H - 1) or c in (0, LEGEND_W - 1):
                coords.append((r, c))
    return coords


def _extract_legend(g: np.ndarray, bg: int) -> dict[int, int]:
    assert g.shape[0] >= LEGEND_H and g.shape[1] >= LEGEND_W
    box = g[:LEGEND_H, :LEGEND_W]
    legend_col = int(box[0, 0])
    assert legend_col != bg
    for r, c in _legend_border_coords():
        assert int(box[r, c]) == legend_col

    mapping: dict[int, int] = {}
    for row in range(1, 1 + LEGEND_ROWS):
        left = int(box[row, 1])
        mid = int(box[row, 2])
        right = int(box[row, 3])
        assert left != bg and right != bg
        assert left != legend_col and right != legend_col
        assert mid == bg
        mapping[left] = right
    assert len(mapping) == LEGEND_ROWS
    return mapping


def _seed_blocks(g: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    outside = ~legend_mask
    vals, cnts = np.unique(g[outside], return_counts=True)
    blocks: list[tuple[int, int, list[tuple[int, int]]]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 4:
            continue
        pts = [tuple(map(int, rc)) for rc in np.argwhere((g == v) & outside)]
        rs = sorted({r for r, _ in pts})
        cs = sorted({c for _, c in pts})
        if len(rs) != 2 or len(cs) != 2:
            continue
        expected = {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}
        if set(pts) != expected:
            continue
        if not _touches_border(pts, h, w):
            continue
        blocks.append((rs[0], cs[0], sorted(pts)))
    blocks.sort(key=lambda x: (x[0], x[1]))
    return [pts for _, _, pts in blocks]


def _seed_line_color(g: np.ndarray, seed_pts: list[tuple[int, int]], bg: int) -> int:
    h, w = g.shape
    seed_set = set(seed_pts)
    seed_col = int(g[seed_pts[0]])
    neigh_cols: list[int] = []
    for r, c in seed_pts:
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seed_set:
                v = int(g[nr, nc])
                if v != bg and v != seed_col:
                    neigh_cols.append(v)
    assert neigh_cols, "seed block must have at least one non-bg 8-neighbor"
    return Counter(neigh_cols).most_common(1)[0][0]


def _is_diag_only_seed_block(g: np.ndarray, seed_pts: list[tuple[int, int]], bg: int) -> bool:
    h, w = g.shape
    seed_set = set(seed_pts)
    seed_col = int(g[seed_pts[0]])
    has_nonseed_8_neighbor = False
    for r, c in seed_pts:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seed_set:
                if int(g[nr, nc]) not in (bg, seed_col):
                    return False
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seed_set:
                if int(g[nr, nc]) not in (bg, seed_col):
                    has_nonseed_8_neighbor = True
    return has_nonseed_8_neighbor


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _find_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    regions: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []
    for pts in bg_comps:
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (reg, _) in enumerate(regions):
        if any(p in reg for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, bnd) in enumerate(regions):
        if i in outside_idxs or not bnd:
            continue
        cavities.append((reg, bnd, {int(g[p]) for p in bnd}))
    return cavities


def _solve_from_seed_blocks(
    g: np.ndarray,
    bg: int,
    legend_mask: np.ndarray,
    fill_lookup: dict[int, int],
) -> np.ndarray:
    h, w = g.shape
    out = g.copy()
    cavities = _find_cavities(g, bg)

    for seed_pts in _seed_blocks(g, bg, legend_mask):
        seed_col = int(g[seed_pts[0]])
        assert seed_col in fill_lookup
        fill_col = int(fill_lookup[seed_col])
        line_col = _seed_line_color(g, seed_pts, bg)
        traced = _nonbg_component8_from(g, seed_pts[0], bg)

        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue

            ok = False
            for r, c in boundary_in_traced:
                for nr, nc in neighbors8(r, c):
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                continue

            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = fill_col

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill frame cavities reached by border seed blocks, using a legend-defined output color.

    Input structure:
    - The most common color is background.
    - A 4x5 legend box sits in the top-left corner; each inner row encodes [seed_color, bg, fill_color].
    - Outside that legend, one or two monochrome 2x2 blocks touch the border and act as seeds.
    - Each seed block is diagonally 8-connected to a thin line that reaches one or more monochrome frame loops.

    Rule:
    - Read the legend to obtain the mapping from seed-block color to fill color.
    - Detect each border-touching 2x2 seed block outside the legend and infer its attached line color from the
      block's 8-neighborhood.
    - Trace the seed block's full non-background 8-connected component.
    - For every enclosed background cavity whose single-color ring is touched by that traced component, flood-fill
      the cavity with the mapped fill color from the legend.

    Key insight:
    The seed block still selects which line-and-frame network is active, but its visible color is only a key into
    the legend; the final cavity color is determined by the legend row, not by the block itself.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_mask = _legend_mask(g.shape)
    legend_map = _extract_legend(g, bg)
    return _solve_from_seed_blocks(g, bg, legend_mask, legend_map)


# --------------- generator (fused D4 seed-template + D19 legend variant) ---------------

GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
SEED_LAYOUT_TRIES = 60
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 1
TOTAL_LOOPS_MAX = 8
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 50
DISTRACTOR_FRAME_TRIES = 40

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_PROB = 0.6
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_SOLO_MAX = 2
DISTRACTOR_NETWORK_FRAMES = (1, 3)

GEN_RETRIES = 400

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = ch + 2, cw + 2
    ring = []
    for dr in range(h):
        for dc in range(w):
            on_border = dr == 0 or dr == h - 1 or dc == 0 or dc == w - 1
            corner = dr in (0, h - 1) and dc in (0, w - 1)
            if on_border and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
    h: int,
    w: int,
) -> bool:
    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return False

    ring_set = set(ring)
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return True
    return False


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
    forbidden: set[tuple[int, int]] | None = None,
    allowed_contact: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    forbidden = set() if forbidden is None else set(forbidden)
    allowed_contact = set() if allowed_contact is None else set(allowed_contact)
    path = [start]
    path_set = {start}
    cur = start
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    while len(path) < target_len:
        dirs = []
        for ddr, ddc in DIAG_DIRS:
            aligned = ddr == dr_bias or ddc == dc_bias
            dirs.extend([(ddr, ddc)] * (DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER))
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            dirs.extend([(ddr, ddc)] * (ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER))
        rng.shuffle(dirs)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1 or (nr, nc) in forbidden:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            blocked_by_existing = False
            for nnr, nnc in neighbors8(nr, nc):
                if not _in_bounds(h, w, nnr, nnc) or (nnr, nnc) == cur:
                    continue
                if (nnr, nnc) in path_set or (nnr, nnc) in allowed_contact:
                    continue
                if grid[nnr, nnc] != -1:
                    blocked_by_existing = True
                    break
            if blocked_by_existing:
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            cur = (nr, nc)
            path.append(cur)
            path_set.add(cur)
            grid[nr, nc] = line_col
            placed = True
            break
        if not placed:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            break
    return path


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
):
    h, w = canvas.shape
    for _ in range(max_tries):
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        lp = rng.choice(line_path)
        fh, fw = ch + 2, cw + 2
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < FRAME_PLACEMENT_PAD or c0 < FRAME_PLACEMENT_PAD or r0 + fh > h - FRAME_PLACEMENT_PAD or c0 + fw > w - FRAME_PLACEMENT_PAD:
                continue
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_distractor_frame(canvas: np.ndarray, frame_col: int, rng: random.Random, max_tries: int = DISTRACTOR_FRAME_TRIES):
    h, w = canvas.shape
    for _ in range(max_tries):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and canvas[rr, cc] != -1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        ring, _ = _frame_ring_coords(r0, c0, ch, cw)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True
    return None


def _try_place_distractor_network(canvas: np.ndarray, line_col: int, frame_colors_pool: list[int], rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(canvas[r + dr, c + dc] != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue
        canvas[r, c] = line_col
        path = _grow_line(canvas, (r, c), line_col, rng.randint(*DISTRACTOR_LINE_LEN), rng)
        if len(path) < 3:
            for pr, pc in path:
                canvas[pr, pc] = -1
            continue
        attached = 0
        for _ in range(rng.randint(*DISTRACTOR_NETWORK_FRAMES)):
            if _try_attach_frame(canvas, path, line_col, rng.choice(frame_colors_pool), rng) is not None:
                attached += 1
        if attached > 0:
            return True
        for pr, pc in path:
            canvas[pr, pc] = -1
    return False


def _seed_block_forbidden(seed_pts: list[tuple[int, int]]) -> set[tuple[int, int]]:
    seed_set = set(seed_pts)
    forbidden = set()
    for r, c in seed_pts:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in seed_set:
                forbidden.add((nr, nc))
    return forbidden


def _sample_seed_block(side: str, h: int, w: int, rng: random.Random):
    if side == "T":
        lo, hi = LEGEND_W + LEGEND_PAD + 1, w - 4
        if lo > hi:
            return None
        c0 = rng.randint(lo, hi)
        seed_pts = [(0, c0), (0, c0 + 1), (1, c0), (1, c0 + 1)]
        first_options = [(2, c0 - 1), (2, c0 + 2)]
    elif side == "B":
        lo, hi = 2, w - 4
        if lo > hi:
            return None
        c0 = rng.randint(lo, hi)
        seed_pts = [(h - 2, c0), (h - 2, c0 + 1), (h - 1, c0), (h - 1, c0 + 1)]
        first_options = [(h - 3, c0 - 1), (h - 3, c0 + 2)]
    elif side == "L":
        lo, hi = LEGEND_H + LEGEND_PAD + 1, h - 4
        if lo > hi:
            return None
        r0 = rng.randint(lo, hi)
        seed_pts = [(r0, 0), (r0, 1), (r0 + 1, 0), (r0 + 1, 1)]
        first_options = [(r0 - 1, 2), (r0 + 2, 2)]
    elif side == "R":
        lo, hi = 2, h - 4
        if lo > hi:
            return None
        r0 = rng.randint(lo, hi)
        seed_pts = [(r0, w - 2), (r0, w - 1), (r0 + 1, w - 2), (r0 + 1, w - 1)]
        first_options = [(r0 - 1, w - 3), (r0 + 2, w - 3)]
    else:
        raise ValueError(side)
    return seed_pts, first_options, _seed_block_forbidden(seed_pts)


def _touching_or_overlapping(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for rr, cc in b:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return True
    return False


def _sample_seed_layout(canvas: np.ndarray, num_seeds: int, rng: random.Random):
    h, w = canvas.shape
    sides = ["T", "B", "L", "R"]
    for _ in range(SEED_LAYOUT_TRIES):
        chosen_sides = rng.sample(sides, num_seeds)
        layout = []
        claimed: set[tuple[int, int]] = set()
        ok = True
        for side in chosen_sides:
            placed = False
            for _ in range(SEED_LAYOUT_TRIES):
                sample = _sample_seed_block(side, h, w, rng)
                if sample is None:
                    continue
                seed_pts, first_options, forbidden = sample
                block_cells = set(seed_pts)
                if any(canvas[r, c] != -1 for r, c in block_cells):
                    continue
                valid_firsts = [
                    p for p in first_options
                    if _in_bounds(h, w, p[0], p[1]) and canvas[p] == -1 and not _touching_or_overlapping({p}, claimed)
                ]
                if not valid_firsts:
                    continue
                candidate_cells = block_cells | {rng.choice(valid_firsts)}
                if _touching_or_overlapping(candidate_cells, claimed):
                    continue
                first = next(iter(candidate_cells - block_cells))
                layout.append((seed_pts, first, forbidden))
                claimed |= candidate_cells
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return layout
    return None


def _distribute_loop_budget(num_seeds: int, num_attached_target: int, rng: random.Random) -> list[int]:
    seed_order = list(range(num_seeds))
    rng.shuffle(seed_order)
    per_seed = [0] * num_seeds
    remaining = num_attached_target
    while remaining > 0 and any(per_seed[i] == 0 for i in range(num_seeds)):
        for i in seed_order:
            if per_seed[i] == 0 and remaining > 0:
                per_seed[i] += 1
                remaining -= 1
    while remaining > 0:
        per_seed[rng.choice(seed_order)] += 1
        remaining -= 1
    return per_seed


def _seed_components_are_separate(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> bool:
    blocks = _seed_blocks(grid, bg, legend_mask)
    traced = [_nonbg_component8_from(grid, block[0], bg) for block in blocks]
    for i in range(len(traced)):
        for j in range(i + 1, len(traced)):
            if traced[i] & traced[j]:
                return False
    return True


def _place_legend(canvas: np.ndarray, legend_col: int, legend_rows: list[tuple[int, int]]) -> None:
    for r in range(LEGEND_H):
        for c in range(LEGEND_W):
            if r in (0, LEGEND_H - 1) or c in (0, LEGEND_W - 1):
                canvas[r, c] = legend_col
    for i, (seed_col, fill_col) in enumerate(legend_rows, start=1):
        canvas[i, 1] = seed_col
        canvas[i, 3] = fill_col


def _reserve_legend_moat(canvas: np.ndarray) -> None:
    h, w = canvas.shape
    max_r = min(h, LEGEND_H + LEGEND_PAD)
    max_c = min(w, LEGEND_W + LEGEND_PAD)
    for r in range(max_r):
        for c in range(max_c):
            if r < LEGEND_H and c < LEGEND_W:
                continue
            if canvas[r, c] == -1:
                canvas[r, c] = RESERVED


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    needed = 1 + num_seeds + num_seeds + num_seeds + (LEGEND_ROWS - num_seeds) * 2 + FRAME_COLORS_MAX
    if needed > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)

    legend_col = palette.pop()
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]
    fill_colors = [palette.pop() for _ in range(num_seeds)]
    dummy_pairs = [(palette.pop(), palette.pop()) for _ in range(LEGEND_ROWS - num_seeds)]
    frame_colors_pool = [palette.pop() for _ in range(FRAME_COLORS_MAX)]

    active_pairs = list(zip(seed_colors, fill_colors))
    legend_rows = active_pairs + dummy_pairs
    rng.shuffle(legend_rows)
    _place_legend(canvas, legend_col, legend_rows)
    _reserve_legend_moat(canvas)

    seed_layout = _sample_seed_layout(canvas, num_seeds, rng)
    if seed_layout is None:
        return None

    lines_info = []
    for i, (seed_pts, first_rc, forbidden) in enumerate(seed_layout):
        for sr, sc in seed_pts:
            if canvas[sr, sc] != -1:
                return None
            canvas[sr, sc] = seed_colors[i]
        if canvas[first_rc] != -1:
            return None
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(
            canvas,
            first_rc,
            line_colors[i],
            target_len,
            rng,
            forbidden=forbidden,
            allowed_contact=set(seed_pts),
        )
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], fill_colors[i], seed_pts, line_colors[i], path))

    total_loops_target = rng.randint(num_seeds, TOTAL_LOOPS_MAX)
    num_distractors_target = rng.randint(0, max(0, total_loops_target - num_seeds))
    num_attached_target = total_loops_target - num_distractors_target
    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)

    for i, (_, _, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            if _try_attach_frame(canvas, path, line_col, rng.choice(frame_colors_pool), rng) is not None:
                attached += 1
        if attached == 0:
            return None

    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
    else:
        for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
            _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas < 0, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    legend_mask = _legend_mask(grid.shape)
    blocks = _seed_blocks(grid, bg, legend_mask)
    if len(blocks) != num_seeds:
        return None
    if not all(_is_diag_only_seed_block(grid, block, bg) for block in blocks):
        return None
    if not any(int(grid[p]) == bg for p in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))):
        return None
    if not _seed_components_are_separate(grid, bg, legend_mask):
        return None

    block_sets = {frozenset(block) for block in blocks}
    for seed_col, fill_col, seed_pts, _, _ in lines_info:
        if seed_col == fill_col:
            return None
        if int(((grid == seed_col) & ~legend_mask).sum()) != 4:
            return None
        if frozenset(seed_pts) not in block_sets:
            return None

    try:
        legend_map = _extract_legend(grid, bg)
        out = main(grid)
        seed_fill_out = _solve_from_seed_blocks(grid, bg, legend_mask, {seed_col: seed_col for seed_col in legend_map})
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if np.array_equal(out, seed_fill_out):
        return None
    if not any(int(((out == fill_col) & ~legend_mask).sum()) > 0 for _, fill_col, _, _, _ in lines_info):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
