# Combo variant of b10624e5 (index 47)
# combo_spec: D6/2+D11/0+D14/0
# Name: 迭代传播 + 路径轨迹 + 干扰色块
#
# Design analysis for this combo variant
# a) Concrete behavior:
#    - The original task copied the TL motif independently into TR / BR / BL.
#    - Here the copies form a fixed chain: TL -> TR -> BR -> BL.
#    - Each hop uses the current hop's source patch, not the pristine TL patch:
#      first reflect+scale TL into TR, then reflect+scale the new TR patch into
#      BR, then reflect+scale the new BR patch into BL.
#    - After each hop, a dedicated trail color marks the movement path between
#      the two hop endpoints. The visible paths therefore encode the order of
#      propagation, not just the final set of destinations.
#    - Additional non-red distractor blocks are already present in the input and
#      stay unchanged in the output.
# b) Complexity calibration:
#    - The chain order is fixed and repeated in every sample, so the new step
#      order is learnable from 2-3 examples.
#    - One dedicated trail color is reserved and excluded from motif/distractor
#      generation so the path signal is visually recoverable.
#    - Distractors are small blocks with a small shape vocabulary; they increase
#      filtering difficulty without creating a second hidden transformation rule.
# c) Hidden assumptions that break from the original seed:
#    - Input can no longer be derived by clearing solved quadrants from `out`,
#      because trails run through multiple quadrants and across the divider.
#      The input is rebuilt from scratch.
#    - Copy steps are no longer order-independent. The solver must preserve the
#      current transformed patch after each hop and use it as the next source.
#    - Non-red foreground outside the TL motif can no longer be treated as part
#      of the puzzle structure; those pixels are distractors and must be ignored.
# d) Placement legality:
#    - Each quadrant still has exactly one red square anchor.
#    - Generator chooses anchor placements that fit each hop result in its own
#      quadrant, and it places distractors only on background cells outside all
#      motif footprints and red anchors.
#    - Trails are allowed to cross the divider, but they never overwrite red
#      anchors, motif pixels, or distractors.
# e) Visual observability:
#    - The chain is observable because the trail graph is a path with internal
#      nodes at TR and BR, not a three-spoke star centered at TL.
#    - Distractors appear in the same regions as trail traffic, so the solver
#      must separate three roles by shape and color: red anchors, trail pixels,
#      and inert colored blocks.
# f) Visual cleanliness:
#    - Distractors are isolated 1x1 / 1x2 / 2x1 / 2x2 blocks and are kept away
#      from motif footprints, so the scene stays readable.
#    - Trails are one-pixel-wide and can touch divider cells, which keeps the
#      propagation order visible without flooding the board.
# g) Inference and new assumptions:
#    - No rule depends on invisible color ordering. The only special colors are
#      red for anchors, the divider color, the background color, and a dedicated
#      trail color.
#    - main() assumes the only TL non-background component containing red is the
#      source motif, and that every non-TL quadrant has exactly one red anchor.
#      generate_input() enforces both conditions and also guarantees distractors
#      do not overlap any future motif or trail cell.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


BG = 4
DIV = 1
RED = 2
TRAIL = 8
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}
CHAIN = [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (1, 0))]

SOURCE_ANCHOR_SIZE = 2
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 9]
TARGET_SIZE_RANGE = (1, 4)
DISTRACTOR_SHAPES = [(1, 1), (1, 2), (2, 1), (2, 2)]
DISTRACTOR_COUNT_RANGE = (4, 7)
MAX_RETRIES = 500


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col quadrant divider: (mid_row, mid_col)."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize supporting fractional scale ratios."""
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _anchor_center(r0: int, c0: int, size: int) -> tuple[int, int]:
    """Pick a stable integer cell inside a square anchor."""
    return int(r0 + min(size // 2, size - 1)), int(c0 + min(size // 2, size - 1))


def _line_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Return a continuous 8-connected integer line between two cells."""
    dr = int(r1 - r0)
    dc = int(c1 - c0)
    steps = max(abs(dr), abs(dc))
    if steps == 0:
        return [(int(r0), int(c0))]
    pts: list[tuple[int, int]] = []
    last = None
    for i in range(steps + 1):
        rr = int(round(r0 + dr * i / steps))
        cc = int(round(c0 + dc * i / steps))
        cur = (rr, cc)
        if cur != last:
            pts.append(cur)
            last = cur
    return pts


def _paint_trail(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    bg: int,
    divider: int,
    trail: int,
) -> None:
    """Paint a thin connector line without overwriting anchors, motifs, or distractors."""
    h, w = out.shape
    for rr, cc in _line_cells(start[0], start[1], end[0], end[1]):
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if int(out[rr, cc]) in (int(bg), int(divider), int(trail)):
            out[rr, cc] = int(trail)


def _find_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Find one red square anchor per quadrant; values are (r0, c0, size)."""
    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchor blocks"
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))
    assert set(anchors) == set(QUADS), "expected one red anchor in every quadrant"
    return anchors


def _extract_source_patch(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> np.ndarray:
    """Return the TL non-background component that contains the red source anchor."""
    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            patch_bb = tuple(map(int, bbox_coords(pts)))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"
    pr0, pc0, pr1, pc1 = patch_bb
    return tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _transform_patch(
    patch: np.ndarray,
    src_key: tuple[int, int],
    dst_key: tuple[int, int],
    src_size: int,
    dst_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reflect the current source patch by side change, then scale by anchor-size ratio."""
    oriented = patch
    if src_key[1] != dst_key[1]:
        oriented = np.fliplr(oriented)
    if src_key[0] != dst_key[0]:
        oriented = np.flipud(oriented)
    ratio = float(dst_size) / float(src_size)
    new_h = max(1, int(round(oriented.shape[0] * ratio)))
    new_w = max(1, int(round(oriented.shape[1] * ratio)))
    return oriented, _resize_nn(oriented, new_h, new_w)


def _build_chain_patches(
    patch: np.ndarray,
    anchor_sizes: dict[tuple[int, int], int],
) -> dict[tuple[int, int], np.ndarray] | None:
    """Return the actual patch used in each quadrant under chain propagation."""
    patches: dict[tuple[int, int], np.ndarray] = {(0, 0): patch}
    current_patch = patch
    current_key = (0, 0)
    current_size = anchor_sizes[(0, 0)]
    for src_key, dst_key in CHAIN:
        assert src_key == current_key, "chain order must consume the previous hop result"
        oriented, next_patch = _transform_patch(current_patch, src_key, dst_key, current_size, anchor_sizes[dst_key])
        roundtrip = _resize_nn(next_patch, oriented.shape[0], oriented.shape[1])
        if not np.array_equal(roundtrip, oriented):
            return None
        patches[dst_key] = next_patch
        current_patch = next_patch
        current_key = dst_key
        current_size = anchor_sizes[dst_key]
    return patches


def _stamp(
    out: np.ndarray,
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    bg: int,
    divider: int,
    trail: int,
    mid_r: int,
    mid_c: int,
) -> None:
    """Stamp a transformed patch without overwriting red anchors or distractors."""
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled source patch must contain red anchor pixels"
    place_r0 = int(anchor_r - int(rr_red.min()))
    place_c0 = int(anchor_c - int(cc_red.min()))

    src_rr, src_cc = np.where(patch_s != bg)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0
    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < out.shape[0])
        & (tgt_cc >= 0)
        & (tgt_cc < out.shape[1])
        & (tgt_rr != mid_r)
        & (tgt_cc != mid_c)
    )
    tgt_rr = tgt_rr[valid]
    tgt_cc = tgt_cc[valid]
    src_rr = src_rr[valid]
    src_cc = src_cc[valid]

    out_vals = out[tgt_rr, tgt_cc]
    paintable = (out_vals == bg) | (out_vals == divider) | (out_vals == trail)
    out[tgt_rr[paintable], tgt_cc[paintable]] = patch_s[src_rr[paintable], src_cc[paintable]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the source motif around the quadrants in a fixed chain, leaving a visible trail for each hop while ignoring inert distractor blocks.

    Input structure:
    - One monochrome full row and one matching full column split the 27x27 board into four 13x13 quadrants.
    - Every quadrant contains exactly one red square anchor block; only the top-left quadrant also contains the full source motif attached to its anchor.
    - Extra small non-red colored blocks may appear anywhere outside that source component; they are distractors and do not participate in any copy step.

    Rule:
    - Extract the top-left motif, then copy it in the fixed order TL -> TR -> BR -> BL.
    - On each hop, reflect the current source patch for the left/right and top/bottom change between the two quadrants, then scale it by the destination-anchor/source-anchor size ratio and align its red region to the destination anchor.
    - After each hop, draw a thin trail from the current source anchor center to the destination anchor center using the dedicated trail color, without overwriting motifs, red anchors, or distractor blocks.

    Key insight:
    - The output is not three independent copies from TL: the trail topology reveals a chain, and the patch used at BR and BL is whatever the previous hop produced.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    divider = int(g[mid_r, mid_c])
    anchors = _find_anchors(g, mid_r, mid_c)
    patch = _extract_source_patch(g, bg, mid_r, mid_c)

    anchor_sizes = {key: size for key, (_r0, _c0, size) in anchors.items()}
    patches = _build_chain_patches(patch, anchor_sizes)
    assert patches is not None, "iterative scaling must be lossless for every chain hop"

    for src_key, dst_key in CHAIN:
        sr0, sc0, ssize = anchors[src_key]
        tr0, tc0, _tsize = anchors[dst_key]
        _stamp(out, patches[dst_key], tr0, tc0, bg, divider, TRAIL, mid_r, mid_c)
        _paint_trail(
            out,
            _anchor_center(sr0, sc0, ssize),
            _anchor_center(tr0, tc0, anchors[dst_key][2]),
            bg,
            divider,
            TRAIL,
        )

    return out


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    """Grow a 4-connected region from seeds until it has n_fill cells."""
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        filled.add(cell)
        frontier.discard(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_colors(
    filled: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray | None:
    """Assign motif colors; all rows and cols inside the bbox must be represented."""
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    non_red = [p for p in filled if p not in red_cells]
    random.shuffle(non_red)
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    extras = random.sample(COLOR_POOL, n_colors - 1)
    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


def _source_patch() -> tuple[np.ndarray, int, int] | None:
    """Build a logical motif and upscale it so every source cell is a 2x2 block."""
    lh, lw = random.choice(BLOCK_DIMS)
    n_cells = lh * lw
    min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    ar, ac = random.randrange(lh), random.randrange(lw)
    filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
    logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
    if logical is None:
        return None
    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    return patch, int(ar * SOURCE_ANCHOR_SIZE), int(ac * SOURCE_ANCHOR_SIZE)


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Pick an anchor position, preferring placements where the whole patch fits."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - size + 1
    anch_c0, anch_c1 = q_c0, q_c1 - size + 1

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _shape_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr in range(h) for dc in range(w)}


def _touches_protected(cells: set[tuple[int, int]], protected: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in protected:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in protected:
                return True
    return False


def _add_distractors(out: np.ndarray) -> list[tuple[set[tuple[int, int]], int]] | None:
    """Add small inert color blocks near the trail but away from motifs and anchors."""
    trail_cells = {(int(r), int(c)) for r, c in np.argwhere(out == TRAIL)}
    if not trail_cells:
        return None

    protected = {(int(r), int(c)) for r, c in np.argwhere((out != BG) & (out != TRAIL))}
    candidates: list[tuple[int, tuple[int, int], set[tuple[int, int]]]] = []
    for qb, qr in QUADS:
        if (qb, qr) == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
        for h, w in DISTRACTOR_SHAPES:
            for r0 in range(q_r0, q_r1 - h + 2):
                for c0 in range(q_c0, q_c1 - w + 2):
                    cells = _shape_cells(r0, c0, h, w)
                    if any(out[r, c] != BG for r, c in cells):
                        continue
                    if _touches_protected(cells, protected):
                        continue
                    dist = min(abs(r - tr) + abs(c - tc) for r, c in cells for tr, tc in trail_cells)
                    if dist > 3:
                        continue
                    candidates.append((dist, (h, w), cells))
    random.shuffle(candidates)
    candidates.sort(key=lambda item: item[0])

    want = random.randint(*DISTRACTOR_COUNT_RANGE)
    n_colors = random.randint(2, min(4, len(COLOR_POOL)))
    palette = random.sample(COLOR_POOL, n_colors)
    chosen: list[tuple[set[tuple[int, int]], int]] = []
    used: set[tuple[int, int]] = set()
    for _dist, _shape, cells in candidates:
        if cells & used:
            continue
        if _touches_protected(cells, used):
            continue
        color = random.choice(palette)
        chosen.append((cells, color))
        used |= cells
        if len(chosen) >= want:
            break
    if len(chosen) < 3:
        return None
    for cells, color in chosen:
        for r, c in cells:
            out[r, c] = int(color)
    return chosen


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, rebuild input, oracle-check."""
    anchor_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
    }

    source = _source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    patches = _build_chain_patches(patch, anchor_sizes)
    if patches is None:
        return None

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (int(tl_r0 + anchor_r_in_patch), int(tl_c0 + anchor_c_in_patch))
    for key in ((0, 1), (1, 1), (1, 0)):
        anchors[key] = _choose_anchor(key[0], key[1], anchor_sizes[key], patches[key])

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in anchor_sizes.items():
        if key == (0, 0):
            continue
        paste(out, np.full((size, size), RED, dtype=int), anchors[key][0], anchors[key][1])

    for src_key, dst_key in CHAIN:
        tr0, tc0 = anchors[dst_key]
        _stamp(out, patches[dst_key], tr0, tc0, BG, DIV, TRAIL, MID, MID)
        _paint_trail(
            out,
            _anchor_center(anchors[src_key][0], anchors[src_key][1], anchor_sizes[src_key]),
            _anchor_center(anchors[dst_key][0], anchors[dst_key][1], anchor_sizes[dst_key]),
            BG,
            DIV,
            TRAIL,
        )

    distractors = _add_distractors(out)
    if distractors is None:
        return None

    trail_cell_count = int(np.count_nonzero(out == TRAIL))
    distractor_cell_count = sum(len(cells) for cells, _color in distractors)
    if trail_cell_count < 12 or distractor_cell_count < 5:
        return None

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    paste(inp, patch, tl_r0, tl_c0)
    for key, size in anchor_sizes.items():
        if key == (0, 0):
            continue
        paste(inp, np.full((size, size), RED, dtype=int), anchors[key][0], anchors[key][1])
    for cells, color in distractors:
        for r, c in cells:
            inp[r, c] = int(color)

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo variant input for b10624e5")
