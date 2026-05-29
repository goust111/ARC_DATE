# Combo variant of b10624e5 (index 32)
# combo_spec: D8/0+D14/1
# Name: 象限边界装饰线 + 显式分割线划分

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete trace:
#    The original task can be solved once a single row+col divider is found.
#    Here the input contains one functional full row/full col pair plus one extra
#    decorative full row and one extra decorative full col of the same color.
#    Example: true split at (12, 13), decorative row at 7, decorative col at 19.
#    The old `_find_dividers()` would see four same-color row/col matches:
#    (7,13), (7,19), (12,13), (12,19). The new solver therefore cannot rely on
#    “unique matching divider color”; it must test which candidate pair yields
#    exactly one red square anchor in each region. The generator forces one
#    right-side anchor to lie in the strip between true and decorative lines, so
#    only the true pair gives the required 1-anchor-per-region partition.
# b) Complexity calibration:
#    One extra full row and one extra full col already create visible ambiguity
#    without making the scene unreadable. Using same-color full lines keeps the
#    interference strong, while limiting the count to one row + one col avoids a
#    messy lattice of candidate partitions.
# c) Hidden assumptions broken from the original solver:
#    - "There is exactly one matching full-row/full-col divider" is false now.
#    - "Any full matching row/col pair is usable" is false now.
#    - The source component extraction still works only if the TL motif does not
#      touch decorative lines, so the generator keeps the TL motif away from them.
#    - Red-anchor detection assumes red components are filled squares; the
#      generator preserves that by keeping decorations non-red.
# d) Placement legality:
#    The generator builds the output first, then derives the input. It keeps
#    anchors off every full line, forces the witness anchor into the strip
#    between true and decorative lines, and requires every target placement to
#    leave at least some non-red motif pixels visible on background cells.
# e/f/g) Human observability and anti-degeneration:
#    The decorative lines are visible full-grid separators, so the variant looks
#    materially different from the original. Yet the actual rule is still
#    inferable: only one row/col pair separates the four red square anchors into
#    four regions with the TL source motif in the top-left region. This also
#    prevents degeneration back to the original task: the original solver fails
#    on these inputs because the divider pair is no longer unique.


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27

SOURCE_ANCHOR_SIZE = 2
TRUE_DIV_RANGE = (12, 14)
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 600

BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]
MIN_VISIBLE_PAYLOAD = 2


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _quadrant_bounds(split_r: int, split_c: int, qb: int, qr: int) -> tuple[int, int, int, int]:
    r0 = 0 if qb == 0 else split_r + 1
    r1 = split_r - 1 if qb == 0 else GRID_SIZE - 1
    c0 = 0 if qr == 0 else split_c + 1
    c1 = split_c - 1 if qr == 0 else GRID_SIZE - 1
    return r0, r1, c0, c1


def _full_lines(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    return rows, cols


def _red_square_anchors(g: np.ndarray) -> list[tuple[int, int, int]]:
    anchors: list[tuple[int, int, int]] = []
    for pts in components_mask(g == RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w:
            continue
        if pts.shape[0] != h * w:
            continue
        if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == RED):
            continue
        anchors.append((int(r0), int(c0), int(h)))
    return anchors


def _classify_anchors(
    anchors: list[tuple[int, int, int]], split_r: int, split_c: int
) -> dict[tuple[int, int], tuple[int, int, int]] | None:
    out: dict[tuple[int, int], tuple[int, int, int]] = {}
    for r0, c0, size in anchors:
        r1 = r0 + size - 1
        c1 = c0 + size - 1
        if r1 < split_r:
            qb = 0
        elif r0 > split_r:
            qb = 1
        else:
            return None
        if c1 < split_c:
            qr = 0
        elif c0 > split_c:
            qr = 1
        else:
            return None
        key = (qb, qr)
        if key in out:
            return None
        out[key] = (r0, c0, size)
    return out if len(out) == 4 else None


def _extract_source_patch(
    g: np.ndarray, bg: int, split_r: int, split_c: int, tl_anchor: tuple[int, int, int]
) -> np.ndarray | None:
    tl = g[:split_r, :split_c]
    ar0, ac0, size = tl_anchor
    comps = components_mask(tl != bg, connectivity=4)
    chosen = None
    for pts in comps:
        if any(ar0 <= r < ar0 + size and ac0 <= c < ac0 + size for r, c in pts):
            if chosen is not None:
                return None
            chosen = pts
    if chosen is None:
        return None
    r0, c0, r1, c1 = bbox_coords(chosen)
    return tl[r0 : r1 + 1, c0 : c1 + 1].copy()


def _select_functional_split(
    g: np.ndarray, bg: int
) -> tuple[int, int, dict[tuple[int, int], tuple[int, int, int]], np.ndarray]:
    rows, cols = _full_lines(g, bg)
    anchors = _red_square_anchors(g)
    assert len(anchors) == 4, "grid must contain exactly four red square anchors"

    valid: list[tuple[int, int, dict[tuple[int, int], tuple[int, int, int]], np.ndarray]] = []
    for r, vr in rows:
        for c, vc in cols:
            if vr != vc:
                continue
            anchor_map = _classify_anchors(anchors, r, c)
            if anchor_map is None or (0, 0) not in anchor_map:
                continue
            patch = _extract_source_patch(g, bg, r, c, anchor_map[(0, 0)])
            if patch is None:
                continue
            valid.append((int(r), int(c), anchor_map, patch))

    assert len(valid) == 1, "grid must have exactly one anchor-consistent functional divider pair"
    return valid[0]


def _anchor_top_left_in_patch(patch: np.ndarray) -> tuple[int, int]:
    rr, cc = np.where(patch == RED)
    assert rr.size > 0, "patch must contain red anchor pixels"
    return int(rr.min()), int(cc.min())


def _stamp_bg_only(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, bg: int) -> None:
    ar0, ac0 = _anchor_top_left_in_patch(patch_s)
    place_r0 = anchor_r - ar0
    place_c0 = anchor_c - ac0
    rr_p, cc_p = np.where(patch_s != bg)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < out.shape[0] and 0 <= cc_t < out.shape[1]):
            continue
        if out[rr_t, cc_t] != bg:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    p = patch
    if qr == 1:
        p = np.fliplr(p)
    if qb == 1:
        p = np.flipud(p)
    ratio = float(target_size) / float(SOURCE_ANCHOR_SIZE)
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    patch_s = _resize_nn(p, new_h, new_w)
    roundtrip = _resize_nn(patch_s, p.shape[0], p.shape[1])
    assert np.array_equal(roundtrip, p), "scaling must preserve the motif exactly"
    return patch_s


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the top-left motif to the other regions, but first identify the true splitter
    among several divider-like full lines.

    Input structure:
    - The grid contains multiple full horizontal/vertical separator-like lines of the same color.
    - Exactly one row+col pair is functional: it partitions the grid into four regions with one
      red square anchor per region.
    - The top-left functional region contains the source motif attached to its red anchor; other
      regions initially show only their red anchors plus decorative full lines.

    Rule:
    - Find the unique full-row/full-col pair that makes the four red square anchors occupy the
      four regions exactly once each.
    - Extract the source motif from the top-left functional region, reflect it by region parity,
      and scale it by the target/source anchor-size ratio.
    - Overlay the transformed motif onto each target anchor, but preserve pre-existing separator
      lines and anchors by only painting onto background cells.

    Key insight:
    - The divider color alone is no longer enough because decorative full lines mimic true
      separators. The usable split is the one supported jointly by anchor geography and the TL
      source component.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    split_r, split_c, anchors, patch = _select_functional_split(g, bg)
    source_size = anchors[(0, 0)][2]
    assert source_size > 0, "top-left anchor must have positive size"

    for (qb, qr), (anchor_r, anchor_c, size) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        patch_s = _scaled_patch(patch, qb, qr, size)
        _stamp_bg_only(out, patch_s, anchor_r, anchor_c, bg)

    return out


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
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
    filled: set[tuple[int, int]], red_cells: set[tuple[int, int]], h: int, w: int
) -> np.ndarray | None:
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
    pool = COLOR_POOL[:]
    random.shuffle(pool)
    extras = pool[: n_colors - 1]
    patch = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        patch[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        patch[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return patch


def _source_patch(needs_block: bool) -> tuple[np.ndarray, int, int] | None:
    s = SOURCE_ANCHOR_SIZE
    use_block = needs_block or random.random() < BLOCK_PROB
    if use_block:
        lh, lw = random.choice(BLOCK_DIMS)
        n_cells = lh * lw
        min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
        max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
        ar, ac = random.randrange(lh), random.randrange(lw)
        filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
        logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
        if logical is None:
            return None
        patch = np.repeat(np.repeat(logical, s, axis=0), s, axis=1)
        return patch, ar * s, ac * s

    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar, ac = random.randint(0, h - s), random.randint(0, w - s)
    red_cells = {(ar + dr, ac + dc) for dr in range(s) for dc in range(s)}
    min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _build_base(split_r: int, split_c: int, decor_r: int, decor_c: int) -> np.ndarray:
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    for r in {split_r, decor_r}:
        out[r, :] = DIV
    for c in {split_c, decor_c}:
        out[:, c] = DIV
    return out


def _target_visible_payload(
    patch_s: np.ndarray, anchor_r: int, anchor_c: int, blocked_rows: set[int], blocked_cols: set[int]
) -> int:
    ar0, ac0 = _anchor_top_left_in_patch(patch_s)
    place_r0 = anchor_r - ar0
    place_c0 = anchor_c - ac0
    count = 0
    rr, cc = np.where((patch_s != BG) & (patch_s != RED))
    for pr, pc in zip(rr.tolist(), cc.tolist(), strict=False):
        tr = place_r0 + pr
        tc = place_c0 + pc
        if not (0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE):
            continue
        if tr in blocked_rows or tc in blocked_cols:
            continue
        count += 1
    return count


def _choose_tl_position(
    patch: np.ndarray,
    anchor_r_in_patch: int,
    split_r: int,
    split_c: int,
    decor_r: int,
    witness_key: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = patch.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(0, split_r - h + 1):
        for c0 in range(0, split_c - w + 1):
            if r0 <= decor_r < r0 + h:
                continue
            if witness_key == (0, 1) and r0 + anchor_r_in_patch >= decor_r:
                continue
            candidates.append((r0, c0))
    return random.choice(candidates) if candidates else None


def _choose_anchor(
    qb: int,
    qr: int,
    size: int,
    patch_s: np.ndarray,
    split_r: int,
    split_c: int,
    decor_r: int,
    decor_c: int,
    require_row_band: bool,
    require_col_band: bool,
) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = _quadrant_bounds(split_r, split_c, qb, qr)
    r_lo, r_hi = q_r0, q_r1 - size + 1
    c_lo, c_hi = q_c0, q_c1 - size + 1
    if r_lo > r_hi or c_lo > c_hi:
        return None

    if require_row_band:
        if decor_r < split_r:
            r_lo = max(r_lo, decor_r + 1)
            r_hi = min(r_hi, split_r - size)
        else:
            r_lo = max(r_lo, split_r + 1)
            r_hi = min(r_hi, decor_r - size)
    if require_col_band:
        c_lo = max(c_lo, split_c + 1)
        c_hi = min(c_hi, decor_c - size)

    blocked_rows = {split_r, decor_r}
    blocked_cols = {split_c, decor_c}
    scored: list[tuple[int, int, int]] = []
    for ar in range(r_lo, r_hi + 1):
        for ac in range(c_lo, c_hi + 1):
            if any(ar <= rr < ar + size for rr in blocked_rows):
                continue
            if any(ac <= cc < ac + size for cc in blocked_cols):
                continue
            visible = _target_visible_payload(patch_s, ar, ac, blocked_rows, blocked_cols)
            if visible < MIN_VISIBLE_PAYLOAD:
                continue
            scored.append((visible, ar, ac))

    if not scored:
        return None
    best = max(score for score, _, _ in scored)
    top = [(ar, ac) for score, ar, ac in scored if score == best]
    return random.choice(top)


def _gen_attempt() -> np.ndarray | None:
    split_r = random.randint(*TRUE_DIV_RANGE)
    split_c = random.randint(*TRUE_DIV_RANGE)
    witness_key = random.choice([(0, 1), (1, 1)])

    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }
    witness_size = target_sizes[witness_key]
    row_offset = random.randint(witness_size + 1, 6)
    col_offset = random.randint(witness_size + 1, 6)
    decor_r = split_r - row_offset if witness_key == (0, 1) else split_r + row_offset
    decor_c = split_c + col_offset

    needs_block = any(size % SOURCE_ANCHOR_SIZE != 0 for key, size in target_sizes.items() if key != (0, 0))
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    tl_pos = _choose_tl_position(patch, anchor_r_in_patch, split_r, split_c, decor_r, witness_key)
    if tl_pos is None:
        return None
    tl_r0, tl_c0 = tl_pos

    try:
        scaled = {
            key: _scaled_patch(patch, key[0], key[1], size)
            for key, size in target_sizes.items()
            if key != (0, 0)
        }
    except AssertionError:
        return None

    anchors: dict[tuple[int, int], tuple[int, int]] = {
        (0, 0): (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    }
    for key in [(0, 1), (1, 0), (1, 1)]:
        pos = _choose_anchor(
            key[0],
            key[1],
            target_sizes[key],
            scaled[key],
            split_r,
            split_c,
            decor_r,
            decor_c,
            require_row_band=(key == witness_key),
            require_col_band=(key == witness_key or key[1] == 1),
        )
        if pos is None:
            return None
        anchors[key] = pos

    out = _build_base(split_r, split_c, decor_r, decor_c)
    paste(out, patch, tl_r0, tl_c0)
    for key in [(0, 1), (1, 0), (1, 1)]:
        ar, ac = anchors[key]
        size = target_sizes[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp_bg_only(out, patch_s, anchors[key][0], anchors[key][1], BG)

    inp = _build_base(split_r, split_c, decor_r, decor_c)
    paste(inp, patch, tl_r0, tl_c0)
    for key in [(0, 1), (1, 0), (1, 1)]:
        ar, ac = anchors[key]
        size = target_sizes[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        try:
            grid = _gen_attempt()
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused b10624e5 input")
