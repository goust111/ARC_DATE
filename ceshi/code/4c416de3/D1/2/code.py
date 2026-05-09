# Variant of 4c416de3
# Dimension: D1 — 匹配条件：对角对称映射 (match_cond)
# Change: 源图案关于角点的对角线轴（dr==dc）对称，通过旋转/翻转映射到四种角点方位（TL/TR/BL/BR）... → 仅水平/垂直镜像对称（取消对角对称约束）

# Design analysis for the D1 variant
# ----------------------------------
# a) Concrete behavior and trace
# - The frame layout stays the same: 2-3 hollow 0-frames on a solid non-zero background.
# - Every useful corner still belongs to one of TL/TR/BL/BR and shares one canonical
#   pattern, expressed in TL-oriented offsets around the corner.
# - The original task used patterns symmetric under transpose (dr, dc) <-> (dc, dr),
#   then mirrored / sign-flipped them to the other three corner orientations.
# - In this variant the corner-to-corner transport still uses only horizontal/vertical
#   mirroring via sign flips, but the canonical pattern itself is intentionally NOT
#   transpose-symmetric. Example:
#     original-style canon: {(-1,0), (0,-1), (1,1)} also implies its transpose pair.
#     variant canon: {(-1,0), (0,-1), (0,0), (1,1), (1,2)}.
#   Copying that canon to TR flips the column sign and produces
#   {(-1,0), (0,1), (0,0), (1,-1), (1,-2)}; this is visibly different from BL, which
#   flips the row sign instead.
# - Targets keep only one marker pixel, but the marker is no longer forced onto a
#   diagonal (k,k). Instead it is a unique deepest strictly-inward cell (dr>0 and dc>0)
#   with maximal dr+dc; generator templates guarantee this cell is off-diagonal.
#
# b) Complexity calibration
# - Patterns keep 5-6 cells, matching the original visual density.
# - Only a small family of templates is used, with repeated support across samples.
# - Every template has 2-3 outward cells plus 2+ inward cells, so source-vs-target
#   distinction remains stable and target completion is still non-trivial.
#
# c) Implicit-assumption audit for main()
# - `_find_corners`, `_edge_metric`, `_colored_around`: depend only on frame geometry;
#   unchanged and still valid.
# - Source detection by outward colored cells: unchanged; templates still include >=2
#   outward cells of one color at every source corner.
# - Source agreement check: unchanged; all sources still share the same canonical pattern.
# - Old diagonal-symmetry assertion: invalid in this variant and removed.
# - Old marker extraction from deepest diagonal `(k,k)`: invalid once the marker may be
#   `(1,2)` or `(2,1)`. Replaced by `_marker_from_canon`, which finds the unique deepest
#   strictly-inward anchor cell. Generator guarantees the uniqueness needed here.
# - Target-color consistency checks still hold because non-source corners retain only the
#   marker cell in input; any already-painted pattern cells at canonical offsets must match
#   that marker color.
#
# d) Placement legality
# - Pattern offsets remain within |dr|,|dc| <= 3, so the existing neighborhood scan radius
#   stays sufficient.
# - Frames remain size 5-8, and the new inward marker offsets are capped at 2, so every
#   kept marker lies safely inside the frame.
# - The output-first generator still validates `main(inp) == output`, which catches overlap,
#   out-of-bounds, and source/target ambiguity regressions automatically.
#
# e) Visual observability
# - Every template is non-transpose-symmetric and uses an off-diagonal inward anchor, so a
#   TL source mirrored to TR vs BL is visibly different. The variant is therefore observable
#   from a few examples instead of collapsing back to diagonal symmetry.
#
# f) Visual cleanliness
# - No extra object types are introduced; only the corner glyph changes shape.
# - The existing frame spacing and modest pattern size keep local neighborhoods legible.
#
# g) Color inference
# - Colors still behave exactly as in the original: each target uses its own marker color.
#   No hidden color ordering or palette arithmetic is introduced.

import numpy as np
from collections import Counter

from arc_utils import get_bg
import random
from arc_utils import draw_line


CORNER_KINDS = ('TL', 'TR', 'BL', 'BR')

# Inward-along-frame-edge directions for each corner kind.
EDGE_DIRS = {
    'TL': ((+1, 0), (0, +1)),
    'TR': ((+1, 0), (0, -1)),
    'BL': ((-1, 0), (0, +1)),
    'BR': ((-1, 0), (0, -1)),
}
# Outward-from-corner directions; cells here must not be 0 (guards mid-edge false positives).
OUT_DIRS = {
    'TL': ((-1, 0), (0, -1)),
    'TR': ((-1, 0), (0, +1)),
    'BL': ((+1, 0), (0, -1)),
    'BR': ((+1, 0), (0, +1)),
}

# Radius for scanning a corner's neighborhood. Must exceed max canonical |dr|,|dc| a pattern can have.
_NEIGH_R = 3


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical (TL-orientation) offset; self-inverse."""
    if kind == 'TL': return (dr, dc)
    if kind == 'TR': return (dr, -dc)
    if kind == 'BL': return (-dr, dc)
    if kind == 'BR': return (-dr, -dc)
    raise ValueError(f'unknown corner kind: {kind}')


def _edge_metric(g, r, c, dr, dc, bg, H, W, max_step=10):
    """Walk from (r,c) along (dr,dc) until bg/boundary. Return (non_bg_count, saw_zero)."""
    cnt = 0
    saw_zero = False
    for k in range(1, max_step + 1):
        rr, cc = r + k * dr, c + k * dc
        if not (0 <= rr < H and 0 <= cc < W):
            break
        v = int(g[rr, cc])
        if v == bg:
            break
        cnt += 1
        if v == 0:
            saw_zero = True
    return cnt, saw_zero


def _find_corners(g, bg):
    H, W = g.shape
    corners = []
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) == bg:
                continue
            for kind in CORNER_KINDS:
                (d1r, d1c), (d2r, d2c) = EDGE_DIRS[kind]
                r1, c1 = r + d1r, c + d1c
                r2, c2 = r + d2r, c + d2c
                if not (0 <= r1 < H and 0 <= c1 < W and 0 <= r2 < H and 0 <= c2 < W):
                    continue
                if int(g[r1, c1]) == bg or int(g[r2, c2]) == bg:
                    continue
                cnt1, sz1 = _edge_metric(g, r, c, d1r, d1c, bg, H, W)
                cnt2, sz2 = _edge_metric(g, r, c, d2r, d2c, bg, H, W)
                if not (cnt1 >= 3 and sz1 and cnt2 >= 3 and sz2):
                    continue
                ok = True
                for (odr, odc) in OUT_DIRS[kind]:
                    rr, cc = r + odr, c + odc
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == 0:
                        ok = False
                        break
                if not ok:
                    continue
                corners.append((r, c, kind))
                break
    return corners


def _colored_around(g, r, c, kind, bg):
    H, W = g.shape
    items = []
    for dr in range(-_NEIGH_R, _NEIGH_R + 1):
        for dc in range(-_NEIGH_R, _NEIGH_R + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(g[rr, cc])
            if v == bg or v == 0:
                continue
            items.append((_rot(dr, dc, kind), v))
    return items


def _marker_from_canon(canon):
    """Pick the unique deepest strictly-inward anchor cell used as the target marker."""
    inward = [(dr, dc) for (dr, dc) in canon if dr > 0 and dc > 0]
    assert inward, 'pattern has no strictly inward anchor cell'
    best_score = max(dr + dc for (dr, dc) in inward)
    best = [(dr, dc) for (dr, dc) in inward if dr + dc == best_score]
    assert len(best) == 1, f'pattern has ambiguous deepest inward marker: {sorted(best)}'
    marker = best[0]
    assert marker[0] != marker[1], 'variant marker must be off diagonal'
    return marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy a pre-drawn corner glyph to every other marker-bearing frame corner.

    Input structure:
    - The background is the most frequent non-zero color, with 2-3 hollow 0-frames.
    - One or more source corners already show the full glyph in a single color. The glyph
      includes outward cells plus an asymmetric inward tail.
    - Other participating corners keep only one colored marker pixel at the glyph's unique
      deepest strictly-inward anchor; that pixel's color is the target output color.

    Rule:
    - Read the source glyph in canonical TL-oriented offsets.
    - Mirror that same canonical glyph into each other corner orientation using only row/col
      sign flips, and paint it with the target corner's marker color.

    Key insight:
    - Unlike the original diagonal-symmetry version, the source glyph is not required to be
      symmetric under (dr, dc) <-> (dc, dr). The useful invariant is instead the corner
      orientation itself: TL/TR differ by a left-right mirror, TL/BL by an up-down mirror,
      and BR applies both. The preserved marker is the unique deepest inward cell, not a
      diagonal `(k,k)` point.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, 'no frame corner detected'

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}

    def _is_outward(off):
        return off[0] < 0 or off[1] < 0

    sources = []
    for c_key, items in per_corner.items():
        outward = [(o, v) for (o, v) in items if _is_outward(o)]
        if len(outward) >= 2:
            colors = {v for (_, v) in outward}
            assert len(colors) == 1, f'source {c_key} has mixed outward colors: {colors}'
            src_color = next(iter(colors))
            same_color_offsets = [o for (o, v) in items if v == src_color]
            if not any(dr > 0 and dc > 0 for (dr, dc) in same_color_offsets):
                continue
            sources.append(c_key)
    assert sources, 'at least one source corner (pre-drawn outward pattern) required'

    def _extract(c_key):
        items = per_corner[c_key]
        pat_col = Counter(v for (_, v) in items).most_common(1)[0][0]
        return frozenset(o for (o, v) in items if v == pat_col), pat_col

    canon, _ = _extract(sources[0])
    for s in sources[1:]:
        other, _ = _extract(s)
        assert other == canon, f'source patterns disagree: {sorted(canon)} vs {sorted(other)}'

    # D1 variant: the canonical glyph must visibly break transpose symmetry.
    transposed = frozenset((dc, dr) for (dr, dc) in canon)
    assert transposed != canon, 'pattern unexpectedly remains diagonally symmetric'

    marker_canon = _marker_from_canon(canon)

    src_set = set(sources)
    targets = []
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        mdr, mdc = _rot(*marker_canon, kind)
        mr, mc = r + mdr, c + mdc
        if not (0 <= mr < H and 0 <= mc < W):
            continue
        v = int(g[mr, mc])
        if v == bg or v == 0:
            continue
        tcol = v
        for (off, color) in per_corner[c_key]:
            if off in canon:
                assert color == tcol, (
                    f'target ({r},{c},{kind}) has conflicting color {color} at '
                    f'canonical offset {off}; expected {tcol}'
                )
        targets.append((r, c, kind, tcol))

    assert targets, 'no target corners detected (every corner appears to be a source)'

    out = g.copy()
    for (r, c, kind, tcol) in targets:
        for (cdr, cdc) in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == tcol, (
                f'paint conflict at ({rr},{cc}) for target ({r},{c},{kind}) '
                f'color {tcol}: existing color {v}'
            )
            out[rr, cc] = tcol

    return out


# --------------- generator (output-first pattern replication (Codex)) ---------------

# --- canvas & layout ---
H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (5, 8)
SOURCE_MIN_SIZE = 6

# --- structural counts ---
N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2

# --- generation retries ---
MAX_ATTEMPTS = 300

# --- canonical (TL-oriented) pattern templates ---
# Every template:
# - stays within |dr|,|dc| <= 3,
# - contains >=2 outward cells,
# - contains a unique deepest strictly-inward marker with dr>0 and dc>0,
# - is intentionally NOT transpose-symmetric.
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1), (1, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 2)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 1)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (2, 1)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (1, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 1)],
)

_CORNER_OFFSETS = {'TL': (0, 0), 'TR': (0, 1), 'BL': (1, 0), 'BR': (1, 1)}


def _split_total(total: int, parts: int) -> list[int]:
    """Partition `total` non-negative units into `parts` bins uniformly at random."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out, prev = [], -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_pattern() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Return (pattern cells, canonical marker offset)."""
    pattern = list(random.choice(PATTERN_TEMPLATES))
    canon = frozenset(pattern)
    assert frozenset((dc, dr) for (dr, dc) in canon) != canon
    marker = _marker_from_canon(canon)
    return pattern, marker


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    """Draw a size x size hollow frame of 0 with TL at (r, c)."""
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _corner_coord(frame: dict, kind: str) -> tuple[int, int]:
    dr, dc = _CORNER_OFFSETS[kind]
    s1 = frame['size'] - 1
    return frame['r'] + dr * s1, frame['c'] + dc * s1


def _sample_frame_pair(span: int) -> tuple[int, int, int, int] | None:
    """Pick two frame sizes and their origins along one axis of length `span`."""
    lo, hi = FRAME_SIZE_RANGE
    pair_limit = span - (2 * BORDER + FRAME_GAP)
    max_first = min(hi, pair_limit - lo)
    if max_first < lo:
        return None
    s1 = random.randint(lo, max_first)
    s2 = random.randint(lo, min(hi, pair_limit - s1))
    slack = span - (2 * BORDER + s1 + FRAME_GAP + s2)
    before, between, _after = _split_total(slack, 3)
    o1 = BORDER + before
    o2 = o1 + s1 + FRAME_GAP + between
    return s1, s2, o1, o2


def _sample_two_frames(H: int, W: int) -> list[dict] | None:
    """Place two frames either side-by-side (shared row) or stacked (shared col)."""
    horizontal = random.random() < TWO_FRAME_HORIZONTAL_PROB
    axis = _sample_frame_pair(W if horizontal else H)
    if axis is None:
        return None
    s1, s2, o1, o2 = axis
    if horizontal:
        row = random.randint(BORDER, H - max(s1, s2) - BORDER)
        return [{'r': row, 'c': o1, 'size': s1}, {'r': row, 'c': o2, 'size': s2}]
    col = random.randint(BORDER, W - max(s1, s2) - BORDER)
    return [{'r': o1, 'c': col, 'size': s1}, {'r': o2, 'c': col, 'size': s2}]


def _sample_three_frames(H: int, W: int) -> list[dict] | None:
    """Place three frames in three of the four corner slots (TL/TR/BL/BR), one missing."""
    lo, hi = FRAME_SIZE_RANGE
    missing = random.choice(CORNER_KINDS)
    present = tuple(k for k in CORNER_KINDS if k != missing)

    def _bands(sizes: dict[str, int]) -> tuple[int, int, int, int]:
        top = max((sizes[s] for s in present if s[0] == 'T'), default=0)
        bot = max((sizes[s] for s in present if s[0] == 'B'), default=0)
        left = max((sizes[s] for s in present if s[1] == 'L'), default=0)
        right = max((sizes[s] for s in present if s[1] == 'R'), default=0)
        return top, bot, left, right

    feasible = []
    n_sizes = hi - lo + 1
    for combo in np.ndindex(*(n_sizes for _ in present)):
        sizes = {slot: lo + i for slot, i in zip(present, combo)}
        top, bot, left, right = _bands(sizes)
        if top + FRAME_GAP + bot > H - 2 * BORDER:
            continue
        if left + FRAME_GAP + right > W - 2 * BORDER:
            continue
        feasible.append(sizes)
    if not feasible:
        return None

    sizes = random.choice(feasible)
    top, bot, left, right = _bands(sizes)
    top_extra, mid_y, _ = _split_total(H - (2 * BORDER + top + FRAME_GAP + bot), 3)
    left_extra, mid_x, _ = _split_total(W - (2 * BORDER + left + FRAME_GAP + right), 3)
    top_row = BORDER + top_extra
    bot_row = top_row + top + FRAME_GAP + mid_y
    left_col = BORDER + left_extra
    right_col = left_col + left + FRAME_GAP + mid_x
    coords = {
        'TL': (top_row, left_col), 'TR': (top_row, right_col),
        'BL': (bot_row, left_col), 'BR': (bot_row, right_col),
    }
    return [{'r': coords[s][0], 'c': coords[s][1], 'size': sizes[s]} for s in present]


def _paint_pattern(grid: np.ndarray, r: int, c: int, kind: str,
                   pattern: list[tuple[int, int]], color: int) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    """Give every corner of every frame an independently sampled non-bg color."""
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame['corner_colors'] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]] | None:
    """Choose 1..N_SOURCES_MAX source corners from frames larger than the minimum size."""
    eligible = [(fi, kind) for fi, f in enumerate(frames)
                if f['size'] >= SOURCE_MIN_SIZE for kind in CORNER_KINDS]
    if not eligible:
        return None
    n = random.randint(1, min(N_SOURCES_MAX, len(eligible)))
    return set(random.sample(eligible, n))


def _render_output(base: np.ndarray, frames: list[dict],
                   pattern: list[tuple[int, int]]) -> np.ndarray:
    """Paint the canonical pattern at every corner using that corner's color."""
    out = base.copy()
    for frame in frames:
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            _paint_pattern(out, r, c, kind, pattern, frame['corner_colors'][kind])
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict],
                  pattern: list[tuple[int, int]], marker_canon: tuple[int, int],
                  sources: set[tuple[int, str]]) -> np.ndarray:
    """Erase every non-source corner's pattern back to base, keeping only the marker."""
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            mdr, mdc = _rot(*marker_canon, kind)
            marker = (r + mdr, c + mdc)
            for dr, dc in pattern:
                rr, cc = _rot(dr, dc, kind)
                rr, cc = r + rr, c + cc
                if (rr, cc) != marker:
                    inp[rr, cc] = base[rr, cc]
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, marker_canon = _choose_pattern()

    frames = (_sample_two_frames(H, W) if random.random() < N_FRAMES_2_PROB
              else _sample_three_frames(H, W))
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame['r'], frame['c'], frame['size'])

    _assign_corner_colors(frames, bg)
    sources = _pick_sources(frames)
    if sources is None:
        return None

    output = _render_output(base, frames, pattern)
    inp = _derive_input(output, base, frames, pattern, marker_canon, sources)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f'failed to generate a valid input after {MAX_ATTEMPTS} attempts')
