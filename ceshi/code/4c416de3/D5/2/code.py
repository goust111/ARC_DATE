# Variant of 4c416de3
# Dimension: D5 — 轨迹：源角点保留策略 (trajectory)
# Change: 源角点的完整图案在输出中保留不变（keep source）... → 源图案缩小为标记像素
#
# Design analysis for this D5 variant
# -----------------------------------
# a) Concrete behavior:
#    - Input keeps the original asymmetry: source corners show the full corner glyph,
#      target corners show only one marker at the deepest inward-diagonal cell.
#    - Output swaps the two roles:
#      * every former target marker grows into the full glyph in its own color;
#      * every former source glyph shrinks back to only its deepest marker pixel.
#    - Small trace example in canonical TL offsets:
#      pattern = {(-1,0), (0,-1), (0,0), (1,1), (2,2)}, marker = (2,2).
#      Original task: input source stays full in output; target marker expands.
#      D5 variant: that same source loses {-1,0), (0,-1), (0,0), (1,1)} in output
#      and keeps only (2,2), while target still expands. The visible difference is
#      that completed corners move from the "unknown" corners to the "known" ones.
# b) Complexity calibration:
#    - The pattern family stays shared across the whole grid, diagonally symmetric,
#      and limited to 4-6 cells. That keeps the task learnable.
#    - We do not add extra marker types or color remappings; the only new behavior is
#      the reverse motion at source corners, which is easy to compare across samples.
# c) Implicit-assumption audit:
#    - Original source detection depends on source inputs having outward colored cells.
#      This still holds because only the OUTPUT changes; INPUT sources remain full.
#    - Original painting logic assumed source cells were left untouched. That no longer
#      holds, so the solver must actively erase source cells back to their base layer.
#    - Erasing needs a base-color model. The generator keeps the original restricted
#      pattern family: (0,0) may sit on the 0-frame corner, outward cells lie on bg,
#      inward-diagonal cells lie inside bg. The solver asserts this shape discipline.
# d) Placement legality:
#    - Frames remain separated by >= FRAME_GAP and patterns stay within radius 3 around
#      their corners, so source-collapse and target-expansion cannot collide.
#    - Source markers are always the deepest diagonal cell already present in the shared
#      pattern, so shrinking never creates an off-pattern artifact.
# e) Visual observability:
#    - Across 2-4 examples, humans can see a stable swap: corners that start full end
#      with a lone diagonal marker, while corners that start with a lone marker end full.
#      No hidden ordering or color-id arithmetic is used.
# f) Visual neatness:
#    - Only one glyph lives near each frame corner, and the frame gap remains large
#      enough that local corner neighborhoods stay readable.
# g) Color-rule inferability:
#    - Colors are unchanged locally: a corner's output uses that same corner's own color.
#      The variant changes shape trajectory only, not color semantics.
# h) New solver assumptions to guarantee:
#    - Canonical pattern cells must be from: outward cells, the corner cell (0,0), and
#      inward-diagonal cells (k,k). No edge-interior cells are allowed.
#    - The generator enforces exactly that by sampling only from the original templates.

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


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
    if kind == 'TL':
        return (dr, dc)
    if kind == 'TR':
        return (dr, -dc)
    if kind == 'BL':
        return (-dr, dc)
    if kind == 'BR':
        return (-dr, -dc)
    raise ValueError(f'bad corner kind: {kind}')


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
                for odr, odc in OUT_DIRS[kind]:
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


def _paint_pattern(grid: np.ndarray, r: int, c: int, kind: str,
                   pattern: list[tuple[int, int]] | frozenset[tuple[int, int]],
                   color: int) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _paint_marker(grid: np.ndarray, r: int, c: int, kind: str,
                  marker_canon: tuple[int, int], color: int) -> None:
    rr, cc = _rot(*marker_canon, kind)
    grid[r + rr, c + cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the corner roles: sources contract to markers while marked targets expand.

    Input structure:
    - bg is the most frequent color. Zero-valued hollow frames define the detectable
      corners; each corner cell may itself be painted over by a source glyph.
    - At least one source corner still carries the full one-color pattern, including
      outward cells plus one or more inward-diagonal cells.
    - Every other participating corner shows only one colored marker at the pattern's
      deepest inward-diagonal anchor. That color is the corner's own output color.

    Rule:
    - Recover the shared canonical pattern from the full source corner(s).
    - Expand each marker-only corner into that rotated pattern using its own color.
    - Simultaneously contract each full source corner back down to only the deepest
      marker pixel of that same pattern, keeping the source's own color.

    Key insight:
    - The input still reveals the template through outward source cells, but the
      output is no longer "copy only". It is a role swap: source corners lose all
      non-marker pattern cells, while target corners gain them.
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
            sources.append(c_key)
    assert sources, 'at least one source corner (pre-drawn outward pattern) required'

    def _extract(c_key):
        items = per_corner[c_key]
        pat_col = Counter(v for (_, v) in items).most_common(1)[0][0]
        return frozenset(o for (o, v) in items if v == pat_col), pat_col

    canon, source_color0 = _extract(sources[0])
    source_colors = {sources[0]: source_color0}
    for s in sources[1:]:
        other, scol = _extract(s)
        source_colors[s] = scol
        assert other == canon, f'source patterns disagree: {sorted(canon)} vs {sorted(other)}'

    for dr, dc in canon:
        assert (dc, dr) in canon, (
            f'pattern not diagonally symmetric: ({dr},{dc}) has no mirror ({dc},{dr})'
        )
        assert (
            (dr, dc) == (0, 0) or dr < 0 or dc < 0 or (dr == dc and dr > 0)
        ), f'unsupported source-collapse offset in canon: {(dr, dc)}'

    diag_ks = [dr for (dr, dc) in canon if dr == dc and dr > 0]
    assert diag_ks, 'pattern has no inward-diagonal cell to anchor markers'
    marker_k = max(diag_ks)
    marker_canon = (marker_k, marker_k)
    assert marker_canon in canon, 'deepest diagonal marker missing from canon'

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
        for off, color in per_corner[c_key]:
            if off in canon:
                assert color == tcol, (
                    f'target ({r},{c},{kind}) has conflicting color {color} at '
                    f'canonical offset {off}; expected {tcol}'
                )
        targets.append((r, c, kind, tcol))

    assert targets, 'no target corners detected (every corner appears to be a source)'

    out = g.copy()

    for r, c, kind in sources:
        scol = source_colors[(r, c, kind)]
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if (cdr, cdc) == marker_canon:
                out[rr, cc] = scol
            elif (cdr, cdc) == (0, 0):
                out[rr, cc] = 0
            else:
                out[rr, cc] = bg

    for r, c, kind, tcol in targets:
        for cdr, cdc in canon:
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


# --------------- generator (output-first role-swapping replication) ---------------

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
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (0, 0)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
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


def _choose_pattern() -> tuple[list[tuple[int, int]], int]:
    """Return (pattern cells, k_max = deepest inward-diagonal offset)."""
    pattern = list(random.choice(PATTERN_TEMPLATES))
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


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
        'TL': (top_row, left_col),
        'TR': (top_row, right_col),
        'BL': (bot_row, left_col),
        'BR': (bot_row, right_col),
    }
    return [{'r': coords[s][0], 'c': coords[s][1], 'size': sizes[s]} for s in present]


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
                   pattern: list[tuple[int, int]], marker_canon: tuple[int, int],
                   sources: set[tuple[int, str]]) -> np.ndarray:
    """Output already shows the role swap: sources collapse to markers, targets stay full."""
    out = base.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            color = frame['corner_colors'][kind]
            if (fi, kind) in sources:
                _paint_marker(out, r, c, kind, marker_canon, color)
            else:
                _paint_pattern(out, r, c, kind, pattern, color)
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict],
                  pattern: list[tuple[int, int]], marker_canon: tuple[int, int],
                  sources: set[tuple[int, str]]) -> np.ndarray:
    """Invert the role swap: expand sources; erase targets back to lone markers."""
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            color = frame['corner_colors'][kind]
            mdr, mdc = _rot(*marker_canon, kind)
            marker = (r + mdr, c + mdc)
            if (fi, kind) in sources:
                _paint_pattern(inp, r, c, kind, pattern, color)
                continue
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
    pattern, k_max = _choose_pattern()
    marker_canon = (k_max, k_max)

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

    output = _render_output(base, frames, pattern, marker_canon, sources)
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
