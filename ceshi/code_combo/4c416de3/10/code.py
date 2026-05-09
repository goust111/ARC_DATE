# Combo variant of 4c416de3 (index 10)
# combo_spec: D5/2+D7/2+D8/1
# Name: 源缩为标记+单框架+极简图案
#
# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior:
#    - The original task uses 2-3 separate frames and medium-size corner glyphs.
#      Here the whole scene is reduced to one large hollow square whose four corners
#      all participate in the same transformation.
#    - Input keeps the asymmetric cue needed for inference:
#      * source corners show the full minimal glyph;
#      * target corners show only the deepest inward-diagonal marker pixel.
#    - Output performs the D5 role swap on that single frame:
#      * each target marker expands into the full rotated glyph in its own color;
#      * each source glyph contracts back to only its marker.
#    - Concrete trace in canonical TL offsets:
#      pattern = {(-1,0), (0,-1), (2,2)}, marker = (2,2).
#      Original task output would keep the source full and only grow targets.
#      This combo instead erases the source's two outward pixels and leaves only the
#      marker, producing a visible source/target role inversion on one shared frame.
# b) Complexity calibration:
#    - "极简图案" could nominally allow 2-3 pixels, but 2 pixels would destroy the
#      original task's key contrast: a source must still reveal outward structure and
#      a target must still have an inward marker. To keep the rule observable and
#      non-degenerate, the generator restricts to 3-pixel templates only.
#    - We still keep a small freedom: the marker depth is either (1,1) or (2,2).
#      That preserves the minimal feel without collapsing every example into the
#      exact same local spacing.
# c) Implicit-assumption audit:
#    - Source detection in the original solver relies on >=2 outward colored cells.
#      This remains valid because input sources still contain the full glyph.
#    - The original solver tolerated multiple frames and inert detected corners.
#      This combo removes that ambiguity: one frame, exactly four detected corners,
#      all of them active as either sources or targets.
#    - Source-collapse now assumes every non-marker source pixel sits on background,
#      because the output must erase it cleanly. Therefore the canonical glyph may
#      only contain two outward cells plus one inward-diagonal marker; (0,0) is
#      disallowed in this combo and the solver asserts that structure.
# d) Placement legality:
#    - The generator uses one large square frame with >=2 background cells of margin
#      to the grid border, so the outward pixels are never clipped.
#    - The frame side is chosen large enough that a radius-3 neighborhood around one
#      corner never reaches another corner, so local corner analysis stays isolated.
# e) Visual observability:
#    - In 2-4 examples, a human can see one stable pattern: within the same big
#      frame, some corners go from full tiny glyphs to lone markers, while the other
#      corners do the opposite. No hidden ordering, counting, or color arithmetic is
#      used.
# f) Visual neatness:
#    - Only one frame is present, all corner neighborhoods are widely separated, and
#      each neighborhood contains at most three painted cells. This keeps the grid
#      sparse and easy to parse despite the role swap.
# g) Color-rule inferability:
#    - Each corner keeps its own local color in both input and output. Distinct corner
#      colors make it easy to see that the transformation copies shape only, not color
#      from source to target.
# h) Generator guarantees for solver assumptions:
#    - Exactly one frame is drawn, and all four corners participate.
#    - There is at least one source and at least one target.
#    - The shared canonical glyph has exactly three cells: two outward cells that are
#      diagonal mirrors of each other, and one inward-diagonal marker at depth 1 or 2.

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


CORNER_KINDS = ('TL', 'TR', 'BL', 'BR')

EDGE_DIRS = {
    'TL': ((+1, 0), (0, +1)),
    'TR': ((+1, 0), (0, -1)),
    'BL': ((-1, 0), (0, +1)),
    'BR': ((-1, 0), (0, -1)),
}
OUT_DIRS = {
    'TL': ((-1, 0), (0, -1)),
    'TR': ((-1, 0), (0, +1)),
    'BL': ((+1, 0), (0, -1)),
    'BR': ((+1, 0), (0, +1)),
}

_NEIGH_R = 3
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (1, 1)],
    [(-1, 0), (0, -1), (2, 2)],
)
_CORNER_OFFSETS = {'TL': (0, 0), 'TR': (0, 1), 'BL': (1, 0), 'BR': (1, 1)}


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical TL-oriented offset; self-inverse."""
    if kind == 'TL':
        return (dr, dc)
    if kind == 'TR':
        return (dr, -dc)
    if kind == 'BL':
        return (-dr, dc)
    if kind == 'BR':
        return (-dr, -dc)
    raise ValueError(f'bad corner kind: {kind}')


def _edge_metric(g, r, c, dr, dc, bg, H, W, max_step=20):
    """Walk from a corner candidate along one inward edge."""
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


def _single_frame_layout(corners):
    """Return a rectangular frame layout if the detected corners form one frame."""
    if len(corners) != 4:
        return None

    by_kind = {}
    for r, c, kind in corners:
        if kind in by_kind:
            return None
        by_kind[kind] = (r, c)
    if set(by_kind) != set(CORNER_KINDS):
        return None

    top, left = by_kind['TL']
    tr_r, right = by_kind['TR']
    bottom, bl_c = by_kind['BL']
    br_r, br_c = by_kind['BR']
    if tr_r != top or bl_c != left or br_r != bottom or br_c != right:
        return None
    if not (top < bottom and left < right):
        return None

    return {
        'top': top,
        'left': left,
        'bottom': bottom,
        'right': right,
        'height': bottom - top + 1,
        'width': right - left + 1,
    }


def _paint_pattern(grid, r, c, kind, pattern, color):
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _paint_marker(grid, r, c, kind, marker_canon, color):
    rr, cc = _rot(*marker_canon, kind)
    grid[r + rr, c + cc] = color


def _corner_coord(frame, kind):
    dr, dc = _CORNER_OFFSETS[kind]
    s1 = frame['size'] - 1
    return frame['r'] + dr * s1, frame['c'] + dc * s1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap roles on a single large frame: full source glyphs shrink to markers while marker-only targets expand.

    Input structure:
    - The background is the most frequent color, and there is exactly one large hollow
      0-frame. Its four corners are the only participating sites.
    - Source corners already show the full tiny one-color glyph: two outward pixels
      plus one inward-diagonal marker pixel.
    - Target corners show only that inward-diagonal marker pixel, whose color is the
      final color for that corner.

    Rule:
    - Detect the four frame corners and recover the shared canonical glyph from the
      full source corner(s).
    - Expand each marker-only target into the same rotated glyph using its own color.
    - Contract each full source back to only its deepest marker pixel in its own color.

    Key insight:
    - The scene is simplified to one frame and a 3-pixel glyph, so the whole task is
      a visible role exchange: corners that reveal the template in the input become
      lone markers in the output, while lone markers become full rotated glyphs.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert len(corners) == 4, f'expected 4 corners from one frame, got {len(corners)}'
    frame_layout = _single_frame_layout(corners)
    assert frame_layout is not None, 'detected corners do not form one shared frame'

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}

    def _is_outward(off):
        return off[0] < 0 or off[1] < 0

    sources = []
    for c_key, items in per_corner.items():
        outward = [(off, v) for off, v in items if _is_outward(off)]
        if len(outward) >= 2:
            colors = {v for _, v in outward}
            assert len(colors) == 1, f'source {c_key} has mixed outward colors: {colors}'
            sources.append(c_key)

    assert 0 < len(sources) < len(corners), 'need both sources and targets on the single frame'

    def _extract(c_key):
        items = per_corner[c_key]
        pat_col = Counter(v for _, v in items).most_common(1)[0][0]
        return frozenset(off for off, v in items if v == pat_col), pat_col

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

    diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
    assert len(canon) == 3, f'expected a 3-pixel minimal glyph, got {sorted(canon)}'
    assert len(diag_ks) == 1, f'expected exactly one inward-diagonal marker, got {sorted(canon)}'
    marker_k = diag_ks[0]
    marker_canon = (marker_k, marker_k)
    outward_cells = canon - {marker_canon}
    assert len(outward_cells) == 2, f'expected two outward cells, got {sorted(canon)}'
    for dr, dc in outward_cells:
        assert dr < 0 or dc < 0, f'non-marker cell must be outward: {(dr, dc)}'

    src_set = set(sources)
    targets = []
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        mdr, mdc = _rot(*marker_canon, kind)
        mr, mc = r + mdr, c + mdc
        assert 0 <= mr < H and 0 <= mc < W, f'marker out of bounds for {c_key}'
        tcol = int(g[mr, mc])
        assert tcol != bg and tcol != 0, f'target {c_key} missing marker'
        for off, color in per_corner[c_key]:
            assert off == marker_canon, (
                f'target {c_key} has extra painted offset {off}; expected only {marker_canon}'
            )
            assert color == tcol, (
                f'target {c_key} has conflicting color {color} at {off}; expected {tcol}'
            )
        targets.append((r, c, kind, tcol))

    out = g.copy()

    for r, c, kind in sources:
        scol = source_colors[(r, c, kind)]
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if (cdr, cdc) == marker_canon:
                out[rr, cc] = scol
            else:
                out[rr, cc] = bg

    for r, c, kind, tcol in targets:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == tcol, (
                f'paint conflict at ({rr},{cc}) for target ({r},{c},{kind}) '
                f'color {tcol}: existing color {v}'
            )
            out[rr, cc] = tcol

    return out


# --------------- generator (single-frame output-first role swap) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 2
FRAME_SIDE_MIN = 8
FRAME_MAX_SLACK = 6
N_SOURCES_CHOICES = (1, 2)
MAX_ATTEMPTS = 300


def _choose_pattern():
    pattern = list(random.choice(PATTERN_TEMPLATES))
    marker_k = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, (marker_k, marker_k)


def _draw_frame(grid, r, c, size):
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _sample_frame(H, W):
    max_side = min(H, W) - 2 * BORDER
    min_side = max(FRAME_SIDE_MIN, max_side - FRAME_MAX_SLACK)
    if min_side > max_side:
        return None
    size = random.randint(min_side, max_side)
    r = random.randint(BORDER, H - size - BORDER)
    c = random.randint(BORDER, W - size - BORDER)
    return {'r': r, 'c': c, 'size': size}


def _assign_corner_colors(frame, bg):
    palette = [color for color in range(1, 10) if color != bg]
    frame['corner_colors'] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _pick_sources():
    n_sources = random.choice(N_SOURCES_CHOICES)
    return set(random.sample(CORNER_KINDS, n_sources))


def _render_output(base, frame, pattern, marker_canon, sources):
    out = base.copy()
    for kind in CORNER_KINDS:
        r, c = _corner_coord(frame, kind)
        color = frame['corner_colors'][kind]
        if kind in sources:
            _paint_marker(out, r, c, kind, marker_canon, color)
        else:
            _paint_pattern(out, r, c, kind, pattern, color)
    return out


def _derive_input(output, base, frame, pattern, marker_canon, sources):
    inp = output.copy()
    for kind in CORNER_KINDS:
        r, c = _corner_coord(frame, kind)
        color = frame['corner_colors'][kind]
        rr, cc = _rot(*marker_canon, kind)
        marker = (r + rr, c + cc)
        if kind in sources:
            _paint_pattern(inp, r, c, kind, pattern, color)
            continue
        for dr, dc in pattern:
            adr, adc = _rot(dr, dc, kind)
            pos = (r + adr, c + adc)
            if pos != marker:
                inp[pos] = base[pos]
    return inp


def _gen_attempt():
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, marker_canon = _choose_pattern()

    frame = _sample_frame(H, W)
    if frame is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    _draw_frame(base, frame['r'], frame['c'], frame['size'])

    _assign_corner_colors(frame, bg)
    sources = _pick_sources()

    output = _render_output(base, frame, pattern, marker_canon, sources)
    inp = _derive_input(output, base, frame, pattern, marker_canon, sources)

    corners = _find_corners(inp, bg)
    if _single_frame_layout(corners) is None:
        return None
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f'failed to generate a valid input after {MAX_ATTEMPTS} attempts')
