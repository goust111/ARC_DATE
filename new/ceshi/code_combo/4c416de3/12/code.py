# Combo variant of 4c416de3 (index 12)
# combo_spec: D2/0+D3/3
# Name: 非对称精灵+拼图式嵌合

import random

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   preserved_backbone:
#     - The grid is still a solid bg color with 2-3 hollow 0-frames.
#     - Corners are still normalized through TL/TR/BL/BR orientation, and at least
#       one corner shows the full transferable source sprite.
#   fused_changes:
#     - D2/0 removes the original diagonal-symmetry guarantee. The source is now a
#       genuinely asymmetric sprite, e.g. {(-1,0),(0,-1),(0,0),(1,1),(2,2),(1,2)},
#       so the solver must preserve the exact offsets rather than relying on mirror
#       pairs across dr==dc.
#     - D3/3 replaces the single marker pixel with a small filled cavity already
#       present inside the target corner. The cavity is the inward-facing connected
#       core of the source sprite: all canonical cells with dr>=0 and dc>=0.
#   behavior_trace:
#     - Seed task, same example canon: a TR target would keep only the deepest
#       marker (2,-2) in its own color, and the solver would repaint the rest.
#     - Combo task: that TR target instead keeps the whole rotated cavity
#       {(0,0),(1,-1),(2,-2),(1,-2)} in one color. The solver first learns the full
#       asymmetric sprite from a source corner, derives that cavity from the source,
#       verifies that each target's prefilled patch matches it exactly, then adds
#       only the missing source cells such as (-1,0) and (0,1).
# complexity_calibration:
#   - Sprite templates stay small (5-6 cells, |dr|,|dc|<=2) so orientation remains
#     readable, but the cavity keeps 3-4 cells, which is visually distinct from the
#     seed task's one-pixel marker.
#   - Every cavity contains the corner cell (0,0) and at least one off-diagonal
#     inward cell. This keeps the jigsaw cue obvious and prevents the combo from
#     collapsing back to a plain diagonal marker variant.
# implicit_assumption_audit:
#   old_assumptions_broken:
#     - The original symmetry assertion is invalid and must be removed.
#     - The original target classifier assumed that exactly one inward-diagonal cell
#       survives. That no longer holds because the target may already contain 3-4
#       colored cells.
#   new_solver_dependencies:
#     - Source detection still depends on having >=2 outward colored cells of a
#       single color; target cavities intentionally have no outward cells.
#     - The solver now assumes that non-source colored cells near a corner equal the
#       full canonical cavity, all in one color. The generator enforces this by
#       erasing every non-source sprite cell except the inward cavity.
#     - Corner localization now also assumes that the two immediately inward
#       frame-edge cells next to a true corner stay 0. Templates therefore start
#       their inward cavity at (1,1) rather than using (1,0) or (0,1), which
#       removes duplicate "corner" detections one step along the frame edge.
#     - The cavity is derived deterministically from the source sprite via
#       dr>=0 and dc>=0. Generator templates are chosen so this derived set is a
#       connected multi-cell patch and remains asymmetric enough to be noticeable.
# placement_legality:
#   - Frame size stays 6-8 and sprite offsets stay within [-1,2] or [-1,2]x[-1,2],
#     so opposite corners of the same frame do not overlap even after keeping a
#     multi-cell cavity at every target.
#   - Frames remain separated by >=4 bg cells, which is larger than the corner
#     scan radius, so colored cells from neighboring frames do not pollute a corner's
#     local neighborhood during source/target extraction.
# visual_observability:
#   - Humans now see two linked signals in each example: the source is asymmetric,
#     and each target already contains a shaped interior patch rather than a lone
#     marker. Across multiple corner orientations, the same cavity visibly flips with
#     the frame corner, making the transform inferable from 2-4 examples.
# cleanliness_and_color:
#   - Each corner still uses exactly one non-bg color, copied directly from that
#     target cavity. There is no color arithmetic or hidden ordering.
#   - The input remains uncluttered because only one sprite family exists globally,
#     and the target cavity occupies the same local region where the completed sprite
#     will land.


CORNER_KINDS = ("TL", "TR", "BL", "BR")

EDGE_DIRS = {
    "TL": ((+1, 0), (0, +1)),
    "TR": ((+1, 0), (0, -1)),
    "BL": ((-1, 0), (0, +1)),
    "BR": ((-1, 0), (0, -1)),
}
OUT_DIRS = {
    "TL": ((-1, 0), (0, -1)),
    "TR": ((-1, 0), (0, +1)),
    "BL": ((+1, 0), (0, -1)),
    "BR": ((+1, 0), (0, +1)),
}

_NEIGH_R = 2


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical TL-oriented offset; this map is self-inverse."""
    if kind == "TL":
        return (dr, dc)
    if kind == "TR":
        return (dr, -dc)
    if kind == "BL":
        return (-dr, dc)
    if kind == "BR":
        return (-dr, -dc)
    raise ValueError(f"unknown corner kind: {kind}")


def _edge_metric(g, r, c, dr, dc, bg, H, W, max_step=10):
    """Walk from (r,c) along an inward edge and measure whether a real frame is present."""
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
                # True corners keep the first inward edge cells as 0-frame cells.
                if int(g[r1, c1]) != 0 or int(g[r2, c2]) != 0:
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


def _is_outward(off):
    dr, dc = off
    return dr < 0 or dc < 0


def _cavity_from_canon(canon):
    assert (1, 0) not in canon and (0, 1) not in canon, (
        "sprite may not paint the first inward edge cells"
    )
    cavity = frozenset((dr, dc) for dr, dc in canon if dr >= 0 and dc >= 0)
    assert cavity, "sprite has no inward cavity"
    assert (0, 0) in cavity, "cavity must include the corner cell"
    assert len(cavity) >= 3, "cavity must remain a visible multi-cell patch"
    assert len(cavity) < len(canon), "cavity must be a proper subset of the sprite"
    return cavity


def _extract_sources_and_canon(per_corner):
    sources = []
    source_colors = {}
    for c_key, items in per_corner.items():
        outward = [(off, v) for off, v in items if _is_outward(off)]
        if len(outward) < 2:
            continue
        colors = {v for _, v in outward}
        assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
        sources.append(c_key)
        source_colors[c_key] = next(iter(colors))
    assert sources, "at least one source corner required"

    canon = None
    for c_key in sources:
        scol = source_colors[c_key]
        sprite = frozenset(off for off, v in per_corner[c_key] if v == scol)
        assert sprite, f"source {c_key} has no sprite cells"
        outward = {off for off in sprite if _is_outward(off)}
        assert len(outward) >= 2, f"source {c_key} lacks outward signature"
        if canon is None:
            canon = sprite
        else:
            assert sprite == canon, (
                f"source sprites disagree: {sorted(canon)} vs {sorted(sprite)}"
            )

    assert canon is not None
    cavity = _cavity_from_canon(canon)
    return sources, source_colors, canon, cavity


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each corner cavity by fitting in the full asymmetric source sprite.

    Input structure:
    - The background is the most frequent color. Hollow 0-frames provide detectable
      TL/TR/BL/BR corners, and a corner cell may stay 0 or already be recolored.
    - One or more source corners show the entire sprite in one non-bg color. The
      sprite is asymmetric and contains both outward cells and an inward core.
    - Every other active corner already contains a small colored cavity patch inside
      the frame. That patch is the inward part of the source sprite, rotated into
      the target corner, and its color is the color of the completed target sprite.

    Rule:
    - Normalize colored source cells into canonical TL-oriented offsets and recover
      the exact asymmetric source sprite.
    - Derive the canonical cavity as the subset of source offsets with dr>=0 and
      dc>=0.
    - For each non-source corner whose colored cells match that rotated cavity,
      paint the full canonical sprite in that corner orientation using the cavity's
      own color.

    Key insight:
    - The target patch is no longer a single marker; it is a partial jigsaw socket.
      The solver must infer the exact full sprite from a source corner, then check
      that each target's prefilled interior cells are the rotated inward core of
      that same sprite. Orientation handling comes from coordinate normalization, so
      the asymmetric sprite and its cavity flip consistently across all corners.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}
    sources, _source_colors, canon, cavity = _extract_sources_and_canon(per_corner)
    src_set = set(sources)

    targets = []
    for c_key in corners:
        if c_key in src_set:
            continue
        items = per_corner[c_key]
        assert items, f"non-source corner {c_key} has no cavity cells"
        colors = {v for _, v in items}
        assert len(colors) == 1, f"target {c_key} has mixed cavity colors: {colors}"
        observed = frozenset(off for off, _ in items)
        assert observed == cavity, (
            f"target {c_key} cavity mismatch: {sorted(observed)} vs {sorted(cavity)}"
        )
        r, c, kind = c_key
        targets.append((r, c, kind, next(iter(colors))))

    assert targets, "no target corners detected"

    out = g.copy()
    for r, c, kind, tcol in targets:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == tcol, (
                f"paint conflict at ({rr},{cc}) for target ({r},{c},{kind}) "
                f"color {tcol}: existing color {v}"
            )
            out[rr, cc] = tcol

    return out


# --------------- generator (output-first combo construction) ---------------

# --- canvas & layout ---
H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (6, 8)
SOURCE_MIN_SIZE = 6

# --- structural counts ---
N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2

# --- generation retries ---
MAX_ATTEMPTS = 300

# --- canonical TL-oriented asymmetric sprite templates ---
# The target cavity is derived from each sprite by taking the nonnegative offsets.
# Every template therefore includes (0,0), has >=2 outward cells, and yields a
# multi-cell cavity of size 3-4.
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1), (1, 2)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 1)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (1, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2), (1, 2)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2), (2, 1)],
)

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}


def _split_total(total, parts):
    """Partition total non-negative units into parts bins uniformly at random."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _is_diagonal_symmetric(pattern):
    cells = set(pattern)
    return all((dc, dr) in cells for dr, dc in cells)


def _choose_pattern():
    pattern = list(random.choice(PATTERN_TEMPLATES))
    cavity = _cavity_from_canon(pattern)
    assert not _is_diagonal_symmetric(pattern), "combo pattern unexpectedly symmetric"
    assert any((dc, dr) not in cavity for dr, dc in cavity), "cavity lost asymmetry"
    return pattern, list(cavity)


def _draw_frame(grid, r, c, size):
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _corner_coord(frame, kind):
    dr, dc = _CORNER_OFFSETS[kind]
    s1 = frame["size"] - 1
    return frame["r"] + dr * s1, frame["c"] + dc * s1


def _sample_frame_pair(span):
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


def _sample_two_frames(H, W):
    horizontal = random.random() < TWO_FRAME_HORIZONTAL_PROB
    axis = _sample_frame_pair(W if horizontal else H)
    if axis is None:
        return None
    s1, s2, o1, o2 = axis
    if horizontal:
        row = random.randint(BORDER, H - max(s1, s2) - BORDER)
        return [{"r": row, "c": o1, "size": s1}, {"r": row, "c": o2, "size": s2}]
    col = random.randint(BORDER, W - max(s1, s2) - BORDER)
    return [{"r": o1, "c": col, "size": s1}, {"r": o2, "c": col, "size": s2}]


def _sample_three_frames(H, W):
    lo, hi = FRAME_SIZE_RANGE
    missing = random.choice(CORNER_KINDS)
    present = tuple(kind for kind in CORNER_KINDS if kind != missing)

    def _bands(sizes):
        top = max((sizes[s] for s in present if s[0] == "T"), default=0)
        bot = max((sizes[s] for s in present if s[0] == "B"), default=0)
        left = max((sizes[s] for s in present if s[1] == "L"), default=0)
        right = max((sizes[s] for s in present if s[1] == "R"), default=0)
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
        "TL": (top_row, left_col),
        "TR": (top_row, right_col),
        "BL": (bot_row, left_col),
        "BR": (bot_row, right_col),
    }
    return [{"r": coords[s][0], "c": coords[s][1], "size": sizes[s]} for s in present]


def _paint_offsets(grid, r, c, kind, offsets, color):
    for dr, dc in offsets:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _assign_corner_colors(frames, bg):
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _pick_sources(frames):
    eligible = [
        (fi, kind)
        for fi, frame in enumerate(frames)
        if frame["size"] >= SOURCE_MIN_SIZE
        for kind in CORNER_KINDS
    ]
    if not eligible:
        return None
    n = random.randint(1, min(N_SOURCES_MAX, len(eligible)))
    return set(random.sample(eligible, n))


def _render_output(base, frames, pattern):
    out = base.copy()
    for frame in frames:
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            _paint_offsets(out, r, c, kind, pattern, frame["corner_colors"][kind])
    return out


def _derive_input(output, base, frames, pattern, cavity, sources):
    inp = output.copy()
    cavity_set = set(cavity)
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            for dr, dc in pattern:
                if (dr, dc) in cavity_set:
                    continue
                rr, cc = _rot(dr, dc, kind)
                inp[r + rr, c + cc] = base[r + rr, c + cc]
    return inp


def _gen_attempt():
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, cavity = _choose_pattern()

    frames = (
        _sample_two_frames(H, W)
        if random.random() < N_FRAMES_2_PROB
        else _sample_three_frames(H, W)
    )
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_corner_colors(frames, bg)
    sources = _pick_sources(frames)
    if sources is None:
        return None

    output = _render_output(base, frames, pattern)
    inp = _derive_input(output, base, frames, pattern, cavity, sources)

    try:
        corners = _find_corners(inp, bg)
        per_corner = {c: _colored_around(inp, *c, bg) for c in corners}
        found_sources, _source_colors, canon, found_cavity = _extract_sources_and_canon(per_corner)
        if _is_diagonal_symmetric(canon):
            return None
        if frozenset(found_cavity) != frozenset(cavity):
            return None
        if set(found_sources) != {
            (_corner_coord(frames[fi], kind)[0], _corner_coord(frames[fi], kind)[1], kind)
            for fi, kind in sources
        }:
            return None
    except Exception:
        return None

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input():
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
