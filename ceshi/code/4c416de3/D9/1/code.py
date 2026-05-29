# Variant of 4c416de3
# Dimension: D9 — 源与目标的基数关系 (multi_step)
# Change: 1-2个源角点对多个目标角点（1:many），所有源共享同一规范图案... → 链式传播（多步序列）

import random

import numpy as np

from arc_utils import draw_line, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the D9 variant
#
# concretization_and_trace:
# - We keep the original visual vocabulary: square 0-frames on a solid bg, a
#   single canonical corner glyph shared by the whole grid, and one colored
#   marker at the deepest inward-diagonal cell of every target corner.
# - The D9 change must be visible in the final output, so the result cannot be
#   the same "paint every target in its own color" broadcast as the base task.
# - Variant rule used here:
#   1. Exactly one initial source corner already carries the full glyph in one
#      color.
#   2. Several target corners show only their deepest diagonal marker.
#   3. Starting from the source, the solver repeatedly picks the unique nearest
#      remaining target, using Manhattan distance between the corner cells.
#   4. When the glyph lands on a target, the outward part of the glyph
#      (canonical dr<0 or dc<0) keeps the previous source color, while the
#      corner/inward part uses the target's own marker color.
#   5. That target's own color becomes the propagation color for the next hop.
# - Small trace:
#   original: source color 3, targets with markers 4 and 7 -> both targets end
#   up fully painted in 4 and 7 independently.
#   variant: source 3 reaches nearest marker 4 first -> target A gets tail 3
#   and head 4; A now propagates color 4 to the next marker 7 -> target B gets
#   tail 4 and head 7. The output records the order locally.
#
# complexity_calibration:
# - A pure nearest-target chain that still produced monochrome targets would
#   collapse to the original output. Splitting each landed glyph into inherited
#   tail + local head makes the multi-step effect readable without introducing a
#   new object type.
# - We keep one source, 2-5 targets, 2-3 frames, and the original small set of
#   canonical glyph templates. That is enough variation to avoid triviality, but
#   not so much that the rule becomes hard to infer from 2-4 examples.
#
# implicit_assumption_audit:
# - Original source detection assumed "outward colored cells => source". That is
#   still true in the input because only the initial source has outward cells.
# - Original target painting loop was order-free; here order matters, so main()
#   cannot gather all targets and paint them independently. It must iteratively
#   choose the next target from the current source position.
# - Original code treated every target color as final paint color. That is no
#   longer valid: target color now controls only the head/inward cells and the
#   next propagation color.
# - Any tie in nearest-target distance would create ambiguity. The generator
#   therefore searches only chains whose next hop is uniquely nearest among the
#   still-active targets.
#
# placement_legality:
# - Frames keep the original non-overlap and border margins.
# - The generator may leave some frame corners inert; only chain corners carry a
#   marker or a full source glyph.
# - Candidate chains are searched over real corner coordinates after frame
#   placement. Rendering rejects any color conflicts caused by overlapping glyph
#   footprints, so generated instances remain solvable by the new main().
#
# visual_observability_and_tidiness:
# - Distinct chain-node colors are sampled so the inherited tail color visibly
#   changes from hop to hop.
# - Only one source is active, which keeps the reading order clean.
# - Inert corners stay as plain 0-frame corners, avoiding clutter.
#
# new_main_assumptions_checked_against_generator:
# - exactly one source corner with a single-color full glyph
# - at least two targets so the process is truly multi-step
# - unique nearest remaining target at every hop
# - no conflicting overlap between glyphs painted by different chain steps
# ---------------------------------------------------------------------------


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

_NEIGH_R = 3
_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical TL-oriented offset; self-inverse."""
    if kind == "TL":
        return (dr, dc)
    if kind == "TR":
        return (dr, -dc)
    if kind == "BL":
        return (-dr, dc)
    if kind == "BR":
        return (-dr, -dc)
    raise ValueError(kind)


def _is_outward(off):
    return off[0] < 0 or off[1] < 0


def _edge_metric(g, r, c, dr, dc, bg, H, W, max_step=10):
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


def _manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _next_unique_nearest(curr_corner, remaining):
    scored = [(_manhattan(curr_corner, t[:2]), idx, t) for idx, t in enumerate(remaining)]
    scored.sort(key=lambda x: x[0])
    assert scored, "no remaining targets"
    best_d = scored[0][0]
    assert len(scored) == 1 or scored[1][0] != best_d, (
        f"nearest-target tie from {curr_corner}: {[t for d, _, t in scored if d == best_d]}"
    )
    return scored[0][2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate one corner glyph through marker corners as a nearest-neighbor chain.

    Input structure:
    - bg is the most frequent color. Several hollow 0-frames sit on the canvas,
      and detectable frame corners may be inert, marked, or already fully drawn.
    - Exactly one corner is the initial source: it already shows the complete
      canonical glyph in a single color, including outward cells.
    - Every active target corner has only one colored marker at the deepest
      inward-diagonal glyph cell; that color is the target's local "head" color.

    Rule:
    - Read the canonical TL-oriented glyph from the source and split it into an
      outward tail part (canonical dr<0 or dc<0) and a corner/inward head part.
    - Starting from the source corner, repeatedly choose the unique nearest
      remaining target by Manhattan distance between corner cells.
    - Paint that target with the rotated glyph: tail cells inherit the current
      source color, head cells use the target marker color. The target's marker
      color then becomes the propagation color for the next hop.

    Key insight:
    - The target's own marker color does not simply fill the whole glyph. It
      colors only the head/inward portion, while the outward tail records the
      previous chain link. The final board therefore encodes the hop order.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"
    per_corner = {c: _colored_around(g, *c, bg) for c in corners}

    sources = []
    for c_key, items in per_corner.items():
        outward = [(off, color) for off, color in items if _is_outward(off)]
        if len(outward) >= 2:
            colors = {color for _, color in outward}
            assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
            src_color = next(iter(colors))
            sources.append((c_key, src_color))
    assert len(sources) == 1, f"expected exactly one source, got {len(sources)}"
    source_key, source_color = sources[0]

    source_items = per_corner[source_key]
    canon = frozenset(off for off, color in source_items if color == source_color)
    assert len(canon) >= 4, "source pattern too small"

    for dr, dc in canon:
        assert (dc, dr) in canon, (
            f"pattern not diagonally symmetric: ({dr},{dc}) has no mirror ({dc},{dr})"
        )

    outward_canon = frozenset(off for off in canon if _is_outward(off))
    head_canon = frozenset(off for off in canon if not _is_outward(off))
    assert len(outward_canon) >= 2, "pattern needs outward tail cells"
    diag_ks = [dr for dr, dc in head_canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal marker cell"
    marker_k = max(diag_ks)
    marker_canon = (marker_k, marker_k)

    targets = []
    for c_key in corners:
        if c_key == source_key:
            continue
        r, c, kind = c_key
        mdr, mdc = _rot(*marker_canon, kind)
        mr, mc = r + mdr, c + mdc
        if not (0 <= mr < H and 0 <= mc < W):
            continue
        v = int(g[mr, mc])
        if v == bg or v == 0:
            continue
        for off, color in per_corner[c_key]:
            if off in canon:
                assert color == v, (
                    f"target {c_key} has conflicting color {color} at {off}; expected {v}"
                )
        targets.append((r, c, kind, v))
    assert len(targets) >= 2, "need at least two targets for chain propagation"

    out = g.copy()
    curr_corner = source_key[:2]
    curr_color = int(source_color)
    remaining = list(targets)

    while remaining:
        target = _next_unique_nearest(curr_corner, remaining)
        r, c, kind, own_color = target
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            paint = curr_color if (cdr, cdc) in outward_canon else own_color
            prev = int(out[rr, cc])
            assert prev == bg or prev == 0 or prev == paint, (
                f"paint conflict at ({rr},{cc}) while landing on {target}: {prev} vs {paint}"
            )
            out[rr, cc] = paint
        curr_corner = (r, c)
        curr_color = int(own_color)
        remaining.remove(target)

    return out


# --------------- generator (output-first chain propagation) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (5, 8)
SOURCE_MIN_SIZE = 6

N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5

MAX_ATTEMPTS = 400

PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (0, 0)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out, prev = [], -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_pattern():
    pattern = list(random.choice(PATTERN_TEMPLATES))
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


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


def _try_paint(grid, r, c, kind, pattern, color_fn, bg):
    pending = []
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        rr, cc = r + rr, c + cc
        if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
            return False
        color = int(color_fn((dr, dc)))
        prev = int(grid[rr, cc])
        if prev != bg and prev != 0 and prev != color:
            return False
        pending.append((rr, cc, color))
    for rr, cc, color in pending:
        grid[rr, cc] = color
    return True


def _pick_chain(frames):
    corners = []
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            corners.append(
                {
                    "fi": fi,
                    "kind": kind,
                    "r": r,
                    "c": c,
                    "source_ok": frame["size"] >= SOURCE_MIN_SIZE,
                }
            )

    source_ids = [i for i, corner in enumerate(corners) if corner["source_ok"]]
    if not source_ids:
        return None

    max_len = min(6, len(corners))
    min_len = min(4, max_len)
    want_lengths = list(range(min_len, max_len + 1))
    random.shuffle(want_lengths)
    random.shuffle(source_ids)

    def dfs(curr_id, pool, seq, want_len):
        if len(seq) == want_len:
            return seq
        options = list(pool)
        random.shuffle(options)
        options.sort(key=lambda nid: _manhattan((corners[curr_id]["r"], corners[curr_id]["c"]), (corners[nid]["r"], corners[nid]["c"])))
        for nid in options:
            d = _manhattan((corners[curr_id]["r"], corners[curr_id]["c"]), (corners[nid]["r"], corners[nid]["c"]))
            next_pool = {
                oid
                for oid in pool
                if oid != nid
                and _manhattan((corners[curr_id]["r"], corners[curr_id]["c"]), (corners[oid]["r"], corners[oid]["c"])) > d
            }
            res = dfs(nid, next_pool, seq + [nid], want_len)
            if res is not None:
                return res
        return None

    for want_len in want_lengths:
        for start in source_ids:
            pool = set(range(len(corners))) - {start}
            seq = dfs(start, pool, [start], want_len)
            if seq is not None and len(seq) >= 4:
                return [corners[i] for i in seq]
    return None


def _build_output(base, pattern, chain, colors, bg):
    out = base.copy()
    outward = {off for off in pattern if _is_outward(off)}

    src = chain[0]
    if not _try_paint(
        out,
        src["r"],
        src["c"],
        src["kind"],
        pattern,
        lambda _off: colors[0],
        bg,
    ):
        return None

    prev_color = colors[0]
    for node, own_color in zip(chain[1:], colors[1:]):
        ok = _try_paint(
            out,
            node["r"],
            node["c"],
            node["kind"],
            pattern,
            lambda off, own=own_color, prev=prev_color: prev if off in outward else own,
            bg,
        )
        if not ok:
            return None
        prev_color = own_color
    return out


def _derive_input(output, base, pattern, chain, colors, k_max):
    inp = output.copy()
    marker = (k_max, k_max)
    for node, own_color in zip(chain[1:], colors[1:]):
        r, c, kind = node["r"], node["c"], node["kind"]
        mdr, mdc = _rot(*marker, kind)
        keep = (r + mdr, c + mdc)
        for dr, dc in pattern:
            rr, cc = _rot(dr, dc, kind)
            rr, cc = r + rr, c + cc
            if (rr, cc) != keep:
                inp[rr, cc] = base[rr, cc]
        inp[keep] = own_color
    return inp


def _gen_attempt():
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()

    frames = _sample_two_frames(H, W) if random.random() < N_FRAMES_2_PROB else _sample_three_frames(H, W)
    if frames is None:
        return None

    chain = _pick_chain(frames)
    if chain is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    palette = [c for c in range(1, 10) if c != bg]
    if len(chain) > len(palette):
        return None
    colors = random.sample(palette, len(chain))

    output = _build_output(base, pattern, chain, colors, bg)
    if output is None:
        return None

    inp = _derive_input(output, base, pattern, chain, colors, k_max)

    solved = main(inp)
    if not np.array_equal(solved, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
