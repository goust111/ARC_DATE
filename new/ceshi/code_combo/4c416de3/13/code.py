# Combo variant of 4c416de3 (index 13)
# combo_spec: D9/1+D14/0
# Name: 链式传播+重叠覆盖规则

import random

import numpy as np

from arc_utils import draw_line, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# concretization_and_trace:
# - We keep the base task's object vocabulary: hollow 0-frames on a solid bg,
#   one canonical corner glyph shared across the grid, one fully drawn source,
#   and marker-only targets.
# - The D9 change makes the target processing order stateful: from the current
#   source corner, jump to the unique nearest remaining target, then continue
#   from there.
# - The D14 change makes that order observable: later landings may repaint cells
#   already colored by earlier landings, and the later color wins.
# - A concrete trace used by the generator:
#   1. Source S is on one frame. Two target markers A and B sit on adjacent
#      corners of a 5x5 frame, so their full glyphs overlap on two cells.
#   2. Because S is geometrically closer to A than to B, the chain must visit A
#      first.
#   3. A paints its whole glyph in color a.
#   4. B is the only remaining target, so the chain continues to B, which paints
#      its whole glyph in color b and overwrites the A/B overlap cells.
# - Difference from the original task:
#   original: every target would simply get its own glyph independently, so this
#   overlapping layout would be illegal or ambiguous.
#   combo: the overlapping layout is intentional, and the nearest-target order
#   decides which target color survives on the shared cells.
#
# complexity_calibration:
# - If overlaps were absent, the chain would collapse back into an order-free
#   repaint and D9 would be visually inert. The generator therefore enforces a
#   positive target-target overlap count in every sample.
# - If both target markers landed on the same input cell, the puzzle would
#   become unreadable. To avoid that, we switch to k_max=1 markers and place the
#   extra overlap on off-diagonal inward cells instead.
# - We use exactly one source and exactly two active targets. That keeps the
#   chain easy to infer while still making "A then B" meaningfully different
#   from "B then A".
#
# implicit_assumption_audit:
# - Base solver assumption "targets can be solved independently" is false now.
#   main() must iteratively choose the next target from the current source
#   position.
# - Base solver assumption "paint conflicts are errors" is false now. A later
#   target may legitimately overwrite an earlier target or even part of the
#   initial source.
# - Source detection still relies on outward colored cells. The generator keeps
#   those outward cells exclusive to the initial source in the input.
# - Nearest-target ties would make the order ambiguous. The generator uses only
#   layouts where the first hop is strictly closer than the alternative.
#
# placement_legality:
# - Both frames are fully inside the grid, and every active glyph footprint is
#   fully in bounds.
# - The source frame and target frame are separated so that the input source
#   glyph does not cover either target marker.
# - The two target markers are distinct input cells, while their full output
#   glyphs overlap on two cells.
#
# visual_observability_and_tidiness:
# - The target pair always lives on adjacent corners of the same 5x5 frame, so
#   the overlap is local and easy to inspect.
# - Source, A, and B use distinct colors to make overwrite cells visible.
# - Inert corners stay as plain 0-frame corners; there is no extra clutter.
#
# new_main_assumptions_checked_against_generator:
# - exactly one fully drawn source corner in the input
# - exactly two marker-only targets, with distinct marker cells
# - the first hop is uniquely nearest
# - the second target overwrites at least one earlier target-painted cell
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

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}
_NEIGH_R = 3


H_RANGE = (18, 24)
W_RANGE = (18, 24)
FRAME_GAP = 2
TARGET_FRAME_SIZE = 5
SOURCE_FRAME_SIZE_RANGE = (6, 8)
MAX_ATTEMPTS = 500

# k_max is always 1 so the two target markers stay distinct in the input.
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1)],
)


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
    scored = [(_manhattan(curr_corner, t[:2]), t) for t in remaining]
    scored.sort(key=lambda x: x[0])
    assert scored, "no remaining targets"
    best_d = scored[0][0]
    assert len(scored) == 1 or scored[1][0] != best_d, (
        f"nearest-target tie from {curr_corner}: {[t for d, t in scored if d == best_d]}"
    )
    return scored[0][1]


def _corner_coord(frame, kind):
    dr, dc = _CORNER_OFFSETS[kind]
    span = frame["size"] - 1
    return frame["r"] + dr * span, frame["c"] + dc * span


def _draw_frame(grid, r, c, size):
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _pattern_positions(r, c, kind, pattern):
    pts = []
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        pts.append((r + rr, c + cc))
    return pts


def _pattern_set(r, c, kind, pattern):
    return set(_pattern_positions(r, c, kind, pattern))


def _marker_pos(r, c, kind, marker_k):
    rr, cc = _rot(marker_k, marker_k, kind)
    return r + rr, c + cc


def _paint_pattern(grid, r, c, kind, pattern, color):
    for rr, cc in _pattern_positions(r, c, kind, pattern):
        grid[rr, cc] = int(color)


def _render_chain(base, steps, pattern):
    out = base.copy()
    owner = {}
    target_overwrites = 0
    for step_idx, (r, c, kind, color) in enumerate(steps):
        for rr, cc in _pattern_positions(r, c, kind, pattern):
            prev = owner.get((rr, cc))
            if prev is not None and prev[0] >= 1 and step_idx >= 1 and prev[1] != int(color):
                target_overwrites += 1
            out[rr, cc] = int(color)
            owner[(rr, cc)] = (step_idx, int(color))
    return out, target_overwrites


def _choose_pattern():
    pattern = list(random.choice(PATTERN_TEMPLATES))
    marker_k = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    assert marker_k == 1
    return pattern, marker_k


def _place_frames(H, W, source_size):
    layouts = []

    need_w = source_size + TARGET_FRAME_SIZE + FRAME_GAP + 3
    if W >= need_w and H >= max(source_size, TARGET_FRAME_SIZE) + 3:
        layouts.append("horizontal")
    need_h = source_size + TARGET_FRAME_SIZE + FRAME_GAP + 3
    if H >= need_h and W >= max(source_size, TARGET_FRAME_SIZE) + 3:
        layouts.append("vertical")
    if not layouts:
        return None

    layout = random.choice(layouts)
    if layout == "horizontal":
        before, between, _after = _split_total(W - need_w, 3)
        row = random.randint(1, H - max(source_size, TARGET_FRAME_SIZE) - 2)
        left_is_source = random.random() < 0.5
        left_c = 1 + before
        right_c = left_c + (source_size if left_is_source else TARGET_FRAME_SIZE) + FRAME_GAP + between
        if left_is_source:
            source = {"r": row, "c": left_c, "size": source_size}
            target = {"r": row, "c": right_c, "size": TARGET_FRAME_SIZE}
        else:
            target = {"r": row, "c": left_c, "size": TARGET_FRAME_SIZE}
            source = {"r": row, "c": right_c, "size": source_size}
        return source, target, layout, left_is_source

    before, between, _after = _split_total(H - need_h, 3)
    col = random.randint(1, W - max(source_size, TARGET_FRAME_SIZE) - 2)
    top_is_source = random.random() < 0.5
    top_r = 1 + before
    bot_r = top_r + (source_size if top_is_source else TARGET_FRAME_SIZE) + FRAME_GAP + between
    if top_is_source:
        source = {"r": top_r, "c": col, "size": source_size}
        target = {"r": bot_r, "c": col, "size": TARGET_FRAME_SIZE}
    else:
        target = {"r": top_r, "c": col, "size": TARGET_FRAME_SIZE}
        source = {"r": bot_r, "c": col, "size": source_size}
    return source, target, layout, top_is_source


def _split_total(total, parts):
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_chain(layout, source_first_flag):
    if layout == "horizontal":
        if source_first_flag:
            if random.random() < 0.5:
                return "TR", "TL", "BL"
            return "BR", "BL", "TL"
        if random.random() < 0.5:
            return "TL", "TR", "BR"
        return "BL", "BR", "TR"

    if source_first_flag:
        if random.random() < 0.5:
            return "BL", "TL", "TR"
        return "BR", "TR", "TL"
    if random.random() < 0.5:
        return "TL", "BL", "BR"
    return "TR", "BR", "BL"


def _build_instance():
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, marker_k = _choose_pattern()
    source_size = random.randint(*SOURCE_FRAME_SIZE_RANGE)

    placed = _place_frames(H, W, source_size)
    if placed is None:
        return None
    source_frame, target_frame, layout, source_first_flag = placed
    source_kind, first_target_kind, second_target_kind = _choose_chain(layout, source_first_flag)

    base = np.full((H, W), bg, dtype=int)
    _draw_frame(base, source_frame["r"], source_frame["c"], source_frame["size"])
    _draw_frame(base, target_frame["r"], target_frame["c"], target_frame["size"])

    source_rc = _corner_coord(source_frame, source_kind)
    first_rc = _corner_coord(target_frame, first_target_kind)
    second_rc = _corner_coord(target_frame, second_target_kind)

    source_steps = [(source_rc[0], source_rc[1], source_kind)]
    target_steps = [
        (first_rc[0], first_rc[1], first_target_kind),
        (second_rc[0], second_rc[1], second_target_kind),
    ]

    all_pts = []
    for r, c, kind in source_steps + target_steps:
        pts = _pattern_positions(r, c, kind, pattern)
        if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in pts):
            return None
        all_pts.append(set(pts))

    first_marker = _marker_pos(first_rc[0], first_rc[1], first_target_kind, marker_k)
    second_marker = _marker_pos(second_rc[0], second_rc[1], second_target_kind, marker_k)
    if first_marker == second_marker:
        return None
    if first_marker in all_pts[0] or second_marker in all_pts[0]:
        return None

    overlap = all_pts[1] & all_pts[2]
    if len(overlap) < 2:
        return None

    source_dist = _manhattan(source_rc, first_rc)
    alt_dist = _manhattan(source_rc, second_rc)
    if source_dist >= alt_dist:
        return None

    palette = [c for c in range(1, 10) if c != bg]
    source_color, first_color, second_color = random.sample(palette, 3)

    steps = [
        (source_rc[0], source_rc[1], source_kind, source_color),
        (first_rc[0], first_rc[1], first_target_kind, first_color),
        (second_rc[0], second_rc[1], second_target_kind, second_color),
    ]
    output, target_overwrites = _render_chain(base, steps, pattern)
    if target_overwrites <= 0:
        return None

    reverse_output, _ = _render_chain(
        base,
        [
            steps[0],
            steps[2],
            steps[1],
        ],
        pattern,
    )
    reverse_diff = int(np.count_nonzero(output != reverse_output))
    if reverse_diff <= 0:
        return None

    inp = base.copy()
    _paint_pattern(inp, source_rc[0], source_rc[1], source_kind, pattern, source_color)
    inp[first_marker] = first_color
    inp[second_marker] = second_color

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None

    return {
        "input": inp,
        "output": output,
        "overwrite_cells": int(target_overwrites),
        "reverse_diff": reverse_diff,
        "distance_margin": int(alt_dist - source_dist),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate one corner glyph through the nearest targets, letting later hops overwrite earlier paint.

    Input structure:
    - The background is the most frequent color. Hollow 0-frames sit on that bg,
      and each detectable corner is either inert, a full source, or a marker-only target.
    - Exactly one corner is the initial source: it already contains the complete
      rotated glyph in a single non-bg color, including outward cells.
    - Each active target shows only one colored marker on the glyph's deepest
      inward diagonal cell. Those marker colors are the colors to use when the
      chain reaches those targets.

    Rule:
    - Read the canonical TL-oriented glyph from the source and find the marker offset.
    - Starting at the source corner, repeatedly jump to the unique nearest
      remaining target by Manhattan distance between corner cells.
    - At each landing, paint the whole rotated glyph in that target's marker
      color. If this new glyph overlaps earlier painted cells, the new color overwrites them.

    Key insight:
    - The propagated state is the source position, not the source color. The
      nearest-target order matters only because later targets are allowed to
      repaint shared cells, so overlap turns the chain order into visible output.
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
            sources.append((c_key, next(iter(colors))))
    assert len(sources) == 1, f"expected exactly one source, got {len(sources)}"
    source_key, source_color = sources[0]

    canon = frozenset(off for off, color in per_corner[source_key] if color == source_color)
    assert len(canon) >= 4, "source pattern too small"
    for dr, dc in canon:
        assert (dc, dr) in canon, (
            f"pattern not diagonally symmetric: ({dr},{dc}) has no mirror ({dc},{dr})"
        )

    diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal marker cell"
    marker_k = max(diag_ks)
    marker_canon = (marker_k, marker_k)

    targets = []
    for c_key in corners:
        if c_key == source_key:
            continue
        r, c, kind = c_key
        mr, mc = _marker_pos(r, c, kind, marker_k)
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
    remaining = list(targets)
    while remaining:
        target = _next_unique_nearest(curr_corner, remaining)
        r, c, kind, tcol = target
        for rr, cc in _pattern_positions(r, c, kind, canon):
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(tcol)
        curr_corner = (r, c)
        remaining.remove(target)

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inst = _build_instance()
        except Exception:
            inst = None
        if inst is not None:
            return inst["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
