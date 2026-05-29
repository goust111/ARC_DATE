# Combo variant of 4c416de3 (index 7)
# combo_spec: D3/0+D8/1
# Name: 部分图案线索+极简图案
#
# Design analysis:
# combo_variant_spec:
#   concretization:
#     pattern:
#       - Canonical corner glyph is shrunk to exactly 3 pixels total.
#       - Each template is diagonally symmetric in TL coordinates and has one
#         positive inward-diagonal anchor, so the task still reads as "copy the
#         same tiny corner glyph into the other corners".
#       - Example template: {(-1,0), (0,-1), (1,1)}.
#     source_corner:
#       - A source still shows the full 3-pixel glyph in one color.
#       - With only one missing pixel separating source from target, the visual
#         difference is intentionally tiny.
#     target_corner:
#       - A target keeps exactly 2 same-colored pixels from that same 3-pixel
#         glyph, not a single marker.
#       - At least one target omits the deepest diagonal pixel, so the input is
#         not solvable by the original "look only at (k_max,k_max)" rule.
#   behavior_trace:
#     original:
#       - Extract a 4-6 pixel source pattern.
#       - Detect targets by a lone deepest-diagonal marker and repaint the full
#         pattern in the marker color.
#     combo:
#       - Extract a 3-pixel source pattern from the corners with the largest
#         visible clue count.
#       - Treat every other active corner as a target when it shows a 2-pixel
#         monochrome subset of that source pattern.
#       - Fill in the missing third pixel using that corner's own color.
#       - Concrete trace: if the canon is {(-1,0),(0,-1),(1,1)}, one BR target
#         may show only {(0,1),(-1,0)} in actual coordinates; the solver must
#         infer that the inward diagonal cell is the missing third pixel.
#   complexity_calibration:
#     - The pattern is forced to 3 pixels, not 2, because a 2-pixel full pattern
#       would make a "multiple clue pixels" target indistinguishable from a full
#       source.
#     - Every non-source corner keeps exactly 2 pixels, guaranteeing that the
#       D3 change is visible in every target while preserving the D8 minimalism.
#     - If there are multiple targets, the generator mixes omission types so the
#       rule reads as "complete the tiny pattern" rather than "always add the
#       same fixed location".
#   implicit_assumptions_review:
#     - Original main assumed a target always has the deepest diagonal marker;
#       this breaks here and is removed.
#     - Original source detection relied on outward cells; with 3-pixel patterns
#       that would be brittle, so source detection is switched to "largest local
#       colored set", which the generator makes reliable by giving sources 3
#       pixels and targets 2 pixels.
#     - The new solver assumes each active corner is monochrome and that no
#       target is already complete; generator construction enforces both.
#     - The new solver also assumes local scans are not polluted by the opposite
#       corner of the same frame, so the generator keeps pattern radius at 1 and
#       frame size at least 6.
#   placement_legality:
#     - Frames remain non-overlapping, with >=4 background cells between frames.
#     - Pattern offsets stay within |dr|,|dc|<=1, so every painted cell remains
#       in-bounds with the existing border margin.
#     - Frame size is restricted to 6-8, preventing opposite-corner bleed into
#       the radius-3 neighborhood scan used by the solver.
#   visual_observability:
#     - Humans can compare one fully drawn 3-pixel source with several 2-pixel
#       near-copies and infer that each target is missing exactly one part of the
#       same tiny rotated glyph.
#     - The forced "missing deepest diagonal" target guarantees at least one
#       example where the new task visibly diverges from the original marker-only
#       behavior.
#   visual_cleanliness:
#     - Each clue stays glued to its own frame corner and uses only one color.
#     - No extra encoding is hidden in color ids, ordering, or counts beyond the
#       plainly visible full-vs-partial distinction.
#   inferability_of_change:
#     - Color behavior is still local and direct: a target's visible clue color
#       is the color of the completed output pattern at that corner.

import random
from itertools import combinations

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


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


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical TL-oriented offset; this transform is self-inverse."""
    if kind == "TL":
        return (dr, dc)
    if kind == "TR":
        return (dr, -dc)
    if kind == "BL":
        return (-dr, dc)
    if kind == "BR":
        return (-dr, -dc)
    raise ValueError(f"unknown corner kind {kind}")


def _edge_metric(g, r, c, dr, dc, bg, H, W, max_step=10):
    """Walk along one edge direction and report (non_bg_count, saw_zero)."""
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


def _mirror(off):
    return (off[1], off[0])


def _canonical_diagonal_depths(pattern):
    return sorted(dr for dr, dc in pattern if dr == dc and dr > 0)


def _active_corner_signature(items):
    if not items:
        return None
    colors = {color for _, color in items}
    assert len(colors) == 1, f"corner has mixed clue colors: {colors}"
    color = next(iter(colors))
    offsets = frozenset(off for off, _ in items)
    return offsets, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete nearly-finished miniature corner glyphs into the shared full pattern.

    Input structure:
    - The background is the most frequent nonzero color, and hollow 0-frames mark
      the objects whose corners matter.
    - One or more source corners already show a full 3-pixel monochrome glyph.
    - Every other active corner shows only a 2-pixel monochrome subset of that
      same glyph, rotated into its corner orientation.

    Rule:
    - Normalize each corner neighborhood into TL-oriented offsets.
    - Use the corners with the largest visible colored set to recover the shared
      canonical glyph.
    - For every smaller monochrome subset corner, paint the missing pixel(s) of
      the same rotated glyph in that corner's own color.

    Key insight:
    - Source and target corners differ by only one pixel, so the task is no
      longer "read a single marker". Instead, you must notice that all targets
      are incomplete copies of the same tiny source glyph and restore the one
      missing piece at each corner.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    signatures = {}
    for c_key in corners:
        sig = _active_corner_signature(_colored_around(g, *c_key, bg))
        if sig is not None:
            signatures[c_key] = sig
    assert signatures, "no colored corner clues detected"

    max_seen = max(len(offsets) for offsets, _ in signatures.values())
    assert max_seen == 3, f"expected a 3-pixel full glyph, got maximum visible size {max_seen}"

    sources = [c_key for c_key, (offsets, _color) in signatures.items() if len(offsets) == max_seen]
    assert sources, "at least one full source corner is required"

    canon, _ = signatures[sources[0]]
    for s in sources[1:]:
        other, _ = signatures[s]
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    assert len(canon) == 3, f"combo requires a 3-pixel canon, got {len(canon)}"
    for off in canon:
        assert _mirror(off) in canon, f"pattern not diagonally symmetric at {off}"
    diag_ks = _canonical_diagonal_depths(canon)
    assert diag_ks, "pattern has no inward diagonal anchor"

    src_set = set(sources)
    targets = []
    for c_key, (visible, color) in signatures.items():
        if c_key in src_set:
            continue
        assert len(visible) == 2, f"target {c_key} should keep exactly 2 clue pixels, got {len(visible)}"
        assert visible.issubset(canon), f"target {c_key} has clue outside canon: {sorted(visible - canon)}"
        targets.append((*c_key, color))

    assert targets, "no target corners detected"

    out = g.copy()
    for r, c, kind, tcol in targets:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == tcol, (
                f"paint conflict at ({rr},{cc}) for target ({r},{c},{kind}) "
                f"color {tcol}: existing color {v}"
            )
            out[rr, cc] = tcol

    return out


# --------------- generator (output-first pattern replication (Codex)) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (6, 8)
SOURCE_MIN_SIZE = 6

N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2

MAX_ATTEMPTS = 300

# Each template is minimal (3 pixels total), diagonally symmetric, and kept
# within radius 1 so same-frame opposite corners never enter one another's local scan.
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (1, 1)],
    [(-1, -1), (0, 0), (1, 1)],
)

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}


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
    pattern = list(random.choice(PATTERN_TEMPLATES))
    k_max = max(_canonical_diagonal_depths(pattern))
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
    s1 = frame["size"] - 1
    return frame["r"] + dr * s1, frame["c"] + dc * s1


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
        return [{"r": row, "c": o1, "size": s1}, {"r": row, "c": o2, "size": s2}]
    col = random.randint(BORDER, W - max(s1, s2) - BORDER)
    return [{"r": o1, "c": col, "size": s1}, {"r": o2, "c": col, "size": s2}]


def _sample_three_frames(H: int, W: int) -> list[dict] | None:
    """Place three frames in three of the four corner slots (TL/TR/BL/BR), one missing."""
    lo, hi = FRAME_SIZE_RANGE
    missing = random.choice(CORNER_KINDS)
    present = tuple(k for k in CORNER_KINDS if k != missing)

    def _bands(sizes: dict[str, int]) -> tuple[int, int, int, int]:
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


def _paint_pattern(
    grid: np.ndarray, r: int, c: int, kind: str, pattern: list[tuple[int, int]], color: int
) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    """Give every corner of every frame an independently sampled non-bg color."""
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]] | None:
    eligible = [
        (fi, kind)
        for fi, f in enumerate(frames)
        if f["size"] >= SOURCE_MIN_SIZE
        for kind in CORNER_KINDS
    ]
    if not eligible:
        return None
    n = random.randint(1, min(N_SOURCES_MAX, len(eligible)))
    return set(random.sample(eligible, n))


def _target_clue_candidates(pattern: list[tuple[int, int]], k_max: int):
    full = tuple(sorted(pattern))
    deepest = (k_max, k_max)
    all_sets = [tuple(subset) for subset in combinations(full, 2)]
    without_deepest = [subset for subset in all_sets if deepest not in subset]
    with_deepest = [subset for subset in all_sets if deepest in subset]
    assert without_deepest and with_deepest, "need both deepest-present and deepest-absent clue subsets"
    return all_sets, with_deepest, without_deepest


def _assign_target_clues(
    frames: list[dict],
    pattern: list[tuple[int, int]],
    k_max: int,
    sources: set[tuple[int, str]],
) -> dict[tuple[int, str], tuple[tuple[int, int], ...]]:
    all_sets, with_deepest, without_deepest = _target_clue_candidates(pattern, k_max)
    targets = [
        (fi, kind)
        for fi, _frame in enumerate(frames)
        for kind in CORNER_KINDS
        if (fi, kind) not in sources
    ]
    assert targets, "need at least one target"

    keep = {}
    force_no_marker = random.choice(targets)
    force_with_marker = None
    if len(targets) >= 2:
        others = [key for key in targets if key != force_no_marker]
        force_with_marker = random.choice(others)

    for key in targets:
        if key == force_no_marker:
            keep[key] = random.choice(without_deepest)
        elif key == force_with_marker:
            keep[key] = random.choice(with_deepest)
        else:
            keep[key] = random.choice(all_sets)
    return keep


def _render_output(base: np.ndarray, frames: list[dict], pattern: list[tuple[int, int]]) -> np.ndarray:
    """Paint the canonical pattern at every corner using that corner's color."""
    out = base.copy()
    for frame in frames:
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            _paint_pattern(out, r, c, kind, pattern, frame["corner_colors"][kind])
    return out


def _derive_input(
    output: np.ndarray,
    base: np.ndarray,
    frames: list[dict],
    pattern: list[tuple[int, int]],
    sources: set[tuple[int, str]],
    target_clues: dict[tuple[int, str], tuple[tuple[int, int], ...]],
) -> np.ndarray:
    """Erase every non-source corner back to a 2-pixel partial clue."""
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            keep = set(target_clues[(fi, kind)])
            for dr, dc in pattern:
                rr, cc = _rot(dr, dc, kind)
                rr, cc = r + rr, c + cc
                if (dr, dc) not in keep:
                    inp[rr, cc] = base[rr, cc]
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()

    frames = _sample_two_frames(H, W) if random.random() < N_FRAMES_2_PROB else _sample_three_frames(H, W)
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_corner_colors(frames, bg)
    sources = _pick_sources(frames)
    if sources is None:
        return None
    target_clues = _assign_target_clues(frames, pattern, k_max, sources)

    output = _render_output(base, frames, pattern)
    inp = _derive_input(output, base, frames, pattern, sources, target_clues)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
