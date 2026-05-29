# Combo variant of 4c416de3 (index 15)
# combo_spec: D4/1+D5/0
# Name: 颜色交换+擦除源
#
# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - Keep the original object vocabulary: 2-3 hollow 0-frames on a solid background, one shared
#      diagonally symmetric corner glyph, and source/target corners around the frames.
#    - Fusion semantics:
#      input source corners show the full glyph in a shared `source_color`;
#      every non-source active corner shows only the deepest inward-diagonal marker in a shared
#      `target_marker_color`.
#      Output expands every target into the full glyph in `source_color`, then the source glyphs are
#      erased back to the frame/background scaffold. So the visible end state contains only targets,
#      all painted in the old source color.
#    - Small trace:
#      suppose canonical offsets are {(-1,0),(0,-1),(0,0),(1,1)}.
#      Original task: source TL blue full glyph stays blue; target BR red marker becomes red glyph.
#      D4/1 alone: BR would become blue, TL would turn red.
#      D5/0 alone: BR would become red, TL would disappear.
#      This combo: BR becomes blue, TL disappears. The marker color participates only as the
#      transferred-to source color before the erase step, so it is absent from the final grid.
# b) Complexity calibration
#    - Plain "swap" with per-target colors would be ambiguous because one source color could not swap
#      consistently with many different target colors. Therefore all targets share one marker color,
#      and all sources share one source color.
#    - We still keep 1-2 sources and many targets, so the transfer is repeated enough to be visible;
#      the task is not simplified into a one-off move.
# c) Hidden assumptions to replace
#    - Original solver assumes each target paints with its own marker color. That no longer holds:
#      target output color must come from the shared source glyph color.
#    - D4/1 alone would recolor sources into the marker color, but D5/0 erases sources afterward.
#      So the solver must not leave any source paint behind in the final grid.
#    - Erasing cannot use one blanket color: canonical offset (0,0) sits on the 0-frame corner,
#      while every other allowed glyph offset must sit on background. The generator must guarantee
#      this base-color invariant.
# d) Placement legality
#    - Frame placement still follows the original non-overlap spacing.
#    - The generator additionally verifies that every rotated glyph cell falls on its expected base:
#      offset (0,0) on 0, all other offsets on bg. This prevents source erasure from damaging frames.
#    - It also rejects any input/output state where colored cells from different corners would overlap.
# e) Visual observability
#    - `source_color != target_marker_color` is mandatory.
#    - Across examples, humans can see dense source glyphs vanish while sparse markers elsewhere
#      blossom into glyphs of the old source color. The repeated use of exactly two task colors makes
#      the transfer legible without relying on numeric color identities.
# f) Visual cleanliness
#    - No glyph/glyph overlap is allowed in either the input state or the final output state.
#    - Targets remain marker-sparse in the input and full-glyph-dense in the output, while sources do
#      the reverse; this keeps each corner role readable.
# g) Inferability of the change
#    - The output uses only one non-structural color: the former source color. The missing marker
#      color is itself a clue that the color did not stay local to targets.
#    - Because several targets share the same output color and all sources disappear, the only simple
#      explanation is "take the source glyph/color, send it to every marker corner, then erase the
#      original source locations."
# - New solver assumptions reviewed against the generator:
#   1. All sources share one glyph color, and all targets share one marker color.
#      `_pick_task_colors()` and `_derive_input()` enforce this.
#   2. Only source corners contain outward painted cells in the input.
#      Targets keep only their deepest diagonal marker.
#   3. Restoring a source offset means writing 0 at canonical (0,0) and bg elsewhere.
#      `_pattern_matches_base()` enforces that every legal glyph placement respects this.
#   4. Solver loops assume no cross-corner write conflicts during erase/paint.
#      `_state_cells()` rejects overlapping colored states for both input and output.

import random
from collections import Counter

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

# Canonical templates never extend beyond |dr|,|dc| == 2. Keeping the probe tight
# avoids absorbing colored cells from neighboring corners on the same small frame.
_NEIGH_R = 2


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical (TL-oriented) offset; self-inverse."""
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
                cnt1, saw_zero_1 = _edge_metric(g, r, c, d1r, d1c, bg, H, W)
                cnt2, saw_zero_2 = _edge_metric(g, r, c, d2r, d2c, bg, H, W)
                if not (cnt1 >= 3 and saw_zero_1 and cnt2 >= 3 and saw_zero_2):
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


def _base_value_for_offset(off, bg):
    return 0 if off == (0, 0) else bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer a source corner glyph onto every marked destination corner, then erase the sources.

    Input structure:
    - The dominant color is the background, and 2-3 hollow 0-frames provide detectable TL/TR/BL/BR
      corners.
    - One or more source corners already contain the full rotated glyph in one shared source color.
    - Every other active corner keeps only the deepest inward-diagonal marker, and all such markers
      share one other color.

    Rule:
    - Recover the shared canonical glyph and the shared source color from the source corner(s).
    - Use that source color to paint the full rotated glyph at every marker-bearing target corner.
    - Remove the glyph from each source corner by restoring the underlying 0/bg scaffold.

    Key insight:
    - D4 contributes the color-transfer rule: targets do not keep their marker color, they inherit
      the source glyph color. D5 contributes the motion rule: after that transfer, the source glyphs
      are erased. The visible output is therefore "source color moved to the targets."
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c_key: _colored_around(g, *c_key, bg) for c_key in corners}

    def _is_outward(off):
        return off[0] < 0 or off[1] < 0

    sources = []
    for c_key, items in per_corner.items():
        outward = [(off, color) for off, color in items if _is_outward(off)]
        if len(outward) >= 2:
            colors = {color for _, color in outward}
            assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
            sources.append(c_key)
    assert sources, "at least one source corner (pre-drawn outward pattern) required"

    def _extract(c_key):
        items = per_corner[c_key]
        pat_col = Counter(color for _, color in items).most_common(1)[0][0]
        return frozenset(off for off, color in items if color == pat_col), pat_col

    canon, source_color = _extract(sources[0])
    for s in sources[1:]:
        other, other_color = _extract(s)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"
        assert other_color == source_color, (
            f"all sources must share one color: {source_color} vs {other_color}"
        )

    for dr, dc in canon:
        assert (dc, dr) in canon, (
            f"pattern not diagonally symmetric: ({dr},{dc}) has no mirror ({dc},{dr})"
        )

    diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal cell to anchor markers"
    marker_canon = (max(diag_ks), max(diag_ks))

    src_set = set(sources)
    targets = []
    target_colors = set()
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        mdr, mdc = _rot(*marker_canon, kind)
        mr, mc = r + mdr, c + mdc
        if not (0 <= mr < H and 0 <= mc < W):
            continue
        tcol = int(g[mr, mc])
        if tcol == bg or tcol == 0:
            continue
        for off, color in per_corner[c_key]:
            if off in canon:
                assert color == tcol, (
                    f"target ({r},{c},{kind}) has conflicting color {color} at "
                    f"canonical offset {off}; expected {tcol}"
                )
        targets.append((r, c, kind))
        target_colors.add(tcol)

    assert targets, "no target corners detected (every corner appears to be a source)"
    assert len(target_colors) == 1, f"all targets must share one marker color: {target_colors}"
    target_color = next(iter(target_colors))
    assert source_color != target_color, "transfer colors must be visually distinct"

    out = g.copy()

    for r, c, kind in sources:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = _base_value_for_offset((cdr, cdc), bg)

    for r, c, kind in targets:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == source_color or v == target_color, (
                f"paint conflict at ({rr},{cc}) for target ({r},{c},{kind}): existing {v}"
            )
            out[rr, cc] = source_color

    return out


# --------------- generator (output-first color-transfer move) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (5, 8)
SOURCE_MIN_SIZE = 6

N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2

MAX_ATTEMPTS = 300

PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (0, 0)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}


def _split_total(total: int, parts: int) -> list[int]:
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
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _corner_coord(frame: dict, kind: str) -> tuple[int, int]:
    dr, dc = _CORNER_OFFSETS[kind]
    span = frame["size"] - 1
    return frame["r"] + dr * span, frame["c"] + dc * span


def _sample_frame_pair(span: int) -> tuple[int, int, int, int] | None:
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
    lo, hi = FRAME_SIZE_RANGE
    missing = random.choice(CORNER_KINDS)
    present = tuple(kind for kind in CORNER_KINDS if kind != missing)

    def _bands(sizes: dict[str, int]) -> tuple[int, int, int, int]:
        top = max((sizes[slot] for slot in present if slot[0] == "T"), default=0)
        bot = max((sizes[slot] for slot in present if slot[0] == "B"), default=0)
        left = max((sizes[slot] for slot in present if slot[1] == "L"), default=0)
        right = max((sizes[slot] for slot in present if slot[1] == "R"), default=0)
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
    return [{"r": coords[slot][0], "c": coords[slot][1], "size": sizes[slot]} for slot in present]


def _paint_pattern(grid: np.ndarray, r: int, c: int, kind: str, pattern, color: int) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]] | None:
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


def _pattern_cells(r: int, c: int, kind: str, pattern) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cells = []
    for off in pattern:
        adr, adc = _rot(*off, kind)
        cells.append(((r + adr, c + adc), off))
    return cells


def _marker_cell(r: int, c: int, kind: str, k_max: int) -> tuple[int, int]:
    mdr, mdc = _rot(k_max, k_max, kind)
    return r + mdr, c + mdc


def _pattern_matches_base(base: np.ndarray, r: int, c: int, kind: str, pattern, bg: int) -> bool:
    H, W = base.shape
    for (rr, cc), off in _pattern_cells(r, c, kind, pattern):
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if int(base[rr, cc]) != _base_value_for_offset(off, bg):
            return False
    return True


def _state_cells(frames: list[dict], pattern, k_max: int, sources: set[tuple[int, str]], *,
                 include_sources_full: bool, include_targets_full: bool) -> set[tuple[int, int]] | None:
    occupied = set()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            is_source = (fi, kind) in sources
            if is_source and include_sources_full:
                for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
                    if (rr, cc) in occupied:
                        return None
                    occupied.add((rr, cc))
            elif (not is_source) and include_targets_full:
                for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
                    if (rr, cc) in occupied:
                        return None
                    occupied.add((rr, cc))
            elif not is_source:
                marker = _marker_cell(r, c, kind, k_max)
                if marker in occupied:
                    return None
                occupied.add(marker)
    return occupied


def _layout_is_legal(base: np.ndarray, frames: list[dict], pattern, k_max: int,
                     sources: set[tuple[int, str]], bg: int) -> bool:
    for frame in frames:
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            if not _pattern_matches_base(base, r, c, kind, pattern, bg):
                return False

    output_cells = _state_cells(
        frames,
        pattern,
        k_max,
        sources,
        include_sources_full=False,
        include_targets_full=True,
    )
    if output_cells is None:
        return False

    input_cells = _state_cells(
        frames,
        pattern,
        k_max,
        sources,
        include_sources_full=True,
        include_targets_full=False,
    )
    if input_cells is None:
        return False

    removed = 0
    for fi, kind in sources:
        r, c = _corner_coord(frames[fi], kind)
        for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
            if (rr, cc) not in output_cells:
                removed += 1
    return removed > 0


def _pick_task_colors(bg: int) -> tuple[int, int]:
    palette = [color for color in range(1, 10) if color != bg]
    return tuple(random.sample(palette, 2))


def _render_output(base: np.ndarray, frames: list[dict], pattern,
                   sources: set[tuple[int, str]], source_color: int) -> np.ndarray:
    out = base.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            _paint_pattern(out, r, c, kind, pattern, source_color)
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict], pattern, k_max: int,
                  sources: set[tuple[int, str]], source_color: int,
                  target_marker_color: int) -> np.ndarray:
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            if (fi, kind) in sources:
                _paint_pattern(inp, r, c, kind, pattern, source_color)
                continue
            marker = _marker_cell(r, c, kind, k_max)
            for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
                if (rr, cc) == marker:
                    inp[rr, cc] = target_marker_color
                else:
                    inp[rr, cc] = base[rr, cc]
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()

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

    sources = _pick_sources(frames)
    if sources is None:
        return None
    if not _layout_is_legal(base, frames, pattern, k_max, sources, bg):
        return None

    source_color, target_marker_color = _pick_task_colors(bg)
    output = _render_output(base, frames, pattern, sources, source_color)
    inp = _derive_input(
        output,
        base,
        frames,
        pattern,
        k_max,
        sources,
        source_color,
        target_marker_color,
    )

    try:
        if not np.array_equal(main(inp), output):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
