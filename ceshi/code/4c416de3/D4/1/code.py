# Variant of 4c416de3
# Dimension: D4 — 变换：颜色传递规则 (transform)
# Change: 目标角点图案使用标记像素的颜色绘制（每个目标独立着色），源角点保持原色不变
#   → 颜色交换：所有目标图案改用源图案颜色绘制，同时所有源图案改成目标标记色
#
# Design analysis for the D4 variant
# a) Concrete behavior
#    - Keep the same object vocabulary as the original task: 2-3 hollow 0-frames on a solid bg,
#      one shared diagonally symmetric corner glyph, and source/target corners.
#    - To make "swap" well-defined, the grid uses exactly two task colors besides bg/0:
#      `source_color` for every pre-drawn source pattern in the input, and `target_color`
#      for every target marker in the input.
#    - Input: each source corner shows the full glyph in `source_color`; every non-source corner
#      shows only the deepest diagonal marker in `target_color`.
#    - Output: every target expands into the full glyph in `source_color`, and every source glyph
#      is recolored to `target_color`.
#    - Small trace:
#      input source TL = blue full glyph, target BR = red marker only.
#      original rule would output BR full glyph in red and keep TL blue.
#      new rule outputs BR full glyph in blue and recolors TL to red.
# b) Complexity calibration
#    - Pairwise per-target colors from the original task would make "swap" ambiguous when there are
#      many targets but only one source color. Therefore all targets share one marker color.
#    - The grid still has many active corners, so the swap is repeated several times and is not a
#      one-off coincidence. Sources may be 1 or 2; targets are the remaining corners.
# c) Implicit assumptions to replace
#    - Original `main()` assumes target output color equals the target marker color. That is no
#      longer valid; target color must now come from the shared source color.
#    - Original `main()` leaves source corners untouched. That is no longer valid; sources must be
#      repainted to the shared target marker color.
#    - Original generator allowed each target to choose an independent color. That must be removed,
#      otherwise the source side would have no single color to swap into.
# d) Placement legality
#    - Frame placement still follows the original non-overlap strategy.
#    - The generator additionally rejects layouts where full corner glyphs from different corners
#      would overlap. This avoids color-conflict artifacts and keeps the swap rule isolated.
# e) Visual observability
#    - `source_color != target_color` is mandatory.
#    - The same two colors repeat across multiple corners, so a human can see that targets receive
#      the old source color while sources take the old marker color.
# f) Visual cleanliness
#    - No full-pattern overlap between corners.
#    - Targets remain sparse in the input (single markers), while sources remain dense; this keeps
#      the before/after contrast readable.
# g) Color-rule inferability
#    - Only two non-structural colors participate in the transformation.
#    - Both colors have structural anchors: full source glyphs identify `source_color`, and repeated
#      deepest-diagonal markers identify `target_color`.

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


CORNER_KINDS = ("TL", "TR", "BL", "BR")

# Inward-along-frame-edge directions for each corner kind.
EDGE_DIRS = {
    "TL": ((+1, 0), (0, +1)),
    "TR": ((+1, 0), (0, -1)),
    "BL": ((-1, 0), (0, +1)),
    "BR": ((-1, 0), (0, -1)),
}
# Outward-from-corner directions; cells here must not be 0 (guards mid-edge false positives).
OUT_DIRS = {
    "TL": ((-1, 0), (0, -1)),
    "TR": ((-1, 0), (0, +1)),
    "BL": ((+1, 0), (0, -1)),
    "BR": ((+1, 0), (0, +1)),
}

# Radius for scanning a corner's neighborhood. The canonical glyph templates stay within 2 cells.
_NEIGH_R = 2


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical (TL-orientation) offset; self-inverse."""
    if kind == "TL":
        return (dr, dc)
    if kind == "TR":
        return (dr, -dc)
    if kind == "BL":
        return (-dr, dc)
    if kind == "BR":
        return (-dr, -dc)
    raise ValueError(kind)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the colors carried by source glyphs and target markers around hollow-frame corners.

    Input structure:
    - bg = most frequent color. Hollow 0-frames define detectable TL/TR/BL/BR corners.
    - >= 1 source corner already contains the full rotated glyph in one shared source color.
    - Every non-source active corner carries only one colored marker at the glyph's deepest
      inward-diagonal offset, and all such markers share one target color.

    Rule:
    - Recover the canonical glyph shape from the source corner(s).
    - Use the source glyph color to fill every target glyph.
    - Repaint every source glyph into the shared target-marker color.

    Key insight: shape extraction is unchanged from the original task; the new information is that
    colors move in both directions. The pre-drawn source identifies the glyph shape and its old
    color, while the repeated target markers identify the color that must replace the sources.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}

    def _is_outward(off):
        return off[0] < 0 or off[1] < 0

    sources = []
    for c_key, items in per_corner.items():
        outward = [(o, v) for (o, v) in items if _is_outward(o)]
        if len(outward) >= 2:
            colors = {v for (_, v) in outward}
            assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
            sources.append(c_key)
    assert sources, "at least one source corner (pre-drawn outward pattern) required"

    def _extract(c_key):
        items = per_corner[c_key]
        pat_col = Counter(v for (_, v) in items).most_common(1)[0][0]
        return frozenset(o for (o, v) in items if v == pat_col), pat_col

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

    diag_ks = [dr for (dr, dc) in canon if dr == dc and dr > 0]
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
        v = int(g[mr, mc])
        if v == bg or v == 0:
            continue
        tcol = v
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
    assert source_color != target_color, "swap colors must be visually distinct"

    out = g.copy()

    for r, c, kind in sources:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == source_color or v == target_color, (
                f"paint conflict at ({rr},{cc}) for source ({r},{c},{kind}): existing {v}"
            )
            out[rr, cc] = target_color

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


# --------------- generator (output-first color-swap replication) ---------------

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


def _paint_pattern(grid: np.ndarray, r: int, c: int, kind: str,
                   pattern: list[tuple[int, int]], color: int) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]] | None:
    """Choose 1..N_SOURCES_MAX source corners from frames larger than the minimum size."""
    eligible = [(fi, kind) for fi, f in enumerate(frames)
                if f["size"] >= SOURCE_MIN_SIZE for kind in CORNER_KINDS]
    if not eligible:
        return None
    n = random.randint(1, min(N_SOURCES_MAX, len(eligible)))
    return set(random.sample(eligible, n))


def _corner_cells(frame: dict, kind: str,
                  pattern: list[tuple[int, int]]) -> set[tuple[int, int]]:
    r, c = _corner_coord(frame, kind)
    cells = set()
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        cells.add((r + rr, c + cc))
    return cells


def _layout_is_clean(frames: list[dict], pattern: list[tuple[int, int]]) -> bool:
    """Reject layouts where full corner glyphs overlap, to keep the color swap unambiguous."""
    seen = set()
    for frame in frames:
        for kind in CORNER_KINDS:
            cells = _corner_cells(frame, kind, pattern)
            if seen & cells:
                return False
            seen |= cells
    return True


def _pick_swap_colors(bg: int) -> tuple[int, int]:
    palette = [c for c in range(1, 10) if c != bg]
    source_color, target_color = random.sample(palette, 2)
    return source_color, target_color


def _render_output(base: np.ndarray, frames: list[dict], pattern: list[tuple[int, int]],
                   sources: set[tuple[int, str]], source_color: int,
                   target_color: int) -> np.ndarray:
    """Render the swapped final state: targets use source_color, sources use target_color."""
    out = base.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            color = target_color if (fi, kind) in sources else source_color
            _paint_pattern(out, r, c, kind, pattern, color)
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict],
                  pattern: list[tuple[int, int]], k_max: int,
                  sources: set[tuple[int, str]], source_color: int,
                  target_color: int) -> np.ndarray:
    """
    Reverse the swap:
    - sources recover their original source_color full glyphs
    - targets are erased back to base except for the deepest-diagonal target_color marker
    """
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            if (fi, kind) in sources:
                _paint_pattern(inp, r, c, kind, pattern, source_color)
                continue
            mdr, mdc = _rot(k_max, k_max, kind)
            marker = (r + mdr, c + mdc)
            for dr, dc in pattern:
                rr, cc = _rot(dr, dc, kind)
                rr, cc = r + rr, c + cc
                if (rr, cc) == marker:
                    inp[rr, cc] = target_color
                else:
                    inp[rr, cc] = base[rr, cc]
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()

    frames = (_sample_two_frames(H, W) if random.random() < N_FRAMES_2_PROB
              else _sample_three_frames(H, W))
    if frames is None or not _layout_is_clean(frames, pattern):
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    sources = _pick_sources(frames)
    if sources is None:
        return None
    source_color, target_color = _pick_swap_colors(bg)

    output = _render_output(base, frames, pattern, sources, source_color, target_color)
    inp = _derive_input(output, base, frames, pattern, k_max, sources, source_color, target_color)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
