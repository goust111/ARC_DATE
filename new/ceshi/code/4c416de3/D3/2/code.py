# Variant of 4c416de3
# Dimension: D3 — 目标对象：标记像素定义 (dst)
# Change: 目标角点仅有一个彩色标记像素，位于图案最深对角偏移(k_max, k_max)处，其颜色决定输出颜色
#   -> 框架边框颜色编码目标色：目标角点不再保留内部 marker，而是在角点相邻的两条边框上
#      各涂一小段同色边框，solver 需要从边框局部着色中读出目标色。

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

# Radius for scanning a corner's neighborhood. Must exceed max canonical |dr|,|dc| a pattern can have.
_NEIGH_R = 3

# Canonical hint cells: one colored border cell on each inward edge adjacent to the corner.
# They do not overlap the original pattern templates, so D3 changes only the target cue.
HINT_OFFSETS = ((1, 0), (0, 1))


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
    """Walk from (r,c) along (dr,dc) until bg/boundary. Return (non_bg_count, saw_zero).
    A real frame edge has both: multiple non-bg cells and at least one 0-cell (to exclude
    pattern-interior paths composed entirely of painted cells)."""
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
                # Each inward edge must be a real frame: >=3 non-bg cells AND >=1 actual 0.
                cnt1, sz1 = _edge_metric(g, r, c, d1r, d1c, bg, H, W)
                cnt2, sz2 = _edge_metric(g, r, c, d2r, d2c, bg, H, W)
                if not (cnt1 >= 3 and sz1 and cnt2 >= 3 and sz2):
                    continue
                # Outward cells must not be 0 (rules out mid-edge false positives).
                ok = True
                for (odr, odc) in OUT_DIRS[kind]:
                    rr, cc = r + odr, c + odc
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == 0:
                        ok = False
                        break
                if not ok:
                    continue
                corners.append((r, c, kind))
                break  # one kind per cell
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


def _read_hint_color(g, r, c, kind, bg):
    """Read the target color from the colored border segments near this corner."""
    H, W = g.shape
    vals = []
    seen_colored = False
    seen_blank = False
    for hdr, hdc in HINT_OFFSETS:
        adr, adc = _rot(hdr, hdc, kind)
        rr, cc = r + adr, c + adc
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        v = int(g[rr, cc])
        if v == bg or v == 0:
            seen_blank = True
            continue
        seen_colored = True
        vals.append(v)
    if not seen_colored:
        return None
    assert not seen_blank, f"partial border hint at corner {(r, c, kind)}"
    colors = set(vals)
    assert len(colors) == 1, f"mixed border hint colors at corner {(r, c, kind)}: {colors}"
    return vals[0]


# Design analysis for the D3 variant
# a) Concrete behavior:
#    - Original targets exposed their output color via one interior diagonal marker at (k_max, k_max).
#    - New targets instead color two frame-edge cells: canonical offsets {(1,0), (0,1)}
#      from the corner. This forms a tiny inward L-shape on the border.
#    - Example trace at a TL target with color 6:
#      original input kept only (1,1) or (2,2) colored -> solver read that diagonal pixel;
#      new input keeps the first cell on the top edge and the first cell on the left edge colored 6,
#      while all actual pattern cells are erased -> solver now reads the border L, then paints the same
#      rotated pattern in color 6.
# b) Complexity calibration:
#    - Two orthogonal hint cells are enough to be seen as an intentional border cue, while still fitting
#      cleanly even on the smallest 5x5 frames.
#    - The hint has exactly one form across all samples, so humans get repeated evidence rather than a
#      combinatorial family of encodings.
# c) Old-assumption audit:
#    - Source detection via outward colored cells still holds because hints are strictly inward.
#    - Canon extraction from sources still works because sources do not receive border hints.
#    - Target classification can no longer read the deepest diagonal cell, so that branch is replaced by
#      `_read_hint_color`; keeping the old logic would silently miss all targets.
# d) Placement legality:
#    - Frame sizes stay 5-8, so the immediate border-adjacent hint cells always fit and never collide
#      with the opposite corner's hint on the same edge.
#    - Hint cells are disjoint from every canonical pattern template used here, preventing overlap bugs.
# e) Visual observability:
#    - Across 2-4 examples, the only persistent target-side cue is the colored border L, which stays in
#      both input and output; the completed output then adds the corner pattern in the same color.
# f) Visual cleanliness:
#    - Hints live on the frame border and the pattern lives outside the frame / on the diagonal, so the
#      two object layers remain visually separable.
# g) Color inference:
#    - The target color is copied directly from the visible hint color; there is no hidden color mapping
#      or numeric ordering.
# 9) New assumptions introduced in `main()` and guaranteed by generator:
#    - Any non-source participating corner has either a full two-cell hint or no hint at all.
#    - Both hint cells share one color.
#    - Sources remain hint-free, so extracting the canonical source pattern is not polluted by cue cells.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a pre-drawn corner pattern to every other border-hint-bearing corner.

    Input structure:
    - bg = most frequent color. Hollow 0-frames are still the main objects, and a
      frame corner may remain 0 or already be painted by a source pattern.
    - >= 1 'source' corner: the full pattern is already drawn in a single color and
      includes outward cells, which distinguishes it from a mere frame corner.
    - Other participating corners: no inward diagonal marker remains. Instead, the
      first cell on each inward frame edge is recolored to the target color.

    Rule:
    - Recover the shared canonical (TL-oriented) pattern from any source corner.
    - For each non-source corner whose border-adjacent hint segments are colored,
      infer that corner's target color from the border and paint the rotated pattern.

    Key insight: the corner glyph is still the same diagonally symmetric pattern as
    in the base task; the only changed evidence is how target color is encoded.
    The colored border L near a target corner supplies the paint color directly.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}

    # Source = corner with >=2 strictly-outward (canonical dr<0 or dc<0) colored cells
    # of a single color.
    def _is_outward(off):
        return off[0] < 0 or off[1] < 0

    sources = []
    for c_key, items in per_corner.items():
        outward = [(o, v) for (o, v) in items if _is_outward(o)]
        if len(outward) >= 2:
            colors = {v for (_, v) in outward}
            assert len(colors) == 1, (
                f"source {c_key} has mixed outward colors: {colors}"
            )
            sources.append(c_key)
    assert sources, "at least one source corner (pre-drawn outward pattern) required"

    def _extract(c_key):
        items = per_corner[c_key]
        pat_col = Counter(v for (_, v) in items).most_common(1)[0][0]
        return frozenset(o for (o, v) in items if v == pat_col), pat_col

    canon, _ = _extract(sources[0])
    for s in sources[1:]:
        other, _ = _extract(s)
        assert other == canon, (
            f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"
        )

    # Pattern must be symmetric about the corner's inward-diagonal axis (dr==dc).
    for (dr, dc) in canon:
        assert (dc, dr) in canon, (
            f"pattern not diagonally symmetric: ({dr},{dc}) has no mirror ({dc},{dr})"
        )

    diag_ks = [dr for (dr, dc) in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal cells"

    # Classify non-source corners as targets by reading the border-color cue.
    src_set = set(sources)
    targets = []
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        tcol = _read_hint_color(g, r, c, kind, bg)
        if tcol is None:
            continue  # no border cue: this detected corner plays no role
        for off, color in per_corner[c_key]:
            if off in HINT_OFFSETS:
                assert color == tcol, (
                    f"target ({r},{c},{kind}) has conflicting hint color {color} at "
                    f"canonical offset {off}; expected {tcol}"
                )
        targets.append((r, c, kind, tcol))

    assert targets, "no target corners detected (every corner appears to be a source)"

    out = g.copy()
    for (r, c, kind, tcol) in targets:
        for (cdr, cdc) in canon:
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

# --- canvas & layout ---
H_RANGE = (18, 24)  # grid height bounds (inclusive)
W_RANGE = (18, 24)  # grid width bounds (inclusive)
BORDER = 1  # min bg gap between a frame and grid edge
FRAME_GAP = 4  # min bg gap between two frames
FRAME_SIZE_RANGE = (5, 8)  # per-side length of a hollow 0-frame (inclusive)
SOURCE_MIN_SIZE = 6  # only frames strictly larger than the minimum may host sources

# --- structural counts ---
N_FRAMES_2_PROB = 0.5  # probability of 2-frame layout (else 3-frame)
TWO_FRAME_HORIZONTAL_PROB = 0.5  # probability of side-by-side vs stacked 2-frame layout
N_SOURCES_MAX = 2  # upper bound on source corners per grid (>= 1)

# --- generation retries ---
MAX_ATTEMPTS = 300

# --- canonical (TL-oriented) pattern templates ---
# Each: list of (dr, dc) offsets from the TL corner cell; must be diagonally
# symmetric (dr==dc axis) and contain >=1 inward-diagonal cell (k, k) with k>=1.
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (0, 0)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)

# Offsets from a frame's top-left origin (r, c) to each corner cell of a size-N frame:
# applied as (r + dr*(size-1), c + dc*(size-1)).
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


def _paint_hint(grid: np.ndarray, r: int, c: int, kind: str, color: int) -> None:
    for hdr, hdc in HINT_OFFSETS:
        rr, cc = _rot(hdr, hdc, kind)
        grid[r + rr, c + cc] = color


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    """Give every corner of every frame an independently sampled non-bg color."""
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]] | None:
    """Choose 1..N_SOURCES_MAX source corners from frames larger than the minimum size."""
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


def _render_output(base: np.ndarray, frames: list[dict], pattern: list[tuple[int, int]],
                   sources: set[tuple[int, str]]) -> np.ndarray:
    """Paint the canonical pattern at every corner; non-sources also keep their border hints."""
    out = base.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            color = frame["corner_colors"][kind]
            _paint_pattern(out, r, c, kind, pattern, color)
            if (fi, kind) not in sources:
                _paint_hint(out, r, c, kind, color)
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict],
                  pattern: list[tuple[int, int]], sources: set[tuple[int, str]]) -> np.ndarray:
    """Erase every non-source pattern back to base, preserving only the border-color hint."""
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            for dr, dc in pattern:
                rr, cc = _rot(dr, dc, kind)
                rr, cc = r + rr, c + cc
                inp[rr, cc] = base[rr, cc]
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, _k_max = _choose_pattern()

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

    output = _render_output(base, frames, pattern, sources)
    inp = _derive_input(output, base, frames, pattern, sources)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
