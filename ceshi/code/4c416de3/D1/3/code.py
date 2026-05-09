# Variant of 4c416de3
# Dimension: D1 — 匹配条件：对角对称映射 (match_cond)
# Change: 源图案关于角点的对角线轴（dr==dc）对称，通过旋转/翻转映射到四种角点方位（TL/TR/BL/BR）... → 旋转90度映射而非反射映射

"""
Design analysis for the D1 rotation variant
===========================================

spec:
  concretization:
    objects:
      - 2-3 hollow square 0-frames on a solid non-zero background.
      - Every frame corner is either a source or a target.
      - A source corner shows the full glyph in one non-bg color.
      - A target corner keeps only one marker at the deepest inward diagonal cell.
    canonical_glyph:
      - Still defined in TL orientation and shared by all corners.
      - No longer required to satisfy diagonal reflection symmetry.
      - Must be visibly asymmetric so that TR/BL results under 90-degree rotation differ
        from the original mirror-based task.
      - Each template keeps 2+ outward cells plus 1+ inward diagonal cells; some also
        contain one off-diagonal inward cell such as (2,1), which is what makes the
        rotation-vs-reflection difference easy to see.
  behavior_trace:
    old_logic:
      - Canonical TL offsets were mapped to corner kinds by sign flips:
        TL=(dr,dc), TR=(dr,-dc), BL=(-dr,dc), BR=(-dr,-dc).
      - This is a reflection family, so diagonal-symmetric patterns look identical after
        transport.
    new_logic:
      - Canonical TL offsets are mapped by quarter-turns:
        TL=0deg, TR=90deg clockwise, BR=180deg, BL=90deg counterclockwise.
      - Example with canonical cell (2,1):
        TL source keeps it at (2,1).
        Under old TR reflection it would land at (2,-1).
        Under new TR rotation it lands at (1,-2).
        That single cell changes the perceived hook direction, so the variant is visible.
  complexity_calibration:
    - Only one mechanism changes: the corner-orientation transform.
    - Frame finding, source/target detection, marker-color reuse, and output-first
      generation all stay unchanged.
    - To avoid invisible degeneration, templates are curated so at least one target kind
      where old/new differ (TR or BL) will show a different footprint.
  implicit_assumptions_review:
    - Original `_rot` was self-inverse because sign-flip reflections are self-inverse.
      That assumption breaks under 90-degree rotation, so the code must split into
      `_to_actual` and `_to_canon`.
    - Original symmetry assert over `(dr,dc) <-> (dc,dr)` depends on the old match
      condition and must be removed.
    - Marker lookup at `(k,k)` is still valid, but only after applying the new rotational
      corner transform.
    - Source classification by "has outward cells of one color" remains valid because
      canonicalization now uses inverse rotation before the test.
  placement_legality:
    - All template cells stay within `|dr|,|dc| <= 3`, so the existing neighborhood scan
      radius is still sufficient.
    - Templates avoid painting long runs on frame edges; this preserves corner detection.
    - Frames still have >=1 background gap from borders and >=4 between each other, which
      keeps local neighborhoods disjoint for extraction.
  visual_observability:
    - Every grid has 2+ frames, hence multiple TR/BL targets are available.
    - Since BR is unchanged between reflection and 180-degree rotation, observability is
      driven by the asymmetric off-diagonal cell(s) on TR/BL.
    - Colors are copied directly from visible marker pixels; no hidden color ordering or
      numeric color semantics are introduced.
  visual_cleanliness:
    - Glyphs are small (4-6 cells), one color per corner, and separated by frame gaps.
    - Only the marker remains on targets in the input, so the puzzle stays readable.
  color_reasoning:
    - Target color equals the visible marker color, exactly as in the original task.
    - Independent per-corner colors are acceptable because the structural cue is local and
      repeated many times in each grid.
  new_main_assumptions_checked_against_generator:
    - At least one source exists and all sources share one canonical pattern.
    - The canonical pattern has a deepest diagonal marker `(k_max,k_max)`.
    - At least one non-source target exists; with 2-3 full frames and <=2 sources this is
      guaranteed.
    - Generated templates satisfy `rotation(TR) != legacy_reflection(TR)`, so the variant
      is not silently collapsing back to the original behavior.
"""

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line, get_bg


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

_ROT_STEPS = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}


def _rotate_cw(dr: int, dc: int, steps: int) -> tuple[int, int]:
    """Rotate a canonical offset around the corner by `steps` quarter-turns clockwise."""
    steps %= 4
    for _ in range(steps):
        dr, dc = dc, -dr
    return dr, dc


def _to_actual(dr: int, dc: int, kind: str) -> tuple[int, int]:
    """Canonical TL-oriented offset -> actual offset for the given corner kind."""
    return _rotate_cw(dr, dc, _ROT_STEPS[kind])


def _to_canon(dr: int, dc: int, kind: str) -> tuple[int, int]:
    """Actual offset near a corner -> canonical TL-oriented offset."""
    return _rotate_cw(dr, dc, -_ROT_STEPS[kind])


def _legacy_reflect(dr: int, dc: int, kind: str) -> tuple[int, int]:
    """Original task's reflection-style mapping, kept only for generator validation."""
    if kind == "TL":
        return dr, dc
    if kind == "TR":
        return dr, -dc
    if kind == "BL":
        return -dr, dc
    return -dr, -dc


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
                for odr, odc in OUT_DIRS[kind]:
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
            items.append((_to_canon(dr, dc, kind), v))
    return items


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a pre-drawn corner glyph to every other marked corner by quarter-turn.

    Input structure:
    - bg is the most common color. Each object is a hollow 0-frame whose corner cell is
      either 0 or already painted.
    - At least one source corner already contains the whole glyph in one color, including
      outward cells plus an inward diagonal anchor.
    - Every non-source target corner keeps only one colored marker at the deepest inward
      diagonal cell; that marker color is the target's output color.

    Rule:
    - Read one source glyph in canonical TL orientation.
    - For every marked target corner, rotate that canonical glyph according to the corner
      kind: TL=0deg, TR=90deg clockwise, BR=180deg, BL=90deg counterclockwise.
    - Paint the rotated glyph using the target corner's own marker color.

    Key insight: the corner type now specifies a quarter-turn, not a mirror. The source
    pattern is intentionally asymmetric, so TR/BL targets reveal rotation rather than
    reflection while still reusing the visible marker color.
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
            assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
            sources.append(c_key)
    assert sources, "at least one source corner (pre-drawn outward pattern) required"

    def _extract(c_key):
        items = per_corner[c_key]
        pat_col = Counter(v for (_, v) in items).most_common(1)[0][0]
        return frozenset(o for (o, v) in items if v == pat_col), pat_col

    canon, _ = _extract(sources[0])
    for s in sources[1:]:
        other, _ = _extract(s)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    # Pattern must anchor at least one inward-diagonal cell so targets have a marker.
    diag_ks = [dr for (dr, dc) in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal cell to anchor markers"
    marker_canon = (max(diag_ks), max(diag_ks))

    # Classify non-source corners as targets by reading their marker cell.
    src_set = set(sources)
    targets = []
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        mdr, mdc = _to_actual(*marker_canon, kind)
        mr, mc = r + mdr, c + mdc
        if not (0 <= mr < H and 0 <= mc < W):
            continue
        v = int(g[mr, mc])
        if v == bg or v == 0:
            continue  # no marker: this detected corner plays no role
        tcol = v
        # All input cells at this corner's canonical pattern offsets must agree in color.
        for off, color in per_corner[c_key]:
            if off in canon:
                assert color == tcol, (
                    f"target ({r},{c},{kind}) has conflicting color {color} at "
                    f"canonical offset {off}; expected {tcol}"
                )
        targets.append((r, c, kind, tcol))

    assert targets, "no target corners detected (every corner appears to be a source)"

    out = g.copy()
    for r, c, kind, tcol in targets:
        for cdr, cdc in canon:
            adr, adc = _to_actual(cdr, cdc, kind)
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


# --------------- generator (output-first pattern replication, rotation variant) ---------------

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

# --- canonical (TL-oriented) asymmetric pattern templates ---
# Each template:
# - stays within |dr|,|dc| <= 3
# - includes >=2 outward cells and >=1 inward-diagonal cell
# - includes at least one cell that makes TR/BL quarter-turns differ from legacy reflection
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 1)],
    [(-1, 0), (0, -1), (-1, -1), (1, 1), (2, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2), (2, 1)],
    [(-1, 0), (0, -1), (-1, -1), (1, 1), (2, 2), (1, 2)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2), (3, 2)],
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


def _pattern_has_rotation_effect(pattern: list[tuple[int, int]]) -> bool:
    """Reject patterns whose TR image would accidentally match the old reflection rule."""
    new_tr = {_to_actual(dr, dc, "TR") for dr, dc in pattern}
    old_tr = {_legacy_reflect(dr, dc, "TR") for dr, dc in pattern}
    return new_tr != old_tr


def _choose_pattern() -> tuple[list[tuple[int, int]], int]:
    """Return (pattern cells, k_max = deepest inward-diagonal offset)."""
    pattern = list(random.choice(PATTERN_TEMPLATES))
    assert _pattern_has_rotation_effect(pattern)
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


def _paint_pattern(
    grid: np.ndarray, r: int, c: int, kind: str, pattern: list[tuple[int, int]], color: int
) -> None:
    for dr, dc in pattern:
        rr, cc = _to_actual(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _pattern_fits_frames(frames: list[dict], pattern: list[tuple[int, int]]) -> bool:
    """Reject layouts where corner glyphs within a frame would overlap each other."""
    occupied: set[tuple[int, int]] = set()
    for frame in frames:
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            cells = set()
            for dr, dc in pattern:
                rr, cc = _to_actual(dr, dc, kind)
                cells.add((r + rr, c + cc))
            if occupied & cells:
                return False
            occupied |= cells
    return True


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    """Give every corner of every frame an independently sampled non-bg color."""
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]] | None:
    """Choose 1..N_SOURCES_MAX source corners from frames larger than the minimum size."""
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
    k_max: int,
    sources: set[tuple[int, str]],
) -> np.ndarray:
    """Erase every non-source corner's pattern back to base, keeping only the marker."""
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            mdr, mdc = _to_actual(k_max, k_max, kind)
            marker = (r + mdr, c + mdc)
            for dr, dc in pattern:
                rr, cc = _to_actual(dr, dc, kind)
                rr, cc = r + rr, c + cc
                if (rr, cc) != marker:
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
    if not _pattern_fits_frames(frames, pattern):
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_corner_colors(frames, bg)
    sources = _pick_sources(frames)
    if sources is None:
        return None

    output = _render_output(base, frames, pattern)
    inp = _derive_input(output, base, frames, pattern, k_max, sources)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
