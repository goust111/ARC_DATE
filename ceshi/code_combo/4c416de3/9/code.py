# Combo variant of 4c416de3 (index 9)
# combo_spec: D1/3+D2/0+D5/0
# Name: 旋转+非对称+擦除

import random

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - The seed task had one shared corner glyph that was effectively transported by
#      reflection-sign-flips, and sources stayed visible in the output.
#    - Here the shared canonical glyph is an actually asymmetric TL-oriented sprite,
#      e.g. {(-1,0),(0,-1),(0,0),(1,1),(2,1)}. If that source sits at TL, then:
#        source input shows the full sprite at TL;
#        a TR target keeps only the marker at canonical (1,1), rotated to actual (1,-1);
#        output paints the full TR sprite by a 90-degree clockwise turn, so the
#        asymmetric "hook" (2,1) lands at (1,-2), not at the reflection location (2,-1).
#      After completion, the TL source is erased back to the bare frame/background.
#      So the fused rule is simultaneously: exact asymmetric-shape recovery + quarter-turn
#      orientation transport + move/erase semantics.
# b) Complexity calibration:
#    - The sprite still stays tiny (5-6 cells, within |dr|,|dc| <= 3) and one-color.
#    - Only one canonical sprite is shared across the whole grid, keeping the reasoning
#      local and repetitive enough for ARC.
#    - Templates are curated to be non-diagonal-symmetric and to differ under TR/BL
#      rotation versus legacy reflection, otherwise the D1 change would become invisible.
# c) Hidden assumptions audited:
#    - The original self-inverse `_rot` no longer works once TL->TR means 90-degree
#      rotation; we split that into `_to_actual` and `_to_canon`.
#    - The original symmetry assert is invalid because the sprite is intentionally
#      asymmetric. Source extraction must preserve the exact offset set.
#    - The original `out = g.copy(); paint only targets` is invalid because sources
#      disappear in the output; sources must be reset to their scaffold colors first.
#    - Source extraction should not rely on "most common color" when asymmetric sprites
#      vary in cell count; instead it keys off the unique outward color that identifies
#      the source sprite.
# d) Placement legality:
#    - The generator verifies that every sprite cell falls on the intended scaffold:
#      canonical (0,0) may overwrite the 0-valued frame corner, every other sprite cell
#      must sit on background.
#    - It rejects any layout where source sprites, target sprites, or target markers
#      would overlap in either input or output states.
# e) Visual observability:
#    - The generator requires a non-source TR or BL corner whose rotated footprint
#      differs from legacy reflection, making the quarter-turn rule human-visible.
#    - It also requires source removal to actually delete colored cells, so the move
#      semantics are visible rather than merely implied.
# f) Visual cleanliness:
#    - Frames keep a 4-cell gap and sprites are small, so each corner remains legible.
#    - Only one sprite family exists per grid; colors are reused only as corner-local
#      paint choices, not as hidden control signals.
# g) Color inference:
#    - Target color still comes directly from the visible marker pixel.
#    - No color ordering or numeric relation is used; structure alone determines which
#      cells to paint, while the marker supplies the paint color.
# New solver assumptions checked against the generator:
#    - Every source has >=2 outward colored cells of one color.
#    - All sources share one exact canonical sprite and include a deepest diagonal
#      marker cell (k,k) with k > 0.
#    - At least one non-source target exists, and at least one such target exhibits a
#      genuine rotation-vs-reflection difference.
#    - Only canonical offset (0,0) may lie on the 0-frame; all other sprite cells are
#      restored to background when a source is erased.


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

_NEIGH_R = 3
_ROT_STEPS = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}


def _rotate_cw(dr: int, dc: int, steps: int) -> tuple[int, int]:
    """Rotate offset `(dr, dc)` around a corner by clockwise quarter-turns."""
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
    """Original reflection-style mapping, kept for generator anti-degeneracy checks."""
    if kind == "TL":
        return dr, dc
    if kind == "TR":
        return dr, -dc
    if kind == "BL":
        return -dr, dc
    if kind == "BR":
        return -dr, -dc
    raise ValueError(f"unknown corner kind: {kind}")


def _edge_metric(g, r, c, dr, dc, bg, H, W, max_step=10):
    """Walk from `(r, c)` along a frame edge; return `(count_non_bg, saw_zero)`."""
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
                cnt1, saw_zero1 = _edge_metric(g, r, c, d1r, d1c, bg, H, W)
                cnt2, saw_zero2 = _edge_metric(g, r, c, d2r, d2c, bg, H, W)
                if not (cnt1 >= 3 and saw_zero1 and cnt2 >= 3 and saw_zero2):
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
            items.append((_to_canon(dr, dc, kind), v))
    return items


def _is_outward(off: tuple[int, int]) -> bool:
    dr, dc = off
    return dr < 0 or dc < 0


def _is_diagonal_symmetric(pattern: list[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    cells = set(pattern)
    return all((dc, dr) in cells for dr, dc in cells)


def _base_value_for_offset(off: tuple[int, int], bg: int) -> int:
    return 0 if off == (0, 0) else bg


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
    assert sources, "at least one source corner (pre-drawn outward sprite) required"

    canon = None
    for c_key in sources:
        source_color = source_colors[c_key]
        sprite = frozenset(off for off, v in per_corner[c_key] if v == source_color)
        assert sprite, f"source {c_key} has no colored sprite cells"
        if canon is None:
            canon = sprite
        else:
            assert sprite == canon, (
                f"source sprites disagree: {sorted(canon)} vs {sorted(sprite)}"
            )

    diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
    assert diag_ks, "sprite has no inward-diagonal cell to anchor markers"
    marker_canon = (max(diag_ks), max(diag_ks))
    assert marker_canon in canon, "deepest diagonal marker missing from source sprite"
    return sources, source_colors, canon, marker_canon


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move an asymmetric corner sprite from every source corner to all marked destination corners.

    Input structure:
    - The most common color is the background. Hollow 0-frames define candidate corners, and
      a corner may either be unpainted, show a full source sprite, or show only one marker.
    - One or more source corners already contain the complete sprite in a single non-bg color.
      That sprite may be asymmetric, but it always includes outward cells plus a deepest
      inward-diagonal anchor cell.
    - Each non-source participating corner keeps only the colored marker at that deepest
      inward-diagonal sprite location; the marker color is the color of the finished target.

    Rule:
    - Normalize a source corner into TL-oriented canonical offsets and recover the exact shared
      sprite shape from its source-colored cells.
    - Erase that sprite from every source corner by restoring the frame corner to 0 and every
      other sprite cell to background.
    - For each marked target corner, rotate the canonical offsets by quarter-turn according to
      the corner kind (TL=0deg, TR=90deg clockwise, BR=180deg, BL=90deg counterclockwise) and
      paint the sprite there using the target marker's color.

    Key insight:
    - The task is no longer a mirror-copy of a symmetric shape. The solver must preserve the
      exact asymmetric source sprite, transport it by corner-dependent rotation, and understand
      that the source is consumed: only rotated target sprites remain in the output.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}
    sources, _source_colors, canon, marker_canon = _extract_sources_and_canon(per_corner)

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
            continue
        tcol = v
        for off, color in per_corner[c_key]:
            if off in canon:
                assert color == tcol, (
                    f"target ({r},{c},{kind}) has conflicting color {color} at "
                    f"canonical offset {off}; expected {tcol}"
                )
        targets.append((r, c, kind, tcol))

    assert targets, "no target corners detected (every corner appears to be a source)"

    out = g.copy()
    for r, c, kind in sources:
        for cdr, cdc in canon:
            adr, adc = _to_actual(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            out[rr, cc] = _base_value_for_offset((cdr, cdc), bg)

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


# --------------- generator (output-first asymmetric rotation+erase movement) ---------------

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
MAX_ATTEMPTS = 400

# --- canonical TL-oriented asymmetric sprite templates ---
# Every template:
# - stays within |dr|,|dc| <= 3
# - has >=2 outward cells and >=1 inward-diagonal cell
# - is not diagonally symmetric
# - differs from the legacy reflection behavior on TR/BL corners
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 1)],
    [(-1, 0), (0, -1), (-1, -1), (1, 1), (2, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2), (2, 1)],
    [(-1, 0), (0, -1), (-1, -1), (1, 1), (2, 2), (1, 2)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2), (3, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 1)],
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


def _pattern_has_rotation_effect(pattern: list[tuple[int, int]]) -> bool:
    """Reject patterns whose quarter-turn transport collapses to the legacy reflection rule."""
    for kind in ("TR", "BL"):
        new_cells = {_to_actual(dr, dc, kind) for dr, dc in pattern}
        old_cells = {_legacy_reflect(dr, dc, kind) for dr, dc in pattern}
        if new_cells != old_cells:
            return True
    return False


def _choose_pattern() -> tuple[list[tuple[int, int]], int]:
    """Return `(pattern, k_max)` where `k_max` is the deepest inward-diagonal marker depth."""
    pattern = list(random.choice(PATTERN_TEMPLATES))
    assert not _is_diagonal_symmetric(pattern), "variant pattern unexpectedly symmetric"
    assert _pattern_has_rotation_effect(pattern), "variant pattern lost rotation observability"
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    """Draw a `size x size` hollow 0-frame with TL corner at `(r, c)`."""
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
    """Place two frames either side-by-side or stacked."""
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
    """Place three frames in three of the four macro corner slots."""
    lo, hi = FRAME_SIZE_RANGE
    missing = random.choice(CORNER_KINDS)
    present = tuple(kind for kind in CORNER_KINDS if kind != missing)

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
    return [{"r": coords[k][0], "c": coords[k][1], "size": sizes[k]} for k in present]


def _paint_pattern(
    grid: np.ndarray, r: int, c: int, kind: str, pattern: list[tuple[int, int]], color: int
) -> None:
    for dr, dc in pattern:
        rr, cc = _to_actual(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _pattern_cells(
    r: int, c: int, kind: str, pattern: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cells = []
    for off in pattern:
        adr, adc = _to_actual(*off, kind)
        cells.append(((r + adr, c + adc), off))
    return cells


def _marker_cell(r: int, c: int, kind: str, k_max: int) -> tuple[int, int]:
    mdr, mdc = _to_actual(k_max, k_max, kind)
    return r + mdr, c + mdc


def _pattern_matches_base(
    base: np.ndarray, r: int, c: int, kind: str, pattern: list[tuple[int, int]], bg: int
) -> bool:
    H, W = base.shape
    for (rr, cc), off in _pattern_cells(r, c, kind, pattern):
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if int(base[rr, cc]) != _base_value_for_offset(off, bg):
            return False
    return True


def _state_cells(
    frames: list[dict],
    pattern: list[tuple[int, int]],
    k_max: int,
    sources: set[tuple[int, str]],
    *,
    include_sources_full: bool,
    include_targets_full: bool,
) -> set[tuple[int, int]] | None:
    occupied = set()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            is_source = (fi, kind) in sources
            if is_source and include_sources_full:
                for (rr, cc), _off in _pattern_cells(r, c, kind, pattern):
                    if (rr, cc) in occupied:
                        return None
                    occupied.add((rr, cc))
            elif (not is_source) and include_targets_full:
                for (rr, cc), _off in _pattern_cells(r, c, kind, pattern):
                    if (rr, cc) in occupied:
                        return None
                    occupied.add((rr, cc))
            elif not is_source:
                marker = _marker_cell(r, c, kind, k_max)
                if marker in occupied:
                    return None
                occupied.add(marker)
    return occupied


def _rotation_is_visible(frames: list[dict], pattern: list[tuple[int, int]], sources: set[tuple[int, str]]) -> bool:
    """Require at least one output target where quarter-turn differs from legacy reflection."""
    for fi, frame in enumerate(frames):
        for kind in ("TR", "BL"):
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            new_cells = {pos for pos, _ in _pattern_cells(r, c, kind, pattern)}
            old_cells = {(r + rr, c + cc) for rr, cc in (_legacy_reflect(dr, dc, kind) for dr, dc in pattern)}
            if new_cells != old_cells:
                return True
    return False


def _layout_is_legal(
    base: np.ndarray,
    frames: list[dict],
    pattern: list[tuple[int, int]],
    k_max: int,
    sources: set[tuple[int, str]],
    bg: int,
) -> bool:
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
        for (rr, cc), _off in _pattern_cells(r, c, kind, pattern):
            if (rr, cc) not in output_cells:
                removed += 1
    return removed > 0 and _rotation_is_visible(frames, pattern, sources)


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    """Give every corner of every frame an independently sampled non-bg color."""
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]] | None:
    """Choose 1..N_SOURCES_MAX source corners from frames large enough for full sprites."""
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


def _render_output(
    base: np.ndarray,
    frames: list[dict],
    pattern: list[tuple[int, int]],
    sources: set[tuple[int, str]],
) -> np.ndarray:
    """Output contains only rotated target sprites; source sprites are absent."""
    out = base.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
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
    """Add full source sprites back, and reduce each target to a single marker."""
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            if (fi, kind) in sources:
                _paint_pattern(inp, r, c, kind, pattern, frame["corner_colors"][kind])
                continue
            marker = _marker_cell(r, c, kind, k_max)
            for (rr, cc), _off in _pattern_cells(r, c, kind, pattern):
                if (rr, cc) != marker:
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

    _assign_corner_colors(frames, bg)
    sources = _pick_sources(frames)
    if sources is None:
        return None
    if not _layout_is_legal(base, frames, pattern, k_max, sources, bg):
        return None

    output = _render_output(base, frames, pattern, sources)
    inp = _derive_input(output, base, frames, pattern, k_max, sources)

    try:
        corners = _find_corners(inp, bg)
        per_corner = {c: _colored_around(inp, *c, bg) for c in corners}
        _sources, _source_colors, canon, _marker = _extract_sources_and_canon(per_corner)
        if _is_diagonal_symmetric(canon):
            return None
    except Exception:
        return None

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
