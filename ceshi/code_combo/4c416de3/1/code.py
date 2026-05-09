# Combo variant of 4c416de3 (index 1)
# combo_spec: D1/3+D5/0
# Name: 旋转映射+擦除源

"""
Design analysis for the combo variant
=====================================

a) Concretization and behavior trace
------------------------------------
- The scene still contains 2-3 hollow 0-frames on one solid non-zero background.
- A source corner still shows the full glyph in one color, while a target corner still
  shows only the deepest inward-diagonal marker.
- The combined rule changes two linked steps at once:
  1. Corner transport is no longer the original sign-flip family. Canonical TL offsets
     are sent by quarter-turns: TL=0deg, TR=90deg clockwise, BR=180deg, BL=90deg
     counterclockwise.
  2. The source glyph is not preserved in the output. It is removed back to the frame's
     base scaffold, so the glyph is moved rather than copied.
- Concrete example: suppose the canonical glyph contains {(-1,0), (0,-1), (1,1), (2,1)}.
  A TL source keeps these offsets in the input.
  Old task, TR target: sign flip sends (2,1) -> (2,-1), and the TL source remains.
  New combo, TR target: quarter-turn sends (2,1) -> (1,-2), and the TL source is erased.
  The output therefore differs both in shape placement and in source persistence.

b) Complexity calibration
-------------------------
- The variant keeps the original object vocabulary: same frame count, same marker logic,
  same source detection rule, same per-corner target color reuse.
- Complexity is raised only where the combo demands it:
  asymmetric templates make rotation distinguishable from reflection, and source erasure
  makes the trajectory a visible move.
- Templates are restricted to 5-6 cells with offsets inside |dr|,|dc| <= 3 so the task
  stays legible and the corner-local scan radius does not need to change.
- The generator keeps 2-3 frames and at most 2 sources, leaving multiple non-source TR/BL
  corners in each grid so the rotation effect has repeated visible support.

c) Hidden assumptions review
----------------------------
- Original `_rot` was self-inverse because the old mapping was built from reflections.
  That assumption breaks here, so the solver must split canonical->actual and
  actual->canonical into `_to_actual` and `_to_canon`.
- The original diagonal-symmetry assert was tied to the reflection variant and must be
  removed. The combo needs asymmetric glyphs so that TR/BL outputs differ from the base
  task.
- `out = g.copy()` followed by painting only targets is no longer correct because sources
  must disappear. The solver must first restore source cells to their base colors.
- That erase step assumes the base value under a glyph cell is structurally known:
  canonical offset `(0,0)` sits on the 0-frame corner, while every other allowed offset
  sits on background. The generator explicitly enforces this.
- Source/target classification by "has outward colored cells" remains valid because the
  corner neighborhood is canonicalized via inverse rotation before testing.

d) Placement legality
---------------------
- Every potential glyph cell must fit in-bounds and lie on the expected base scaffold:
  `(0,0)` on 0, all other glyph offsets on bg.
- The generator separately checks occupied colored cells for the input state
  (source full glyphs + target markers) and for the output state (target full glyphs only)
  so no state contains overlapping colored evidence.
- It also requires at least one visibly removed outward source cell and at least one
  non-source TR/BL corner whose rotated footprint differs from the legacy reflection rule.

e) Visual observability
-----------------------
- Human solvers can see both changes directly:
  1. the source's outward tail vanishes in the output;
  2. TR/BL targets bend the asymmetric hook by rotation rather than mirror.
- No hidden use of numeric color identity is introduced. Target color is still read from
  the visible marker already present in the input.

f) Visual cleanliness
---------------------
- Frames keep a minimum 4-cell inter-frame gap and glyphs are tiny, so local neighborhoods
  do not visually tangle.
- Rejecting layouts where painted states overlap prevents ambiguous mixed-color stacks.

g) Color inference
------------------
- Colors remain local and structurally grounded: each target uses its own marker color.
- Palette size is modest and no extra color mapping is layered on top of the new combo,
  so the only new inference burden is geometry plus source removal.

New solver assumptions checked against the generator
----------------------------------------------------
- At least one source exists and all sources encode the same canonical glyph.
- The glyph contains at least one inward diagonal `(k,k)` marker anchor.
- At least one target exists.
- At least one non-source TR or BL corner remains to expose quarter-turn behavior.
- At least one outward source pixel disappears in the output, so the move semantics is
  visible rather than degenerate.
"""

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line, get_bg


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
_ROT_STEPS = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}
_ROTATION_VISIBLE_KINDS = ("TR", "BL")


def _rotate_cw(dr: int, dc: int, steps: int) -> tuple[int, int]:
    """Rotate an offset around the corner by `steps` quarter-turns clockwise."""
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
    """Original task's reflection-style mapping, kept for anti-degeneracy checks."""
    if kind == "TL":
        return dr, dc
    if kind == "TR":
        return dr, -dc
    if kind == "BL":
        return -dr, dc
    return -dr, -dc


def _base_value_for_offset(off: tuple[int, int], bg: int) -> int:
    return 0 if off == (0, 0) else bg


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
            items.append((_to_canon(dr, dc, kind), v))
    return items


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a pre-drawn corner glyph onto other marked corners using quarter-turn transport.

    Input structure:
    - The most common color is the background, and hollow 0-frames provide the candidate
      corner sites.
    - One or more source corners already show the entire one-color glyph, including
      outward tail cells and inward cells.
    - Every participating non-source corner keeps only a single colored marker at the
      deepest inward-diagonal anchor; that marker color is the destination color.

    Rule:
    - Recover the shared canonical TL-oriented glyph from any source corner.
    - Erase that glyph from every source corner by restoring the underlying bg/0 scaffold.
    - For each marked target corner, rotate the canonical glyph by corner kind
      (TL=0deg, TR=90deg clockwise, BR=180deg, BL=90deg counterclockwise) and paint it
      in the target marker's color.

    Key insight:
    - The puzzle combines two independent visual cues: only sources have outward colored
      cells, which reveals the movable glyph, and the asymmetric glyph's TR/BL targets
      only match the examples if corner transport is read as rotation rather than mirror.
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

    canon, _ = _extract(sources[0])
    for s in sources[1:]:
        other, _ = _extract(s)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal cell to anchor markers"
    marker_canon = (max(diag_ks), max(diag_ks))

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


# --------------- generator (output-first movement with rotational transport) ---------------

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
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 1)],
    [(-1, 0), (0, -1), (-1, -1), (1, 1), (2, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2), (2, 1)],
    [(-1, 0), (0, -1), (-1, -1), (1, 1), (2, 2), (1, 2)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2), (3, 2)],
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


def _kind_has_rotation_effect(pattern: list[tuple[int, int]], kind: str) -> bool:
    new_cells = {_to_actual(dr, dc, kind) for dr, dc in pattern}
    old_cells = {_legacy_reflect(dr, dc, kind) for dr, dc in pattern}
    return new_cells != old_cells


def _choose_pattern() -> tuple[list[tuple[int, int]], int]:
    pattern = list(random.choice(PATTERN_TEMPLATES))
    assert any(_kind_has_rotation_effect(pattern, kind) for kind in _ROTATION_VISIBLE_KINDS)
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
    s1 = frame["size"] - 1
    return frame["r"] + dr * s1, frame["c"] + dc * s1


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
                cells = [pos for pos, _ in _pattern_cells(r, c, kind, pattern)]
            elif (not is_source) and include_targets_full:
                cells = [pos for pos, _ in _pattern_cells(r, c, kind, pattern)]
            elif not is_source:
                cells = [_marker_cell(r, c, kind, k_max)]
            else:
                cells = []
            for cell in cells:
                if cell in occupied:
                    return None
                occupied.add(cell)
    return occupied


def _has_visible_rotation_target(
    frames: list[dict], pattern: list[tuple[int, int]], sources: set[tuple[int, str]]
) -> bool:
    for fi, _frame in enumerate(frames):
        for kind in _ROTATION_VISIBLE_KINDS:
            if (fi, kind) not in sources and _kind_has_rotation_effect(pattern, kind):
                return True
    return False


def _has_visible_source_erasure(
    frames: list[dict],
    pattern: list[tuple[int, int]],
    sources: set[tuple[int, str]],
    output_cells: set[tuple[int, int]],
) -> bool:
    for fi, kind in sources:
        r, c = _corner_coord(frames[fi], kind)
        for (rr, cc), off in _pattern_cells(r, c, kind, pattern):
            if (off[0] < 0 or off[1] < 0) and (rr, cc) not in output_cells:
                return True
    return False


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    for frame in frames:
        palette = [color for color in range(1, 10) if color != bg]
        frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


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

    if not _has_visible_rotation_target(frames, pattern, sources):
        return False
    if not _has_visible_source_erasure(frames, pattern, sources, output_cells):
        return False
    return True


def _render_output(
    base: np.ndarray, frames: list[dict], pattern: list[tuple[int, int]], sources: set[tuple[int, str]]
) -> np.ndarray:
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

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
