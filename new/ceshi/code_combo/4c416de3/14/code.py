# Combo variant of 4c416de3 (index 14)
# combo_spec: D1/3+D10/2
# Name: 控制像素指示变换+旋转映射

"""
Design analysis for combo variant D1/3 + D10/2
==============================================

spec concretization
-------------------
- Keep the original scene grammar: 2-3 hollow 0-frames on one solid non-zero
  background, with a shared corner glyph copied across corners.
- Keep D1/3's asymmetric quarter-turn glyph family. The canonical glyph is still
  defined in TL-local coordinates, but it is no longer diagonal-mirror symmetric.
  That asymmetry is required so a quarter-turn visibly changes the hook shape.
- Add one extra control pixel near every target corner. The target color still comes
  from the remaining visible marker pixel, while the control pixel's *position*
  chooses the target's extra local quarter-turn.
- Sources stay fully drawn and have no control pixel. They establish the canonical
  step-0 glyph. Targets show exactly two colored pixels in the input:
  1. the output-color marker, which is one real glyph cell after the chosen turn;
  2. the control pixel, placed in one of four reserved slots that encode 0/90/180/270.

behavior trace on a concrete corner
-----------------------------------
- Suppose the canonical glyph contains the asymmetric cell `(2, 1)` and the deepest
  diagonal marker is `(1, 1)`.
- Old original task: a TR corner always used the corner kind itself to decide the
  transport. The corner type alone fixed the result.
- D1/3 alone: that TR corner would always receive the canonical glyph turned 90 deg
  clockwise relative to TL, so `(2, 1) -> (1, -2)`.
- This combo: first normalize the target corner into TL-local coordinates; then read
  its control slot. If the control pixel sits in the "180 deg" slot, the target uses
  an extra 180 deg local turn, so `(2, 1) -> (-2, -1)` before the corner-kind embed.
  The color still comes from the visible marker pixel. The difference is now driven by
  the control slot instead of being forced by the corner type alone.

complexity calibration
----------------------
- Only one new symbolic channel is added: a single control pixel per target, with four
  fixed slot positions. This keeps the degrees of freedom small and repeated.
- The glyph templates remain small (5-6 cells) and asymmetric, so a non-zero control
  turn visibly changes the footprint.
- Generation intentionally repeats exactly two step values per grid whenever enough
  targets exist: step 0 and one non-zero quarter-turn. That gives human-solvable
  repetition instead of a one-off arbitrary mapping.

implicit assumption review of the solver
----------------------------------------
- `_to_actual` / `_to_canon` from D1/3 are still required because the base
  source-extraction logic depends on canonical TL-local coordinates.
- The old diagonal-symmetry assert is invalid under D1/3 and remains removed.
- The original target test assumed the marker always stayed at canonical `(k, k)` after
  the corner-kind mapping. That assumption breaks here: the marker position depends on
  the control-selected local quarter-turn. The solver must decode the control slot
  first, then rotate `(k, k)` accordingly.
- The original source detector counted outward cells directly. With control pixels
  added, that would misclassify some targets as sources when the rotated marker also
  lands outward. The fix is to remove reserved control-slot offsets from the source
  test before counting outward cells.
- The new solver assumes control pixels never overlap true glyph cells. The generator
  guarantees this by using reserved slots outside the union of all rotated templates.

placement legality
------------------
- Control slots use offsets with magnitude 3, so the old `BORDER=1` is no longer safe.
  The generator raises the minimum border gap to 3 and uses slightly larger grids.
- A generalized layout-fit check rejects any sample where:
  - a target control pixel would land on a frame edge or outside the grid,
  - two corner outputs would overlap,
  - any input-only control pixel would collide with another corner's source pattern,
    target marker, or another control.
- The generator remains output-first: render the final turned glyphs first, then erase
  target glyphs back to one marker and add one control pixel.

visual observability and cleanliness
------------------------------------
- The control pixel is a singleton in a reserved slot and uses the same color as that
  target's marker, so the pair is visually grouped without needing an extra legend.
- The per-grid step palette is intentionally small and repeated, so the mapping from
  slot-position to local rotation is inferable from 2-4 examples.
- Because the final output contains no control pixels, the input/output difference is
  clean: isolated guide pixels disappear and are replaced by a full rotated glyph.
- Asymmetry in the glyph ensures the control really matters. A 180 deg or 90 deg turn
  produces a visibly different hook from the step-0 source.

new solver assumptions audited against the generator
----------------------------------------------------
- Every non-source corner has exactly one control pixel and exactly one non-control
  colored pixel left in the input.
- That non-control pixel is exactly the deepest diagonal marker after the selected
  local quarter-turn.
- All sources share one canonical step-0 glyph and contain no control-slot pixels.
- At least one target uses a non-zero turn, and the selected glyph templates are
  asymmetric enough that the output differs from the step-0 baseline on those targets.
"""

from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np

try:
    from arc_utils import draw_line, get_bg
except ImportError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
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

# Control-slot orbit: a target keeps one control pixel in exactly one of these TL-local
# slots; the chosen slot encodes the extra local quarter-turn.
CONTROL_SLOTS = {
    0: (-1, 3),
    1: (3, 1),
    2: (1, -3),
    3: (-3, -1),
}
SLOT_TO_STEP = {off: step for step, off in CONTROL_SLOTS.items()}


def _rotate_cw(dr: int, dc: int, steps: int) -> tuple[int, int]:
    steps %= 4
    for _ in range(steps):
        dr, dc = dc, -dr
    return dr, dc


def _to_actual(dr: int, dc: int, kind: str) -> tuple[int, int]:
    return _rotate_cw(dr, dc, _ROT_STEPS[kind])


def _to_canon(dr: int, dc: int, kind: str) -> tuple[int, int]:
    return _rotate_cw(dr, dc, -_ROT_STEPS[kind])


def _step_actual(dr: int, dc: int, kind: str, step: int) -> tuple[int, int]:
    return _to_actual(*_rotate_cw(dr, dc, step), kind)


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
            items.append((_to_canon(dr, dc, kind), v))
    return items


def _split_control(items):
    controls = []
    payload = []
    for off, color in items:
        if off in SLOT_TO_STEP:
            controls.append((off, color))
        else:
            payload.append((off, color))
    return controls, payload


def _color_map(items):
    mp = {}
    for off, color in items:
        if off in mp:
            assert mp[off] == color, f"conflicting colors at canonical offset {off}: {mp[off]} vs {color}"
        mp[off] = color
    return mp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a source corner glyph to marked targets, with each target's turn chosen by a control pixel.

    Input structure:
    - The background is the most frequent non-zero color. Each object is a hollow 0-frame.
    - One or more source corners already contain the full asymmetric glyph in a single color and have no
      control pixel.
    - Every target corner keeps exactly two same-color guide pixels: one true glyph marker and one extra
      control pixel in a reserved nearby slot. The marker gives the output color; the control slot gives the
      target's extra local quarter-turn.

    Rule:
    - Read any source corner in canonical TL-local coordinates to recover the shared step-0 glyph.
    - For each target, decode the extra turn from the control slot, find the marker at the turned deepest
      diagonal cell, erase the input-only control pixel, and paint the whole turned glyph in the marker's color.

    Key insight:
    - Corner type still tells you how to normalize a neighborhood, but it no longer decides the target
      orientation by itself. The visible singleton control pixel is the new instruction channel: its position
      selects 0/90/180/270 within the corner's local frame, and the asymmetric glyph makes that choice visible.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}
    split = {c: _split_control(items) for c, items in per_corner.items()}

    def _is_outward(off):
        return off[0] < 0 or off[1] < 0

    sources = []
    for c_key in corners:
        controls, payload = split[c_key]
        if controls:
            continue
        outward = [(off, v) for off, v in payload if _is_outward(off)]
        if len(outward) >= 2:
            colors = {v for _, v in payload}
            assert len(colors) == 1, f"source {c_key} has mixed payload colors: {colors}"
            sources.append(c_key)
    assert sources, "at least one source corner required"

    def _extract(c_key):
        _controls, payload = split[c_key]
        pat_col = Counter(v for _, v in payload).most_common(1)[0][0]
        return frozenset(off for off, v in payload if v == pat_col), pat_col

    canon, _ = _extract(sources[0])
    for s in sources[1:]:
        other, _ = _extract(s)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal marker anchor"
    marker_canon = (max(diag_ks), max(diag_ks))

    src_set = set(sources)
    targets = []
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        controls, payload = split[c_key]
        assert len(controls) == 1, f"target {c_key} must have exactly one control pixel"
        (ctrl_off, ctrl_color) = controls[0]
        step = SLOT_TO_STEP[ctrl_off]
        expected_marker = _rotate_cw(*marker_canon, step)

        payload_map = _color_map(payload)
        assert expected_marker in payload_map, (
            f"target {c_key} missing marker at turned canonical offset {expected_marker}"
        )
        tcol = payload_map[expected_marker]
        assert ctrl_color == tcol, f"target {c_key} control color {ctrl_color} != marker color {tcol}"
        assert set(payload_map) == {expected_marker}, (
            f"target {c_key} has extra non-control pixels: {sorted(payload_map)}"
        )
        targets.append((r, c, kind, step, tcol, ctrl_off))

    assert targets, "no target corners detected"
    assert len(src_set) + len(targets) == len(corners), "every detected corner must be source or target"

    out = g.copy()
    for r, c, kind, _step, _tcol, ctrl_off in targets:
        cdr, cdc = _to_actual(*ctrl_off, kind)
        rr, cc = r + cdr, c + cdc
        out[rr, cc] = bg

    for r, c, kind, step, tcol, _ctrl_off in targets:
        for cdr, cdc in canon:
            adr, adc = _step_actual(cdr, cdc, kind, step)
            rr, cc = r + adr, c + adc
            assert 0 <= rr < H and 0 <= cc < W, f"turned glyph left grid at {(rr, cc)}"
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == tcol, (
                f"paint conflict at ({rr},{cc}) for target ({r},{c},{kind}) "
                f"turn {step} color {tcol}: existing color {v}"
            )
            out[rr, cc] = tcol

    return out


# --------------- generator (output-first combo D1/3 + D10/2) ---------------

H_RANGE = (26, 30)
W_RANGE = (26, 30)
BORDER = 3
FRAME_GAP = 4
FRAME_SIZE_RANGE = (8, 8)
SOURCE_MIN_SIZE = 8

N_FRAMES_2_PROB = 0.65
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2

MAX_ATTEMPTS = 500

# Cycle the non-zero branch across calls so the 90/180/270 control-slot variants
# all appear regularly in generated examples instead of whichever one happens to fit
# layouts most often.
_ALT_STEP_ORDER = [1, 2, 3]
random.shuffle(_ALT_STEP_ORDER)
_ALT_STEP_INDEX = 0

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


def _pattern_has_turn_effect(pattern: list[tuple[int, int]]) -> bool:
    base = set(pattern)
    return any({_rotate_cw(dr, dc, step) for dr, dc in pattern} != base for step in (1, 2, 3))


def _choose_pattern() -> tuple[list[tuple[int, int]], int]:
    pattern = list(random.choice(PATTERN_TEMPLATES))
    assert _pattern_has_turn_effect(pattern)
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
    grid: np.ndarray,
    r: int,
    c: int,
    kind: str,
    pattern: list[tuple[int, int]],
    color: int,
    step: int,
) -> None:
    for dr, dc in pattern:
        rr, cc = _step_actual(dr, dc, kind, step)
        grid[r + rr, c + cc] = color


def _pattern_cells(r: int, c: int, kind: str, pattern: list[tuple[int, int]], step: int) -> set[tuple[int, int]]:
    cells = set()
    for dr, dc in pattern:
        rr, cc = _step_actual(dr, dc, kind, step)
        cells.add((r + rr, c + cc))
    return cells


def _marker_cell(r: int, c: int, kind: str, k_max: int, step: int) -> tuple[int, int]:
    rr, cc = _step_actual(k_max, k_max, kind, step)
    return r + rr, c + cc


def _control_cell(r: int, c: int, kind: str, step: int) -> tuple[int, int]:
    rr, cc = _to_actual(*CONTROL_SLOTS[step], kind)
    return r + rr, c + cc


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
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


def _next_alt_step() -> int:
    global _ALT_STEP_INDEX
    alt = _ALT_STEP_ORDER[_ALT_STEP_INDEX]
    _ALT_STEP_INDEX = (_ALT_STEP_INDEX + 1) % len(_ALT_STEP_ORDER)
    if _ALT_STEP_INDEX == 0:
        random.shuffle(_ALT_STEP_ORDER)
    return alt


def _assign_target_steps(
    frames: list[dict],
    sources: set[tuple[int, str]],
    alt: int,
) -> dict[tuple[int, str], int] | None:
    targets = [
        (fi, kind)
        for fi, _frame in enumerate(frames)
        for kind in CORNER_KINDS
        if (fi, kind) not in sources
    ]
    if not targets:
        return None

    random.shuffle(targets)
    steps = {}
    if len(targets) == 1:
        steps[targets[0]] = alt
        return steps

    steps[targets[0]] = 0
    steps[targets[1]] = alt
    if len(targets) >= 4:
        steps[targets[2]] = 0
        steps[targets[3]] = alt
    for target in targets:
        steps.setdefault(target, random.choice([0, alt]))
    assert any(step != 0 for step in steps.values())
    return steps


def _layout_fits(
    base: np.ndarray,
    frames: list[dict],
    pattern: list[tuple[int, int]],
    k_max: int,
    sources: set[tuple[int, str]],
    target_steps: dict[tuple[int, str], int],
) -> bool:
    H, W = base.shape
    output_occ: set[tuple[int, int]] = set()
    input_occ: set[tuple[int, int]] = set()
    corner_input: dict[tuple[int, str], set[tuple[int, int]]] = {}

    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            step = 0 if (fi, kind) in sources else target_steps[(fi, kind)]
            out_cells = _pattern_cells(r, c, kind, pattern, step)

            for rr, cc in out_cells:
                if not (0 <= rr < H and 0 <= cc < W):
                    return False
            if output_occ & out_cells:
                return False
            output_occ |= out_cells

            if (fi, kind) in sources:
                if input_occ & out_cells:
                    return False
                input_occ |= out_cells
                corner_input[(fi, kind)] = set(out_cells)
                continue

            marker = _marker_cell(r, c, kind, k_max, step)
            control = _control_cell(r, c, kind, step)
            if not (0 <= marker[0] < H and 0 <= marker[1] < W):
                return False
            if not (0 <= control[0] < H and 0 <= control[1] < W):
                return False
            if int(base[control]) != int(get_bg(base)):
                return False
            target_input = {marker, control}
            if len(target_input) != 2:
                return False
            if input_occ & target_input:
                return False
            input_occ |= target_input
            corner_input[(fi, kind)] = target_input

    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            local_seen = {
                cell
                for cell in input_occ
                if abs(cell[0] - r) <= _NEIGH_R and abs(cell[1] - c) <= _NEIGH_R
            }
            if local_seen != corner_input[(fi, kind)]:
                return False

    return True


def _render_output(
    base: np.ndarray,
    frames: list[dict],
    pattern: list[tuple[int, int]],
    sources: set[tuple[int, str]],
    target_steps: dict[tuple[int, str], int],
) -> np.ndarray:
    out = base.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            step = 0 if (fi, kind) in sources else target_steps[(fi, kind)]
            color = frame["corner_colors"][kind]
            _paint_pattern(out, r, c, kind, pattern, color, step)
    return out


def _derive_input(
    output: np.ndarray,
    base: np.ndarray,
    frames: list[dict],
    pattern: list[tuple[int, int]],
    k_max: int,
    sources: set[tuple[int, str]],
    target_steps: dict[tuple[int, str], int],
) -> np.ndarray:
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            step = target_steps[(fi, kind)]
            marker = _marker_cell(r, c, kind, k_max, step)
            for rr, cc in _pattern_cells(r, c, kind, pattern, step):
                if (rr, cc) != marker:
                    inp[rr, cc] = base[rr, cc]
            ctrl_r, ctrl_c = _control_cell(r, c, kind, step)
            inp[ctrl_r, ctrl_c] = frame["corner_colors"][kind]
    return inp


def _gen_attempt(alt: int) -> np.ndarray | None:
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
    target_steps = _assign_target_steps(frames, sources, alt)
    if target_steps is None:
        return None
    if not _layout_fits(base, frames, pattern, k_max, sources, target_steps):
        return None

    output = _render_output(base, frames, pattern, sources, target_steps)
    inp = _derive_input(output, base, frames, pattern, k_max, sources, target_steps)
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    alt = _next_alt_step()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(alt)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
