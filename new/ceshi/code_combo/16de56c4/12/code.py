# Combo variant of 16de56c4 (index 12)
# combo_spec: D1/0+D14/1
# Name: GCD多点间距+不等数量标记

from __future__ import annotations

from collections import defaultdict
from math import gcd
import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# ----------------
# a) Concrete behavior:
#    - Original 16de56c4 used exactly two same-colored marker dots on each
#      active shooting line, so the period was read directly from one gap.
#    - This combo keeps the same global scene: one winning boundary chooses the
#      shooting direction, each active line has one repeated marker color, and a
#      different-colored singleton may act as either aligned companion or
#      off-beat bystander.
#    - The fused change is that a marker color may now appear 2-4 times.  When
#      there are 3-4 markers, the visible gaps can be unequal and the real step
#      is the GCD of adjacent marker gaps.
#    - Small trace on a left-shooting row:
#        old input line: marker at [0, 6], bystander at [5]
#        old step: 6, filled beats: [0, 6]
#        combo input line: marker at [0, 6, 9], bystander at [5]
#        combo gaps: [6, 3], step = gcd(6, 3) = 3, filled beats: [0, 3, 6, 9]
#      The difference is visible because a new beat appears between the first
#      two markers.
#    - Lines with only 2 markers are still legal.  They behave exactly like the
#      original pair case, which makes the 3-4 marker lines stand out as the
#      new ingredient instead of replacing every line with one pattern.
#
# b) Complexity calibration:
#    - The generator mixes marker counts 2, 3, and 4 rather than forcing one
#      count everywhere.  This matches the "2-4 markers" rule while keeping the
#      scene readable: the repeated color is still the only multi-dot color on
#      its line.
#    - Every generated grid is forced to contain at least one "GCD-critical"
#      line whose first visible gap is larger than the true step and whose later
#      gaps expose a smaller common divisor.  Without this, the combo could
#      silently collapse back to pair-spacing behavior.
#    - Other lines may remain simple pairs or uniform multi-marker lines, which
#      gives the solver repeated evidence for both the unchanged and changed
#      regimes without adding unrelated freedom.
#
# c) Old assumptions that must change:
#    - `len(mpos) == 2` is no longer valid; the solver must accept 2-4 markers.
#    - `step = b - a` is only valid for the 2-marker subcase.  In general the
#      solver must sort all marker positions and reduce adjacent gaps by GCD.
#    - The repeated-color lookup should still return exactly one marker color
#      per active line; allowing many repeated colors would create ambiguous
#      fills and make companion attribution unclear.
#    - Companion detection still depends on modular alignment, but the modular
#      class now comes from the GCD-derived step.  For a 3-4 marker line, an
#      aligned singleton may match the true step even when it does not align to
#      the first visible marker gap.
#    - The write guard `cur == bg or cur == marker` remains safe because the
#      generator still uses at most one singleton extra color on a line and that
#      extra color never overlaps the marker positions.
#
# d) Placement legality:
#    - Marker positions are built on a beat lattice emanating from the shooting
#      edge.  For 3-4 marker GCD lines, the generator chooses a non-uniform
#      subsequence of beats whose adjacent beat-gaps have GCD 1, then scales it
#      by the chosen base step.
#    - One marker always sits on the shooting edge, extra markers stay inside
#      bounds, companions are placed on a later beat beyond the outermost marker
#      while avoiding the opposite boundary, and bystanders are placed off-beat.
#    - All dots are checked for overlap before placement, and the finished grid
#      is validated by running `main()` on it.
#
# e) Visual observability:
#    - The combo is visible from examples because at least one line per grid has
#      3-4 markers with unequal gaps, yet the output paints a denser regular
#      rhythm whose spacing equals the GCD rather than any single large gap.
#    - Pair-marker lines remain present, so a human can compare "plain pair"
#      lines against "multi-marker GCD" lines within the same task family.
#    - No hidden use of numeric color ordering is involved; humans only need to
#      read repetition, alignment, and spacing.
#
# f) Visual cleanliness:
#    - Each active line contains one repeated color and at most one singleton of
#      another color.  That keeps roles legible and prevents multiple foreign
#      dots from obscuring which object is the companion.
#    - Opposite-edge placements are avoided where practical so the winning edge
#      remains visually unambiguous and the sparse-dot layout stays close to the
#      original task style.
#
# g) Change inferability:
#    - Color changes remain structurally cued.  If a singleton lands on the same
#      beat class as the markers, it becomes the companion color and bounds the
#      fill span; otherwise it is a bystander and stays unchanged.
#    - Because the only new inference burden is "derive step from 2-4 markers,
#      sometimes via GCD", the examples stay within a human-solvable ARC-scale
#      rule set rather than introducing arbitrary color remapping.
#
# Post-change solver assumptions reviewed against the generator:
#    - Exactly one repeated color per active line: guaranteed.
#    - Marker count is always between 2 and 4: guaranteed.
#    - Every marker position on a line shares one beat class under the GCD step:
#      guaranteed by construction.
#    - At most one aligned non-marker color per line: guaranteed.
#    - At least one line per generated grid is GCD-critical, so the combo does
#      not degenerate into pure 2-marker behavior.


def _gcd_of_gaps(positions: list[int]) -> int:
    assert len(positions) >= 2, f"Need at least two marker positions: {positions}"
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    step = gaps[0]
    for gap in gaps[1:]:
        step = gcd(step, gap)
    return step


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill shooting-line beats whose period may need a multi-gap GCD.

    Input structure:
    - The background is the most common color and all non-background cells are
      sparse dots.
    - Exactly one boundary has strictly more non-background dots than the other
      three; that boundary is the shooting edge and fixes whether rays travel
      along rows or columns.
    - Each active shooting-direction line has exactly one repeated marker color
      appearing 2-4 times.  Those marker positions all lie on one beat lattice,
      but for 3-4 markers the consecutive gaps need not be equal.
    - A line may also contain one singleton dot of another color.  If it lands
      on the same beat class it is a companion; otherwise it is a bystander.

    Rule:
    - Find the unique shooting edge; left/right imply row-wise lines, while
      top/bottom imply column-wise lines.
    - On each active line, sort the repeated marker positions.  If there are
      two markers, their gap is the step; otherwise compute the step as the GCD
      of all adjacent marker gaps.  The beat offset is the first marker modulo
      that step.
    - If a different-colored dot shares that beat class, it is the companion:
      paint with the companion color and only within the span covered by the
      marker dots together with that companion dot.
    - If no aligned companion exists, paint the whole line on that beat class
      using the marker color.  Off-beat singleton dots remain unchanged.

    Key insight: the densest boundary still reveals the shooting direction, but
    the visible marker spacing is no longer always the true rhythm.  The solver
    must sometimes combine several unequal marker gaps and recover their common
    divisor before it can decide which cells belong to the periodic ray.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge_counts = {
        "top": sum(1 for c in range(W) if g[0, c] != bg),
        "bottom": sum(1 for c in range(W) if g[H - 1, c] != bg),
        "left": sum(1 for r in range(H) if g[r, 0] != bg),
        "right": sum(1 for r in range(H) if g[r, W - 1] != bg),
    }
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg dots on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    edge = winners[0]

    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for pos in range(line_len):
            value = int(g[cell(li, pos)])
            if value != bg:
                colors[value].append(pos)

        if not colors:
            continue

        repeating = {color: sorted(posns) for color, posns in colors.items() if len(posns) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has repeating colors "
            f"{list(repeating)}; expected exactly one"
        )
        marker, mpos = next(iter(repeating.items()))
        assert 2 <= len(mpos) <= 4, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has {len(mpos)} dots (expected 2-4): {mpos}"
        )

        step = _gcd_of_gaps(mpos)
        assert step > 0, f"Non-positive step on line {li}: {mpos}"
        offset = mpos[0] % step
        assert all(pos % step == offset for pos in mpos), (
            f"Marker positions on line {li} do not share one beat class: {mpos}"
        )

        aligned_others = [
            other_color
            for other_color, other_positions in colors.items()
            if other_color != marker and any(pos % step == offset for pos in other_positions)
        ]
        assert len(aligned_others) <= 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has multiple companions: {aligned_others}"
        )
        companion = aligned_others[0] if aligned_others else None

        if companion is not None:
            paint = companion
            extent_positions = mpos + colors[companion]
            lo, hi = min(extent_positions), max(extent_positions)
        else:
            paint = marker
            lo, hi = 0, line_len - 1

        for pos in range(lo, hi + 1):
            if pos % step != offset:
                continue
            rc = cell(li, pos)
            current = int(out[rc])
            if current == bg or current == marker:
                out[rc] = paint

    return out


# --------------- generator (combo variant) ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_MARKER_COUNT_RANGE = (2, 4)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.35
_P_COMPANION = 0.35

_MAX_RETRIES = 500


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
    min_num_lines = 2 * n_active - 1
    by_row = edge in ("left", "right")
    if by_row:
        lo = max(_ROWS_RANGE[0], min_num_lines)
        if lo > _ROWS_RANGE[1]:
            return None
        H = random.randint(lo, _ROWS_RANGE[1])
        W = random.randint(*_COLS_RANGE)
        return H, W, H, W

    lo = max(_COLS_RANGE[0], min_num_lines)
    if lo > _COLS_RANGE[1]:
        return None
    W = random.randint(lo, _COLS_RANGE[1])
    H = random.randint(*_ROWS_RANGE)
    return H, W, W, H


def _spaced_indices(n: int, total: int) -> list[int]:
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _choose_line_kind() -> str:
    roll = random.random()
    if roll < _P_MARKER_ONLY:
        return "marker_only"
    if roll < _P_MARKER_ONLY + _P_COMPANION:
        return "companion"
    return "bystander"


def _profile_feasible(
    line_len: int,
    step: int,
    marker_count: int,
    critical: bool,
    need_room_after: bool,
) -> bool:
    max_beat = (line_len - 1) // step
    reserve = 2 if need_room_after else 1
    max_last = max_beat - reserve
    if marker_count == 2:
        return max_last >= 1
    if critical:
        return max_last >= marker_count
    return max_last >= marker_count - 1


def _build_marker_indices(
    max_beat: int,
    marker_count: int,
    critical: bool,
    need_room_after: bool,
    edge: str,
) -> list[int] | None:
    reserve = 2 if need_room_after else 1
    max_last = max_beat - reserve

    if marker_count == 2:
        if max_last < 1:
            return None
        return [0, random.randint(1, max_last)]

    if critical:
        if max_last < marker_count:
            return None

        increments = [1] * (marker_count - 1)
        big_idx = 0 if edge in ("left", "top") else marker_count - 2
        increments[big_idx] = 2
        capacities = [2] * (marker_count - 1)
        capacities[big_idx] = 1
        protected_choices = [idx for idx in range(marker_count - 1) if idx != big_idx]
        protected = random.choice(protected_choices)
        capacities[protected] = 0

        spare = min(max_last - sum(increments), sum(capacities))
        extra = random.randint(0, spare)
        while extra > 0:
            choices = [i for i, cap in enumerate(capacities) if cap > 0]
            idx = random.choice(choices)
            increments[idx] += 1
            capacities[idx] -= 1
            extra -= 1

        indices = [0]
        for inc in increments:
            indices.append(indices[-1] + inc)
        return indices

    if max_last < marker_count - 1:
        return None
    beat_gap = random.randint(1, max_last // (marker_count - 1))
    return [i * beat_gap for i in range(marker_count)]


def _positions_from_indices(edge: str, line_len: int, step: int, beat_indices: list[int]) -> list[int]:
    if edge in ("left", "top"):
        return [idx * step for idx in beat_indices]
    edge_pos = line_len - 1
    return [edge_pos - idx * step for idx in beat_indices]


def _candidate_marker_profiles(require_critical: bool) -> list[tuple[int, bool]]:
    if require_critical:
        return [(3, True), (3, True), (4, True)]
    return [
        (2, False),
        (2, False),
        (2, False),
        (3, False),
        (3, False),
        (4, False),
        (3, True),
        (4, True),
    ]


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    require_critical: bool,
) -> tuple[list[tuple[int, int]], dict[str, int | bool]] | None:
    kind = _choose_line_kind()
    opposite_pos = line_len - 1 if edge in ("left", "top") else 0
    max_step = min(_STEP_RANGE[1], line_len - 1)

    def feasible_choices(line_kind: str) -> list[tuple[int, bool, int]]:
        choices: list[tuple[int, bool, int]] = []
        for marker_count, critical in _candidate_marker_profiles(require_critical):
            for step in range(_STEP_RANGE[0], max_step + 1):
                if not _profile_feasible(line_len, step, marker_count, critical, line_kind == "companion"):
                    continue
                if line_kind == "bystander" and step == 1:
                    continue
                choices.append((marker_count, critical, step))
        return choices

    choices = feasible_choices(kind)
    if not choices and kind != "marker_only":
        kind = "marker_only"
        choices = feasible_choices(kind)
    if not choices:
        return None

    marker_count, critical, step = random.choice(choices)
    max_beat = (line_len - 1) // step
    beat_indices = _build_marker_indices(max_beat, marker_count, critical, kind == "companion", edge)
    assert beat_indices is not None
    marker_positions = _positions_from_indices(edge, line_len, step, beat_indices)
    offset = marker_positions[0] % step

    dots = [(pos, marker_color) for pos in marker_positions]

    if kind == "companion":
        later_beats = list(range(beat_indices[-1] + 1, max_beat))
        if not later_beats:
            return None
        other_colors = [color for color in _COLORS if color != marker_color]
        companion_pos = _positions_from_indices(edge, line_len, step, [random.choice(later_beats)])[0]
        dots.append((companion_pos, random.choice(other_colors)))

    if kind == "bystander":
        other_colors = [color for color in _COLORS if color != marker_color]
        non_beat = [
            pos
            for pos in range(line_len)
            if pos % step != offset and pos not in marker_positions and pos != opposite_pos
        ]
        if not other_colors or not non_beat:
            return None
        dots.append((random.choice(non_beat), random.choice(other_colors)))

    return dots, {
        "step": step,
        "marker_count": marker_count,
        "critical": critical,
        "kind_is_companion": kind == "companion",
    }


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]],
    edge: str,
    H: int,
    W: int,
) -> dict[str, int]:
    boundary = {
        "top": lambda r, c: r == 0,
        "bottom": lambda r, c: r == H - 1,
        "left": lambda r, c: c == 0,
        "right": lambda r, c: c == W - 1,
    }
    counts = {}
    for other_edge in _EDGES:
        if other_edge == edge:
            continue
        check = boundary[other_edge]
        counts[other_edge] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _line_signature(values: list[int], bg: int = 0) -> tuple[int, int, int, bool]:
    colors: dict[int, list[int]] = defaultdict(list)
    for pos, value in enumerate(values):
        if value != bg:
            colors[value].append(pos)
    repeating = [sorted(posns) for posns in colors.values() if len(posns) >= 2]
    assert len(repeating) == 1, f"Expected one repeated marker color, got {colors}"
    marker_positions = repeating[0]
    gaps = [b - a for a, b in zip(marker_positions, marker_positions[1:])]
    first_gap = gaps[0]
    return len(marker_positions), _gcd_of_gaps(marker_positions), first_gap, len(set(gaps)) > 1


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None
    forced_critical_line = random.choice(active_lines)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    critical_lines = 0
    multi_marker_lines = 0

    for li in active_lines:
        marker_color = random.choice(_COLORS)
        result = _build_line_dots(edge, line_len, marker_color, require_critical=li == forced_critical_line)
        if result is None:
            return None
        dots, meta = result

        seen_positions = set()
        for pos, color in dots:
            if pos in seen_positions:
                return None
            seen_positions.add(pos)
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        values = [int(inp[cell(li, pos)]) for pos in range(line_len)]
        marker_count, gcd_step, first_gap, is_nonuniform = _line_signature(values)
        multi_marker_lines += int(marker_count >= 3)
        critical_lines += int(marker_count >= 3 and is_nonuniform and first_gap != gcd_step)
        shooting_edge_dots += 1

        if li == forced_critical_line and not (marker_count >= 3 and is_nonuniform and first_gap != gcd_step):
            return None
        if marker_count != int(meta["marker_count"]):
            return None

    if critical_lines == 0 or multi_marker_lines == 0:
        return None

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
