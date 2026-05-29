# Variant of 16de56c4
# Dimension: D1 — 路径步长计算方式 (path)
# Change: step = 两个标记点之间的间距（即差值），offset = 第一个标记点位置 mod step... → GCD多点间距

from __future__ import annotations

from collections import defaultdict
from math import gcd
import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# ----------------
# a) Concrete behavior:
#    - The original task put exactly two same-colored marker dots on each active
#      shooting line, so the period was readable from a single gap.
#    - This variant keeps the same global layout (one winning edge, sparse dots,
#      optional companion/bystander), but each active line now has 3-5 marker
#      dots of one color. Their successive gaps are not all equal.
#    - The real period is the GCD of those successive gaps. Example on a
#      left-shooting row:
#        old line: marker at [0, 4] -> step = 4
#        new line: marker at [0, 4, 6] -> successive gaps = [4, 2], step = 2
#      The output therefore fills columns 0,2,4,6,... instead of only 0,4,8,...
#    - Companion / bystander behavior stays the same after the period is found:
#      an aligned foreign-color dot limits the fill span and sets the paint
#      color; an unaligned foreign-color dot is preserved.
#
# b) Complexity calibration:
#    - Every active line is forced to have at least one large gap and at least
#      one unit gap in beat-space. This makes the GCD inference visible instead
#      of collapsing back to evenly spaced markers.
#    - The line still contains only one repeated color and at most one extra
#      singleton dot, so the scene stays readable.
#
# c) Old assumptions that must change:
#    - `len(mpos) == 2` no longer holds; the solver must sort 3+ positions.
#    - `step = b - a` is no longer valid; step must be accumulated via GCD over
#      adjacent gaps.
#    - Companion detection still depends on modular alignment, but now the
#      modular class comes from the GCD-derived step, not the first gap.
#    - The write guard `cur == bg or cur == marker` remains safe, because the
#      generator still guarantees companions/bystanders use different colors and
#      appear at most once per line.
#
# d) Placement legality:
#    - Generator builds marker positions on one arithmetic beat lattice, then
#      selects a non-uniform subset whose gap-GCD is the base step.
#    - Marker/companion/bystander positions are chosen inside bounds, never
#      overlap, and companions are always beyond the outermost interior marker.
#    - Extra dots are kept away from the opposite boundary to reduce edge-count
#      ambiguity; the finished input is still validated by the solver.
#
# e) Visual observability:
#    - The new effect is observable because every active line has non-uniform
#      marker gaps: one gap is intentionally larger than the actual fill period,
#      while another remains a single beat. Sample pairs therefore show filled
#      beats appearing between some markers.
#
# f) Visual cleanliness:
#    - One active line contains one repeated marker color plus at most one extra
#      singleton color. This avoids dense overlap and keeps the periodic bands
#      easy to parse.
#
# g) Color inference:
#    - No hidden color ordering is used. Colors only indicate roles already
#      visible in the grid: marker, aligned companion, or unaligned bystander.
#
# Post-change solver assumptions reviewed against the generator:
#    - Exactly one repeated color per active line: guaranteed.
#    - Marker count >= 3 on every active line: guaranteed.
#    - At most one aligned non-marker color per line: guaranteed.
#    - At least one line changes under the GCD rule, because generated marker
#      gaps are non-uniform and input != output is rechecked.


def _gcd_of_gaps(positions: list[int]) -> int:
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    step = gaps[0]
    for gap in gaps[1:]:
        step = gcd(step, gap)
    return step


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic rays whose step is inferred from multiple marker gaps.

    Input structure:
    - bg is the most common color and the grid contains sparse non-bg dots.
    - One grid edge has strictly more non-bg dots than the others; that edge is
      the unique shooting side and defines whether rays run across rows or columns.
    - Each active shooting-direction line has exactly one repeated marker color
      with 3 or more dots on a common beat lattice.  The visible gaps between
      consecutive marker dots may differ.
    - A line may also contain one singleton dot of another color, either aligned
      to the same beat lattice (companion) or off-beat (bystander).

    Rule:
    - Find the unique shooting edge; left/right imply row-wise rays, top/bottom
      imply column-wise rays.
    - On each active line, sort the repeated marker positions and compute the
      step as the GCD of all adjacent marker gaps.  The beat offset is the first
      marker position modulo that step.
    - If another-colored dot shares that beat class, it is a companion: the ray
      is painted with the companion color and only between the marker/companion
      span.  An off-beat singleton is a bystander and stays untouched.
    - Without a companion, the marker color fills the whole line on that beat.

    Key insight: the edge with the densest boundary dots still identifies the
    shooting direction, but the period is no longer a single observed distance.
    The solver must combine several marker gaps and recover their common divisor
    before deciding which cells lie on the ray.
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
        for p in range(line_len):
            value = int(g[cell(li, p)])
            if value != bg:
                colors[value].append(p)

        if not colors:
            continue

        repeating = {color: sorted(posns) for color, posns in colors.items() if len(posns) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has repeating colors "
            f"{list(repeating)}; expected exactly one"
        )
        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) >= 3, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has {len(mpos)} dots (expected >= 3): {mpos}"
        )

        step = _gcd_of_gaps(mpos)
        assert step > 0, f"Non-positive step on line {li}: {mpos}"
        offset = mpos[0] % step
        assert all(p % step == offset for p in mpos), (
            f"Marker positions on line {li} do not share one beat class: {mpos}"
        )

        aligned_others = [
            other_color
            for other_color, other_positions in colors.items()
            if other_color != marker and any(p % step == offset for p in other_positions)
        ]
        assert len(aligned_others) <= 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has multiple companions: {aligned_others}"
        )
        companion = aligned_others[0] if aligned_others else None

        if companion is not None:
            paint = companion
            all_positions = mpos + colors[companion]
            lo, hi = min(all_positions), max(all_positions)
        else:
            paint = marker
            lo, hi = 0, line_len - 1

        for p in range(lo, hi + 1):
            if p % step != offset:
                continue
            rc = cell(li, p)
            current = int(out[rc])
            if current == bg or current == marker:
                out[rc] = paint

    return out


# --------------- generator (gcd-spacing variant) ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_MARKER_COUNT_RANGE = (3, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.35
_P_COMPANION = 0.35

_MAX_RETRIES = 400


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


def _build_marker_indices(max_beat: int, need_room_after: bool) -> list[int] | None:
    reserve = 2 if need_room_after else 1
    max_last = max_beat - reserve
    feasible_counts = [
        count
        for count in range(_MARKER_COUNT_RANGE[0], _MARKER_COUNT_RANGE[1] + 1)
        if max_last >= count
    ]
    if not feasible_counts:
        return None

    count = random.choice(feasible_counts)
    gap_count = count - 1

    # Force first beat-gap > 1 and preserve at least one later unit gap so the
    # GCD is smaller than the first visible marker gap.
    increments = [2] + [1] * (gap_count - 1)
    capacities = [1] + [2] * (gap_count - 1)
    protected = random.randrange(1, gap_count)
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


def _positions_from_indices(edge: str, line_len: int, step: int, beat_indices: list[int]) -> list[int]:
    if edge in ("left", "top"):
        return [idx * step for idx in beat_indices]
    edge_pos = line_len - 1
    return [edge_pos - idx * step for idx in beat_indices]


def _build_line_dots(edge: str, line_len: int, marker_color: int) -> list[tuple[int, int]] | None:
    kind = _choose_line_kind()
    max_step = min(_STEP_RANGE[1], line_len - 1)

    def feasible_steps(line_kind: str) -> list[int]:
        result = []
        for step in range(_STEP_RANGE[0], max_step + 1):
            max_beat = (line_len - 1) // step
            need_room_after = line_kind == "companion"
            if _build_marker_indices(max_beat, need_room_after) is None:
                continue
            if line_kind == "bystander" and step == 1:
                continue
            result.append(step)
        return result

    steps = feasible_steps(kind)
    if not steps:
        fallback = "marker_only" if kind != "marker_only" else None
        if fallback is None:
            return None
        kind = fallback
        steps = feasible_steps(kind)
        if not steps:
            return None

    step = random.choice(steps)
    max_beat = (line_len - 1) // step
    beat_indices = _build_marker_indices(max_beat, kind == "companion")
    assert beat_indices is not None
    marker_positions = _positions_from_indices(edge, line_len, step, beat_indices)

    dots = [(pos, marker_color) for pos in marker_positions]
    offset = marker_positions[0] % step

    if kind == "companion":
        later_beats = list(range(beat_indices[-1] + 1, max_beat))
        if later_beats:
            other_colors = [color for color in _COLORS if color != marker_color]
            beat_idx = random.choice(later_beats)
            companion_pos = _positions_from_indices(edge, line_len, step, [beat_idx])[0]
            dots.append((companion_pos, random.choice(other_colors)))
        else:
            kind = "marker_only"

    if kind == "bystander":
        other_colors = [color for color in _COLORS if color != marker_color]
        non_beat = [
            pos
            for pos in range(line_len)
            if pos % step != offset and pos not in marker_positions
        ]
        if other_colors and non_beat:
            dots.append((random.choice(non_beat), random.choice(other_colors)))

    return dots


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


def _line_signature(values: list[int]) -> tuple[int, bool]:
    colors: dict[int, list[int]] = defaultdict(list)
    for pos, value in enumerate(values):
        if value != 0:
            colors[value].append(pos)
    marker_positions = max(colors.values(), key=len)
    gaps = [b - a for a, b in zip(marker_positions, marker_positions[1:])]
    return _gcd_of_gaps(marker_positions), len(set(gaps)) > 1


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

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    gcd_critical_lines = 0

    for li in active_lines:
        marker_color = random.choice(_COLORS)
        dots = _build_line_dots(edge, line_len, marker_color)
        if dots is None:
            return None

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

        values = [int(inp[cell(li, p)]) for p in range(line_len)]
        _, is_critical = _line_signature(values)
        gcd_critical_lines += int(is_critical)
        shooting_edge_dots += 1

    if gcd_critical_lines == 0:
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
