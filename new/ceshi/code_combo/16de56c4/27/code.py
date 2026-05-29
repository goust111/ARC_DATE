# Combo variant of 16de56c4 (index 27)
# combo_spec: D1/0+D16/3
# Name: GCD多点间距+指定长度填充

from __future__ import annotations

from collections import defaultdict
from math import gcd
import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# ---------------
# a) Concrete behavior:
#    - Original 16de56c4 used exactly two marker dots, so the visible gap was
#      both the period and the only distance cue. Marker-only lines then filled
#      the whole line, while companion lines filled the marker/companion span.
#    - In this combo, one active line has 3+ same-colored markers. Their gaps
#      are deliberately non-uniform, so the visible first gap is no longer the
#      true step. Example on a left-shooting row:
#        input markers at beat indices [0, 3, 4] with base step 2
#        visible positions = [0, 6, 8]
#        old reading from first two markers alone would suggest step 6
#        combo reading uses gcd(6, 2) = 2
#      After recovering step 2, the first interior marker at position 6 is
#      three beats away from the edge, so output paints exactly 3 beats:
#      positions 6, 8, 10. If a companion sits at 10, those three beats become
#      the companion color; otherwise they keep the marker color.
#    - This differs from both source behaviors:
#      1. D1/0 alone would still fill the full line or full companion span once
#         the GCD step is found.
#      2. D16/3 alone would still read the first gap itself as the step, which
#         would be wrong on [0, 6, 8].
#
# b) Complexity calibration:
#    - The generator only uses first interior beat distances 3 or 4. Distance 2
#      is too weak visually because the fixed-length run can collapse to already
#      occupied markers; larger counts become hard to fit in short columns.
#    - Every active line includes a late unit beat-gap after a larger first
#      beat-gap. This makes the GCD inference visible without introducing many
#      extra dots.
#    - At most one singleton non-marker dot appears on a line, preserving the
#      original "marker / companion / bystander" role system.
#
# c) Hidden assumptions that must change:
#    - The original solver assumed `len(mpos) == 2`; here the repeated color has
#      3-4 dots, so markers must be ordered by distance from the shooting edge.
#    - The original `step = b - a` is invalid; step must be the GCD of adjacent
#      marker gaps.
#    - The original D16 count rule used the only interior marker. In the combo,
#      the designated count marker is the first interior marker after the edge,
#      so the solver must identify marker order after it knows the true step.
#    - Companion detection can no longer define the fill extent. It only changes
#      paint color, and only if it lies on the same beat class beyond that first
#      interior marker.
#
# d) Placement legality:
#    - Marker beat indices are constructed as `[0, count, count+1, ...]`, so
#      one gap is large and a later gap is 1 beat. This guarantees GCD recovery.
#    - The fixed-length run always ends strictly before the opposite boundary, so
#      output cells do not create a rival boundary count.
#    - Companion dots, when present, are placed on the final target beat; this
#      keeps them aligned, beyond the count marker, in-bounds, and non-overlapping.
#    - Bystanders are chosen off-beat and outside the target set.
#
# e) Visual observability:
#    - A human can see that the first marker gap is too large to be the true
#      rhythm because later markers introduce a finer cadence in the same row or
#      column.
#    - The painted run begins at the first interior marker, not at the edge, and
#      has a short countable length. That visibly separates this combo from the
#      original full-line/span behavior.
#
# f) Visual cleanliness:
#    - Active lines remain spaced apart and sparse. Each line shows one repeated
#      color plus at most one singleton of another color.
#    - Later markers stay near the fixed-length run, so the causal relation
#      between marker rhythm and painted targets remains easy to read.
#
# g) Color inference:
#    - Colors do not encode numbers. Marker color paints unless an aligned
#      companion is present; then the companion color paints the same fixed run.
#      This is the same visible color logic as the base task.
#
# Post-change solver assumptions checked against the generator:
#    - Exactly one repeated color per active line: guaranteed.
#    - Marker count >= 3 and ordered edge-first in beat distance: guaranteed.
#    - The first interior marker is 3 or 4 beats from the edge: guaranteed.
#    - At most one aligned non-marker color exists beyond the count marker:
#      guaranteed by one-dot companion construction.
#    - Every sample exercises both modifications, because all lines need GCD to
#      recover the base step and use the first large gap to determine fill count.


def _gcd_of_gaps(positions: list[int]) -> int:
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    assert gaps, f"Need at least two positions to compute gaps: {positions}"
    step = gaps[0]
    for gap in gaps[1:]:
        step = gcd(step, gap)
    return step


def _positions_by_edge_distance(edge_pos: int, step: int, positions: list[int]) -> list[tuple[int, int]]:
    ranked = sorted((abs(pos - edge_pos) // step, pos) for pos in positions)
    assert ranked[0][0] == 0, f"Markers do not touch the shooting edge: {positions}"
    return ranked


def _target_positions(start_pos: int, fill_dir: int, step: int, count: int) -> list[int]:
    return [start_pos + fill_dir * step * k for k in range(count)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a hidden beat from multiple markers, then paint a count-sized run from the first interior marker.

    Input structure:
    - bg is the most common color and the grid is mostly empty except for sparse dots.
    - Exactly one boundary has strictly more non-bg dots than the other three; that
      boundary is the shooting edge and determines whether rays travel across rows
      or across columns.
    - Each active shooting-direction line contains one repeated marker color with
      3 or more dots. One marker touches the shooting edge, and the later markers
      share a finer beat lattice whose true step is the GCD of adjacent gaps.
    - A line may also contain one singleton of another color: aligned to that beat
      class as a companion, or off-beat as a bystander.

    Rule:
    - Find the unique shooting edge; left/right imply row-wise rays, top/bottom
      imply column-wise rays.
    - On each active line, sort the repeated marker positions and compute the base
      step as the GCD of adjacent marker gaps. Then re-order those markers by
      distance from the shooting edge.
    - The first interior marker after the edge determines the fill length: if it is
      `k` beats from the edge, paint exactly `k` beat positions starting at that
      marker and continuing away from the edge using the recovered step.
    - An aligned companion beyond that count marker only changes the paint color to
      its own color; it does not change the run length. An off-beat singleton is a
      bystander and remains untouched.

    Key insight: the first visible marker gap is intentionally larger than the true
    period. The solver must first combine several marker gaps into their GCD, then
    reinterpret that first large gap as a beat-count for a short outward progression.
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
    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    fill_dir = 1 if edge in ("left", "top") else -1

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
        assert all(pos % step == offset for pos in mpos), (
            f"Marker positions on line {li} do not share one beat class: {mpos}"
        )

        ranked = _positions_by_edge_distance(edge_pos, step, mpos)
        beat_distances = [dist for dist, _pos in ranked]
        assert beat_distances[0] == 0, f"Missing edge marker on line {li}: {ranked}"
        assert len(beat_distances) >= 2 and beat_distances[1] > 0, (
            f"Need an interior marker on line {li}: {ranked}"
        )

        count = beat_distances[1]
        start_pos = ranked[1][1]
        targets = _target_positions(start_pos, fill_dir, step, count)
        assert all(0 <= pos < line_len for pos in targets), (
            f"Line {li} ({'row' if by_row else 'col'}) cannot host {count} beats: {targets}"
        )

        aligned_others = [
            other_color
            for other_color, other_positions in colors.items()
            if other_color != marker
            and any(
                pos % step == offset and (pos - start_pos) * fill_dir > 0
                for pos in other_positions
            )
        ]
        assert len(aligned_others) <= 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has multiple companions: {aligned_others}"
        )
        paint = aligned_others[0] if aligned_others else marker

        for pos in targets:
            rc = cell(li, pos)
            current = int(out[rc])
            if current == bg or current == marker:
                out[rc] = paint

    return out


# --------------- generator ---------------

_ROWS_RANGE = (12, 18)
_COLS_RANGE = (12, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (2, 3)
_COUNT_OPTIONS = (3, 4)
_MARKER_COUNT_RANGE = (3, 4)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_MIN_LINE_LEN = 17

_P_MARKER_ONLY = 0.30
_P_COMPANION = 0.45

_MAX_RETRIES = 500


def _valid_step_count_pairs(line_len: int) -> list[tuple[int, int]]:
    pairs = []
    for step in range(_STEP_RANGE[0], _STEP_RANGE[1] + 1):
        for count in _COUNT_OPTIONS:
            # Keep the final painted beat off the opposite boundary.
            if (2 * count - 1) * step <= line_len - 2:
                pairs.append((step, count))
    return pairs


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
    min_num_lines = 2 * n_active - 1
    by_row = edge in ("left", "right")
    if by_row:
        rows_lo = max(_ROWS_RANGE[0], min_num_lines)
        cols_lo = max(_COLS_RANGE[0], _MIN_LINE_LEN)
        if rows_lo > _ROWS_RANGE[1] or cols_lo > _COLS_RANGE[1]:
            return None
        H = random.randint(rows_lo, _ROWS_RANGE[1])
        W = random.randint(cols_lo, _COLS_RANGE[1])
        if not _valid_step_count_pairs(W):
            return None
        return H, W, H, W

    cols_lo = max(_COLS_RANGE[0], min_num_lines)
    rows_lo = max(_ROWS_RANGE[0], _MIN_LINE_LEN)
    if cols_lo > _COLS_RANGE[1] or rows_lo > _ROWS_RANGE[1]:
        return None
    W = random.randint(cols_lo, _COLS_RANGE[1])
    H = random.randint(rows_lo, _ROWS_RANGE[1])
    if not _valid_step_count_pairs(H):
        return None
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


def _marker_indices_for_count(count: int) -> list[int]:
    indices = [0, count, count + 1]
    candidates = list(range(count + 2, 2 * count - 1))
    max_markers = min(_MARKER_COUNT_RANGE[1], len(indices) + len(candidates))
    marker_total = random.randint(_MARKER_COUNT_RANGE[0], max_markers)
    extra_need = marker_total - len(indices)
    if extra_need > 0:
        indices.extend(random.sample(candidates, extra_need))
    return sorted(indices)


def _positions_from_indices(edge: str, line_len: int, step: int, beat_indices: list[int]) -> list[int]:
    if edge in ("left", "top"):
        return [idx * step for idx in beat_indices]
    edge_pos = line_len - 1
    return [edge_pos - idx * step for idx in beat_indices]


def _fill_positions(edge: str, line_len: int, step: int, count: int) -> list[int]:
    marker_positions = _positions_from_indices(edge, line_len, step, [count])
    fill_dir = 1 if edge in ("left", "top") else -1
    return _target_positions(marker_positions[0], fill_dir, step, count)


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    palette: list[int],
) -> tuple[list[tuple[int, int]], str, tuple[int, int]]:
    kind = _choose_line_kind()
    step, count = random.choice(_valid_step_count_pairs(line_len))

    beat_indices = _marker_indices_for_count(count)
    marker_positions = _positions_from_indices(edge, line_len, step, beat_indices)
    targets = _fill_positions(edge, line_len, step, count)
    target_set = set(targets)
    dots = [(pos, marker_color) for pos in marker_positions]

    other_colors = [color for color in palette if color != marker_color]
    if kind == "companion" and other_colors:
        companion_pos = targets[-1]
        assert companion_pos not in marker_positions
        dots.append((companion_pos, random.choice(other_colors)))
    elif kind == "bystander" and other_colors:
        edge_pos = 0 if edge in ("left", "top") else line_len - 1
        non_beat = [
            pos
            for pos in range(line_len)
            if pos not in marker_positions
            and pos not in target_set
            and pos % step != edge_pos % step
        ]
        if non_beat:
            dots.append((random.choice(non_beat), random.choice(other_colors)))
        else:
            kind = "marker_only"
    else:
        kind = "marker_only"

    return dots, kind, (step, count)


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


def _line_metrics(values: list[int], edge: str) -> tuple[bool, bool]:
    colors: dict[int, list[int]] = defaultdict(list)
    for pos, value in enumerate(values):
        if value != 0:
            colors[value].append(pos)
    repeating = [sorted(posns) for posns in colors.values() if len(posns) >= 2]
    if len(repeating) != 1:
        return False, False

    marker_positions = repeating[0]
    step = _gcd_of_gaps(marker_positions)
    edge_pos = 0 if edge in ("left", "top") else len(values) - 1
    ranked = _positions_by_edge_distance(edge_pos, step, marker_positions)
    beat_distances = [dist for dist, _pos in ranked]
    count = beat_distances[1]
    gcd_visible = len(marker_positions) >= 3 and any(
        later - earlier > 1
        for earlier, later in zip(beat_distances, beat_distances[1:])
    )
    count_visible = count >= 3
    return gcd_visible, count_visible


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    palette_size = random.randint(2, min(8, len(_COLORS)))
    palette = random.sample(_COLORS, palette_size)

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    line_types: list[str] = []
    line_params: list[tuple[int, int]] = []
    gcd_visible_lines = 0
    count_visible_lines = 0

    for li in active_lines:
        marker_color = random.choice(palette)
        dots, line_type, params = _build_line_dots(edge, line_len, marker_color, palette)
        line_types.append(line_type)
        line_params.append(params)

        used_positions = set()
        for pos, color in dots:
            if pos in used_positions:
                return None
            used_positions.add(pos)
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        values = [int(inp[cell(li, p)]) for p in range(line_len)]
        gcd_visible, count_visible = _line_metrics(values, edge)
        gcd_visible_lines += int(gcd_visible)
        count_visible_lines += int(count_visible)
        shooting_edge_dots += 1

    if gcd_visible_lines != n_active or count_visible_lines != n_active:
        return None
    type_counts = defaultdict(int)
    for line_type in line_types:
        type_counts[line_type] += 1
    if any(type_counts[line_type] == 0 for line_type in ("marker_only", "companion", "bystander")):
        return None
    if not any(step == 3 for step, _count in line_params):
        return None
    if not any(count == 4 for _step, count in line_params):
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
