# Variant of 16de56c4
# Dimension: D16 — 填充范围计算 (dst)
# Change: 无伴随时填整行(0到line_len-1)；有伴随时填min到max的span... → 半行填充

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis:
# spec:
#   preserved_core:
#     - background is the most common color
#     - exactly one outer border acts as the shooting edge
#     - left/right shoot along rows; top/bottom shoot along columns
#     - an aligned foreign-color dot is still a companion that recolors the beat
#       and clips painting to the min..max span of marker and companion dots
#     - unaligned foreign dots are bystanders and stay unchanged
#   d16_change:
#     - old no-companion behavior painted the entire row/column on the inferred beat
#     - new no-companion behavior paints only from the shooting edge to the last
#       visible same-color marker on that beat; the ray no longer extends forever
#   observability_adjustment:
#     - with the original "exactly two marker dots" construction, the new bounded
#       default would often collapse to the two existing endpoints and become
#       visually indistinguishable from "do nothing"
#     - to make D16 visible, some no-companion lines now contain 3 same-color
#       marker dots on the same beat: the edge marker, the first inward marker
#       that defines the period, and a farther marker on the same modular class
#     - the input omits one or more intermediate beat cells between the first and
#       far marker, so the output visibly fills a finite segment but still stops
#       before the opposite side of the grid
#   concrete_trace:
#     - example line from the left edge, length 13:
#       markers at positions [0, 2, 8], so the period is 2 from the first gap
#     - old solver: no companion => fill every even position 0..12
#       output positions become [0, 2, 4, 6, 8, 10, 12]
#     - new solver: no companion => fill only from the edge to the furthest
#       marker, so output positions become [0, 2, 4, 6, 8]
#     - difference: the same input now produces a finite half-line instead of an
#       edge-to-edge infinite progression
#   complexity_calibration:
#     - only two meaningful branch types are needed for inference:
#       companion lines and no-companion half-lines
#     - a small fraction of half-lines additionally get one unaligned bystander,
#       retaining the original "spectator" concept without introducing new cases
#     - marker colors may repeat across lines, but each active line still has only
#       one repeating marker color, preventing ambiguous line parsing
#   implicit_assumption_audit:
#     - edge detection still relies on counting non-background dots on each border;
#       generator must keep the shooting border strictly dominant
#     - the solver can no longer assert "exactly two" marker dots; instead it must
#       use the edge-nearest pair to infer the period and allow extra same-color
#       markers further inward on the same beat
#     - the first marker on every active line must still lie on the shooting edge;
#       generator guarantees this
#     - every extra same-color marker on a no-companion line must lie on the same
#       modular class as the first pair; generator guarantees this too
#     - companion selection still assumes at most one aligned foreign color per
#       line; generation enforces that by placing at most one companion dot
#   placement_legality:
#     - active lines remain gap-separated
#     - all dots on a line have distinct positions and stay within bounds
#     - far markers for half-lines are kept away from the opposite shooting edge
#       when possible so the default finite segment is easy to see
#     - bystanders are chosen from non-beat cells, so they cannot be mistaken for
#       companions
#   visual_readability:
#     - companion lines still show long recolored spans
#     - half-lines show shorter same-color fills that stop at their last marker
#     - bystanders remain isolated off-beat dots, so the grid stays legible
#   generator_checks:
#     - every generated puzzle must contain at least one half-line and at least one
#       companion line, ensuring the D16 branch is both exercised and contrasted
#     - solver validation plus a later old-vs-new comparison confirms that the new
#       inputs do not degenerate back to the original full-line behavior


def _progress_from_edge(edge: str, line_len: int, pos: int) -> int:
    """Distance from the shooting edge along the active line."""
    return pos if edge in ("left", "top") else (line_len - 1 - pos)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint beat-aligned rays launched from one dominant edge.

    Input structure:
    - The background is the most frequent color; all other cells are sparse dots.
    - Exactly one border (top, bottom, left, or right) contains more non-bg dots
      than the other three and therefore defines the shooting side.
    - Each active shooting-direction line has one repeating marker color.  One
      marker sits on the shooting edge, the next inward marker fixes the period,
      and some no-companion lines may contain extra same-color markers farther in.
    - A different color on the same beat is a companion; a different color off
      the beat is a bystander.

    Rule:
    - Find the unique shooting edge; left/right act on rows, top/bottom on columns.
    - On each active line, sort the repeating marker positions by distance from the
      shooting edge.  The gap between the first two markers is the step, and their
      beat determines the modular class to paint.
    - If another color shares that beat, repaint only the beat cells inside the
      span covered by the marker and companion dots, using the companion color.
    - If no companion exists, repaint only from the shooting edge through the
      furthest visible marker of that repeating color; do not extend past it.
    - Bystanders are never modified.

    Key insight:
    - Border counts reveal the launch side, while the edge-nearest marker gap
      reveals the beat.  An aligned foreign dot still acts as a wall and color
      override; otherwise the markers' own visible extent now caps the ray.
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

        repeating = {color: poses for color, poses in colors.items() if len(poses) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has repeating colors "
            f"{sorted(repeating)} from dots {dict(colors)}"
        )

        marker, marker_positions = next(iter(repeating.items()))
        ordered = sorted(
            marker_positions,
            key=lambda p: _progress_from_edge(edge, line_len, p),
        )
        assert ordered[0] == edge_pos, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"does not touch the shooting edge: {ordered}"
        )
        assert len(ordered) >= 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"needs at least 2 dots: {ordered}"
        )

        step = _progress_from_edge(edge, line_len, ordered[1])
        assert step > 0, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has zero step: {ordered}"
        )
        offset = edge_pos % step

        for p in ordered:
            progress = _progress_from_edge(edge, line_len, p)
            assert progress % step == 0, (
                f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
                f"breaks the inferred beat {step}: {ordered}"
            )
            assert p % step == offset, (
                f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
                f"breaks the inferred offset {offset}: {ordered}"
            )

        companion_color = None
        companion_positions: list[int] = []
        for other_color, other_positions in colors.items():
            if other_color == marker:
                continue
            aligned = [p for p in other_positions if p % step == offset]
            if aligned:
                companion_color = other_color
                companion_positions = aligned
                break

        if companion_color is not None:
            paint = companion_color
            all_positions = ordered + companion_positions
            lo, hi = min(all_positions), max(all_positions)
        else:
            paint = marker
            lo, hi = sorted((edge_pos, ordered[-1]))

        for p in range(lo, hi + 1):
            if p % step != offset:
                continue
            rc = cell(li, p)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_EXTRA_BYSTANDER = 0.30
_P_EXTRA_COMPANION = 0.45

_MAX_RETRIES = 400


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
    """Pick (H, W, num_lines, line_len) compatible with the edge and line count."""
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
    """Choose n indices with at least one empty gap between consecutive picks."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _candidate_steps_for_half(line_len: int) -> list[int]:
    upper = min(_STEP_RANGE[1], line_len - 1)
    return [step for step in range(_STEP_RANGE[0], upper + 1) if (line_len - 1) // step >= 4]


def _candidate_steps_for_companion(line_len: int) -> list[int]:
    upper = min(_STEP_RANGE[1], line_len - 1)
    return [step for step in range(_STEP_RANGE[0], upper + 1) if (line_len - 1) // step >= 2]


def _beats_from_edge(edge: str, line_len: int, step: int, start_beat: int) -> list[int]:
    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    max_beat = (line_len - 1) // step
    beats = []
    for beat in range(start_beat, max_beat + 1):
        pos = edge_pos + beat * step if edge_pos == 0 else edge_pos - beat * step
        if 0 <= pos < line_len:
            beats.append(pos)
    return beats


def _build_half_line(edge: str, line_len: int, marker_color: int) -> list[tuple[int, int]]:
    step = random.choice(_candidate_steps_for_half(line_len))
    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    max_beat = (line_len - 1) // step
    far_beat = random.randint(3, max_beat - 1)
    far_pos = edge_pos + far_beat * step if edge_pos == 0 else edge_pos - far_beat * step
    first_pos = edge_pos + step if edge_pos == 0 else edge_pos - step
    dots = [(edge_pos, marker_color), (first_pos, marker_color), (far_pos, marker_color)]

    if random.random() < _P_EXTRA_BYSTANDER:
        offset = edge_pos % step
        other_colors = [c for c in _COLORS if c != marker_color]
        non_beat = [
            p
            for p in range(line_len)
            if p % step != offset and p not in {edge_pos, first_pos, far_pos, 0, line_len - 1}
        ]
        if other_colors and non_beat:
            dots.append((random.choice(non_beat), random.choice(other_colors)))

    return dots


def _build_companion_line(edge: str, line_len: int, marker_color: int) -> list[tuple[int, int]]:
    step = random.choice(_candidate_steps_for_companion(line_len))
    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    first_pos = edge_pos + step if edge_pos == 0 else edge_pos - step
    dots = [(edge_pos, marker_color), (first_pos, marker_color)]

    other_colors = [c for c in _COLORS if c != marker_color]
    beat_positions = _beats_from_edge(edge, line_len, step, start_beat=2)
    interior_beats = [p for p in beat_positions if p not in (0, line_len - 1)]
    companion_pos = random.choice(interior_beats if interior_beats else beat_positions)
    dots.append((companion_pos, random.choice(other_colors)))

    if random.random() < _P_EXTRA_COMPANION:
        offset = edge_pos % step
        non_beat = [
            p
            for p in range(line_len)
            if p % step != offset and p not in {edge_pos, first_pos, companion_pos, 0, line_len - 1}
        ]
        if non_beat:
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


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    if not _candidate_steps_for_half(line_len) or not _candidate_steps_for_companion(line_len):
        return None

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    line_kinds = ["half", "companion"]
    while len(line_kinds) < n_active:
        line_kinds.append("companion" if random.random() < 0.45 else "half")
    random.shuffle(line_kinds)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    seen_half = 0
    seen_companion = 0

    for li, kind in zip(active_lines, line_kinds):
        marker_color = random.choice(_COLORS)
        if kind == "half":
            dots = _build_half_line(edge, line_len, marker_color)
            seen_half += 1
        else:
            dots = _build_companion_line(edge, line_len, marker_color)
            seen_companion += 1

        occupied = set()
        for pos, color in dots:
            if pos in occupied:
                return None
            occupied.add(pos)
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1

    if seen_half == 0 or seen_companion == 0:
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
