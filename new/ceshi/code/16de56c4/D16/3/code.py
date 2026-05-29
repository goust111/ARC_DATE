# Variant of 16de56c4
# Dimension: D16 — 填充范围计算 (dst)
# Change: 无伴随时填整行(0到line_len-1)；有伴随时填min到max的span... → 指定长度填充

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D16 variant
# -----------------------------------
# a) Concrete behavior:
#    The original task used the two marker dots to define a periodic beat, then
#    decided how far to paint by either taking the whole line (no companion) or
#    the marker/companion span (with companion).  In this variant the marker gap
#    still defines the beat, but the gap also becomes a count: if the inner
#    marker is 3 cells away from the shooting edge, we paint exactly 3 beat
#    positions starting at that inner marker and moving away from the edge.
#    Example on a left-shooting row with markers at positions 0 and 3:
#    - original marker-only output: 0,3,6,9,12,... across the row
#    - new marker-only output:      0 stays, and only 3,6,9 are painted
#    Example with a companion at 9:
#    - original companion output:   0,3,6,9 all become companion color
#    - new companion output:        0 stays marker-colored; 3,6,9 become companion color
#    So D16 changes only the destination/range computation, not edge detection,
#    beat detection, companion color override, or bystander preservation.
#
# b) Complexity calibration:
#    If step=1, "paint exactly one beat from the inner marker outward" can
#    collapse to no visible change on marker-only lines, so generation uses
#    steps 2-4 only.  Step 5 is intentionally dropped because the grid's maximum
#    line length (21) cannot safely host 5 beat positions spaced 5 apart.
#    Companion lines remain present so the old color-override mechanism is still
#    repeated across examples, but the fill extent is now controlled by count.
#
# c) Hidden assumptions from the original solver that must change:
#    - `lo, hi = 0, line_len - 1` for marker-only lines no longer holds.
#    - `lo, hi = min(...), max(...)` for companion lines no longer holds.
#    - Iterating over every position in `[lo, hi]` and filtering by modulus is
#      replaced by constructing an explicit list of exactly `step` beat targets.
#    Assumptions that still hold:
#    - exactly one shooting edge has the most boundary dots
#    - each active line has exactly one repeating color with exactly two dots
#    - one of those marker dots lies on the shooting edge
#    - any aligned non-marker dot is beyond the inner marker and acts as companion
#
# d) Placement legality:
#    To make "paint `step` beat positions" physically possible, the generator
#    only uses steps satisfying `step * step <= line_len - 2`.  That guarantees
#    every target beat is in-bounds and also leaves at least one background cell
#    beyond the final filled beat, preventing the last beat from landing on the
#    opposite edge and confusing edge-count inference.
#
# e) Visual observability:
#    The count rule is readable from 2-4 examples because each row/column shows:
#    - a marker on the shooting edge
#    - an inner marker exactly `step` cells away
#    - a short arithmetic progression of exactly `step` painted beats extending
#      farther inward
#    This is visibly different from the original full-line or span fill.  On
#    companion lines the unchanged edge marker next to a recolored inner run is
#    especially diagnostic.
#
# f) Visual cleanliness:
#    Active lines are still spaced apart, each line still carries at most three
#    dots in the input, and bystanders remain off-beat.  The output therefore
#    shows clean, non-overlapping horizontal or vertical progressions.
#
# g) Color inference:
#    No color-id arithmetic is used.  Marker color stays the paint color unless
#    an aligned companion is present; then the companion's visible color becomes
#    the fill color.  This keeps the color rule identical to the original task.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a fixed number of periodic beats that shoot away from one dominant edge.

    Input structure:
    - bg = most common color. Sparse non-bg dots lie on several rows or columns.
    - One grid edge (top, bottom, left, or right) has strictly more non-bg dots
      than the other three; that edge is the unique shooting side.
    - Every active shooting-direction line contains exactly one repeating color
      with exactly two dots: one on the shooting edge and one interior.
    - Optional third dots are either aligned companions on the same beat class or
      unaligned bystanders that must stay unchanged.

    Rule:
    - Determine the shooting edge; left/right shoot along rows, top/bottom along columns.
    - On each active line, the marker gap defines both the beat spacing and the
      fill count: if the interior marker is `step` cells away from the edge, then
      exactly `step` beat positions are painted, starting at the interior marker
      and continuing away from the edge.
    - An aligned companion still changes the paint color to its own color, but it
      no longer sets the fill span; it only recolors that fixed-length progression.
    - Unaligned bystanders remain untouched.

    Key insight: the distance from the inner marker back to the shooting edge is
    itself the number of beats to emit, so the marker pair encodes both period and
    run length while companions only control color.
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

        repeating = {color: ps for color, ps in colors.items() if len(ps) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) must have exactly one repeating color: "
            f"{dict(colors)}"
        )
        marker, marker_positions = next(iter(repeating.items()))
        assert len(marker_positions) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has {len(marker_positions)} dots (expected 2): {marker_positions}"
        )
        assert edge_pos in marker_positions, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"must touch shooting edge {edge}: {marker_positions}"
        )

        inner_pos = next(pos for pos in marker_positions if pos != edge_pos)
        step = abs(inner_pos - edge_pos)
        assert step > 0, "Marker dots must be distinct"

        companion_color = None
        for other_color, other_positions in colors.items():
            if other_color == marker:
                continue
            aligned = [
                pos
                for pos in other_positions
                if (pos - edge_pos) % step == 0 and (pos - inner_pos) * fill_dir > 0
            ]
            if aligned:
                companion_color = other_color
                break

        paint = companion_color if companion_color is not None else marker
        targets = [inner_pos + fill_dir * step * k for k in range(step)]
        assert all(0 <= pos < line_len for pos in targets), (
            f"Line {li} ({'row' if by_row else 'col'}) cannot host {step} beats: {targets}"
        )

        for pos in targets:
            rc = cell(li, pos)
            current = int(out[rc])
            if current == bg or current == marker:
                out[rc] = paint

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (2, 4)  # 1 is often visually degenerate; 5 cannot fit this variant
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.35
_P_COMPANION = 0.45

_MAX_RETRIES = 400


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
    """Pick (H, W, num_lines, line_len) compatible with edge and active count."""
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
    """Choose n indices from [0, total) with gaps >= 1 between neighbors."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _marker_positions(edge: str, line_len: int, step: int) -> tuple[int, int]:
    if edge in ("left", "top"):
        return 0, step
    return line_len - 1, line_len - 1 - step


def _fill_positions(edge: str, line_len: int, step: int) -> list[int]:
    edge_pos, inner_pos = _marker_positions(edge, line_len, step)
    fill_dir = 1 if edge in ("left", "top") else -1
    positions = [inner_pos + fill_dir * step * k for k in range(step)]
    assert edge_pos not in positions
    return positions


def _valid_steps(line_len: int) -> list[int]:
    return [step for step in range(_STEP_RANGE[0], _STEP_RANGE[1] + 1) if step * step <= line_len - 2]


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    palette: list[int],
    step: int,
) -> tuple[list[tuple[int, int]], str]:
    """Return the input dots for one active line and the chosen line type."""
    edge_pos, inner_pos = _marker_positions(edge, line_len, step)
    beat_positions = set(_fill_positions(edge, line_len, step))
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]

    roll = random.random()
    if roll < _P_MARKER_ONLY:
        return dots, "marker_only"

    other_colors = [color for color in palette if color != marker_color]
    if roll < _P_MARKER_ONLY + _P_COMPANION and other_colors:
        companion_pos = max(beat_positions) if edge in ("left", "top") else min(beat_positions)
        companion_color = random.choice(other_colors)
        dots.append((companion_pos, companion_color))
        return dots, "companion"

    non_beat = [
        pos
        for pos in range(line_len)
        if pos not in (edge_pos, inner_pos) and pos not in beat_positions and (pos - edge_pos) % step != 0
    ]
    if other_colors and non_beat:
        dots.append((random.choice(non_beat), random.choice(other_colors)))
        return dots, "bystander"
    return dots, "marker_only"


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]],
    edge: str,
    H: int,
    W: int,
) -> dict[str, int]:
    """Count how many placed dots land on each non-shooting edge."""
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

    valid_steps = _valid_steps(line_len)
    if not valid_steps:
        return None

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

    for li in active_lines:
        step = random.choice(valid_steps)
        marker_color = random.choice(palette)
        dots, line_type = _build_line_dots(edge, line_len, marker_color, palette, step)
        line_types.append(line_type)

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

        shooting_edge_dots += 1

    if all(line_type == "marker_only" for line_type in line_types):
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
