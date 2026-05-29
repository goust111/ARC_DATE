# Variant of 16de56c4
# Dimension: D7 — 变换规则 (transform)
# Change: 背景和标记色单元格被填充为轨迹颜色，旁观者点保持不变... → 标记点擦除

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D7 variant
# ----------------------------------
# spec:
#   unchanged_geometry:
#     - The dominant shooting edge is still the boundary with the most non-bg dots.
#     - left/right edges shoot along rows; top/bottom edges shoot along columns.
#     - Each active line still contains exactly two same-colored marker dots; their gap is the step.
#     - An aligned different-colored companion still clips the ray and supplies the paint color.
#     - An unaligned different-colored bystander still survives unchanged.
#   transform_change:
#     - The two repeated marker dots are now instructions only.
#     - Output keeps the periodic ray they define, but the marker cells themselves are erased to bg.
#     - Companion cells remain if present; bystanders remain; only the marker cells disappear.
#   concrete_trace:
#     - Example row from the left edge: input positions [(0,4), (3,4), (9,7)].
#     - Original task: step=3, companion=7, fill beats 0/3/6/9 with 7, so markers become 7.
#     - New variant: fill only non-marker beats on that span, so position 6 becomes 7,
#       companion at 9 stays 7, then positions 0 and 3 are erased to bg.
#     - Visible effect: the input's two 4 dots vanish while a same-beat trail remains/grows.
#   complexity_calibration:
#     - Only D7 changes; the edge-detection and modular-beat logic stay identical.
#     - To keep the erase effect visible, the generator only uses steps/companions that leave
#       at least one non-marker beat cell in the output trajectory.
#     - Companion and bystander branches are still both available, so the puzzle does not collapse
#       into a single trivial "copy-every-kth cell" pattern.
#   implicit_assumptions_checked:
#     - Original `out = g.copy()` is still safe, but marker cells must later be forced to bg.
#     - Original fill condition `cur == bg or cur == marker` relied on markers being repaintable;
#       this is no longer valid because markers must not survive as paint.
#     - Processing by line still assumes each non-empty line has a repeated color; generator
#       preserves that by placing exactly one repeated marker color per active line.
#     - Companion detection still assumes aligned color membership is enough; generator keeps at
#       most one aligned foreign dot so the first match is well-defined.
#   placement_legality_and_cleanliness:
#     - Active lines are still spaced apart, so row/column patterns do not overlap.
#     - Each line places at most one optional third dot, never on a marker cell, never off-grid.
#     - Companion beats are chosen strictly beyond the inner marker and far enough away to leave a
#       visible non-marker track cell after marker erasure.
#   human_observability:
#     - Across samples, the repeated dots consistently disappear while same-beat cells appear
#       elsewhere on the line; this is readable from 2-3 examples without relying on color ids.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace periodic rays from a unique shooting edge, but erase the marker dots that define them.

    Input structure:
    - bg = most common color. Sparse non-bg dots of several colors.
    - One grid edge (top, bottom, left, or right) has the most non-bg dots and sets the
      shooting direction; left/right act on rows, top/bottom act on columns.
    - On each active shooting line, exactly one color repeats twice; those two dots are the
      markers and their gap is the period.
    - A different-colored aligned dot on the same beat is a companion; an unaligned foreign dot
      is a bystander.

    Rule:
    - Identify the unique shooting edge, then scan each row/column parallel to that direction.
    - For every repeated marker pair, compute step from the distance between the two markers and
      compute the beat from the first marker modulo that step.
    - Without a companion, paint every beat cell across the whole line in the marker color.
      With a companion, paint only the beat cells between the markers and the companion, using
      the companion color.
    - After decoding a line, erase the two marker cells back to background; bystanders are left
      untouched.

    Key insight: the repeated dots are no longer output content. They only encode a modular beat
    and, once that beat has generated its ray, the two encoding dots disappear.
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

        repeating = {color: positions for color, positions in colors.items() if len(positions) >= 2}
        assert repeating, (
            f"Line {li} ({'row' if by_row else 'col'}) has non-bg dots "
            f"{dict(colors)} but no repeating color"
        )

        for marker, mpos in repeating.items():
            assert len(mpos) == 2, (
                f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
                f"has {len(mpos)} dots (expected 2): {mpos}"
            )
            a, b = sorted(mpos)
            step = b - a
            offset = a % step

            companion = None
            for other_color, other_positions in colors.items():
                if other_color == marker:
                    continue
                if any(pos % step == offset for pos in other_positions):
                    companion = other_color
                    break

            if companion is not None:
                paint = companion
                all_positions = mpos + colors[companion]
                lo, hi = min(all_positions), max(all_positions)
            else:
                paint = marker
                lo, hi = 0, line_len - 1

            marker_positions = set(mpos)
            for p in range(lo, hi + 1):
                if p % step != offset or p in marker_positions:
                    continue
                rc = cell(li, p)
                if int(out[rc]) == bg:
                    out[rc] = paint

            for p in marker_positions:
                out[cell(li, p)] = bg

    return out


# --------------- generator (D7 variant) ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.40
_P_COMPANION = 0.35

_MAX_RETRIES = 300


def _choose_dims(edge: str, n_active: int):
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


def _valid_steps(line_len: int) -> list[int]:
    hi = min(_STEP_RANGE[1], line_len - 1)
    return [step for step in range(_STEP_RANGE[0], hi + 1) if 2 * step <= line_len - 1]


def _build_line_dots(edge: str, line_len: int, marker_color: int, step: int):
    if edge in ("left", "top"):
        edge_pos, inner_pos = 0, step
        companion_beats = list(range(inner_pos + 2 * step, line_len, step))
    else:
        edge_pos, inner_pos = line_len - 1, line_len - 1 - step
        companion_beats = list(range(inner_pos - 2 * step, -1, -step))

    offset = edge_pos % step
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]

    other_colors = [color for color in _COLORS if color != marker_color]
    non_beat = [
        pos for pos in range(line_len)
        if pos % step != offset and pos not in (edge_pos, inner_pos)
    ]

    roll = random.random()
    if roll < _P_MARKER_ONLY:
        line_type = "marker_only"
    elif roll < _P_MARKER_ONLY + _P_COMPANION and companion_beats and other_colors:
        line_type = "companion"
    elif non_beat and other_colors:
        line_type = "bystander"
    else:
        line_type = "marker_only"

    if line_type == "companion":
        dots.append((random.choice(companion_beats), random.choice(other_colors)))
    elif line_type == "bystander":
        dots.append((random.choice(non_beat), random.choice(other_colors)))

    return dots, line_type


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]], edge: str, H: int, W: int
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

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    valid_steps = _valid_steps(line_len)
    if not valid_steps:
        return None

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []
    seen_companion = False

    for li in active_lines:
        marker_color = random.choice(_COLORS)
        step = random.choice(valid_steps)
        dots, line_type = _build_line_dots(edge, line_len, marker_color, step)

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1
        seen_companion = seen_companion or line_type == "companion"

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None

    # Enforce the D7 signature: marker cells must disappear, and at least one new trail cell
    # should appear somewhere so the output is not just a deletion.
    erased_markers = 0
    gained_trail = 0
    for (r, c), value in np.ndenumerate(inp):
        out_value = int(out[r, c])
        if int(value) != 0 and out_value == 0:
            erased_markers += 1
        if int(value) == 0 and out_value != 0:
            gained_trail += 1
    if erased_markers < 2 * n_active or gained_trail == 0:
        return None

    # Keep the companion branch present in many grids, but not mandatory for every one.
    if n_active >= 5 and not seen_companion and random.random() < 0.5:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
