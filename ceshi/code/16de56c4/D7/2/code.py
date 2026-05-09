# Variant of 16de56c4
# Dimension: D7 — 变换规则 (transform)
# Change: 背景和标记色单元格被填充为轨迹颜色，旁观者点保持不变... → 伴随点变色

# Design analysis
# - concretization:
#   Each active row/column still contains exactly one repeating marker color with
#   two dots that define the beat. A same-beat third dot is still the companion,
#   but the output is no longer "everything becomes companion color". Instead the
#   ray cells use the companion color while the companion dot itself flips to the
#   marker color. Small trace:
#   input line [M . . M . . C . .], step=3.
#   original output -> [C . . C . . C . .]
#   variant output  -> [C . . C . . M . .]
#   This keeps the bounded ray shape while adding an explicit two-way color swap.
# - complexity calibration:
#   The rule changes only the companion branch; marker-only and bystander branches
#   stay unchanged. That keeps the original task recognizable while making the
#   transformed line visibly different in a way that is easy to compare across
#   multiple examples. Generator coverage is biased toward companion lines so the
#   new behavior appears often, but not exclusively.
# - implicit assumptions review:
#   The original solver relied on `if cur == bg or cur == marker` so the aligned
#   companion cell naturally stayed companion-colored. That is no longer valid:
#   the companion position must be detected and overwritten with the marker color
#   before the generic fill rule runs. The span computation and edge detection
#   assumptions remain valid because geometry is unchanged.
# - placement legality:
#   Active lines remain distinct, so dots on different lines never overlap. For a
#   companion line we require an extra beat beyond the inner marker (`2*step <
#   line_len`) so the companion can sit farther from the shooting edge. For a
#   bystander line we require `step > 1` so at least one off-beat cell exists.
# - visual observability:
#   Every generated grid contains at least one companion line, and companion lines
#   are common. On such a line the output necessarily contains both colors on the
#   same beat sequence: marker dots change to companion color while the companion
#   endpoint changes back to marker color. That difference is directly visible in
#   2-3 examples without relying on color IDs or hidden ordering.
# - cleanliness and color inference:
#   There is still at most one extra dot per active line, so the display stays
#   sparse and readable. The swapped colors are locally explained by alignment:
#   only the aligned third dot triggers the swap, while unaligned bystanders stay
#   unchanged, giving the human solver a clean contrast.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend periodic rays fired from one unique grid edge, with color swapping at
    aligned stopper dots.

    Input structure:
    - bg is the majority color, with a sparse set of non-bg dots.
    - One border owns the launch pattern: it has strictly more non-bg cells than
      the other three borders, so it determines whether rows or columns act as
      shooting lines.
    - On each active shooting line, one color appears exactly twice; those two
      dots define the beat spacing.
    - An optional third dot on the same beat but in a different color is a
      companion; an off-beat extra dot is just a bystander.

    Rule:
    - Find the unique shooting edge; left/right means process rows, top/bottom
      means process columns.
    - For each active line, use the two repeated marker dots to recover the step
      and beat class.
    - Without a companion, fill the whole line on that beat using the marker
      color, exactly as in the base task.
    - With a companion, fill only the beat cells inside the marker-companion span
      using the companion color, but recolor the companion dot itself to the
      marker color; bystanders are not changed.

    Key insight: the geometry is still controlled by the repeated marker pair,
    but an aligned third dot now performs a local color exchange: it limits the
    ray, lends its color to the track, and receives the marker color in return.
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
            v = int(g[cell(li, p)])
            if v != bg:
                colors[v].append(p)

        if not colors:
            continue

        repeating = {color: ps for color, ps in colors.items() if len(ps) >= 2}
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

            companion_color = None
            companion_positions: list[int] = []
            for other_color, other_positions in colors.items():
                if other_color == marker:
                    continue
                aligned = [p for p in other_positions if p % step == offset]
                if aligned:
                    assert companion_color is None, (
                        f"Line {li} ({'row' if by_row else 'col'}) has multiple companions"
                    )
                    companion_color = other_color
                    companion_positions = aligned

            if companion_color is not None:
                assert len(companion_positions) == 1, (
                    f"Line {li} ({'row' if by_row else 'col'}) expected 1 companion, "
                    f"got {companion_positions}"
                )
                companion_pos = companion_positions[0]
                paint = companion_color
                lo = min(a, b, companion_pos)
                hi = max(a, b, companion_pos)
            else:
                companion_pos = None
                paint = marker
                lo, hi = 0, line_len - 1

            for p in range(lo, hi + 1):
                if p % step != offset:
                    continue
                rc = cell(li, p)
                if companion_pos is not None and p == companion_pos:
                    out[rc] = marker
                    continue
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

_P_EXTRA_COMPANION = 0.55
_P_EXTRA_MARKER_ONLY = 0.20

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


def _valid_steps_for_mode(line_len: int, mode: str) -> list[int]:
    hi = min(_STEP_RANGE[1], line_len - 1)
    steps = []
    for step in range(_STEP_RANGE[0], hi + 1):
        if mode == "companion" and 2 * step >= line_len:
            continue
        if mode == "bystander" and step == 1:
            continue
        steps.append(step)
    return steps


def _choose_line_modes(n_active: int) -> list[str]:
    modes = ["companion"]
    for _ in range(n_active - 1):
        roll = random.random()
        if roll < _P_EXTRA_COMPANION:
            modes.append("companion")
        elif roll < _P_EXTRA_COMPANION + _P_EXTRA_MARKER_ONLY:
            modes.append("marker_only")
        else:
            modes.append("bystander")
    random.shuffle(modes)
    return modes


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    mode: str,
    step: int,
) -> list[tuple[int, int]] | None:
    if edge in ("left", "top"):
        edge_pos, inner_pos = 0, step
        away_beats = list(range(inner_pos + step, line_len, step))
    else:
        edge_pos, inner_pos = line_len - 1, line_len - 1 - step
        away_beats = list(range(inner_pos - step, -1, -step))
    offset = edge_pos % step
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]

    if mode == "marker_only":
        return dots

    other_colors = [c for c in _COLORS if c != marker_color]
    if not other_colors:
        return None

    if mode == "companion":
        if not away_beats:
            return None
        dots.append((random.choice(away_beats), random.choice(other_colors)))
        return dots

    non_beat = [
        p
        for p in range(line_len)
        if p not in (edge_pos, inner_pos) and p % step != offset
    ]
    if not non_beat:
        return None
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

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    line_modes = _choose_line_modes(n_active)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []

    for li, mode in zip(active_lines, line_modes):
        valid_steps = _valid_steps_for_mode(line_len, mode)
        if not valid_steps:
            return None
        marker_color = random.choice(_COLORS)
        step = random.choice(valid_steps)
        dots = _build_line_dots(edge, line_len, marker_color, mode, step)
        if dots is None:
            return None

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

    shooting_edge_dots = n_active
    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None

    companion_swaps = 0
    for li in range(num_lines):
        line_in = inp[li, :] if by_row else inp[:, li]
        line_out = out[li, :] if by_row else out[:, li]
        colors: dict[int, list[int]] = defaultdict(list)
        for p, value in enumerate(line_in):
            v = int(value)
            if v != 0:
                colors[v].append(p)
        repeating = {color: ps for color, ps in colors.items() if len(ps) == 2}
        if not repeating:
            continue
        marker, mpos = next(iter(repeating.items()))
        a, b = sorted(mpos)
        step = b - a
        offset = a % step
        for other_color, positions in colors.items():
            if other_color == marker:
                continue
            aligned = [p for p in positions if p % step == offset]
            if aligned:
                p = aligned[0]
                if int(line_out[p]) == marker and any(int(line_out[q]) == other_color for q in mpos):
                    companion_swaps += 1

    if companion_swaps == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
