# Variant of 16de56c4
# Dimension: D7 — 变换规则 (transform)
# Change: 背景和标记色单元格被填充为轨迹颜色，旁观者点保持不变... → 保留源+填充

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D7 variant
# ----------------------------------
# a) Concrete behavior:
#    Each active row/column still contains exactly two same-color marker dots
#    that define a periodic beat, plus at most one extra non-marker dot.
#    The only rule change is at write time. Example on a left-shooting row:
#    input  = [2,0,2,0,0,0,7,0,0]  (markers at 0,2 -> step 2; companion at 6)
#    old out= [7,0,7,0,7,0,7,0,0]
#    new out= [2,0,2,0,7,0,7,0,0]
#    The companion still sets the fill color and span, but marker sources stay
#    marker-colored; only blank beat cells become products.
# b) Complexity calibration:
#    The task remains the same edge-detection / modular-beat problem. We keep
#    marker-only, companion, and bystander lines so the grid still contains the
#    original structural variety, but we force companion lines to appear often
#    enough that the new keep-source behavior is visible.
# c) Old implicit assumption that breaks here:
#    In the original solver, `if cur == bg or cur == marker: out[rc] = paint`
#    was harmless because markers were supposed to be overwritten. Under this
#    variant that line is wrong: marker cells are sources, not fill targets.
#    The fix is to paint only when `cur == bg`.
# d) Placement legality:
#    Active lines are spaced apart as before. Companion lines only use steps
#    that leave room for a blank beat between the inner marker and companion,
#    so the variant produces visible product cells without overlap or edge
#    ambiguity. Bystanders are forced onto non-beat positions.
# e) Visual observability:
#    A sample must contain companion lines; otherwise marker-only lines look the
#    same under both rules. The generator therefore requires at least two
#    companion lines and rejects cases where the new and legacy outputs match.
# f) Clean visualization:
#    Each shooting-direction line still contains one repeating color and at
#    most one extra dot, so there is no multi-object stacking in a line.
# g) Color inference:
#    Colors are never ordered or transformed by id. The companion color is
#    readable from the aligned endpoint itself; the preserved marker color is
#    readable from the two repeated source dots.


def _solve(grid: np.ndarray, *, keep_source: bool) -> np.ndarray:
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

            for p in range(lo, hi + 1):
                if p % step != offset:
                    continue
                rc = cell(li, p)
                cur = int(out[rc])
                if cur == bg:
                    out[rc] = paint
                elif not keep_source and cur == marker:
                    out[rc] = paint

    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, keep_source=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic rays from a unique boundary while keeping the marker sources intact.

    Input structure:
    - The background is the most common color, with sparse colored dots elsewhere.
    - One boundary has strictly more non-background dots than the other three; that
      boundary is the sole shooting edge, so rays run along rows or columns.
    - Every active line contains one repeated color appearing exactly twice; those
      two marker dots define the period, and a third dot may appear on or off-beat.

    Rule:
    - Detect the unique shooting edge, then inspect each row/column parallel to the
      ray direction independently.
    - For the repeated color on a line, use the gap between its two dots as the step
      and use their modular class as the beat positions.
    - An aligned third dot is a companion: it sets the fill color and limits filling
      to the span between the markers and that companion. An unaligned third dot is a
      bystander and stays untouched.
    - Only empty beat cells are painted. Marker dots keep their original color even
      when a companion controls the fill color.

    Key insight:
    The puzzle is still solved by finding the edge with the densest boundary dots and
    using paired marker spacing as a period; the variant changes only the write target:
    the periodic pattern creates new trail cells, but the original marker sources are preserved.
    """
    return _solve(grid, keep_source=True)


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_MIN_COMPANION_LINES = 2
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


def _edge_positions(edge: str, line_len: int, step: int) -> tuple[int, int, int]:
    if edge in ("left", "top"):
        edge_pos = 0
        inner_pos = step
    else:
        edge_pos = line_len - 1
        inner_pos = line_len - 1 - step
    return edge_pos, inner_pos, edge_pos % step


def _has_non_beat(edge: str, line_len: int, step: int) -> bool:
    edge_pos, inner_pos, offset = _edge_positions(edge, line_len, step)
    return any(
        pos % step != offset and pos not in (edge_pos, inner_pos)
        for pos in range(line_len)
    )


def _allowed_steps(edge: str, line_len: int, line_type: str) -> list[int]:
    steps = list(range(_STEP_RANGE[0], min(_STEP_RANGE[1], line_len - 1) + 1))
    if line_type == "companion":
        return [step for step in steps if 3 * step <= line_len - 1]
    if line_type == "bystander":
        return [step for step in steps if _has_non_beat(edge, line_len, step)]
    return steps


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    companion_color: int | None,
    step: int,
    line_type: str,
) -> list[tuple[int, int]] | None:
    edge_pos, inner_pos, offset = _edge_positions(edge, line_len, step)
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]

    if line_type == "marker_only":
        return dots

    if line_type == "companion":
        assert companion_color is not None
        if edge in ("left", "top"):
            beats = list(range(inner_pos + 2 * step, line_len, step))
        else:
            beats = list(range(inner_pos - 2 * step, -1, -step))
        if not beats:
            return None
        dots.append((random.choice(beats), companion_color))
        return dots

    if line_type == "bystander":
        other_color = companion_color
        assert other_color is not None
        non_beat = [
            pos
            for pos in range(line_len)
            if pos % step != offset and pos not in (edge_pos, inner_pos)
        ]
        if not non_beat:
            return None
        dots.append((random.choice(non_beat), other_color))
        return dots

    raise ValueError(f"Unknown line_type: {line_type}")


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
    counts: dict[str, int] = {}
    for other_edge in _EDGES:
        if other_edge == edge:
            continue
        check = boundary[other_edge]
        counts[other_edge] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _plan_line_types(n_active: int) -> list[str]:
    n_companion = random.randint(_MIN_COMPANION_LINES, n_active)
    remaining = n_active - n_companion
    n_bystander = random.randint(0, remaining)
    n_marker_only = remaining - n_bystander
    line_types = (
        ["companion"] * n_companion
        + ["bystander"] * n_bystander
        + ["marker_only"] * n_marker_only
    )
    random.shuffle(line_types)
    return line_types


def _variant_score(inp: np.ndarray, out: np.ndarray) -> int:
    legacy = _legacy_main(inp)
    return int(np.count_nonzero(legacy != out))


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

    palette_size = random.randint(2, min(8, len(_COLORS)))
    palette = random.sample(_COLORS, palette_size)
    line_types = _plan_line_types(n_active)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []

    for li, line_type in zip(active_lines, line_types):
        steps = _allowed_steps(edge, line_len, line_type)
        if not steps:
            return None
        step = random.choice(steps)

        marker_color = random.choice(palette)
        other_choices = [color for color in palette if color != marker_color]
        if not other_choices and line_type != "marker_only":
            return None
        extra_color = random.choice(other_choices) if other_choices else None

        dots = _build_line_dots(edge, line_len, marker_color, extra_color, step, line_type)
        if dots is None:
            return None

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None
    if _variant_score(inp, out) == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
