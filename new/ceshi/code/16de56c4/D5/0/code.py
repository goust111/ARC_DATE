# Variant of 16de56c4
# Dimension: D5 — 终止条件 (dst)
# Change: 无伴随点时填满整行；有伴随点时填充标记点与伴随点的span范围内... → 碰撞停止

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D5 collision-stop variant
# -------------------------------------------------
# a) Concrete behavior and trace
#    Each active row/column still contains one repeated color with exactly two
#    dots. One marker sits on the shooting edge, the other lies step cells
#    inward; their gap defines the beat. The output still paints the same
#    periodic beat positions, but the stopping rule changes:
#    - Original: an aligned extra dot changes the paint color and defines a
#      [marker..companion] span; an unaligned extra dot is ignored.
#    - Variant: the ray keeps the marker color and travels physically away from
#      the edge. The first extra non-bg pixel beyond the inner marker is a hard
#      obstacle regardless of alignment. Beats strictly before that obstacle are
#      painted; the obstacle cell itself stays unchanged; nothing beyond it is
#      filled.
#    Example on a left-shooting row of length 11:
#      input  : 3 . 3 . . 8 . . . . .
#      step   : 2 from positions 0 and 2
#      old    : 3 . 3 . 3 . 3 . 3 . 3   (8 is off-beat, so ignored)
#      new    : 3 . 3 . 3 8 . . . . .   (ray reaches cell 4, then stops before 6)
#    This makes the D5 change directly visible even when the blocker is not on
#    the periodic beat.
#
# b) Complexity calibration
#    The variant keeps the original object vocabulary minimal: one marker pair
#    plus at most one extra dot per active line. To avoid degenerating back to
#    the old task, the generator deliberately mixes:
#    - clear lines: no blocker, fill reaches the far side
#    - aligned blockers: previously "companion-like", now only stop the ray
#    - offset blockers: previously harmless bystanders, now also stop the ray
#    This is enough to make the rule observable without introducing several new
#    object classes or multiple competing blockers on the same line.
#
# c) Original-code assumptions that break under the new D5
#    - Old solver searched for an aligned different-colored dot and repainted
#      the whole span with that dot's color. That no longer holds because the
#      extra dot never donates color.
#    - Old solver let marker-only lines fill the whole line and also let
#      off-beat dots remain irrelevant. Under the variant, an off-beat dot can
#      truncate the periodic fill if it lies farther inward on the same line.
#    - The old "if cur == bg or cur == marker: out[rc] = paint" pattern was
#      harmless because paint could legitimately overwrite marker cells within
#      the span. Now the ray should never recolor the obstacle itself, so the
#      stop position must be excluded from the fill loop.
#
# d) Placement legality guaranteed by the generator
#    - Exactly one repeated color appears on each active line.
#    - The extra dot, if present, lies strictly beyond the inner marker in the
#      shooting direction, so the two marker dots remain the first two hits that
#      define the step.
#    - There is at most one extra dot per line, so the first collision is
#      unambiguous and no overlapping placements occur.
#    - Blockers are kept off the opposite shooting edge to reduce accidental
#      edge-count ties; we still validate edge uniqueness before accepting.
#
# e) Visual observability and cleanliness
#    Humans can compare several lines inside one grid:
#    - some lines run to the far edge,
#    - some stop at an aligned blocker,
#    - some stop early at an off-beat blocker.
#    The palette stays small per grid, and the stopping decision depends only on
#    visible geometry along a line, never on color-id ordering or hidden state.
#
# f) New solver assumptions reviewed against the generator
#    - The solver assumes the repeated color has exactly two dots; generator
#      enforces that.
#    - It assumes any extra dot relevant to stopping is farther inward than the
#      inner marker; generator enforces that.
#    - It assumes one unique shooting edge by boundary counts; generator checks
#      and retries until that is true.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic rays that launch from one uniquely populated edge.

    Input structure:
    - Background is the most common color; non-bg content is sparse.
    - One grid edge has strictly more non-bg dots than the others; that edge
      determines the shooting direction (left/right -> rows, top/bottom -> columns).
    - Each active shooting line contains exactly one repeated marker color with
      two dots: one on the shooting edge and one step cells inward.
    - A line may also contain one extra non-bg dot farther along the same line;
      it can be aligned to the beat or off the beat.

    Rule:
    - Use the repeated marker pair to recover the step and the shooting direction.
    - Starting from the edge marker, extend the marker color along the periodic
      beat positions moving away from the shooting edge.
    - The first extra non-background dot beyond the inner marker is a solid
      blocker regardless of its color or alignment; the ray stops before that cell.
    - If no blocker exists, the periodic fill continues to the far boundary.

    Key insight: the extra dot no longer defines a color-changing span; instead,
    all that matters is whether a non-bg obstacle appears ahead of the periodic
    progression and where the first collision occurs.
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
    forward = 1 if edge in ("left", "top") else -1

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
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) must have exactly one "
            f"repeating marker color: {dict(colors)}"
        )
        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} has "
            f"{len(mpos)} dots (expected 2): {mpos}"
        )

        a, b = sorted(mpos)
        step = b - a
        assert step > 0, f"Line {li} has zero marker gap"
        if forward == 1:
            edge_pos, inner_pos = a, b
            assert edge_pos == 0, f"Line {li} should start on the near edge: {mpos}"
        else:
            edge_pos, inner_pos = b, a
            assert edge_pos == line_len - 1, f"Line {li} should start on the near edge: {mpos}"

        blocker_positions: list[int] = []
        for color, ps in colors.items():
            if color == marker:
                continue
            for p in ps:
                if forward == 1:
                    assert p > inner_pos, (
                        f"Line {li} blocker must be beyond inner marker: "
                        f"marker={mpos}, blocker={p}"
                    )
                else:
                    assert p < inner_pos, (
                        f"Line {li} blocker must be beyond inner marker: "
                        f"marker={mpos}, blocker={p}"
                    )
                blocker_positions.append(p)

        if blocker_positions:
            stop_pos = min(blocker_positions) if forward == 1 else max(blocker_positions)
        else:
            stop_pos = line_len if forward == 1 else -1

        p = edge_pos
        while 0 <= p < line_len:
            if (forward == 1 and p >= stop_pos) or (forward == -1 and p <= stop_pos):
                break
            rc = cell(li, p)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = marker
            p += forward * step

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_LINE_TYPES = ("clear", "aligned_blocker", "offset_blocker")
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


def _line_candidates(edge: str, line_len: int, step: int) -> tuple[int, int, list[int], list[int]]:
    if edge in ("left", "top"):
        edge_pos = 0
        inner_pos = step
        future_beats = list(range(inner_pos + step, line_len, step))
        aligned = [p for p in future_beats if p != line_len - 1]
        offbeat = [
            p
            for p in range(inner_pos + 1, line_len - 1)
            if p % step != edge_pos % step and future_beats and p < future_beats[-1]
        ]
    else:
        edge_pos = line_len - 1
        inner_pos = line_len - 1 - step
        future_beats = list(range(inner_pos - step, -1, -step))
        aligned = [p for p in future_beats if p != 0]
        offbeat = [
            p
            for p in range(1, inner_pos)
            if p % step != edge_pos % step and future_beats and p > future_beats[-1]
        ]
    return edge_pos, inner_pos, aligned, offbeat


def _feasible_steps(line_len: int, line_type: str) -> list[int]:
    steps: list[int] = []
    for step in range(_STEP_RANGE[0], min(_STEP_RANGE[1], line_len - 1) + 1):
        _, _, aligned, offbeat = _line_candidates("left", line_len, step)
        if line_type == "clear":
            steps.append(step)
        elif line_type == "aligned_blocker" and aligned:
            steps.append(step)
        elif line_type == "offset_blocker" and offbeat:
            steps.append(step)
    return steps


def _sample_line_dots(
    edge: str,
    line_len: int,
    palette: list[int],
    line_type: str,
) -> list[tuple[int, int]] | None:
    steps = _feasible_steps(line_len, line_type)
    if not steps:
        return None
    step = random.choice(steps)
    edge_pos, inner_pos, aligned, offbeat = _line_candidates(edge, line_len, step)
    marker_color = random.choice(palette)
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]

    if line_type == "clear":
        return dots

    other_colors = [c for c in palette if c != marker_color]
    if not other_colors:
        other_colors = [c for c in _COLORS if c != marker_color]
    if not other_colors:
        return None

    if line_type == "aligned_blocker":
        candidates = aligned
    else:
        candidates = offbeat
    if not candidates:
        return None

    if len(candidates) >= 2 and random.random() < 0.7:
        blocker_pos = random.choice(candidates[1:])
    else:
        blocker_pos = random.choice(candidates)
    blocker_color = random.choice(other_colors)
    dots.append((blocker_pos, blocker_color))
    return dots


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]], edge: str, H: int, W: int
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

    palette_size = random.randint(3, min(6, len(_COLORS)))
    palette = random.sample(_COLORS, palette_size)

    line_types = ["clear", "aligned_blocker", "offset_blocker"]
    while len(line_types) < n_active:
        line_types.append(random.choice(_LINE_TYPES))
    random.shuffle(line_types)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []
    blocker_lines = 0
    offset_lines = 0

    for li, line_type in zip(active_lines, line_types):
        dots = _sample_line_dots(edge, line_len, palette, line_type)
        if dots is None:
            return None
        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))
        shooting_edge_dots += 1
        if line_type != "clear":
            blocker_lines += 1
        if line_type == "offset_blocker":
            offset_lines += 1

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    if blocker_lines == 0 or offset_lines == 0:
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
